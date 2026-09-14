"""
SupportGraph AI — Human Review Service (Phase 5.5)

Implements domain logic and data structures for human reviewers to evaluate
AI-assisted intent suggestions:
- Inspect customer opening messages, dialogue context, and AI suggestions
- Accept AI suggestion (`annotation_status = "reviewed"`)
- Override AI suggestion with a different taxonomy intent (`annotation_status = "overridden_ai_suggestion"`)
- Mark ambiguous messages as `unclear_needs_review` (`annotation_status = "reviewed"`)
- Skip records for subsequent review
- Track progress metrics: total, reviewed, accepted, overridden, and pending.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd

try:
    from app.core.logging import get_logger
    from app.nlp.taxonomy_finalization import (
        DEFAULT_CANDIDATE_TAXONOMY_PATH,
        create_initial_candidate_taxonomy,
        load_candidate_taxonomy,
    )
except ModuleNotFoundError:
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]
    from backend.app.nlp.taxonomy_finalization import (  # type: ignore[no-redef]
        DEFAULT_CANDIDATE_TAXONOMY_PATH,
        create_initial_candidate_taxonomy,
        load_candidate_taxonomy,
    )

logger = get_logger(__name__)

DEFAULT_PROGRESS_PATH = "data/golden/golden_annotation_progress.json"


class ReviewAction(str, Enum):
    ACCEPT = "ACCEPT"
    OVERRIDE = "OVERRIDE"
    UNCLEAR = "UNCLEAR"
    SKIP = "SKIP"
    GROUP_APPROVE = "GROUP_APPROVE"
    CALIBRATION_ACCEPT = "CALIBRATION_ACCEPT"


class InvalidReviewActionError(Exception):
    """Raised when an unrecognized review action or invalid taxonomy label is provided."""


def get_allowed_taxonomy_labels(
    taxonomy_path: Optional[Path | str] = None,
) -> set[str]:
    """Retrieve set of approved taxonomy intent names plus unclear_needs_review."""
    tax_path = Path(taxonomy_path) if taxonomy_path else Path(DEFAULT_CANDIDATE_TAXONOMY_PATH)
    if tax_path.exists():
        taxonomy = load_candidate_taxonomy(tax_path)
        labels = set(taxonomy.intent_names)
    else:
        taxonomy = create_initial_candidate_taxonomy()
        labels = set(taxonomy.intent_names)
    labels.add("unclear_needs_review")
    return labels


def is_valid_ai_suggestion(
    suggested_label: Any,
    allowed_labels: Optional[set[str] | list[str]] = None,
    suggestion_status: Optional[Any] = None,
) -> bool:
    """
    Determine whether an AI suggestion value is complete, valid, and eligible.

    Returns False if:
    - None or pd.isna
    - Empty string or whitespace
    - String literals 'nan', 'none', 'null', 'undefined' (case-insensitive)
    - suggestion_status is explicitly 'failed', 'invalid_model_output', or 'error'
    - suggested_label is not in approved taxonomy labels
    """
    if suggested_label is None or pd.isna(suggested_label):
        return False

    val = str(suggested_label).strip()
    if not val or val.lower() in ("nan", "none", "null", "undefined"):
        return False

    if suggestion_status is not None and not pd.isna(suggestion_status):
        status_val = str(suggestion_status).strip().lower()
        if status_val in ("failed", "invalid_model_output", "error", "pending"):
            return False

    if allowed_labels is not None:
        if val not in set(allowed_labels):
            return False

    return True


def is_suggestion_complete(
    record: dict[str, Any] | pd.Series,
    allowed_labels: Optional[set[str] | list[str]] = None,
) -> bool:
    """
    Check if a record has a complete and valid AI suggestion.
    """
    label = record.get("model_suggested_label") if hasattr(record, "get") else None
    status = record.get("suggestion_status") if hasattr(record, "get") else None
    return is_valid_ai_suggestion(label, allowed_labels, status)


def apply_review_decision(
    record: dict[str, Any] | pd.Series,
    action: str | ReviewAction,
    annotator: str,
    selected_label: Optional[str] = None,
    notes: Optional[str] = None,
    allowed_labels: Optional[set[str]] = None,
    review_mode: Optional[str] = None,
    approval_type: Optional[str] = None,
    group_id: Optional[str] = None,
    approved_group_label: Optional[str] = None,
    group_size: Optional[int] = None,
    annotation_timestamp: Optional[str] = None,
) -> dict[str, Any]:
    """
    Apply an explicit human review decision to a record dictionary.

    Args:
        record: Target golden record containing AI suggestion fields.
        action: One of ACCEPT, OVERRIDE, UNCLEAR, SKIP, GROUP_APPROVE, or CALIBRATION_ACCEPT.
        annotator: Human reviewer name or identifier (mandatory for non-SKIP).
        selected_label: Mandatory if action is OVERRIDE; optional intent override for GROUP_APPROVE/CALIBRATION_ACCEPT.
        notes: Optional human annotator notes.
        allowed_labels: Optional set of allowed taxonomy labels for validation.
        review_mode: Audit field ("individual_human_review" or "group_human_approval").
        approval_type: Audit field ("accepted_ai_suggestion", "human_override", "group_approved_suggestion", "individual_calibration_acceptance", "unclear").
        group_id: Optional group identifier when approved through group review.
        approved_group_label: Optional intent label of the approved group.
        group_size: Optional total record count in approved group.
        annotation_timestamp: Timestamp string (defaults to current UTC ISO timestamp).

    Returns:
        Updated dictionary with human annotation fields and audit trail populated.
    """
    rec = dict(record)
    act_str = str(action).upper().replace("REVIEWACTION.", "")

    valid_actions = {
        ReviewAction.ACCEPT.value,
        ReviewAction.OVERRIDE.value,
        ReviewAction.UNCLEAR.value,
        ReviewAction.SKIP.value,
        ReviewAction.GROUP_APPROVE.value,
        ReviewAction.CALIBRATION_ACCEPT.value,
    }
    if act_str not in valid_actions:
        raise InvalidReviewActionError(
            f"Invalid review action: '{action}'. Must be one of {sorted(list(valid_actions))}."
        )

    if act_str != ReviewAction.SKIP.value and not annotator.strip():
        raise ValueError("Annotator name or ID is required when recording an annotation decision.")

    if allowed_labels is None:
        allowed_labels = get_allowed_taxonomy_labels()

    now_ts = annotation_timestamp or datetime.now(timezone.utc).isoformat()

    if act_str == ReviewAction.ACCEPT.value:
        sug_label = rec.get("model_suggested_label")
        status = rec.get("suggestion_status")
        if not is_valid_ai_suggestion(sug_label, allowed_labels, status):
            cleaned = str(sug_label).strip() if sug_label is not None and not pd.isna(sug_label) else ""
            raise InvalidReviewActionError(
                f"Cannot ACCEPT suggestion: model_suggested_label '{cleaned}' is missing, invalid, or belongs to a failed suggestion run."
            )

        rec["annotation_label"] = str(sug_label).strip()
        rec["annotator"] = annotator.strip()
        rec["annotation_status"] = "reviewed"
        rec["review_mode"] = review_mode or "individual_human_review"
        rec["approval_type"] = approval_type or "accepted_ai_suggestion"
        rec["annotation_timestamp"] = now_ts
        if notes:
            rec["notes"] = notes.strip()

    elif act_str == ReviewAction.CALIBRATION_ACCEPT.value:
        calib_label = selected_label or rec.get("calibration_candidate_label") or rec.get("model_suggested_label")
        if not is_valid_ai_suggestion(calib_label, allowed_labels):
            cleaned = str(calib_label).strip() if calib_label is not None and not pd.isna(calib_label) else ""
            raise InvalidReviewActionError(
                f"Cannot CALIBRATION_ACCEPT: label '{cleaned}' is missing or invalid in taxonomy."
            )

        rec["annotation_label"] = str(calib_label).strip()
        rec["annotator"] = annotator.strip()
        rec["annotation_status"] = "reviewed"
        rec["review_mode"] = review_mode or "individual_human_review"
        rec["approval_type"] = approval_type or "individual_calibration_acceptance"
        rec["annotation_timestamp"] = now_ts
        if notes:
            rec["notes"] = notes.strip()

    elif act_str == ReviewAction.GROUP_APPROVE.value:
        target_intent = selected_label or rec.get("model_suggested_label")
        status = rec.get("suggestion_status")
        if not is_valid_ai_suggestion(target_intent, allowed_labels, status):
            cleaned = str(target_intent).strip() if target_intent is not None and not pd.isna(target_intent) else ""
            raise InvalidReviewActionError(
                f"Cannot GROUP_APPROVE: intent '{cleaned}' is missing or invalid in taxonomy."
            )

        rec["annotation_label"] = str(target_intent).strip()
        rec["annotator"] = annotator.strip()
        rec["annotation_status"] = "reviewed"
        rec["review_mode"] = review_mode or "group_human_approval"
        rec["approval_type"] = approval_type or "group_approved_suggestion"
        rec["annotation_timestamp"] = now_ts
        if group_id:
            rec["group_id"] = str(group_id).strip()
        if approved_group_label or target_intent:
            rec["approved_group_label"] = str(approved_group_label or target_intent).strip()
        if group_size is not None:
            rec["group_size"] = str(group_size)
        if notes:
            rec["notes"] = notes.strip()

    elif act_str == ReviewAction.OVERRIDE.value:
        if not selected_label or not str(selected_label).strip():
            raise InvalidReviewActionError("Action OVERRIDE requires a non-empty selected_label.")
        ovr_label = str(selected_label).strip()
        if ovr_label.lower() in ("nan", "none", "null") or ovr_label not in allowed_labels:
            raise InvalidReviewActionError(
                f"Selected override label '{ovr_label}' not in approved taxonomy: {sorted(list(allowed_labels))}"
            )

        rec["annotation_label"] = ovr_label
        rec["annotator"] = annotator.strip()
        rec["annotation_status"] = "overridden_ai_suggestion"
        rec["review_mode"] = review_mode or "individual_human_review"
        rec["approval_type"] = approval_type or "human_override"
        rec["annotation_timestamp"] = now_ts
        if notes:
            rec["notes"] = notes.strip()

    elif act_str == ReviewAction.UNCLEAR.value:
        rec["annotation_label"] = "unclear_needs_review"
        rec["annotator"] = annotator.strip()
        rec["annotation_status"] = "reviewed"
        rec["review_mode"] = review_mode or "individual_human_review"
        rec["approval_type"] = approval_type or "unclear"
        rec["annotation_timestamp"] = now_ts
        if notes:
            rec["notes"] = notes.strip()

    elif act_str == ReviewAction.SKIP.value:
        rec["annotation_label"] = ""
        rec["annotator"] = ""
        rec["annotation_status"] = "pending_human_review"
        rec["review_mode"] = ""
        rec["approval_type"] = ""
        rec["group_id"] = ""
        rec["approved_group_label"] = ""
        rec["group_size"] = ""
        rec["annotation_timestamp"] = ""
        if notes:
            rec["notes"] = notes.strip()

    return rec


def compute_review_progress(
    df: pd.DataFrame,
    allowed_labels: Optional[set[str]] = None,
) -> dict[str, Any]:
    """
    Calculate high-level human review progress metrics across a dataset.
    Safely handles NaN values, float-inferred columns, and string 'nan'.
    """
    total = len(df)
    if total == 0:
        return {
            "total_records": 0,
            "ai_suggestions_completed": 0,
            "human_reviewed": 0,
            "pending_human_review": 0,
            "accepted_ai_suggestions": 0,
            "overridden_ai_suggestions": 0,
            "unclear_records": 0,
            "group_approved_records": 0,
            "individual_reviewed_records": 0,
            "calibration_accepted_records": 0,
        }

    if allowed_labels is None:
        allowed_labels = get_allowed_taxonomy_labels()

    sug_completed = 0
    human_reviewed = 0
    accepted_count = 0
    overridden_count = 0
    unclear_count = 0
    group_approved_count = 0
    individual_reviewed_count = 0
    calibration_accepted_count = 0

    for _, r in df.iterrows():
        raw_sug = r.get("model_suggested_label")
        raw_status = r.get("suggestion_status")
        if is_valid_ai_suggestion(raw_sug, allowed_labels, raw_status):
            sug_completed += 1

        raw_lbl = r.get("annotation_label")
        lbl = str(raw_lbl).strip() if raw_lbl is not None and not pd.isna(raw_lbl) else ""
        if lbl.lower() in ("nan", "none", "null"):
            lbl = ""

        raw_astatus = r.get("annotation_status")
        astatus = str(raw_astatus).strip().lower() if raw_astatus is not None and not pd.isna(raw_astatus) else ""

        raw_rmode = r.get("review_mode")
        rmode = str(raw_rmode).strip().lower() if raw_rmode is not None and not pd.isna(raw_rmode) else ""

        raw_apptype = r.get("approval_type")
        apptype = str(raw_apptype).strip().lower() if raw_apptype is not None and not pd.isna(raw_apptype) else ""

        clean_sug = str(raw_sug).strip() if raw_sug is not None and not pd.isna(raw_sug) else ""

        if lbl and astatus not in ("pending_human_review", "pending", ""):
            human_reviewed += 1
            if rmode == "group_human_approval" or apptype == "group_approved_suggestion":
                group_approved_count += 1
            else:
                individual_reviewed_count += 1

            if apptype == "individual_calibration_acceptance":
                calibration_accepted_count += 1

            if astatus == "reviewed" and (apptype == "accepted_ai_suggestion" or (lbl == clean_sug and lbl != "unclear_needs_review")):
                accepted_count += 1
            elif astatus == "overridden_ai_suggestion":
                overridden_count += 1
            elif lbl == "unclear_needs_review":
                unclear_count += 1

    pending_human_review = total - human_reviewed

    return {
        "total_records": total,
        "ai_suggestions_completed": sug_completed,
        "human_reviewed": human_reviewed,
        "pending_human_review": pending_human_review,
        "accepted_ai_suggestions": accepted_count,
        "overridden_ai_suggestions": overridden_count,
        "unclear_records": unclear_count,
        "group_approved_records": group_approved_count,
        "individual_reviewed_records": individual_reviewed_count,
        "calibration_accepted_records": calibration_accepted_count,
    }


def load_review_progress_manifest(
    progress_path: Path | str = DEFAULT_PROGRESS_PATH,
) -> dict[str, Any]:
    """
    Safely load review progress manifest JSON with backward-compatible defaults.
    """
    p = Path(progress_path)
    default_manifest: dict[str, Any] = {
        "total_records": 0,
        "ai_suggestions_completed": 0,
        "human_reviewed": 0,
        "pending_human_review": 0,
        "accepted_ai_suggestions": 0,
        "overridden_ai_suggestions": 0,
        "unclear_records": 0,
        "group_approved_records": 0,
        "individual_reviewed_records": 0,
        "current_mode": "smart",
        "current_group_identifier": "",
        "completed_group_ids": [],
        "partially_reviewed_group_ids": [],
    }
    if not p.exists():
        return default_manifest

    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                # Overlay onto defaults
                for k, v in data.items():
                    default_manifest[k] = v
                return default_manifest
    except Exception as err:
        logger.warning("Could not read progress manifest from %s: %s. Using defaults.", p, err)

    return default_manifest


def save_review_progress_manifest(
    progress_stats: dict[str, Any],
    output_path: Path | str = DEFAULT_PROGRESS_PATH,
) -> Path:
    """Save progress statistics as JSON artifact."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(progress_stats, f, indent=2)
    logger.info("Saved review progress manifest to: %s", p)
    return p
