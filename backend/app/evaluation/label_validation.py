"""
SupportGraph AI — Golden Set Label Validation Module (Phase 5)

Validates completed and in-progress golden evaluation datasets:
- Schema and required column enforcement
- Allowed taxonomy label verification (including 'unclear_needs_review')
- Strict empty label detection
- Duplicate tweet ID and duplicate text detection
- Label distribution and class imbalance analysis
- Fails loudly with descriptive error messages; never silently alters labels.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd

try:
    from app.core.logging import get_logger
except ModuleNotFoundError:
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]

logger = get_logger(__name__)

REQUIRED_GOLDEN_COLUMNS = [
    "golden_id",
    "tweet_id",
    "conversation_id",
    "customer_message",
    "normalized_message",
    "candidate_intent",
    "source_cluster",
    "conversation_context",
    "annotation_label",
    "annotation_status",
    "annotator",
    "notes",
]


@dataclass
class ValidationReport:
    """Structured report of golden set validation findings."""
    is_valid: bool
    total_records: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    label_distribution: dict[str, int] = field(default_factory=dict)
    completion_stats: dict[str, Any] = field(default_factory=dict)


class GoldenValidationError(Exception):
    """Raised when a golden dataset fails strict validation checks."""
    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        message = f"Golden set validation failed with {len(errors)} error(s):\n" + "\n".join(f" - {e}" for e in errors)
        super().__init__(message)


def validate_required_columns(df: pd.DataFrame) -> list[str]:
    """Check that all required golden dataset columns exist."""
    missing = set(REQUIRED_GOLDEN_COLUMNS) - set(df.columns)
    if missing:
        return [f"Missing required golden columns: {sorted(list(missing))}"]
    return []


def validate_no_duplicate_tweet_ids(df: pd.DataFrame) -> list[str]:
    """Verify that every tweet_id in the golden set is unique."""
    dupes = df[df.duplicated(subset=["tweet_id"], keep=False)]
    if not dupes.empty:
        dupe_ids = dupes["tweet_id"].unique().tolist()
        return [f"Found {len(dupes)} rows with duplicate tweet_id(s): {dupe_ids[:5]}"]
    return []


def validate_no_duplicate_normalized_messages(df: pd.DataFrame) -> list[str]:
    """Verify that every normalized customer message is unique."""
    dupes = df[df.duplicated(subset=["normalized_message"], keep=False)]
    if not dupes.empty:
        sample_dupes = dupes["normalized_message"].head(3).tolist()
        return [f"Found {len(dupes)} rows with duplicate normalized text. Examples: {sample_dupes}"]
    return []


def compute_annotation_completion(df: pd.DataFrame) -> dict[str, Any]:
    """Calculate annotation completion statistics."""
    total = len(df)
    if total == 0:
        return {"total": 0, "labeled": 0, "empty": 0, "percent_complete": 0.0}

    # An entry is labeled if annotation_label is non-empty string and not null
    labels = df["annotation_label"].fillna("").astype(str).str.strip()
    labeled_count = int((labels != "").sum())
    empty_count = total - labeled_count
    pct = round((labeled_count / total) * 100.0, 2)

    return {
        "total": total,
        "labeled": labeled_count,
        "empty": empty_count,
        "percent_complete": pct,
    }


def validate_no_empty_labels(df: pd.DataFrame) -> list[str]:
    """Verify that no records have empty or whitespace-only annotation labels."""
    labels = df["annotation_label"].fillna("").astype(str).str.strip()
    empty_mask = labels == ""
    empty_count = int(empty_mask.sum())
    if empty_count > 0:
        empty_ids = df.loc[empty_mask, "golden_id"].head(5).tolist()
        return [f"Golden set contains {empty_count} unlabelled record(s). Sample IDs: {empty_ids}"]
    return []


def validate_allowed_labels(
    df: pd.DataFrame,
    allowed_labels: list[str] | set[str],
    allow_empty: bool = False,
    allow_unclear: bool = True,
) -> list[str]:
    """
    Check that all labels in df belong to the approved taxonomy list (or 'unclear_needs_review').
    """
    valid_set = set(allowed_labels)
    if allow_unclear:
        valid_set.add("unclear_needs_review")

    labels = df["annotation_label"].fillna("").astype(str).str.strip()
    errors: list[str] = []

    for idx, (gid, lbl) in enumerate(zip(df.get("golden_id", range(len(df))), labels)):
        if lbl == "":
            if not allow_empty:
                errors.append(f"Row {idx} ({gid}): annotation_label is empty.")
        elif lbl not in valid_set:
            errors.append(
                f"Row {idx} ({gid}): invalid label '{lbl}'. Must be one of: {sorted(list(valid_set))}"
            )
            if len(errors) >= 10:
                errors.append("... truncated further label validation errors.")
                break

    return errors


def compute_label_distribution(df: pd.DataFrame) -> dict[str, int]:
    """Compute counts per intent label in the dataset."""
    labels = df["annotation_label"].fillna("").astype(str).str.strip()
    non_empty = labels[labels != ""]
    return non_empty.value_counts().to_dict()


def detect_class_imbalance(
    df: pd.DataFrame,
    min_class_ratio: float = 0.03,
) -> list[str]:
    """
    Warn if any annotated class has representation below min_class_ratio of the total.
    """
    dist = compute_label_distribution(df)
    total = sum(dist.values())
    if total == 0:
        return []

    warnings: list[str] = []
    for cls, count in dist.items():
        ratio = count / total
        if ratio < min_class_ratio:
            warnings.append(
                f"Class '{cls}' has low representation: {count}/{total} ({ratio*100:.1f}% < {min_class_ratio*100:.1f}%)"
            )
    return warnings


def validate_golden_set(
    df: pd.DataFrame,
    allowed_labels: list[str] | set[str],
    require_complete: bool = True,
    allow_unclear: bool = True,
) -> ValidationReport:
    """
    Full validation pipeline for a golden evaluation dataset.

    Args:
        df: DataFrame to validate.
        allowed_labels: List or set of approved intent labels.
        require_complete: If True, requires 100% completion (zero empty labels).
        allow_unclear: If True, allows 'unclear_needs_review' as an explicit label.

    Returns:
        ValidationReport summarizing findings.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # 1. Required columns
    col_errors = validate_required_columns(df)
    errors.extend(col_errors)
    if col_errors:
        return ValidationReport(
            is_valid=False,
            total_records=len(df),
            errors=errors,
            warnings=warnings,
            label_distribution={},
            completion_stats={},
        )

    # 2. Duplicate checks
    errors.extend(validate_no_duplicate_tweet_ids(df))
    errors.extend(validate_no_duplicate_normalized_messages(df))

    # 3. Completion and empty label checks
    completion = compute_annotation_completion(df)
    if require_complete:
        errors.extend(validate_no_empty_labels(df))

    # 4. Allowed labels check
    errors.extend(
        validate_allowed_labels(
            df,
            allowed_labels=allowed_labels,
            allow_empty=not require_complete,
            allow_unclear=allow_unclear,
        )
    )

    # 5. Class distribution and warnings
    dist = compute_label_distribution(df)
    warnings.extend(detect_class_imbalance(df))

    is_valid = len(errors) == 0
    return ValidationReport(
        is_valid=is_valid,
        total_records=len(df),
        errors=errors,
        warnings=warnings,
        label_distribution=dist,
        completion_stats=completion,
    )
