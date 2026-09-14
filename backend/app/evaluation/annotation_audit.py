"""
SupportGraph AI — Model-Human Annotation Audit Service (Phase 5.5)

Audits agreement between AI advisory suggestions and human review decisions.

SCIENTIFIC INTEGRITY PRINCIPLE:
AI-human agreement is NOT inter-annotator agreement.
A model's suggestion compared against a human's label measures model recommendation
acceptance/alignment, NOT human inter-rater reliability.
Cohen's Kappa between an AI model and a human annotator must NOT be calculated or
represented as human inter-annotator agreement.
This metric is strictly designated: `MODEL_HUMAN_AGREEMENT`.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    from app.core.logging import get_logger
except ModuleNotFoundError:
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]

logger = get_logger(__name__)

DISCLAIMER_TEXT = (
    "SCIENTIFIC INTEGRITY NOTICE: This metric measures MODEL_HUMAN_AGREEMENT "
    "(how frequently human reviewers accepted the AI model's suggestions). "
    "It is NOT an inter-annotator agreement score and must never be reported as human inter-rater reliability."
)


@dataclass
class ModelHumanAuditReport:
    """Structured audit report analyzing AI suggestion acceptance and human overrides."""
    metric_name: str = "MODEL_HUMAN_AGREEMENT"
    total_records: int = 0
    ai_suggestions_generated: int = 0
    human_reviewed_records: int = 0
    pending_human_review: int = 0
    accepted_suggestions: int = 0
    overridden_suggestions: int = 0
    unclear_records: int = 0
    model_human_agreement_rate: float = 0.0
    per_intent_agreement: dict[str, float] = field(default_factory=dict)
    disagreements: list[dict[str, Any]] = field(default_factory=list)
    scientific_integrity_disclaimer: str = DISCLAIMER_TEXT

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_model_human_audit(df: pd.DataFrame) -> ModelHumanAuditReport:
    """
    Compute MODEL_HUMAN_AGREEMENT statistics across an annotated dataframe.

    Args:
        df: DataFrame containing model suggestion fields and human annotation fields.

    Returns:
        ModelHumanAuditReport containing detailed audit metrics.
    """
    total = len(df)
    if total == 0:
        return ModelHumanAuditReport()

    sug_col = df.get("model_suggested_label", pd.Series([""] * total)).fillna("").astype(str).str.strip()
    lbl_col = df.get("annotation_label", pd.Series([""] * total)).fillna("").astype(str).str.strip()
    status_col = df.get("annotation_status", pd.Series([""] * total)).fillna("").astype(str).str.strip()

    suggestions_generated = int((sug_col != "").sum())
    reviewed_mask = (lbl_col != "") & (status_col != "pending_human_review") & (status_col != "pending")
    reviewed_count = int(reviewed_mask.sum())
    pending_count = total - reviewed_count

    accepted_mask = (status_col == "reviewed") & (lbl_col == sug_col) & (lbl_col != "unclear_needs_review")
    accepted_count = int(accepted_mask.sum())
    overridden_count = int((status_col == "overridden_ai_suggestion").sum())
    unclear_count = int((lbl_col == "unclear_needs_review").sum())

    if reviewed_count == 0:
        return ModelHumanAuditReport(
            metric_name="MODEL_HUMAN_AGREEMENT",
            total_records=total,
            ai_suggestions_generated=suggestions_generated,
            human_reviewed_records=0,
            pending_human_review=total,
            accepted_suggestions=0,
            overridden_suggestions=0,
            unclear_records=0,
            model_human_agreement_rate=0.0,
            per_intent_agreement={},
            disagreements=[],
            scientific_integrity_disclaimer=DISCLAIMER_TEXT,
        )

    # Filter to reviewed rows where both suggestion and human label exist
    df_reviewed = df[reviewed_mask].copy()
    agreed_mask = df_reviewed["model_suggested_label"].astype(str).str.strip() == df_reviewed["annotation_label"].astype(str).str.strip()
    agreed_count = int(agreed_mask.sum())
    overall_agreement = round((agreed_count / len(df_reviewed)) * 100.0, 2)

    # Per-intent agreement (grouped by final human annotation label)
    per_intent: dict[str, float] = {}
    for human_intent, group in df_reviewed.groupby("annotation_label"):
        intent_str = str(human_intent).strip()
        if not intent_str:
            continue
        matched = (group["model_suggested_label"].astype(str).str.strip() == intent_str).sum()
        pct = round((matched / len(group)) * 100.0, 2)
        per_intent[intent_str] = pct

    # Collect disagreements
    disagreements: list[dict[str, Any]] = []
    for idx, row in df_reviewed[~agreed_mask].iterrows():
        disagreements.append({
            "golden_id": row.get("golden_id", f"row_{idx}"),
            "tweet_id": row.get("tweet_id", ""),
            "customer_message": row.get("customer_message", ""),
            "model_suggested_label": row.get("model_suggested_label", ""),
            "model_confidence": row.get("model_confidence", 0.0),
            "model_reasoning": row.get("model_reasoning_summary", ""),
            "human_label": row.get("annotation_label", ""),
            "annotator": row.get("annotator", ""),
            "notes": row.get("notes", ""),
        })

    return ModelHumanAuditReport(
        metric_name="MODEL_HUMAN_AGREEMENT",
        total_records=total,
        ai_suggestions_generated=suggestions_generated,
        human_reviewed_records=reviewed_count,
        pending_human_review=pending_count,
        accepted_suggestions=accepted_count,
        overridden_suggestions=overridden_count,
        unclear_records=unclear_count,
        model_human_agreement_rate=overall_agreement,
        per_intent_agreement=per_intent,
        disagreements=disagreements,
        scientific_integrity_disclaimer=DISCLAIMER_TEXT,
    )
