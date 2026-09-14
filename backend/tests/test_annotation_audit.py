"""
SupportGraph AI — Unit Tests for Model-Human Annotation Audit (Phase 5.5)

Tests:
1. Strict designation of MODEL_HUMAN_AGREEMENT metric name
2. Proper calculation of overall model-human agreement rate
3. Per-intent agreement breakdown calculation
4. Itemized disagreement recording
5. Scientific integrity disclaimer presence
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from app.evaluation.annotation_audit import (
        ModelHumanAuditReport,
        compute_model_human_audit,
    )
except ModuleNotFoundError:
    from backend.app.evaluation.annotation_audit import (  # type: ignore[no-redef]
        ModelHumanAuditReport,
        compute_model_human_audit,
    )


def test_empty_reviewed_dataset_audit() -> None:
    """Verify audit report when no human reviews have been performed yet."""
    df = pd.DataFrame([
        {
            "golden_id": "gold_001",
            "model_suggested_label": "battery_power_issue",
            "annotation_label": "",
            "annotation_status": "pending_human_review",
        }
    ])
    report = compute_model_human_audit(df)
    assert report.metric_name == "MODEL_HUMAN_AGREEMENT"
    assert report.total_records == 1
    assert report.human_reviewed_records == 0
    assert report.model_human_agreement_rate == 0.0
    assert "NOT an inter-annotator agreement score" in report.scientific_integrity_disclaimer


def test_compute_model_human_agreement_metrics() -> None:
    """Verify agreement percentage and per-intent metrics on reviewed records."""
    df = pd.DataFrame([
        # 3 Battery cases: 2 agreed, 1 overridden
        {
            "golden_id": "g1",
            "tweet_id": "t1",
            "customer_message": "Battery drops from 50 to 1",
            "model_suggested_label": "battery_power_issue",
            "model_confidence": 0.90,
            "annotation_label": "battery_power_issue",
            "annotation_status": "reviewed",
            "annotator": "Alice",
        },
        {
            "golden_id": "g2",
            "tweet_id": "t2",
            "customer_message": "Phone dies instantly",
            "model_suggested_label": "battery_power_issue",
            "model_confidence": 0.85,
            "annotation_label": "battery_power_issue",
            "annotation_status": "reviewed",
            "annotator": "Alice",
        },
        {
            "golden_id": "g3",
            "tweet_id": "t3",
            "customer_message": "Screen crack after battery puffed",
            "model_suggested_label": "battery_power_issue",
            "model_confidence": 0.65,
            "annotation_label": "display_touch_issue",  # Override!
            "annotation_status": "overridden_ai_suggestion",
            "annotator": "Alice",
        },
        # 1 Software Update case: agreed
        {
            "golden_id": "g4",
            "tweet_id": "t4",
            "customer_message": "Bricked on Apple logo after 11.0.3",
            "model_suggested_label": "software_update_problem",
            "model_confidence": 0.95,
            "annotation_label": "software_update_problem",
            "annotation_status": "reviewed",
            "annotator": "Alice",
        },
    ])

    report = compute_model_human_audit(df)

    assert report.metric_name == "MODEL_HUMAN_AGREEMENT"
    assert report.total_records == 4
    assert report.human_reviewed_records == 4
    assert report.accepted_suggestions == 3
    assert report.overridden_suggestions == 1
    # 3 out of 4 agreed = 75.0%
    assert report.model_human_agreement_rate == 75.0

    # Per intent: battery_power_issue (2 out of 2 agreed = 100%), display_touch_issue (0 out of 1 = 0%)
    assert report.per_intent_agreement["battery_power_issue"] == 100.0
    assert report.per_intent_agreement["software_update_problem"] == 100.0
    assert report.per_intent_agreement["display_touch_issue"] == 0.0

    # Disagreements
    assert len(report.disagreements) == 1
    dis = report.disagreements[0]
    assert dis["golden_id"] == "g3"
    assert dis["model_suggested_label"] == "battery_power_issue"
    assert dis["human_label"] == "display_touch_issue"
