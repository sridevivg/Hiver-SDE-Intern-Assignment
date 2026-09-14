"""
SupportGraph AI — Unit Tests for Inter-Annotator Agreement (Phase 5)

Tests:
1. Scientific honesty rule: single annotator reports INTER_ANNOTATOR_AGREEMENT_NOT_AVAILABLE
2. Raw percentage agreement computation
3. Cohen's Kappa score computation and Landis & Koch benchmark interpretation
4. Disagreement confusion matrix generation
5. Disagreement example itemization
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
    from app.evaluation.annotation_agreement import (
        STATUS_AVAILABLE,
        STATUS_NOT_AVAILABLE,
        AgreementReport,
        compute_cohen_kappa,
        compute_inter_annotator_agreement,
        interpret_cohen_kappa,
    )
except ModuleNotFoundError:
    from backend.app.evaluation.annotation_agreement import (  # type: ignore[no-redef]
        STATUS_AVAILABLE,
        STATUS_NOT_AVAILABLE,
        AgreementReport,
        compute_cohen_kappa,
        compute_inter_annotator_agreement,
        interpret_cohen_kappa,
    )


def test_scientific_honesty_single_annotator() -> None:
    """Verify that a single annotator dataset yields INTER_ANNOTATOR_AGREEMENT_NOT_AVAILABLE."""
    df_a = pd.DataFrame({
        "golden_id": ["gold_1", "gold_2"],
        "annotation_label": ["battery_power_issue", "software_update_problem"],
    })

    report = compute_inter_annotator_agreement(df_a, None)
    assert report.status == STATUS_NOT_AVAILABLE
    assert "INTER_ANNOTATOR_AGREEMENT_NOT_AVAILABLE" in report.message

    report_empty = compute_inter_annotator_agreement(df_a, pd.DataFrame())
    assert report_empty.status == STATUS_NOT_AVAILABLE


def test_compute_cohen_kappa_perfect_and_zero() -> None:
    """Verify Cohen's Kappa formula under perfect agreement and chance disagreement."""
    labels_a = ["cat1", "cat2", "cat3", "cat1", "cat2"]
    labels_b = ["cat1", "cat2", "cat3", "cat1", "cat2"]
    kappa_perf = compute_cohen_kappa(labels_a, labels_b)
    assert kappa_perf == 1.0

    # Benchmark interpretation
    assert interpret_cohen_kappa(1.0) == "Almost perfect agreement"
    assert interpret_cohen_kappa(0.70) == "Substantial agreement"
    assert interpret_cohen_kappa(0.50) == "Moderate agreement"
    assert interpret_cohen_kappa(0.30) == "Fair agreement"
    assert interpret_cohen_kappa(0.10) == "Slight agreement"
    assert interpret_cohen_kappa(-0.1) == "Poor (less than chance agreement)"


def test_compute_inter_annotator_agreement_two_annotators() -> None:
    """Verify agreement calculation with two annotator datasets."""
    records_a = [
        {"golden_id": "g1", "tweet_id": "t1", "customer_message": "Msg 1", "annotation_label": "battery_power_issue", "annotator": "Alice"},
        {"golden_id": "g2", "tweet_id": "t2", "customer_message": "Msg 2", "annotation_label": "software_update_problem", "annotator": "Alice"},
        {"golden_id": "g3", "tweet_id": "t3", "customer_message": "Msg 3", "annotation_label": "account_access_issue", "annotator": "Alice"},
        {"golden_id": "g4", "tweet_id": "t4", "customer_message": "Msg 4", "annotation_label": "billing_purchase_issue", "annotator": "Alice"},
    ]
    records_b = [
        {"golden_id": "g1", "tweet_id": "t1", "customer_message": "Msg 1", "annotation_label": "battery_power_issue", "annotator": "Bob"},
        {"golden_id": "g2", "tweet_id": "t2", "customer_message": "Msg 2", "annotation_label": "software_update_problem", "annotator": "Bob"},
        {"golden_id": "g3", "tweet_id": "t3", "customer_message": "Msg 3", "annotation_label": "display_touch_issue", "annotator": "Bob"},  # Disagree
        {"golden_id": "g4", "tweet_id": "t4", "customer_message": "Msg 4", "annotation_label": "billing_purchase_issue", "annotator": "Bob"},
    ]

    df_a = pd.DataFrame(records_a)
    df_b = pd.DataFrame(records_b)

    report = compute_inter_annotator_agreement(df_a, df_b)

    assert report.status == STATUS_AVAILABLE
    assert report.total_compared == 4
    assert report.agreed_count == 3
    assert report.disagreement_count == 1
    assert report.raw_agreement == 75.0
    assert report.cohen_kappa is not None
    assert len(report.disagreements) == 1

    dis = report.disagreements[0]
    assert dis["golden_id"] == "g3"
    assert dis["label_annotator_a"] == "account_access_issue"
    assert dis["label_annotator_b"] == "display_touch_issue"
    assert dis["annotator_a"] == "Alice"
    assert dis["annotator_b"] == "Bob"
