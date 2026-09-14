"""
SupportGraph AI — Unit Tests for Annotation Pattern Analysis & Taxonomy Calibration (Phase 5.9)

Tests:
1. Completed vs pending filtering (verifying pending records are ignored in agreement calculations).
2. Agreement rate, counts, and Wilson score confidence interval calculations.
3. Confusion matrix construction and row/column totals.
4. Dynamic semantic n-gram extraction.
5. general_device_support deep-dive audit.
6. Taxonomy overlap boundary detection.
7. Annotation guideline generation.
8. Handling of empty datasets and missing columns.
9. Dataset read-only immutability (verifying SHA-256 checksum preservation).
10. Full report export (JSON and Markdown).
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.app.evaluation.annotation_pattern_analysis import (
    AgreementMetrics,
    audit_general_device_support,
    compute_agreement_metrics,
    compute_confusion_matrix,
    compute_per_intent_statistics,
    discover_semantic_patterns,
    export_reports_to_directory,
    extract_frequent_ngrams,
    extract_reviewed_dataset,
    generate_annotation_guidelines,
    generate_guidelines_markdown,
    generate_markdown_report,
    run_annotation_pattern_analysis,
)


@pytest.fixture
def mock_reviewed_dataset() -> pd.DataFrame:
    """Create a controlled 10-record dataset (6 reviewed, 4 pending)."""
    return pd.DataFrame([
        # 1. Agreed: battery_power_issue
        {
            "golden_id": "gold_001",
            "customer_message": "My iPhone battery dies in 2 hours.",
            "model_suggested_label": "battery_power_issue",
            "annotation_label": "battery_power_issue",
            "annotation_status": "reviewed",
            "annotator": "sridevi",
        },
        # 2. Overridden: AI general_device_support -> Human software_update_problem
        {
            "golden_id": "gold_002",
            "customer_message": "Since updating to iOS 11 my phone is buggy and restarts.",
            "model_suggested_label": "general_device_support",
            "annotation_label": "software_update_problem",
            "annotation_status": "overridden_ai_suggestion",
            "annotator": "sridevi",
        },
        # 3. Overridden: AI general_device_support -> Human software_update_problem
        {
            "golden_id": "gold_003",
            "customer_message": "After installing the new update my phone crashes.",
            "model_suggested_label": "general_device_support",
            "annotation_label": "software_update_problem",
            "annotation_status": "overridden_ai_suggestion",
            "annotator": "sridevi",
        },
        # 4. Agreed: general_device_support
        {
            "golden_id": "gold_004",
            "customer_message": "Can I bring my device to the Apple store tomorrow?",
            "model_suggested_label": "general_device_support",
            "annotation_label": "general_device_support",
            "annotation_status": "reviewed",
            "annotator": "sridevi",
        },
        # 5. Overridden: AI display_touch_issue -> Human keyboard_typing_issue
        {
            "golden_id": "gold_005",
            "customer_message": "Please fix this letter eye problem with autocorrect changing i to A [?].",
            "model_suggested_label": "display_touch_issue",
            "annotation_label": "keyboard_typing_issue",
            "annotation_status": "overridden_ai_suggestion",
            "annotator": "sridevi",
        },
        # 6. Unclear: unclear_needs_review
        {
            "golden_id": "gold_006",
            "customer_message": "help please",
            "model_suggested_label": "general_device_support",
            "annotation_label": "unclear_needs_review",
            "annotation_status": "unclear",
            "annotator": "sridevi",
        },
        # 7-10. Pending records (should NOT be included in agreement)
        {
            "golden_id": "gold_007",
            "customer_message": "AirPods have no sound.",
            "model_suggested_label": "hardware_audio_connection_issue",
            "annotation_label": "",
            "annotation_status": "pending_human_review",
            "annotator": "",
        },
        {
            "golden_id": "gold_008",
            "customer_message": "Locked out of Apple ID.",
            "model_suggested_label": "account_access_issue",
            "annotation_label": "",
            "annotation_status": "pending_human_review",
            "annotator": "",
        },
        {
            "golden_id": "gold_009",
            "customer_message": "MacBook keeps freezing on boot.",
            "model_suggested_label": "mac_software_issue",
            "annotation_label": "",
            "annotation_status": "pending_human_review",
            "annotator": "",
        },
        {
            "golden_id": "gold_010",
            "customer_message": "Charged twice for subscription.",
            "model_suggested_label": "billing_purchase_issue",
            "annotation_label": "",
            "annotation_status": "pending_human_review",
            "annotator": "",
        },
    ])


def test_extract_reviewed_dataset(mock_reviewed_dataset: pd.DataFrame) -> None:
    """Verify that only reviewed rows are extracted and pending rows are excluded."""
    reviewed = extract_reviewed_dataset(mock_reviewed_dataset)
    assert len(reviewed) == 6
    assert set(reviewed["golden_id"]) == {"gold_001", "gold_002", "gold_003", "gold_004", "gold_005", "gold_006"}
    assert all(reviewed["annotation_label"] != "")


def test_compute_agreement_metrics(mock_reviewed_dataset: pd.DataFrame) -> None:
    """Verify agreement calculation: 2 agreed / 6 reviewed = 33.3%."""
    metrics = compute_agreement_metrics(mock_reviewed_dataset, total_records=10)
    assert metrics.total_golden_records == 10
    assert metrics.completed_reviews == 6
    assert metrics.pending_reviews == 4
    assert metrics.agreed_count == 2
    assert metrics.overridden_count == 4
    assert metrics.unclear_count == 1
    assert pytest.approx(metrics.observed_agreement_rate, 0.01) == 0.3333
    assert metrics.confidence_interval_95[0] < metrics.observed_agreement_rate < metrics.confidence_interval_95[1]
    assert metrics.sample_coverage_pct == 60.0
    assert "OBSERVED ANNOTATION AGREEMENT" in metrics.scientific_disclaimer


def test_compute_confusion_matrix(mock_reviewed_dataset: pd.DataFrame) -> None:
    """Verify confusion matrix structure and row/col tallies."""
    reviewed = extract_reviewed_dataset(mock_reviewed_dataset)
    matrix, all_intents = compute_confusion_matrix(reviewed)

    assert "general_device_support" in all_intents
    assert "software_update_problem" in all_intents
    assert "keyboard_typing_issue" in all_intents

    # AI general_device_support -> Human software_update_problem: count is 2
    assert matrix["general_device_support"]["software_update_problem"] == 2
    # AI general_device_support -> Human general_device_support: count is 1
    assert matrix["general_device_support"]["general_device_support"] == 1
    # AI display_touch_issue -> Human keyboard_typing_issue: count is 1
    assert matrix["display_touch_issue"]["keyboard_typing_issue"] == 1


def test_per_intent_statistics(mock_reviewed_dataset: pd.DataFrame) -> None:
    """Verify per-intent precision, recall, and support calculations."""
    reviewed = extract_reviewed_dataset(mock_reviewed_dataset)
    matrix, all_intents = compute_confusion_matrix(reviewed)
    stats = compute_per_intent_statistics(reviewed, all_intents)

    stat_map = {s.intent: s for s in stats}
    gds = stat_map["general_device_support"]
    assert gds.ground_truth_support == 1
    assert gds.ai_suggested_support == 4
    assert gds.agreed_matches == 1
    assert pytest.approx(gds.observed_precision, 0.01) == 0.25  # 1 / 4
    assert pytest.approx(gds.observed_recall, 0.01) == 1.0     # 1 / 1

    sup = stat_map["software_update_problem"]
    assert sup.ground_truth_support == 2
    assert sup.ai_suggested_support == 0
    assert sup.agreed_matches == 0
    assert sup.ai_false_negatives == 2


def test_dynamic_ngram_extraction() -> None:
    """Verify dynamic token and n-gram extraction from text messages."""
    texts = [
        "Since updating to iOS 11 my phone keeps restarting.",
        "After updating to iOS 11 battery drains fast.",
        "Updating to iOS 11 caused so many bugs.",
    ]
    ngrams = extract_frequent_ngrams(texts, top_k=5)
    signals = [s["signal"] for s in ngrams]
    assert any("ios" in s or "updating" in s or "updating to" in s for s in signals)


def test_discover_semantic_patterns(mock_reviewed_dataset: pd.DataFrame) -> None:
    """Verify discovery of confusion pairs from override instances."""
    reviewed = extract_reviewed_dataset(mock_reviewed_dataset)
    pairs = discover_semantic_patterns(reviewed)

    assert len(pairs) >= 2
    top_pair = pairs[0]
    assert top_pair.ai_suggested_label == "general_device_support"
    assert top_pair.human_ground_truth_label == "software_update_problem"
    assert top_pair.count == 2
    assert len(top_pair.representative_examples) == 2
    assert len(top_pair.recurring_semantic_signals) > 0


def test_general_device_support_audit(mock_reviewed_dataset: pd.DataFrame) -> None:
    """Verify specialized audit of general_device_support."""
    reviewed = extract_reviewed_dataset(mock_reviewed_dataset)
    audit = audit_general_device_support(reviewed)

    assert audit.ai_predicted_count == 4
    assert audit.human_confirmed_count == 1
    assert audit.human_overridden_count == 3
    assert pytest.approx(audit.override_rate, 0.01) == 0.75  # 3 / 4
    assert "OVER-USED AS CATCH-ALL" in audit.audit_verdict
    assert len(audit.top_target_intents) > 0
    assert audit.top_target_intents[0]["intent"] == "software_update_problem"


def test_empty_dataset_handling() -> None:
    """Verify graceful handling when dataset is empty."""
    empty_df = pd.DataFrame(columns=["golden_id", "customer_message", "model_suggested_label", "annotation_label", "annotation_status"])
    report = run_annotation_pattern_analysis(empty_df, total_records=200)

    assert report.metrics.completed_reviews == 0
    assert report.metrics.observed_agreement_rate == 0.0
    assert report.per_intent_agreement == []
    assert report.top_confusion_pairs == []
    assert report.general_device_support_audit.ai_predicted_count == 0


def test_export_reports_to_directory(mock_reviewed_dataset: pd.DataFrame) -> None:
    """Verify complete export of JSON and Markdown reports to a temporary directory."""
    report = run_annotation_pattern_analysis(mock_reviewed_dataset, total_records=10)

    with tempfile.TemporaryDirectory() as tmp_dir:
        created = export_reports_to_directory(report, tmp_dir)

        assert "summary_json" in created
        assert "confusion_matrix_json" in created
        assert "correction_patterns_json" in created
        assert "taxonomy_overlap_json" in created
        assert "annotation_guidelines_md" in created
        assert "annotation_pattern_analysis_md" in created

        # Verify JSON file validity
        with open(created["summary_json"], "r", encoding="utf-8") as f:
            summary_loaded = json.load(f)
            assert summary_loaded["metrics"]["completed_reviews"] == 6

        # Verify Markdown report content
        with open(created["annotation_pattern_analysis_md"], "r", encoding="utf-8") as f:
            md_content = f.read()
            assert "SupportGraph AI — Phase 5.9 Annotation Pattern Analysis" in md_content
            assert "general_device_support" in md_content


def test_source_dataset_immutability() -> None:
    """
    CRITICAL SCIENTIFIC SAFETY TEST:
    Verify that executing the full pattern analysis pipeline does not modify
    the source golden dataset file in any way.
    """
    golden_path = _PROJECT_ROOT / "data" / "golden" / "golden_set_human_review.csv"
    if not golden_path.exists():
        pytest.skip("Golden dataset file does not exist in workspace.")

    # Calculate SHA-256 before analysis
    with open(golden_path, "rb") as f:
        sha_before = hashlib.sha256(f.read()).hexdigest()

    df = pd.read_csv(golden_path, dtype=str)
    report = run_annotation_pattern_analysis(df, total_records=200)

    # Calculate SHA-256 after analysis
    with open(golden_path, "rb") as f:
        sha_after = hashlib.sha256(f.read()).hexdigest()

    assert sha_before == sha_after, "SOURCE DATASET WAS MODIFIED! Scientific immutability violated."
    assert report.metrics.completed_reviews == 57
