"""
SupportGraph AI — Test Suite for Routing Calibrator (Phase 6.1)

Covers:
- Pre-caching benchmark record predictions
- Safety and automation metric calculations
- Threshold candidate evaluation and safety scoring
- Configuration categorization (UNSAFE, CONSERVATIVE, BALANCED, RECOMMENDED)
- Grid search execution and baseline vs calibrated comparison
- Artifact generation and schema validation
- Source dataset immutability verification
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from backend.app.evaluation.routing_calibrator import (
    CachedRecordPrediction,
    CalibrationReport,
    ConfigurationEvaluationResult,
    RoutingCalibrator,
    ThresholdCandidate,
)
from backend.app.intent.classifier import TopKIntentClassifier
from backend.app.schemas.intent_routing import IntentAnalysis, IntentPrediction
from backend.scripts.calibrate_routing_thresholds import compute_sha256, export_calibration_artifacts


# ---------------------------------------------------------------------------
# Test Fixtures & Mock Helpers
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_classifier():
    mock_llm = MagicMock()
    mock_llm.provider = "mock"
    mock_llm.model = "mock-model"
    classifier = TopKIntentClassifier(llm_client=mock_llm)
    classifier.classify = lambda customer_message, normalized_message="", conversation_context="", top_k=3: (
        IntentAnalysis(
            top_predictions=classifier.classify_heuristic_fallback(customer_message, requested_k=top_k or 3),
            top_confidence=0.90,
            confidence_margin=0.80,
            normalized_entropy=0.25,
            model_name="mock_classifier",
        )
    )
    return classifier


@pytest.fixture
def sample_cached_records():
    """Create 4 synthetic cached records with known ground truth and predictions."""
    return [
        CachedRecordPrediction(
            golden_id="g1",
            customer_message="Battery is dead",
            ground_truth_label="battery_power_issue",
            top_1_intent="battery_power_issue",
            top_1_confidence=0.95,
            top_2_intent="general_device_support",
            top_2_confidence=0.05,
            confidence_margin=0.90,
            normalized_entropy=0.20,
            top_predictions=["battery_power_issue", "general_device_support"],
            is_top_1_correct=True,
            is_top_2_correct=True,
            is_top_3_correct=True,
        ),
        CachedRecordPrediction(
            golden_id="g2",
            customer_message="Screen flickers and sound is crackling",
            ground_truth_label="hardware_audio_connection_issue",
            top_1_intent="display_touch_issue",  # Model error on Top-1
            top_1_confidence=0.55,
            top_2_intent="hardware_audio_connection_issue",  # Correct in Top-2
            top_2_confidence=0.45,
            confidence_margin=0.10,
            normalized_entropy=0.90,
            top_predictions=["display_touch_issue", "hardware_audio_connection_issue"],
            is_top_1_correct=False,
            is_top_2_correct=True,
            is_top_3_correct=True,
        ),
        CachedRecordPrediction(
            golden_id="g3",
            customer_message="iCloud locked password reset",
            ground_truth_label="account_access_issue",
            top_1_intent="account_access_issue",
            top_1_confidence=0.92,
            top_2_intent="billing_purchase_issue",
            top_2_confidence=0.08,
            confidence_margin=0.84,
            normalized_entropy=0.30,
            top_predictions=["account_access_issue", "billing_purchase_issue"],
            is_top_1_correct=True,
            is_top_2_correct=True,
            is_top_3_correct=True,
        ),
        CachedRecordPrediction(
            golden_id="g4",
            customer_message="Charged twice on card for subscription",
            ground_truth_label="billing_purchase_issue",
            top_1_intent="general_device_support",  # Model error
            top_1_confidence=0.50,
            top_2_intent="billing_purchase_issue",  # Correct in Top-2
            top_2_confidence=0.50,
            confidence_margin=0.00,
            normalized_entropy=1.00,
            top_predictions=["general_device_support", "billing_purchase_issue"],
            is_top_1_correct=False,
            is_top_2_correct=True,
            is_top_3_correct=True,
        ),
    ]


# ---------------------------------------------------------------------------
# Test Metric Calculations & Threshold Logic
# ---------------------------------------------------------------------------
class TestThresholdEvaluation:
    def test_permissive_thresholds_lead_to_unsafe_auto_handling(self, sample_cached_records):
        """Very loose thresholds (0.50 conf, 0.0 margin, 1.0 entropy) auto-handle errors."""
        calibrator = RoutingCalibrator()
        candidate = ThresholdCandidate(conf_threshold=0.50, margin_threshold=0.00, entropy_threshold=1.00)
        res = calibrator.evaluate_configuration(sample_cached_records, candidate)

        assert res.total_records == 4
        assert res.auto_handle_count == 4
        assert res.auto_handle_rate == 1.0
        assert res.correct_auto_handles == 2
        assert res.incorrect_auto_handles == 2  # Unsafe!
        assert res.auto_handle_precision == 0.50
        assert res.category == "UNSAFE"
        assert res.error_interception_rate == 0.0

    def test_strict_thresholds_intercept_errors(self, sample_cached_records):
        """Strict thresholds (0.90 conf, 0.20 margin, 0.50 entropy) cleanly separate safe cases from errors."""
        calibrator = RoutingCalibrator()
        candidate = ThresholdCandidate(conf_threshold=0.90, margin_threshold=0.20, entropy_threshold=0.50)
        res = calibrator.evaluate_configuration(sample_cached_records, candidate)

        assert res.total_records == 4
        assert res.auto_handle_count == 2  # g1 and g3
        assert res.correct_auto_handles == 2
        assert res.incorrect_auto_handles == 0  # Zero unsafe auto-handles!
        assert res.auto_handle_precision == 1.0
        assert res.escalate_count == 2  # g2 and g4
        assert res.intercepted_errors == 2  # 100% of errors intercepted
        assert res.error_interception_rate == 1.0
        assert res.human_assistance_top2_rate == 1.0  # Both errors had ground truth in Top-2!
        assert res.category in ["BALANCED", "CONSERVATIVE", "RECOMMENDED"]


# ---------------------------------------------------------------------------
# Test Grid Search & Recommendation Engine
# ---------------------------------------------------------------------------
class TestGridSearchAndRanking:
    def test_run_grid_search_on_mock_df(self, mock_classifier):
        calibrator = RoutingCalibrator(
            classifier=mock_classifier,
            confidence_candidates=[0.70, 0.85, 0.90],
            margin_candidates=[0.10, 0.15, 0.20],
            entropy_candidates=[0.50, 0.65],
        )
        mock_df = pd.DataFrame([
            {
                "golden_id": "g1",
                "customer_message": "Battery is dead",
                "annotation_label": "battery_power_issue",
                "annotation_status": "reviewed",
            },
            {
                "golden_id": "g2",
                "customer_message": "Screen is shattered",
                "annotation_label": "display_touch_issue",
                "annotation_status": "reviewed",
            },
        ])
        report = calibrator.run_grid_search(mock_df)

        assert report.total_evaluated_records == 2
        assert report.total_configurations_evaluated == 3 * 3 * 2  # 18
        assert report.recommended_configuration is not None
        assert report.baseline_configuration is not None
        assert "Comparison" in report.comparison_table_markdown() or "|" in report.comparison_table_markdown()


# ---------------------------------------------------------------------------
# Test Artifact Export & Schema Validation
# ---------------------------------------------------------------------------
class TestArtifactExport:
    def test_export_calibration_artifacts(self, tmp_path, sample_cached_records):
        calibrator = RoutingCalibrator()
        candidate_base = ThresholdCandidate(conf_threshold=0.85, margin_threshold=0.15, entropy_threshold=0.65)
        candidate_rec = ThresholdCandidate(conf_threshold=0.90, margin_threshold=0.20, entropy_threshold=0.50)

        base_res = calibrator.evaluate_configuration(sample_cached_records, candidate_base)
        rec_res = calibrator.evaluate_configuration(sample_cached_records, candidate_rec)

        report = CalibrationReport(
            total_evaluated_records=4,
            total_configurations_evaluated=10,
            baseline_configuration=base_res,
            recommended_configuration=rec_res,
            all_results=[base_res, rec_res],
            unsafe_count=1,
            conservative_count=0,
            balanced_count=1,
        )

        paths = export_calibration_artifacts(report, tmp_path)
        assert len(paths) == 6
        for name, p in paths.items():
            assert p.exists()
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                assert isinstance(data, (dict, list))


# ---------------------------------------------------------------------------
# Test Dataset Immutability
# ---------------------------------------------------------------------------
class TestDatasetImmutability:
    def test_golden_dataset_sha256_unmodified(self):
        golden_file = Path("data/golden/golden_set_human_review.csv")
        assert golden_file.exists()
        current_sha = compute_sha256(golden_file)
        expected_sha = "1d3e9b3b8bdef3750437e59b17ab7c27167fd1548c2bb29de984151296c5b45a"
        assert current_sha == expected_sha
