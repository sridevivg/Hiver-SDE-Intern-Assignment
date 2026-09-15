"""
SupportGraph AI — Tests: Phase 10.2 Evaluator (Phase 10.2)

Tests:
- OperationalCoverageEvaluator initialization and metrics model
- Recovery math and delta comparisons
- Dataset SHA-256 verification logic
"""
from __future__ import annotations

from pathlib import Path
import pytest

from backend.app.evaluation.operational_coverage_evaluator import (
    GOLDEN_CSV_PATH,
    GOLDEN_SHA256,
    OperationalCoverageEvaluator,
    Phase102EvaluationMetrics,
    compute_sha256,
)


def test_sha256_computation():
    """Verify SHA-256 computation on golden dataset."""
    actual = compute_sha256(GOLDEN_CSV_PATH)
    assert actual == GOLDEN_SHA256


def test_metrics_model_validation():
    """Verify Phase102EvaluationMetrics model defaults and constraints."""
    metrics = Phase102EvaluationMetrics(
        usable_evidence_coverage=0.55,
        direct_problem_match_rate=0.35,
        related_problem_match_rate=0.20,
        good_recovery_count=18,
        auto_handle_precision=1.0,
    )
    assert metrics.usable_evidence_coverage == 0.55
    assert metrics.good_recovery_count == 18
    assert metrics.bad_recovery_count == 0
    assert metrics.safety_regression_detected is False


def test_evaluator_structure():
    """Verify OperationalCoverageEvaluator initial state."""
    evaluator = OperationalCoverageEvaluator()
    assert evaluator.golden_path.exists()
