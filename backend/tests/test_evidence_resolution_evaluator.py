"""
Tests for backend/app/evaluation/evidence_resolution_evaluator.py (Phase 8)
"""
import pandas as pd
import pytest

from backend.app.evaluation.evidence_resolution_evaluator import (
    EvidenceResolutionEvaluator,
    Phase8BenchmarkMetrics,
    compute_sha256,
)
from backend.app.intent.classifier import TopKIntentClassifier
from backend.app.resolution.support_resolution_engine import SupportResolutionEngine
from backend.app.schemas.intent_routing import IntentAnalysis, IntentPrediction


@pytest.fixture
def fast_engine() -> SupportResolutionEngine:
    """Fast engine with deterministic heuristic classifier for rapid test execution."""
    clf = TopKIntentClassifier(api_key="")
    clf.classify = lambda customer_message, normalized_message="", conversation_context="", top_k=3: (
        IntentAnalysis(
            top_predictions=clf.classify_heuristic_fallback(customer_message, requested_k=top_k or 3),
            top_confidence=clf.classify_heuristic_fallback(customer_message, requested_k=top_k or 3)[0].confidence,
            confidence_margin=0.40,
            normalized_entropy=0.20,
            model_name="heuristic_test_model",
        )
    )
    return SupportResolutionEngine(classifier=clf)


def test_sha256_computation():
    sha = compute_sha256("data/golden/golden_set_human_review.csv")
    assert sha == "1d3e9b3b8bdef3750437e59b17ab7c27167fd1548c2bb29de984151296c5b45a"


def test_evaluator_benchmark_execution(fast_engine):
    evaluator = EvidenceResolutionEvaluator(engine=fast_engine)
    metrics: Phase8BenchmarkMetrics = evaluator.run_benchmark()

    assert metrics.total_evaluated_records == 77
    assert metrics.completed_human_ground_truth_records == 77
    assert metrics.dataset_immutability_pass is True
    assert metrics.pre_eval_sha256 == metrics.post_eval_sha256
    assert metrics.pre_eval_sha256 == "1d3e9b3b8bdef3750437e59b17ab7c27167fd1548c2bb29de984151296c5b45a"

    # Core metrics
    assert metrics.primary_intent_accuracy >= 0.50
    assert metrics.top_3_coverage >= 0.75
    assert metrics.auto_handle_accuracy >= 0.50
    assert metrics.response_verification_pass_rate >= 0.80
    assert "Phase_8_Evidence_Grounded_Resolution" in metrics.progression_comparison
