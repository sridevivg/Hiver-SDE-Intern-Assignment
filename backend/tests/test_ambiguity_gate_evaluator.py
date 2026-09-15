"""
Tests for AmbiguityGateEvaluator & 3-Strategy Comparison (Phase 7.1)
"""
import pandas as pd
import pytest

from backend.app.evaluation.ambiguity_gate_evaluator import (
    AmbiguityGateEvaluationReport,
    AmbiguityGateEvaluator,
    categorize_unsafe_error,
)
from backend.app.intent.clarity_signals import (
    ClaritySignalsProfile,
    EvidenceAgreementLevel,
    MultiSymptomLevel,
    ProblemStrengthLevel,
    RetrievalQualityLevel,
    SufficiencyLevel,
)
from backend.app.understanding.problem_extractor import CustomerProblemProfile


from backend.app.intent.classifier import TopKIntentClassifier
from backend.app.schemas.intent_routing import IntentAnalysis


@pytest.fixture
def evaluator() -> AmbiguityGateEvaluator:
    clf = TopKIntentClassifier(api_key="")
    clf.classify = lambda customer_message, normalized_message="", conversation_context="", top_k=3: (
        clf.classify_heuristic_fallback(customer_message, requested_k=top_k or 3)
        and IntentAnalysis(
            top_predictions=clf.classify_heuristic_fallback(customer_message, requested_k=top_k or 3),
            top_confidence=clf.classify_heuristic_fallback(customer_message, requested_k=top_k or 3)[0].confidence,
            confidence_margin=0.50,
            normalized_entropy=0.10,
            model_name="heuristic_baseline",
        )
    )
    return AmbiguityGateEvaluator(classifier=clf)


def test_evaluate_dataframe_3_strategies(evaluator: AmbiguityGateEvaluator) -> None:
    # Create synthetic test dataset of diverse ground-truth cases
    data = [
        {
            "golden_id": "test_1",
            "customer_message": "My battery is draining within two hours.",
            "annotation_label": "battery_power_issue",
            "annotation_status": "reviewed",
        },
        {
            "golden_id": "test_2",
            "customer_message": "Apple please fix this.",
            "annotation_label": "general_device_support",
            "annotation_status": "reviewed",
        },
        {
            "golden_id": "test_3",
            "customer_message": "My screen flickers and speakers make buzzing sound.",
            "annotation_label": "display_touch_issue",
            "annotation_status": "reviewed",
        },
    ]
    df = pd.DataFrame(data)

    report: AmbiguityGateEvaluationReport = evaluator.evaluate_dataframe(df)

    assert report.total_records == 3
    assert report.strategy_a_confidence_only.total_records == 3
    assert report.strategy_b_phase_7_ambiguity.total_records == 3
    assert report.strategy_c_phase_7_1_gate.total_records == 3
    assert len(report.signal_interception_analysis) > 0
    assert report.summary_markdown() != ""


def test_categorize_unsafe_error() -> None:
    profile = CustomerProblemProfile(
        device="iPhone",
        primary_symptom="battery draining",
        update_related=True,
    )
    signals = ClaritySignalsProfile(
        sufficiency=SufficiencyLevel.SUFFICIENT,
        problem_strength=ProblemStrengthLevel.STRONG,
        candidate_conflict="CAUSE_VS_SYMPTOM",
        evidence_agreement=EvidenceAgreementLevel.STRONG_AGREEMENT,
        retrieval_quality=RetrievalQualityLevel.NO_EVIDENCE,
        multi_symptom=MultiSymptomLevel.SINGLE_SYMPTOM,
    )

    cat, exp = categorize_unsafe_error(
        message="update caused issues",
        truth="software_update_problem",
        pred="battery_power_issue",
        profile=profile,
        signals=signals,
    )

    assert cat in ("MULTIPLE_VALID_INTERPRETATIONS", "TAXONOMY_LIMITATION", "RETRIEVAL_MISMATCH")
    assert exp != ""
