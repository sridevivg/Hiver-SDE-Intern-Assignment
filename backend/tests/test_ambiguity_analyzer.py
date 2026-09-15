"""
Tests for PrimaryProblemSelector and AmbiguityAnalyzer (Phase 7 Disambiguation & Ambiguity Layer)
"""
import pytest

from backend.app.intent.ambiguity_analyzer import (
    AmbiguityAnalysisResult,
    AmbiguityAnalyzer,
    AmbiguityType,
)
from backend.app.intent.classifier import TopKIntentClassifier
from backend.app.intent.primary_problem_selector import PrimaryProblemSelector
from backend.app.schemas.intent_routing import (
    IntentAnalysis,
    IntentPrediction,
    RoutingDecisionType,
)
from backend.app.understanding.problem_extractor import ProblemExtractor


@pytest.fixture
def extractor() -> ProblemExtractor:
    return ProblemExtractor()


@pytest.fixture
def selector() -> PrimaryProblemSelector:
    return PrimaryProblemSelector()


@pytest.fixture
def analyzer() -> AmbiguityAnalyzer:
    return AmbiguityAnalyzer()


def test_primary_problem_selector_decouples_update_from_battery(
    extractor: ProblemExtractor,
    selector: PrimaryProblemSelector,
) -> None:
    msg = "Battery is draining in 2 hours since updating to iOS 11"
    profile = extractor.extract(msg)

    # Simulated classifier analysis predicting software_update_problem
    analysis = IntentAnalysis(
        top_predictions=[
            IntentPrediction(intent="software_update_problem", confidence=0.75),
            IntentPrediction(intent="battery_power_issue", confidence=0.20),
        ],
        top_confidence=0.75,
        confidence_margin=0.55,
        model_name="test_model",
    )

    primary, cause, reason = selector.select_primary_problem(msg, profile, analysis)
    assert primary == "battery_power_issue"
    assert cause == "software_update_problem"
    assert "battery" in reason.lower()


def test_ambiguity_analyzer_cause_vs_symptom(
    extractor: ProblemExtractor,
    selector: PrimaryProblemSelector,
    analyzer: AmbiguityAnalyzer,
) -> None:
    msg = "Battery is draining in 2 hours since updating to iOS 11"
    profile = extractor.extract(msg)
    analysis = IntentAnalysis(
        top_predictions=[
            IntentPrediction(intent="software_update_problem", confidence=0.75),
            IntentPrediction(intent="battery_power_issue", confidence=0.20),
        ],
        top_confidence=0.75,
        confidence_margin=0.55,
        model_name="test_model",
    )
    primary, cause, _ = selector.select_primary_problem(msg, profile, analysis)

    result: AmbiguityAnalysisResult = analyzer.analyze(
        message=msg,
        profile=profile,
        analysis=analysis,
        primary_intent=primary,
        contextual_cause_intent=cause,
    )

    assert result.ambiguity_type == AmbiguityType.CAUSE_VS_SYMPTOM
    assert result.contextual_cause == "software_update_problem"
    assert result.routing_recommendation == RoutingDecisionType.AUTO_HANDLE


def test_ambiguity_analyzer_genuine_ambiguity(
    extractor: ProblemExtractor,
    analyzer: AmbiguityAnalyzer,
) -> None:
    msg = "I have an issue with my phone"
    profile = extractor.extract(msg)
    analysis = IntentAnalysis(
        top_predictions=[
            IntentPrediction(intent="general_device_support", confidence=0.36),
            IntentPrediction(intent="hardware_audio_connection_issue", confidence=0.34),
        ],
        top_confidence=0.36,
        confidence_margin=0.02,
        normalized_entropy=0.98,
        model_name="test_model",
    )

    result = analyzer.analyze(
        message=msg,
        profile=profile,
        analysis=analysis,
        primary_intent="general_device_support",
    )

    assert result.is_ambiguous is True
    assert result.routing_recommendation == RoutingDecisionType.ESCALATE_TO_HUMAN


def test_ambiguity_analyzer_clear_primary(
    extractor: ProblemExtractor,
    analyzer: AmbiguityAnalyzer,
) -> None:
    msg = "How do I reset my Apple ID password?"
    profile = extractor.extract(msg)
    analysis = IntentAnalysis(
        top_predictions=[
            IntentPrediction(intent="account_access_issue", confidence=0.95),
            IntentPrediction(intent="billing_purchase_issue", confidence=0.03),
        ],
        top_confidence=0.95,
        confidence_margin=0.92,
        normalized_entropy=0.15,
        model_name="test_model",
    )

    result = analyzer.analyze(
        message=msg,
        profile=profile,
        analysis=analysis,
        primary_intent="account_access_issue",
    )

    assert result.ambiguity_type == AmbiguityType.CLEAR_PRIMARY
    assert result.is_ambiguous is False
    assert result.routing_recommendation == RoutingDecisionType.AUTO_HANDLE
