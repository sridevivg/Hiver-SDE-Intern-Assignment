"""
Tests for AmbiguityDecisionGate and Safety Veto Policy (Phase 7.1)
"""
import pytest

from backend.app.intent.ambiguity_decision_gate import (
    AmbiguityDecisionGate,
    DecisionGateResult,
)
from backend.app.intent.clarity_signals import (
    CandidateConflictType,
    EvidenceAgreementLevel,
    SufficiencyLevel,
)
from backend.app.retrieval.evidence_ranker import (
    EvidenceMatchTier,
    RetrievedEvidenceCase,
)
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
def decision_gate() -> AmbiguityDecisionGate:
    return AmbiguityDecisionGate()


def test_veto_high_confidence_insufficient_information(
    extractor: ProblemExtractor,
    decision_gate: AmbiguityDecisionGate,
) -> None:
    # High confidence (97%) with vague/insufficient input
    msg = "Apple please fix this."
    profile = extractor.extract(msg)
    analysis = IntentAnalysis(
        top_predictions=[
            IntentPrediction(intent="software_update_problem", confidence=0.97),
            IntentPrediction(intent="general_device_support", confidence=0.02),
        ],
        top_confidence=0.97,
        confidence_margin=0.95,
        model_name="test_model",
    )

    result: DecisionGateResult = decision_gate.evaluate_gate(
        message=msg,
        profile=profile,
        analysis=analysis,
        primary_intent="software_update_problem",
    )

    # VETO must override high confidence!
    assert result.decision == RoutingDecisionType.ESCALATE_TO_HUMAN
    assert result.is_clear is False
    assert len(result.escalation_reasons) > 0
    assert any("insufficient" in r.lower() or "veto" in r.lower() for r in result.escalation_reasons)


def test_veto_high_confidence_evidence_disagreement(
    extractor: ProblemExtractor,
    decision_gate: AmbiguityDecisionGate,
) -> None:
    # Model predicts audio (94%) for a WiFi inquiry
    msg = "My WiFi is not connecting."
    profile = extractor.extract(msg)
    analysis = IntentAnalysis(
        top_predictions=[
            IntentPrediction(intent="hardware_audio_connection_issue", confidence=0.94),
        ],
        top_confidence=0.94,
        confidence_margin=0.90,
        model_name="test_model",
    )

    result = decision_gate.evaluate_gate(
        message=msg,
        profile=profile,
        analysis=analysis,
        primary_intent="hardware_audio_connection_issue",
    )

    # VETO must override high confidence!
    assert result.decision == RoutingDecisionType.ESCALATE_TO_HUMAN
    assert result.is_clear is False
    assert any("contradiction" in r.lower() or "disagree" in r.lower() or "veto" in r.lower() for r in result.escalation_reasons)


def test_cause_vs_symptom_safe_auto_handle(
    extractor: ProblemExtractor,
    decision_gate: AmbiguityDecisionGate,
) -> None:
    msg = "My battery drains within two hours after updating to iOS 11."
    profile = extractor.extract(msg)
    analysis = IntentAnalysis(
        top_predictions=[
            IntentPrediction(intent="software_update_problem", confidence=0.60),
            IntentPrediction(intent="battery_power_issue", confidence=0.35),
        ],
        top_confidence=0.88,
        confidence_margin=0.50,
        model_name="test_model",
    )

    result = decision_gate.evaluate_gate(
        message=msg,
        profile=profile,
        analysis=analysis,
        primary_intent="battery_power_issue",
        contextual_cause_intent="software_update_problem",
    )

    assert result.decision == RoutingDecisionType.AUTO_HANDLE
    assert result.is_clear is True
    assert result.signals.candidate_conflict == CandidateConflictType.CAUSE_VS_SYMPTOM


def test_genuine_ambiguity_escalation(
    extractor: ProblemExtractor,
    decision_gate: AmbiguityDecisionGate,
) -> None:
    msg = "My phone is messed up after the update."
    profile = extractor.extract(msg)
    analysis = IntentAnalysis(
        top_predictions=[
            IntentPrediction(intent="software_update_problem", confidence=0.51),
            IntentPrediction(intent="general_device_support", confidence=0.44),
        ],
        top_confidence=0.51,
        confidence_margin=0.07,
        model_name="test_model",
    )

    result = decision_gate.evaluate_gate(
        message=msg,
        profile=profile,
        analysis=analysis,
        primary_intent="software_update_problem",
        contextual_cause_intent="software_update_problem",
    )

    assert result.decision == RoutingDecisionType.ESCALATE_TO_HUMAN
    assert result.is_clear is False


def test_clear_battery_auto_handle(
    extractor: ProblemExtractor,
    decision_gate: AmbiguityDecisionGate,
) -> None:
    msg = "My iPhone 7 battery is draining very fast and health is at 74%."
    profile = extractor.extract(msg)
    analysis = IntentAnalysis(
        top_predictions=[
            IntentPrediction(intent="battery_power_issue", confidence=0.92),
            IntentPrediction(intent="general_device_support", confidence=0.05),
        ],
        top_confidence=0.92,
        confidence_margin=0.87,
        model_name="test_model",
    )
    evidence = [
        RetrievedEvidenceCase(
            case_id="c1",
            similarity_score=0.85,
            operational_similarity=0.90,
            match_tier=EvidenceMatchTier.DIRECT_PROBLEM_MATCH,
            historical_customer_message="battery draining fast",
            historical_brand_response="Check battery health.",
            historical_intent="battery_power_issue",
        )
    ]

    result = decision_gate.evaluate_gate(
        message=msg,
        profile=profile,
        analysis=analysis,
        primary_intent="battery_power_issue",
        evidence_cases=evidence,
    )

    assert result.decision == RoutingDecisionType.AUTO_HANDLE
    assert result.is_clear is True
    assert all(result.checklist.values())
