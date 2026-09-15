"""
Tests for Multi-Signal Clarity Evaluator (Phase 7.1)
"""
import pytest

from backend.app.intent.clarity_signals import (
    CandidateConflictType,
    ClaritySignalEvaluator,
    ClaritySignalsProfile,
    EvidenceAgreementLevel,
    MultiSymptomLevel,
    ProblemStrengthLevel,
    RetrievalQualityLevel,
    SufficiencyLevel,
)
from backend.app.retrieval.evidence_ranker import (
    EvidenceMatchTier,
    RetrievedEvidenceCase,
)
from backend.app.schemas.intent_routing import IntentAnalysis, IntentPrediction
from backend.app.understanding.problem_extractor import ProblemExtractor


@pytest.fixture
def extractor() -> ProblemExtractor:
    return ProblemExtractor()


@pytest.fixture
def signal_evaluator() -> ClaritySignalEvaluator:
    return ClaritySignalEvaluator()


def test_clarity_signals_clear_battery(
    extractor: ProblemExtractor,
    signal_evaluator: ClaritySignalEvaluator,
) -> None:
    msg = "My iPhone battery dies within two hours."
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

    signals: ClaritySignalsProfile = signal_evaluator.evaluate(
        message=msg,
        profile=profile,
        analysis=analysis,
        primary_intent="battery_power_issue",
        evidence_cases=evidence,
    )

    assert signals.sufficiency == SufficiencyLevel.SUFFICIENT
    assert signals.problem_strength == ProblemStrengthLevel.STRONG
    assert signals.candidate_conflict == CandidateConflictType.NO_CONFLICT
    assert signals.evidence_agreement == EvidenceAgreementLevel.STRONG_AGREEMENT
    assert signals.retrieval_quality == RetrievalQualityLevel.STRONG_EVIDENCE
    assert signals.multi_symptom == MultiSymptomLevel.SINGLE_SYMPTOM
    assert signals.is_vetoed is False


def test_clarity_signals_insufficient_info(
    extractor: ProblemExtractor,
    signal_evaluator: ClaritySignalEvaluator,
) -> None:
    msg = "Apple please fix this."
    profile = extractor.extract(msg)
    analysis = IntentAnalysis(
        top_predictions=[
            IntentPrediction(intent="general_device_support", confidence=0.95),
        ],
        top_confidence=0.95,
        confidence_margin=0.95,
        model_name="test_model",
    )

    signals = signal_evaluator.evaluate(
        message=msg,
        profile=profile,
        analysis=analysis,
        primary_intent="general_device_support",
    )

    assert signals.sufficiency == SufficiencyLevel.INSUFFICIENT
    assert signals.is_vetoed is True
    assert len(signals.veto_reasons) > 0


def test_clarity_signals_contradictory_evidence(
    extractor: ProblemExtractor,
    signal_evaluator: ClaritySignalEvaluator,
) -> None:
    # WiFi inquiry wrongly predicted as audio issue
    msg = "Cannot connect to WiFi network on iPhone."
    profile = extractor.extract(msg)
    analysis = IntentAnalysis(
        top_predictions=[
            IntentPrediction(intent="hardware_audio_connection_issue", confidence=0.94),
        ],
        top_confidence=0.94,
        confidence_margin=0.90,
        model_name="test_model",
    )

    signals = signal_evaluator.evaluate(
        message=msg,
        profile=profile,
        analysis=analysis,
        primary_intent="hardware_audio_connection_issue",
    )

    assert signals.evidence_agreement == EvidenceAgreementLevel.NO_AGREEMENT
    assert signals.is_vetoed is True


def test_clarity_signals_unrelated_multi_symptom(
    extractor: ProblemExtractor,
    signal_evaluator: ClaritySignalEvaluator,
) -> None:
    msg = "Screen is broken, speaker has no audio, and wifi does not connect."
    profile = extractor.extract(msg)
    analysis = IntentAnalysis(
        top_predictions=[
            IntentPrediction(intent="display_touch_issue", confidence=0.88),
        ],
        top_confidence=0.88,
        confidence_margin=0.70,
        model_name="test_model",
    )

    signals = signal_evaluator.evaluate(
        message=msg,
        profile=profile,
        analysis=analysis,
        primary_intent="display_touch_issue",
    )

    assert signals.multi_symptom == MultiSymptomLevel.UNRELATED_MULTI_SYMPTOM
    assert signals.is_vetoed is True
