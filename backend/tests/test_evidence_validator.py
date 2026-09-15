"""
Tests for backend/app/resolution/evidence_validator.py (Phase 8)
"""
import pytest

from backend.app.resolution.evidence_validator import (
    EvidenceDimensionMatch,
    EvidenceValidationResult,
    EvidenceVerdict,
    ResolutionEvidenceValidator,
)
from backend.app.retrieval.evidence_ranker import (
    EvidenceMatchTier,
    RetrievedEvidenceCase,
)
from backend.app.understanding.problem_extractor import CustomerProblemProfile


@pytest.fixture
def validator():
    return ResolutionEvidenceValidator()


@pytest.fixture
def battery_profile():
    return CustomerProblemProfile(
        device="iPhone 7",
        product_or_service="iOS 11",
        primary_symptom="battery_power",
        possible_cause="software_update_problem",
        update_related=True,
        sufficiency="sufficient",
    )


def test_strong_evidence_validation(validator, battery_profile):
    evidence_cases = [
        RetrievedEvidenceCase(
            case_id="case_101",
            similarity_score=0.85,
            operational_similarity=0.92,
            match_tier=EvidenceMatchTier.DIRECT_PROBLEM_MATCH,
            historical_customer_message="@AppleSupport My iPhone battery is draining extremely fast after updating.",
            historical_intent="battery_power_issue",
            retrieval_explanation="Direct problem match on battery drain.",
        ),
        RetrievedEvidenceCase(
            case_id="case_102",
            similarity_score=0.78,
            operational_similarity=0.84,
            match_tier=EvidenceMatchTier.DIRECT_PROBLEM_MATCH,
            historical_customer_message="@AppleSupport iPhone 7 battery percentage dropping quickly.",
            historical_intent="battery_power_issue",
            retrieval_explanation="Direct match on battery drain symptom.",
        ),
    ]

    result: EvidenceValidationResult = validator.evaluate_evidence(
        profile=battery_profile,
        primary_intent="battery_power_issue",
        evidence_cases=evidence_cases,
    )

    assert result.evidence_verdict == EvidenceVerdict.STRONG_EVIDENCE
    assert result.auto_resolution_allowed is True
    assert result.direct_problem_matches == 2
    assert result.intent_agreement == 1.0
    assert result.symptom_agreement >= 0.80
    assert result.resolution_support_strength >= 0.70


def test_conflicting_evidence_validation(validator, battery_profile):
    evidence_cases = [
        RetrievedEvidenceCase(
            case_id="case_201",
            similarity_score=0.60,
            operational_similarity=0.40,
            match_tier=EvidenceMatchTier.WEAK_SEMANTIC_MATCH,
            historical_customer_message="@AppleSupport I was double billed for my iTunes subscription.",
            historical_intent="billing_purchase_issue",
            retrieval_explanation="Conflicting billing intent.",
        ),
        RetrievedEvidenceCase(
            case_id="case_202",
            similarity_score=0.55,
            operational_similarity=0.35,
            match_tier=EvidenceMatchTier.WEAK_SEMANTIC_MATCH,
            historical_customer_message="@AppleSupport How do I cancel my subscription on iPhone?",
            historical_intent="billing_purchase_issue",
            retrieval_explanation="Billing contradiction.",
        ),
    ]

    result = validator.evaluate_evidence(
        profile=battery_profile,
        primary_intent="battery_power_issue",
        evidence_cases=evidence_cases,
    )

    assert result.evidence_verdict == EvidenceVerdict.CONFLICTING_EVIDENCE
    assert result.auto_resolution_allowed is False


def test_weak_evidence_validation(validator, battery_profile):
    evidence_cases = [
        RetrievedEvidenceCase(
            case_id="case_301",
            similarity_score=0.30,
            operational_similarity=0.20,
            match_tier=EvidenceMatchTier.WEAK_SEMANTIC_MATCH,
            historical_customer_message="@AppleSupport Hi there Apple Support.",
            historical_intent="general_device_support",
            retrieval_explanation="Weak semantic match.",
        )
    ]

    result = validator.evaluate_evidence(
        profile=battery_profile,
        primary_intent="battery_power_issue",
        evidence_cases=evidence_cases,
    )

    assert result.evidence_verdict == EvidenceVerdict.WEAK_EVIDENCE
    assert result.auto_resolution_allowed is False


def test_empty_evidence_validation(validator, battery_profile):
    result = validator.evaluate_evidence(
        profile=battery_profile,
        primary_intent="battery_power_issue",
        evidence_cases=[],
    )

    assert result.evidence_verdict == EvidenceVerdict.INSUFFICIENT_EVIDENCE
    assert result.auto_resolution_allowed is False
