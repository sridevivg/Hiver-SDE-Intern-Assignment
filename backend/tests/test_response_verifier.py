"""
Tests for backend/app/resolution/response_verifier.py (Phase 8)
"""
import pytest

from backend.app.resolution.response_generator import GroundedResponseCandidate
from backend.app.resolution.response_verifier import (
    ResponseGroundingResult,
    ResponseGroundingVerifier,
    VerificationStatus,
)
from backend.app.retrieval.evidence_ranker import (
    EvidenceMatchTier,
    RetrievedEvidenceCase,
)
from backend.app.understanding.problem_extractor import CustomerProblemProfile


@pytest.fixture
def verifier():
    return ResponseGroundingVerifier()


@pytest.fixture
def battery_profile():
    return CustomerProblemProfile(
        device="iPhone 12",
        primary_symptom="battery_power",
        update_related=False,
        sufficiency="sufficient",
    )


def test_valid_response_passes_verification(verifier, battery_profile):
    candidate = GroundedResponseCandidate(
        response_text=(
            "We understand how important battery life is on your iPhone 12. "
            "Please check Settings > Battery > Battery Health to view your maximum capacity. "
            "DM us if you need further help: https://apple.co/DM"
        ),
        grounded_intent="battery_power_issue",
        referenced_case_ids=["case_01"],
        contains_diagnostic_step=True,
        contains_dm_link=True,
    )
    evidence = [
        RetrievedEvidenceCase(
            case_id="case_01",
            similarity_score=0.88,
            operational_similarity=0.92,
            match_tier=EvidenceMatchTier.DIRECT_PROBLEM_MATCH,
            historical_customer_message="@AppleSupport Battery drains fast.",
            historical_intent="battery_power_issue",
        )
    ]

    result: ResponseGroundingResult = verifier.verify_response(
        customer_message="@AppleSupport My battery drains fast.",
        profile=battery_profile,
        primary_intent="battery_power_issue",
        candidate_response=candidate,
        evidence_cases=evidence,
    )

    assert result.verification_status == VerificationStatus.PASS
    assert result.grounded is True
    assert result.addresses_primary_symptom is True
    assert len(result.unsupported_claims) == 0
    assert result.support_score >= 0.70


def test_unsupported_claim_fails_verification(verifier, battery_profile):
    candidate = GroundedResponseCandidate(
        response_text=(
            "We can fix this by helping you jailbreak your iPhone 12 and wipe the entire disk without backup. "
            "We also offer a guaranteed refund: https://apple.co/DM"
        ),
        grounded_intent="battery_power_issue",
        referenced_case_ids=[],
        contains_diagnostic_step=True,
        contains_dm_link=True,
    )

    result = verifier.verify_response(
        customer_message="@AppleSupport My battery drains fast.",
        profile=battery_profile,
        primary_intent="battery_power_issue",
        candidate_response=candidate,
        evidence_cases=[],
    )

    assert result.verification_status == VerificationStatus.FAIL
    assert result.grounded is False
    assert len(result.unsupported_claims) >= 2


def test_symptom_mismatch_fails_verification(verifier, battery_profile):
    candidate = GroundedResponseCandidate(
        response_text=(
            "You can reset your Apple ID password directly at https://iforgot.apple.com. "
            "DM us if you need help: https://apple.co/DM"
        ),
        grounded_intent="battery_power_issue",
        referenced_case_ids=[],
        contains_diagnostic_step=True,
        contains_dm_link=True,
    )

    result = verifier.verify_response(
        customer_message="@AppleSupport My iPhone 12 battery is dead.",
        profile=battery_profile,
        primary_intent="battery_power_issue",
        candidate_response=candidate,
        evidence_cases=[],
    )

    assert result.verification_status == VerificationStatus.FAIL
    assert result.addresses_primary_symptom is False
