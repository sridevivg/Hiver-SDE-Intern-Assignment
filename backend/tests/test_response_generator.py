"""
Tests for backend/app/resolution/response_generator.py (Phase 8)
"""
import pytest

from backend.app.resolution.response_generator import (
    EvidenceGroundedResponseGenerator,
    GroundedResponseCandidate,
)
from backend.app.retrieval.evidence_ranker import (
    EvidenceMatchTier,
    RetrievedEvidenceCase,
)
from backend.app.understanding.problem_extractor import CustomerProblemProfile


@pytest.fixture
def generator():
    return EvidenceGroundedResponseGenerator()


def test_battery_response_generation(generator):
    profile = CustomerProblemProfile(
        device="iPhone 11",
        product_or_service="iOS 15",
        primary_symptom="battery_power",
        possible_cause="",
        update_related=False,
        sufficiency="sufficient",
    )
    evidence = [
        RetrievedEvidenceCase(
            case_id="case_101",
            similarity_score=0.85,
            operational_similarity=0.90,
            match_tier=EvidenceMatchTier.DIRECT_PROBLEM_MATCH,
            historical_customer_message="@AppleSupport Battery draining on iPhone 11.",
            historical_intent="battery_power_issue",
        )
    ]

    candidate: GroundedResponseCandidate = generator.generate_response(
        profile=profile,
        primary_intent="battery_power_issue",
        evidence_cases=evidence,
    )

    assert "iPhone 11" in candidate.response_text
    assert "Settings > Battery > Battery Health" in candidate.response_text
    assert "https://apple.co/DM" in candidate.response_text
    assert "case_101" in candidate.referenced_case_ids
    assert candidate.contains_diagnostic_step is True
    assert candidate.contains_dm_link is True


def test_update_context_tailoring(generator):
    profile = CustomerProblemProfile(
        device="iPhone 7",
        product_or_service="iOS 11",
        primary_symptom="battery_power",
        possible_cause="software_update_problem",
        update_related=True,
        sufficiency="sufficient",
    )
    candidate = generator.generate_response(
        profile=profile,
        primary_intent="battery_power_issue",
        evidence_cases=[],
    )

    assert "following the recent update" in candidate.response_text
    assert "Battery Health" in candidate.response_text
