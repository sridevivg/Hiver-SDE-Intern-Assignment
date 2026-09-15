"""
Tests for backend/app/resolution/evidence_conflict_detector.py (Phase 10)
"""
import pytest

from backend.app.resolution.evidence_conflict_detector import (
    ConflictDetectionResult,
    ConflictType,
    EvidenceConflictDetector,
)
from backend.app.resolution.evidence_validator import (
    CompositeEvidenceVerdict,
    EvidenceDimensionMatch,
    ResolutionEvidenceValidator,
)
from backend.app.retrieval.evidence_ranker import (
    EvidenceMatchTier,
    RetrievedEvidenceCase,
)
from backend.app.understanding.problem_extractor import CustomerProblemProfile


@pytest.fixture
def detector():
    return EvidenceConflictDetector()


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


def test_no_conflicts_detected(detector, validator, battery_profile):
    """Clean evidence with matching intents should have no conflicts."""
    cases = [
        RetrievedEvidenceCase(
            case_id="case_1",
            similarity_score=0.88,
            operational_similarity=0.90,
            match_tier=EvidenceMatchTier.DIRECT_PROBLEM_MATCH,
            historical_customer_message="My iPhone battery drains so fast after update.",
            historical_intent="battery_power_issue",
            retrieval_explanation="Direct problem match.",
        ),
        RetrievedEvidenceCase(
            case_id="case_2",
            similarity_score=0.75,
            operational_similarity=0.80,
            match_tier=EvidenceMatchTier.RELATED_SYMPTOM,
            historical_customer_message="Battery percentage drops 20% in 10 minutes.",
            historical_intent="battery_power_issue",
            retrieval_explanation="Related symptom match.",
        ),
    ]
    dims = [validator.evaluate_dimension(battery_profile, "battery_power_issue", c) for c in cases]

    result: ConflictDetectionResult = detector.detect_conflicts(
        primary_intent="battery_power_issue",
        profile=battery_profile,
        evidence_cases=cases,
        dimension_matches=dims,
    )

    assert result.has_conflict is False
    assert result.conflict_type == ConflictType.NONE
    assert len(result.allowed_case_ids) == 2
    assert len(result.contradicting_case_ids) == 0
    assert result.forced_verdict is None


def test_intent_conflict_detected_and_excluded(detector, validator, battery_profile):
    """Mutually exclusive intent pairs (e.g. battery vs billing) must be flagged and excluded."""
    cases = [
        RetrievedEvidenceCase(
            case_id="case_battery",
            similarity_score=0.80,
            operational_similarity=0.85,
            match_tier=EvidenceMatchTier.DIRECT_PROBLEM_MATCH,
            historical_customer_message="Battery draining fast on iOS 11.",
            historical_intent="battery_power_issue",
            retrieval_explanation="Direct match.",
        ),
        RetrievedEvidenceCase(
            case_id="case_billing",
            similarity_score=0.65,
            operational_similarity=0.40,
            match_tier=EvidenceMatchTier.WEAK_SEMANTIC_MATCH,
            historical_customer_message="I got charged twice for my subscription.",
            historical_intent="billing_purchase_issue",
            retrieval_explanation="Conflicting billing intent.",
        ),
    ]
    dims = [validator.evaluate_dimension(battery_profile, "battery_power_issue", c) for c in cases]

    result = detector.detect_conflicts(
        primary_intent="battery_power_issue",
        profile=battery_profile,
        evidence_cases=cases,
        dimension_matches=dims,
    )

    assert result.has_conflict is True
    assert result.conflict_type == ConflictType.INTENT_CONFLICT
    assert "case_battery" in result.allowed_case_ids
    assert "case_billing" in result.contradicting_case_ids
    assert any("battery_power_issue vs billing_purchase_issue" in p for p in result.conflicting_pairs)


def test_dominant_conflict_forces_verdict(detector, validator, battery_profile):
    """When all cases have conflicting intents, forced verdict must be CONFLICTING_COMPOSITE_EVIDENCE."""
    cases = [
        RetrievedEvidenceCase(
            case_id="case_billing_1",
            similarity_score=0.70,
            operational_similarity=0.30,
            match_tier=EvidenceMatchTier.WEAK_SEMANTIC_MATCH,
            historical_customer_message="Double charge on iTunes receipt.",
            historical_intent="billing_purchase_issue",
            retrieval_explanation="Billing case.",
        ),
        RetrievedEvidenceCase(
            case_id="case_account_2",
            similarity_score=0.65,
            operational_similarity=0.25,
            match_tier=EvidenceMatchTier.WEAK_SEMANTIC_MATCH,
            historical_customer_message="Forgot Apple ID password.",
            historical_intent="account_access_issue",
            retrieval_explanation="Account case.",
        ),
    ]
    dims = [validator.evaluate_dimension(battery_profile, "battery_power_issue", c) for c in cases]

    result = detector.detect_conflicts(
        primary_intent="battery_power_issue",
        profile=battery_profile,
        evidence_cases=cases,
        dimension_matches=dims,
    )

    assert result.has_conflict is True
    assert len(result.allowed_case_ids) == 0
    assert result.forced_verdict == CompositeEvidenceVerdict.CONFLICTING_COMPOSITE_EVIDENCE


def test_empty_cases_conflict_detection(detector, battery_profile):
    """Empty cases list should return no conflict cleanly."""
    result = detector.detect_conflicts(
        primary_intent="battery_power_issue",
        profile=battery_profile,
        evidence_cases=[],
        dimension_matches=[],
    )
    assert result.has_conflict is False
    assert result.conflict_type == ConflictType.NONE
    assert result.forced_verdict is None
