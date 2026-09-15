"""
Tests for backend/app/resolution/evidence_synthesizer.py (Phase 10)
"""
import pytest

from backend.app.resolution.evidence_conflict_detector import EvidenceConflictDetector
from backend.app.resolution.evidence_synthesizer import (
    CaseEvidenceContribution,
    CompositeEvidencePackage,
    ContributionStrength,
    MultiCaseEvidenceSynthesizer,
)
from backend.app.resolution.evidence_validator import (
    CompositeEvidenceVerdict,
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
def synthesizer():
    return MultiCaseEvidenceSynthesizer()


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


def test_passthrough_strong_evidence(synthesizer, validator, battery_profile):
    """When single-case validation already returned STRONG_EVIDENCE, synthesis passes it through directly."""
    cases = [
        RetrievedEvidenceCase(
            case_id="case_strong_1",
            similarity_score=0.90,
            operational_similarity=0.95,
            match_tier=EvidenceMatchTier.DIRECT_PROBLEM_MATCH,
            historical_customer_message="iPhone 7 battery draining fast on iOS 11 update.",
            historical_intent="battery_power_issue",
            retrieval_explanation="Direct problem match.",
        ),
        RetrievedEvidenceCase(
            case_id="case_strong_2",
            similarity_score=0.88,
            operational_similarity=0.90,
            match_tier=EvidenceMatchTier.DIRECT_PROBLEM_MATCH,
            historical_customer_message="Battery drain after iOS 11 update on iPhone.",
            historical_intent="battery_power_issue",
            retrieval_explanation="Direct problem match.",
        ),
    ]
    single_val: EvidenceValidationResult = validator.evaluate_evidence(
        profile=battery_profile,
        primary_intent="battery_power_issue",
        evidence_cases=cases,
    )
    assert single_val.evidence_verdict == EvidenceVerdict.STRONG_EVIDENCE

    pkg: CompositeEvidencePackage = synthesizer.synthesize(
        primary_intent="battery_power_issue",
        profile=battery_profile,
        evidence_cases=cases,
        single_case_validation=single_val,
        dimension_matches=single_val.dimension_matches,
    )

    assert pkg.composite_verdict == CompositeEvidenceVerdict.DIRECT_STRONG_EVIDENCE
    assert pkg.auto_resolution_allowed is True
    assert "Passthrough" in pkg.synthesis_rationale


def test_multi_case_synthesis_composite_strong(synthesizer, validator, battery_profile):
    """
    When no single case is a complete DIRECT match, but multiple cases together cover
    symptom + update context + resolution pattern, synthesis produces COMPOSITE_STRONG_EVIDENCE.
    """
    cases = [
        # Case A covers symptom (battery drain)
        RetrievedEvidenceCase(
            case_id="case_symptom",
            similarity_score=0.72,
            operational_similarity=0.75,
            match_tier=EvidenceMatchTier.RELATED_SYMPTOM,
            historical_customer_message="My phone battery percentage drops from 100 to 20 very fast.",
            historical_intent="battery_power_issue",
            historical_brand_response="Please check Settings > Battery > Battery Health to see max capacity.",
            retrieval_explanation="Related symptom match on battery drain.",
        ),
        # Case B covers contextual trigger (iOS 11 update) & resolution pattern
        RetrievedEvidenceCase(
            case_id="case_context",
            similarity_score=0.70,
            operational_similarity=0.72,
            match_tier=EvidenceMatchTier.RELATED_CONTEXT,
            historical_customer_message="Ever since I updated to iOS 11 my device has issues after update.",
            historical_intent="battery_power_issue",
            historical_brand_response="After an update indexing may take 48 hours. Ensure device is plugged in.",
            retrieval_explanation="Related context on iOS update.",
        ),
    ]
    single_val = validator.evaluate_evidence(
        profile=battery_profile,
        primary_intent="battery_power_issue",
        evidence_cases=cases,
    )
    # Single-case validator gives MODERATE or WEAK (no DIRECT_PROBLEM_MATCH)
    assert single_val.evidence_verdict != EvidenceVerdict.STRONG_EVIDENCE

    pkg: CompositeEvidencePackage = synthesizer.synthesize(
        primary_intent="battery_power_issue",
        profile=battery_profile,
        evidence_cases=cases,
        single_case_validation=single_val,
        dimension_matches=single_val.dimension_matches,
    )

    assert pkg.symptom_covered is True
    assert len(pkg.covered_dimensions) >= 2
    assert "symptom" in pkg.covered_dimensions
    assert pkg.composite_verdict == CompositeEvidenceVerdict.COMPOSITE_STRONG_EVIDENCE
    assert pkg.auto_resolution_allowed is True


def test_mandatory_symptom_coverage_rule(synthesizer, validator, battery_profile):
    """
    NON-NEGOTIABLE SAFETY RULE:
    If cases only cover context/device but NOT the primary symptom,
    synthesis MUST NEVER produce COMPOSITE_STRONG_EVIDENCE.
    """
    cases = [
        # Case only covers update context
        RetrievedEvidenceCase(
            case_id="case_update_only",
            similarity_score=0.60,
            operational_similarity=0.55,
            match_tier=EvidenceMatchTier.RELATED_CONTEXT,
            historical_customer_message="Updated to iOS 11 today on my iPhone 7.",
            historical_intent="general_device_support",
            historical_brand_response="Restart your iPhone 7 after updating.",
            retrieval_explanation="Context only, no symptom.",
        ),
    ]
    single_val = validator.evaluate_evidence(
        profile=battery_profile,
        primary_intent="battery_power_issue",
        evidence_cases=cases,
    )

    pkg = synthesizer.synthesize(
        primary_intent="battery_power_issue",
        profile=battery_profile,
        evidence_cases=cases,
        single_case_validation=single_val,
        dimension_matches=single_val.dimension_matches,
    )

    assert pkg.symptom_covered is False
    assert pkg.composite_verdict != CompositeEvidenceVerdict.COMPOSITE_STRONG_EVIDENCE
    assert pkg.auto_resolution_allowed is False
    assert any("symptom" in r.lower() for r in pkg.why_not_composite_strong)


def test_weak_semantic_matches_cannot_upgrade(synthesizer, validator, battery_profile):
    """
    NON-NEGOTIABLE SAFETY RULE:
    Multiple WEAK_SEMANTIC_MATCH cases must never become STRONG evidence.
    """
    cases = [
        RetrievedEvidenceCase(
            case_id="case_weak_1",
            similarity_score=0.45,
            operational_similarity=0.30,
            match_tier=EvidenceMatchTier.WEAK_SEMANTIC_MATCH,
            historical_customer_message="Apple Support can you help me with my phone?",
            historical_intent="general_device_support",
            retrieval_explanation="Weak match 1.",
        ),
        RetrievedEvidenceCase(
            case_id="case_weak_2",
            similarity_score=0.42,
            operational_similarity=0.28,
            match_tier=EvidenceMatchTier.WEAK_SEMANTIC_MATCH,
            historical_customer_message="Having a problem with my Apple device today.",
            historical_intent="general_device_support",
            retrieval_explanation="Weak match 2.",
        ),
    ]
    single_val = validator.evaluate_evidence(
        profile=battery_profile,
        primary_intent="battery_power_issue",
        evidence_cases=cases,
    )

    pkg = synthesizer.synthesize(
        primary_intent="battery_power_issue",
        profile=battery_profile,
        evidence_cases=cases,
        single_case_validation=single_val,
        dimension_matches=single_val.dimension_matches,
    )

    assert pkg.composite_verdict in (
        CompositeEvidenceVerdict.WEAK_COMPOSITE_EVIDENCE,
        CompositeEvidenceVerdict.INSUFFICIENT_COMPOSITE_EVIDENCE,
    )
    assert pkg.auto_resolution_allowed is False


def test_near_duplicate_deduplication(synthesizer, validator, battery_profile):
    """
    Near-duplicate cases (identical / near-identical customer messages)
    must be deduplicated so that repeated fragments cannot inflate evidence strength.
    """
    cases = [
        RetrievedEvidenceCase(
            case_id="case_orig",
            similarity_score=0.75,
            operational_similarity=0.78,
            match_tier=EvidenceMatchTier.RELATED_SYMPTOM,
            historical_customer_message="My iPhone battery drops 20% in 10 minutes.",
            historical_intent="battery_power_issue",
            historical_brand_response="Check Battery Health in Settings.",
            retrieval_explanation="Original case.",
        ),
        # Near duplicate of case_orig
        RetrievedEvidenceCase(
            case_id="case_dup",
            similarity_score=0.75,
            operational_similarity=0.78,
            match_tier=EvidenceMatchTier.RELATED_SYMPTOM,
            historical_customer_message="My iPhone battery drops 20% in 10 minutes.",
            historical_intent="battery_power_issue",
            historical_brand_response="Check Battery Health in Settings.",
            retrieval_explanation="Duplicate copy.",
        ),
    ]
    single_val = validator.evaluate_evidence(
        profile=battery_profile,
        primary_intent="battery_power_issue",
        evidence_cases=cases,
    )

    pkg = synthesizer.synthesize(
        primary_intent="battery_power_issue",
        profile=battery_profile,
        evidence_cases=cases,
        single_case_validation=single_val,
        dimension_matches=single_val.dimension_matches,
    )

    assert "case_dup" in pkg.duplicate_case_ids
    assert pkg.unique_contributing_cases == 1
    dup_contrib = next(c for c in pkg.case_contributions if c.case_id == "case_dup")
    assert dup_contrib.is_duplicate is True
    assert dup_contrib.allowed_for_synthesis is False


def test_empty_cases_synthesis(synthesizer, validator, battery_profile):
    """Empty cases list produces INSUFFICIENT_COMPOSITE_EVIDENCE cleanly."""
    single_val = validator.evaluate_evidence(
        profile=battery_profile,
        primary_intent="battery_power_issue",
        evidence_cases=[],
    )
    pkg = synthesizer.synthesize(
        primary_intent="battery_power_issue",
        profile=battery_profile,
        evidence_cases=[],
        single_case_validation=single_val,
        dimension_matches=[],
    )
    assert pkg.composite_verdict == CompositeEvidenceVerdict.INSUFFICIENT_COMPOSITE_EVIDENCE
    assert pkg.auto_resolution_allowed is False


def test_case_evidence_contribution_explainability(synthesizer, validator, battery_profile):
    """Every case must have an explicit CaseEvidenceContribution record with explainable notes."""
    cases = [
        RetrievedEvidenceCase(
            case_id="case_exp_1",
            similarity_score=0.75,
            operational_similarity=0.80,
            match_tier=EvidenceMatchTier.RELATED_SYMPTOM,
            historical_customer_message="iPhone battery draining very fast.",
            historical_intent="battery_power_issue",
            historical_brand_response="Go to Settings > Battery.",
            retrieval_explanation="Symptom match.",
        )
    ]
    single_val = validator.evaluate_evidence(
        profile=battery_profile,
        primary_intent="battery_power_issue",
        evidence_cases=cases,
    )
    pkg = synthesizer.synthesize(
        primary_intent="battery_power_issue",
        profile=battery_profile,
        evidence_cases=cases,
        single_case_validation=single_val,
        dimension_matches=single_val.dimension_matches,
    )
    assert len(pkg.case_contributions) == 1
    contrib = pkg.case_contributions[0]
    assert contrib.case_id == "case_exp_1"
    assert contrib.match_tier == EvidenceMatchTier.RELATED_SYMPTOM
    assert contrib.contribution_notes != ""
