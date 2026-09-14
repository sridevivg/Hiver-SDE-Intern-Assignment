"""
Unit tests for EscalationQualityAnalyzer (Phase 9).

Tests the 7-category escalation taxonomy classification under diverse
safety-critical and edge-case scenarios.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

try:
    from app.intent.ambiguity_analyzer import AmbiguityAnalysisResult, AmbiguityType
    from app.intent.ambiguity_decision_gate import DecisionGateResult
    from app.resolution.escalation_quality_analyzer import (
        EscalationCategory,
        EscalationQualityAnalyzer,
        EscalationQualityResult,
    )
    from app.resolution.evidence_validator import EvidenceValidationResult, EvidenceVerdict
    from app.resolution.response_verifier import ResponseGroundingResult, VerificationStatus
    from app.resolution.support_resolution_engine import (
        HumanEscalationPackage,
        SupportResolutionResult,
    )
    from app.schemas.intent_routing import RoutingDecisionType
except ModuleNotFoundError:
    from backend.app.intent.ambiguity_analyzer import AmbiguityAnalysisResult, AmbiguityType
    from backend.app.intent.ambiguity_decision_gate import DecisionGateResult
    from backend.app.resolution.escalation_quality_analyzer import (
        EscalationCategory,
        EscalationQualityAnalyzer,
        EscalationQualityResult,
    )
    from backend.app.resolution.evidence_validator import EvidenceValidationResult, EvidenceVerdict
    from backend.app.resolution.response_verifier import ResponseGroundingResult, VerificationStatus
    from backend.app.resolution.support_resolution_engine import (
        HumanEscalationPackage,
        SupportResolutionResult,
    )
    from backend.app.schemas.intent_routing import RoutingDecisionType


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_ambiguity(ambiguity_type: AmbiguityType, reason: str = "test") -> AmbiguityAnalysisResult:
    return AmbiguityAnalysisResult(
        ambiguity_type=ambiguity_type,
        is_ambiguous=ambiguity_type != AmbiguityType.CLEAR_PRIMARY,
        primary_intent="battery_power_issue",
        ambiguity_reason=reason,
        routing_recommendation=RoutingDecisionType.ESCALATE_TO_HUMAN,
    )


def _make_evidence(
    verdict: EvidenceVerdict,
    direct_matches: int = 0,
    allowed: bool = False,
) -> EvidenceValidationResult:
    return EvidenceValidationResult(
        evidence_verdict=verdict,
        auto_resolution_allowed=allowed,
        direct_problem_matches=direct_matches,
        related_symptom_matches=0,
        related_context_matches=0,
        weak_semantic_matches=0,
        intent_agreement=0.3,
        symptom_agreement=0.2,
        evidence_consistency=0.5,
        resolution_support_strength=0.3,
        reasons=["test reason"],
    )


def _make_grounding(status: VerificationStatus) -> ResponseGroundingResult:
    is_pass = (status == VerificationStatus.PASS)
    return ResponseGroundingResult(
        verification_status=status,
        grounded=is_pass,
        support_score=0.85 if is_pass else 0.2,
        addresses_primary_symptom=is_pass,
        avoids_cause_overfocus=is_pass,
        unsupported_claims=[] if is_pass else ["unsupported claim"],
        explanation="Test grounding result.",
    )


def _make_gate(checklist: dict) -> MagicMock:
    gate = MagicMock(spec=DecisionGateResult)
    gate.checklist = checklist
    gate.escalation_reasons = []
    gate.decision = RoutingDecisionType.ESCALATE_TO_HUMAN
    return gate


def _make_escalated_result(
    ambiguity_type: AmbiguityType = AmbiguityType.CLEAR_PRIMARY,
    evidence_verdict: EvidenceVerdict = EvidenceVerdict.STRONG_EVIDENCE,
    direct_matches: int = 1,
    grounding_status: VerificationStatus = VerificationStatus.PASS,
    primary_intent: str = "battery_power_issue",
    checklist: dict | None = None,
    customer_message: str = "my iPhone battery drains too fast",
) -> SupportResolutionResult:
    """Helper to build a minimal escalated SupportResolutionResult."""
    if checklist is None:
        checklist = {
            "sufficient_information": True,
            "primary_symptom_identified": True,
            "no_safety_critical_issue": True,
            "strong_primary_problem": True,
        }
    ambiguity = _make_ambiguity(ambiguity_type)
    evidence = _make_evidence(evidence_verdict, direct_matches, allowed=False)
    grounding = _make_grounding(grounding_status)
    gate = _make_gate(checklist)

    pkg = MagicMock(spec=HumanEscalationPackage)
    pkg.customer_message = customer_message
    pkg.top_candidates = []

    return SupportResolutionResult(
        routing_decision=RoutingDecisionType.ESCALATE_TO_HUMAN,
        primary_intent=primary_intent,
        confidence=0.55,
        problem_summary="Test summary.",
        ambiguity_analysis=ambiguity,
        gate_result=gate,
        evidence_validation=evidence,
        evidence_cases=[],
        resolution_strategy="Check battery settings.",
        grounded_response=None,
        response_grounding=grounding,
        escalation_package=pkg,
        explanation="Test escalation.",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Tests: EscalationCategory assignment
# ──────────────────────────────────────────────────────────────────────────────

class TestEscalationQualityAnalyzer:
    """Tests for the 7-category classification decision tree."""

    def setup_method(self):
        self.analyzer = EscalationQualityAnalyzer()

    def test_non_escalated_result_is_not_recoverable(self):
        """A non-escalated case is not subject to Phase 9 analysis."""
        auto_result = _make_escalated_result()
        auto_result = auto_result.model_copy(update={"routing_decision": RoutingDecisionType.AUTO_HANDLE})
        quality = self.analyzer.analyze_escalation(auto_result)
        assert quality.is_recoverable is False

    def test_human_required_account_access_intent(self):
        """Escalation with account_access_issue intent → HUMAN_REQUIRED."""
        result = _make_escalated_result(
            primary_intent="account_access_issue",
            evidence_verdict=EvidenceVerdict.STRONG_EVIDENCE,
            direct_matches=2,
            grounding_status=VerificationStatus.PASS,
        )
        quality = self.analyzer.analyze_escalation(result)
        assert quality.escalation_category == EscalationCategory.HUMAN_REQUIRED
        assert quality.is_recoverable is False

    def test_human_required_billing_intent(self):
        """Escalation with billing_purchase_issue intent → HUMAN_REQUIRED."""
        result = _make_escalated_result(primary_intent="billing_purchase_issue")
        quality = self.analyzer.analyze_escalation(result)
        assert quality.escalation_category == EscalationCategory.HUMAN_REQUIRED
        assert quality.is_recoverable is False

    def test_human_required_hardware_damage_keyword(self):
        """Message containing 'cracked' screen keyword → HUMAN_REQUIRED."""
        result = _make_escalated_result(
            customer_message="my iphone screen is cracked and won't turn on"
        )
        quality = self.analyzer.analyze_escalation(result)
        assert quality.escalation_category == EscalationCategory.HUMAN_REQUIRED
        assert quality.is_recoverable is False

    def test_insufficient_information_unclear_insufficient(self):
        """UNCLEAR_INSUFFICIENT ambiguity type → INSUFFICIENT_INFORMATION."""
        result = _make_escalated_result(ambiguity_type=AmbiguityType.UNCLEAR_INSUFFICIENT)
        quality = self.analyzer.analyze_escalation(result)
        assert quality.escalation_category == EscalationCategory.INSUFFICIENT_INFORMATION
        assert quality.is_recoverable is False

    def test_insufficient_information_gate_veto_sufficient(self):
        """Gate veto on sufficient_information → INSUFFICIENT_INFORMATION."""
        result = _make_escalated_result(
            ambiguity_type=AmbiguityType.CLEAR_PRIMARY,
            checklist={
                "sufficient_information": False,
                "primary_symptom_identified": True,
                "no_safety_critical_issue": True,
            },
        )
        quality = self.analyzer.analyze_escalation(result)
        assert quality.escalation_category == EscalationCategory.INSUFFICIENT_INFORMATION
        assert quality.is_recoverable is False

    def test_verification_veto_grounding_fail(self):
        """Grounding verification FAIL → VERIFICATION_VETO."""
        result = _make_escalated_result(
            ambiguity_type=AmbiguityType.CLEAR_PRIMARY,
            evidence_verdict=EvidenceVerdict.STRONG_EVIDENCE,
            direct_matches=2,
            grounding_status=VerificationStatus.FAIL,
        )
        quality = self.analyzer.analyze_escalation(result)
        assert quality.escalation_category == EscalationCategory.VERIFICATION_VETO
        assert quality.is_recoverable is False

    def test_evidence_limited_weak_evidence(self):
        """WEAK_EVIDENCE verdict → EVIDENCE_LIMITED."""
        result = _make_escalated_result(
            ambiguity_type=AmbiguityType.CLEAR_PRIMARY,
            evidence_verdict=EvidenceVerdict.WEAK_EVIDENCE,
            direct_matches=0,
            grounding_status=VerificationStatus.PASS,
        )
        quality = self.analyzer.analyze_escalation(result)
        assert quality.escalation_category == EscalationCategory.EVIDENCE_LIMITED
        assert quality.is_recoverable is False

    def test_evidence_limited_insufficient_evidence(self):
        """INSUFFICIENT_EVIDENCE verdict → EVIDENCE_LIMITED."""
        result = _make_escalated_result(
            evidence_verdict=EvidenceVerdict.INSUFFICIENT_EVIDENCE,
        )
        quality = self.analyzer.analyze_escalation(result)
        assert quality.escalation_category == EscalationCategory.EVIDENCE_LIMITED
        assert quality.is_recoverable is False

    def test_evidence_limited_conflicting_evidence(self):
        """CONFLICTING_EVIDENCE verdict → EVIDENCE_LIMITED."""
        result = _make_escalated_result(
            evidence_verdict=EvidenceVerdict.CONFLICTING_EVIDENCE,
        )
        quality = self.analyzer.analyze_escalation(result)
        assert quality.escalation_category == EscalationCategory.EVIDENCE_LIMITED
        assert quality.is_recoverable is False

    def test_genuine_ambiguity(self):
        """GENUINE_AMBIGUITY ambiguity type → GENUINE_AMBIGUITY category."""
        result = _make_escalated_result(
            ambiguity_type=AmbiguityType.GENUINE_AMBIGUITY,
            evidence_verdict=EvidenceVerdict.STRONG_EVIDENCE,
            direct_matches=2,
            grounding_status=VerificationStatus.PASS,
        )
        quality = self.analyzer.analyze_escalation(result)
        assert quality.escalation_category == EscalationCategory.GENUINE_AMBIGUITY
        assert quality.is_recoverable is False

    def test_multi_problem_complexity(self):
        """MULTI_SYMPTOM ambiguity → MULTI_PROBLEM_COMPLEXITY category."""
        result = _make_escalated_result(
            ambiguity_type=AmbiguityType.MULTI_SYMPTOM,
            evidence_verdict=EvidenceVerdict.STRONG_EVIDENCE,
            direct_matches=2,
            grounding_status=VerificationStatus.PASS,
        )
        quality = self.analyzer.analyze_escalation(result)
        assert quality.escalation_category == EscalationCategory.MULTI_PROBLEM_COMPLEXITY
        assert quality.is_recoverable is False

    def test_recoverable_escalation_all_conditions_met(self):
        """All safety conditions met → RECOVERABLE_ESCALATION."""
        result = _make_escalated_result(
            ambiguity_type=AmbiguityType.CAUSE_VS_SYMPTOM,
            evidence_verdict=EvidenceVerdict.STRONG_EVIDENCE,
            direct_matches=2,
            grounding_status=VerificationStatus.PASS,
            checklist={
                "sufficient_information": True,
                "primary_symptom_identified": True,
                "no_safety_critical_issue": True,
            },
        )
        quality = self.analyzer.analyze_escalation(result)
        assert quality.escalation_category == EscalationCategory.RECOVERABLE_ESCALATION
        assert quality.is_recoverable is True
        assert quality.recovery_blocking_reasons == []

    def test_recoverable_escalation_clear_primary(self):
        """CLEAR_PRIMARY ambiguity + STRONG_EVIDENCE + PASS → RECOVERABLE_ESCALATION."""
        result = _make_escalated_result(
            ambiguity_type=AmbiguityType.CLEAR_PRIMARY,
            evidence_verdict=EvidenceVerdict.STRONG_EVIDENCE,
            direct_matches=1,
            grounding_status=VerificationStatus.PASS,
        )
        quality = self.analyzer.analyze_escalation(result)
        assert quality.escalation_category == EscalationCategory.RECOVERABLE_ESCALATION
        assert quality.is_recoverable is True

    def test_recoverable_requires_direct_matches_with_moderate(self):
        """MODERATE_EVIDENCE without direct matches → NOT recoverable."""
        result = _make_escalated_result(
            ambiguity_type=AmbiguityType.CLEAR_PRIMARY,
            evidence_verdict=EvidenceVerdict.MODERATE_EVIDENCE,
            direct_matches=0,
            grounding_status=VerificationStatus.PASS,
        )
        quality = self.analyzer.analyze_escalation(result)
        # MODERATE_EVIDENCE without direct matches → _is_safely_recoverable blocks it
        assert quality.is_recoverable is False

    def test_high_confidence_alone_not_recoverable(self):
        """
        A case with high model confidence but WEAK_EVIDENCE must NOT be RECOVERABLE.
        This is the core anti-overfitting safety check.
        """
        result = _make_escalated_result(
            ambiguity_type=AmbiguityType.CLEAR_PRIMARY,
            evidence_verdict=EvidenceVerdict.WEAK_EVIDENCE,
            direct_matches=0,
            grounding_status=VerificationStatus.PASS,
        )
        result = result.model_copy(update={"confidence": 0.99})
        quality = self.analyzer.analyze_escalation(result)
        assert quality.escalation_category != EscalationCategory.RECOVERABLE_ESCALATION
        assert quality.is_recoverable is False


class TestEscalationQualityBatchAndDistribution:
    """Tests for batch analysis and distribution computation."""

    def setup_method(self):
        self.analyzer = EscalationQualityAnalyzer()

    def test_batch_skips_auto_handle_cases(self):
        """Batch analysis excludes AUTO_HANDLE cases."""
        escalated = _make_escalated_result(ambiguity_type=AmbiguityType.UNCLEAR_INSUFFICIENT)
        auto = escalated.model_copy(update={"routing_decision": RoutingDecisionType.AUTO_HANDLE})
        results = self.analyzer.analyze_batch([escalated, auto, escalated])
        # Only the 2 ESCALATE_TO_HUMAN cases should be analyzed
        assert len(results) == 2

    def test_distribution_counts_all_categories(self):
        """Distribution dict contains all 7 categories."""
        quality_results = [
            EscalationQualityResult(
                escalation_category=EscalationCategory.EVIDENCE_LIMITED,
                is_recoverable=False,
                classification_rationale="test",
            ),
            EscalationQualityResult(
                escalation_category=EscalationCategory.RECOVERABLE_ESCALATION,
                is_recoverable=True,
                classification_rationale="test",
            ),
        ]
        distribution = self.analyzer.compute_distribution(quality_results)
        assert len(distribution) == 7
        assert distribution[EscalationCategory.EVIDENCE_LIMITED.value] == 1
        assert distribution[EscalationCategory.RECOVERABLE_ESCALATION.value] == 1
        assert distribution[EscalationCategory.GENUINE_AMBIGUITY.value] == 0
