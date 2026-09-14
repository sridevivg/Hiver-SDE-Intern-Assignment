"""
Unit tests for EscalationRecoveryEngine (Phase 9).

Tests the selective recovery mechanism including:
- Safety gate enforcement (non-recoverable categories are rejected)
- Successful recovery path
- Recovery failure paths (safety constraints caught)
- Batch recovery and stats computation
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

try:
    from app.intent.ambiguity_analyzer import AmbiguityAnalysisResult, AmbiguityType
    from app.resolution.escalation_quality_analyzer import (
        EscalationCategory,
        EscalationQualityResult,
    )
    from app.resolution.escalation_recovery_engine import (
        EscalationRecoveryEngine,
        EscalationRecoveryResult,
        RecoveryOutcome,
    )
    from app.resolution.evidence_validator import EvidenceValidationResult, EvidenceVerdict
    from app.resolution.response_verifier import ResponseGroundingResult, VerificationStatus
    from app.resolution.support_resolution_engine import (
        SupportResolutionEngine,
        SupportResolutionResult,
    )
    from app.schemas.intent_routing import RoutingDecisionType
except ModuleNotFoundError:
    from backend.app.intent.ambiguity_analyzer import AmbiguityAnalysisResult, AmbiguityType
    from backend.app.resolution.escalation_quality_analyzer import (
        EscalationCategory,
        EscalationQualityResult,
    )
    from backend.app.resolution.escalation_recovery_engine import (
        EscalationRecoveryEngine,
        EscalationRecoveryResult,
        RecoveryOutcome,
    )
    from backend.app.resolution.evidence_validator import EvidenceValidationResult, EvidenceVerdict
    from backend.app.resolution.response_verifier import ResponseGroundingResult, VerificationStatus
    from backend.app.resolution.support_resolution_engine import (
        SupportResolutionEngine,
        SupportResolutionResult,
    )
    from backend.app.schemas.intent_routing import RoutingDecisionType


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_recoverable_quality() -> EscalationQualityResult:
    return EscalationQualityResult(
        escalation_category=EscalationCategory.RECOVERABLE_ESCALATION,
        is_recoverable=True,
        classification_rationale="Marginal signal escalation only.",
    )


def _make_non_recoverable_quality(cat: EscalationCategory) -> EscalationQualityResult:
    return EscalationQualityResult(
        escalation_category=cat,
        is_recoverable=False,
        classification_rationale=f"Blocked: {cat.value}",
        recovery_blocking_reasons=[f"Category {cat.value} blocks recovery."],
    )


def _make_strong_evidence() -> EvidenceValidationResult:
    return EvidenceValidationResult(
        evidence_verdict=EvidenceVerdict.STRONG_EVIDENCE,
        auto_resolution_allowed=True,
        direct_problem_matches=2,
        related_symptom_matches=1,
        related_context_matches=0,
        weak_semantic_matches=0,
        intent_agreement=0.9,
        symptom_agreement=0.85,
        evidence_consistency=0.9,
        resolution_support_strength=0.88,
    )


def _make_weak_evidence() -> EvidenceValidationResult:
    return EvidenceValidationResult(
        evidence_verdict=EvidenceVerdict.WEAK_EVIDENCE,
        auto_resolution_allowed=False,
        direct_problem_matches=0,
        related_symptom_matches=0,
        related_context_matches=0,
        weak_semantic_matches=2,
        intent_agreement=0.2,
        symptom_agreement=0.1,
        evidence_consistency=0.3,
        resolution_support_strength=0.15,
        reasons=["Weak evidence."],
    )


def _make_pass_grounding() -> ResponseGroundingResult:
    return ResponseGroundingResult(
        verification_status=VerificationStatus.PASS,
        grounded=True,
        support_score=0.92,
        addresses_primary_symptom=True,
        avoids_cause_overfocus=True,
        unsupported_claims=[],
        explanation="All checks passed.",
    )


def _make_fail_grounding() -> ResponseGroundingResult:
    return ResponseGroundingResult(
        verification_status=VerificationStatus.FAIL,
        grounded=False,
        support_score=0.1,
        addresses_primary_symptom=False,
        avoids_cause_overfocus=False,
        unsupported_claims=["jailbreak"],
        explanation="Grounding failed.",
    )


def _make_auto_handle_result(
    intent: str = "battery_power_issue",
    evidence: EvidenceValidationResult | None = None,
    grounding: ResponseGroundingResult | None = None,
) -> SupportResolutionResult:
    """Build a SupportResolutionResult that looks like a successful AUTO_HANDLE."""
    ambiguity = AmbiguityAnalysisResult(
        ambiguity_type=AmbiguityType.CLEAR_PRIMARY,
        is_ambiguous=False,
        primary_intent=intent,
        ambiguity_reason="Clear primary problem.",
        routing_recommendation=RoutingDecisionType.AUTO_HANDLE,
    )
    # gate_result is Optional[DecisionGateResult] — pass None for test simplicity
    return SupportResolutionResult(
        routing_decision=RoutingDecisionType.AUTO_HANDLE,
        primary_intent=intent,
        confidence=0.82,
        problem_summary="Battery drain on iPhone 12.",
        ambiguity_analysis=ambiguity,
        gate_result=None,
        evidence_validation=evidence or _make_strong_evidence(),
        evidence_cases=[],
        resolution_strategy="Check Battery Health.",
        grounded_response="Hi! We understand your iPhone 12 is experiencing battery drain...",
        response_grounding=grounding or _make_pass_grounding(),
        escalation_package=None,
        explanation="All gates passed.",
    )


def _make_still_escalated_result() -> SupportResolutionResult:
    """SupportResolutionResult that remains ESCALATE_TO_HUMAN after re-run."""
    ambiguity = AmbiguityAnalysisResult(
        ambiguity_type=AmbiguityType.GENUINE_AMBIGUITY,
        is_ambiguous=True,
        primary_intent="battery_power_issue",
        ambiguity_reason="Still ambiguous.",
        routing_recommendation=RoutingDecisionType.ESCALATE_TO_HUMAN,
    )
    return SupportResolutionResult(
        routing_decision=RoutingDecisionType.ESCALATE_TO_HUMAN,
        primary_intent="battery_power_issue",
        confidence=0.51,
        problem_summary="Unclear problem.",
        ambiguity_analysis=ambiguity,
        gate_result=None,
        evidence_validation=_make_weak_evidence(),
        evidence_cases=[],
        resolution_strategy="",
        grounded_response=None,
        response_grounding=_make_fail_grounding(),
        escalation_package=None,
        explanation="Escalated.",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestEscalationRecoveryEngineSafetyGates:
    """Tests enforcing that non-recoverable categories are rejected immediately."""

    def setup_method(self):
        self.mock_engine = MagicMock(spec=SupportResolutionEngine)
        self.recovery = EscalationRecoveryEngine(engine=self.mock_engine)

    @pytest.mark.parametrize("category", [
        EscalationCategory.GENUINE_AMBIGUITY,
        EscalationCategory.MULTI_PROBLEM_COMPLEXITY,
        EscalationCategory.INSUFFICIENT_INFORMATION,
        EscalationCategory.EVIDENCE_LIMITED,
        EscalationCategory.VERIFICATION_VETO,
        EscalationCategory.HUMAN_REQUIRED,
    ])
    def test_non_recoverable_categories_are_skipped(self, category):
        """All non-RECOVERABLE_ESCALATION categories → RECOVERY_SKIPPED_NOT_RECOVERABLE."""
        quality = _make_non_recoverable_quality(category)
        result = self.recovery.attempt_recovery(
            customer_message="test message",
            quality_result=quality,
        )
        assert result.recovery_outcome == RecoveryOutcome.RECOVERY_SKIPPED_NOT_RECOVERABLE
        # Engine must NOT be called for any non-recoverable category
        self.mock_engine.process_message.assert_not_called()

    def test_is_recoverable_false_blocks_recovery(self):
        """Even if category is RECOVERABLE_ESCALATION but is_recoverable=False → skip."""
        quality = EscalationQualityResult(
            escalation_category=EscalationCategory.RECOVERABLE_ESCALATION,
            is_recoverable=False,  # Defensive flag
            classification_rationale="Defensive block.",
        )
        result = self.recovery.attempt_recovery(
            customer_message="test message",
            quality_result=quality,
        )
        assert result.recovery_outcome == RecoveryOutcome.RECOVERY_SKIPPED_NOT_RECOVERABLE


class TestEscalationRecoveryEngineSuccessPath:
    """Tests for successful recovery path."""

    def test_successful_recovery_produces_auto_handle(self):
        """RECOVERABLE_ESCALATION + re-run AUTO_HANDLE → RECOVERED_AUTO_HANDLE."""
        mock_engine = MagicMock(spec=SupportResolutionEngine)
        mock_engine.process_message.return_value = _make_auto_handle_result()
        recovery = EscalationRecoveryEngine(engine=mock_engine)

        quality = _make_recoverable_quality()
        result = recovery.attempt_recovery(
            customer_message="my iPhone battery drains super fast",
            quality_result=quality,
        )

        assert result.recovery_outcome == RecoveryOutcome.RECOVERED_AUTO_HANDLE
        assert result.recovered_resolution is not None
        assert result.recovered_resolution.routing_decision == RoutingDecisionType.AUTO_HANDLE
        assert result.recovery_failure_reasons == []

    def test_successful_recovery_calls_engine_once(self):
        """Recovery engine calls process_message exactly once."""
        mock_engine = MagicMock(spec=SupportResolutionEngine)
        mock_engine.process_message.return_value = _make_auto_handle_result()
        recovery = EscalationRecoveryEngine(engine=mock_engine)

        quality = _make_recoverable_quality()
        recovery.attempt_recovery("test message", quality)
        mock_engine.process_message.assert_called_once()


class TestEscalationRecoveryEngineFailurePaths:
    """Tests for safety-blocked recovery attempts."""

    def test_recovery_fails_if_re_run_still_escalates(self):
        """Re-run still produces ESCALATE_TO_HUMAN → RECOVERY_FAILED_SAFETY."""
        mock_engine = MagicMock(spec=SupportResolutionEngine)
        mock_engine.process_message.return_value = _make_still_escalated_result()
        recovery = EscalationRecoveryEngine(engine=mock_engine)

        quality = _make_recoverable_quality()
        result = recovery.attempt_recovery("test message", quality)

        assert result.recovery_outcome == RecoveryOutcome.RECOVERY_FAILED_SAFETY
        assert len(result.recovery_failure_reasons) > 0
        assert result.recovered_resolution is None

    def test_recovery_fails_if_evidence_insufficient_in_rerun(self):
        """Re-run AUTO_HANDLE but evidence verdict WEAK → RECOVERY_FAILED_SAFETY."""
        mock_engine = MagicMock(spec=SupportResolutionEngine)
        mock_engine.process_message.return_value = _make_auto_handle_result(
            evidence=_make_weak_evidence()
        )
        recovery = EscalationRecoveryEngine(engine=mock_engine)

        quality = _make_recoverable_quality()
        result = recovery.attempt_recovery("test message", quality)

        assert result.recovery_outcome == RecoveryOutcome.RECOVERY_FAILED_SAFETY

    def test_recovery_fails_if_grounding_fails_in_rerun(self):
        """Re-run AUTO_HANDLE but grounding FAIL → RECOVERY_FAILED_SAFETY."""
        mock_engine = MagicMock(spec=SupportResolutionEngine)
        mock_engine.process_message.return_value = _make_auto_handle_result(
            grounding=_make_fail_grounding()
        )
        recovery = EscalationRecoveryEngine(engine=mock_engine)

        quality = _make_recoverable_quality()
        result = recovery.attempt_recovery("test message", quality)

        assert result.recovery_outcome == RecoveryOutcome.RECOVERY_FAILED_SAFETY

    def test_pipeline_exception_returns_failed_safety(self):
        """Engine exception during re-run → RECOVERY_FAILED_SAFETY (no crash)."""
        mock_engine = MagicMock(spec=SupportResolutionEngine)
        mock_engine.process_message.side_effect = RuntimeError("Model unavailable")
        recovery = EscalationRecoveryEngine(engine=mock_engine)

        quality = _make_recoverable_quality()
        result = recovery.attempt_recovery("test message", quality)

        assert result.recovery_outcome == RecoveryOutcome.RECOVERY_FAILED_SAFETY
        assert "Model unavailable" in result.recovery_failure_reasons[0]


class TestRecoveryStats:
    """Tests for recovery statistics computation."""

    def test_compute_recovery_stats_all_successful(self):
        """All recovered → recovery_success_rate = 1.0."""
        results = [
            EscalationRecoveryResult(
                recovery_outcome=RecoveryOutcome.RECOVERED_AUTO_HANDLE,
                original_escalation_category=EscalationCategory.RECOVERABLE_ESCALATION.value,
                customer_message="test",
            )
        ] * 5
        engine = EscalationRecoveryEngine(engine=MagicMock())
        stats = engine.compute_recovery_stats(results)
        assert stats["recovery_success_rate"] == 1.0
        assert stats["recovered_auto_handle"] == 5

    def test_compute_recovery_stats_mixed(self):
        """Mixed outcomes compute correct rates."""
        results = [
            EscalationRecoveryResult(
                recovery_outcome=RecoveryOutcome.RECOVERED_AUTO_HANDLE,
                original_escalation_category=EscalationCategory.RECOVERABLE_ESCALATION.value,
                customer_message="test",
            ),
            EscalationRecoveryResult(
                recovery_outcome=RecoveryOutcome.RECOVERY_FAILED_SAFETY,
                original_escalation_category=EscalationCategory.RECOVERABLE_ESCALATION.value,
                customer_message="test",
            ),
            EscalationRecoveryResult(
                recovery_outcome=RecoveryOutcome.RECOVERY_SKIPPED_NOT_RECOVERABLE,
                original_escalation_category=EscalationCategory.GENUINE_AMBIGUITY.value,
                customer_message="test",
            ),
        ]
        engine = EscalationRecoveryEngine(engine=MagicMock())
        stats = engine.compute_recovery_stats(results)
        assert stats["total_submitted"] == 3
        assert stats["recovered_auto_handle"] == 1
        assert stats["recovery_failed_safety"] == 1
        assert stats["recovery_skipped_not_recoverable"] == 1
        assert stats["eligible_for_recovery"] == 2
        assert stats["recovery_success_rate"] == 0.5
