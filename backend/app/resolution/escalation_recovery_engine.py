"""
SupportGraph AI — Escalation Recovery Engine (Phase 9)

Selectively attempts to resolve RECOVERABLE_ESCALATION cases through the
Phase 8 resolution pipeline without bypassing any safety constraint.

SAFETY CONTRACT (non-negotiable):
  - Only cases classified as EscalationCategory.RECOVERABLE_ESCALATION by the
    EscalationQualityAnalyzer may be submitted to this engine.
  - Any case involving insufficient information, genuine ambiguity, multi-symptom
    complexity, evidence limitations, verification vetoes, or human-required domains
    MUST NOT be passed to this engine.
  - This engine runs the full Phase 8 verification pipeline again from scratch.
  - If ANY safety gate fails during recovery, the case is immediately re-escalated
    with an explicit recovery failure reason.
  - High model confidence NEVER bypasses safety checks.
  - The golden benchmark dataset remains 100% read-only and is never modified.
"""
from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

try:
    from app.core.logging import get_logger
    from app.resolution.escalation_quality_analyzer import (
        EscalationCategory,
        EscalationQualityResult,
    )
    from app.resolution.evidence_validator import EvidenceVerdict
    from app.resolution.response_verifier import VerificationStatus
    from app.resolution.support_resolution_engine import (
        SupportResolutionEngine,
        SupportResolutionResult,
    )
    from app.schemas.intent_routing import RoutingDecisionType
except ModuleNotFoundError:
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]
    from backend.app.resolution.escalation_quality_analyzer import (  # type: ignore[no-redef]
        EscalationCategory,
        EscalationQualityResult,
    )
    from backend.app.resolution.evidence_validator import (  # type: ignore[no-redef]
        EvidenceVerdict,
    )
    from backend.app.resolution.response_verifier import (  # type: ignore[no-redef]
        VerificationStatus,
    )
    from backend.app.resolution.support_resolution_engine import (  # type: ignore[no-redef]
        SupportResolutionEngine,
        SupportResolutionResult,
    )
    from backend.app.schemas.intent_routing import RoutingDecisionType  # type: ignore[no-redef]

logger = get_logger(__name__)


class RecoveryOutcome(str, Enum):
    """Outcome of an escalation recovery attempt."""
    RECOVERED_AUTO_HANDLE = "RECOVERED_AUTO_HANDLE"
    RECOVERY_FAILED_SAFETY = "RECOVERY_FAILED_SAFETY"
    RECOVERY_SKIPPED_NOT_RECOVERABLE = "RECOVERY_SKIPPED_NOT_RECOVERABLE"


class EscalationRecoveryResult(BaseModel):
    """Complete output of a single escalation recovery attempt."""
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    recovery_outcome: RecoveryOutcome = Field(
        ..., description="Result of the recovery attempt"
    )
    original_escalation_category: str = Field(
        ..., description="Phase 9 category of the original escalation"
    )
    customer_message: str = Field(default="", description="Original customer message")
    recovered_resolution: Optional[SupportResolutionResult] = Field(
        default=None,
        description="The recovered AUTO_HANDLE resolution (only if RECOVERED_AUTO_HANDLE)",
    )
    recovery_failure_reasons: list[str] = Field(
        default_factory=list,
        description="Explicit safety reasons why recovery failed (if RECOVERY_FAILED_SAFETY)",
    )
    explanation: str = Field(default="", description="Human-readable recovery outcome explanation")

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(exclude={"recovered_resolution"})


class EscalationRecoveryEngine:
    """
    Selective recovery engine for RECOVERABLE_ESCALATION cases.

    Attempts to re-process a customer message through the full Phase 8 resolution
    pipeline and promotes it to AUTO_HANDLE if and only if all safety constraints
    are simultaneously satisfied in the re-run.

    WHAT THIS ENGINE DOES:
      - Accepts ONLY EscalationCategory.RECOVERABLE_ESCALATION cases.
      - Re-runs the full SupportResolutionEngine pipeline from scratch.
      - Verifies the recovered result independently satisfies auto-handle criteria.
      - Returns the recovered AUTO_HANDLE result if all checks pass.
      - Returns a RECOVERY_FAILED_SAFETY result if any check fails — even if the
        original escalation reason was marginal.

    WHAT THIS ENGINE NEVER DOES:
      - Override insufficient information signals.
      - Override genuine ambiguity or multi-symptom complexity.
      - Override WEAK, INSUFFICIENT, or CONFLICTING evidence verdicts.
      - Override response grounding verification failures.
      - Override hard safety veto signals from the decision gate.
      - Bypass the mandatory response verification step.
    """

    def __init__(
        self,
        engine: Optional[SupportResolutionEngine] = None,
    ) -> None:
        self.engine = engine or SupportResolutionEngine()

    def attempt_recovery(
        self,
        customer_message: str,
        quality_result: EscalationQualityResult,
        case_id: Optional[str] = None,
    ) -> EscalationRecoveryResult:
        """
        Attempt recovery of a single RECOVERABLE_ESCALATION case.

        Args:
            customer_message: The original customer support inquiry text.
            quality_result: The Phase 9 quality classification result (must be RECOVERABLE_ESCALATION).
            case_id: Optional case identifier for audit tracking.

        Returns:
            EscalationRecoveryResult with outcome, recovered resolution (if successful),
            and explicit failure reasons (if recovery was blocked by safety constraints).
        """
        # === SAFETY GATE 1: Only RECOVERABLE_ESCALATION cases may enter ===
        if quality_result.escalation_category != EscalationCategory.RECOVERABLE_ESCALATION:
            return EscalationRecoveryResult(
                recovery_outcome=RecoveryOutcome.RECOVERY_SKIPPED_NOT_RECOVERABLE,
                original_escalation_category=quality_result.escalation_category.value,
                customer_message=customer_message,
                recovered_resolution=None,
                recovery_failure_reasons=[
                    f"Category '{quality_result.escalation_category.value}' is not eligible for recovery. "
                    "Only RECOVERABLE_ESCALATION cases may enter the recovery engine."
                ],
                explanation=(
                    f"Recovery skipped: case is categorized as '{quality_result.escalation_category.value}', "
                    "which requires human review and cannot be safely recovered."
                ),
            )

        if not quality_result.is_recoverable:
            return EscalationRecoveryResult(
                recovery_outcome=RecoveryOutcome.RECOVERY_SKIPPED_NOT_RECOVERABLE,
                original_escalation_category=quality_result.escalation_category.value,
                customer_message=customer_message,
                recovered_resolution=None,
                recovery_failure_reasons=[
                    "Quality result has is_recoverable=False despite RECOVERABLE_ESCALATION category. "
                    "This is a defensive safety check — recovery is rejected."
                ],
                explanation="Recovery skipped: is_recoverable flag is False.",
            )

        logger.info(
            "Attempting escalation recovery for case_id=%s (original category: %s)",
            case_id or "unknown",
            quality_result.escalation_category.value,
        )

        # === RE-RUN FULL PHASE 8 PIPELINE ===
        try:
            recovered: SupportResolutionResult = self.engine.process_message(
                customer_message=customer_message,
                case_id=case_id,
                log_audit=True,
            )
        except Exception as exc:
            logger.error("Recovery pipeline execution failed: %s", exc)
            return EscalationRecoveryResult(
                recovery_outcome=RecoveryOutcome.RECOVERY_FAILED_SAFETY,
                original_escalation_category=quality_result.escalation_category.value,
                customer_message=customer_message,
                recovered_resolution=None,
                recovery_failure_reasons=[f"Pipeline execution error during recovery: {exc}"],
                explanation="Recovery failed due to an internal pipeline error.",
            )

        # === SAFETY GATE 2: Recovered result must be AUTO_HANDLE ===
        if recovered.routing_decision != RoutingDecisionType.AUTO_HANDLE:
            failure_reasons = self._extract_failure_reasons(recovered)
            logger.info(
                "Recovery failed for case_id=%s — re-run still produced ESCALATE_TO_HUMAN.",
                case_id or "unknown",
            )
            return EscalationRecoveryResult(
                recovery_outcome=RecoveryOutcome.RECOVERY_FAILED_SAFETY,
                original_escalation_category=quality_result.escalation_category.value,
                customer_message=customer_message,
                recovered_resolution=None,
                recovery_failure_reasons=failure_reasons,
                explanation=(
                    "Recovery failed: re-running the full Phase 8 pipeline still produced "
                    "ESCALATE_TO_HUMAN. The case remains a genuine escalation."
                ),
            )

        # === SAFETY GATE 3: Independently verify evidence verdict ===
        ev = recovered.evidence_validation
        if ev is None or ev.evidence_verdict not in (
            EvidenceVerdict.STRONG_EVIDENCE, EvidenceVerdict.MODERATE_EVIDENCE
        ):
            ev_val = ev.evidence_verdict.value if ev else "None"
            return EscalationRecoveryResult(
                recovery_outcome=RecoveryOutcome.RECOVERY_FAILED_SAFETY,
                original_escalation_category=quality_result.escalation_category.value,
                customer_message=customer_message,
                recovered_resolution=None,
                recovery_failure_reasons=[
                    f"Recovered result has insufficient evidence: verdict='{ev_val}'. "
                    "Recovery requires STRONG_EVIDENCE or MODERATE_EVIDENCE."
                ],
                explanation="Recovery blocked: insufficient operational evidence in re-run.",
            )

        # === SAFETY GATE 4: Independently verify response grounding ===
        gr = recovered.response_grounding
        if gr is None or gr.verification_status != VerificationStatus.PASS:
            gr_status = gr.verification_status.value if gr else "None"
            return EscalationRecoveryResult(
                recovery_outcome=RecoveryOutcome.RECOVERY_FAILED_SAFETY,
                original_escalation_category=quality_result.escalation_category.value,
                customer_message=customer_message,
                recovered_resolution=None,
                recovery_failure_reasons=[
                    f"Recovered response grounding verification status='{gr_status}'. "
                    "Recovery requires VerificationStatus.PASS."
                ],
                explanation="Recovery blocked: re-run response failed grounding verification.",
            )

        # === ALL SAFETY GATES PASSED — recovery successful ===
        logger.info(
            "Escalation recovery SUCCESSFUL for case_id=%s. "
            "Evidence: %s, Grounding: %s, Intent: %s, Confidence: %.2f",
            case_id or "unknown",
            ev.evidence_verdict.value,
            gr.verification_status.value,
            recovered.primary_intent,
            recovered.confidence,
        )

        return EscalationRecoveryResult(
            recovery_outcome=RecoveryOutcome.RECOVERED_AUTO_HANDLE,
            original_escalation_category=quality_result.escalation_category.value,
            customer_message=customer_message,
            recovered_resolution=recovered,
            recovery_failure_reasons=[],
            explanation=(
                f"Recovery successful. Case re-processed through full Phase 8 pipeline and "
                f"promoted to AUTO_HANDLE. Evidence: {ev.evidence_verdict.value}. "
                f"Grounding: {gr.verification_status.value}. "
                f"Intent: {recovered.primary_intent} (confidence: {recovered.confidence:.2f})."
            ),
        )

    def attempt_batch_recovery(
        self,
        messages_with_quality: list[tuple[str, EscalationQualityResult]],
    ) -> list[EscalationRecoveryResult]:
        """
        Attempt recovery for a batch of (customer_message, quality_result) pairs.

        Only RECOVERABLE_ESCALATION cases are actually processed; all others
        are returned with RECOVERY_SKIPPED_NOT_RECOVERABLE.

        Args:
            messages_with_quality: List of (customer_message, EscalationQualityResult) tuples.

        Returns:
            List of EscalationRecoveryResult in the same order as input.
        """
        results = []
        for msg, qr in messages_with_quality:
            results.append(self.attempt_recovery(customer_message=msg, quality_result=qr))
        return results

    def compute_recovery_stats(
        self,
        recovery_results: list[EscalationRecoveryResult],
    ) -> dict[str, Any]:
        """
        Compute aggregate recovery statistics from a batch of recovery results.

        Returns:
            Dictionary with counts and rates for each recovery outcome.
        """
        total = len(recovery_results)
        recovered = sum(
            1 for r in recovery_results
            if r.recovery_outcome == RecoveryOutcome.RECOVERED_AUTO_HANDLE
        )
        failed_safety = sum(
            1 for r in recovery_results
            if r.recovery_outcome == RecoveryOutcome.RECOVERY_FAILED_SAFETY
        )
        skipped = sum(
            1 for r in recovery_results
            if r.recovery_outcome == RecoveryOutcome.RECOVERY_SKIPPED_NOT_RECOVERABLE
        )
        eligible = recovered + failed_safety  # Skipped cases were never eligible

        return {
            "total_submitted": total,
            "recovered_auto_handle": recovered,
            "recovery_failed_safety": failed_safety,
            "recovery_skipped_not_recoverable": skipped,
            "eligible_for_recovery": eligible,
            "recovery_success_rate": round(recovered / eligible, 4) if eligible > 0 else 0.0,
            "recovery_failure_rate": round(failed_safety / eligible, 4) if eligible > 0 else 0.0,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _extract_failure_reasons(
        self, result: SupportResolutionResult
    ) -> list[str]:
        """Extract the concrete failure reasons from an escalated re-run result."""
        reasons: list[str] = []

        if result.gate_result and result.gate_result.escalation_reasons:
            reasons.extend(result.gate_result.escalation_reasons)

        if result.evidence_validation and not result.evidence_validation.auto_resolution_allowed:
            reasons.extend(result.evidence_validation.reasons)

        if (
            result.response_grounding
            and result.response_grounding.verification_status == VerificationStatus.FAIL
        ):
            reasons.append(
                f"Grounding verification FAIL: {result.response_grounding.explanation}"
            )

        if not reasons:
            reasons.append(
                "Re-run pipeline produced ESCALATE_TO_HUMAN with no explicit reasons extracted."
            )

        return reasons
