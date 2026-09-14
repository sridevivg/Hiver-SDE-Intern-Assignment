"""
SupportGraph AI — Escalation Quality Analyzer (Phase 9)

Classifies all ESCALATE_TO_HUMAN cases into one of seven mutually exclusive
escalation quality categories to distinguish genuinely unresolvable cases from
cases that might be safely recovered without sacrificing system safety.

ESCALATION TAXONOMY:
  1. GENUINE_AMBIGUITY         — Message is fundamentally vague, multi-interpretable,
                                 or ambiguous across competing operational intents.
  2. MULTI_PROBLEM_COMPLEXITY  — Customer reports multiple distinct problems that
                                 individually require different resolution paths.
  3. INSUFFICIENT_INFORMATION  — Customer message lacks the minimum operational context
                                 needed for safe automated resolution.
  4. EVIDENCE_LIMITED          — Retrieved historical case corpus is too weak or sparse
                                 to support evidence-grounded automated response.
  5. VERIFICATION_VETO         — The generated response failed grounding verification
                                 (primary symptom mismatch or unsupported claims detected).
  6. RECOVERABLE_ESCALATION    — Escalated solely due to marginal confidence/entropy
                                 signals without structural ambiguity, missing information,
                                 or evidence deficiency. The response passes verification.
  7. HUMAN_REQUIRED            — Case involves account security, billing dispute, hardware
                                 damage, or policy matters mandating human judgment.

CRITICAL SAFETY CONSTRAINTS:
  - Categories GENUINE_AMBIGUITY, MULTI_PROBLEM_COMPLEXITY, INSUFFICIENT_INFORMATION,
    EVIDENCE_LIMITED, VERIFICATION_VETO, and HUMAN_REQUIRED MUST NEVER be classified
    as RECOVERABLE_ESCALATION.
  - High model confidence alone NEVER makes a case RECOVERABLE_ESCALATION.
  - RECOVERABLE_ESCALATION requires ALL of the following to be simultaneously true:
      * Gate decision was ESCALATE due to confidence/entropy only (no veto flags)
      * Evidence verdict is STRONG_EVIDENCE or MODERATE_EVIDENCE with direct matches
      * Response grounding verification passes (VerificationStatus.PASS)
      * No safety veto signals present in gate checklist
      * Information sufficiency is NOT insufficient
      * Ambiguity type is NOT GENUINE_AMBIGUITY or UNCLEAR_INSUFFICIENT
"""
from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

try:
    from app.core.logging import get_logger
    from app.intent.ambiguity_analyzer import AmbiguityType
    from app.resolution.evidence_validator import EvidenceValidationResult, EvidenceVerdict
    from app.resolution.response_verifier import ResponseGroundingResult, VerificationStatus
    from app.resolution.support_resolution_engine import (
        HumanEscalationPackage,
        SupportResolutionResult,
    )
    from app.schemas.intent_routing import RoutingDecisionType
except ModuleNotFoundError:
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]
    from backend.app.intent.ambiguity_analyzer import AmbiguityType  # type: ignore[no-redef]
    from backend.app.resolution.evidence_validator import (  # type: ignore[no-redef]
        EvidenceValidationResult,
        EvidenceVerdict,
    )
    from backend.app.resolution.response_verifier import (  # type: ignore[no-redef]
        ResponseGroundingResult,
        VerificationStatus,
    )
    from backend.app.resolution.support_resolution_engine import (  # type: ignore[no-redef]
        HumanEscalationPackage,
        SupportResolutionResult,
    )
    from backend.app.schemas.intent_routing import RoutingDecisionType  # type: ignore[no-redef]

logger = get_logger(__name__)


class EscalationCategory(str, Enum):
    """Phase 9 escalation quality taxonomy — seven mutually exclusive categories."""
    GENUINE_AMBIGUITY = "GENUINE_AMBIGUITY"
    MULTI_PROBLEM_COMPLEXITY = "MULTI_PROBLEM_COMPLEXITY"
    INSUFFICIENT_INFORMATION = "INSUFFICIENT_INFORMATION"
    EVIDENCE_LIMITED = "EVIDENCE_LIMITED"
    VERIFICATION_VETO = "VERIFICATION_VETO"
    RECOVERABLE_ESCALATION = "RECOVERABLE_ESCALATION"
    HUMAN_REQUIRED = "HUMAN_REQUIRED"


# Keyword indicators for human-required policy matters
_HUMAN_REQUIRED_INDICATORS: dict[str, set[str]] = {
    "account_security": {
        "hack", "hacked", "stolen", "unauthorized", "locked out", "compromised",
        "account stolen", "identity", "fraud", "scam",
    },
    "billing_dispute": {
        "charge", "charged", "refund", "dispute", "unauthorized charge",
        "bill", "billing", "overcharged", "fraud charge", "stolen card",
    },
    "hardware_damage": {
        "cracked", "broken", "water damage", "liquid", "dropped",
        "shattered", "physically damaged", "screen cracked", "bent",
    },
    "warranty_service": {
        "warranty", "repair center", "genius bar", "apple store appointment",
        "service appointment", "authorized repair",
    },
}

# Intents whose escalations are inherently human-sensitive
_HUMAN_SENSITIVE_INTENTS: frozenset[str] = frozenset({
    "account_access_issue",
    "billing_purchase_issue",
})


class EscalationQualityResult(BaseModel):
    """Structured output of escalation quality classification for a single case."""
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    escalation_category: EscalationCategory = Field(
        ..., description="Phase 9 escalation quality category"
    )
    is_recoverable: bool = Field(
        ..., description="True only for RECOVERABLE_ESCALATION cases"
    )
    classification_rationale: str = Field(
        ..., description="Explainable justification for the assigned category"
    )
    recovery_blocking_reasons: list[str] = Field(
        default_factory=list,
        description="Explicit reasons why this case cannot be safely recovered (empty for RECOVERABLE_ESCALATION)",
    )
    # Evidence signals captured for downstream analysis
    evidence_verdict: str = Field(default="")
    ambiguity_type: str = Field(default="")
    grounding_status: str = Field(default="")
    has_direct_evidence: bool = Field(default=False)
    confidence: float = Field(default=0.0)
    gate_had_veto: bool = Field(default=False)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class EscalationQualityAnalyzer:
    """
    Analyzes the quality of ESCALATE_TO_HUMAN routing decisions from Phase 8.

    For each escalated SupportResolutionResult, classifies the escalation into
    one of seven categories using a deterministic, safety-preserving decision tree.

    Classification priority (most restrictive first):
      1. HUMAN_REQUIRED (account, billing, hardware damage)
      2. INSUFFICIENT_INFORMATION (missing context / gate veto)
      3. VERIFICATION_VETO (grounding FAIL)
      4. EVIDENCE_LIMITED (WEAK / INSUFFICIENT / CONFLICTING evidence)
      5. GENUINE_AMBIGUITY (ambiguity type GENUINE_AMBIGUITY)
      6. MULTI_PROBLEM_COMPLEXITY (ambiguity type MULTI_SYMPTOM)
      7. RECOVERABLE_ESCALATION (all safety conditions met)
      8. Fallback: GENUINE_AMBIGUITY (conservative)
    """

    def analyze_escalation(
        self,
        result: SupportResolutionResult,
    ) -> EscalationQualityResult:
        """Classify a single ESCALATE_TO_HUMAN result into the Phase 9 taxonomy."""
        if result.routing_decision != RoutingDecisionType.ESCALATE_TO_HUMAN:
            return EscalationQualityResult(
                escalation_category=EscalationCategory.GENUINE_AMBIGUITY,
                is_recoverable=False,
                classification_rationale="Case was not escalated; quality analysis not applicable.",
                recovery_blocking_reasons=["Not an escalated case."],
            )

        pkg: Optional[HumanEscalationPackage] = result.escalation_package
        ev: Optional[EvidenceValidationResult] = result.evidence_validation
        gr: Optional[ResponseGroundingResult] = result.response_grounding
        ambiguity = result.ambiguity_analysis
        gate = result.gate_result

        # --- Extract signals ---
        ambiguity_type = (
            ambiguity.ambiguity_type if ambiguity else AmbiguityType.UNCLEAR_INSUFFICIENT
        )
        evidence_verdict = ev.evidence_verdict if ev else EvidenceVerdict.INSUFFICIENT_EVIDENCE
        grounding_status = gr.verification_status if gr else VerificationStatus.FAIL
        has_direct_evidence = (ev.direct_problem_matches > 0) if ev else False
        confidence = result.confidence

        # Gate safety veto signals
        gate_had_veto = False
        _safety_veto_keys = {"sufficient_information", "no_safety_critical_issue", "primary_symptom_identified"}
        if gate and gate.checklist:
            for key in _safety_veto_keys:
                if key in gate.checklist and not gate.checklist[key]:
                    gate_had_veto = True
                    break

        primary_intent = result.primary_intent or ""
        customer_msg = (pkg.customer_message or "").lower() if pkg else ""

        blocking_reasons: list[str] = []

        # === 1. HUMAN_REQUIRED ===
        if self._is_human_required(primary_intent, customer_msg, blocking_reasons):
            return EscalationQualityResult(
                escalation_category=EscalationCategory.HUMAN_REQUIRED,
                is_recoverable=False,
                classification_rationale=(
                    f"Case involves a human-sensitive domain: {blocking_reasons[0] if blocking_reasons else primary_intent}. "
                    "Policy mandates human judgment for account security, billing disputes, or physical damage."
                ),
                recovery_blocking_reasons=blocking_reasons,
                evidence_verdict=evidence_verdict.value,
                ambiguity_type=ambiguity_type.value,
                grounding_status=grounding_status.value,
                has_direct_evidence=has_direct_evidence,
                confidence=confidence,
                gate_had_veto=gate_had_veto,
            )

        # === 2. INSUFFICIENT_INFORMATION ===
        if self._is_insufficient_information(ambiguity_type, gate_had_veto, gate, blocking_reasons):
            return EscalationQualityResult(
                escalation_category=EscalationCategory.INSUFFICIENT_INFORMATION,
                is_recoverable=False,
                classification_rationale=(
                    "Customer message lacks minimum operational context required for safe resolution. "
                    f"Signals: {'; '.join(blocking_reasons)}"
                ),
                recovery_blocking_reasons=blocking_reasons,
                evidence_verdict=evidence_verdict.value,
                ambiguity_type=ambiguity_type.value,
                grounding_status=grounding_status.value,
                has_direct_evidence=has_direct_evidence,
                confidence=confidence,
                gate_had_veto=gate_had_veto,
            )

        # === 3. VERIFICATION_VETO ===
        if grounding_status == VerificationStatus.FAIL:
            blocking_reasons.append(
                f"Response grounding verification FAILED: {gr.explanation if gr else 'unknown reason'}"
            )
            return EscalationQualityResult(
                escalation_category=EscalationCategory.VERIFICATION_VETO,
                is_recoverable=False,
                classification_rationale=(
                    "The AI-generated response failed grounding verification — "
                    "it either did not address the primary symptom or contained unsupported claims."
                ),
                recovery_blocking_reasons=blocking_reasons,
                evidence_verdict=evidence_verdict.value,
                ambiguity_type=ambiguity_type.value,
                grounding_status=grounding_status.value,
                has_direct_evidence=has_direct_evidence,
                confidence=confidence,
                gate_had_veto=gate_had_veto,
            )

        # === 4. EVIDENCE_LIMITED ===
        if self._is_evidence_limited(evidence_verdict, blocking_reasons):
            return EscalationQualityResult(
                escalation_category=EscalationCategory.EVIDENCE_LIMITED,
                is_recoverable=False,
                classification_rationale=(
                    f"Evidence verdict '{evidence_verdict.value}' is insufficient for safe automated resolution. "
                    f"{'; '.join(blocking_reasons)}"
                ),
                recovery_blocking_reasons=blocking_reasons,
                evidence_verdict=evidence_verdict.value,
                ambiguity_type=ambiguity_type.value,
                grounding_status=grounding_status.value,
                has_direct_evidence=has_direct_evidence,
                confidence=confidence,
                gate_had_veto=gate_had_veto,
            )

        # === 5. GENUINE_AMBIGUITY ===
        if ambiguity_type == AmbiguityType.GENUINE_AMBIGUITY:
            blocking_reasons.append(
                f"Ambiguity type GENUINE_AMBIGUITY: {ambiguity.ambiguity_reason if ambiguity else 'no reason provided'}"
            )
            return EscalationQualityResult(
                escalation_category=EscalationCategory.GENUINE_AMBIGUITY,
                is_recoverable=False,
                classification_rationale=(
                    "Customer message is fundamentally ambiguous across competing operational intents. "
                    "The system cannot safely select a single resolution path without human clarification."
                ),
                recovery_blocking_reasons=blocking_reasons,
                evidence_verdict=evidence_verdict.value,
                ambiguity_type=ambiguity_type.value,
                grounding_status=grounding_status.value,
                has_direct_evidence=has_direct_evidence,
                confidence=confidence,
                gate_had_veto=gate_had_veto,
            )

        # === 6. MULTI_PROBLEM_COMPLEXITY ===
        if ambiguity_type == AmbiguityType.MULTI_SYMPTOM:
            blocking_reasons.append(
                "Ambiguity type MULTI_SYMPTOM: customer reports multiple distinct hardware/software problems."
            )
            return EscalationQualityResult(
                escalation_category=EscalationCategory.MULTI_PROBLEM_COMPLEXITY,
                is_recoverable=False,
                classification_rationale=(
                    "Customer describes multiple distinct operational problems requiring independent resolution paths. "
                    "Consolidated auto-handling risks applying the wrong troubleshooting steps."
                ),
                recovery_blocking_reasons=blocking_reasons,
                evidence_verdict=evidence_verdict.value,
                ambiguity_type=ambiguity_type.value,
                grounding_status=grounding_status.value,
                has_direct_evidence=has_direct_evidence,
                confidence=confidence,
                gate_had_veto=gate_had_veto,
            )

        # === 7. RECOVERABLE_ESCALATION ===
        if self._is_safely_recoverable(
            ambiguity_type, evidence_verdict, grounding_status,
            has_direct_evidence, gate_had_veto, ev, blocking_reasons
        ):
            return EscalationQualityResult(
                escalation_category=EscalationCategory.RECOVERABLE_ESCALATION,
                is_recoverable=True,
                classification_rationale=(
                    f"Escalated due to marginal confidence/entropy signals only — no structural ambiguity, "
                    f"missing information, evidence gaps, or verification failures. "
                    f"Ambiguity: {ambiguity_type.value}. Evidence: {evidence_verdict.value}. "
                    f"Confidence: {confidence:.2f}. Grounding: PASS."
                ),
                recovery_blocking_reasons=[],
                evidence_verdict=evidence_verdict.value,
                ambiguity_type=ambiguity_type.value,
                grounding_status=grounding_status.value,
                has_direct_evidence=has_direct_evidence,
                confidence=confidence,
                gate_had_veto=gate_had_veto,
            )

        # === Fallback: Conservative GENUINE_AMBIGUITY ===
        blocking_reasons.append("No specific category matched; conservatively classified as GENUINE_AMBIGUITY.")
        return EscalationQualityResult(
            escalation_category=EscalationCategory.GENUINE_AMBIGUITY,
            is_recoverable=False,
            classification_rationale=(
                "Case could not be assigned to a specific category. "
                "Conservatively classified as GENUINE_AMBIGUITY to ensure human review."
            ),
            recovery_blocking_reasons=blocking_reasons,
            evidence_verdict=evidence_verdict.value if ev else "",
            ambiguity_type=ambiguity_type.value,
            grounding_status=grounding_status.value if gr else "",
            has_direct_evidence=has_direct_evidence,
            confidence=confidence,
            gate_had_veto=gate_had_veto,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Private classification helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _is_human_required(
        self,
        primary_intent: str,
        customer_msg: str,
        blocking_reasons: list[str],
    ) -> bool:
        if primary_intent in _HUMAN_SENSITIVE_INTENTS:
            blocking_reasons.append(
                f"Intent '{primary_intent}' is a human-sensitive domain "
                "(account security or billing requires human judgment)."
            )
            return True
        for domain, keywords in _HUMAN_REQUIRED_INDICATORS.items():
            for kw in keywords:
                if kw in customer_msg:
                    blocking_reasons.append(
                        f"Customer message contains human-required keyword '{kw}' "
                        f"associated with domain '{domain}'."
                    )
                    return True
        return False

    def _is_insufficient_information(
        self,
        ambiguity_type: AmbiguityType,
        gate_had_veto: bool,
        gate,
        blocking_reasons: list[str],
    ) -> bool:
        if ambiguity_type == AmbiguityType.UNCLEAR_INSUFFICIENT:
            blocking_reasons.append(
                "Ambiguity type UNCLEAR_INSUFFICIENT: message lacks minimum operational context."
            )
            return True
        if gate_had_veto and gate and gate.checklist:
            if not gate.checklist.get("sufficient_information", True):
                blocking_reasons.append(
                    "Gate veto: 'sufficient_information' is False — "
                    "customer message lacks device, symptom, or operational context."
                )
                return True
            if not gate.checklist.get("primary_symptom_identified", True):
                blocking_reasons.append(
                    "Gate veto: 'primary_symptom_identified' is False — "
                    "no concrete operational symptom extracted from the message."
                )
                return True
        return False

    def _is_evidence_limited(
        self,
        evidence_verdict: EvidenceVerdict,
        blocking_reasons: list[str],
    ) -> bool:
        if evidence_verdict == EvidenceVerdict.INSUFFICIENT_EVIDENCE:
            blocking_reasons.append(
                "Evidence verdict INSUFFICIENT_EVIDENCE: no relevant historical cases retrieved."
            )
            return True
        if evidence_verdict == EvidenceVerdict.CONFLICTING_EVIDENCE:
            blocking_reasons.append(
                "Evidence verdict CONFLICTING_EVIDENCE: retrieved cases contradict primary intent."
            )
            return True
        if evidence_verdict == EvidenceVerdict.WEAK_EVIDENCE:
            blocking_reasons.append(
                "Evidence verdict WEAK_EVIDENCE: only weak lexical matches; no operational alignment."
            )
            return True
        return False

    def _is_safely_recoverable(
        self,
        ambiguity_type: AmbiguityType,
        evidence_verdict: EvidenceVerdict,
        grounding_status: VerificationStatus,
        has_direct_evidence: bool,
        gate_had_veto: bool,
        ev: Optional[EvidenceValidationResult],
        blocking_reasons: list[str],
    ) -> bool:
        """
        Returns True ONLY when ALL safety conditions for recovery are simultaneously met.
        """
        # Must not be a structural ambiguity type
        if ambiguity_type in (
            AmbiguityType.GENUINE_AMBIGUITY,
            AmbiguityType.UNCLEAR_INSUFFICIENT,
            AmbiguityType.MULTI_SYMPTOM,
        ):
            blocking_reasons.append(f"Ambiguity type {ambiguity_type.value} blocks recovery.")
            return False

        # Evidence must be STRONG or MODERATE
        if evidence_verdict not in (EvidenceVerdict.STRONG_EVIDENCE, EvidenceVerdict.MODERATE_EVIDENCE):
            blocking_reasons.append(f"Evidence verdict {evidence_verdict.value} insufficient for recovery.")
            return False

        # MODERATE_EVIDENCE alone (without any direct match) is not sufficient
        if evidence_verdict == EvidenceVerdict.MODERATE_EVIDENCE and not has_direct_evidence:
            blocking_reasons.append(
                "MODERATE_EVIDENCE without direct problem matches is insufficient for recovery."
            )
            return False

        # Grounding must PASS
        if grounding_status != VerificationStatus.PASS:
            blocking_reasons.append("Response grounding verification did not PASS.")
            return False

        # No hard safety veto
        if gate_had_veto:
            blocking_reasons.append("Gate had at least one hard safety veto signal.")
            return False

        return True

    def analyze_batch(
        self,
        results: list[SupportResolutionResult],
    ) -> list[EscalationQualityResult]:
        """Classify all escalated cases in a batch. AUTO_HANDLE cases are skipped."""
        return [
            self.analyze_escalation(res)
            for res in results
            if res.routing_decision == RoutingDecisionType.ESCALATE_TO_HUMAN
        ]

    def compute_distribution(
        self,
        quality_results: list[EscalationQualityResult],
    ) -> dict[str, int]:
        """Return category count distribution from a batch of quality results."""
        distribution: dict[str, int] = {cat.value: 0 for cat in EscalationCategory}
        for r in quality_results:
            distribution[r.escalation_category.value] += 1
        return distribution
