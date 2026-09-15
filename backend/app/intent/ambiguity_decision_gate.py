"""
SupportGraph AI — Safe Ambiguity Decision Gate & Veto Policy (Phase 7.1)

Enforces empirical multi-signal gating and safety vetoes:
- Core Rule: CLEAR != HIGH CONFIDENCE.
- High model probability (>90%) is strictly prohibited from overriding missing information,
  weak symptoms, symptom-intent contradictions, or multi-symptom complexity.
- Evaluates 6 independent clarity signals from ClaritySignalEvaluator.
- Produces transparent, explainable decision checklists (✓/✗) and rich escalation packages.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

try:
    from app.core.logging import get_logger
    from app.intent.clarity_signals import (
        CandidateConflictType,
        ClaritySignalEvaluator,
        ClaritySignalsProfile,
        EvidenceAgreementLevel,
        MultiSymptomLevel,
        ProblemStrengthLevel,
        RetrievalQualityLevel,
        SufficiencyLevel,
    )
    from app.retrieval.evidence_ranker import RetrievedEvidenceCase
    from app.schemas.intent_routing import (
        IntentAnalysis,
        IntentPrediction,
        RoutingDecisionType,
    )
    from app.understanding.problem_extractor import CustomerProblemProfile
except ModuleNotFoundError:
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]
    from backend.app.intent.clarity_signals import (  # type: ignore[no-redef]
        CandidateConflictType,
        ClaritySignalEvaluator,
        ClaritySignalsProfile,
        EvidenceAgreementLevel,
        MultiSymptomLevel,
        ProblemStrengthLevel,
        RetrievalQualityLevel,
        SufficiencyLevel,
    )
    from backend.app.retrieval.evidence_ranker import (  # type: ignore[no-redef]
        RetrievedEvidenceCase,
    )
    from backend.app.schemas.intent_routing import (  # type: ignore[no-redef]
        IntentAnalysis,
        IntentPrediction,
        RoutingDecisionType,
    )
    from backend.app.understanding.problem_extractor import (  # type: ignore[no-redef]
        CustomerProblemProfile,
    )

logger = get_logger(__name__)


class DecisionGateResult(BaseModel):
    """Complete output of the Multi-Signal Ambiguity Decision Gate."""
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    decision: RoutingDecisionType = Field(..., description="AUTO_HANDLE or ESCALATE_TO_HUMAN")
    is_clear: bool = Field(..., description="Whether the inquiry passed all clarity criteria")
    primary_intent: str = Field(..., description="Selected primary operational intent")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence score")
    signals: ClaritySignalsProfile = Field(..., description="Evaluated 6-signal clarity profile")
    checklist: dict[str, bool] = Field(default_factory=dict, description="Operational clarity verification checklist")
    explanation: str = Field(default="", description="Detailed human-readable decision explanation")
    escalation_reasons: list[str] = Field(default_factory=list, description="Specific reasons for escalation if any")

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class AmbiguityDecisionGate:
    """
    Multi-signal safety gate that evaluates whether a customer inquiry is clear enough to auto-handle.
    """

    def __init__(self, signal_evaluator: Optional[ClaritySignalEvaluator] = None) -> None:
        self.signal_evaluator = signal_evaluator or ClaritySignalEvaluator()

    def evaluate_gate(
        self,
        message: str,
        profile: CustomerProblemProfile,
        analysis: IntentAnalysis,
        primary_intent: str,
        contextual_cause_intent: Optional[str] = None,
        evidence_cases: Optional[list[RetrievedEvidenceCase]] = None,
    ) -> DecisionGateResult:
        """
        Evaluate inquiry through the multi-signal safe decision gate.
        """
        evidence_cases = evidence_cases or []
        signals: ClaritySignalsProfile = self.signal_evaluator.evaluate(
            message=message,
            profile=profile,
            analysis=analysis,
            primary_intent=primary_intent,
            contextual_cause_intent=contextual_cause_intent,
            evidence_cases=evidence_cases,
        )

        top_conf = analysis.top_confidence
        margin = analysis.confidence_margin
        escalation_reasons: list[str] = []

        # =========================================================================
        # 1. CHECKLIST EVALUATION
        # =========================================================================
        is_info_sufficient = (signals.sufficiency == SufficiencyLevel.SUFFICIENT)
        is_problem_strong = (signals.problem_strength in (ProblemStrengthLevel.STRONG, ProblemStrengthLevel.MODERATE))
        is_conflict_resolved = (signals.candidate_conflict != CandidateConflictType.GENUINE_CONFLICT)
        is_evidence_agreed = (signals.evidence_agreement in (EvidenceAgreementLevel.STRONG_AGREEMENT, EvidenceAgreementLevel.PARTIAL_AGREEMENT))
        is_single_or_related = (signals.multi_symptom != MultiSymptomLevel.UNRELATED_MULTI_SYMPTOM)
        has_sufficient_confidence = (top_conf >= 0.70 and margin >= 0.10) or (
            signals.evidence_agreement == EvidenceAgreementLevel.STRONG_AGREEMENT
            and signals.problem_strength == ProblemStrengthLevel.STRONG
            and top_conf >= 0.60
        )

        checklist = {
            "information_sufficient": is_info_sufficient,
            "problem_strength_valid": is_problem_strong,
            "candidate_conflict_resolved": is_conflict_resolved,
            "symptom_intent_agreed": is_evidence_agreed,
            "multi_symptom_manageable": is_single_or_related,
            "confidence_margin_valid": has_sufficient_confidence,
            "no_safety_veto": not signals.is_vetoed,
        }

        # =========================================================================
        # 2. VETO EVALUATION (Overrides High Confidence)
        # =========================================================================
        if signals.is_vetoed:
            escalation_reasons.extend(signals.veto_reasons)

        if not is_info_sufficient:
            escalation_reasons.append(f"Insufficient information: {signals.sufficiency_reason}")
        if not is_problem_strong:
            escalation_reasons.append(f"Weak primary problem: {signals.strength_reason}")
        if not is_conflict_resolved:
            escalation_reasons.append(f"Unresolved candidate conflict: {signals.conflict_reason}")
        if not is_evidence_agreed:
            escalation_reasons.append(f"Symptom-intent contradiction: {signals.agreement_reason}")
        if not is_single_or_related:
            escalation_reasons.append(f"Unrelated multi-symptom complexity: {signals.multi_symptom_reason}")
        if not has_sufficient_confidence:
            escalation_reasons.append(
                f"Borderline model confidence ({top_conf * 100:.1f}%) or margin ({margin * 100:.1f}%) "
                "insufficient for automated resolution."
            )

        # Deduplicate escalation reasons while preserving order
        unique_reasons: list[str] = []
        for r in escalation_reasons:
            if r not in unique_reasons:
                unique_reasons.append(r)

        # =========================================================================
        # 3. FINAL DECISION
        # =========================================================================
        all_checks_passed = all(checklist.values())

        if all_checks_passed and not signals.is_vetoed:
            decision = RoutingDecisionType.AUTO_HANDLE
            is_clear = True
            explanation = (
                f"Safe auto-handle: All 6 clarity criteria verified. Primary symptom '{profile.primary_symptom}' "
                f"strongly aligns with intent '{primary_intent}' (Conf: {top_conf * 100:.1f}%, Margin: {margin * 100:.1f}%)."
            )
        else:
            decision = RoutingDecisionType.ESCALATE_TO_HUMAN
            is_clear = False
            primary_reason = unique_reasons[0] if unique_reasons else "Ambiguity threshold triggered."
            explanation = (
                f"Escalated to human review: {primary_reason} "
                f"AI confidence ({top_conf * 100:.1f}%) does not override operational safety requirements."
            )

        return DecisionGateResult(
            decision=decision,
            is_clear=is_clear,
            primary_intent=primary_intent,
            confidence=top_conf,
            signals=signals,
            checklist=checklist,
            explanation=explanation,
            escalation_reasons=unique_reasons,
        )
