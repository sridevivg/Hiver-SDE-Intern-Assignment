"""
SupportGraph AI — Multi-Signal Clarity Analysis Layer (Phase 7.1)

Evaluates 6 independent operational signals to determine whether customer inquiries
are genuinely clear or ambiguous:
- Signal A: Information Sufficiency (SUFFICIENT, PARTIAL, INSUFFICIENT)
- Signal B: Primary Problem Strength (STRONG, MODERATE, WEAK)
- Signal C: Candidate Conflict (NO_CONFLICT, CAUSE_VS_SYMPTOM, GENUINE_CONFLICT)
- Signal D: Operational Evidence Agreement (STRONG_AGREEMENT, PARTIAL_AGREEMENT, NO_AGREEMENT)
- Signal E: Retrieval Evidence Quality (STRONG_EVIDENCE, MODERATE_EVIDENCE, WEAK_EVIDENCE, NO_EVIDENCE)
- Signal F: Multi-Symptom Complexity (SINGLE_SYMPTOM, RELATED_MULTI_SYMPTOM, UNRELATED_MULTI_SYMPTOM)

Core Principle: CLEAR != HIGH CONFIDENCE. High probability must never override contradictory signals.
"""
from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

try:
    from app.core.logging import get_logger
    from app.retrieval.evidence_ranker import EvidenceMatchTier, RetrievedEvidenceCase
    from app.schemas.intent_routing import IntentAnalysis, IntentPrediction
    from app.understanding.problem_extractor import CustomerProblemProfile
except ModuleNotFoundError:
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]
    from backend.app.retrieval.evidence_ranker import (  # type: ignore[no-redef]
        EvidenceMatchTier,
        RetrievedEvidenceCase,
    )
    from backend.app.schemas.intent_routing import (  # type: ignore[no-redef]
        IntentAnalysis,
        IntentPrediction,
    )
    from backend.app.understanding.problem_extractor import (  # type: ignore[no-redef]
        CustomerProblemProfile,
    )

logger = get_logger(__name__)


class SufficiencyLevel(str, Enum):
    """Signal A: Information sufficiency level."""
    SUFFICIENT = "SUFFICIENT"
    PARTIAL = "PARTIAL"
    INSUFFICIENT = "INSUFFICIENT"


class ProblemStrengthLevel(str, Enum):
    """Signal B: Primary problem specificity and strength."""
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"


class CandidateConflictType(str, Enum):
    """Signal C: Nature of candidate intent competition."""
    NO_CONFLICT = "NO_CONFLICT"
    CAUSE_VS_SYMPTOM = "CAUSE_VS_SYMPTOM"
    GENUINE_CONFLICT = "GENUINE_CONFLICT"


class EvidenceAgreementLevel(str, Enum):
    """Signal D: Agreement between extracted symptom and predicted intent."""
    STRONG_AGREEMENT = "STRONG_AGREEMENT"
    PARTIAL_AGREEMENT = "PARTIAL_AGREEMENT"
    NO_AGREEMENT = "NO_AGREEMENT"


class RetrievalQualityLevel(str, Enum):
    """Signal E: Quality of historical support evidence."""
    STRONG_EVIDENCE = "STRONG_EVIDENCE"
    MODERATE_EVIDENCE = "MODERATE_EVIDENCE"
    WEAK_EVIDENCE = "WEAK_EVIDENCE"
    NO_EVIDENCE = "NO_EVIDENCE"


class MultiSymptomLevel(str, Enum):
    """Signal F: Multi-symptom complexity."""
    SINGLE_SYMPTOM = "SINGLE_SYMPTOM"
    RELATED_MULTI_SYMPTOM = "RELATED_MULTI_SYMPTOM"
    UNRELATED_MULTI_SYMPTOM = "UNRELATED_MULTI_SYMPTOM"


# Mapping from operational intents to primary symptom keyword patterns
INTENT_SYMPTOM_MAPPING = {
    "battery_power_issue": re.compile(
        r"\b(battery|drain|draining|charge|charging|power|percentage|overheating|dies|dead)\b",
        re.IGNORECASE,
    ),
    "software_update_problem": re.compile(
        r"\b(update|updated|updating|ios\s*11|ios|macos|install|verify|restore|slow|lag|freeze|crash)\b",
        re.IGNORECASE,
    ),
    "hardware_audio_connection_issue": re.compile(
        r"\b(speaker|sound|audio|volume|crackling|mic|microphone|loudspeaker|airpod|airpods|earpod|earpods|headphone|headphones|bluetooth|pair|pairing)\b",
        re.IGNORECASE,
    ),
    "display_touch_issue": re.compile(
        r"\b(screen|display|cracked|black\s+screen|flicker|touch|unresponsive|digitizer|lines)\b",
        re.IGNORECASE,
    ),
    "keyboard_typing_issue": re.compile(
        r"\b(keyboard|type|typing|autocorrect|letter\s+i|predictive|keys|shortcuts)\b",
        re.IGNORECASE,
    ),
    "account_access_issue": re.compile(
        r"\b(password|passcode|apple\s*id|icloud|locked|disabled|login|sign\s+in|2fa|verification)\b",
        re.IGNORECASE,
    ),
    "billing_purchase_issue": re.compile(
        r"\b(charge|charged|bill|billing|refund|subscription|itunes|app\s+store|receipt|renewal)\b",
        re.IGNORECASE,
    ),
    "mac_software_issue": re.compile(
        r"\b(macbook|imac|mac\s*pro|macos|safari|high\s*sierra|sierra|disk\s*utility|finder)\b",
        re.IGNORECASE,
    ),
    "general_device_support": re.compile(
        r"\b(wifi|wi-fi|cellular|network|signal|restart|settings|device|phone|setup)\b",
        re.IGNORECASE,
    ),
}

WEAK_PROBLEM_PATTERNS = re.compile(
    r"\b(phone\s+is\s+weird|something\s+is\s+broken|please\s+fix|broken|messed\s+up|help|issue|bug|problem|bad)\b",
    re.IGNORECASE,
)


class ClaritySignalsProfile(BaseModel):
    """Comprehensive multi-signal evaluation profile for an incoming inquiry."""
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    sufficiency: SufficiencyLevel = Field(..., description="Signal A: Information Sufficiency")
    problem_strength: ProblemStrengthLevel = Field(..., description="Signal B: Primary Problem Strength")
    candidate_conflict: CandidateConflictType = Field(..., description="Signal C: Candidate Conflict")
    evidence_agreement: EvidenceAgreementLevel = Field(..., description="Signal D: Operational Evidence Agreement")
    retrieval_quality: RetrievalQualityLevel = Field(..., description="Signal E: Retrieval Evidence Quality")
    multi_symptom: MultiSymptomLevel = Field(..., description="Signal F: Multi-Symptom Complexity")

    # Details & Explanations
    sufficiency_reason: str = Field(default="", description="Why sufficiency was assigned")
    strength_reason: str = Field(default="", description="Why problem strength was assigned")
    conflict_reason: str = Field(default="", description="Why candidate conflict was assigned")
    agreement_reason: str = Field(default="", description="Why evidence agreement was assigned")
    retrieval_reason: str = Field(default="", description="Why retrieval quality was assigned")
    multi_symptom_reason: str = Field(default="", description="Why multi-symptom complexity was assigned")

    # Veto check
    is_vetoed: bool = Field(default=False, description="Whether a hard safety veto was triggered")
    veto_reasons: list[str] = Field(default_factory=list, description="Specific safety veto violations")

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class ClaritySignalEvaluator:
    """
    Evaluator that extracts and evaluates all 6 operational clarity signals.
    """

    def evaluate(
        self,
        message: str,
        profile: CustomerProblemProfile,
        analysis: IntentAnalysis,
        primary_intent: str,
        contextual_cause_intent: Optional[str] = None,
        evidence_cases: Optional[list[RetrievedEvidenceCase]] = None,
    ) -> ClaritySignalsProfile:
        """
        Evaluate all 6 operational signals for a customer message.
        """
        evidence_cases = evidence_cases or []
        msg_clean = message.strip()
        msg_lower = msg_clean.lower()
        words = msg_clean.split()
        veto_reasons: list[str] = []

        # Safety Veto: Urgent physical/thermal hardware hazard
        HAZARD_PATTERNS = re.compile(
            r"\b(smoking|smoke|sparking|sparks|burning|burnt|fire|flames|swelling|swollen|bulging|hissing|popping)\b",
            re.IGNORECASE,
        )
        if HAZARD_PATTERNS.search(msg_lower):
            veto_reasons.append("VETO: Urgent hardware/thermal hazard detected in customer inquiry.")

        # =========================================================================
        # SIGNAL A: INFORMATION SUFFICIENCY
        # =========================================================================
        if profile.information_sufficiency == "insufficient" or (len(words) <= 4 and not profile.hardware_related and not profile.financial_related):
            sufficiency = SufficiencyLevel.INSUFFICIENT
            suff_reason = "Customer message is too brief or lacks operational details to identify a concrete problem."
            veto_reasons.append("VETO: Insufficient operational information to auto-handle.")
        elif profile.information_sufficiency == "partial" or len(words) < 7 and "unspecified" in profile.primary_symptom.lower():
            sufficiency = SufficiencyLevel.PARTIAL
            suff_reason = "Customer message provides partial context but lacks full operational specificity."
        else:
            sufficiency = SufficiencyLevel.SUFFICIENT
            suff_reason = "Customer message provides sufficient detail and operational context."

        # =========================================================================
        # SIGNAL B: PRIMARY PROBLEM STRENGTH
        # =========================================================================
        sym_lower = profile.primary_symptom.lower()
        if "unspecified" in sym_lower or (WEAK_PROBLEM_PATTERNS.fullmatch(msg_clean) and not profile.hardware_related):
            problem_strength = ProblemStrengthLevel.WEAK
            strength_reason = f"Primary symptom '{profile.primary_symptom}' is non-specific or vague."
            veto_reasons.append("VETO: Primary problem is too weak or non-specific.")
        elif profile.primary_symptom == "general post-update instability or sluggishness" and len(profile.secondary_symptoms) == 0 and len(words) < 8:
            problem_strength = ProblemStrengthLevel.MODERATE
            strength_reason = "Problem describes general post-update behavior without a specific component failure."
        else:
            problem_strength = ProblemStrengthLevel.STRONG
            strength_reason = f"Concrete operational symptom identified: '{profile.primary_symptom}'."

        # =========================================================================
        # SIGNAL C: CANDIDATE CONFLICT
        # =========================================================================
        top_1 = analysis.top_1_intent
        top_2 = analysis.top_2_intent
        margin = analysis.confidence_margin

        if contextual_cause_intent is not None and primary_intent != contextual_cause_intent:
            candidate_conflict = CandidateConflictType.CAUSE_VS_SYMPTOM
            conflict_reason = (
                f"Candidate competition reflects cause vs symptom: primary symptom '{primary_intent}' "
                f"with contextual cause '{contextual_cause_intent}'."
            )
        elif top_2 is not None and margin < 0.15 and top_1 != "unclear_needs_review":
            candidate_conflict = CandidateConflictType.GENUINE_CONFLICT
            conflict_reason = (
                f"Genuine conflict between competing candidates '{top_1}' and '{top_2}' (margin: {margin:.2f})."
            )
            if problem_strength == ProblemStrengthLevel.WEAK or sufficiency != SufficiencyLevel.SUFFICIENT:
                veto_reasons.append(f"VETO: Unresolved candidate conflict ({top_1} vs {top_2}).")
        else:
            candidate_conflict = CandidateConflictType.NO_CONFLICT
            conflict_reason = f"Decisive candidate separation for '{primary_intent}' (margin: {margin:.2f})."

        # =========================================================================
        # SIGNAL D: OPERATIONAL EVIDENCE AGREEMENT
        # =========================================================================
        pat = INTENT_SYMPTOM_MAPPING.get(primary_intent)
        if pat and (pat.search(sym_lower) or pat.search(msg_lower)):
            evidence_agreement = EvidenceAgreementLevel.STRONG_AGREEMENT
            agreement_reason = f"Extracted symptom '{profile.primary_symptom}' strongly aligns with intent '{primary_intent}'."
        elif primary_intent == "general_device_support" and profile.device is not None:
            evidence_agreement = EvidenceAgreementLevel.PARTIAL_AGREEMENT
            agreement_reason = "General device inquiry with recognized device hardware/system context."
        elif primary_intent == "software_update_problem" and profile.update_related:
            evidence_agreement = EvidenceAgreementLevel.STRONG_AGREEMENT
            agreement_reason = "Update-related inquiry aligns with software_update_problem intent."
        else:
            evidence_agreement = EvidenceAgreementLevel.NO_AGREEMENT
            agreement_reason = (
                f"Extracted symptom '{profile.primary_symptom}' does not support predicted intent '{primary_intent}'."
            )
            veto_reasons.append(f"VETO: Symptom-intent contradiction ({profile.primary_symptom} vs {primary_intent}).")

        # =========================================================================
        # SIGNAL E: RETRIEVAL EVIDENCE QUALITY
        # =========================================================================
        if not evidence_cases:
            retrieval_quality = RetrievalQualityLevel.NO_EVIDENCE
            retrieval_reason = "No historical evidence retrieved."
        else:
            has_direct = any(c.match_tier == EvidenceMatchTier.DIRECT_PROBLEM_MATCH for c in evidence_cases)
            has_related_symptom = any(c.match_tier == EvidenceMatchTier.RELATED_SYMPTOM for c in evidence_cases)
            has_related_context = any(c.match_tier == EvidenceMatchTier.RELATED_CONTEXT for c in evidence_cases)

            if has_direct:
                retrieval_quality = RetrievalQualityLevel.STRONG_EVIDENCE
                retrieval_reason = "Historical corpus contains direct operational problem match cases."
            elif has_related_symptom:
                retrieval_quality = RetrievalQualityLevel.MODERATE_EVIDENCE
                retrieval_reason = "Historical corpus contains related symptom match cases."
            elif has_related_context:
                retrieval_quality = RetrievalQualityLevel.WEAK_EVIDENCE
                retrieval_reason = "Historical corpus contains related context match cases."
            else:
                retrieval_quality = RetrievalQualityLevel.NO_EVIDENCE
                retrieval_reason = "Retrieved historical cases share only weak semantic keywords without operational match."

        # =========================================================================
        # SIGNAL F: MULTI-SYMPTOM COMPLEXITY
        # =========================================================================
        distinct_symptoms_count = 1 + len(profile.secondary_symptoms)
        if distinct_symptoms_count >= 3:
            multi_symptom = MultiSymptomLevel.UNRELATED_MULTI_SYMPTOM
            multi_reason = (
                f"Customer reports {distinct_symptoms_count} distinct functional issues "
                f"({profile.primary_symptom}, {', '.join(profile.secondary_symptoms)})."
            )
            veto_reasons.append("VETO: Unrelated multi-symptom customer complaint requiring human prioritization.")
        elif distinct_symptoms_count == 2:
            # Check if symptoms are related (e.g. battery + heat, or update + lag)
            s2 = profile.secondary_symptoms[0].lower()
            if ("battery" in sym_lower and "power" in s2) or ("audio" in sym_lower and "speaker" in s2):
                multi_symptom = MultiSymptomLevel.RELATED_MULTI_SYMPTOM
                multi_reason = f"Customer reports related symptoms ({profile.primary_symptom} and {profile.secondary_symptoms[0]})."
            else:
                multi_symptom = MultiSymptomLevel.RELATED_MULTI_SYMPTOM
                multi_reason = f"Customer reports dual symptoms ({profile.primary_symptom} and {profile.secondary_symptoms[0]})."
        else:
            multi_symptom = MultiSymptomLevel.SINGLE_SYMPTOM
            multi_reason = f"Customer reports a single cohesive operational symptom ({profile.primary_symptom})."

        is_vetoed = len(veto_reasons) > 0

        return ClaritySignalsProfile(
            sufficiency=sufficiency,
            problem_strength=problem_strength,
            candidate_conflict=candidate_conflict,
            evidence_agreement=evidence_agreement,
            retrieval_quality=retrieval_quality,
            multi_symptom=multi_symptom,
            sufficiency_reason=suff_reason,
            strength_reason=strength_reason,
            conflict_reason=conflict_reason,
            agreement_reason=agreement_reason,
            retrieval_reason=retrieval_reason,
            multi_symptom_reason=multi_reason,
            is_vetoed=is_vetoed,
            veto_reasons=veto_reasons,
        )
