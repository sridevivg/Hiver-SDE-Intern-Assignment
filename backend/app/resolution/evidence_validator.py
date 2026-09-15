"""
SupportGraph AI — Operational Evidence Validator (Phase 8)

Validates whether retrieved historical support cases provide sufficiently strong
operational evidence to support automated customer resolution:
- Evaluates 5 operational dimensions:
  1. Device Match (exact, compatible, unspecified, mismatch)
  2. Product/Service Match (exact, compatible, unspecified, mismatch)
  3. Primary Symptom Match (direct, related, unrelated, contradictory)
  4. Operational Intent Match (direct, related, conflicting)
  5. Cause Match (supporting context vs primary symptom)
- Computes comprehensive evidence metrics and produces structured verdicts:
  - STRONG_EVIDENCE: High operational match across multiple historical cases, zero contradictions
  - MODERATE_EVIDENCE: Valid symptom alignment with partial context or minor semantic noise
  - WEAK_EVIDENCE: Low similarity or purely lexical overlap
  - CONFLICTING_EVIDENCE: Retrieved cases suggest competing operational intents or contradict symptoms
  - INSUFFICIENT_EVIDENCE: No relevant cases retrieved or empty corpus
"""
from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

try:
    from app.core.logging import get_logger
    from app.retrieval.evidence_ranker import (
        EvidenceMatchTier,
        RetrievedEvidenceCase,
    )
    from app.understanding.problem_extractor import CustomerProblemProfile
except ModuleNotFoundError:
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]
    from backend.app.retrieval.evidence_ranker import (  # type: ignore[no-redef]
        EvidenceMatchTier,
        RetrievedEvidenceCase,
    )
    from backend.app.understanding.problem_extractor import (  # type: ignore[no-redef]
        CustomerProblemProfile,
    )

logger = get_logger(__name__)


class EvidenceVerdict(str, Enum):
    """Overall operational evidence verdict for automated resolution."""
    STRONG_EVIDENCE = "STRONG_EVIDENCE"
    MODERATE_EVIDENCE = "MODERATE_EVIDENCE"
    WEAK_EVIDENCE = "WEAK_EVIDENCE"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class CompositeEvidenceVerdict(str, Enum):
    """
    Phase 10: Composite multi-case evidence verdict.

    Separate from EvidenceVerdict to preserve full backward compatibility
    with all Phase 8 and Phase 9 code.

    DECISION HIERARCHY (most authoritative first):
      DIRECT_STRONG_EVIDENCE         — single-case STRONG_EVIDENCE passthrough
      COMPOSITE_STRONG_EVIDENCE      — multiple cases cover ≥2 independent dimensions
                                       (symptom coverage MANDATORY)
      MODERATE_COMPOSITE_EVIDENCE    — partial coverage; symptom or resolution missing
      WEAK_COMPOSITE_EVIDENCE        — only semantic/lexical overlap, no operational match
      CONFLICTING_COMPOSITE_EVIDENCE — retrieved cases support incompatible intents
      INSUFFICIENT_COMPOSITE_EVIDENCE — no usable cases at all
    """
    DIRECT_STRONG_EVIDENCE = "DIRECT_STRONG_EVIDENCE"
    COMPOSITE_STRONG_EVIDENCE = "COMPOSITE_STRONG_EVIDENCE"
    MODERATE_COMPOSITE_EVIDENCE = "MODERATE_COMPOSITE_EVIDENCE"
    WEAK_COMPOSITE_EVIDENCE = "WEAK_COMPOSITE_EVIDENCE"
    CONFLICTING_COMPOSITE_EVIDENCE = "CONFLICTING_COMPOSITE_EVIDENCE"
    INSUFFICIENT_COMPOSITE_EVIDENCE = "INSUFFICIENT_COMPOSITE_EVIDENCE"

    @property
    def allows_auto_resolution(self) -> bool:
        """True if this verdict can authorize automatic resolution (subject to grounding check)."""
        return self in (
            CompositeEvidenceVerdict.DIRECT_STRONG_EVIDENCE,
            CompositeEvidenceVerdict.COMPOSITE_STRONG_EVIDENCE,
        )

    @property
    def is_strong(self) -> bool:
        return self in (
            CompositeEvidenceVerdict.DIRECT_STRONG_EVIDENCE,
            CompositeEvidenceVerdict.COMPOSITE_STRONG_EVIDENCE,
        )


class EvidenceDimensionMatch(BaseModel):
    """Operational alignment breakdown across 5 key dimensions for a single historical case."""
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    case_id: str
    match_tier: EvidenceMatchTier
    device_match: Literal["exact", "compatible", "unspecified", "mismatch"] = "unspecified"
    service_match: Literal["exact", "compatible", "unspecified", "mismatch"] = "unspecified"
    symptom_match: Literal["direct", "related", "unrelated", "contradictory"] = "unrelated"
    intent_match: Literal["direct", "related", "conflicting"] = "related"
    cause_match: Literal["supporting_cause", "differing_cause", "no_cause"] = "no_cause"
    operational_score: float = Field(default=0.0, ge=0.0, le=1.0)
    match_notes: str = ""


class EvidenceValidationResult(BaseModel):
    """Comprehensive evaluation of retrieved historical support evidence."""
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    evidence_verdict: EvidenceVerdict = Field(..., description="Overall operational evidence verdict")
    auto_resolution_allowed: bool = Field(..., description="True if evidence is strong enough to permit auto-handling")
    direct_problem_matches: int = Field(default=0, ge=0, description="Count of Tier 1 direct problem matches")
    related_problem_matches: int = Field(default=0, ge=0, description="Count of Tier 2 related problem family matches")
    related_symptom_matches: int = Field(default=0, ge=0, description="Count of Tier 3 related symptom matches")
    related_context_matches: int = Field(default=0, ge=0, description="Count of Tier 4 related context matches")
    weak_semantic_matches: int = Field(default=0, ge=0, description="Count of Tier 5 weak semantic matches")
    intent_agreement: float = Field(default=0.0, ge=0.0, le=1.0, description="Proportion of cases agreeing with predicted intent")
    symptom_agreement: float = Field(default=0.0, ge=0.0, le=1.0, description="Degree of functional symptom alignment")
    device_context_agreement: float = Field(default=0.0, ge=0.0, le=1.0, description="Degree of device/OS alignment")
    evidence_consistency: float = Field(default=0.0, ge=0.0, le=1.0, description="Inter-case operational consistency")
    resolution_support_strength: float = Field(default=0.0, ge=0.0, le=1.0, description="Composite score of resolution support")
    reasons: list[str] = Field(default_factory=list, description="Explainable justifications for evidence verdict")
    dimension_matches: list[EvidenceDimensionMatch] = Field(default_factory=list, description="Per-case dimensional breakdown")

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


# Mutually exclusive/contradictory intent pairings
CONTRADICTORY_INTENT_PAIRS = {
    ("battery_power_issue", "billing_purchase_issue"),
    ("battery_power_issue", "account_access_issue"),
    ("hardware_audio_connection_issue", "billing_purchase_issue"),
    ("display_touch_issue", "account_access_issue"),
    ("keyboard_typing_issue", "billing_purchase_issue"),
    ("account_access_issue", "hardware_audio_connection_issue"),
}


class ResolutionEvidenceValidator:
    """
    Validates retrieved historical evidence to determine if automatic resolution is operationally justified.
    """

    def __init__(
        self,
        min_support_strength_for_auto: float = 0.55,
        min_intent_agreement: float = 0.45,
    ) -> None:
        self.min_support_strength_for_auto = min_support_strength_for_auto
        self.min_intent_agreement = min_intent_agreement

    def evaluate_dimension(
        self,
        profile: CustomerProblemProfile,
        primary_intent: str,
        case: RetrievedEvidenceCase,
    ) -> EvidenceDimensionMatch:
        """Evaluate operational match across 5 distinct dimensions for a retrieved case."""
        hist_text = case.historical_customer_message.lower()
        hist_intent = case.historical_intent

        # 1. Device Match
        device_match: Literal["exact", "compatible", "unspecified", "mismatch"] = "unspecified"
        if profile.device:
            q_dev = profile.device.lower()
            if q_dev in hist_text:
                device_match = "exact"
            elif ("iphone" in q_dev and "iphone" in hist_text) or ("macbook" in q_dev and "mac" in hist_text):
                device_match = "compatible"
            elif ("iphone" in q_dev and "mac" in hist_text) or ("ipad" in q_dev and "apple watch" in hist_text):
                device_match = "mismatch"
            else:
                device_match = "unspecified"

        # 2. Product/Service Match
        service_match: Literal["exact", "compatible", "unspecified", "mismatch"] = "unspecified"
        if profile.product_or_service:
            q_serv = profile.product_or_service.lower()
            if q_serv in hist_text:
                service_match = "exact"
            elif ("ios" in q_serv and "ios" in hist_text) or ("macos" in q_serv and "mac" in hist_text):
                service_match = "compatible"
            else:
                service_match = "unspecified"

        # 3. Symptom Match
        symptom_match: Literal["direct", "related", "unrelated", "contradictory"] = "unrelated"
        q_sym = profile.primary_symptom.lower()

        # Check direct keywords
        has_direct_symptom = False
        if "battery" in q_sym and re.search(r"\b(battery|drain|charge|power)\b", hist_text):
            has_direct_symptom = True
        elif "audio" in q_sym and re.search(r"\b(speaker|sound|volume|mic|audio|airpod|headphone)\b", hist_text):
            has_direct_symptom = True
        elif "display" in q_sym and re.search(r"\b(screen|display|touch|cracked|flicker|black screen)\b", hist_text):
            has_direct_symptom = True
        elif "keyboard" in q_sym and re.search(r"\b(keyboard|type|typing|autocorrect|letter\s+i)\b", hist_text):
            has_direct_symptom = True
        elif "account" in q_sym and re.search(r"\b(password|apple\s*id|locked|sign\s+in|2fa|verification)\b", hist_text):
            has_direct_symptom = True
        elif "billing" in q_sym and re.search(r"\b(charge|billed|refund|subscription|itunes|receipt)\b", hist_text):
            has_direct_symptom = True
        elif "wifi" in q_sym and re.search(r"\b(wifi|wi-fi|network|cellular|lte)\b", hist_text):
            has_direct_symptom = True

        if has_direct_symptom or case.match_tier == EvidenceMatchTier.DIRECT_PROBLEM_MATCH:
            symptom_match = "direct"
        elif case.match_tier == EvidenceMatchTier.RELATED_PROBLEM_MATCH:
            symptom_match = "direct" if has_direct_symptom else "related"
        elif case.match_tier == EvidenceMatchTier.RELATED_SYMPTOM:
            symptom_match = "related"
        elif (primary_intent, hist_intent) in CONTRADICTORY_INTENT_PAIRS:
            symptom_match = "contradictory"
        else:
            symptom_match = "unrelated"

        # 4. Intent Match
        intent_match: Literal["direct", "related", "conflicting"] = "related"
        if hist_intent == primary_intent or case.match_tier in (EvidenceMatchTier.DIRECT_PROBLEM_MATCH, EvidenceMatchTier.RELATED_PROBLEM_MATCH):
            intent_match = "direct"
        elif (primary_intent, hist_intent) in CONTRADICTORY_INTENT_PAIRS or (hist_intent, primary_intent) in CONTRADICTORY_INTENT_PAIRS:
            intent_match = "conflicting"
        else:
            intent_match = "related"

        # 5. Cause Match
        cause_match: Literal["supporting_cause", "differing_cause", "no_cause"] = "no_cause"
        if profile.possible_cause or profile.update_related:
            if re.search(r"\b(update|updated|updating|ios\s*11|ios11|upgrade)\b", hist_text):
                cause_match = "supporting_cause"
            else:
                cause_match = "differing_cause"

        # Compute dimension score
        score = case.operational_similarity
        if symptom_match == "direct":
            score = min(1.0, score + 0.15)
        elif symptom_match == "contradictory":
            score = max(0.0, score - 0.40)

        if intent_match == "direct":
            score = min(1.0, score + 0.10)
        elif intent_match == "conflicting":
            score = max(0.0, score - 0.35)

        return EvidenceDimensionMatch(
            case_id=case.case_id,
            match_tier=case.match_tier,
            device_match=device_match,
            service_match=service_match,
            symptom_match=symptom_match,
            intent_match=intent_match,
            cause_match=cause_match,
            operational_score=round(score, 4),
            match_notes=case.retrieval_explanation,
        )

    def evaluate_evidence(
        self,
        profile: CustomerProblemProfile,
        primary_intent: str,
        evidence_cases: list[RetrievedEvidenceCase],
    ) -> EvidenceValidationResult:
        """
        Evaluate full set of retrieved historical cases for operational evidence validity.
        """
        if not evidence_cases:
            return EvidenceValidationResult(
                evidence_verdict=EvidenceVerdict.INSUFFICIENT_EVIDENCE,
                auto_resolution_allowed=False,
                reasons=["No historical support evidence cases were retrieved."],
            )

        # Dimension matches
        dim_matches: list[EvidenceDimensionMatch] = [
            self.evaluate_dimension(profile, primary_intent, c) for c in evidence_cases
        ]

        # Counts by tier
        tier_counts = {
            EvidenceMatchTier.DIRECT_PROBLEM_MATCH: sum(1 for c in evidence_cases if c.match_tier == EvidenceMatchTier.DIRECT_PROBLEM_MATCH),
            EvidenceMatchTier.RELATED_PROBLEM_MATCH: sum(1 for c in evidence_cases if c.match_tier == EvidenceMatchTier.RELATED_PROBLEM_MATCH),
            EvidenceMatchTier.RELATED_SYMPTOM: sum(1 for c in evidence_cases if c.match_tier == EvidenceMatchTier.RELATED_SYMPTOM),
            EvidenceMatchTier.RELATED_CONTEXT: sum(1 for c in evidence_cases if c.match_tier == EvidenceMatchTier.RELATED_CONTEXT),
            EvidenceMatchTier.WEAK_SEMANTIC_MATCH: sum(1 for c in evidence_cases if c.match_tier == EvidenceMatchTier.WEAK_SEMANTIC_MATCH),
        }

        # Intent Agreement
        matching_intent_count = sum(1 for d in dim_matches if d.intent_match == "direct")
        intent_agreement = round(matching_intent_count / len(dim_matches), 4)

        # Symptom Agreement
        direct_symptom_count = sum(1 for d in dim_matches if d.symptom_match == "direct")
        related_symptom_count = sum(1 for d in dim_matches if d.symptom_match == "related")
        symptom_score_sum = (direct_symptom_count * 1.0) + (related_symptom_count * 0.5)
        symptom_agreement = round(min(1.0, symptom_score_sum / len(dim_matches)), 4)

        # Device/Context Agreement
        dev_match_count = sum(1 for d in dim_matches if d.device_match in ("exact", "compatible"))
        device_agreement = round(dev_match_count / len(dim_matches), 4) if profile.device else 1.0

        # Conflicting cases
        conflicting_count = sum(1 for d in dim_matches if d.intent_match == "conflicting" or d.symptom_match == "contradictory")

        # Evidence Consistency
        if len(dim_matches) > 1:
            intent_entropy_penalty = 1.0 - (matching_intent_count / len(dim_matches))
            consistency = max(0.0, 1.0 - (0.5 * intent_entropy_penalty) - (0.4 * (conflicting_count / len(dim_matches))))
        else:
            consistency = 1.0 if matching_intent_count == 1 else 0.5
        consistency = round(consistency, 4)

        # Composite Support Strength
        base_op_scores = [d.operational_score for d in dim_matches]
        avg_op_score = sum(base_op_scores) / len(base_op_scores) if base_op_scores else 0.0

        resolution_support_strength = round(
            (0.35 * avg_op_score) +
            (0.30 * symptom_agreement) +
            (0.25 * intent_agreement) +
            (0.10 * consistency),
            4,
        )

        reasons: list[str] = []

        # Determine Verdict
        if conflicting_count > 0 and conflicting_count >= matching_intent_count:
            verdict = EvidenceVerdict.CONFLICTING_EVIDENCE
            auto_allowed = False
            reasons.append(f"Conflicting evidence detected: {conflicting_count} retrieved case(s) contradict the proposed intent '{primary_intent}'.")

        elif (tier_counts[EvidenceMatchTier.DIRECT_PROBLEM_MATCH] + tier_counts[EvidenceMatchTier.RELATED_PROBLEM_MATCH]) >= 1 and symptom_agreement >= 0.50 and intent_agreement >= 0.33:
            verdict = EvidenceVerdict.STRONG_EVIDENCE
            auto_allowed = True
            strong_count = tier_counts[EvidenceMatchTier.DIRECT_PROBLEM_MATCH] + tier_counts[EvidenceMatchTier.RELATED_PROBLEM_MATCH]
            reasons.append(
                f"Strong operational evidence: {strong_count} direct/related problem match case(s) "
                f"with {symptom_agreement:.0%} symptom agreement."
            )

        elif (tier_counts[EvidenceMatchTier.DIRECT_PROBLEM_MATCH] + tier_counts[EvidenceMatchTier.RELATED_PROBLEM_MATCH] + tier_counts[EvidenceMatchTier.RELATED_SYMPTOM]) >= 1 and symptom_agreement >= 0.33:
            verdict = EvidenceVerdict.MODERATE_EVIDENCE
            # Moderate evidence permits auto handling only if support strength meets threshold and no conflict
            auto_allowed = resolution_support_strength >= self.min_support_strength_for_auto and conflicting_count == 0
            if auto_allowed:
                reasons.append(
                    f"Moderate operational evidence: {tier_counts[EvidenceMatchTier.RELATED_SYMPTOM] + tier_counts[EvidenceMatchTier.RELATED_PROBLEM_MATCH]} related case(s) "
                    f"supporting resolution strength {resolution_support_strength:.2f}."
                )
            else:
                reasons.append(
                    f"Moderate evidence insufficient for safe auto-resolution (support strength {resolution_support_strength:.2f} < {self.min_support_strength_for_auto})."
                )

        elif tier_counts[EvidenceMatchTier.WEAK_SEMANTIC_MATCH] == len(evidence_cases):
            verdict = EvidenceVerdict.WEAK_EVIDENCE
            auto_allowed = False
            reasons.append(
                "Weak semantic evidence: retrieved historical cases share lexical keywords but fail to match the operational customer problem."
            )

        else:
            verdict = EvidenceVerdict.WEAK_EVIDENCE
            auto_allowed = False
            reasons.append(
                f"Insufficient operational support: overall resolution support strength ({resolution_support_strength:.2f}) below safe threshold."
            )

        return EvidenceValidationResult(
            evidence_verdict=verdict,
            auto_resolution_allowed=auto_allowed,
            direct_problem_matches=tier_counts[EvidenceMatchTier.DIRECT_PROBLEM_MATCH],
            related_problem_matches=tier_counts[EvidenceMatchTier.RELATED_PROBLEM_MATCH],
            related_symptom_matches=tier_counts[EvidenceMatchTier.RELATED_SYMPTOM],
            related_context_matches=tier_counts[EvidenceMatchTier.RELATED_CONTEXT],
            weak_semantic_matches=tier_counts[EvidenceMatchTier.WEAK_SEMANTIC_MATCH],
            intent_agreement=intent_agreement,
            symptom_agreement=symptom_agreement,
            device_context_agreement=device_agreement,
            evidence_consistency=consistency,
            resolution_support_strength=resolution_support_strength,
            reasons=reasons,
            dimension_matches=dim_matches,
        )
