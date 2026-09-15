"""
SupportGraph AI — Multi-Case Evidence Synthesizer (Phase 10)

Implements the core Phase 10 principle:

    NO SINGLE EXACT MATCH DOES NOT NECESSARILY MEAN NO SUFFICIENT EVIDENCE

    MULTIPLE WEAK SEMANTIC MATCHES MUST NEVER AUTOMATICALLY BECOME STRONG EVIDENCE

Synthesizes composite evidence packages across 5 independent operational dimensions
from multiple retrieved historical cases. Evaluated AFTER single-case validation
fails to find STRONG_EVIDENCE — providing a second-chance evidence pathway that
preserves all existing safety constraints.

SAFETY CONSTRAINTS (non-negotiable):
  - Symptom coverage is MANDATORY for COMPOSITE_STRONG_EVIDENCE.
  - WEAK_SEMANTIC_MATCH tier cases CANNOT contribute HIGH-strength evidence.
  - RELATED_CONTEXT cases CANNOT independently establish symptom coverage.
  - Duplicate / near-duplicate cases are deduplicated before synthesis —
    repeated copies of the same evidence cannot inflate evidence strength.
  - Conflict detection is called before composite verdict is finalized —
    contradicting cases are excluded from synthesis.
  - COMPOSITE_STRONG_EVIDENCE requires coverage across ≥2 independent dimensions
    (e.g. symptom + intent, or symptom + resolution_pattern).
  - This module NEVER bypasses response grounding verification.
  - High model confidence alone NEVER upgrades weak evidence.
"""
from __future__ import annotations

import re
from enum import Enum
from typing import Any, Literal, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from app.core.logging import get_logger
    from app.resolution.evidence_conflict_detector import (
        ConflictDetectionResult,
        EvidenceConflictDetector,
    )
    from app.resolution.evidence_validator import (
        CompositeEvidenceVerdict,
        EvidenceDimensionMatch,
        EvidenceValidationResult,
        EvidenceVerdict,
    )
    from app.retrieval.evidence_ranker import EvidenceMatchTier, RetrievedEvidenceCase
    from app.understanding.problem_extractor import CustomerProblemProfile
except ModuleNotFoundError:
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]
    from backend.app.resolution.evidence_conflict_detector import (  # type: ignore[no-redef]
        ConflictDetectionResult,
        EvidenceConflictDetector,
    )
    from backend.app.resolution.evidence_validator import (  # type: ignore[no-redef]
        CompositeEvidenceVerdict,
        EvidenceDimensionMatch,
        EvidenceValidationResult,
        EvidenceVerdict,
    )
    from backend.app.retrieval.evidence_ranker import (  # type: ignore[no-redef]
        EvidenceMatchTier,
        RetrievedEvidenceCase,
    )
    from backend.app.understanding.problem_extractor import (  # type: ignore[no-redef]
        CustomerProblemProfile,
    )

logger = get_logger(__name__)

# Minimum cosine similarity between two case messages to consider them near-duplicates
DUPLICATE_SIMILARITY_THRESHOLD = 0.92

# Minimum number of independent dimensions that must be covered for COMPOSITE_STRONG_EVIDENCE
MIN_DIMENSIONS_FOR_COMPOSITE_STRONG = 2


class ContributionStrength(str, Enum):
    """Evidence contribution strength for a single case-dimension pair."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NONE = "NONE"


class CaseEvidenceContribution(BaseModel):
    """
    Explicit evidence contribution record for a single historical case.

    States exactly what dimensions each case contributes to and at what strength,
    enabling full explainability of composite evidence decisions.
    """
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    case_id: str
    match_tier: EvidenceMatchTier
    historical_intent: str
    contributes_to: list[str] = Field(
        default_factory=list,
        description="Dimensions this case contributes to: symptom, context, device, intent, resolution_pattern",
    )
    contribution_strength: ContributionStrength = ContributionStrength.NONE
    is_duplicate: bool = Field(False, description="True if near-duplicate of another case")
    allowed_for_synthesis: bool = Field(True, description="False if excluded by conflict detection")
    contribution_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class DimensionCoverage(BaseModel):
    """Coverage assessment for a single evidence dimension."""
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    dimension: str  # symptom | context | device | intent | resolution_pattern
    is_covered: bool = False
    coverage_strength: ContributionStrength = ContributionStrength.NONE
    contributing_case_ids: list[str] = Field(default_factory=list)
    coverage_notes: str = ""


class CompositeEvidencePackage(BaseModel):
    """
    Complete structured output of multi-case evidence synthesis.

    Contains the composite verdict, per-dimension coverage analysis,
    per-case contribution records, and explainable reasoning.
    """
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    composite_verdict: CompositeEvidenceVerdict = Field(
        ..., description="Phase 10 composite evidence verdict"
    )
    auto_resolution_allowed: bool = Field(
        ..., description="Whether composite evidence authorizes automated resolution"
    )
    # Dimension analysis
    dimension_coverage: list[DimensionCoverage] = Field(
        default_factory=list,
        description="Coverage assessment per evidence dimension",
    )
    covered_dimensions: list[str] = Field(
        default_factory=list,
        description="Names of dimensions with at least MEDIUM coverage",
    )
    missing_dimensions: list[str] = Field(
        default_factory=list,
        description="Names of dimensions not covered by any retrieved case",
    )
    # Case contributions
    case_contributions: list[CaseEvidenceContribution] = Field(
        default_factory=list,
        description="Per-case explicit contribution records",
    )
    # Deduplication
    duplicate_case_ids: list[str] = Field(
        default_factory=list,
        description="Case IDs excluded as near-duplicates",
    )
    unique_contributing_cases: int = Field(
        0, description="Count of non-duplicate cases that contributed evidence"
    )
    # Conflict detection
    conflict_result: Optional[ConflictDetectionResult] = Field(
        default=None,
        description="Evidence conflict detection result",
    )
    # Synthesis metrics
    symptom_covered: bool = Field(False, description="True if primary symptom has coverage")
    intent_consistency: float = Field(0.0, ge=0.0, le=1.0, description="Proportion of allowed cases with matching intent")
    composite_support_score: float = Field(0.0, ge=0.0, le=1.0, description="Overall composite evidence strength [0,1]")
    # Reasoning
    synthesis_rationale: str = Field("", description="Human-readable synthesis explanation")
    why_not_composite_strong: list[str] = Field(
        default_factory=list,
        description="Specific reasons why COMPOSITE_STRONG_EVIDENCE was not reached (if applicable)",
    )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(exclude={"conflict_result"})


class MultiCaseEvidenceSynthesizer:
    """
    Synthesizes composite evidence packages from multiple retrieved historical cases.

    Called by SupportResolutionEngine AFTER single-case ResolutionEvidenceValidator
    returns WEAK_EVIDENCE or INSUFFICIENT_EVIDENCE. Provides a second-chance
    evidence pathway based on multi-case dimensional coverage.

    CORE LOGIC:
      1. Deduplicate near-identical retrieved cases (prevent evidence inflation)
      2. Run conflict detection (exclude contradicting cases)
      3. Evaluate 5 evidence dimensions across allowed cases
      4. Produce CompositeEvidenceVerdict and auto_resolution_allowed flag

    SAFETY RULES ENFORCED:
      - WEAK_SEMANTIC_MATCH → max ContributionStrength.LOW (never HIGH or MEDIUM)
      - RELATED_CONTEXT → cannot establish symptom_coverage (context ≠ symptom)
      - Symptom coverage is MANDATORY for COMPOSITE_STRONG_EVIDENCE
      - ≥2 independent dimensions required for COMPOSITE_STRONG_EVIDENCE
      - Conflict detection may force CONFLICTING_COMPOSITE_EVIDENCE regardless of coverage
    """

    def __init__(
        self,
        conflict_detector: Optional[EvidenceConflictDetector] = None,
        duplicate_threshold: float = DUPLICATE_SIMILARITY_THRESHOLD,
        min_dimensions_for_strong: int = MIN_DIMENSIONS_FOR_COMPOSITE_STRONG,
    ) -> None:
        self.conflict_detector = conflict_detector or EvidenceConflictDetector()
        self.duplicate_threshold = duplicate_threshold
        self.min_dimensions_for_strong = min_dimensions_for_strong

    def synthesize(
        self,
        primary_intent: str,
        profile: CustomerProblemProfile,
        evidence_cases: list[RetrievedEvidenceCase],
        single_case_validation: EvidenceValidationResult,
        dimension_matches: list[EvidenceDimensionMatch],
    ) -> CompositeEvidencePackage:
        """
        Synthesize composite evidence from multiple retrieved cases.

        Args:
            primary_intent: The selected primary intent.
            profile: Structured customer problem profile.
            evidence_cases: Retrieved historical cases.
            single_case_validation: Result of Phase 8 ResolutionEvidenceValidator.
            dimension_matches: Per-case dimensional evaluations from ResolutionEvidenceValidator.

        Returns:
            CompositeEvidencePackage with composite verdict and full explainability.
        """
        # ── Shortcut: single-case already STRONG → passthrough ───────────────
        if single_case_validation.evidence_verdict == EvidenceVerdict.STRONG_EVIDENCE:
            return self._passthrough_strong(
                evidence_cases=evidence_cases,
                dimension_matches=dimension_matches,
                single_case_validation=single_case_validation,
            )

        # ── No cases at all ───────────────────────────────────────────────────
        if not evidence_cases:
            return CompositeEvidencePackage(
                composite_verdict=CompositeEvidenceVerdict.INSUFFICIENT_COMPOSITE_EVIDENCE,
                auto_resolution_allowed=False,
                synthesis_rationale="No historical cases retrieved for synthesis.",
                why_not_composite_strong=["No cases available."],
            )

        # ── Step 1: Deduplication ─────────────────────────────────────────────
        unique_cases, duplicate_ids = self._deduplicate(evidence_cases)

        # ── Step 2: Conflict Detection ────────────────────────────────────────
        unique_dim_matches = [d for d in dimension_matches if d.case_id not in duplicate_ids]
        conflict_result = self.conflict_detector.detect_conflicts(
            primary_intent=primary_intent,
            profile=profile,
            evidence_cases=unique_cases,
            dimension_matches=unique_dim_matches,
        )

        # Forced verdict from conflict detector
        if conflict_result.forced_verdict is not None:
            return CompositeEvidencePackage(
                composite_verdict=conflict_result.forced_verdict,
                auto_resolution_allowed=False,
                duplicate_case_ids=duplicate_ids,
                conflict_result=conflict_result,
                synthesis_rationale=conflict_result.explanation,
                why_not_composite_strong=[conflict_result.explanation],
            )

        # Filter to allowed cases only
        allowed_cases = [c for c in unique_cases if c.case_id in conflict_result.allowed_case_ids]
        allowed_dims = [d for d in unique_dim_matches if d.case_id in conflict_result.allowed_case_ids]

        if not allowed_cases:
            return CompositeEvidencePackage(
                composite_verdict=CompositeEvidenceVerdict.CONFLICTING_COMPOSITE_EVIDENCE,
                auto_resolution_allowed=False,
                duplicate_case_ids=duplicate_ids,
                conflict_result=conflict_result,
                synthesis_rationale=(
                    "All retrieved cases were excluded by conflict detection. "
                    + conflict_result.explanation
                ),
                why_not_composite_strong=["No non-conflicting cases remain for synthesis."],
            )

        # ── Step 3: Build case contributions ──────────────────────────────────
        dim_map = {d.case_id: d for d in allowed_dims}
        case_contributions: list[CaseEvidenceContribution] = []
        excluded_contributions: list[CaseEvidenceContribution] = []

        for case in evidence_cases:
            is_dup = case.case_id in duplicate_ids
            is_allowed = (case.case_id in conflict_result.allowed_case_ids) and not is_dup
            contrib = self._build_contribution(
                case=case,
                profile=profile,
                primary_intent=primary_intent,
                dim=dim_map.get(case.case_id),
                is_duplicate=is_dup,
                allowed=is_allowed,
                conflict_result=conflict_result,
            )
            if is_allowed:
                case_contributions.append(contrib)
            else:
                excluded_contributions.append(contrib)

        # Also record duplicate cases
        all_contributions = case_contributions + excluded_contributions

        # ── Step 4: Evaluate dimensions ───────────────────────────────────────
        dimension_coverage = self._evaluate_dimensions(
            allowed_cases=allowed_cases,
            case_contributions=case_contributions,
            profile=profile,
            primary_intent=primary_intent,
        )

        covered_dims = [d.dimension for d in dimension_coverage if d.is_covered and d.coverage_strength != ContributionStrength.NONE]
        missing_dims = [d.dimension for d in dimension_coverage if not d.is_covered]
        symptom_covered = any(d.dimension == "symptom" and d.is_covered for d in dimension_coverage)

        # ── Step 5: Intent consistency ────────────────────────────────────────
        intent_matching = sum(1 for c in allowed_cases if c.historical_intent == primary_intent)
        intent_consistency = round(intent_matching / len(allowed_cases), 4) if allowed_cases else 0.0

        # ── Step 6: Composite support score ──────────────────────────────────
        composite_score = self._compute_composite_score(
            dimension_coverage=dimension_coverage,
            intent_consistency=intent_consistency,
            num_allowed=len(allowed_cases),
        )

        # ── Step 7: Determine verdict ─────────────────────────────────────────
        verdict, auto_allowed, why_not_strong = self._determine_verdict(
            symptom_covered=symptom_covered,
            covered_dims=covered_dims,
            conflict_result=conflict_result,
            composite_score=composite_score,
            intent_consistency=intent_consistency,
            single_case_validation=single_case_validation,
        )

        # ── Build synthesis rationale ─────────────────────────────────────────
        rationale = self._build_rationale(
            verdict=verdict,
            covered_dims=covered_dims,
            missing_dims=missing_dims,
            case_contributions=case_contributions,
            duplicate_ids=duplicate_ids,
            conflict_result=conflict_result,
            composite_score=composite_score,
            intent_consistency=intent_consistency,
        )

        return CompositeEvidencePackage(
            composite_verdict=verdict,
            auto_resolution_allowed=auto_allowed,
            dimension_coverage=dimension_coverage,
            covered_dimensions=covered_dims,
            missing_dimensions=missing_dims,
            case_contributions=all_contributions,
            duplicate_case_ids=duplicate_ids,
            unique_contributing_cases=len(allowed_cases),
            conflict_result=conflict_result,
            symptom_covered=symptom_covered,
            intent_consistency=intent_consistency,
            composite_support_score=composite_score,
            synthesis_rationale=rationale,
            why_not_composite_strong=why_not_strong,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Private: Deduplication
    # ─────────────────────────────────────────────────────────────────────────

    def _deduplicate(
        self,
        evidence_cases: list[RetrievedEvidenceCase],
    ) -> tuple[list[RetrievedEvidenceCase], list[str]]:
        """
        Identify and remove near-duplicate cases to prevent evidence inflation.

        Near-duplicates: two cases whose customer messages have TF-IDF cosine
        similarity >= DUPLICATE_SIMILARITY_THRESHOLD. Only the first (highest-tier)
        occurrence is kept.

        Returns:
            (unique_cases, duplicate_case_ids)
        """
        if len(evidence_cases) <= 1:
            return list(evidence_cases), []

        messages = [c.historical_customer_message for c in evidence_cases]
        try:
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
            mat = vec.fit_transform(messages)
            sims = cosine_similarity(mat)
        except Exception:
            # If vectorization fails (e.g., all empty strings), treat all as unique
            return list(evidence_cases), []

        unique_cases: list[RetrievedEvidenceCase] = []
        duplicate_ids: list[str] = []
        seen_indices: set[int] = set()

        for i, case in enumerate(evidence_cases):
            if i in seen_indices:
                continue
            unique_cases.append(case)
            # Mark all similar later cases as duplicates
            for j in range(i + 1, len(evidence_cases)):
                if j not in seen_indices and sims[i, j] >= self.duplicate_threshold:
                    seen_indices.add(j)
                    duplicate_ids.append(evidence_cases[j].case_id)
                    logger.debug(
                        "Duplicate detected: case_id=%s similar to case_id=%s (sim=%.3f)",
                        evidence_cases[j].case_id, case.case_id, sims[i, j],
                    )

        return unique_cases, duplicate_ids

    # ─────────────────────────────────────────────────────────────────────────
    # Private: Contribution building
    # ─────────────────────────────────────────────────────────────────────────

    def _build_contribution(
        self,
        case: RetrievedEvidenceCase,
        profile: CustomerProblemProfile,
        primary_intent: str,
        dim: Optional[EvidenceDimensionMatch],
        is_duplicate: bool,
        allowed: bool,
        conflict_result: ConflictDetectionResult,
    ) -> CaseEvidenceContribution:
        """Build explicit contribution record for a single retrieved case."""
        contributes_to: list[str] = []
        notes_parts: list[str] = []

        # Tier-based maximum contribution strength
        if case.match_tier == EvidenceMatchTier.DIRECT_PROBLEM_MATCH:
            max_strength = ContributionStrength.HIGH
        elif case.match_tier == EvidenceMatchTier.RELATED_PROBLEM_MATCH:
            max_strength = ContributionStrength.HIGH
        elif case.match_tier == EvidenceMatchTier.RELATED_SYMPTOM:
            max_strength = ContributionStrength.MEDIUM
        elif case.match_tier == EvidenceMatchTier.RELATED_CONTEXT:
            max_strength = ContributionStrength.LOW
        else:  # WEAK_SEMANTIC_MATCH
            max_strength = ContributionStrength.LOW  # Cannot go higher

        if is_duplicate:
            return CaseEvidenceContribution(
                case_id=case.case_id,
                match_tier=case.match_tier,
                historical_intent=case.historical_intent,
                contributes_to=[],
                contribution_strength=ContributionStrength.NONE,
                is_duplicate=True,
                allowed_for_synthesis=False,
                contribution_notes="Excluded: near-duplicate of another retrieved case.",
            )

        if not allowed:
            reason = next(
                (lb.conflict_reason for lb in conflict_result.case_labels if lb.case_id == case.case_id),
                "Excluded by conflict detection.",
            )
            return CaseEvidenceContribution(
                case_id=case.case_id,
                match_tier=case.match_tier,
                historical_intent=case.historical_intent,
                contributes_to=[],
                contribution_strength=ContributionStrength.NONE,
                is_duplicate=False,
                allowed_for_synthesis=False,
                contribution_notes=f"Excluded by conflict detection: {reason}",
            )

        # ── Dimension contributions for allowed, non-duplicate cases ──────────

        # 1. SYMPTOM coverage
        # RELATED_CONTEXT alone CANNOT establish symptom coverage (context ≠ symptom)
        if case.match_tier in (
            EvidenceMatchTier.DIRECT_PROBLEM_MATCH,
            EvidenceMatchTier.RELATED_PROBLEM_MATCH,
            EvidenceMatchTier.RELATED_SYMPTOM,
        ):
            if dim and dim.symptom_match in ("direct", "related"):
                contributes_to.append("symptom")
                notes_parts.append(f"symptom({dim.symptom_match})")

        # 2. CONTEXT coverage
        if profile.update_related or profile.possible_cause:
            hist_lower = case.historical_customer_message.lower()
            if re.search(r"\b(update|updated|updating|ios\s*11|ios11|after\s+update|upgrade)\b", hist_lower):
                contributes_to.append("context")
                notes_parts.append("context(update)")
            elif re.search(r"\b(bought|purchased|subscription|renewal)\b", hist_lower):
                contributes_to.append("context")
                notes_parts.append("context(purchase)")

        # 3. DEVICE coverage
        if profile.device and dim and dim.device_match in ("exact", "compatible"):
            contributes_to.append("device")
            notes_parts.append(f"device({dim.device_match})")

        # 4. INTENT coverage
        if case.historical_intent == primary_intent:
            contributes_to.append("intent")
            notes_parts.append("intent(direct)")
        elif case.match_tier in (EvidenceMatchTier.DIRECT_PROBLEM_MATCH, EvidenceMatchTier.RELATED_PROBLEM_MATCH):
            contributes_to.append("intent")
            notes_parts.append("intent(strong)")

        # 5. RESOLUTION PATTERN coverage
        # A case contributes resolution pattern if it has a brand response that mentions
        # the relevant troubleshooting steps
        brand_resp = case.historical_brand_response.lower() if case.historical_brand_response else ""
        if len(brand_resp) > 20:  # Non-trivial response
            contributes_to.append("resolution_pattern")
            notes_parts.append("resolution_pattern(present)")

        # Compute strength: based on tier, number of dimensions, and symptom presence
        if not contributes_to:
            strength = ContributionStrength.NONE
        elif "symptom" in contributes_to and max_strength == ContributionStrength.HIGH:
            strength = ContributionStrength.HIGH
        elif "symptom" in contributes_to:
            strength = ContributionStrength.MEDIUM
        elif max_strength == ContributionStrength.HIGH and len(contributes_to) >= 2:
            strength = ContributionStrength.MEDIUM
        elif max_strength == ContributionStrength.LOW:
            strength = ContributionStrength.LOW
        else:
            strength = ContributionStrength.LOW

        return CaseEvidenceContribution(
            case_id=case.case_id,
            match_tier=case.match_tier,
            historical_intent=case.historical_intent,
            contributes_to=list(set(contributes_to)),
            contribution_strength=strength,
            is_duplicate=False,
            allowed_for_synthesis=True,
            contribution_notes="; ".join(notes_parts) if notes_parts else "No contributing dimensions.",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Private: Dimension evaluation
    # ─────────────────────────────────────────────────────────────────────────

    def _evaluate_dimensions(
        self,
        allowed_cases: list[RetrievedEvidenceCase],
        case_contributions: list[CaseEvidenceContribution],
        profile: CustomerProblemProfile,
        primary_intent: str,
    ) -> list[DimensionCoverage]:
        """Evaluate each of the 5 evidence dimensions across all allowed contributions."""
        all_dims = ["symptom", "context", "device", "intent", "resolution_pattern"]
        result: list[DimensionCoverage] = []

        for dim_name in all_dims:
            contributing_cases = [
                c for c in case_contributions
                if dim_name in c.contributes_to and c.allowed_for_synthesis
            ]
            is_covered = len(contributing_cases) > 0

            # Strength = max strength among contributing cases
            if not is_covered:
                strength = ContributionStrength.NONE
            else:
                strength_order = {
                    ContributionStrength.HIGH: 4,
                    ContributionStrength.MEDIUM: 3,
                    ContributionStrength.LOW: 2,
                    ContributionStrength.NONE: 1,
                }
                strength = max(
                    (c.contribution_strength for c in contributing_cases),
                    key=lambda s: strength_order[s],
                )

            notes = ""
            if is_covered:
                notes = f"{len(contributing_cases)} case(s) contribute {strength.value} evidence."
            else:
                if dim_name == "symptom":
                    notes = "No cases address the primary customer symptom."
                elif dim_name == "context":
                    notes = "No contextual trigger evidence found." if (profile.update_related or profile.possible_cause) else "No context signals in customer message."
                elif dim_name == "device":
                    notes = "No device coverage." if profile.device else "No device specified in customer message."
                elif dim_name == "intent":
                    notes = "No case directly matches primary intent."
                elif dim_name == "resolution_pattern":
                    notes = "No resolution pattern evidence found."

            result.append(DimensionCoverage(
                dimension=dim_name,
                is_covered=is_covered and strength != ContributionStrength.NONE,
                coverage_strength=strength,
                contributing_case_ids=[c.case_id for c in contributing_cases],
                coverage_notes=notes,
            ))

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Private: Composite score
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_composite_score(
        self,
        dimension_coverage: list[DimensionCoverage],
        intent_consistency: float,
        num_allowed: int,
    ) -> float:
        """Compute normalized composite evidence support score [0, 1]."""
        strength_weights = {
            ContributionStrength.HIGH: 1.0,
            ContributionStrength.MEDIUM: 0.6,
            ContributionStrength.LOW: 0.25,
            ContributionStrength.NONE: 0.0,
        }
        # Dimension importance weights
        dim_weights = {
            "symptom": 0.35,
            "intent": 0.25,
            "resolution_pattern": 0.20,
            "context": 0.10,
            "device": 0.10,
        }
        dim_score = sum(
            dim_weights.get(d.dimension, 0.1) * strength_weights[d.coverage_strength]
            for d in dimension_coverage
        )
        score = round(0.75 * dim_score + 0.25 * intent_consistency, 4)
        return min(1.0, score)

    # ─────────────────────────────────────────────────────────────────────────
    # Private: Verdict determination
    # ─────────────────────────────────────────────────────────────────────────

    def _determine_verdict(
        self,
        symptom_covered: bool,
        covered_dims: list[str],
        conflict_result: ConflictDetectionResult,
        composite_score: float,
        intent_consistency: float,
        single_case_validation: EvidenceValidationResult,
    ) -> tuple[CompositeEvidenceVerdict, bool, list[str]]:
        """
        Determine the composite evidence verdict and auto_resolution_allowed flag.

        Returns:
            (verdict, auto_resolution_allowed, why_not_strong_reasons)
        """
        why_not: list[str] = []

        # Rule 1: Conflict → CONFLICTING
        if conflict_result.has_conflict and len(conflict_result.allowed_case_ids) == 0:
            return CompositeEvidenceVerdict.CONFLICTING_COMPOSITE_EVIDENCE, False, [
                "All non-conflicting cases excluded."
            ]

        # Rule 2: COMPOSITE_STRONG requires symptom (mandatory) + ≥2 independent dims
        if not symptom_covered:
            why_not.append("Primary symptom not covered by any retrieved case.")

        meaningful_dims = [d for d in covered_dims if d in ("symptom", "intent", "resolution_pattern", "context")]
        if len(meaningful_dims) < self.min_dimensions_for_strong:
            why_not.append(
                f"Only {len(meaningful_dims)} meaningful dimension(s) covered "
                f"(need {self.min_dimensions_for_strong})."
            )

        if composite_score < 0.40:
            why_not.append(f"Composite support score ({composite_score:.2f}) below 0.40 threshold.")

        if symptom_covered and len(meaningful_dims) >= self.min_dimensions_for_strong and composite_score >= 0.40:
            return CompositeEvidenceVerdict.COMPOSITE_STRONG_EVIDENCE, True, []

        # Rule 3: MODERATE if symptom is covered but not enough dimensions/score
        if symptom_covered and composite_score >= 0.25:
            return CompositeEvidenceVerdict.MODERATE_COMPOSITE_EVIDENCE, False, why_not

        # Rule 4: WEAK if only semantic overlap
        if single_case_validation.evidence_verdict == EvidenceVerdict.WEAK_EVIDENCE:
            why_not.append("Single-case validation produced WEAK_EVIDENCE; synthesis did not improve it.")
            return CompositeEvidenceVerdict.WEAK_COMPOSITE_EVIDENCE, False, why_not

        # Rule 5: Insufficient
        if single_case_validation.evidence_verdict == EvidenceVerdict.INSUFFICIENT_EVIDENCE:
            return CompositeEvidenceVerdict.INSUFFICIENT_COMPOSITE_EVIDENCE, False, why_not

        # Fallback WEAK
        return CompositeEvidenceVerdict.WEAK_COMPOSITE_EVIDENCE, False, why_not

    # ─────────────────────────────────────────────────────────────────────────
    # Private: Rationale building
    # ─────────────────────────────────────────────────────────────────────────

    def _build_rationale(
        self,
        verdict: CompositeEvidenceVerdict,
        covered_dims: list[str],
        missing_dims: list[str],
        case_contributions: list[CaseEvidenceContribution],
        duplicate_ids: list[str],
        conflict_result: ConflictDetectionResult,
        composite_score: float,
        intent_consistency: float,
    ) -> str:
        parts = [f"Composite verdict: {verdict.value}."]
        parts.append(f"Unique contributing cases: {len([c for c in case_contributions if c.allowed_for_synthesis])}.")
        if duplicate_ids:
            parts.append(f"Deduplicated: {len(duplicate_ids)} near-duplicate(s) excluded.")
        if conflict_result.has_conflict:
            parts.append(f"Conflicts: {conflict_result.explanation}")
        if covered_dims:
            parts.append(f"Covered dimensions: {', '.join(covered_dims)}.")
        if missing_dims:
            parts.append(f"Missing: {', '.join(missing_dims)}.")
        parts.append(f"Composite score: {composite_score:.2f}, intent consistency: {intent_consistency:.2f}.")
        return " ".join(parts)

    # ─────────────────────────────────────────────────────────────────────────
    # Private: Passthrough for STRONG single-case evidence
    # ─────────────────────────────────────────────────────────────────────────

    def _passthrough_strong(
        self,
        evidence_cases: list[RetrievedEvidenceCase],
        dimension_matches: list[EvidenceDimensionMatch],
        single_case_validation: EvidenceValidationResult,
    ) -> CompositeEvidencePackage:
        """Return DIRECT_STRONG_EVIDENCE when single-case validation already found STRONG."""
        return CompositeEvidencePackage(
            composite_verdict=CompositeEvidenceVerdict.DIRECT_STRONG_EVIDENCE,
            auto_resolution_allowed=True,
            unique_contributing_cases=single_case_validation.direct_problem_matches,
            symptom_covered=single_case_validation.symptom_agreement > 0.0,
            intent_consistency=single_case_validation.intent_agreement,
            composite_support_score=single_case_validation.resolution_support_strength,
            synthesis_rationale=(
                f"Passthrough: single-case evidence is already STRONG "
                f"({single_case_validation.direct_problem_matches} direct match(es), "
                f"{single_case_validation.symptom_agreement:.0%} symptom agreement). "
                "Multi-case synthesis not required."
            ),
        )
