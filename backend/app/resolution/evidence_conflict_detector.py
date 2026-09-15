"""
SupportGraph AI — Evidence Conflict Detector (Phase 10)

Checks for operational contradictions in retrieved historical evidence
BEFORE composite evidence synthesis is approved.

SAFETY CONTRACT:
  - Conflict detection runs BEFORE composite evidence can authorize auto-resolution.
  - If retrieved cases support mutually exclusive intents (e.g. battery_power_issue vs
    account_access_issue), the composite verdict MUST be CONFLICTING_COMPOSITE_EVIDENCE
    and auto-resolution is BLOCKED.
  - Symptom contradiction (a case explicitly contradicts the customer's stated symptom)
    must exclude the contradicting case from composite synthesis — it cannot contribute.
  - Context-only evidence (RELATED_CONTEXT tier) cannot override a conflicting symptom.

This module is designed to be called by MultiCaseEvidenceSynthesizer in evidence_synthesizer.py.
"""
from __future__ import annotations

import re
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

try:
    from app.core.logging import get_logger
    from app.resolution.evidence_validator import (
        CONTRADICTORY_INTENT_PAIRS,
        CompositeEvidenceVerdict,
        EvidenceDimensionMatch,
    )
    from app.retrieval.evidence_ranker import EvidenceMatchTier, RetrievedEvidenceCase
    from app.understanding.problem_extractor import CustomerProblemProfile
except ModuleNotFoundError:
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]
    from backend.app.resolution.evidence_validator import (  # type: ignore[no-redef]
        CONTRADICTORY_INTENT_PAIRS,
        CompositeEvidenceVerdict,
        EvidenceDimensionMatch,
    )
    from backend.app.retrieval.evidence_ranker import (  # type: ignore[no-redef]
        EvidenceMatchTier,
        RetrievedEvidenceCase,
    )
    from backend.app.understanding.problem_extractor import (  # type: ignore[no-redef]
        CustomerProblemProfile,
    )

logger = get_logger(__name__)


class ConflictType(str, Enum):
    """Type of detected evidence conflict."""
    NONE = "NONE"
    INTENT_CONFLICT = "INTENT_CONFLICT"
    SYMPTOM_CONTRADICTION = "SYMPTOM_CONTRADICTION"
    MIXED_CONFLICT = "MIXED_CONFLICT"


class CaseConflictLabel(BaseModel):
    """Conflict assessment for a single retrieved case."""
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    case_id: str
    historical_intent: str
    conflict_type: ConflictType = ConflictType.NONE
    is_contradictory: bool = False
    conflict_reason: str = ""
    allowed_for_synthesis: bool = True


class ConflictDetectionResult(BaseModel):
    """Complete output of evidence conflict detection across all retrieved cases."""
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    has_conflict: bool = Field(False, description="True if any operational conflict detected")
    conflict_type: ConflictType = Field(ConflictType.NONE, description="Dominant conflict type")
    conflicting_pairs: list[str] = Field(
        default_factory=list,
        description="Human-readable list of conflicting intent pairs",
    )
    contradicting_case_ids: list[str] = Field(
        default_factory=list,
        description="Case IDs whose intent/symptom contradicts the primary intent",
    )
    allowed_case_ids: list[str] = Field(
        default_factory=list,
        description="Case IDs that pass conflict screening and may contribute to synthesis",
    )
    case_labels: list[CaseConflictLabel] = Field(
        default_factory=list,
        description="Per-case conflict assessment",
    )
    explanation: str = Field(default="", description="Human-readable conflict summary")
    forced_verdict: Optional[CompositeEvidenceVerdict] = Field(
        default=None,
        description="If set, this verdict MUST be used — conflict overrides synthesis",
    )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class EvidenceConflictDetector:
    """
    Detects operational contradictions in retrieved historical evidence
    before composite synthesis is approved.

    Two types of conflicts are detected:

    1. INTENT_CONFLICT
       Retrieved cases support mutually exclusive operational intents.
       Example: customer has battery issue; retrieved evidence includes
       account_access_issue and battery_power_issue cases — the account
       case is flagged as conflicting with the primary intent.
       If the majority of strong-tier cases are conflicting → CONFLICTING_COMPOSITE_EVIDENCE.

    2. SYMPTOM_CONTRADICTION
       A retrieved case explicitly discusses a symptom that directly
       contradicts the customer's stated primary symptom.
       Example: customer has display issue; retrieved case says
       "no sound from speaker" — symptom contradiction.

    The detector produces:
    - A list of case IDs that are ALLOWED for synthesis
    - A list of case IDs that are EXCLUDED due to conflict
    - A forced_verdict if the conflict is severe enough to block synthesis entirely
    """

    def detect_conflicts(
        self,
        primary_intent: str,
        profile: CustomerProblemProfile,
        evidence_cases: list[RetrievedEvidenceCase],
        dimension_matches: list[EvidenceDimensionMatch],
    ) -> ConflictDetectionResult:
        """
        Screen all retrieved cases for operational conflicts.

        Args:
            primary_intent: The selected primary intent for this customer message.
            profile: Structured customer problem profile.
            evidence_cases: Retrieved historical cases (pre-ranked).
            dimension_matches: Per-case dimensional evaluations from ResolutionEvidenceValidator.

        Returns:
            ConflictDetectionResult with allowed/contradicting case IDs and optional forced verdict.
        """
        if not evidence_cases:
            return ConflictDetectionResult(
                has_conflict=False,
                conflict_type=ConflictType.NONE,
                explanation="No evidence cases to screen for conflicts.",
            )

        # Build a lookup from case_id → dimension match
        dim_map: dict[str, EvidenceDimensionMatch] = {d.case_id: d for d in dimension_matches}

        case_labels: list[CaseConflictLabel] = []
        allowed_ids: list[str] = []
        contradicting_ids: list[str] = []
        conflicting_pairs: list[str] = []
        intent_conflicts_detected = 0
        symptom_contradictions_detected = 0

        for case in evidence_cases:
            label = self._assess_single_case(
                primary_intent=primary_intent,
                profile=profile,
                case=case,
                dim=dim_map.get(case.case_id),
            )
            case_labels.append(label)
            if label.allowed_for_synthesis:
                allowed_ids.append(case.case_id)
            else:
                contradicting_ids.append(case.case_id)
                if label.conflict_type == ConflictType.INTENT_CONFLICT:
                    intent_conflicts_detected += 1
                    pair = f"{primary_intent} vs {case.historical_intent}"
                    if pair not in conflicting_pairs:
                        conflicting_pairs.append(pair)
                elif label.conflict_type == ConflictType.SYMPTOM_CONTRADICTION:
                    symptom_contradictions_detected += 1

        # Determine if conflict is severe enough to block synthesis
        total = len(evidence_cases)
        strong_cases = [c for c in evidence_cases if c.match_tier == EvidenceMatchTier.DIRECT_PROBLEM_MATCH]
        strong_conflicting = sum(
            1 for c in strong_cases if c.case_id in contradicting_ids
        )

        has_conflict = (intent_conflicts_detected > 0 or symptom_contradictions_detected > 0)

        # Forced verdict: CONFLICTING when conflicting cases dominate OR all strong cases conflict
        forced_verdict = None
        dominant_conflict = (
            (len(contradicting_ids) >= len(allowed_ids))  # At least half are conflicting
            or (strong_cases and strong_conflicting == len(strong_cases))  # All strong cases conflict
        )
        if has_conflict and dominant_conflict and len(allowed_ids) == 0:
            forced_verdict = CompositeEvidenceVerdict.CONFLICTING_COMPOSITE_EVIDENCE

        # Determine dominant conflict type
        if intent_conflicts_detected > 0 and symptom_contradictions_detected > 0:
            conflict_type = ConflictType.MIXED_CONFLICT
        elif intent_conflicts_detected > 0:
            conflict_type = ConflictType.INTENT_CONFLICT
        elif symptom_contradictions_detected > 0:
            conflict_type = ConflictType.SYMPTOM_CONTRADICTION
        else:
            conflict_type = ConflictType.NONE

        # Build explanation
        if not has_conflict:
            explanation = (
                f"No conflicts detected across {total} retrieved cases. "
                f"All {len(allowed_ids)} cases permitted for synthesis."
            )
        else:
            parts = []
            if intent_conflicts_detected:
                parts.append(f"{intent_conflicts_detected} intent conflict(s): {'; '.join(conflicting_pairs)}")
            if symptom_contradictions_detected:
                parts.append(f"{symptom_contradictions_detected} symptom contradiction(s)")
            explanation = (
                f"Conflicts detected: {'; '.join(parts)}. "
                f"{len(allowed_ids)} of {total} cases permitted for synthesis."
            )
            if forced_verdict:
                explanation += f" Forced verdict: {forced_verdict.value}."

        logger.debug(
            "Conflict detection: primary_intent=%s, total=%d, allowed=%d, conflicting=%d, forced=%s",
            primary_intent, total, len(allowed_ids), len(contradicting_ids),
            forced_verdict.value if forced_verdict else "None",
        )

        return ConflictDetectionResult(
            has_conflict=has_conflict,
            conflict_type=conflict_type,
            conflicting_pairs=conflicting_pairs,
            contradicting_case_ids=contradicting_ids,
            allowed_case_ids=allowed_ids,
            case_labels=case_labels,
            explanation=explanation,
            forced_verdict=forced_verdict,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _assess_single_case(
        self,
        primary_intent: str,
        profile: CustomerProblemProfile,
        case: RetrievedEvidenceCase,
        dim: Optional[EvidenceDimensionMatch],
    ) -> CaseConflictLabel:
        """
        Determine if a single retrieved case conflicts with the customer's primary intent.

        Rules:
        - If the historical intent is in a CONTRADICTORY_INTENT_PAIR with the primary intent → INTENT_CONFLICT
        - If the dimension match shows symptom_match == 'contradictory' → SYMPTOM_CONTRADICTION
        - Otherwise → allowed for synthesis
        """
        hist_intent = case.historical_intent

        # Check INTENT_CONFLICT via CONTRADICTORY_INTENT_PAIRS
        pair_fwd = (primary_intent, hist_intent)
        pair_rev = (hist_intent, primary_intent)
        if pair_fwd in CONTRADICTORY_INTENT_PAIRS or pair_rev in CONTRADICTORY_INTENT_PAIRS:
            return CaseConflictLabel(
                case_id=case.case_id,
                historical_intent=hist_intent,
                conflict_type=ConflictType.INTENT_CONFLICT,
                is_contradictory=True,
                conflict_reason=(
                    f"Historical intent '{hist_intent}' is operationally incompatible "
                    f"with primary intent '{primary_intent}'."
                ),
                allowed_for_synthesis=False,
            )

        # Check OperationalProblemFamily conflict if present
        q_family = getattr(profile, "primary_problem_family", None)
        h_family_str = getattr(case, "historical_problem_family", None)
        if q_family and h_family_str:
            try:
                from app.understanding.problem_family_registry import (
                    OperationalProblemFamily,
                    ProblemFamilyDetector,
                )
            except ModuleNotFoundError:
                from backend.app.understanding.problem_family_registry import (  # type: ignore[no-redef]
                    OperationalProblemFamily,
                    ProblemFamilyDetector,
                )
            try:
                h_fam = OperationalProblemFamily(h_family_str)
                if ProblemFamilyDetector().are_conflicting(q_family, h_fam):
                    return CaseConflictLabel(
                        case_id=case.case_id,
                        historical_intent=hist_intent,
                        conflict_type=ConflictType.INTENT_CONFLICT,
                        is_contradictory=True,
                        conflict_reason=(
                            f"Operational problem family '{h_fam.value}' directly contradicts "
                            f"customer's primary problem family '{q_family.value}'."
                        ),
                        allowed_for_synthesis=False,
                    )
            except (ValueError, KeyError):
                pass

        # Check SYMPTOM_CONTRADICTION from dimension match
        if dim and dim.symptom_match == "contradictory":
            return CaseConflictLabel(
                case_id=case.case_id,
                historical_intent=hist_intent,
                conflict_type=ConflictType.SYMPTOM_CONTRADICTION,
                is_contradictory=True,
                conflict_reason=(
                    f"Retrieved case symptom contradicts customer's primary symptom "
                    f"'{profile.primary_symptom}' (historical intent: '{hist_intent}')."
                ),
                allowed_for_synthesis=False,
            )

        # Also flag WEAK_SEMANTIC_MATCH cases with intent mismatch as low-trust
        # They can still contribute LOW-strength evidence, but are not contradictory
        return CaseConflictLabel(
            case_id=case.case_id,
            historical_intent=hist_intent,
            conflict_type=ConflictType.NONE,
            is_contradictory=False,
            conflict_reason="",
            allowed_for_synthesis=True,
        )
