"""
SupportGraph AI — Evidence Quality Scorer (Phase 13).

Scores candidate human resolutions across 8 operational dimensions:
1. Problem Alignment (0.15)
2. Fact Alignment (0.10)
3. Action Validity (0.20)
4. Resolution Completeness (0.15)
5. Provenance Credibility (0.10)
6. Internal Consistency (0.10)
7. Safety Compliance (0.10)
8. Reproducibility (0.10)

Acceptance Threshold: >= 0.80 overall score with zero safety vetoes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import re

try:
    from app.core.logging import get_logger
    from app.feedback.schemas import (
        EvidenceQualityEvaluation,
        EvidenceQualityVerdict,
        HumanResolution,
    )
except ModuleNotFoundError:
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]
    from backend.app.feedback.schemas import (  # type: ignore[no-redef]
        EvidenceQualityEvaluation,
        EvidenceQualityVerdict,
        HumanResolution,
    )

logger = get_logger(__name__)

ACCEPTANCE_THRESHOLD = 0.80
HIGH_QUALITY_THRESHOLD = 0.90


class EvidenceQualityScorer:
    """
    Computes a weighted, multi-dimensional quality score for human resolutions.
    Enforces hard safety vetoes that immediately override numerical scores.
    """

    def evaluate(self, resolution: HumanResolution) -> EvidenceQualityEvaluation:
        """
        Evaluate candidate resolution across 8 dimensions.
        """
        dim_scores: Dict[str, float] = {}
        positive_factors: List[str] = []
        blocking_factors: List[str] = []
        vetoed = False

        res_text = resolution.final_resolution.strip()
        prob_text = resolution.customer_problem.strip()
        res_lower = res_text.lower()

        # 1. Problem Alignment (Weight: 0.15)
        # Check keyword overlap or problem reference
        prob_words = set(re.findall(r"\w{4,}", prob_text.lower()))
        res_words = set(re.findall(r"\w{4,}", res_lower))
        overlap = len(prob_words.intersection(res_words))
        if overlap >= 2 or len(res_words) >= 12:
            dim_scores["problem_alignment"] = 1.0
            positive_factors.append("Strong problem alignment with customer inquiry keywords.")
        elif overlap >= 1:
            dim_scores["problem_alignment"] = 0.75
        else:
            dim_scores["problem_alignment"] = 0.50
            blocking_factors.append("Weak direct keyword overlap between problem and resolution.")

        # 2. Fact Alignment (Weight: 0.10)
        facts = resolution.confirmed_facts or {}
        if facts:
            dim_scores["fact_alignment"] = 1.0
            positive_factors.append(f"Confirmed customer facts present ({len(facts)} attributes).")
        else:
            dim_scores["fact_alignment"] = 0.80  # Default neutral score when facts not strictly required

        # 3. Action Validity (Weight: 0.20)
        if resolution.actions_attempted and len(resolution.actions_attempted) >= 1:
            if resolution.actions_that_worked:
                dim_scores["action_validity"] = 1.0
                positive_factors.append(f"Clear verified working actions identified: {', '.join(resolution.actions_that_worked)}.")
            else:
                dim_scores["action_validity"] = 0.85
                positive_factors.append("Canonical troubleshooting actions documented.")
        else:
            dim_scores["action_validity"] = 0.30
            blocking_factors.append("No structured troubleshooting actions documented.")

        # 4. Resolution Completeness (Weight: 0.15)
        word_count = len(res_text.split())
        if word_count >= 20:
            dim_scores["resolution_completeness"] = 1.0
            positive_factors.append("Comprehensive, detailed resolution guidance provided.")
        elif word_count >= 10:
            dim_scores["resolution_completeness"] = 0.80
        else:
            dim_scores["resolution_completeness"] = 0.40
            blocking_factors.append("Resolution text is overly terse or brief.")

        # 5. Provenance Credibility (Weight: 0.10)
        if resolution.evidence_sources and len(resolution.evidence_sources) > 0:
            dim_scores["provenance_credibility"] = 1.0
            positive_factors.append("External or internal documentation sources cited.")
        else:
            dim_scores["provenance_credibility"] = 0.75  # Specialist direct resolution without formal URL

        # 6. Internal Consistency (Weight: 0.10)
        if resolution.resolution_status in ("RESOLVED", "ESCALATED") and resolution.problem_family:
            dim_scores["internal_consistency"] = 1.0
            positive_factors.append("Logical coherence between problem family and resolution state.")
        else:
            dim_scores["internal_consistency"] = 0.50
            blocking_factors.append("Inconsistent problem family or resolution status.")

        # 7. Safety Compliance (Weight: 0.10)
        # Check for safety red flags
        if re.search(r"\b(jailbreak|bypass|hack|pirate|crack)\b", res_lower):
            dim_scores["safety_compliance"] = 0.0
            vetoed = True
            blocking_factors.append("SAFETY VETO: Resolution references unauthorized or unsafe tools.")
        else:
            dim_scores["safety_compliance"] = 1.0
            positive_factors.append("Full compliance with safety and brand standards.")

        # 8. Reproducibility (Weight: 0.10)
        # Step-by-step markers or settings navigation cues
        if re.search(r"\b(settings|navigate|tap|click|restart|select|step|first|then)\b", res_lower):
            dim_scores["reproducibility"] = 1.0
            positive_factors.append("Clear actionable navigation steps ensuring reproducibility.")
        else:
            dim_scores["reproducibility"] = 0.70

        # Weighted calculation
        weights = {
            "problem_alignment": 0.15,
            "fact_alignment": 0.10,
            "action_validity": 0.20,
            "resolution_completeness": 0.15,
            "provenance_credibility": 0.10,
            "internal_consistency": 0.10,
            "safety_compliance": 0.10,
            "reproducibility": 0.10,
        }

        overall_score = sum(dim_scores.get(k, 0.5) * w for k, w in weights.items())
        overall_score = round(min(1.0, max(0.0, overall_score)), 3)

        # Verdict assignment
        if vetoed or overall_score < ACCEPTANCE_THRESHOLD:
            verdict = EvidenceQualityVerdict.REJECTED_QUALITY
        elif overall_score >= HIGH_QUALITY_THRESHOLD:
            verdict = EvidenceQualityVerdict.HIGH_QUALITY
        else:
            verdict = EvidenceQualityVerdict.ACCEPTABLE_QUALITY

        return EvidenceQualityEvaluation(
            overall_score=overall_score,
            verdict=verdict,
            dimension_scores=dim_scores,
            positive_factors=positive_factors,
            blocking_factors=blocking_factors,
            vetoed=vetoed,
        )
