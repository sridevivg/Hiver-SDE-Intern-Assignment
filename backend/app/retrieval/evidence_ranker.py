"""
SupportGraph AI — Evidence Ranker & Operational Match Classifier (Phase 10.2)

Classifies retrieved historical cases into operational evidence tiers across 5 dimensions:
  - Dimension 1: Primary Problem Family (exact, compatible, or mismatch)
  - Dimension 2: Primary Functional Symptom (direct symptom match, related symptom)
  - Dimension 3: Device / Product Context Compatibility
  - Dimension 4: Trigger / Context Alignment (update, charging, accessories, login)
  - Dimension 5: Resolution Pattern Actionability

Evidence Tiers:
  - DIRECT_PROBLEM_MATCH: Shares primary problem family & core functional symptom
  - RELATED_PROBLEM_MATCH: Shares operational problem family with device/trigger variation
  - RELATED_SYMPTOM: Shares functional symptom across different operational context
  - RELATED_CONTEXT: Shares contextual trigger (e.g. OS update) but different primary problem
  - WEAK_SEMANTIC_MATCH: Lexical/token similarity without operational problem alignment
"""
from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field

try:
    from app.core.logging import get_logger
    from app.understanding.problem_extractor import CustomerProblemProfile
    from app.understanding.problem_family_registry import (
        PROBLEM_FAMILY_DEFINITIONS,
        OperationalProblemFamily,
        ProblemFamilyDetector,
    )
except ModuleNotFoundError:
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]
    from backend.app.understanding.problem_extractor import (  # type: ignore[no-redef]
        CustomerProblemProfile,
    )
    from backend.app.understanding.problem_family_registry import (  # type: ignore[no-redef]
        PROBLEM_FAMILY_DEFINITIONS,
        OperationalProblemFamily,
        ProblemFamilyDetector,
    )

logger = get_logger(__name__)


class EvidenceMatchTier(str, Enum):
    """Operational relevance tier for retrieved historical evidence."""

    DIRECT_PROBLEM_MATCH = "DIRECT_PROBLEM_MATCH"
    RELATED_PROBLEM_MATCH = "RELATED_PROBLEM_MATCH"
    RELATED_SYMPTOM = "RELATED_SYMPTOM"
    RELATED_CONTEXT = "RELATED_CONTEXT"
    WEAK_SEMANTIC_MATCH = "WEAK_SEMANTIC_MATCH"


class RetrievedEvidenceCase(BaseModel):
    """A single retrieved historical support case with operational evidence classification."""
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    case_id: str = Field(..., description="Historical conversation or golden ID")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Dense semantic/lexical similarity score")
    operational_similarity: float = Field(..., ge=0.0, le=1.0, description="Calibrated operational problem match score")
    match_tier: EvidenceMatchTier = Field(..., description="Operational evidence tier")
    historical_customer_message: str = Field(..., description="Original customer message in historical case")
    historical_brand_response: str = Field(default="", description="Official AppleSupport reply turn")
    historical_intent: str = Field(default="general_device_support", description="Historical operational intent")
    historical_problem_family: str = Field(default="GENERAL_DEVICE_FUNCTIONALITY", description="Operational problem family")
    retrieval_explanation: str = Field(default="", description="Why this historical case was retrieved and how it relates")
    # Phase 13 Provenance & Quality fields
    source_type: str = Field(default="HISTORICAL_CORPUS_EVIDENCE", description="HISTORICAL_CORPUS_EVIDENCE or HUMAN_VALIDATED_EVIDENCE")
    provenance: str = Field(default="historical_apple_support_corpus", description="Audit provenance identifier")
    evidence_quality_score: Optional[float] = Field(default=None, description="Quality score of the evidence item")
    version: Optional[str] = Field(default=None, description="Evidence version if human-derived")

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class EvidenceRanker:
    """
    Ranks and categorizes retrieved historical support cases across 5 operational dimensions.
    """

    def __init__(self, detector: Optional[ProblemFamilyDetector] = None) -> None:
        self.detector = detector or ProblemFamilyDetector()

    def classify_evidence_tier(
        self,
        query_intent: str,
        query_profile: CustomerProblemProfile,
        hist_intent: str,
        hist_text: str,
        base_similarity: float,
        hist_family: Optional[OperationalProblemFamily | str] = None,
        hist_brand_response: str = "",
    ) -> tuple[EvidenceMatchTier, float, str]:
        """
        Classify evidence match tier and compute calibrated operational similarity score across 5 dimensions.

        Returns:
            (match_tier, operational_similarity, explanation)
        """
        hist_lower = hist_text.lower()
        q_symptom = query_profile.primary_symptom.lower()

        # ── Dimension 1: Problem Family Resolution ────────────────────────────
        q_family = getattr(
            query_profile, "primary_problem_family", OperationalProblemFamily.GENERAL_DEVICE_FUNCTIONALITY
        )
        if isinstance(q_family, str):
            try:
                q_family = OperationalProblemFamily(q_family)
            except ValueError:
                q_family = OperationalProblemFamily.GENERAL_DEVICE_FUNCTIONALITY

        # Fallback to detector if query_family is generic
        if q_family == OperationalProblemFamily.GENERAL_DEVICE_FUNCTIONALITY:
            detected_q_fam, _, _ = self.detector.detect_family(
                query_profile.user_action_or_failure or q_symptom,
                primary_symptom=q_symptom,
                candidate_intent=query_intent,
            )
            if detected_q_fam != OperationalProblemFamily.GENERAL_DEVICE_FUNCTIONALITY:
                q_family = detected_q_fam

        # Resolve historical family
        if hist_family is not None:
            if isinstance(hist_family, str):
                try:
                    h_family = OperationalProblemFamily(hist_family)
                except ValueError:
                    h_family, _, _ = self.detector.detect_family(hist_text, candidate_intent=hist_intent)
            else:
                h_family = hist_family
        else:
            h_family, _, _ = self.detector.detect_family(hist_text, candidate_intent=hist_intent)

        same_family = (
            q_family == h_family and q_family != OperationalProblemFamily.GENERAL_DEVICE_FUNCTIONALITY
        )
        compatible_family = self.detector.are_compatible(q_family, h_family)
        conflicting_family = self.detector.are_conflicting(q_family, h_family)

        same_intent = (query_intent == hist_intent and query_intent != "general_device_support")

        # ── Dimension 2: Functional Symptom Overlap ──────────────────────────
        shared_symptom = False

        # Check registered keywords for the query family
        defn = PROBLEM_FAMILY_DEFINITIONS.get(q_family)
        if defn:
            for pat_str in defn.patterns:
                if re.search(pat_str, hist_lower):
                    shared_symptom = True
                    break

        # Check specific functional symptom markers
        if not shared_symptom:
            if "battery" in q_symptom and re.search(r"\b(battery|drain|draining|charge|power)\b", hist_lower):
                shared_symptom = True
            elif "charge" in q_symptom and re.search(r"\b(charge|charging|charger|won'?t\s*charge)\b", hist_lower):
                shared_symptom = True
            elif "audio" in q_symptom and re.search(r"\b(speaker|sound|volume|mic|microphone|audio|airpod)\b", hist_lower):
                shared_symptom = True
            elif "display" in q_symptom and re.search(r"\b(screen|display|touch|cracked|flicker|black\s*screen)\b", hist_lower):
                shared_symptom = True
            elif "keyboard" in q_symptom and re.search(r"\b(keyboard|type|typing|autocorrect|letter\s+i)\b", hist_lower):
                shared_symptom = True
            elif "account" in q_symptom and re.search(r"\b(password|apple\s*id|locked|sign\s*in|2fa)\b", hist_lower):
                shared_symptom = True
            elif "billing" in q_symptom and re.search(r"\b(charge|billed|refund|subscription|itunes|receipt)\b", hist_lower):
                shared_symptom = True
            elif "wi-fi" in q_symptom or "wifi" in q_symptom:
                if re.search(r"\b(wi-?fi|wireless|network)\b", hist_lower):
                    shared_symptom = True
            elif "bluetooth" in q_symptom and re.search(r"\b(bluetooth|pair|pairing|carplay)\b", hist_lower):
                shared_symptom = True
            elif "cellular" in q_symptom and re.search(r"\b(no\s*service|cellular|lte|carrier|sim)\b", hist_lower):
                shared_symptom = True
            elif "freeze" in q_symptom or "lag" in q_symptom or "crash" in q_symptom:
                if re.search(r"\b(freeze|frozen|crash|crashing|beach\s*ball|slow|lag)\b", hist_lower):
                    shared_symptom = True
            elif "sync" in q_symptom and re.search(r"\b(sync|syncing|icloud|backup|photos\s*not\s*syncing)\b", hist_lower):
                shared_symptom = True
            elif "app" in q_symptom and re.search(r"\b(app\s*store|apps?|safari|music)\b", hist_lower):
                shared_symptom = True

        # ── Dimension 3: Device / Platform Compatibility ─────────────────────
        device_conflict = False
        if query_profile.device:
            q_dev = query_profile.device.lower()
            if "mac" in q_dev and ("watch" in hist_lower or "airpods" in hist_lower):
                device_conflict = True
            elif "watch" in q_dev and "macbook" in hist_lower:
                device_conflict = True

        # ── Dimension 4: Context / Trigger Alignment ─────────────────────────
        shared_update_context = query_profile.update_related and bool(
            re.search(r"\b(update|updated|updating|ios\s*11|ios11|sierra|high\s*sierra)\b", hist_lower)
        )

        # ── Dimension 5: Resolution Pattern Presence ──────────────────────────
        has_resolution = bool(hist_brand_response and len(hist_brand_response.strip()) >= 20)

        # ── Tier Assignment Evaluation ────────────────────────────────────────

        # Condition 0: Conflicting problem families or conflicting devices -> WEAK_SEMANTIC_MATCH
        if conflicting_family or device_conflict:
            op_sim = round(base_similarity * 0.25, 4)
            explanation = (
                f"Weak semantic match: conflicting operational problem families "
                f"(Query: '{q_family.value}' vs Historical: '{h_family.value}') or incompatible device context."
            )
            return (EvidenceMatchTier.WEAK_SEMANTIC_MATCH, op_sim, explanation)

        # Tier 1: Direct Problem Match
        # (Same family OR same specific intent) AND (shared symptom OR high lexical match)
        if (same_family or same_intent) and (shared_symptom or base_similarity >= 0.50):
            res_boost = 0.05 if has_resolution else 0.0
            op_sim = round(min(1.0, base_similarity * 1.15 + 0.12 + res_boost), 4)
            explanation = (
                f"Direct problem match: shares operational problem family '{q_family.value}' "
                f"and core functional symptom ({query_profile.primary_symptom})."
            )
            return (EvidenceMatchTier.DIRECT_PROBLEM_MATCH, op_sim, explanation)

        # Tier 2: Related Problem Match (NEW in Phase 10.2)
        # Same problem family or compatible family, with context or device variation
        if same_family or (compatible_family and shared_symptom):
            res_boost = 0.04 if has_resolution else 0.0
            op_sim = round(min(1.0, base_similarity * 1.05 + 0.08 + res_boost), 4)
            explanation = (
                f"Related problem match: addresses operational problem family '{q_family.value}' "
                f"(Historical: '{h_family.value}') with context/device variation."
            )
            return (EvidenceMatchTier.RELATED_PROBLEM_MATCH, op_sim, explanation)

        # Tier 3: Related Symptom Match
        # Shared functional symptom across different operational context
        if shared_symptom:
            op_sim = round(min(1.0, base_similarity * 0.95 + 0.05), 4)
            explanation = (
                f"Related symptom match: addresses the same functional symptom ({query_profile.primary_symptom}) "
                f"under historical family '{h_family.value}' (Intent: '{hist_intent}')."
            )
            return (EvidenceMatchTier.RELATED_SYMPTOM, op_sim, explanation)

        # Tier 4: Related Context Match
        # Shared trigger (e.g. software update) but differing primary operational problem
        if shared_update_context:
            op_sim = round(base_similarity * 0.70, 4)
            explanation = (
                f"Related context match: shares software update trigger context, but involves "
                f"different operational symptoms (Query: '{q_family.value}', Historical: '{h_family.value}')."
            )
            return (EvidenceMatchTier.RELATED_CONTEXT, op_sim, explanation)

        # Tier 5: Weak Semantic Match
        op_sim = round(base_similarity * 0.35, 4)
        explanation = (
            f"Weak semantic match: shares vocabulary/keywords but addresses a different "
            f"operational customer problem (Query: '{q_family.value}', Historical: '{h_family.value}')."
        )
        return (EvidenceMatchTier.WEAK_SEMANTIC_MATCH, op_sim, explanation)
