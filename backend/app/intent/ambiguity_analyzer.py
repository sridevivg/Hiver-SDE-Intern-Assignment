"""
SupportGraph AI — Ambiguity Analyzer (Phase 7)

Analyzes whether competing candidate intents represent:
- GENUINE_AMBIGUITY: Message is vague, incomplete, or lacks a concrete failing function
- CAUSE_VS_SYMPTOM: Competing intents reflect a causal trigger (update) vs an isolated symptom (battery)
- MULTI_SYMPTOM: Customer reports multiple distinct hardware/software malfunctions
- CLEAR_PRIMARY: Unambiguous operational intent with sufficient supporting evidence
- UNCLEAR_INSUFFICIENT: Insufficient information to classify

Generates transparent human-readable escalation rationales when ambiguity is detected.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

try:
    from app.core.logging import get_logger
    from app.schemas.intent_routing import IntentAnalysis, RoutingDecisionType
    from app.understanding.problem_extractor import CustomerProblemProfile
except ModuleNotFoundError:
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]
    from backend.app.schemas.intent_routing import (  # type: ignore[no-redef]
        IntentAnalysis,
        RoutingDecisionType,
    )
    from backend.app.understanding.problem_extractor import CustomerProblemProfile  # type: ignore[no-redef]

logger = get_logger(__name__)


class AmbiguityType(str, Enum):
    """Categorical classification of prediction ambiguity."""
    CLEAR_PRIMARY = "CLEAR_PRIMARY"
    CAUSE_VS_SYMPTOM = "CAUSE_VS_SYMPTOM"
    GENUINE_AMBIGUITY = "GENUINE_AMBIGUITY"
    MULTI_SYMPTOM = "MULTI_SYMPTOM"
    UNCLEAR_INSUFFICIENT = "UNCLEAR_INSUFFICIENT"


class AmbiguityAnalysisResult(BaseModel):
    """Result of ambiguity analysis over customer message and candidate predictions."""
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    ambiguity_type: AmbiguityType = Field(..., description="Nature of the ambiguity detected")
    is_ambiguous: bool = Field(..., description="Whether the case is ambiguous and requires human review")
    primary_intent: str = Field(..., description="Selected primary operational intent")
    contextual_cause_intent: Optional[str] = Field(default=None, description="Contextual causal intent if any")
    competing_candidates: list[str] = Field(default_factory=list, description="List of competing intent names")
    confidence_margin: float = Field(default=0.0, description="Margin between top two candidates")
    ambiguity_reason: str = Field(..., description="Explainable reason for ambiguity detection or clarity")
    routing_recommendation: RoutingDecisionType = Field(
        ...,
        description="Recommended routing destination (AUTO_HANDLE or ESCALATE_TO_HUMAN)",
    )

    @property
    def contextual_cause(self) -> Optional[str]:
        return self.contextual_cause_intent

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class AmbiguityAnalyzer:
    """
    Evaluates candidate predictions and extracted problem structure for operational ambiguity.
    """

    def analyze(
        self,
        message: str,
        profile: CustomerProblemProfile,
        analysis: IntentAnalysis,
        primary_intent: str,
        contextual_cause_intent: Optional[str] = None,
    ) -> AmbiguityAnalysisResult:
        """
        Analyze ambiguity type and determine whether escalation is warranted.
        """
        top_1 = analysis.top_1_intent
        top_2 = analysis.top_2_intent
        margin = analysis.confidence_margin
        top_conf = analysis.top_confidence
        entropy = analysis.normalized_entropy or 0.0

        candidates = [p.intent for p in analysis.top_predictions]

        # Case 1: Insufficient Information
        if profile.information_sufficiency == "insufficient" or top_1 == "unclear_needs_review":
            return AmbiguityAnalysisResult(
                ambiguity_type=AmbiguityType.UNCLEAR_INSUFFICIENT,
                is_ambiguous=True,
                primary_intent="unclear_needs_review",
                contextual_cause_intent=contextual_cause_intent,
                competing_candidates=candidates[:2],
                confidence_margin=margin,
                ambiguity_reason=(
                    "The customer message contains insufficient operational detail or is fragmented; "
                    "requires human operator clarification."
                ),
                routing_recommendation=RoutingDecisionType.ESCALATE_TO_HUMAN,
            )

        # Case 2: Multi-Symptom Inquiry
        if len(profile.secondary_symptoms) >= 2 or (len(profile.secondary_symptoms) >= 1 and margin < 0.25):
            return AmbiguityAnalysisResult(
                ambiguity_type=AmbiguityType.MULTI_SYMPTOM,
                is_ambiguous=True,
                primary_intent=primary_intent,
                contextual_cause_intent=contextual_cause_intent,
                competing_candidates=candidates[:2],
                confidence_margin=margin,
                ambiguity_reason=(
                    f"Customer reports multiple distinct symptoms ({profile.primary_symptom} and "
                    f"{', '.join(profile.secondary_symptoms)}); requires human review to prioritize resolution."
                ),
                routing_recommendation=RoutingDecisionType.ESCALATE_TO_HUMAN,
            )

        # Case 3: Genuine Ambiguity (Vague post-update complaint without specific symptom)
        if profile.update_related and "unspecified" in profile.primary_symptom.lower():
            return AmbiguityAnalysisResult(
                ambiguity_type=AmbiguityType.GENUINE_AMBIGUITY,
                is_ambiguous=True,
                primary_intent=primary_intent,
                contextual_cause_intent="software_update_problem",
                competing_candidates=candidates[:2],
                confidence_margin=margin,
                ambiguity_reason=(
                    "The customer attributes a problem to a software update but does not specify "
                    "which concrete device function is failing; competing between software_update_problem "
                    "and general_device_support."
                ),
                routing_recommendation=RoutingDecisionType.ESCALATE_TO_HUMAN,
            )

        # Case 4: Cause vs Symptom Disambiguation
        if contextual_cause_intent is not None:
            # If primary intent has reasonable confidence (e.g. battery drain), it is safely resolved
            if top_conf >= 0.70 and margin >= 0.10:
                return AmbiguityAnalysisResult(
                    ambiguity_type=AmbiguityType.CAUSE_VS_SYMPTOM,
                    is_ambiguous=False,
                    primary_intent=primary_intent,
                    contextual_cause_intent=contextual_cause_intent,
                    competing_candidates=candidates[:2],
                    confidence_margin=margin,
                    ambiguity_reason=(
                        f"Resolved cause vs symptom: '{primary_intent}' is identified as the primary operational "
                        f"symptom, with '{contextual_cause_intent}' serving as contextual causality."
                    ),
                    routing_recommendation=RoutingDecisionType.AUTO_HANDLE,
                )
            else:
                return AmbiguityAnalysisResult(
                    ambiguity_type=AmbiguityType.CAUSE_VS_SYMPTOM,
                    is_ambiguous=True,
                    primary_intent=primary_intent,
                    contextual_cause_intent=contextual_cause_intent,
                    competing_candidates=candidates[:2],
                    confidence_margin=margin,
                    ambiguity_reason=(
                        f"Ambiguous separation between cause ('{contextual_cause_intent}') and symptom "
                        f"('{primary_intent}'); margin ({margin:.2f}) is below decisive threshold."
                    ),
                    routing_recommendation=RoutingDecisionType.ESCALATE_TO_HUMAN,
                )

        # Case 5: Close Intent Competition with Low Margin
        if margin < 0.15 and top_2 is not None:
            return AmbiguityAnalysisResult(
                ambiguity_type=AmbiguityType.GENUINE_AMBIGUITY,
                is_ambiguous=True,
                primary_intent=primary_intent,
                contextual_cause_intent=contextual_cause_intent,
                competing_candidates=candidates[:2],
                confidence_margin=margin,
                ambiguity_reason=(
                    f"Close competition between '{top_1}' and '{top_2}' (margin: {margin:.2f}); "
                    f"insufficient separation to safely auto-handle."
                ),
                routing_recommendation=RoutingDecisionType.ESCALATE_TO_HUMAN,
            )

        # Case 6: Clear Primary Problem
        if top_conf >= 0.85 and margin >= 0.15:
            return AmbiguityAnalysisResult(
                ambiguity_type=AmbiguityType.CLEAR_PRIMARY,
                is_ambiguous=False,
                primary_intent=primary_intent,
                contextual_cause_intent=None,
                competing_candidates=candidates[:2],
                confidence_margin=margin,
                ambiguity_reason=(
                    f"Clear operational intent '{primary_intent}' supported by high confidence ({top_conf:.2f}) "
                    f"and decisive separation margin ({margin:.2f})."
                ),
                routing_recommendation=RoutingDecisionType.AUTO_HANDLE,
            )

        # Default fallback escalation for borderline confidence
        return AmbiguityAnalysisResult(
            ambiguity_type=AmbiguityType.GENUINE_AMBIGUITY,
            is_ambiguous=True,
            primary_intent=primary_intent,
            contextual_cause_intent=None,
            competing_candidates=candidates[:2],
            confidence_margin=margin,
            ambiguity_reason=(
                f"Moderate confidence ({top_conf:.2f}) below auto-handle safety threshold; "
                f"escalating for human confirmation."
            ),
            routing_recommendation=RoutingDecisionType.ESCALATE_TO_HUMAN,
        )
