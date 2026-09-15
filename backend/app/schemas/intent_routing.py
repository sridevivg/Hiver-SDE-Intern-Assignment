"""
SupportGraph AI — Intent Routing & HITL Schemas (Phase 6)

Defines Pydantic v2 data models for:
- Top-K intent predictions with calibrated confidences
- Prediction uncertainty metrics (confidence margin, normalized entropy)
- Explainable deterministic routing decisions (AUTO_HANDLE vs ESCALATE_TO_HUMAN)
- Customer message requests and full routing responses
- Human review actions and audit log records for runtime escalation
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class RoutingDecisionType(str, Enum):
    """Routing decision enum."""
    AUTO_HANDLE = "AUTO_HANDLE"
    ESCALATE_TO_HUMAN = "ESCALATE_TO_HUMAN"


class HumanReviewActionType(str, Enum):
    """Actions available to a human reviewer adjudicating an escalated case."""
    ACCEPT_TOP_1 = "ACCEPT_TOP_1"
    SELECT_TOP_2 = "SELECT_TOP_2"
    OVERRIDE_INTENT = "OVERRIDE_INTENT"
    MARK_UNCLEAR = "MARK_UNCLEAR"


class IntentPrediction(BaseModel):
    """A single candidate intent prediction with confidence and reasoning."""
    model_config = ConfigDict(extra="ignore")

    intent: str = Field(..., description="Operational intent label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score [0.0, 1.0]")
    reasoning: str = Field(default="", description="Reasoning or evidence for this candidate intent")


class IntentAnalysis(BaseModel):
    """Structured intent analysis containing Top-K predictions and uncertainty metrics."""
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    top_predictions: list[IntentPrediction] = Field(
        ...,
        min_length=1,
        description="Top-K predicted intents ranked descending by confidence",
    )
    top_confidence: float = Field(..., ge=0.0, le=1.0, description="Top-1 candidate confidence")
    confidence_margin: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Margin between Top-1 and Top-2 confidences (top_1_conf - top_2_conf)",
    )
    normalized_entropy: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Normalized Shannon entropy [0.0 = completely certain, 1.0 = maximum confusion]",
    )
    model_name: str = Field(default="", description="Name or identifier of the classifier model")

    @property
    def top_1_intent(self) -> str:
        """Helper property for Top-1 intent name."""
        return self.top_predictions[0].intent if self.top_predictions else ""

    @property
    def top_2_intent(self) -> Optional[str]:
        """Helper property for Top-2 intent name if available."""
        return self.top_predictions[1].intent if len(self.top_predictions) > 1 else None

    @property
    def top_2_confidence(self) -> Optional[float]:
        """Helper property for Top-2 confidence score if available."""
        return self.top_predictions[1].confidence if len(self.top_predictions) > 1 else None


class RoutingDecision(BaseModel):
    """Deterministic routing decision with explainable rationale and applied thresholds."""
    model_config = ConfigDict(extra="ignore")

    decision: RoutingDecisionType = Field(..., description="Routing target: AUTO_HANDLE or ESCALATE_TO_HUMAN")
    reason: str = Field(..., description="Transparent, human-readable rationale for the routing decision")
    confidence_threshold: float = Field(..., description="Auto-handle confidence threshold applied")
    margin_threshold: float = Field(..., description="Minimum confidence margin threshold applied")
    entropy_threshold: Optional[float] = Field(default=None, description="Maximum uncertainty entropy threshold")


class CustomerMessageRequest(BaseModel):
    """Input payload representing an incoming customer support message."""
    model_config = ConfigDict(extra="ignore")

    customer_message: str = Field(..., min_length=1, description="Raw incoming customer message text")
    normalized_message: Optional[str] = Field(default=None, description="Optional pre-cleaned message text")
    conversation_id: Optional[str] = Field(default=None, description="Optional conversation or thread identifier")
    metadata: Optional[dict[str, Any]] = Field(default=None, description="Optional metadata payload")


class IntentRoutingResponse(BaseModel):
    """Complete API response for intent classification and uncertainty-aware routing."""
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    customer_message: str = Field(..., description="Original customer message")
    intent_analysis: IntentAnalysis = Field(..., description="Top-K intent analysis and uncertainty metrics")
    routing: RoutingDecision = Field(..., description="Routing decision and rationale")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 timestamp of analysis",
    )
    model_name: str = Field(default="", description="Underlying model identifier")


class RoutingThresholdConfig(BaseModel):
    """Configuration schema exposing active routing thresholds."""
    model_config = ConfigDict(extra="ignore")

    auto_handle_confidence_threshold: float
    min_confidence_margin: float
    max_uncertainty_entropy: float
    top_k_predictions_count: int


class HumanReviewActionRequest(BaseModel):
    """Payload submitted when a human reviewer adjudicates an escalated case."""
    model_config = ConfigDict(extra="ignore")

    case_id: str = Field(..., description="Unique case or message ID")
    customer_message: str = Field(..., description="Original customer message")
    action: HumanReviewActionType = Field(..., description="Reviewer action type")
    selected_intent: Optional[str] = Field(
        default=None,
        description="Explicit intent selected (required for OVERRIDE_INTENT and SELECT_TOP_2)",
    )
    reviewer_notes: Optional[str] = Field(default="", description="Optional human reviewer notes")
    reviewer_id: str = Field(default="human_reviewer", description="Identifier of the reviewer")
    top_predictions: Optional[list[IntentPrediction]] = Field(
        default=None,
        description="Candidate predictions presented to reviewer",
    )
    confidence_margin: Optional[float] = Field(default=None, description="Confidence margin of AI predictions")
    routing_reason: Optional[str] = Field(default="", description="Original routing explanation")


class HumanReviewRecord(BaseModel):
    """Audit log entry persisted to the append-only runtime review log."""
    model_config = ConfigDict(extra="ignore")

    case_id: str
    timestamp: str
    customer_message: str
    top_1_intent: str
    top_1_confidence: float
    top_2_intent: str
    top_2_confidence: float
    confidence_margin: float
    routing_decision: str
    routing_reason: str
    action_taken: str
    human_final_intent: str
    is_ai_accepted: bool
    reviewer_id: str
    notes: str
