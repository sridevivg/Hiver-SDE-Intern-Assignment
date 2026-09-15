"""
SupportGraph AI — Live Human Review Queue Data Models.

Defines schemas for live human review cases, lifecycle statuses,
actions, and structural case source categorization.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid

from pydantic import BaseModel, ConfigDict, Field


class CaseSource(str, Enum):
    """Structural origin of a support case."""
    LIVE_SUPPORT = "LIVE_SUPPORT"
    DEMO = "DEMO"
    EVALUATION = "EVALUATION"
    BENCHMARK = "BENCHMARK"
    SYNTHETIC = "SYNTHETIC"
    TEST = "TEST"


class ReviewCaseStatus(str, Enum):
    """Lifecycle state of a live human review case."""
    NEW = "NEW"
    UNDER_REVIEW = "UNDER_REVIEW"
    APPROVED = "APPROVED"
    EDITED = "EDITED"
    ESCALATED = "ESCALATED"
    RESOLVED = "RESOLVED"


class LiveReviewCase(BaseModel):
    """
    Representation of a live case awaiting or having undergone human specialist review.
    Only cases with source == LIVE_SUPPORT and status in [NEW, UNDER_REVIEW] appear in the active queue.
    """
    model_config = ConfigDict(extra="ignore", use_enum_values=True)

    case_id: str = Field(default_factory=lambda: f"rev_{uuid.uuid4().hex[:10]}")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source: CaseSource = CaseSource.LIVE_SUPPORT
    status: ReviewCaseStatus = ReviewCaseStatus.NEW

    # Query & Customer Context
    customer_query: str
    conversation_id: Optional[str] = None
    priority: str = "NORMAL"  # "URGENT" for safety hazards, "NORMAL" otherwise

    # Decision & Reasons
    decision: str = "HUMAN_REVIEW_REQUIRED"
    outcome: Optional[str] = None
    escalation_reason: str
    decision_explanation: Optional[Dict[str, Any]] = None

    # Generated Guidance & Understanding
    ai_suggested_response: Optional[str] = None
    problem_understanding: Optional[Dict[str, Any]] = None
    intent: Optional[str] = None
    problem_family: Optional[str] = None

    # Evidence Context
    evidence_summary: Optional[str] = None
    related_historical_cases: List[Dict[str, Any]] = Field(default_factory=list)

    # Specialist Review Audit
    reviewer_id: Optional[str] = None
    reviewer_notes: Optional[str] = None
    edited_response: Optional[str] = None
    completed_at: Optional[str] = None


class LiveReviewQueueResponse(BaseModel):
    """Response containing active live review cases."""
    cases: List[LiveReviewCase]
    count: int


class ReviewActionRequest(BaseModel):
    """Request payload for adjudicating a review case."""
    reviewer_id: str = Field(default="specialist_01", description="Identifier of human specialist")
    notes: Optional[str] = Field(default=None, description="Optional adjudication notes")
    edited_response: Optional[str] = Field(default=None, description="Specialist edited response text if applicable")
    escalation_tier: Optional[str] = Field(default="TIER_2_TECHNICAL", description="Escalation target tier if escalating")
