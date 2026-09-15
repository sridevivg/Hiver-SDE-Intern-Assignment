"""
SupportGraph AI — End-to-End Outcome Taxonomy & Structured Decision Explanation (Phase 12).

Provides:
- EndToEndOutcome: Mutually exclusive terminal outcome enum for end-to-end evaluation
- DecisionExplanation: Structured "Why Did the AI Decide This?" explanation model
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class EndToEndOutcome(str, Enum):
    """
    Mutually exclusive final outcome for an end-to-end support resolution trajectory.
    Contradictory states (e.g. RESOLVED + ESCALATED simultaneously) are strictly prohibited.
    """
    SUCCESSFULLY_RESOLVED = "SUCCESSFULLY_RESOLVED"
    SAFE_AUTO_HANDLED = "SAFE_AUTO_HANDLED"
    CLARIFICATION_REQUIRED = "CLARIFICATION_REQUIRED"
    ESCALATED_EVIDENCE_LIMITED = "ESCALATED_EVIDENCE_LIMITED"
    ESCALATED_AMBIGUOUS = "ESCALATED_AMBIGUOUS"
    ESCALATED_CONFLICT = "ESCALATED_CONFLICT"
    ESCALATED_HUMAN_REQUIRED = "ESCALATED_HUMAN_REQUIRED"
    ESCALATED_VERIFICATION_FAILURE = "ESCALATED_VERIFICATION_FAILURE"
    ESCALATED_SYSTEM_FAILURE = "ESCALATED_SYSTEM_FAILURE"


class DecisionExplanation(BaseModel):
    """
    Structured, deterministic explanation answering: "Why Did the AI Decide This?".
    Combines verified positive factors, blocking negative factors, decision checklists,
    and actionable instructions for human specialists.
    """
    model_config = ConfigDict(extra="ignore", use_enum_values=True)

    decision: str = Field(..., description="High-level decision: AUTO_HANDLE, ESCALATE_TO_HUMAN, CLARIFY, or RESOLVED")
    outcome: EndToEndOutcome = Field(..., description="Specific terminal outcome in the Phase 12 taxonomy")
    summary: str = Field(..., description="Plain-language synthesis of why this decision was reached")
    positive_factors: List[str] = Field(default_factory=list, description="Verified checks and strong signals in favor")
    negative_factors: List[str] = Field(default_factory=list, description="Failed checks, missing evidence, or safety vetoes")
    recommended_human_actions: List[str] = Field(default_factory=list, description="Prioritized recommendations for human agents")
    checklist: Dict[str, bool] = Field(default_factory=dict, description="Multi-signal validation checklist results")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return self.model_dump()
