"""
SupportGraph AI — Conversation State & Multi-Turn Models.

Tracks full conversation trajectory, confirmed vs inferred facts, attempted actions,
turn classifications, and safe resolution / escalation states.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field
import uuid

try:
    from app.schemas.decision_explanation import (
        DecisionExplanation,
        EndToEndOutcome,
    )
except ModuleNotFoundError:
    from backend.app.schemas.decision_explanation import (  # type: ignore[no-redef]
        DecisionExplanation,
        EndToEndOutcome,
    )


class ConversationStatus(str, Enum):
    """Overall status of the conversation."""
    ACTIVE = "ACTIVE"
    AWAITING_CUSTOMER = "AWAITING_CUSTOMER"
    RESOLVED = "RESOLVED"
    ESCALATED = "ESCALATED"
    ABANDONED = "ABANDONED"


class ResolutionStage(str, Enum):
    """Current progressive stage in the resolution lifecycle."""
    NEW = "NEW"
    UNDERSTANDING = "UNDERSTANDING"
    CLARIFYING = "CLARIFYING"
    TROUBLESHOOTING = "TROUBLESHOOTING"
    AWAITING_RESULT = "AWAITING_RESULT"
    RESOLVED = "RESOLVED"
    ESCALATED = "ESCALATED"


class MessageRoleType(str, Enum):
    """Classified semantic role of an incoming customer message."""
    NEW_PROBLEM = "NEW_PROBLEM"
    CONTINUATION = "CONTINUATION"
    CLARIFICATION_ANSWER = "CLARIFICATION_ANSWER"
    ACTION_ATTEMPT_RESULT = "ACTION_ATTEMPT_RESULT"
    PROBLEM_WORSENING = "PROBLEM_WORSENING"
    RESOLUTION_CONFIRMATION = "RESOLUTION_CONFIRMATION"
    NEGATIVE_RESOLUTION_RESULT = "NEGATIVE_RESOLUTION_RESULT"
    NEW_PROBLEM_IN_EXISTING_CONVERSATION = "NEW_PROBLEM_IN_EXISTING_CONVERSATION"
    UNCLEAR_MESSAGE = "UNCLEAR_MESSAGE"


class ActionStatus(str, Enum):
    """Lifecycle status of a troubleshooting action."""
    NOT_ATTEMPTED = "NOT_ATTEMPTED"
    IN_PROGRESS = "IN_PROGRESS"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class TroubleshootingAction(BaseModel):
    """A discrete troubleshooting action with execution status."""
    action_id: str = Field(default_factory=lambda: f"act_{uuid.uuid4().hex[:8]}")
    canonical_name: str
    description: str
    prerequisites: List[str] = Field(default_factory=list)
    expected_outcome: Optional[str] = None
    is_destructive: bool = False
    status: ActionStatus = ActionStatus.NOT_ATTEMPTED
    result_notes: Optional[str] = None
    attempted_turn_index: Optional[int] = None

    model_config = ConfigDict(use_enum_values=True)


class ConversationTurn(BaseModel):
    """A single turn within the conversation history."""
    turn_index: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    speaker: str  # "customer", "agent", or "system"
    content: str
    role_type: Optional[MessageRoleType] = None
    identified_facts: Dict[str, Any] = Field(default_factory=dict)
    action_referenced: Optional[str] = None
    action_outcome: Optional[str] = None
    evidence_used: List[Dict[str, Any]] = Field(default_factory=list)
    confidence_score: Optional[float] = None

    model_config = ConfigDict(use_enum_values=True)


class EscalationPackage(BaseModel):
    """Structured package prepared when escalating to human support."""
    reason: str
    summary: str
    confirmed_facts: Dict[str, Any] = Field(default_factory=dict)
    attempted_actions: List[Dict[str, Any]] = Field(default_factory=list)
    recommended_specialist_tier: str = "TIER_2_TECHNICAL"
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(use_enum_values=True)


class ConversationState(BaseModel):
    """
    Complete state representation of an active or completed support conversation.
    Strictly distinguishes confirmed facts from unconfirmed inferences.
    """
    conversation_id: str = Field(default_factory=lambda: f"conv_{uuid.uuid4().hex[:12]}")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    status: ConversationStatus = ConversationStatus.ACTIVE
    stage: ResolutionStage = ResolutionStage.NEW
    customer_id: Optional[str] = None

    problem_family: Optional[str] = None
    operational_goal: Optional[str] = None

    # Strict separation between confirmed (stated by customer) and inferred facts
    confirmed_facts: Dict[str, Any] = Field(default_factory=dict)
    inferred_facts: Dict[str, Any] = Field(default_factory=dict)

    missing_information: List[str] = Field(default_factory=list)
    clarification_questions_asked: List[str] = Field(default_factory=list)

    # Troubleshooting tracking
    attempted_actions: List[TroubleshootingAction] = Field(default_factory=list)
    recommended_action_history: List[str] = Field(default_factory=list)
    current_action: Optional[TroubleshootingAction] = None

    # Conversation trajectory
    turns: List[ConversationTurn] = Field(default_factory=list)
    consecutive_unclear_turns: int = 0

    # Escalation / Resolution outcomes
    escalation_package: Optional[EscalationPackage] = None
    resolution_summary: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(use_enum_values=True)

    def add_turn(
        self,
        speaker: str,
        content: str,
        role_type: Optional[MessageRoleType] = None,
        identified_facts: Optional[Dict[str, Any]] = None,
        action_referenced: Optional[str] = None,
        action_outcome: Optional[str] = None,
        evidence_used: Optional[List[Dict[str, Any]]] = None,
        confidence_score: Optional[float] = None,
    ) -> ConversationTurn:
        """Append a new turn to the conversation."""
        turn_index = len(self.turns)
        turn = ConversationTurn(
            turn_index=turn_index,
            timestamp=datetime.utcnow(),
            speaker=speaker,
            content=content,
            role_type=role_type,
            identified_facts=identified_facts or {},
            action_referenced=action_referenced,
            action_outcome=action_outcome,
            evidence_used=evidence_used or [],
            confidence_score=confidence_score,
        )
        self.turns.append(turn)
        self.updated_at = datetime.utcnow()
        return turn

    def update_fact(self, key: str, value: Any, confirmed: bool = True) -> None:
        """
        Record a fact. If confirmed, saves to confirmed_facts and removes from inferred_facts.
        Never silently elevates inferences to confirmed facts without explicit flag.
        """
        if confirmed:
            self.confirmed_facts[key] = value
            self.inferred_facts.pop(key, None)
            if key in self.missing_information:
                self.missing_information.remove(key)
        else:
            if key not in self.confirmed_facts:
                self.inferred_facts[key] = value
        self.updated_at = datetime.utcnow()

    def record_attempted_action(
        self,
        canonical_name: str,
        description: str,
        status: ActionStatus = ActionStatus.FAILED,
        result_notes: Optional[str] = None,
        turn_index: Optional[int] = None,
        is_destructive: bool = False,
    ) -> TroubleshootingAction:
        """Add or update an attempted action in the record."""
        # Check if already in attempted_actions
        for existing in self.attempted_actions:
            if existing.canonical_name == canonical_name:
                existing.status = status
                if result_notes:
                    existing.result_notes = result_notes
                if turn_index is not None:
                    existing.attempted_turn_index = turn_index
                self.updated_at = datetime.utcnow()
                return existing

        action = TroubleshootingAction(
            canonical_name=canonical_name,
            description=description,
            status=status,
            result_notes=result_notes,
            attempted_turn_index=turn_index if turn_index is not None else len(self.turns) - 1,
            is_destructive=is_destructive,
        )
        self.attempted_actions.append(action)
        self.updated_at = datetime.utcnow()
        return action

    def is_action_attempted(self, canonical_name: str) -> bool:
        """Check if an action (by canonical name) was already attempted."""
        return any(act.canonical_name == canonical_name for act in self.attempted_actions)

    def mark_resolved(self, summary: Optional[str] = None, resolved_by_action: Optional[str] = None) -> None:
        """Mark conversation as successfully resolved."""
        self.status = ConversationStatus.RESOLVED
        self.stage = ResolutionStage.RESOLVED
        self.resolution_summary = {
            "resolved_at": datetime.utcnow().isoformat(),
            "problem_family": self.problem_family,
            "total_turns": len(self.turns),
            "summary": summary or "Resolved successfully through evidence-grounded steps.",
            "resolved_by_action": resolved_by_action,
            "attempted_actions_count": len(self.attempted_actions),
        }
        self.updated_at = datetime.utcnow()

    def escalate(
        self,
        reason: str,
        summary: Optional[str] = None,
        specialist_tier: str = "TIER_2_TECHNICAL",
    ) -> EscalationPackage:
        """Escalate the conversation to a human specialist with full context."""
        self.status = ConversationStatus.ESCALATED
        self.stage = ResolutionStage.ESCALATED
        package = EscalationPackage(
            reason=reason,
            summary=summary or f"Escalated due to: {reason}. Problem family: {self.problem_family}",
            confirmed_facts=dict(self.confirmed_facts),
            attempted_actions=[act.model_dump() for act in self.attempted_actions],
            recommended_specialist_tier=specialist_tier,
            timestamp=datetime.utcnow(),
        )
        self.escalation_package = package
        self.updated_at = datetime.utcnow()
        return package

    def get_decision_explanation(self) -> DecisionExplanation:
        """Construct structured 'Why Did AI Decide This?' explanation for current state."""
        if self.status == ConversationStatus.RESOLVED:
            outcome = EndToEndOutcome.SUCCESSFULLY_RESOLVED
            decision = "RESOLVED"
            summary = "Conversation successfully resolved through evidence-grounded progressive troubleshooting."
            pos_factors = [
                f"✓ Problem identified: {self.problem_family}",
                f"✓ Successfully resolved via action: {self.resolution_summary.get('resolved_by_action') if self.resolution_summary else 'troubleshooting'}",
                f"✓ Total turns to resolution: {len(self.turns)}",
            ]
            neg_factors = []
            rec_actions = []
        elif self.status == ConversationStatus.ESCALATED:
            decision = "ESCALATE_TO_HUMAN"
            esc = self.escalation_package
            reason = esc.reason if esc else "MANUAL_ESCALATION"
            if reason == "CUSTOMER_REPORTED_PROBLEM_WORSENING":
                outcome = EndToEndOutcome.ESCALATED_HUMAN_REQUIRED
                summary = "Escalated urgently: Customer reported hardware condition or symptom worsening."
            elif reason == "UNRESOLVED_AMBIGUITY_OR_UNCLEAR_MESSAGES":
                outcome = EndToEndOutcome.ESCALATED_AMBIGUOUS
                summary = "Escalated to human: Customer inquiry could not be clarified after multiple attempts."
            elif reason == "ALL_TROUBLESHOOTING_ACTIONS_EXHAUSTED":
                outcome = EndToEndOutcome.ESCALATED_EVIDENCE_LIMITED
                summary = f"Escalated to human: All progressive non-destructive troubleshooting actions exhausted for {self.problem_family}."
            else:
                outcome = EndToEndOutcome.ESCALATED_HUMAN_REQUIRED
                summary = f"Escalated to human: {reason}"

            pos_factors = [f"✓ Problem family tracked: {self.problem_family}"] if self.problem_family else []
            neg_factors = [f"✗ Escalation reason: {reason}"]
            rec_actions = [
                f"Adjudicate case for {self.problem_family} with confirmed facts: {list(self.confirmed_facts.keys())}",
                f"Review {len(self.attempted_actions)} already-attempted action(s) to avoid repeating steps.",
            ]
        elif self.stage == ResolutionStage.CLARIFYING:
            outcome = EndToEndOutcome.CLARIFICATION_REQUIRED
            decision = "CLARIFY"
            summary = "Targeted clarification required to disambiguate customer inquiry before safe troubleshooting."
            pos_factors = [f"✓ Context gathered: {list(self.confirmed_facts.keys())}"] if self.confirmed_facts else []
            neg_factors = [f"✗ Missing information: {self.missing_information}"]
            rec_actions = ["Awaiting customer response to targeted inquiry."]
        else:
            outcome = EndToEndOutcome.SAFE_AUTO_HANDLED
            decision = "AUTO_HANDLE"
            current_act = self.current_action.canonical_name if self.current_action else "troubleshooting"
            summary = f"Autonomous troubleshooting in progress: Recommending {current_act} for {self.problem_family}."
            pos_factors = [
                f"✓ Problem family: {self.problem_family}",
                f"✓ Current progressive action: {current_act}",
                f"✓ Attempted actions tracked: {len(self.attempted_actions)}",
            ]
            neg_factors = []
            rec_actions = []

        checklist = {
            "problem_identified": bool(self.problem_family),
            "no_duplicate_actions": len(self.attempted_actions) == len(set(a.canonical_name for a in self.attempted_actions)),
            "safe_non_destructive": not (self.current_action.is_destructive if self.current_action else False),
            "facts_retained": bool(self.confirmed_facts),
        }

        return DecisionExplanation(
            decision=decision,
            outcome=outcome,
            summary=summary,
            positive_factors=pos_factors,
            negative_factors=neg_factors,
            recommended_human_actions=rec_actions,
            checklist=checklist,
        )
