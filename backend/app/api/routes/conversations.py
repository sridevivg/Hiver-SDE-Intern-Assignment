"""
SupportGraph AI — Multi-Turn Support Conversation API Endpoints (Phase 11).

Provides FastAPI endpoints:
- POST /api/v1/conversations/start: Initialize a new multi-turn support conversation
- POST /api/v1/conversations/{conversation_id}/message: Process incoming customer turn
- GET  /api/v1/conversations/{conversation_id}/state: Retrieve full current conversation state
- GET  /api/v1/conversations/{conversation_id}/history: Retrieve ordered conversation turns
- GET  /api/v1/conversations/{conversation_id}/resolution-summary: Resolution outcome or escalation package
- POST /api/v1/conversations/{conversation_id}/resolve: Mark conversation as resolved
- GET  /api/v1/conversations/{conversation_id}/audit: Append-only audit trail
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

try:
    from app.conversation.conversation_manager import get_conversation_manager
    from app.conversation.conversation_state import (
        ConversationState,
        ConversationStatus,
        ConversationTurn,
        EscalationPackage,
        ResolutionStage,
    )
    from app.core.logging import get_logger
except ModuleNotFoundError:
    from backend.app.conversation.conversation_manager import (  # type: ignore[no-redef]
        get_conversation_manager,
    )
    from backend.app.conversation.conversation_state import (  # type: ignore[no-redef]
        ConversationState,
        ConversationStatus,
        ConversationTurn,
        EscalationPackage,
        ResolutionStage,
    )
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]

try:
    from app.feedback.live_queue_manager import get_live_queue_manager
    from app.schemas.human_review import CaseSource
except ModuleNotFoundError:
    from backend.app.feedback.live_queue_manager import get_live_queue_manager  # type: ignore[no-redef]
    from backend.app.schemas.human_review import CaseSource  # type: ignore[no-redef]

logger = get_logger(__name__)

router = APIRouter(prefix="/conversations", tags=["Multi-Turn Support Conversations"])


class StartConversationRequest(BaseModel):
    """Request to initiate a multi-turn conversation."""
    customer_id: Optional[str] = Field(default=None, description="Optional customer identifier")
    initial_message: Optional[str] = Field(default=None, description="Optional initial problem message")


class StartConversationResponse(BaseModel):
    """Response when starting a conversation."""
    conversation_id: str
    status: str
    stage: str
    problem_family: Optional[str] = None
    initial_agent_response: Optional[str] = None
    confirmed_facts: Dict[str, Any] = Field(default_factory=dict)
    turns_count: int = 0


class CustomerTurnRequest(BaseModel):
    """Customer message payload for subsequent turns."""
    message: str = Field(..., min_length=1, description="Customer message text")


class TurnResponse(BaseModel):
    """Response to a customer turn."""
    conversation_id: str
    turn_index: int
    agent_response: str
    role_type: str
    status: str
    stage: str
    confirmed_facts: Dict[str, Any] = Field(default_factory=dict)
    current_action: Optional[Dict[str, Any]] = None
    escalation_package: Optional[Dict[str, Any]] = None


class ResolveRequest(BaseModel):
    """Request to explicitly mark conversation resolved."""
    summary: Optional[str] = Field(default=None, description="Resolution summary notes")


@router.post(
    "/start",
    response_model=StartConversationResponse,
    summary="Start a new multi-turn conversation",
    status_code=status.HTTP_201_CREATED,
)
async def start_conversation(request: StartConversationRequest) -> StartConversationResponse:
    """Initialize a stateful conversation."""
    manager = get_conversation_manager()
    state, first_turn = manager.create_conversation(
        customer_id=request.customer_id,
        initial_message=request.initial_message,
    )
    if (state.status == "ESCALATED" or state.escalation_package) and request.initial_message:
        try:
            queue_mgr = get_live_queue_manager()
            esc_reason = state.escalation_package.reason if state.escalation_package else "Conversation escalated to human specialist."
            queue_mgr.create_case(
                customer_query=request.initial_message,
                escalation_reason=esc_reason,
                source=CaseSource.LIVE_SUPPORT,
                conversation_id=state.conversation_id,
                decision="HUMAN_REVIEW_REQUIRED",
                ai_suggested_response=first_turn.content if first_turn else "A specialist will assist you shortly.",
                problem_family=state.problem_family or "GENERAL_DEVICE_SUPPORT",
                problem_understanding={"confirmed_facts": state.confirmed_facts},
            )
        except Exception as q_exc:
            logger.warning("Failed to record live review case in start_conversation: %s", q_exc)

    return StartConversationResponse(
        conversation_id=state.conversation_id,
        status=state.status,
        stage=state.stage,
        problem_family=state.problem_family,
        initial_agent_response=first_turn.content if first_turn else None,
        confirmed_facts=state.confirmed_facts,
        turns_count=len(state.turns),
    )


@router.post(
    "/{conversation_id}/message",
    response_model=TurnResponse,
    summary="Send customer message to conversation",
)
async def send_message(conversation_id: str, request: CustomerTurnRequest) -> TurnResponse:
    """Process customer message within active conversation state."""
    manager = get_conversation_manager()
    try:
        state, agent_turn = manager.process_message(conversation_id, request.message)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

    if state.status == "ESCALATED" or state.escalation_package:
        try:
            queue_mgr = get_live_queue_manager()
            esc_reason = state.escalation_package.reason if state.escalation_package else "Conversation escalated to human specialist."
            queue_mgr.create_case(
                customer_query=request.message,
                escalation_reason=esc_reason,
                source=CaseSource.LIVE_SUPPORT,
                conversation_id=conversation_id,
                decision="HUMAN_REVIEW_REQUIRED",
                ai_suggested_response=agent_turn.content,
                problem_family=state.problem_family or "GENERAL_DEVICE_SUPPORT",
                problem_understanding={"confirmed_facts": state.confirmed_facts},
            )
        except Exception as q_exc:
            logger.warning("Failed to record live review case in send_message: %s", q_exc)

    return TurnResponse(
        conversation_id=conversation_id,
        turn_index=agent_turn.turn_index,
        agent_response=agent_turn.content,
        role_type=agent_turn.role_type or "UNKNOWN",
        status=state.status,
        stage=state.stage,
        confirmed_facts=state.confirmed_facts,
        current_action=state.current_action.model_dump() if state.current_action else None,
        escalation_package=state.escalation_package.model_dump() if state.escalation_package else None,
    )


@router.get(
    "/{conversation_id}/state",
    response_model=Dict[str, Any],
    summary="Get complete conversation state",
)
async def get_state(conversation_id: str) -> Dict[str, Any]:
    """Retrieve full internal conversation state."""
    manager = get_conversation_manager()
    state = manager.get_conversation(conversation_id)
    if not state:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    return state.model_dump()


@router.get(
    "/{conversation_id}/history",
    response_model=List[Dict[str, Any]],
    summary="Get conversation turns history",
)
async def get_history(conversation_id: str) -> List[Dict[str, Any]]:
    """Retrieve chronologically ordered conversation turns."""
    manager = get_conversation_manager()
    history = manager.get_history(conversation_id)
    if not history:
        # Check if conversation exists but has no turns yet
        state = manager.get_conversation(conversation_id)
        if not state:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    return history


@router.get(
    "/{conversation_id}/resolution-summary",
    response_model=Dict[str, Any],
    summary="Get resolution or escalation outcome summary",
)
async def get_resolution_summary(conversation_id: str) -> Dict[str, Any]:
    """Retrieve resolution summary or escalation package."""
    manager = get_conversation_manager()
    state = manager.get_conversation(conversation_id)
    if not state:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    if state.resolution_summary:
        return {
            "type": "RESOLVED",
            "summary": state.resolution_summary,
            "confirmed_facts": state.confirmed_facts,
            "total_turns": len(state.turns),
        }
    elif state.escalation_package:
        return {
            "type": "ESCALATED",
            "escalation": state.escalation_package.model_dump(),
            "confirmed_facts": state.confirmed_facts,
            "total_turns": len(state.turns),
        }
    else:
        return {
            "type": "ACTIVE",
            "stage": state.stage,
            "confirmed_facts": state.confirmed_facts,
            "total_turns": len(state.turns),
        }


@router.get(
    "/{conversation_id}/decision-explanation",
    response_model=Dict[str, Any],
    summary="Get structured 'Why Did AI Decide This?' explanation",
)
async def get_decision_explanation(conversation_id: str) -> Dict[str, Any]:
    """Retrieve structured deterministic decision explanation."""
    manager = get_conversation_manager()
    state = manager.get_conversation(conversation_id)
    if not state:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    return state.get_decision_explanation().model_dump()


@router.post(
    "/{conversation_id}/resolve",
    response_model=Dict[str, Any],
    summary="Explicitly mark conversation as resolved",
)
async def resolve_conversation(conversation_id: str, request: ResolveRequest) -> Dict[str, Any]:
    """Mark conversation as resolved."""
    manager = get_conversation_manager()
    try:
        state = manager.resolve_conversation(conversation_id, summary=request.summary)
        return state.model_dump()
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))


@router.get(
    "/{conversation_id}/audit",
    response_model=List[Dict[str, Any]],
    summary="Get append-only audit trail",
)
async def get_audit_trail(conversation_id: str) -> List[Dict[str, Any]]:
    """Retrieve full audit log entries for conversation."""
    manager = get_conversation_manager()
    state = manager.get_conversation(conversation_id)
    if not state:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    return manager.get_audit_trail(conversation_id)
