"""
SupportGraph AI — Live Human Review Queue API Endpoints.

Provides FastAPI routes under /api/v1/human-review:
- GET  /queue           -> Active live review queue (newest first, source == LIVE_SUPPORT only)
- GET  /{case_id}       -> Details of a specific live review case
- POST /{case_id}/approve  -> Specialist approval action (moves out of active queue)
- POST /{case_id}/edit     -> Specialist edit action (moves out of active queue)
- POST /{case_id}/escalate -> Specialist escalation action (moves out of active queue)
"""

from __future__ import annotations

from typing import Any, Dict, List
from fastapi import APIRouter, Depends, HTTPException, status

try:
    from app.core.logging import get_logger
    from app.feedback.live_queue_manager import LiveQueueManager, get_live_queue_manager
    from app.schemas.human_review import (
        LiveReviewCase,
        LiveReviewQueueResponse,
        ReviewActionRequest,
    )
except ModuleNotFoundError:
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]
    from backend.app.feedback.live_queue_manager import (  # type: ignore[no-redef]
        LiveQueueManager,
        get_live_queue_manager,
    )
    from backend.app.schemas.human_review import (  # type: ignore[no-redef]
        LiveReviewCase,
        LiveReviewQueueResponse,
        ReviewActionRequest,
    )

logger = get_logger(__name__)

router = APIRouter(prefix="/human-review", tags=["Live Human Review"])


@router.get(
    "/queue",
    response_model=LiveReviewQueueResponse,
    summary="Get Active Live Human Review Queue",
    description="Returns active cases (source=LIVE_SUPPORT, status in [NEW, UNDER_REVIEW]) sorted newest first.",
)
async def get_queue(
    manager: LiveQueueManager = Depends(get_live_queue_manager),
) -> LiveReviewQueueResponse:
    """Retrieve active live review queue."""
    cases = manager.get_active_queue()
    return LiveReviewQueueResponse(cases=cases, count=len(cases))


@router.get(
    "/{case_id}",
    response_model=LiveReviewCase,
    summary="Get Single Review Case",
    description="Retrieve details of a review case by ID.",
)
async def get_case(
    case_id: str,
    manager: LiveQueueManager = Depends(get_live_queue_manager),
) -> LiveReviewCase:
    """Retrieve a single live review case."""
    case = manager.get_case(case_id)
    if not case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Live review case '{case_id}' not found.",
        )
    return case


@router.post(
    "/{case_id}/approve",
    response_model=LiveReviewCase,
    summary="Approve Review Case Response",
    description="Specialist approves the response. Case moves out of active queue and status becomes APPROVED.",
)
async def approve_case(
    case_id: str,
    request: ReviewActionRequest,
    manager: LiveQueueManager = Depends(get_live_queue_manager),
) -> LiveReviewCase:
    """Approve a review case."""
    case = manager.approve_case(
        case_id=case_id,
        reviewer_id=request.reviewer_id,
        notes=request.notes,
    )
    if not case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Live review case '{case_id}' not found.",
        )
    return case


@router.post(
    "/{case_id}/edit",
    response_model=LiveReviewCase,
    summary="Edit Review Case Response",
    description="Specialist provides an edited response. Case moves out of active queue and status becomes EDITED.",
)
async def edit_case(
    case_id: str,
    request: ReviewActionRequest,
    manager: LiveQueueManager = Depends(get_live_queue_manager),
) -> LiveReviewCase:
    """Edit response and complete review for a case."""
    if not request.edited_response or not request.edited_response.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Edited response text cannot be empty.",
        )
    case = manager.edit_case(
        case_id=case_id,
        edited_response=request.edited_response.strip(),
        reviewer_id=request.reviewer_id,
        notes=request.notes,
    )
    if not case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Live review case '{case_id}' not found.",
        )
    return case


@router.post(
    "/{case_id}/escalate",
    response_model=LiveReviewCase,
    summary="Escalate Review Case",
    description="Escalate to a higher tier specialist. Case moves out of active queue and status becomes ESCALATED.",
)
async def escalate_case(
    case_id: str,
    request: ReviewActionRequest,
    manager: LiveQueueManager = Depends(get_live_queue_manager),
) -> LiveReviewCase:
    """Escalate a review case to next tier."""
    case = manager.escalate_case(
        case_id=case_id,
        tier=request.escalation_tier or "TIER_2_TECHNICAL",
        reviewer_id=request.reviewer_id,
        notes=request.notes,
    )
    if not case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Live review case '{case_id}' not found.",
        )
    return case
