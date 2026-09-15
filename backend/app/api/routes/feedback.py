"""
SupportGraph AI — Human-in-the-Loop Feedback & Evidence REST API (Phase 13).

Provides endpoints under /api/v1/feedback/:
- POST /resolutions: Capture human resolution
- GET  /resolutions/{resolution_id}: Retrieve candidate resolution
- POST /resolutions/{resolution_id}/review: Submit for review / approve
- GET  /resolutions/{resolution_id}/validation: Execute deterministic validation gate
- POST /resolutions/{resolution_id}/evaluate: Run offline simulation evaluation
- POST /resolutions/{resolution_id}/promote: Promote approved resolution to evidence
- GET  /evidence: List approved evidence items
- GET  /evidence/{evidence_id}: Retrieve single approved evidence item
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

try:
    from app.core.logging import get_logger
    from app.feedback.approved_store import ApprovedEvidenceStore
    from app.feedback.candidate_store import CandidateEvidenceStore
    from app.feedback.promotion_pipeline import PromotionPipeline
    from app.feedback.schemas import (
        ApprovedEvidenceItem,
        HumanResolution,
        ResolutionLifecycleState,
        ReviewDecision,
    )
except ModuleNotFoundError:
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]
    from backend.app.feedback.approved_store import (  # type: ignore[no-redef]
        ApprovedEvidenceStore,
    )
    from backend.app.feedback.candidate_store import (  # type: ignore[no-redef]
        CandidateEvidenceStore,
    )
    from backend.app.feedback.promotion_pipeline import (  # type: ignore[no-redef]
        PromotionPipeline,
    )
    from backend.app.feedback.schemas import (  # type: ignore[no-redef]
        ApprovedEvidenceItem,
        HumanResolution,
        ResolutionLifecycleState,
        ReviewDecision,
    )

logger = get_logger(__name__)

router = APIRouter(prefix="/feedback", tags=["Human-in-the-Loop Feedback"])

_pipeline = PromotionPipeline()


class ReviewSubmissionRequest(BaseModel):
    reviewer_id: str = Field(..., description="Reviewer or specialist ID")
    notes: Optional[str] = Field(default="", description="Reviewer feedback notes")


class PromotionRequest(BaseModel):
    reviewer_id: str = Field(..., description="Authorizing reviewer ID")
    version: str = Field(default="1.0.0", description="Target evidence version")


@router.post(
    "/resolutions",
    response_model=HumanResolution,
    status_code=status.HTTP_201_CREATED,
    summary="Capture Specialist Resolution",
)
async def capture_resolution(resolution: HumanResolution) -> HumanResolution:
    """Capture a human specialist resolution as candidate evidence."""
    try:
        return _pipeline.capture_resolution(resolution)
    except Exception as exc:
        logger.error("Failed to capture resolution: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))


@router.get(
    "/resolutions/{resolution_id}",
    response_model=HumanResolution,
    summary="Get Resolution Details",
)
async def get_resolution(resolution_id: str) -> HumanResolution:
    """Retrieve candidate human resolution by ID."""
    res = _pipeline.candidate_store.get_candidate(resolution_id)
    if not res:
        raise HTTPException(status_code=404, detail=f"Resolution {resolution_id} not found.")
    return res


@router.get(
    "/resolutions",
    response_model=List[HumanResolution],
    summary="List Candidate Resolutions",
)
async def list_resolutions(
    status: Optional[ResolutionLifecycleState] = None,
    problem_family: Optional[str] = None,
) -> List[HumanResolution]:
    """List candidate human resolutions with optional filters."""
    return _pipeline.candidate_store.list_candidates(status=status, problem_family=problem_family)


@router.post(
    "/resolutions/{resolution_id}/submit-review",
    response_model=HumanResolution,
    summary="Submit Resolution for Review",
)
async def submit_for_review(
    resolution_id: str,
    req: ReviewSubmissionRequest,
) -> HumanResolution:
    """Move candidate resolution into review queue."""
    try:
        return _pipeline.submit_for_review(resolution_id, actor_id=req.reviewer_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get(
    "/resolutions/{resolution_id}/validation",
    response_model=ReviewDecision,
    summary="Validate Resolution Deterministically",
)
async def validate_resolution(resolution_id: str) -> ReviewDecision:
    """Execute 10-check deterministic validation gate and quality scoring."""
    try:
        _, decision = _pipeline.validate_resolution(resolution_id)
        return decision
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post(
    "/resolutions/{resolution_id}/evaluate",
    response_model=Dict[str, Any],
    summary="Run Offline Simulation Evaluation",
)
async def evaluate_resolution(
    resolution_id: str,
    req: ReviewSubmissionRequest,
) -> Dict[str, Any]:
    """Run offline simulation evaluation in test isolation."""
    try:
        return _pipeline.run_offline_evaluation(resolution_id, actor_id=req.reviewer_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post(
    "/resolutions/{resolution_id}/approve",
    response_model=HumanResolution,
    summary="Approve Resolution for Promotion",
)
async def approve_resolution(
    resolution_id: str,
    req: ReviewSubmissionRequest,
) -> HumanResolution:
    """Authorize candidate resolution for promotion."""
    try:
        return _pipeline.approve_resolution(
            resolution_id=resolution_id,
            reviewer_id=req.reviewer_id,
            notes=req.notes or "",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post(
    "/resolutions/{resolution_id}/promote",
    response_model=ApprovedEvidenceItem,
    summary="Promote Resolution to Approved Evidence",
)
async def promote_resolution(
    resolution_id: str,
    req: PromotionRequest,
) -> ApprovedEvidenceItem:
    """Promote validated, approved resolution into trusted ApprovedEvidenceStore."""
    try:
        return _pipeline.promote_to_evidence(
            resolution_id=resolution_id,
            reviewer_id=req.reviewer_id,
            version=req.version,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get(
    "/evidence",
    response_model=List[ApprovedEvidenceItem],
    summary="List Approved Evidence Items",
)
async def list_approved_evidence(
    problem_family: Optional[str] = None,
) -> List[ApprovedEvidenceItem]:
    """List approved operational evidence items accessible to retrieval."""
    return _pipeline.approved_store.list_evidence(problem_family=problem_family)


@router.get(
    "/evidence/{evidence_id}",
    response_model=ApprovedEvidenceItem,
    summary="Get Single Approved Evidence Item",
)
async def get_approved_evidence(evidence_id: str) -> ApprovedEvidenceItem:
    """Retrieve approved operational evidence item by ID."""
    item = _pipeline.approved_store.get_evidence(evidence_id)
    if not item:
        raise HTTPException(status_code=404, detail=f"Evidence item {evidence_id} not found.")
    return item
