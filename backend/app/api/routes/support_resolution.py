"""
SupportGraph AI — Evidence-Grounded Support Resolution API Endpoints (Phase 8)

Provides FastAPI endpoints:
- POST /api/v1/resolution/resolve: End-to-end evidence-grounded resolution pipeline
- POST /api/v1/resolution/understand: Problem extractor endpoint (structured device, symptom, causality)
- POST /api/v1/resolution/retrieve-evidence: Historical evidence retrieval with operational tier classification
- POST /api/v1/resolution/validate-evidence: Operational evidence validation endpoint
- POST /api/v1/resolution/verify-response: Response grounding and safety verification endpoint
- GET  /api/v1/resolution/audit/stats: Runtime resolution audit metrics
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

try:
    from app.core.config import settings
    from app.core.logging import get_logger
    from app.resolution.evidence_validator import (
        EvidenceValidationResult,
        ResolutionEvidenceValidator,
    )
    from app.resolution.resolution_auditor import ResolutionAuditor
    from app.resolution.response_generator import (
        EvidenceGroundedResponseGenerator,
        GroundedResponseCandidate,
    )
    from app.resolution.response_verifier import (
        ResponseGroundingResult,
        ResponseGroundingVerifier,
    )
    from app.resolution.support_resolution_engine import (
        SupportResolutionEngine,
        SupportResolutionResult,
    )
    from app.retrieval.case_retriever import CaseRetriever
    from app.retrieval.evidence_ranker import RetrievedEvidenceCase
    from app.schemas.intent_routing import CustomerMessageRequest
    from app.understanding.problem_extractor import (
        CustomerProblemProfile,
        ProblemExtractor,
    )
except ModuleNotFoundError:
    from backend.app.core.config import settings  # type: ignore[no-redef]
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]
    from backend.app.resolution.evidence_validator import (  # type: ignore[no-redef]
        EvidenceValidationResult,
        ResolutionEvidenceValidator,
    )
    from backend.app.resolution.resolution_auditor import (  # type: ignore[no-redef]
        ResolutionAuditor,
    )
    from backend.app.resolution.response_generator import (  # type: ignore[no-redef]
        EvidenceGroundedResponseGenerator,
        GroundedResponseCandidate,
    )
    from backend.app.resolution.response_verifier import (  # type: ignore[no-redef]
        ResponseGroundingResult,
        ResponseGroundingVerifier,
    )
    from backend.app.resolution.support_resolution_engine import (  # type: ignore[no-redef]
        SupportResolutionEngine,
        SupportResolutionResult,
    )
    from backend.app.retrieval.case_retriever import CaseRetriever  # type: ignore[no-redef]
    from backend.app.retrieval.evidence_ranker import (  # type: ignore[no-redef]
        RetrievedEvidenceCase,
    )
    from backend.app.schemas.intent_routing import (  # type: ignore[no-redef]
        CustomerMessageRequest,
    )
    from backend.app.understanding.problem_extractor import (  # type: ignore[no-redef]
        CustomerProblemProfile,
        ProblemExtractor,
    )

try:
    from app.feedback.live_queue_manager import get_live_queue_manager
    from app.schemas.human_review import CaseSource
except ModuleNotFoundError:
    from backend.app.feedback.live_queue_manager import get_live_queue_manager  # type: ignore[no-redef]
    from backend.app.schemas.human_review import CaseSource  # type: ignore[no-redef]

logger = get_logger(__name__)

router = APIRouter(prefix="/resolution", tags=["Support Resolution"])

# Dependency Singletons
_resolution_engine: SupportResolutionEngine | None = None
_problem_extractor: ProblemExtractor | None = None
_case_retriever: CaseRetriever | None = None
_evidence_validator: ResolutionEvidenceValidator | None = None
_response_verifier: ResponseGroundingVerifier | None = None
_auditor: ResolutionAuditor | None = None


def get_resolution_engine() -> SupportResolutionEngine:
    global _resolution_engine
    if _resolution_engine is None:
        _resolution_engine = SupportResolutionEngine()
    return _resolution_engine


def get_problem_extractor() -> ProblemExtractor:
    global _problem_extractor
    if _problem_extractor is None:
        _problem_extractor = ProblemExtractor()
    return _problem_extractor


def get_case_retriever() -> CaseRetriever:
    global _case_retriever
    if _case_retriever is None:
        _case_retriever = CaseRetriever()
    return _case_retriever


def get_evidence_validator() -> ResolutionEvidenceValidator:
    global _evidence_validator
    if _evidence_validator is None:
        _evidence_validator = ResolutionEvidenceValidator()
    return _evidence_validator


def get_response_verifier() -> ResponseGroundingVerifier:
    global _response_verifier
    if _response_verifier is None:
        _response_verifier = ResponseGroundingVerifier()
    return _response_verifier


def get_auditor() -> ResolutionAuditor:
    global _auditor
    if _auditor is None:
        _auditor = ResolutionAuditor()
    return _auditor


class ResolveRequest(BaseModel):
    """Request payload for end-to-end evidence-grounded support resolution."""
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    customer_message: str = Field(..., min_length=1, description="Customer inquiry message text")
    normalized_message: Optional[str] = Field(default=None, description="Optional pre-normalized message text")
    top_k_evidence: int = Field(default=3, ge=1, le=10, description="Number of evidence cases to retrieve")
    case_id: Optional[str] = Field(default=None, description="Optional tracking/case identifier")
    log_audit: bool = Field(default=True, description="Whether to record decision into runtime audit log")
    source: str = Field(default="LIVE_SUPPORT", description="Source of inquiry: LIVE_SUPPORT, DEMO, TEST, BENCHMARK")


class RetrieveEvidenceRequest(BaseModel):
    """Request payload for historical evidence retrieval."""
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    query_text: str = Field(..., min_length=1, description="Customer query message text")
    query_intent: Optional[str] = Field(default=None, description="Target operational intent")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of cases to retrieve")


class ValidateEvidenceRequest(BaseModel):
    """Request payload to validate retrieved historical evidence."""
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    customer_message: str = Field(..., min_length=1, description="Customer message")
    primary_intent: str = Field(..., description="Target operational intent")
    top_k_evidence: int = Field(default=3, ge=1, le=10, description="Number of cases to evaluate")


class VerifyResponseRequest(BaseModel):
    """Request payload to verify candidate response grounding."""
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    customer_message: str = Field(..., min_length=1, description="Customer inquiry message text")
    primary_intent: str = Field(..., description="Target operational intent")
    candidate_response: str = Field(..., min_length=1, description="Generated brand response text")
    referenced_case_ids: list[str] = Field(default_factory=list, description="IDs of historical cases supporting response")


@router.post(
    "/resolve",
    response_model=SupportResolutionResult,
    summary="Evidence-Grounded Support Resolution (Phase 8)",
    description=(
        "Executes the full Phase 8 resolution pipeline: problem understanding, top-k intent candidates, "
        "causal disambiguation, ambiguity analysis, operational evidence retrieval, evidence validation, "
        "response synthesis, response grounding verification, and safe auto-handling or rich human escalation packaging."
    ),
)
async def resolve_inquiry(
    request: ResolveRequest,
    engine: SupportResolutionEngine = Depends(get_resolution_engine),
) -> SupportResolutionResult:
    """Process customer message through evidence-grounded resolution pipeline."""
    try:
        result = engine.process_message(
            customer_message=request.customer_message,
            normalized_message=request.normalized_message or "",
            top_k_evidence=request.top_k_evidence,
            case_id=request.case_id,
            log_audit=request.log_audit,
        )

        # Live Human Review Queue Integration:
        # Strictly admit ONLY live queries that require human review into the active queue
        decision_str = str(result.routing_decision.value if hasattr(result.routing_decision, "value") else result.routing_decision)
        if request.source == "LIVE_SUPPORT" and "ESCALATE_TO_HUMAN" in decision_str:
            try:
                queue_mgr = get_live_queue_manager()

                reason_text = ""
                if result.escalation_package and result.escalation_package.why_not_auto_handled:
                    reason_text = "; ".join(result.escalation_package.why_not_auto_handled)
                elif result.explanation:
                    reason_text = result.explanation
                else:
                    reason_text = "Human review required by SupportGraph safety gate."

                hist_cases = [c.model_dump() for c in result.evidence_cases] if result.evidence_cases else []
                p_dict = (
                    result.escalation_package.problem_profile.model_dump()
                    if (result.escalation_package and result.escalation_package.problem_profile)
                    else {}
                )

                queue_mgr.create_case(
                    customer_query=request.customer_message,
                    escalation_reason=reason_text,
                    source=CaseSource.LIVE_SUPPORT,
                    outcome=result.outcome.value if hasattr(result.outcome, "value") else str(result.outcome) if result.outcome else None,
                    decision_explanation=result.decision_explanation.model_dump() if result.decision_explanation else None,
                    ai_suggested_response=result.resolution_strategy or result.grounded_response or "Review by specialist recommended.",
                    problem_understanding=p_dict,
                    intent=result.primary_intent,
                    problem_family=p_dict.get("problem_family") or result.primary_intent,
                    related_historical_cases=hist_cases,
                    case_id=request.case_id,
                )
            except Exception as q_exc:
                logger.warning("Failed to record live review case: %s", q_exc)

        return result
    except Exception as exc:
        logger.error("Error during support resolution: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Resolution processing failed: {exc}",
        )


@router.post(
    "/understand",
    response_model=CustomerProblemProfile,
    summary="Extract Structured Problem Profile",
    description="Analyzes customer message to extract structured device, symptom, and causality features.",
)
async def understand_problem(
    request: CustomerMessageRequest,
    extractor: ProblemExtractor = Depends(get_problem_extractor),
) -> CustomerProblemProfile:
    """Extract structured problem profile from message."""
    try:
        return extractor.extract(message=request.customer_message)
    except Exception as exc:
        logger.error("Error during problem extraction: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Problem extraction failed: {exc}",
        )


@router.post(
    "/retrieve-evidence",
    response_model=List[RetrievedEvidenceCase],
    summary="Retrieve Historical Evidence Cases",
    description="Retrieves historically similar AppleSupport cases categorized into operational match tiers.",
)
async def retrieve_evidence(
    request: RetrieveEvidenceRequest,
    retriever: CaseRetriever = Depends(get_case_retriever),
) -> list[RetrievedEvidenceCase]:
    """Retrieve operational evidence cases for a customer inquiry."""
    try:
        return retriever.retrieve(
            query_text=request.query_text,
            query_intent=request.query_intent or "",
            top_k=request.top_k,
        )
    except Exception as exc:
        logger.error("Error during evidence retrieval: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evidence retrieval failed: {exc}",
        )


@router.post(
    "/validate-evidence",
    response_model=EvidenceValidationResult,
    summary="Validate Operational Evidence (Phase 8)",
    description="Evaluates whether retrieved historical support cases provide sufficiently strong operational evidence for resolution.",
)
async def validate_evidence_endpoint(
    request: ValidateEvidenceRequest,
    extractor: ProblemExtractor = Depends(get_problem_extractor),
    retriever: CaseRetriever = Depends(get_case_retriever),
    validator: ResolutionEvidenceValidator = Depends(get_evidence_validator),
) -> EvidenceValidationResult:
    """Validate operational evidence strength and consistency."""
    try:
        profile = extractor.extract(request.customer_message)
        evidence = retriever.retrieve(
            query_text=request.customer_message,
            query_intent=request.primary_intent,
            query_profile=profile,
            top_k=request.top_k_evidence,
        )
        return validator.evaluate_evidence(
            profile=profile,
            primary_intent=request.primary_intent,
            evidence_cases=evidence,
        )
    except Exception as exc:
        logger.error("Error validating evidence: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evidence validation failed: {exc}",
        )


@router.post(
    "/verify-response",
    response_model=ResponseGroundingResult,
    summary="Verify Response Grounding (Phase 8)",
    description="Performs safety and operational evidence grounding checks on a candidate brand troubleshooting response.",
)
async def verify_response_endpoint(
    request: VerifyResponseRequest,
    extractor: ProblemExtractor = Depends(get_problem_extractor),
    retriever: CaseRetriever = Depends(get_case_retriever),
    verifier: ResponseGroundingVerifier = Depends(get_response_verifier),
) -> ResponseGroundingResult:
    """Verify that a response is grounded in evidence and symptom-aligned."""
    try:
        profile = extractor.extract(request.customer_message)
        evidence = retriever.retrieve(
            query_text=request.customer_message,
            query_intent=request.primary_intent,
            query_profile=profile,
            top_k=3,
        )
        candidate = GroundedResponseCandidate(
            response_text=request.candidate_response,
            grounded_intent=request.primary_intent,
            referenced_case_ids=request.referenced_case_ids,
            contains_dm_link=("https://apple.co/dm" in request.candidate_response.lower() or "dm" in request.candidate_response.lower()),
        )
        return verifier.verify_response(
            customer_message=request.customer_message,
            profile=profile,
            primary_intent=request.primary_intent,
            candidate_response=candidate,
            evidence_cases=evidence,
        )
    except Exception as exc:
        logger.error("Error verifying response: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Response verification failed: {exc}",
        )


@router.get(
    "/audit/stats",
    summary="Get Runtime Resolution Audit Statistics",
    description="Retrieves aggregate metrics from the append-only runtime resolution audit log.",
)
async def get_audit_stats(
    auditor: ResolutionAuditor = Depends(get_auditor),
) -> dict[str, Any]:
    """Return runtime audit statistics."""
    try:
        return auditor.get_audit_stats()
    except Exception as exc:
        logger.error("Error fetching audit stats: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch audit stats: {exc}",
        )
