"""
SupportGraph AI — Intent Classification & Routing API Endpoints (Phase 6)

Provides FastAPI endpoints:
- POST /api/v1/intent/classify-and-route: End-to-end Top-K classification, uncertainty analysis, and routing decision
- POST /api/v1/intent/classify: Top-K intent prediction with confidences
- POST /api/v1/intent/route: Evaluate candidate predictions against deterministic routing rules
- POST /api/v1/intent/review: Submit and log human review decision for escalated case
- GET  /api/v1/intent/config: Inspect active routing threshold parameters
- GET  /api/v1/intent/escalations/stats: Aggregate metrics on runtime HITL review queue
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status

try:
    from app.core.config import settings
    from app.core.logging import get_logger
    from app.intent.classifier import TopKIntentClassifier
    from app.intent.escalation import RuntimeEscalationManager
    from app.intent.router import IntentRouter
    from app.schemas.intent_routing import (
        CustomerMessageRequest,
        HumanReviewActionRequest,
        HumanReviewRecord,
        IntentAnalysis,
        IntentRoutingResponse,
        RoutingDecision,
        RoutingThresholdConfig,
    )
except ModuleNotFoundError:
    from backend.app.core.config import settings  # type: ignore[no-redef]
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]
    from backend.app.intent.classifier import TopKIntentClassifier  # type: ignore[no-redef]
    from backend.app.intent.escalation import RuntimeEscalationManager  # type: ignore[no-redef]
    from backend.app.intent.router import IntentRouter  # type: ignore[no-redef]
    from backend.app.schemas.intent_routing import (  # type: ignore[no-redef]
        CustomerMessageRequest,
        HumanReviewActionRequest,
        HumanReviewRecord,
        IntentAnalysis,
        IntentRoutingResponse,
        RoutingDecision,
        RoutingThresholdConfig,
    )

logger = get_logger(__name__)

router = APIRouter(prefix="/intent", tags=["Intent & Routing"])

# Singletons / Dependency Helpers
_classifier: TopKIntentClassifier | None = None
_router_engine: IntentRouter | None = None
_escalation_mgr: RuntimeEscalationManager | None = None


def get_classifier() -> TopKIntentClassifier:
    global _classifier
    if _classifier is None:
        _classifier = TopKIntentClassifier()
    return _classifier


def get_router_engine() -> IntentRouter:
    global _router_engine
    if _router_engine is None:
        _router_engine = IntentRouter()
    return _router_engine


def get_escalation_manager() -> RuntimeEscalationManager:
    global _escalation_mgr
    if _escalation_mgr is None:
        _escalation_mgr = RuntimeEscalationManager()
    return _escalation_mgr


@router.post(
    "/classify-and-route",
    response_model=IntentRoutingResponse,
    summary="Classify Intent & Make Routing Decision",
    description=(
        "Processes an incoming customer support message, predicts Top-K candidate intents "
        "with calibrated confidences, computes uncertainty metrics, and determines whether "
        "the case is AUTO_HANDLE or ESCALATE_TO_HUMAN."
    ),
)
async def classify_and_route(
    request: CustomerMessageRequest,
    classifier: TopKIntentClassifier = Depends(get_classifier),
    router_engine: IntentRouter = Depends(get_router_engine),
) -> IntentRoutingResponse:
    """Classify message intent and compute uncertainty-aware routing decision."""
    try:
        analysis = classifier.classify(
            customer_message=request.customer_message,
            normalized_message=request.normalized_message or "",
        )
        decision = router_engine.route(analysis)
        return IntentRoutingResponse(
            customer_message=request.customer_message,
            intent_analysis=analysis,
            routing=decision,
            model_name=analysis.model_name,
        )
    except Exception as exc:
        logger.error("Error during classify_and_route: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Classification and routing failed: {exc}",
        )


@router.post(
    "/classify",
    response_model=IntentAnalysis,
    summary="Top-K Intent Prediction Only",
    description="Predicts Top-K candidate intents and returns uncertainty metrics.",
)
async def classify(
    request: CustomerMessageRequest,
    classifier: TopKIntentClassifier = Depends(get_classifier),
) -> IntentAnalysis:
    """Predict candidate intents and calculate uncertainty signals."""
    try:
        return classifier.classify(
            customer_message=request.customer_message,
            normalized_message=request.normalized_message or "",
        )
    except Exception as exc:
        logger.error("Error during intent classification: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Intent classification failed: {exc}",
        )


@router.post(
    "/route",
    response_model=RoutingDecision,
    summary="Evaluate Routing Rules on Intent Analysis",
    description="Applies deterministic uncertainty routing rules to a provided IntentAnalysis payload.",
)
async def route_decision(
    analysis: IntentAnalysis,
    router_engine: IntentRouter = Depends(get_router_engine),
) -> RoutingDecision:
    """Evaluate IntentAnalysis against threshold rules."""
    return router_engine.route(analysis)


@router.post(
    "/review",
    response_model=HumanReviewRecord,
    summary="Record Human Review Decision for Escalated Case",
    description="Logs a human reviewer's final intent selection to the append-only audit trail.",
)
async def submit_human_review(
    request: HumanReviewActionRequest,
    escalation_mgr: RuntimeEscalationManager = Depends(get_escalation_manager),
) -> HumanReviewRecord:
    """Record human review action to append-only runtime CSV log."""
    try:
        return escalation_mgr.record_human_decision(request)
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )
    except Exception as exc:
        logger.error("Failed to record human review decision: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record review decision: {exc}",
        )


@router.get(
    "/config",
    response_model=RoutingThresholdConfig,
    summary="Get Active Routing Thresholds",
    description="Returns current confidence thresholds, margin parameters, and uncertainty settings.",
)
async def get_routing_config() -> RoutingThresholdConfig:
    """Get active threshold configuration."""
    return RoutingThresholdConfig(
        auto_handle_confidence_threshold=settings.auto_handle_confidence_threshold,
        min_confidence_margin=settings.min_confidence_margin,
        max_uncertainty_entropy=settings.max_uncertainty_entropy,
        top_k_predictions_count=settings.top_k_predictions_count,
    )


@router.get(
    "/escalations/stats",
    summary="Get Runtime Escalation Statistics",
    description="Returns summary statistics on human reviewer activity in the runtime escalation queue.",
)
async def get_escalation_stats(
    escalation_mgr: RuntimeEscalationManager = Depends(get_escalation_manager),
) -> dict[str, Any]:
    """Retrieve runtime review queue statistics."""
    return escalation_mgr.get_summary_stats()
