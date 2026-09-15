"""
SupportGraph AI — Schemas Package (Phase 6)

Exposes Pydantic data models for intent classification, uncertainty analysis,
routing decisions, and runtime human review actions.
"""
from __future__ import annotations

try:
    from app.schemas.decision_explanation import (
        DecisionExplanation,
        EndToEndOutcome,
    )
    from app.schemas.intent_routing import (
        CustomerMessageRequest,
        HumanReviewActionRequest,
        HumanReviewActionType,
        HumanReviewRecord,
        IntentAnalysis,
        IntentPrediction,
        IntentRoutingResponse,
        RoutingDecision,
        RoutingDecisionType,
        RoutingThresholdConfig,
    )
except ModuleNotFoundError:
    from backend.app.schemas.decision_explanation import (  # type: ignore[no-redef]
        DecisionExplanation,
        EndToEndOutcome,
    )
    from backend.app.schemas.intent_routing import (  # type: ignore[no-redef]
        CustomerMessageRequest,
        HumanReviewActionRequest,
        HumanReviewActionType,
        HumanReviewRecord,
        IntentAnalysis,
        IntentPrediction,
        IntentRoutingResponse,
        RoutingDecision,
        RoutingDecisionType,
        RoutingThresholdConfig,
    )

__all__ = [
    "CustomerMessageRequest",
    "DecisionExplanation",
    "EndToEndOutcome",
    "HumanReviewActionRequest",
    "HumanReviewActionType",
    "HumanReviewRecord",
    "IntentAnalysis",
    "IntentPrediction",
    "IntentRoutingResponse",
    "RoutingDecision",
    "RoutingDecisionType",
    "RoutingThresholdConfig",
]
