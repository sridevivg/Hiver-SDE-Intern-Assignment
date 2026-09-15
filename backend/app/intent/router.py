"""
SupportGraph AI — Uncertainty-Aware Intent Router (Phase 6)

Implements the deterministic, explainable runtime routing engine:
- Evaluates Top-K intent predictions against calibrated uncertainty signals:
  1. Top-1 prediction confidence
  2. Top-1 vs Top-2 confidence margin (p1 - p2)
  3. Normalized Shannon entropy
  4. Unclear / ambiguous intent detection
- Emits either AUTO_HANDLE or ESCALATE_TO_HUMAN
- Formulates transparent, auditable rationales for every decision.
"""
from __future__ import annotations

import logging
from typing import Optional

try:
    from app.core.config import settings
    from app.core.logging import get_logger
    from app.schemas.intent_routing import (
        IntentAnalysis,
        RoutingDecision,
        RoutingDecisionType,
    )
except ModuleNotFoundError:
    from backend.app.core.config import settings  # type: ignore[no-redef]
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]
    from backend.app.schemas.intent_routing import (  # type: ignore[no-redef]
        IntentAnalysis,
        RoutingDecision,
        RoutingDecisionType,
    )

logger = get_logger(__name__)


class IntentRouter:
    """
    Deterministic uncertainty-aware routing engine for intent predictions.
    """

    def __init__(
        self,
        auto_handle_confidence_threshold: Optional[float] = None,
        min_confidence_margin: Optional[float] = None,
        max_uncertainty_entropy: Optional[float] = None,
    ) -> None:
        """
        Initialize the router with configurable thresholds.

        Args:
            auto_handle_confidence_threshold: Minimum Top-1 confidence required to auto-handle.
            min_confidence_margin: Minimum separation margin (p1 - p2) required to auto-handle.
            max_uncertainty_entropy: Maximum normalized entropy allowed before escalating.
        """
        self.auto_handle_confidence_threshold: float = (
            auto_handle_confidence_threshold
            if auto_handle_confidence_threshold is not None
            else settings.auto_handle_confidence_threshold
        )
        self.min_confidence_margin: float = (
            min_confidence_margin
            if min_confidence_margin is not None
            else settings.min_confidence_margin
        )
        self.max_uncertainty_entropy: float = (
            max_uncertainty_entropy
            if max_uncertainty_entropy is not None
            else settings.max_uncertainty_entropy
        )

    def route(
        self,
        analysis: IntentAnalysis,
        auto_handle_confidence_threshold: Optional[float] = None,
        min_confidence_margin: Optional[float] = None,
        max_uncertainty_entropy: Optional[float] = None,
    ) -> RoutingDecision:
        """
        Evaluate an IntentAnalysis and determine routing destination.

        Returns:
            RoutingDecision containing target (AUTO_HANDLE / ESCALATE_TO_HUMAN) and explainable reason.
        """
        conf_thresh = (
            auto_handle_confidence_threshold
            if auto_handle_confidence_threshold is not None
            else self.auto_handle_confidence_threshold
        )
        margin_thresh = (
            min_confidence_margin
            if min_confidence_margin is not None
            else self.min_confidence_margin
        )
        entropy_thresh = (
            max_uncertainty_entropy
            if max_uncertainty_entropy is not None
            else self.max_uncertainty_entropy
        )

        top_1 = analysis.top_predictions[0] if analysis.top_predictions else None
        top_2 = analysis.top_predictions[1] if len(analysis.top_predictions) > 1 else None

        if top_1 is None:
            return RoutingDecision(
                decision=RoutingDecisionType.ESCALATE_TO_HUMAN,
                reason="No candidate intent predictions were produced; escalating for human triage.",
                confidence_threshold=conf_thresh,
                margin_threshold=margin_thresh,
                entropy_threshold=entropy_thresh,
            )

        top_1_intent = top_1.intent
        top_1_conf = top_1.confidence
        top_2_intent = top_2.intent if top_2 else "none"
        top_2_conf = top_2.confidence if top_2 else 0.0
        margin = analysis.confidence_margin
        entropy = analysis.normalized_entropy

        # Rule 1: Explicitly unclear or non-support
        if top_1_intent == "unclear_needs_review":
            return RoutingDecision(
                decision=RoutingDecisionType.ESCALATE_TO_HUMAN,
                reason=(
                    "Customer message is ambiguous, fragmented, or missing critical operational "
                    "context (classified as 'unclear_needs_review'); requires human clarification."
                ),
                confidence_threshold=conf_thresh,
                margin_threshold=margin_thresh,
                entropy_threshold=entropy_thresh,
            )

        # Rule 2: Low absolute confidence on winning candidate
        if top_1_conf < conf_thresh:
            return RoutingDecision(
                decision=RoutingDecisionType.ESCALATE_TO_HUMAN,
                reason=(
                    f"Top-1 prediction confidence ({top_1_conf:.2f} for '{top_1_intent}') "
                    f"is below the minimum auto-handle threshold ({conf_thresh:.2f})."
                ),
                confidence_threshold=conf_thresh,
                margin_threshold=margin_thresh,
                entropy_threshold=entropy_thresh,
            )

        # Rule 3: Low separation margin between Top-1 and Top-2
        if top_2 is not None and margin < margin_thresh:
            return RoutingDecision(
                decision=RoutingDecisionType.ESCALATE_TO_HUMAN,
                reason=(
                    f"Ambiguous intent competition: confidence margin between Top-1 '{top_1_intent}' "
                    f"({top_1_conf:.2f}) and Top-2 '{top_2_intent}' ({top_2_conf:.2f}) is only "
                    f"{margin:.2f}, below the required minimum separation margin ({margin_thresh:.2f})."
                ),
                confidence_threshold=conf_thresh,
                margin_threshold=margin_thresh,
                entropy_threshold=entropy_thresh,
            )

        # Rule 4: High normalized entropy / probability distribution spread
        if entropy is not None and entropy > entropy_thresh:
            return RoutingDecision(
                decision=RoutingDecisionType.ESCALATE_TO_HUMAN,
                reason=(
                    f"High multi-candidate uncertainty: normalized entropy ({entropy:.2f}) "
                    f"exceeds maximum threshold ({entropy_thresh:.2f})."
                ),
                confidence_threshold=conf_thresh,
                margin_threshold=margin_thresh,
                entropy_threshold=entropy_thresh,
            )

        # Default: Safe for Auto Handling
        return RoutingDecision(
            decision=RoutingDecisionType.AUTO_HANDLE,
            reason=(
                f"High confidence ({top_1_conf:.2f} >= {conf_thresh:.2f}) with decisive "
                f"separation margin ({margin:.2f} >= {margin_thresh:.2f}) for '{top_1_intent}'."
            ),
            confidence_threshold=conf_thresh,
            margin_threshold=margin_thresh,
            entropy_threshold=entropy_thresh,
        )
