"""
SupportGraph AI — Runtime HITL Escalation Manager (Phase 6)

Manages runtime escalation cases and human reviewer decisions:
- Appends human decisions to an isolated, append-only review log:
  `data/runtime/runtime_escalation_reviews.csv`
- Records original AI predictions, uncertainty metrics, routing reasons,
  and explicit human reviewer selections.
- Strictly isolated from dataset QA workflows (Phase 5.x) and protected golden datasets.
"""
from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    from app.core.config import settings
    from app.core.logging import get_logger
    from app.schemas.intent_routing import (
        HumanReviewActionRequest,
        HumanReviewActionType,
        HumanReviewRecord,
        IntentAnalysis,
        RoutingDecision,
    )
except ModuleNotFoundError:
    from backend.app.core.config import settings  # type: ignore[no-redef]
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]
    from backend.app.schemas.intent_routing import (  # type: ignore[no-redef]
        HumanReviewActionRequest,
        HumanReviewActionType,
        HumanReviewRecord,
        IntentAnalysis,
        RoutingDecision,
    )

logger = get_logger(__name__)

CSV_HEADERS = [
    "case_id",
    "timestamp",
    "customer_message",
    "top_1_intent",
    "top_1_confidence",
    "top_2_intent",
    "top_2_confidence",
    "confidence_margin",
    "routing_decision",
    "routing_reason",
    "action_taken",
    "human_final_intent",
    "is_ai_accepted",
    "reviewer_id",
    "notes",
]


class RuntimeEscalationManager:
    """
    Manages logging and audit history for runtime human-in-the-loop escalations.
    """

    def __init__(self, log_path: Optional[Path | str] = None) -> None:
        """
        Initialize the escalation manager with a specified log file path.
        """
        if log_path is not None:
            self.log_path = Path(log_path)
        else:
            self.log_path = settings.runtime_escalations_log_path

        self._ensure_log_initialized()

    def _ensure_log_initialized(self) -> None:
        """Create runtime data directory and initialize CSV with headers if not present."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists() or self.log_path.stat().st_size == 0:
            with open(self.log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(CSV_HEADERS)
            logger.info("Initialized runtime escalation log: %s", self.log_path)

    def record_human_decision(
        self,
        request: HumanReviewActionRequest,
        analysis: Optional[IntentAnalysis] = None,
        routing: Optional[RoutingDecision] = None,
    ) -> HumanReviewRecord:
        """
        Record a human review decision to the append-only audit log.

        Args:
            request: The human review action payload.
            analysis: Optional IntentAnalysis object with Top-K predictions.
            routing: Optional RoutingDecision with routing explanation.

        Returns:
            HumanReviewRecord dataclass/Pydantic model representing the recorded row.
        """
        now_iso = datetime.now(timezone.utc).isoformat()

        # Extract predictions
        top_1_intent = ""
        top_1_conf = 0.0
        top_2_intent = ""
        top_2_conf = 0.0
        margin = request.confidence_margin or 0.0

        if analysis is not None and analysis.top_predictions:
            top_1_intent = analysis.top_predictions[0].intent
            top_1_conf = analysis.top_predictions[0].confidence
            if len(analysis.top_predictions) > 1:
                top_2_intent = analysis.top_predictions[1].intent
                top_2_conf = analysis.top_predictions[1].confidence
            margin = analysis.confidence_margin
        elif request.top_predictions:
            top_1_intent = request.top_predictions[0].intent
            top_1_conf = request.top_predictions[0].confidence
            if len(request.top_predictions) > 1:
                top_2_intent = request.top_predictions[1].intent
                top_2_conf = request.top_predictions[1].confidence
            if request.confidence_margin is not None:
                margin = request.confidence_margin
            elif len(request.top_predictions) > 1:
                margin = round(max(0.0, top_1_conf - top_2_conf), 4)

        # Determine human final intent and acceptance
        human_final_intent = ""
        is_accepted = False

        if request.action == HumanReviewActionType.ACCEPT_TOP_1:
            human_final_intent = top_1_intent
            is_accepted = True
        elif request.action == HumanReviewActionType.SELECT_TOP_2:
            human_final_intent = request.selected_intent or top_2_intent
            is_accepted = False
        elif request.action == HumanReviewActionType.OVERRIDE_INTENT:
            if not request.selected_intent:
                raise ValueError("OVERRIDE_INTENT action requires a non-empty 'selected_intent'.")
            human_final_intent = request.selected_intent
            is_accepted = (human_final_intent == top_1_intent)
        elif request.action == HumanReviewActionType.MARK_UNCLEAR:
            human_final_intent = "unclear_needs_review"
            is_accepted = (top_1_intent == "unclear_needs_review")

        routing_decision_str = routing.decision.value if routing else "ESCALATE_TO_HUMAN"
        routing_reason_str = routing.reason if routing else (request.routing_reason or "Escalated for human review.")

        record = HumanReviewRecord(
            case_id=request.case_id,
            timestamp=now_iso,
            customer_message=request.customer_message,
            top_1_intent=top_1_intent,
            top_1_confidence=top_1_conf,
            top_2_intent=top_2_intent,
            top_2_confidence=top_2_conf,
            confidence_margin=margin,
            routing_decision=routing_decision_str,
            routing_reason=routing_reason_str,
            action_taken=request.action.value,
            human_final_intent=human_final_intent,
            is_ai_accepted=is_accepted,
            reviewer_id=request.reviewer_id,
            notes=request.reviewer_notes or "",
        )

        # Append to CSV
        self._ensure_log_initialized()
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                record.case_id,
                record.timestamp,
                record.customer_message,
                record.top_1_intent,
                record.top_1_confidence,
                record.top_2_intent,
                record.top_2_confidence,
                record.confidence_margin,
                record.routing_decision,
                record.routing_reason,
                record.action_taken,
                record.human_final_intent,
                record.is_ai_accepted,
                record.reviewer_id,
                record.notes,
            ])

        logger.info(
            "Recorded human review decision for case %s: action=%s, final_intent=%s",
            record.case_id,
            record.action_taken,
            record.human_final_intent,
        )
        return record

    def load_review_history(self) -> pd.DataFrame:
        """Load all recorded runtime human decisions as a DataFrame."""
        self._ensure_log_initialized()
        try:
            return pd.read_csv(self.log_path, dtype=str)
        except Exception as exc:
            logger.warning("Failed to read review history: %s", exc)
            return pd.DataFrame(columns=CSV_HEADERS)

    def get_summary_stats(self) -> dict[str, Any]:
        """Compute aggregated statistics over runtime human reviews."""
        df = self.load_review_history()
        total = len(df)
        if total == 0:
            return {
                "total_reviews": 0,
                "ai_accepted_count": 0,
                "ai_accepted_rate": 0.0,
                "top_2_selected_count": 0,
                "overridden_count": 0,
                "unclear_count": 0,
            }

        accepted = int((df["is_ai_accepted"].astype(str).str.lower() == "true").sum())
        actions = df["action_taken"].value_counts().to_dict()

        return {
            "total_reviews": total,
            "ai_accepted_count": accepted,
            "ai_accepted_rate": round(accepted / total, 4),
            "top_2_selected_count": int(actions.get("SELECT_TOP_2", 0)),
            "overridden_count": int(actions.get("OVERRIDE_INTENT", 0)),
            "unclear_count": int(actions.get("MARK_UNCLEAR", 0)),
        }
