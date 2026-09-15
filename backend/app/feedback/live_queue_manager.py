"""
SupportGraph AI — Live Human Review Queue Manager.

Manages the active live human review queue:
- Persists cases to data/runtime/live_human_review_cases.jsonl.
- Strictly admits only cases with source == LIVE_SUPPORT.
- Active queue returns only cases with status in [NEW, UNDER_REVIEW], sorted newest first.
- Completed cases (APPROVED, EDITED, ESCALATED, RESOLVED) leave the active queue while
  remaining auditable.
- Strictly isolated from benchmark runs, demo scripts, synthetic records, and test fixtures.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

try:
    from app.core.logging import get_logger
    from app.schemas.human_review import (
        CaseSource,
        LiveReviewCase,
        ReviewCaseStatus,
    )
except ModuleNotFoundError:
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]
    from backend.app.schemas.human_review import (  # type: ignore[no-redef]
        CaseSource,
        LiveReviewCase,
        ReviewCaseStatus,
    )

logger = get_logger(__name__)

DEFAULT_LIVE_QUEUE_PATH = Path("data/runtime/live_human_review_cases.jsonl")


class LiveQueueManager:
    """
    Thread-safe manager for live human review cases.
    """

    def __init__(self, storage_path: Optional[Path] = None) -> None:
        self.storage_path = storage_path or DEFAULT_LIVE_QUEUE_PATH
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._cases: Dict[str, LiveReviewCase] = {}
        self._load()

    def _load(self) -> None:
        """Load stored cases from disk into memory."""
        if not self.storage_path.exists():
            return

        with self._lock:
            self._cases.clear()
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                data = json.loads(line)
                                # Structural isolation: Only load cases marked as LIVE_SUPPORT
                                if data.get("source") == CaseSource.LIVE_SUPPORT.value or data.get("source") == "LIVE_SUPPORT":
                                    case = LiveReviewCase(**data)
                                    self._cases[case.case_id] = case
                            except Exception as exc:
                                logger.warning("Skipping corrupted live review case record: %s", exc)
            except Exception as exc:
                logger.error("Failed to load live review queue from %s: %s", self.storage_path, exc)

    def _save_all(self) -> None:
        """Atomically persist all cases to disk."""
        temp_path = self.storage_path.with_suffix(".tmp")
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                for case in self._cases.values():
                    f.write(json.dumps(case.model_dump(), default=str) + "\n")
            os.replace(temp_path, self.storage_path)
        except Exception as exc:
            logger.error("Failed to write live review queue to %s: %s", self.storage_path, exc)
            if temp_path.exists():
                temp_path.unlink()

    def create_case(
        self,
        customer_query: str = "",
        escalation_reason: str = "",
        source: CaseSource = CaseSource.LIVE_SUPPORT,
        conversation_id: Optional[str] = None,
        priority: str = "NORMAL",
        decision: str = "HUMAN_REVIEW_REQUIRED",
        outcome: Optional[str] = None,
        decision_explanation: Optional[Dict[str, Any]] = None,
        ai_suggested_response: Optional[str] = None,
        problem_understanding: Optional[Dict[str, Any]] = None,
        intent: Optional[str] = None,
        problem_family: Optional[str] = None,
        evidence_summary: Optional[str] = None,
        related_historical_cases: Optional[List[Dict[str, Any]]] = None,
        case_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[LiveReviewCase]:
        """
        Create and persist a new live review case.
        Strict isolation: Only LIVE_SUPPORT cases enter the queue.
        """
        # Enforce structural isolation: Reject test, demo, evaluation, or benchmark sources
        if source != CaseSource.LIVE_SUPPORT and str(source) != "LIVE_SUPPORT" and getattr(source, "value", None) != "LIVE_SUPPORT":
            logger.debug("Rejecting non-live case with source: %s", source)
            return None

        # Clean customer message preview
        clean_query = customer_query.strip()
        if not clean_query:
            return None

        # Automatic priority assignment if thermal / safety hazard
        if "hazard" in escalation_reason.lower() or "thermal" in escalation_reason.lower() or "smoking" in clean_query.lower() or "burning" in clean_query.lower():
            priority = "URGENT"

        case = LiveReviewCase(
            case_id=case_id or f"live_rev_{datetime.now(timezone.utc).strftime('%y%m%d%H%M%S')}_{os.urandom(2).hex()}",
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
            source=CaseSource.LIVE_SUPPORT,
            status=ReviewCaseStatus.NEW,
            customer_query=clean_query,
            conversation_id=conversation_id,
            priority=priority,
            decision=decision,
            outcome=outcome,
            escalation_reason=escalation_reason,
            decision_explanation=decision_explanation,
            ai_suggested_response=ai_suggested_response,
            problem_understanding=problem_understanding,
            intent=intent,
            problem_family=problem_family,
            evidence_summary=evidence_summary,
            related_historical_cases=related_historical_cases or [],
        )

        with self._lock:
            self._cases[case.case_id] = case
            self._save_all()

        logger.info("Created LIVE_SUPPORT review case %s (priority=%s)", case.case_id, case.priority)
        return case

    def get_active_queue(self) -> List[LiveReviewCase]:
        """
        Return all active LIVE_SUPPORT review cases (status in [NEW, UNDER_REVIEW]).
        Sorted newest first by created_at.
        """
        with self._lock:
            active_statuses = {ReviewCaseStatus.NEW.value, ReviewCaseStatus.UNDER_REVIEW.value, "NEW", "UNDER_REVIEW"}
            active = [
                case
                for case in self._cases.values()
                if (case.source == CaseSource.LIVE_SUPPORT.value or case.source == "LIVE_SUPPORT")
                and (case.status in active_statuses)
            ]
            # Sort newest first
            return sorted(active, key=lambda c: c.created_at, reverse=True)

    def get_active_cases(self) -> List[LiveReviewCase]:
        """Alias for get_active_queue()."""
        return self.get_active_queue()

    def get_case(self, case_id: str) -> Optional[LiveReviewCase]:
        """Retrieve a single review case by ID."""
        with self._lock:
            return self._cases.get(case_id)

    def approve_case(
        self,
        case_id: str,
        reviewer_id: str = "specialist_01",
        notes: Optional[str] = None,
    ) -> Optional[LiveReviewCase]:
        """
        Approve case response and transition status to APPROVED.
        Leaves the active queue.
        """
        with self._lock:
            case = self._cases.get(case_id)
            if not case:
                return None

            now = datetime.now(timezone.utc).isoformat()
            case.status = ReviewCaseStatus.APPROVED.value
            case.reviewer_id = reviewer_id
            case.reviewer_notes = notes
            case.updated_at = now
            case.completed_at = now
            self._save_all()
            logger.info("Live review case %s APPROVED by %s", case_id, reviewer_id)
            return case

    def edit_case(
        self,
        case_id: str,
        edited_response: str,
        reviewer_id: str = "specialist_01",
        notes: Optional[str] = None,
    ) -> Optional[LiveReviewCase]:
        """
        Specialist edits response text and transitions status to EDITED.
        Leaves the active queue.
        """
        with self._lock:
            case = self._cases.get(case_id)
            if not case:
                return None

            now = datetime.now(timezone.utc).isoformat()
            case.status = ReviewCaseStatus.EDITED.value
            case.edited_response = edited_response
            case.reviewer_id = reviewer_id
            case.reviewer_notes = notes
            case.updated_at = now
            case.completed_at = now
            self._save_all()
            logger.info("Live review case %s EDITED by %s", case_id, reviewer_id)
            return case

    def escalate_case(
        self,
        case_id: str,
        tier: str = "TIER_2_TECHNICAL",
        reviewer_id: str = "specialist_01",
        notes: Optional[str] = None,
    ) -> Optional[LiveReviewCase]:
        """
        Escalate to higher tier specialist and transition status to ESCALATED.
        Leaves the active queue.
        """
        with self._lock:
            case = self._cases.get(case_id)
            if not case:
                return None

            now = datetime.now(timezone.utc).isoformat()
            case.status = ReviewCaseStatus.ESCALATED.value
            case.priority = "URGENT"
            case.reviewer_id = reviewer_id
            case.reviewer_notes = f"Escalated to {tier}. {notes or ''}".strip()
            case.updated_at = now
            case.completed_at = now
            self._save_all()
            logger.info("Live review case %s ESCALATED to %s by %s", case_id, tier, reviewer_id)
            return case

    def clear_for_tests(self) -> None:
        """Clear cases (used strictly during isolated unit testing)."""
        with self._lock:
            self._cases.clear()
            if self.storage_path.exists():
                self.storage_path.unlink()


# Singleton Instance
_live_queue_manager: Optional[LiveQueueManager] = None


def get_live_queue_manager() -> LiveQueueManager:
    """Retrieve or initialize singleton LiveQueueManager."""
    global _live_queue_manager
    if _live_queue_manager is None:
        _live_queue_manager = LiveQueueManager()
    return _live_queue_manager
