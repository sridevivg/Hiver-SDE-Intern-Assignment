"""
SupportGraph AI — Feedback & Promotion Auditor (Phase 13).

Maintains an append-only audit trail of all Human-in-the-Loop transitions,
validation decisions, review judgments, and evidence promotions.
File path: data/runtime/feedback_audit.jsonl
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from app.core.logging import get_logger
    from app.feedback.schemas import ResolutionLifecycleState
except ModuleNotFoundError:
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]
    from backend.app.feedback.schemas import (  # type: ignore[no-redef]
        ResolutionLifecycleState,
    )

logger = get_logger(__name__)

DEFAULT_AUDIT_PATH = Path("data/runtime/feedback_audit.jsonl")


class FeedbackAuditor:
    """
    Append-only auditor recording every state transition and validation decision.
    """

    def __init__(self, audit_path: Optional[Path | str] = None) -> None:
        self.audit_path = Path(audit_path or DEFAULT_AUDIT_PATH)
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)

    def log_event(
        self,
        event_type: str,
        resolution_id: str,
        previous_state: Optional[ResolutionLifecycleState | str],
        new_state: ResolutionLifecycleState | str,
        actor_id: str,
        reason: str = "",
        evidence_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Append an audit record to the log file."""
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "resolution_id": resolution_id,
            "evidence_id": evidence_id,
            "previous_state": str(previous_state.value if hasattr(previous_state, "value") else previous_state) if previous_state else None,
            "new_state": str(new_state.value if hasattr(new_state, "value") else new_state),
            "actor_id": actor_id,
            "reason": reason,
            "details": details or {},
        }

        try:
            with open(self.audit_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as exc:
            logger.error("Failed to write feedback audit record: %s", exc)

        return record

    def read_events(self, resolution_id: Optional[str] = None) -> list[Dict[str, Any]]:
        """Read all audit events from disk, optionally filtered by resolution_id."""
        if not self.audit_path.exists():
            return []
        events = []
        try:
            with open(self.audit_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        ev = json.loads(line.strip())
                        if resolution_id and ev.get("resolution_id") != resolution_id:
                            continue
                        events.append(ev)
        except Exception as exc:
            logger.error("Failed to read audit events: %s", exc)
        return events
