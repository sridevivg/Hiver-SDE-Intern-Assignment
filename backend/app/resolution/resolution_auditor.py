"""
SupportGraph AI — Runtime Resolution Auditor (Phase 8)

Maintains an isolated, append-only runtime audit log for all evidence-grounded
support resolution decisions in `data/runtime/evidence_resolution_audit.csv`.

CRITICAL SCIENTIFIC RULE:
- This auditor writes EXCLUSIVELY to runtime audit storage.
- It NEVER modifies or accesses `data/golden/golden_set_human_review.csv`.
"""
from __future__ import annotations

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

try:
    from app.core.logging import get_logger
except ModuleNotFoundError:
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]

logger = get_logger(__name__)

DEFAULT_RUNTIME_AUDIT_CSV = Path("data/runtime/evidence_resolution_audit.csv")

AUDIT_CSV_HEADERS = [
    "timestamp",
    "case_id",
    "customer_message",
    "device",
    "primary_symptom",
    "possible_cause",
    "primary_intent",
    "top_confidence",
    "candidate_intents",
    "evidence_verdict",
    "direct_matches_count",
    "routing_decision",
    "retrieved_case_ids",
    "match_tiers",
    "response_grounding_status",
    "grounding_score",
    "human_escalation_reason",
]


class ResolutionAuditRecord(BaseModel):
    """Structured audit entry for a single runtime resolution adjudication."""
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    case_id: str
    customer_message: str
    device: Optional[str] = None
    primary_symptom: str = "general_unspecified"
    possible_cause: Optional[str] = None
    primary_intent: str
    top_confidence: float
    candidate_intents: str = ""
    evidence_verdict: str = "STRONG_EVIDENCE"
    direct_matches_count: int = 0
    routing_decision: str = "AUTO_HANDLE"
    retrieved_case_ids: str = ""
    match_tiers: str = ""
    response_grounding_status: str = "PASS"
    grounding_score: float = 1.0
    human_escalation_reason: str = ""

    def to_csv_row(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "case_id": self.case_id,
            "customer_message": self.customer_message,
            "device": self.device or "",
            "primary_symptom": self.primary_symptom,
            "possible_cause": self.possible_cause or "",
            "primary_intent": self.primary_intent,
            "top_confidence": round(self.top_confidence, 4),
            "candidate_intents": self.candidate_intents,
            "evidence_verdict": self.evidence_verdict,
            "direct_matches_count": self.direct_matches_count,
            "routing_decision": self.routing_decision,
            "retrieved_case_ids": self.retrieved_case_ids,
            "match_tiers": self.match_tiers,
            "response_grounding_status": self.response_grounding_status,
            "grounding_score": round(self.grounding_score, 4),
            "human_escalation_reason": self.human_escalation_reason,
        }


class ResolutionAuditor:
    """
    Isolated append-only auditor for runtime support resolution decisions.
    """

    def __init__(self, audit_path: Optional[Path | str] = None) -> None:
        self.audit_path = Path(audit_path or DEFAULT_RUNTIME_AUDIT_CSV)
        self._ensure_storage()

    def _ensure_storage(self) -> None:
        """Ensure runtime storage directory and CSV header exists."""
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.audit_path.exists():
            with open(self.audit_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=AUDIT_CSV_HEADERS)
                writer.writeheader()
            logger.info("Initialized runtime audit log at %s", self.audit_path)

    def log_decision(self, record: ResolutionAuditRecord) -> None:
        """Append a decision record to runtime audit storage."""
        try:
            self._ensure_storage()
            with open(self.audit_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=AUDIT_CSV_HEADERS)
                writer.writerow(record.to_csv_row())
            logger.debug("Logged resolution decision for case %s (%s)", record.case_id, record.routing_decision)
        except Exception as exc:
            logger.error("Failed to append to runtime audit log %s: %s", self.audit_path, exc)

    def get_audit_stats(self) -> dict[str, Any]:
        """Compute aggregate statistics from runtime audit log."""
        if not self.audit_path.exists():
            return {
                "total_logged_decisions": 0,
                "auto_handle_count": 0,
                "escalation_count": 0,
                "evidence_verdicts": {},
                "grounding_pass_count": 0,
            }

        total = 0
        auto_handle = 0
        escalation = 0
        verdicts: dict[str, int] = {}
        grounding_pass = 0

        try:
            with open(self.audit_path, mode="r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    total += 1
                    decision = row.get("routing_decision", "")
                    if decision == "AUTO_HANDLE":
                        auto_handle += 1
                    elif decision == "ESCALATE_TO_HUMAN":
                        escalation += 1

                    v = row.get("evidence_verdict", "UNKNOWN")
                    verdicts[v] = verdicts.get(v, 0) + 1

                    if row.get("response_grounding_status") == "PASS":
                        grounding_pass += 1
        except Exception as exc:
            logger.warning("Error reading audit stats: %s", exc)

        return {
            "total_logged_decisions": total,
            "auto_handle_count": auto_handle,
            "escalation_count": escalation,
            "auto_handle_rate": round(auto_handle / total, 4) if total > 0 else 0.0,
            "escalation_rate": round(escalation / total, 4) if total > 0 else 0.0,
            "evidence_verdicts": verdicts,
            "grounding_pass_count": grounding_pass,
            "grounding_pass_rate": round(grounding_pass / total, 4) if total > 0 else 0.0,
        }
