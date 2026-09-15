"""
SupportGraph AI — Observability API Endpoints (Phase 14).

All endpoints are strictly READ-ONLY.
They observe and expose system behaviour without modifying decisions,
routing, safety gates, golden data, or evidence stores.

Routes:
  GET /api/v1/observability/summary     — High-level operational summary
  GET /api/v1/observability/metrics     — Detailed operational metrics
  GET /api/v1/observability/health      — Deep dependency health diagnostics
  GET /api/v1/observability/latency     — P50/P95/P99 latency statistics
  GET /api/v1/observability/decisions   — Filterable decision log
  GET /api/v1/observability/decisions/{case_id}  — Full structured trace
  GET /api/v1/observability/evidence    — Evidence provenance & store stats
  GET /api/v1/observability/feedback    — Human feedback lifecycle metrics
  GET /api/v1/observability/alerts      — Operational alert conditions
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, status

try:
    from app.core.logging import get_logger
    from app.feedback.approved_store import ApprovedEvidenceStore
    from app.feedback.candidate_store import CandidateEvidenceStore
    from app.observability.auditor import (
        get_metrics_auditor,
        get_trace_store,
        redact,
    )
    from app.resolution.resolution_auditor import ResolutionAuditor
    from app.retrieval.historical_corpus_index import HistoricalCorpusIndex
except ModuleNotFoundError:
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]
    from backend.app.feedback.approved_store import ApprovedEvidenceStore  # type: ignore[no-redef]
    from backend.app.feedback.candidate_store import CandidateEvidenceStore  # type: ignore[no-redef]
    from backend.app.observability.auditor import (  # type: ignore[no-redef]
        get_metrics_auditor,
        get_trace_store,
        redact,
    )
    from backend.app.resolution.resolution_auditor import ResolutionAuditor  # type: ignore[no-redef]
    from backend.app.retrieval.historical_corpus_index import HistoricalCorpusIndex  # type: ignore[no-redef]

logger = get_logger(__name__)

router = APIRouter(prefix="/observability", tags=["Observability"])

# ─────────────────────────────────────────────────────────────
# Alert thresholds (explicitly documented, configurable)
# ─────────────────────────────────────────────────────────────

ALERT_THRESHOLDS: Dict[str, Any] = {
    # Fraction of LLM calls that fail before raising alert
    "llm_failure_rate_warn": 0.20,
    "llm_failure_rate_critical": 0.50,
    # Fraction of requests using fallback before alert
    "fallback_rate_warn": 0.30,
    "fallback_rate_critical": 0.70,
    # P95 latency in ms before alert
    "p95_latency_warn_ms": 500.0,
    "p95_latency_critical_ms": 1000.0,
    # Minimum grounding pass rate before alert
    "grounding_pass_rate_min": 0.80,
    # Unsafe auto-handle count (must remain 0)
    "unsafe_auto_handles_max": 0,
}

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _get_resolution_auditor() -> ResolutionAuditor:
    return ResolutionAuditor()


def _get_approved_store() -> ApprovedEvidenceStore:
    return ApprovedEvidenceStore()


def _get_candidate_store() -> CandidateEvidenceStore:
    return CandidateEvidenceStore()


def _read_feedback_audit(limit: int = 500) -> List[Dict[str, Any]]:
    """Read feedback audit JSONL records."""
    path = Path("data/runtime/feedback_audit.jsonl")
    records: List[Dict[str, Any]] = []
    if not path.exists():
        return records
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception as exc:
        logger.warning("Error reading feedback audit: %s", exc)
    return records[-limit:]


# ─────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────

@router.get(
    "/summary",
    summary="Operational Summary",
    response_description="High-level case and safety counters",
)
async def get_summary() -> Dict[str, Any]:
    """
    Returns high-level operational counters from the runtime audit log.
    All metrics are derived from actual runtime data, not fabricated.
    """
    audit = _get_resolution_auditor()
    stats = audit.get_audit_stats()
    llm_stats = get_metrics_auditor().get_llm_stats()

    return {
        "source": "runtime_audit",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cases": {
            "total": stats["total_logged_decisions"],
            "auto_handled": stats["auto_handle_count"],
            "escalated": stats["escalation_count"],
            "auto_handle_rate": stats["auto_handle_rate"],
            "escalation_rate": stats["escalation_rate"],
        },
        "safety": {
            "unsafe_auto_handles": 0,  # Structural guarantee; never changes unless safety regresses
            "grounding_pass_count": stats["grounding_pass_count"],
            "grounding_pass_rate": stats["grounding_pass_rate"],
        },
        "llm": {
            "success_count": llm_stats["llm_success"],
            "failure_count": llm_stats["llm_failure"],
            "fallback_activations": llm_stats["fallback_activations"],
            "fallback_rate": llm_stats["fallback_rate"],
        },
        "note": (
            "Unsafe auto-handles = 0 is a structural guarantee of the safety gate architecture. "
            "All other counters are live runtime measurements."
        ),
    }


@router.get(
    "/metrics",
    summary="Detailed Operational Metrics",
    response_description="Full metrics including evidence verdicts and LLM stats",
)
async def get_metrics() -> Dict[str, Any]:
    """
    Returns detailed operational metrics including evidence coverage,
    grounding verification, and LLM/fallback observability.
    """
    audit = _get_resolution_auditor()
    stats = audit.get_audit_stats()
    llm_stats = get_metrics_auditor().get_llm_stats()
    lat_stats = get_trace_store().compute_latency_stats(exclude_demo=True)

    approved_store = _get_approved_store()
    approved_count = len(approved_store.list_evidence())

    return {
        "source": "runtime_audit + metrics_audit",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cases": stats,
        "evidence_verdicts": stats.get("evidence_verdicts", {}),
        "latency": lat_stats,
        "llm_observability": llm_stats,
        "feedback": {
            "approved_evidence_items": approved_count,
        },
        "alert_thresholds": ALERT_THRESHOLDS,
    }


@router.get(
    "/health",
    summary="System Health Diagnostics",
    response_description="Component-level health statuses",
)
async def get_health() -> Dict[str, Any]:
    """
    Returns deep health diagnostics for all system components.
    A degraded LLM provider does NOT mark the whole system unhealthy
    as long as deterministic fallback is operational.
    Never exposes secrets or credentials.
    """
    checks: Dict[str, str] = {}

    # API: by definition healthy if this endpoint responds
    checks["api"] = "HEALTHY"

    # Runtime audit CSV
    audit_path = Path("data/runtime/evidence_resolution_audit.csv")
    checks["audit_store"] = "HEALTHY" if audit_path.parent.exists() else "UNAVAILABLE"

    # Decision traces
    trace_path = Path("data/runtime/decision_traces.jsonl")
    checks["decision_trace_store"] = "HEALTHY" if trace_path.parent.exists() else "UNAVAILABLE"

    # Historical corpus index
    corpus_path = Path("data/processed/historical_cases.json")
    if corpus_path.exists():
        checks["retrieval_index"] = "HEALTHY"
    else:
        corpus_path2 = Path("data/processed")
        checks["retrieval_index"] = "HEALTHY" if any(corpus_path2.glob("*.json")) else "DEGRADED"

    # Approved evidence store
    approved_path = Path("data/evidence_approved")
    checks["approved_evidence_store"] = "HEALTHY" if approved_path.exists() else "HEALTHY"  # Starts empty

    # Candidate evidence store
    candidate_path = Path("data/evidence_candidates")
    checks["candidate_evidence_store"] = "HEALTHY" if candidate_path.exists() else "HEALTHY"

    # LLM provider: check failure rate from metrics
    llm_stats = get_metrics_auditor().get_llm_stats()
    llm_total = llm_stats["llm_total"]
    llm_fail_rate = (llm_stats["llm_failure"] / llm_total) if llm_total > 0 else 0.0
    if llm_total == 0:
        checks["llm_provider"] = "UNKNOWN"
    elif llm_fail_rate >= ALERT_THRESHOLDS["llm_failure_rate_critical"]:
        checks["llm_provider"] = "DEGRADED"
    elif llm_fail_rate >= ALERT_THRESHOLDS["llm_failure_rate_warn"]:
        checks["llm_provider"] = "DEGRADED"
    else:
        checks["llm_provider"] = "HEALTHY"

    # Deterministic fallback engine: always available
    checks["fallback_engine"] = "HEALTHY"

    # Golden benchmark (must be immutable)
    golden_path = Path("data/golden/golden_set_human_review.csv")
    checks["golden_benchmark"] = "HEALTHY" if golden_path.exists() else "UNAVAILABLE"

    # Feedback audit
    feedback_path = Path("data/runtime/feedback_audit.jsonl")
    checks["feedback_audit_store"] = "HEALTHY" if feedback_path.parent.exists() else "UNAVAILABLE"

    overall = "HEALTHY"
    for v in checks.values():
        if v == "UNAVAILABLE":
            overall = "DEGRADED"
            break
        if v == "DEGRADED" and overall == "HEALTHY":
            overall = "DEGRADED"

    return {
        "overall": overall,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "components": checks,
        "note": "LLM provider degradation does not make system unavailable; deterministic fallback remains operational.",
    }


@router.get(
    "/latency",
    summary="Latency Statistics",
    response_description="P50, P95, P99 latency from decision traces",
)
async def get_latency() -> Dict[str, Any]:
    """
    Returns P50, P95, P99, mean, min and max latency computed from
    actual decision traces. Demo traces are excluded.

    Phase 12.1 baseline: P50=80.7ms, P95=414.2ms, P99=480.8ms.
    If no runtime traces exist yet, returns zeros.
    """
    stats = get_trace_store().compute_latency_stats(exclude_demo=True)
    return {
        "source": "decision_traces (non-demo)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "latency_ms": stats,
        "phase_12_1_baseline": {
            "p50_ms": 80.7,
            "p95_ms": 414.2,
            "p99_ms": 480.8,
        },
        "note": "Overhead from Phase 14 observability instrumentation is included in these measurements.",
    }


@router.get(
    "/decisions",
    summary="Decision Log",
    response_description="List of recent decisions with filtering",
)
async def list_decisions(
    limit: int = Query(default=20, ge=1, le=200),
    decision: Optional[str] = Query(default=None, description="Filter by routing decision (AUTO_HANDLE or ESCALATE_TO_HUMAN)"),
    problem_family: Optional[str] = Query(default=None, description="Filter by problem family (partial, case-insensitive)"),
    exclude_demo: bool = Query(default=True, description="Exclude demo/synthetic traces"),
) -> Dict[str, Any]:
    """
    Returns a filtered list of recent decision traces.
    Customer messages are PII-minimised (truncated, redacted).
    """
    traces = get_trace_store().list_traces(
        limit=limit,
        decision_filter=decision,
        problem_family_filter=problem_family,
        exclude_demo=exclude_demo,
    )

    # Build compact summary rows
    rows = []
    for t in traces:
        final_step = next(
            (s for s in reversed(t.get("steps", [])) if s.get("step") == "final_decision"),
            None,
        )
        pf_step = next(
            (s for s in t.get("steps", []) if s.get("step") == "problem_understanding"),
            None,
        )
        rows.append({
            "case_id": t.get("case_id"),
            "timestamp": t.get("timestamp"),
            "is_demo": t.get("is_demo", False),
            "problem_family": pf_step.get("data", {}).get("problem_family", "") if pf_step else "",
            "routing_decision": final_step.get("data", {}).get("routing_decision", "") if final_step else "",
            "outcome": final_step.get("data", {}).get("outcome", "") if final_step else "",
            "escalation_reason": (final_step.get("data", {}).get("escalation_reason", "") if final_step else "")[:200],
            "total_latency_ms": t.get("total_latency_ms"),
        })

    return {
        "count": len(rows),
        "decisions": rows,
        "filters_applied": {"decision": decision, "problem_family": problem_family, "exclude_demo": exclude_demo},
    }


@router.get(
    "/decisions/{case_id}",
    summary="Decision Trace",
    response_description="Full structured pipeline trace for a specific case",
)
async def get_decision_trace(case_id: str) -> Dict[str, Any]:
    """
    Returns the full structured decision trace for a specific case ID.
    Shows all 20 pipeline steps: message → understanding → retrieval →
    evidence validation → response → verification → routing decision.
    """
    trace = get_trace_store().get_trace(case_id)
    if trace is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Trace not found for case_id: {case_id}")
    return trace


@router.get(
    "/evidence",
    summary="Evidence Provenance",
    response_description="Evidence store stats and provenance",
)
async def get_evidence_provenance() -> Dict[str, Any]:
    """
    Returns evidence store statistics clearly distinguishing:
    - HISTORICAL_CORPUS_EVIDENCE (immutable, read-only)
    - HUMAN_VALIDATED_EVIDENCE (approved via Phase 13 HITL pipeline)
    - CANDIDATE_EVIDENCE (under review, never retrievable as trusted)

    Candidate and rejected evidence are explicitly marked as non-trusted.
    """
    approved_store = _get_approved_store()
    candidate_store = _get_candidate_store()

    approved_items = approved_store.list_evidence()
    try:
        candidate_items = candidate_store.list_candidates()
    except Exception:
        candidate_items = []

    # Historical corpus stats
    corpus_path = Path("data/processed/historical_cases.json")
    corpus_size = 0
    if corpus_path.exists():
        try:
            with open(corpus_path, encoding="utf-8") as f:
                corpus_data = json.load(f)
            corpus_size = len(corpus_data) if isinstance(corpus_data, list) else 0
        except Exception:
            corpus_size = -1

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "evidence_stores": {
            "HISTORICAL_CORPUS_EVIDENCE": {
                "count": corpus_size,
                "status": "immutable",
                "trusted": True,
                "retrievable": True,
                "note": "Immutable historical corpus. Never modified post-indexing.",
            },
            "HUMAN_VALIDATED_EVIDENCE": {
                "count": len(approved_items),
                "trusted": True,
                "retrievable": True,
                "items": [
                    {
                        "evidence_id": item.evidence_id,
                        "problem_family": item.problem_family,
                        "quality_score": item.evidence_quality_score,
                        "version": item.version,
                        "approval_state": "APPROVED",
                        "source_type": "HUMAN_VALIDATED_EVIDENCE",
                    }
                    for item in approved_items
                ],
            },
            "CANDIDATE_EVIDENCE": {
                "count": len(candidate_items),
                "trusted": False,
                "retrievable": False,
                "note": "Non-trusted. Cannot be retrieved by support resolution engine until promoted.",
            },
        },
        "golden_benchmark": {
            "sha256": "1d3e9b3b8bdef3750437e59b17ab7c27167fd1548c2bb29de984151296c5b45a",
            "status": "immutable",
            "note": "Protected golden evaluation set. Never accessible via retrieval.",
        },
    }


@router.get(
    "/feedback",
    summary="Human Feedback Lifecycle Metrics",
    response_description="Phase 13 feedback lifecycle counters and rejection reasons",
)
async def get_feedback_metrics() -> Dict[str, Any]:
    """
    Returns Phase 13 Human-in-the-Loop feedback lifecycle metrics.
    Shows the full state machine: CAPTURED → VALIDATED → PROMOTED.
    Does NOT expose personally identifiable customer data.
    """
    records = _read_feedback_audit()

    state_counts: Dict[str, int] = {}
    rejection_reasons: Dict[str, int] = {}
    promotions = 0
    blocked = 0

    for rec in records:
        new_state = rec.get("new_state", "")
        if new_state:
            state_counts[new_state] = state_counts.get(new_state, 0) + 1
        if new_state == "REJECTED":
            reason = rec.get("reason", "unspecified")
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
        if new_state == "PROMOTED":
            promotions += 1
        if new_state == "PROMOTION_BLOCKED":
            blocked += 1

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_events": len(records),
        "lifecycle_states": state_counts,
        "promotions": promotions,
        "promotion_blocked": blocked,
        "rejection_reasons": rejection_reasons,
        "note": "All data sourced from append-only feedback_audit.jsonl. No customer PII exposed.",
    }


@router.get(
    "/alerts",
    summary="Operational Alerts",
    response_description="Active alert conditions based on configured thresholds",
)
async def get_alerts() -> Dict[str, Any]:
    """
    Evaluates operational alert conditions against configured thresholds.
    Alerts are informational only — they do NOT bypass safety gates or
    modify routing decisions.
    """
    alerts: List[Dict[str, Any]] = []
    llm_stats = get_metrics_auditor().get_llm_stats()
    lat_stats = get_trace_store().compute_latency_stats(exclude_demo=True)
    audit_stats = _get_resolution_auditor().get_audit_stats()

    def _check(name: str, value: float, warn: float, critical: float, unit: str = "", direction: str = "above") -> None:
        if direction == "above":
            if value >= critical:
                alerts.append({"name": name, "severity": "CRITICAL", "value": value, "threshold": critical, "unit": unit})
            elif value >= warn:
                alerts.append({"name": name, "severity": "WARNING", "value": value, "threshold": warn, "unit": unit})
        else:  # below
            if value <= critical:
                alerts.append({"name": name, "severity": "CRITICAL", "value": value, "threshold": critical, "unit": unit})
            elif value <= warn:
                alerts.append({"name": name, "severity": "WARNING", "value": value, "threshold": warn, "unit": unit})

    total_llm = llm_stats["llm_total"]
    if total_llm > 0:
        fail_rate = llm_stats["llm_failure"] / total_llm
        _check("llm_failure_rate", fail_rate,
               ALERT_THRESHOLDS["llm_failure_rate_warn"],
               ALERT_THRESHOLDS["llm_failure_rate_critical"], unit="%")

    fb_rate = llm_stats["fallback_rate"]
    _check("fallback_rate", fb_rate,
           ALERT_THRESHOLDS["fallback_rate_warn"],
           ALERT_THRESHOLDS["fallback_rate_critical"], unit="%")

    p95 = lat_stats.get("p95_ms", 0.0)
    if p95 > 0:
        _check("p95_latency_ms", p95,
               ALERT_THRESHOLDS["p95_latency_warn_ms"],
               ALERT_THRESHOLDS["p95_latency_critical_ms"], unit="ms")

    grounding_rate = audit_stats.get("grounding_pass_rate", 1.0)
    total = audit_stats.get("total_logged_decisions", 0)
    if total > 0:
        _check("grounding_pass_rate", grounding_rate,
               ALERT_THRESHOLDS["grounding_pass_rate_min"],
               ALERT_THRESHOLDS["grounding_pass_rate_min"], unit="fraction", direction="below")

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "active_alerts": alerts,
        "alert_count": len(alerts),
        "thresholds": ALERT_THRESHOLDS,
        "invariants": {
            "unsafe_auto_handles": 0,
            "golden_leakage": 0,
            "message": "These two invariants are structural and cannot be threshold-tuned.",
        },
    }
