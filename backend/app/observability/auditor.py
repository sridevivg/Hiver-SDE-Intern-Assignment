"""
SupportGraph AI — Decision Trace Auditor (Phase 14).

Captures structured, append-only decision traces for every support case.
Each trace records the full pipeline execution: message → understanding →
retrieval → evidence validation → routing → outcome.

Storage: data/runtime/decision_traces.jsonl
Strictly read-only with respect to support decisions.
Implements PII minimisation and secret redaction.
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from app.core.logging import get_logger
except ModuleNotFoundError:
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]

logger = get_logger(__name__)

DEFAULT_TRACE_PATH = Path("data/runtime/decision_traces.jsonl")
DEFAULT_METRICS_PATH = Path("data/runtime/system_metrics_audit.jsonl")

# ─────────────────────────────────────────────────────────────
# Secret / PII redaction
# ─────────────────────────────────────────────────────────────

_REDACT_PATTERNS = [
    # Bearer tokens (JWT format with dots) — must run BEFORE the generic keyword pattern
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9\-._~+/]+=*(\.[A-Za-z0-9\-._~+/]+=*)*"), "Bearer [REDACTED]"),
    # Generic secrets/credentials
    (re.compile(r"(?i)(api[_\-]?key|token|secret|password|passwd|authorization)\s*[=:]\s*\S+"), r"\1=[REDACTED]"),
    (re.compile(r"\b(?:\d[ -]?){13,16}\b"), "[CARD_REDACTED]"),           # card numbers
    (re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"), "[EMAIL_REDACTED]"),  # email
]


def redact(text: str) -> str:
    """Apply all redaction patterns to a string."""
    if not text:
        return text
    for pattern, replacement in _REDACT_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def redact_dict(obj: Any, max_message_len: int = 500) -> Any:
    """
    Recursively redact sensitive fields from a dict/list/string.
    Also truncates raw customer messages to avoid storing excessive PII.
    """
    if isinstance(obj, dict):
        return {k: redact_dict(v, max_message_len) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact_dict(v, max_message_len) for v in obj]
    if isinstance(obj, str):
        cleaned = redact(obj)
        if len(cleaned) > max_message_len:
            cleaned = cleaned[:max_message_len] + " …[truncated]"
        return cleaned
    return obj


# ─────────────────────────────────────────────────────────────
# Decision Trace Model
# ─────────────────────────────────────────────────────────────

class DecisionTrace:
    """
    Immutable record of a single end-to-end support case execution.
    Contains 20 annotated pipeline steps.
    """
    def __init__(
        self,
        case_id: str,
        is_demo: bool = False,
    ) -> None:
        self.case_id = case_id
        self.is_demo = is_demo
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.steps: List[Dict[str, Any]] = []
        self._timers: Dict[str, float] = {}
        self._start_total = time.perf_counter()

    # ── Timing helpers ──────────────────────────────────────

    def start_step(self, name: str) -> None:
        self._timers[name] = time.perf_counter()

    def end_step(self, name: str, status: str = "ok", data: Optional[Dict[str, Any]] = None) -> float:
        elapsed_ms = 0.0
        if name in self._timers:
            elapsed_ms = (time.perf_counter() - self._timers.pop(name)) * 1000
        self.steps.append({
            "step": name,
            "status": status,
            "latency_ms": round(elapsed_ms, 2),
            "data": data or {},
        })
        return elapsed_ms

    # ── Convenience annotation helpers ──────────────────────

    def record_customer_message(self, message: str) -> None:
        self.steps.append({
            "step": "customer_message",
            "status": "received",
            "latency_ms": 0.0,
            "data": {
                "message_length": len(message),
                "preview": redact(message[:200]),
            },
        })

    def record_problem_understanding(
        self,
        device: Optional[str],
        primary_symptom: str,
        problem_family: str,
        possible_cause: Optional[str],
        latency_ms: float,
    ) -> None:
        self.steps.append({
            "step": "problem_understanding",
            "status": "ok",
            "latency_ms": round(latency_ms, 2),
            "data": {
                "device": device,
                "primary_symptom": primary_symptom,
                "problem_family": problem_family,
                "possible_cause": possible_cause,
            },
        })

    def record_intent_candidates(self, candidates: List[Dict[str, Any]], latency_ms: float) -> None:
        self.steps.append({
            "step": "intent_candidates",
            "status": "ok",
            "latency_ms": round(latency_ms, 2),
            "data": {"candidates": candidates[:5]},
        })

    def record_primary_problem(self, primary_intent: str, cause_intent: Optional[str], reason: str, latency_ms: float) -> None:
        self.steps.append({
            "step": "primary_problem_selection",
            "status": "ok",
            "latency_ms": round(latency_ms, 2),
            "data": {
                "primary_intent": primary_intent,
                "cause_intent": cause_intent,
                "reason": reason,
            },
        })

    def record_ambiguity(self, ambiguity_type: str, ambiguity_reason: str, latency_ms: float) -> None:
        self.steps.append({
            "step": "ambiguity_analysis",
            "status": "ok",
            "latency_ms": round(latency_ms, 2),
            "data": {
                "ambiguity_type": ambiguity_type,
                "reason": ambiguity_reason,
            },
        })

    def record_safety_gate(self, gate_decision: str, safety_signals: Dict[str, Any], latency_ms: float) -> None:
        self.steps.append({
            "step": "safety_gate",
            "status": "ok",
            "latency_ms": round(latency_ms, 2),
            "data": {
                "gate_decision": gate_decision,
                "signals": safety_signals,
            },
        })

    def record_evidence_retrieval(self, case_count: int, tiers: List[str], latency_ms: float) -> None:
        self.steps.append({
            "step": "evidence_retrieval",
            "status": "ok" if case_count > 0 else "no_evidence",
            "latency_ms": round(latency_ms, 2),
            "data": {
                "retrieved_count": case_count,
                "tiers": tiers,
            },
        })

    def record_evidence_validation(
        self,
        verdict: str,
        direct_matches: int,
        symptom_agreement: float,
        auto_resolution_allowed: bool,
        latency_ms: float,
    ) -> None:
        self.steps.append({
            "step": "evidence_validation",
            "status": "ok",
            "latency_ms": round(latency_ms, 2),
            "data": {
                "verdict": verdict,
                "direct_matches": direct_matches,
                "symptom_agreement": round(symptom_agreement, 3),
                "auto_resolution_allowed": auto_resolution_allowed,
            },
        })

    def record_conflict_check(self, has_conflict: bool, conflicting_pairs: List[str], latency_ms: float) -> None:
        self.steps.append({
            "step": "conflict_check",
            "status": "conflict_detected" if has_conflict else "no_conflict",
            "latency_ms": round(latency_ms, 2),
            "data": {
                "has_conflict": has_conflict,
                "conflicting_pairs": conflicting_pairs,
            },
        })

    def record_response_generation(self, latency_ms: float, fallback_used: bool = False) -> None:
        self.steps.append({
            "step": "response_generation",
            "status": "fallback" if fallback_used else "ok",
            "latency_ms": round(latency_ms, 2),
            "data": {"fallback_used": fallback_used},
        })

    def record_grounding_verification(
        self,
        status: str,
        score: float,
        explanation: str,
        latency_ms: float,
    ) -> None:
        self.steps.append({
            "step": "grounding_verification",
            "status": status,
            "latency_ms": round(latency_ms, 2),
            "data": {
                "score": round(score, 3),
                "explanation": redact(explanation[:300]),
            },
        })

    def record_final_decision(
        self,
        routing_decision: str,
        outcome: str,
        escalation_reason: str = "",
    ) -> None:
        total_ms = (time.perf_counter() - self._start_total) * 1000
        self.steps.append({
            "step": "final_decision",
            "status": routing_decision,
            "latency_ms": round(total_ms, 2),
            "data": {
                "routing_decision": routing_decision,
                "outcome": outcome,
                "escalation_reason": escalation_reason,
                "total_latency_ms": round(total_ms, 2),
            },
        })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "timestamp": self.timestamp,
            "is_demo": self.is_demo,
            "steps": self.steps,
            "total_latency_ms": round((time.perf_counter() - self._start_total) * 1000, 2),
        }


# ─────────────────────────────────────────────────────────────
# Decision Trace Store
# ─────────────────────────────────────────────────────────────

class DecisionTraceStore:
    """
    Append-only store for structured decision traces.
    Provides read-only query capability for the observability APIs.
    """

    def __init__(self, trace_path: Optional[Path | str] = None) -> None:
        self.trace_path = Path(trace_path or DEFAULT_TRACE_PATH)
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, trace: DecisionTrace) -> None:
        """Persist a completed decision trace to disk."""
        try:
            record = trace.to_dict()
            with open(self.trace_path, mode="a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as exc:
            logger.error("Failed to save decision trace for case %s: %s", trace.case_id, exc)

    def get_trace(self, case_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve the most recent trace for a given case ID."""
        if not self.trace_path.exists():
            return None
        try:
            with open(self.trace_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if record.get("case_id") == case_id:
                        return record
        except Exception as exc:
            logger.warning("Error reading trace for case %s: %s", case_id, exc)
        return None

    def list_traces(
        self,
        limit: int = 50,
        decision_filter: Optional[str] = None,
        problem_family_filter: Optional[str] = None,
        exclude_demo: bool = False,
    ) -> List[Dict[str, Any]]:
        """Return recent traces matching optional filters."""
        if not self.trace_path.exists():
            return []
        results: List[Dict[str, Any]] = []
        try:
            with open(self.trace_path, encoding="utf-8") as f:
                all_lines = f.readlines()
            for line in reversed(all_lines):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if exclude_demo and record.get("is_demo"):
                    continue
                # Extract final decision step for filtering
                final = next(
                    (s for s in reversed(record.get("steps", [])) if s.get("step") == "final_decision"),
                    None,
                )
                if decision_filter and final:
                    if decision_filter.upper() not in final.get("data", {}).get("routing_decision", ""):
                        continue
                if problem_family_filter:
                    pf_step = next(
                        (s for s in record.get("steps", []) if s.get("step") == "problem_understanding"),
                        None,
                    )
                    if pf_step:
                        if problem_family_filter.upper() not in pf_step.get("data", {}).get("problem_family", "").upper():
                            continue
                results.append(record)
                if len(results) >= limit:
                    break
        except Exception as exc:
            logger.warning("Error listing traces: %s", exc)
        return results

    def compute_latency_stats(self, exclude_demo: bool = True) -> Dict[str, Any]:
        """Compute P50, P95, P99 latency statistics from stored traces."""
        if not self.trace_path.exists():
            return {"p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "count": 0, "mean_ms": 0.0}
        latencies: List[float] = []
        try:
            with open(self.trace_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if exclude_demo and record.get("is_demo"):
                        continue
                    lat = record.get("total_latency_ms")
                    if lat is not None:
                        latencies.append(float(lat))
        except Exception as exc:
            logger.warning("Error computing latency stats: %s", exc)
        if not latencies:
            return {"p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "count": 0, "mean_ms": 0.0}
        latencies.sort()
        n = len(latencies)
        def percentile(pct: float) -> float:
            idx = min(int(pct / 100 * n), n - 1)
            return round(latencies[idx], 2)
        return {
            "count": n,
            "mean_ms": round(sum(latencies) / n, 2),
            "p50_ms": percentile(50),
            "p95_ms": percentile(95),
            "p99_ms": percentile(99),
            "min_ms": round(latencies[0], 2),
            "max_ms": round(latencies[-1], 2),
        }


# ─────────────────────────────────────────────────────────────
# System Metrics Auditor
# ─────────────────────────────────────────────────────────────

class SystemMetricsAuditor:
    """
    Records LLM call outcomes, fallback events, and aggregate counters.
    Provides queryable snapshots for the observability metrics endpoint.
    Strictly read-only w.r.t. decision logic.
    """

    def __init__(self, metrics_path: Optional[Path | str] = None) -> None:
        self.metrics_path = Path(metrics_path or DEFAULT_METRICS_PATH)
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)

    def _append(self, record: Dict[str, Any]) -> None:
        try:
            record["timestamp"] = datetime.now(timezone.utc).isoformat()
            with open(self.metrics_path, mode="a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as exc:
            logger.error("SystemMetricsAuditor write failed: %s", exc)

    def log_llm_request(self, success: bool, latency_ms: float, reason: Optional[str] = None) -> None:
        """Log an LLM inference request outcome (success or failure)."""
        self._append({
            "event": "llm_request",
            "success": success,
            "latency_ms": round(latency_ms, 2),
            "reason": reason or ("ok" if success else "unknown_failure"),
        })

    def log_fallback(self, reason: str, latency_ms: float) -> None:
        """Log activation of the deterministic heuristic fallback."""
        self._append({
            "event": "fallback_activation",
            "reason": redact(reason[:200]),
            "latency_ms": round(latency_ms, 2),
        })

    def log_rate_limit(self) -> None:
        """Log a rate-limit event from the LLM provider."""
        self._append({"event": "rate_limit"})

    def get_llm_stats(self) -> Dict[str, Any]:
        """Compute aggregate LLM observability stats."""
        if not self.metrics_path.exists():
            return {
                "llm_total": 0, "llm_success": 0, "llm_failure": 0,
                "rate_limit_events": 0, "fallback_activations": 0,
                "fallback_rate": 0.0, "llm_success_rate": 0.0,
            }
        total = success = failure = rate_limits = fallbacks = 0
        try:
            with open(self.metrics_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    event_type = ev.get("event", "")
                    if event_type == "llm_request":
                        total += 1
                        if ev.get("success"):
                            success += 1
                        else:
                            failure += 1
                    elif event_type == "rate_limit":
                        rate_limits += 1
                    elif event_type == "fallback_activation":
                        fallbacks += 1
        except Exception as exc:
            logger.warning("Error reading metrics: %s", exc)
        return {
            "llm_total": total,
            "llm_success": success,
            "llm_failure": failure,
            "rate_limit_events": rate_limits,
            "fallback_activations": fallbacks,
            "llm_success_rate": round(success / total, 4) if total > 0 else 0.0,
            "fallback_rate": round(fallbacks / max(total + fallbacks, 1), 4),
        }


# ─────────────────────────────────────────────────────────────
# Singleton accessors (process-level)
# ─────────────────────────────────────────────────────────────

_trace_store: Optional[DecisionTraceStore] = None
_metrics_auditor: Optional[SystemMetricsAuditor] = None


def get_trace_store() -> DecisionTraceStore:
    global _trace_store
    if _trace_store is None:
        _trace_store = DecisionTraceStore()
    return _trace_store


def get_metrics_auditor() -> SystemMetricsAuditor:
    global _metrics_auditor
    if _metrics_auditor is None:
        _metrics_auditor = SystemMetricsAuditor()
    return _metrics_auditor
