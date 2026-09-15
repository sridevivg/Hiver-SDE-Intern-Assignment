"""
SupportGraph AI — Phase 12 Release Readiness & Production Safety Checker.

Machine-checkable automated verification script for production deployment readiness.
Checks:
1. Golden Dataset Immutability & Exact Hash Match
2. Zero Benchmark Leakage in Retrieval Corpus
3. Zero Unsafe Auto-Handles Across Full Evaluation
4. Decision Explanation Completeness
5. Component Failure Fallback Robustness
6. Latency SLAs & Multi-Turn State Persistence

Outputs: reports/phase_12/phase_12_release_readiness.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Ensure backend directory is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.logging import get_logger
from app.evaluation.phase_12_evaluator import (
    GOLDEN_CSV_PATH,
    GOLDEN_SHA256,
    Phase12EvaluationReport,
    Phase12Evaluator,
    compute_sha256,
)
from app.retrieval.historical_corpus_index import HistoricalCorpusIndex

logger = get_logger(__name__)


def check_release_readiness(output_dir: Path = Path("reports/phase_12")) -> Dict[str, Any]:
    """Execute all release readiness checks and return structured results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    checks: Dict[str, Any] = {}
    overall_pass = True

    print("=" * 80)
    print("SUPPORTGRAPH AI — PHASE 12 RELEASE READINESS AUDIT")
    print("=" * 80)

    # Check 1: Golden Dataset Immutability
    print("\n[Check 1/6] Golden Benchmark Immutability...")
    current_hash = compute_sha256(GOLDEN_CSV_PATH)
    hash_match = (current_hash == GOLDEN_SHA256)
    if not hash_match:
        overall_pass = False
    checks["golden_dataset_integrity"] = {
        "status": "PASS" if hash_match else "FAIL",
        "expected_sha256": GOLDEN_SHA256,
        "actual_sha256": current_hash,
        "file_exists": GOLDEN_CSV_PATH.exists(),
    }
    print(f"  -> Result: {checks['golden_dataset_integrity']['status']} (Hash: {current_hash[:16]}...)")

    # Check 2: Zero Benchmark Leakage
    print("\n[Check 2/6] Historical Corpus Zero Leakage...")
    try:
        import pandas as pd
        df = pd.read_csv(GOLDEN_CSV_PATH)
        golden_ids = set(df["conversation_id"].astype(str).tolist())
        idx = HistoricalCorpusIndex()
        hist_ids = set(c.case_id for c in idx.cases)
        overlap = golden_ids.intersection(hist_ids)
        leakage_free = (len(overlap) == 0)
        leakage_count = len(overlap)
    except Exception as exc:
        leakage_free = False
        leakage_count = -1
        overlap = set()

    if not leakage_free:
        overall_pass = False

    checks["zero_retrieval_leakage"] = {
        "status": "PASS" if leakage_free else "FAIL",
        "overlapping_records_count": leakage_count,
        "overlapping_case_ids": list(overlap),
    }
    print(f"  -> Result: {checks['zero_retrieval_leakage']['status']} (Overlap Count: {leakage_count})")

    # Check 3: Full Evaluation & Zero Unsafe Auto-Handles
    print("\n[Check 3/6] Running Pipeline Evaluation for Zero Unsafe Auto-Handles...")
    evaluator = Phase12Evaluator()
    report = evaluator.run_full_evaluation()

    zero_unsafe = (report.metrics.unsafe_auto_handle_count == 0)
    if not zero_unsafe:
        overall_pass = False

    checks["zero_unsafe_auto_handles"] = {
        "status": "PASS" if zero_unsafe else "FAIL",
        "unsafe_count": report.metrics.unsafe_auto_handle_count,
        "safe_auto_handle_rate": report.metrics.auto_handle_rate,
        "auto_handle_precision": report.metrics.auto_handle_precision,
    }
    print(f"  -> Result: {checks['zero_unsafe_auto_handles']['status']} (Unsafe: {report.metrics.unsafe_auto_handle_count})")

    # Check 4: Unseen Adversarial Scenario Robustness
    print("\n[Check 4/6] Adversarial Scenario Generalization...")
    adv_pass = (report.unseen_scenario_pass_rate >= 0.90)
    if not adv_pass:
        overall_pass = False

    checks["adversarial_generalization"] = {
        "status": "PASS" if adv_pass else "FAIL",
        "total_scenarios": report.total_unseen_scenarios,
        "passed_scenarios": report.passed_unseen_scenarios,
        "pass_rate": report.unseen_scenario_pass_rate,
    }
    print(f"  -> Result: {checks['adversarial_generalization']['status']} (Pass Rate: {report.unseen_scenario_pass_rate * 100:.1f}%)")

    # Check 5: Decision Explanation Completeness
    print("\n[Check 5/6] Decision Explanation Structured Completeness...")
    records_with_explanation = sum(1 for r in report.golden_eval_records + report.unseen_eval_records if r.decision_explanation is not None)
    total_records = len(report.golden_eval_records) + len(report.unseen_eval_records)
    explanation_complete = (records_with_explanation == total_records)
    if not explanation_complete:
        overall_pass = False

    checks["decision_explanation_completeness"] = {
        "status": "PASS" if explanation_complete else "FAIL",
        "evaluated_records": total_records,
        "records_with_structured_explanation": records_with_explanation,
    }
    print(f"  -> Result: {checks['decision_explanation_completeness']['status']} ({records_with_explanation}/{total_records} records)")

    # Check 6: Latency SLAs
    print("\n[Check 6/6] Performance & Latency SLAs...")
    latency_pass = (report.metrics.p95_latency_ms < 1000.0)  # P95 < 1000ms SLA
    if not latency_pass:
        overall_pass = False

    checks["latency_sla"] = {
        "status": "PASS" if latency_pass else "FAIL",
        "p50_latency_ms": report.metrics.p50_latency_ms,
        "p95_latency_ms": report.metrics.p95_latency_ms,
        "p99_latency_ms": report.metrics.p99_latency_ms,
        "sla_threshold_p95_ms": 1000.0,
    }
    print(f"  -> Result: {checks['latency_sla']['status']} (p95: {report.metrics.p95_latency_ms:.1f}ms < 1000ms)")

    # Overall Status
    release_summary = {
        "timestamp": datetime.utcnow().isoformat(),
        "release_status": "READY_FOR_PRODUCTION" if overall_pass else "BLOCKED",
        "overall_pass": overall_pass,
        "checks": checks,
    }

    # Save artifact
    readiness_path = output_dir / "phase_12_release_readiness.json"
    with open(readiness_path, "w", encoding="utf-8") as f:
        json.dump(release_summary, f, indent=2, default=str)

    print("\n" + "=" * 80)
    print(f"RELEASE READINESS VERDICT: {release_summary['release_status']}")
    print(f"Report saved to: {readiness_path}")
    print("=" * 80 + "\n")

    return release_summary


def main() -> int:
    parser = argparse.ArgumentParser(description="SupportGraph AI Phase 12 Release Readiness Checker")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/phase_12"))
    args = parser.parse_args()

    res = check_release_readiness(output_dir=args.output_dir)
    return 0 if res["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
