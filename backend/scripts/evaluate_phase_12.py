"""
SupportGraph AI — Phase 12 Full Pipeline Evaluation Runner.

Executes the complete SupportGraph AI end-to-end evaluation harness across:
1. 77 protected golden benchmark records (checking 0 leakage & 0 unsafe auto-handles)
2. 30 unseen synthetic adversarial scenarios across 5 critical groups

Outputs structured JSON artifacts and evaluation reports to reports/phase_12/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

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

logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SupportGraph AI Phase 12 Full Pipeline Evaluator",
    )
    parser.add_argument(
        "--golden-set",
        type=Path,
        default=GOLDEN_CSV_PATH,
        help="Path to protected golden set CSV",
    )
    parser.add_argument(
        "--adversarial-scenarios",
        type=Path,
        default=Path("data/evaluation/phase_12_adversarial_scenarios.json"),
        help="Path to 30 unseen adversarial scenarios JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/phase_12"),
        help="Directory to save evaluation reports and metrics",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("SUPPORTGRAPH AI — PHASE 12 END-TO-END PIPELINE EVALUATION")
    print("=" * 80)

    # 1. Pre-Run Golden Set Integrity
    pre_hash = compute_sha256(args.golden_set)
    print(f"\n[1/4] Verifying Protected Golden Dataset Integrity...")
    print(f"  Path: {args.golden_set}")
    print(f"  Expected SHA-256: {GOLDEN_SHA256}")
    print(f"  Actual SHA-256:   {pre_hash}")
    if pre_hash != GOLDEN_SHA256:
        print("  [ERROR] Golden dataset hash mismatch! Aborting evaluation.")
        return 1
    print("  [PASS] Golden dataset integrity confirmed.")

    # 2. Run Complete Evaluation
    print(f"\n[2/4] Running End-to-End Evaluation...")
    evaluator = Phase12Evaluator()
    report: Phase12EvaluationReport = evaluator.run_full_evaluation(
        golden_csv_path=args.golden_set,
        adversarial_json_path=args.adversarial_scenarios,
    )

    # 3. Post-Run Golden Set Integrity
    post_hash = compute_sha256(args.golden_set)
    print(f"\n[3/4] Verifying Post-Evaluation Immutability...")
    print(f"  Post SHA-256:     {post_hash}")
    if post_hash != GOLDEN_SHA256:
        print("  [ERROR] Golden dataset was modified during evaluation! Aborting.")
        return 1
    print("  [PASS] Golden dataset remained completely unmodified.")

    # 4. Save Artifacts
    print(f"\n[4/4] Exporting Evaluation Artifacts to {args.output_dir}...")

    # Metrics
    metrics_path = args.output_dir / "phase_12_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(report.metrics.model_dump(mode="json"), f, indent=2, default=str)
    print(f"  -> Saved {metrics_path}")

    # Adversarial Results
    adv_path = args.output_dir / "phase_12_adversarial_results.json"
    with open(adv_path, "w", encoding="utf-8") as f:
        json.dump(
            [r.model_dump(mode="json") for r in report.unseen_eval_records],
            f,
            indent=2,
            default=str,
        )
    print(f"  -> Saved {adv_path}")

    # Leakage Verification
    leakage_data = {
        "timestamp": report.timestamp,
        "golden_dataset_path": str(args.golden_set),
        "golden_sha256": post_hash,
        "golden_records_count": report.golden_dataset_records,
        "historical_corpus_overlap_count": report.leakage_count,
        "zero_leakage_status": "PASS" if report.leakage_count == 0 else "FAIL",
    }
    leakage_path = args.output_dir / "phase_12_leakage_verification.json"
    with open(leakage_path, "w", encoding="utf-8") as f:
        json.dump(leakage_data, f, indent=2)
    print(f"  -> Saved {leakage_path}")

    # Latency Breakdown
    latency_data = {
        "timestamp": report.timestamp,
        "avg_latency_ms": report.metrics.avg_latency_ms,
        "p50_latency_ms": report.metrics.p50_latency_ms,
        "p95_latency_ms": report.metrics.p95_latency_ms,
        "p99_latency_ms": report.metrics.p99_latency_ms,
        "component_breakdown_ms": report.metrics.component_latency_breakdown_ms,
    }
    latency_path = args.output_dir / "phase_12_latency.json"
    with open(latency_path, "w", encoding="utf-8") as f:
        json.dump(latency_data, f, indent=2)
    print(f"  -> Saved {latency_path}")

    # Print Summary Report
    print("\n" + "=" * 80)
    print("PHASE 12 EVALUATION SUMMARY")
    print("=" * 80)
    m = report.metrics
    print(f"1. PROBLEM UNDERSTANDING:")
    print(f"   - Primary Intent Accuracy:       {m.primary_intent_accuracy * 100:.1f}%")
    print(f"   - Top-2 Candidate Coverage:      {m.top_2_candidate_coverage * 100:.1f}%")
    print(f"   - Top-3 Candidate Coverage:      {m.top_3_candidate_coverage * 100:.1f}%")
    print(f"   - Problem Family Accuracy:       {m.problem_family_accuracy * 100:.1f}%")

    print(f"\n2. EVIDENCE COVERAGE & GROUNDING:")
    print(f"   - Usable Evidence Coverage:      {m.usable_evidence_coverage * 100:.1f}%")
    print(f"   - Direct Problem Match Rate:     {m.direct_problem_match_rate * 100:.1f}%")
    print(f"   - Evidence-Limited Rate:         {m.evidence_limited_rate * 100:.1f}%")
    print(f"   - Weak Semantic Match Rate:      {m.weak_semantic_match_rate * 100:.1f}%")
    print(f"   - Avg Grounding Score:           {m.avg_evidence_grounding_score:.2f}")

    print(f"\n3. DECISION SAFETY & GUARDRAILS:")
    print(f"   - Safe Auto-Handle Rate:         {m.auto_handle_rate * 100:.1f}%")
    print(f"   - Auto-Handle Precision:         {m.auto_handle_precision * 100:.1f}%")
    print(f"   - UNSAFE AUTO-HANDLES:           {m.unsafe_auto_handle_count} (0 required)")
    print(f"   - Benchmark Leakage:             {report.leakage_count} (0 required)")

    print(f"\n4. MULTI-TURN & CONVERSATION QUALITY:")
    print(f"   - Context Retention Rate:        {m.context_retention_rate * 100:.1f}%")
    print(f"   - Repeat Prevention Rate:        {m.repeat_prevention_rate * 100:.1f}%")
    print(f"   - Clarification Precision:       {m.clarification_precision * 100:.1f}%")
    print(f"   - Resolution Confirmation:       {m.resolution_confirmation_accuracy * 100:.1f}%")

    print(f"\n5. ADVERSARIAL EVALUATION:")
    print(f"   - Total Unseen Scenarios:        {report.total_unseen_scenarios}")
    print(f"   - Passed Scenarios:              {report.passed_unseen_scenarios} / {report.total_unseen_scenarios}")
    print(f"   - Unseen Pass Rate:              {report.unseen_scenario_pass_rate * 100:.1f}%")

    print(f"\n6. PIPELINE RELIABILITY & LATENCY:")
    print(f"   - Pipeline Failure Rate:         {m.pipeline_failure_rate * 100:.1f}%")
    print(f"   - Fallback Success Rate:         {m.fallback_success_rate * 100:.1f}%")
    print(f"   - Avg Latency:                   {m.avg_latency_ms:.2f} ms")
    print(f"   - p50 / p95 / p99 Latency:       {m.p50_latency_ms:.1f} ms / {m.p95_latency_ms:.1f} ms / {m.p99_latency_ms:.1f} ms")

    print(f"\n7. OUTCOME DISTRIBUTION:")
    for outcome_key, count in sorted(report.outcome_distribution.items()):
        print(f"   - {outcome_key:<32}: {count}")

    print("=" * 80)
    print("PHASE 12 EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
