"""
SupportGraph AI — CLI Benchmark Runner for Phase 8 Evidence-Grounded Resolution

Runs the comprehensive evaluation suite against the 77 protected human ground truth records:
- Verifies SHA-256 pre- and post-eval immutability
- Computes Intent Classification, Evidence Quality, Routing & Safety, and Response Grounding metrics
- Generates a multi-phase comparison matrix
- Saves results to artifacts/reports/phase_8_evidence_resolution_evaluation.json
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from backend.app.core.logging import get_logger
    from backend.app.evaluation.evidence_resolution_evaluator import (
        EvidenceResolutionEvaluator,
        Phase8BenchmarkMetrics,
    )
except ModuleNotFoundError:
    from app.core.logging import get_logger  # type: ignore[no-redef]
    from app.evaluation.evidence_resolution_evaluator import (  # type: ignore[no-redef]
        EvidenceResolutionEvaluator,
        Phase8BenchmarkMetrics,
    )

logger = get_logger(__name__)

ARTIFACTS_DIR = Path("artifacts/reports")
OUTPUT_JSON_PATH = ARTIFACTS_DIR / "phase_8_evidence_resolution_evaluation.json"


def main() -> None:
    print("\n" + "=" * 80)
    print("SUPPORTGRAPH AI — PHASE 8 BENCHMARK EVALUATION")
    print("Evidence-Grounded Response Generation & Resolution Verification")
    print("=" * 80)

    evaluator = EvidenceResolutionEvaluator()
    metrics: Phase8BenchmarkMetrics = evaluator.run_benchmark()

    print(f"\n[1] DATASET INTEGRITY VERIFICATION")
    print(f"  • Pre-Eval  SHA-256: {metrics.pre_eval_sha256}")
    print(f"  • Post-Eval SHA-256: {metrics.post_eval_sha256}")
    print(f"  • Dataset Immutability: {'PASS (100% Intact & Untouched)' if metrics.dataset_immutability_pass else 'FAIL (Corrupted!)'}")
    print(f"  • Evaluated Completed Human Records: {metrics.completed_human_ground_truth_records}")

    print(f"\n[2] INTENT CLASSIFICATION PERFORMANCE")
    print(f"  • Primary Intent Accuracy (Top-1): {metrics.primary_intent_accuracy:.1%}")
    print(f"  • Top-2 Candidate Coverage:       {metrics.top_2_coverage:.1%}")
    print(f"  • Top-3 Candidate Coverage:       {metrics.top_3_coverage:.1%}")
    print(f"  • Macro F1 Score:                 {metrics.macro_f1:.4f}")

    print(f"\n[3] OPERATIONAL EVIDENCE VALIDATION")
    print(f"  • Direct Evidence Availability:   {metrics.direct_evidence_availability_rate:.1%}")
    print(f"  • Average Intent Agreement:       {metrics.average_intent_agreement:.1%}")
    print(f"  • Average Symptom Agreement:      {metrics.average_symptom_agreement:.1%}")
    print(f"  • Evidence Consistency:           {metrics.average_evidence_consistency:.1%}")
    print("  • Verdict Distribution:")
    for v, c in metrics.verdict_distribution.items():
        print(f"    - {v:24s}: {c:2d} ({c/metrics.total_evaluated_records:.1%})")
    print("  • Retrieved Match Tiers (Total Cases):")
    for t, c in metrics.match_tier_counts.items():
        print(f"    - {t:24s}: {c:2d}")

    print(f"\n[4] ROUTING & SAFETY BENCHMARK")
    print(f"  • Auto-Handle Count / Rate:       {metrics.auto_handle_count} / {metrics.auto_handle_rate:.1%}")
    print(f"  • Auto-Handle Precision:          {metrics.auto_handle_accuracy:.1%}")
    print(f"  • Unsafe Auto-Handles (Errors):   {metrics.unsafe_auto_handles_count}")
    print(f"  • Escalation Count / Rate:        {metrics.escalation_count} / {metrics.escalation_rate:.1%}")
    print(f"  • Error Interception Rate:        {metrics.error_interception_rate:.1%} (intercepts errors safely)")
    print(f"  • Human Assistance (Top-2):       {metrics.human_assistance_top_2_rate:.1%}")

    print(f"\n[5] RESPONSE GROUNDING & VERIFICATION")
    print(f"  • Response Grounding Pass Rate:   {metrics.response_verification_pass_rate:.1%}")
    print(f"  • Average Grounding Score:        {metrics.average_grounding_score:.2f}")
    print(f"  • Grounded Response Rate:         {metrics.evidence_grounded_response_rate:.1%}")
    print(f"  • Unsupported Response Rate:      {metrics.unsupported_response_rate:.1%}")

    print(f"\n[6] PROGRESSION COMPARISON MATRIX ACROSS PHASES")
    print("-" * 80)
    print(f"{'Phase / Architecture':<36s} | {'Auto-Rate':<10s} | {'Precision':<10s} | {'Unsafe Errs':<11s} | {'Interception':<12s}")
    print("-" * 80)
    for phase_name, p_data in metrics.progression_comparison.items():
        ar = f"{p_data.get('auto_handle_rate', 0.0):.1%}"
        pr = f"{p_data.get('auto_handle_precision', 0.0):.1%}"
        err = f"{p_data.get('unsafe_auto_handles', 0)}"
        ir = f"{p_data.get('error_interception_rate', 0.0):.1%}"
        print(f"{phase_name:<36s} | {ar:<10s} | {pr:<10s} | {err:<11s} | {ir:<12s}")
    print("-" * 80)

    # Save to artifacts
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics.model_dump(), f, indent=2)
    print(f"\nSaved benchmark evaluation artifact to: {OUTPUT_JSON_PATH}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
