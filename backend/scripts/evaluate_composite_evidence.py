"""
SupportGraph AI — Phase 10 Benchmark Evaluation CLI

Runs the Phase 10 Evidence Coverage Expansion & Multi-Case Evidence Synthesis
evaluation against the 77 protected human ground-truth records.

Usage:
    ./backend/.venv/bin/python -m backend.scripts.evaluate_composite_evidence

Output:
    - Console report with all Phase 10 metrics
    - artifacts/reports/phase_10_composite_evidence_evaluation.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from app.evaluation.composite_evidence_evaluator import CompositeEvidenceEvaluator
except ModuleNotFoundError:
    from backend.app.evaluation.composite_evidence_evaluator import CompositeEvidenceEvaluator


REPORT_DIR = Path("artifacts/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON = REPORT_DIR / "phase_10_composite_evidence_evaluation.json"


def format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def delta_str(value: float, unit: str = "%") -> str:
    arrow = "↑" if value > 0 else ("↓" if value < 0 else "→")
    if unit == "%":
        return f"{arrow} {abs(value * 100):.1f}pp"
    return f"{arrow} {value:+.4f}"


def main() -> None:
    print("=" * 70)
    print("  SupportGraph AI — Phase 10: Composite Evidence Synthesis")
    print("  Benchmark Evaluation on 77 Human Ground-Truth Records")
    print("=" * 70)
    print()
    print("  SHA-256 Protected Dataset: data/golden/golden_set_human_review.csv")
    print("  Verifying dataset integrity before evaluation...")
    print()

    evaluator = CompositeEvidenceEvaluator()

    start = time.time()
    metrics = evaluator.run_benchmark()
    elapsed = time.time() - start

    print(f"  ✓ Evaluation complete in {elapsed:.1f}s")
    print()

    # ── Dataset Integrity ──────────────────────────────────────────────────
    print("─" * 70)
    print("  DATASET INTEGRITY")
    print("─" * 70)
    integrity = "✓ PASS (100% Immutable)" if metrics.dataset_immutability_pass else "✗ FAIL — CONTAMINATION DETECTED"
    print(f"  SHA-256 Pre-Eval:  {metrics.pre_eval_sha256[:32]}...")
    print(f"  SHA-256 Post-Eval: {metrics.post_eval_sha256[:32]}...")
    print(f"  Immutability:      {integrity}")
    print()

    # ── Intent Classification Performance ──────────────────────────────────
    print("─" * 70)
    print("  INTENT CLASSIFICATION PERFORMANCE")
    print("─" * 70)
    print(f"  Total records evaluated:   {metrics.total_evaluated_records}")
    print(f"  Primary Intent Accuracy:   {format_pct(metrics.primary_intent_accuracy)}")
    print(f"  Top-2 Candidate Coverage:  {format_pct(metrics.top_2_coverage)}")
    print(f"  Top-3 Candidate Coverage:  {format_pct(metrics.top_3_coverage)}")
    print(f"  Macro F1:                  {metrics.macro_f1:.4f}")
    print()

    # ── Composite Evidence Verdict Distribution ────────────────────────────
    print("─" * 70)
    print("  COMPOSITE EVIDENCE VERDICT DISTRIBUTION")
    print("─" * 70)
    dist = metrics.composite_verdict_distribution
    tot = metrics.total_evaluated_records
    for verdict_name, count in dist.items():
        pct = count / tot if tot > 0 else 0
        bar = "█" * int(pct * 30)
        print(f"  {verdict_name:<32} {count:>3} cases ({format_pct(pct):>6})  {bar}")
    print()

    # ── Evidence Coverage Expansion ─────────────────────────────────────────
    print("─" * 70)
    print("  EVIDENCE COVERAGE EXPANSION")
    print("─" * 70)
    print(f"  Direct Strong Evidence:    {metrics.direct_strong_evidence_count:>3} cases ({format_pct(metrics.direct_strong_evidence_count / tot)})")
    print(f"  Composite Strong Evidence: {metrics.composite_strong_evidence_count:>3} cases ({format_pct(metrics.composite_strong_evidence_count / tot)})")
    print(f"  Total Usable Evidence:     {metrics.usable_evidence_count:>3} cases ({format_pct(metrics.usable_evidence_coverage_rate)})")
    print(f"  Coverage Expansion Delta:  {delta_str(metrics.coverage_expansion_delta)}")
    print()

    # ── Dimension Coverage & Operational Quality ───────────────────────────
    print("─" * 70)
    print("  DIMENSION COVERAGE & OPERATIONAL ALIGNMENT")
    print("─" * 70)
    print(f"  Symptom Coverage Rate:     {format_pct(metrics.symptom_coverage_rate)}")
    print(f"  Context Coverage Rate:     {format_pct(metrics.context_coverage_rate)}")
    print(f"  Device Coverage Rate:      {format_pct(metrics.device_coverage_rate)}")
    print(f"  Resolution Pattern Rate:   {format_pct(metrics.resolution_pattern_coverage_rate)}")
    print(f"  Avg Intent Consistency:    {format_pct(metrics.average_intent_consistency)}")
    print(f"  Avg Composite Score:       {metrics.average_composite_support_score:.4f}")
    print()

    # ── Conflict Detection & Deduplication ──────────────────────────────────
    print("─" * 70)
    print("  CONFLICT DETECTION & DEDUPLICATION SAFETY")
    print("─" * 70)
    print(f"  Cases with Conflicts:      {metrics.conflict_detected_count} ({format_pct(metrics.conflict_detected_count / tot)})")
    print(f"  Intent Conflicts:          {metrics.intent_conflict_count}")
    print(f"  Symptom Contradictions:    {metrics.symptom_contradiction_count}")
    print(f"  Total Cases Excluded:      {metrics.total_excluded_cases_count}")
    print(f"  Deduplicated Cases:        {metrics.deduplicated_cases_count}")
    print()

    # ── Routing Decision & Safety Metrics ──────────────────────────────────
    print("─" * 70)
    print("  ROUTING DECISION & SAFETY METRICS")
    print("─" * 70)
    print(f"  Auto-Handle Rate:          {format_pct(metrics.auto_handle_rate)}  ({metrics.auto_handle_count} cases)")
    print(f"  Auto-Handle Precision:     {format_pct(metrics.auto_handle_accuracy)}")
    print(f"  Unsafe Auto-Handles:       {metrics.unsafe_auto_handles_count}")
    print(f"  Escalation Rate:           {format_pct(metrics.escalation_rate)}  ({metrics.escalation_count} cases)")
    print(f"  Error Interception Rate:   {format_pct(metrics.error_interception_rate)}")
    print()

    # ── Multi-Phase Progression Comparison ─────────────────────────────────
    print("─" * 70)
    print("  MULTI-PHASE EVOLUTION PROGRESSION")
    print("─" * 70)
    print(f"  {'Phase':<32} {'Auto-Handle':>12} {'Precision':>10} {'Unsafe':>8} {'Intercept':>10}")
    print(f"  {'-'*32} {'-'*12} {'-'*10} {'-'*8} {'-'*10}")
    for phase_name, p_metrics in metrics.progression_comparison.items():
        ah_rate = format_pct(p_metrics['auto_handle_rate'])
        ah_acc = format_pct(p_metrics['auto_handle_acc'])
        unsafe = str(p_metrics['unsafe_auto_handles'])
        interc = format_pct(p_metrics['error_interception_rate'])
        print(f"  {phase_name:<32} {ah_rate:>12} {ah_acc:>10} {unsafe:>8} {interc:>10}")
    print("─" * 70)
    print()

    # Save JSON report
    with open(OUTPUT_JSON, "w") as f:
        json.dump(metrics.model_dump(), f, indent=2)
    print(f"  ✓ Full JSON metrics saved to {OUTPUT_JSON}")
    print("=" * 70)


if __name__ == "__main__":
    main()
