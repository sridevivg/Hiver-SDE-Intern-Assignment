"""
SupportGraph AI — Phase 9 Benchmark Evaluation CLI

Runs the Phase 9 Escalation Quality Analysis & Selective Recovery evaluation
against the 77 protected human ground-truth records.

Usage:
    ./backend/.venv/bin/python -m backend.scripts.evaluate_escalation_quality

Output:
    - Console report with all Phase 9 metrics
    - artifacts/reports/phase_9_escalation_quality_evaluation.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from app.evaluation.escalation_quality_evaluator import EscalationQualityEvaluator
except ModuleNotFoundError:
    from backend.app.evaluation.escalation_quality_evaluator import EscalationQualityEvaluator


REPORT_DIR = Path("artifacts/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON = REPORT_DIR / "phase_9_escalation_quality_evaluation.json"


def format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def delta_str(value: float, unit: str = "%") -> str:
    arrow = "↑" if value > 0 else ("↓" if value < 0 else "→")
    if unit == "%":
        return f"{arrow} {abs(value * 100):.1f}pp"
    return f"{arrow} {value:+.4f}"


def main() -> None:
    print("=" * 70)
    print("  SupportGraph AI — Phase 9: Escalation Quality & Selective Recovery")
    print("  Benchmark Evaluation on 77 Human Ground-Truth Records")
    print("=" * 70)
    print()
    print("  SHA-256 Protected Dataset: data/golden/golden_set_human_review.csv")
    print("  Verifying dataset integrity before evaluation...")
    print()

    evaluator = EscalationQualityEvaluator()

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

    # ── Phase 8 Baseline ──────────────────────────────────────────────────
    print("─" * 70)
    print("  PHASE 8 BASELINE (re-measured from this run)")
    print("─" * 70)
    print(f"  Total records evaluated:   {metrics.total_evaluated_records}")
    print(f"  Primary Intent Accuracy:   {format_pct(metrics.phase8_primary_intent_accuracy)}")
    print(f"  Macro F1:                  {metrics.phase8_macro_f1:.4f}")
    print(f"  Auto-Handle Rate:          {format_pct(metrics.phase8_auto_handle_rate)}  ({metrics.phase8_auto_handle_count} cases)")
    print(f"  Auto-Handle Precision:     {format_pct(metrics.phase8_auto_handle_accuracy)}")
    print(f"  Unsafe Auto-Handles:       {metrics.phase8_unsafe_auto_handles}")
    print(f"  Escalation Rate:           {format_pct(metrics.phase8_escalation_rate)}  ({metrics.phase8_escalation_count} cases)")
    print(f"  Error Interception Rate:   {format_pct(metrics.phase8_error_interception_rate)}")
    print()

    # ── Phase 9: Escalation Category Distribution ─────────────────────────
    print("─" * 70)
    print("  PHASE 9: ESCALATION QUALITY CATEGORY DISTRIBUTION")
    print("─" * 70)
    total_esc = metrics.phase8_escalation_count
    dist = metrics.escalation_category_distribution
    categories = [
        ("GENUINE_AMBIGUITY",        "Genuine Ambiguity"),
        ("MULTI_PROBLEM_COMPLEXITY", "Multi-Problem Complexity"),
        ("INSUFFICIENT_INFORMATION", "Insufficient Information"),
        ("EVIDENCE_LIMITED",         "Evidence Limited"),
        ("VERIFICATION_VETO",        "Verification Veto"),
        ("RECOVERABLE_ESCALATION",   "Recoverable Escalation  ← Target"),
        ("HUMAN_REQUIRED",           "Human Required"),
    ]
    for key, label in categories:
        count = dist.get(key, 0)
        pct = f"{(count / total_esc * 100):.1f}%" if total_esc > 0 else "0.0%"
        marker = "  ►" if key == "RECOVERABLE_ESCALATION" else "   "
        print(f"{marker}  {label:<32}  {count:3d}  ({pct} of escalations)")

    print()
    print(f"  Recoverable escalations: {metrics.recoverable_escalation_count} "
          f"({format_pct(metrics.recoverable_escalation_rate_of_total)} of total, "
          f"{format_pct(metrics.recoverable_escalation_rate_of_escalated)} of escalated)")
    print()

    # ── Phase 9: Selective Recovery Results ──────────────────────────────
    print("─" * 70)
    print("  PHASE 9: SELECTIVE RECOVERY RESULTS")
    print("─" * 70)
    print(f"  Recovery attempts:         {metrics.recovery_attempted_count}")
    print(f"  Successfully recovered:    {metrics.recovery_success_count}  "
          f"(success rate: {format_pct(metrics.recovery_success_rate)})")
    print(f"  Blocked by safety:         {metrics.recovery_failed_safety_count}")
    print()

    # ── Phase 9 Post-Recovery Metrics ────────────────────────────────────
    print("─" * 70)
    print("  PHASE 9: POST-RECOVERY ROUTING METRICS")
    print("─" * 70)
    print(f"  Auto-Handle Rate:          {format_pct(metrics.phase9_auto_handle_rate)}  "
          f"({metrics.phase9_auto_handle_count} cases)  "
          f"[{delta_str(metrics.auto_handle_rate_delta)} vs Phase 8]")
    print(f"  Auto-Handle Precision:     {format_pct(metrics.phase9_auto_handle_accuracy)}  "
          f"[{delta_str(metrics.auto_handle_accuracy_delta)} vs Phase 8]")
    print(f"  Unsafe Auto-Handles:       {metrics.phase9_unsafe_auto_handles}  "
          f"[Δ {metrics.unsafe_auto_handles_delta:+d} vs Phase 8]")
    print(f"  Escalation Rate:           {format_pct(metrics.phase9_escalation_rate)}  "
          f"({metrics.phase9_escalation_count} cases)")
    print(f"  Error Interception Rate:   {format_pct(metrics.phase9_error_interception_rate)}  "
          f"[{delta_str(metrics.error_interception_rate_delta)} vs Phase 8]")
    print()

    # Safety check warning
    if metrics.error_interception_rate_delta < -0.02:
        print("  ⚠  WARNING: Error interception rate degraded significantly vs Phase 8!")
        print("     Recovery may be introducing unsafe auto-handles — review immediately.")
    elif metrics.phase9_unsafe_auto_handles > metrics.phase8_unsafe_auto_handles:
        print("  ⚠  WARNING: Unsafe auto-handles INCREASED vs Phase 8.")
        print("     Recovery engine safety gate may require tightening.")
    else:
        print("  ✓  SAFETY PRESERVED: Error interception rate maintained or improved.")
    print()

    # ── Multi-Phase Progression ───────────────────────────────────────────
    print("─" * 70)
    print("  MULTI-PHASE PROGRESSION MATRIX")
    print("─" * 70)
    print(f"  {'Phase':<40} {'Auto%':>7} {'Prec%':>7} {'Unsafe':>7} {'EIR%':>7}")
    print("  " + "─" * 64)
    prog = metrics.progression_comparison
    for phase_key, data in prog.items():
        name = phase_key.replace("_", " ")[:38]
        ah = f"{data.get('auto_handle_rate', 0) * 100:.1f}%"
        pr = f"{data.get('auto_handle_precision', 0) * 100:.1f}%"
        us = str(data.get('unsafe_auto_handles', '-'))
        ei = f"{data.get('error_interception_rate', 0) * 100:.1f}%"
        print(f"  {name:<40} {ah:>7} {pr:>7} {us:>7} {ei:>7}")
    print()

    # ── Save JSON Report ──────────────────────────────────────────────────
    output = metrics.model_dump()
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  ✓ JSON report saved to: {OUTPUT_JSON}")
    print()
    print("=" * 70)
    print("  Phase 9 benchmark evaluation complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
