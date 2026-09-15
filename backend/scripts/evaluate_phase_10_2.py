"""
SupportGraph AI — Phase 10.2 Evaluation Driver CLI

Runs the full Phase 10.2 evaluation:
  1. Computes pre-evaluation SHA-256 for data/golden/golden_set_human_review.csv
  2. Evaluates the 77-record human-reviewed benchmark with full operational retrieval
  3. Computes post-evaluation SHA-256 and asserts immutability
  4. Generates required Phase 10.2 artifacts:
       reports/phase_10_2/phase_10_1_vs_10_2_comparison.json
       reports/phase_10_2/evidence_recovery_analysis.json
       reports/phase_10_2/retrieval_quality_analysis.json
       reports/phase_10_2/safety_regression_analysis.json
"""
from __future__ import annotations

import json
from pathlib import Path

from backend.app.evaluation.operational_coverage_evaluator import (
    GOLDEN_CSV_PATH,
    GOLDEN_SHA256,
    OperationalCoverageEvaluator,
    compute_sha256,
)

OUTPUT_DIR = Path("reports/phase_10_2")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  SUPPORTGRAPH AI — PHASE 10.2 EVALUATION BENCHMARK")
    print("  Operational Evidence Coverage Expansion & Historical Corpus Retrieval")
    print("=" * 70)

    # 1. Pre-evaluation SHA-256
    sha_pre = compute_sha256(GOLDEN_CSV_PATH)
    print(f"\n[1] Golden Dataset Verification (Pre-Evaluation)")
    print(f"    Target File:   {GOLDEN_CSV_PATH}")
    print(f"    Expected SHA:  {GOLDEN_SHA256}")
    print(f"    Actual SHA:    {sha_pre}")
    assert sha_pre == GOLDEN_SHA256, "PRE-EVALUATION INTEGRITY CHECK FAILED!"
    print("    Integrity:     PASSED (Byte-for-byte verified)")

    # 2. Run Evaluation
    print("\n[2] Executing Phase 10.2 Benchmark Evaluation (77 Records)...")
    evaluator = OperationalCoverageEvaluator()
    metrics, case_results, comp_report = evaluator.evaluate()

    # 3. Post-evaluation SHA-256
    sha_post = compute_sha256(GOLDEN_CSV_PATH)
    print(f"\n[3] Golden Dataset Verification (Post-Evaluation)")
    print(f"    Actual SHA:    {sha_post}")
    assert sha_post == GOLDEN_SHA256, "POST-EVALUATION INTEGRITY CHECK FAILED!"
    assert sha_pre == sha_post, "DATASET MODIFIED DURING EVALUATION!"
    print("    Integrity:     PASSED (Dataset remained completely immutable)")

    # 4. Save Artifacts
    comparison_file = OUTPUT_DIR / "phase_10_1_vs_10_2_comparison.json"
    with open(comparison_file, "w") as f:
        json.dump(comp_report, f, indent=2)

    recovery_file = OUTPUT_DIR / "evidence_recovery_analysis.json"
    recovery_data = {
        "recovery_summary": comp_report["recovery_analysis"],
        "case_diagnostics": [c for c in case_results if c.get("is_good_recovery") or c.get("is_usable_evidence")],
    }
    with open(recovery_file, "w") as f:
        json.dump(recovery_data, f, indent=2)

    retrieval_file = OUTPUT_DIR / "retrieval_quality_analysis.json"
    retrieval_data = {
        "problem_family_match_rate": metrics.problem_family_match_rate,
        "top_1_operational_relevance_rate": metrics.top_1_operational_relevance_rate,
        "top_3_usable_evidence_coverage": metrics.top_3_usable_evidence_coverage,
        "usable_evidence_coverage": metrics.usable_evidence_coverage,
        "direct_problem_match_rate": metrics.direct_problem_match_rate,
        "related_problem_match_rate": metrics.related_problem_match_rate,
        "weak_semantic_match_rate": metrics.weak_semantic_match_rate,
    }
    with open(retrieval_file, "w") as f:
        json.dump(retrieval_data, f, indent=2)

    safety_file = OUTPUT_DIR / "safety_regression_analysis.json"
    safety_data = {
        "auto_handle_rate": metrics.auto_handle_rate,
        "auto_handle_precision": metrics.auto_handle_precision,
        "unsafe_auto_handles": metrics.unsafe_auto_handles,
        "error_interception_rate": metrics.error_interception_rate,
        "safety_regression_detected": metrics.safety_regression_detected,
    }
    with open(safety_file, "w") as f:
        json.dump(safety_data, f, indent=2)

    # 5. Print Results
    print("\n" + "=" * 70)
    print("  PHASE 10.1 vs PHASE 10.2 BENCHMARK RESULTS")
    print("=" * 70)
    print(f"  {'Metric':<34} {'Phase 10.1':<12} {'Phase 10.2':<12} {'Delta':<10}")
    print("  " + "-" * 66)

    comp = comp_report["metrics_comparison"]
    for k, v in comp.items():
        p1 = f"{v['phase_10_1']:.1%}" if isinstance(v['phase_10_1'], float) else str(v['phase_10_1'])
        p2 = f"{v['phase_10_2']:.1%}" if isinstance(v['phase_10_2'], float) else str(v['phase_10_2'])
        d = f"{v['delta']:+.1%}" if isinstance(v['delta'], float) else str(v['delta'])
        print(f"  {k:<34} {p1:<12} {p2:<12} {d:<10}")

    print("\n" + "=" * 70)
    print("  EVIDENCE RECOVERY & SAFETY SUMMARY")
    print("=" * 70)
    rec = comp_report["recovery_analysis"]
    print(f"  Previously Evidence-Limited: {rec['previously_evidence_limited']}")
    print(f"  Cases Recovered (Good):      {rec['good_recovery_count']} ({rec['recovery_rate']:.1%})")
    print(f"  Bad Recoveries (Unsafe):     {rec['bad_recovery_count']}")
    print(f"  Remaining True Knowledge:    {rec['remaining_true_knowledge_gaps']}")
    print(f"  Auto-Handle Precision:       {metrics.auto_handle_precision:.1%}")
    print(f"  Unsafe Auto-Handles:         {metrics.unsafe_auto_handles}")
    print(f"  Safety Regression Detected:  {metrics.safety_regression_detected}")
    print("=" * 70)
    print(f"\nAll artifacts generated in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
