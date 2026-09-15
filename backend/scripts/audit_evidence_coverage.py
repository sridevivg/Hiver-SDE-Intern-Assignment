"""
SupportGraph AI — Evidence Coverage Audit CLI (Phase 10.1)

Runs the complete evidence coverage diagnostic audit:

  1. Verifies golden dataset SHA-256 (before)
  2. Loads the protected benchmark (77 human-reviewed records)
  3. Runs production pipeline to identify evidence-limited cases
  4. Builds leakage-safe extended corpus (historical parquet, golden IDs excluded)
  5. Runs root-cause diagnostic audit on each evidence-limited case
  6. Produces root-cause statistics
  7. Exports structured JSON artifacts
  8. Verifies golden dataset SHA-256 (after)
  9. Prints engineering conclusion

Usage:
  ./backend/.venv/bin/python -m backend.scripts.audit_evidence_coverage
  ./backend/.venv/bin/python -m backend.scripts.audit_evidence_coverage --corpus-size 10000
  ./backend/.venv/bin/python -m backend.scripts.audit_evidence_coverage --no-json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure backend package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from app.evaluation.evidence_corpus_builder import (
        GOLDEN_SHA256,
        compute_sha256,
    )
    from app.evaluation.evidence_coverage_evaluator import (
        EvidenceCoverageEvaluator,
        Phase101BenchmarkMetrics,
    )
    from app.evaluation.evidence_coverage_auditor import AuditRootCause
    from app.evaluation.evidence_corpus_builder import EvidenceCorpusBuilder
except ModuleNotFoundError:
    from backend.app.evaluation.evidence_corpus_builder import (  # type: ignore[no-redef]
        GOLDEN_SHA256,
        compute_sha256,
    )
    from backend.app.evaluation.evidence_coverage_evaluator import (  # type: ignore[no-redef]
        EvidenceCoverageEvaluator,
        Phase101BenchmarkMetrics,
    )
    from backend.app.evaluation.evidence_coverage_auditor import (  # type: ignore[no-redef]
        AuditRootCause,
    )
    from backend.app.evaluation.evidence_corpus_builder import (  # type: ignore[no-redef]
        EvidenceCorpusBuilder,
    )

GOLDEN_CSV = Path("data/golden/golden_set_human_review.csv")
ARTIFACTS_DIR = Path("reports/evidence_coverage_audit")


def _verify_sha256(label: str) -> str:
    actual = compute_sha256(GOLDEN_CSV)
    match = actual == GOLDEN_SHA256
    status = "✓ PASS" if match else "✗ FAIL"
    print(f"\n  SHA-256 ({label}): {status}")
    print(f"  Expected : {GOLDEN_SHA256}")
    print(f"  Actual   : {actual}")
    if not match:
        print("\n  ⚠️  GOLDEN DATASET HASH MISMATCH — STOPPING")
        sys.exit(1)
    return actual


def _fmt_pct(rate: float) -> str:
    return f"{rate * 100:.1f}%"


def _bar(count: int, total: int, width: int = 20) -> str:
    if total == 0:
        return " " * width
    filled = int(count / total * width)
    return "█" * filled + "░" * (width - filled)


def print_results(metrics: Phase101BenchmarkMetrics) -> None:
    ev = metrics.evidence_limited_count or 1  # avoid div by zero in display

    print("\n" + "=" * 64)
    print("SUPPORTGRAPH AI — EVIDENCE COVERAGE AUDIT (Phase 10.1)")
    print("=" * 64)
    print(f"\n  Golden Dataset SHA-256 : {'PASS ✓' if metrics.dataset_immutability_pass else 'FAIL ✗'}")
    print(f"  Audit Corpus Size      : {metrics.audit_corpus_size:,} entries")
    print(f"  Golden IDs Excluded    : {metrics.audit_corpus_excluded_golden_ids}")
    print(f"  Extended Search Depth  : top-{metrics.audit_extended_search_depth}")

    print("\n" + "-" * 64)
    print("  BENCHMARK BREAKDOWN")
    print("-" * 64)
    print(f"  Total Evaluated Records : {metrics.total_evaluated_records}")
    print(f"  Auto-Handled            : {metrics.auto_handled_count}")
    print(f"  Total Escalated         : {metrics.total_escalated}")
    print(f"  ├─ Evidence-Limited     : {metrics.evidence_limited_count}  ← AUDITED")
    print(f"  └─ Other Escalations    : {metrics.other_escalation_count}")

    print("\n" + "-" * 64)
    print("  ROOT CAUSE DISTRIBUTION (Evidence-Limited Cases)")
    print("-" * 64)

    causes_ordered = [
        (AuditRootCause.RANKING_MISS.value, metrics.ranking_miss_count),
        (AuditRootCause.RETRIEVAL_MISS.value, metrics.retrieval_miss_count),
        (AuditRootCause.VALIDATION_OVER_REJECTION.value, metrics.validation_over_rejection_count),
        (AuditRootCause.REPRESENTATION_LIMITATION.value, metrics.representation_limitation_count),
        (AuditRootCause.INSUFFICIENT_INFORMATION.value, metrics.insufficient_information_count),
        (AuditRootCause.TRUE_KNOWLEDGE_GAP.value, metrics.true_knowledge_gap_count),
    ]

    for cause, count in causes_ordered:
        pct = metrics.root_cause_percentages.get(cause, 0.0)
        bar = _bar(count, ev)
        tag = " ← PRIMARY" if cause == metrics.primary_bottleneck else ""
        print(f"\n  {cause:<35} : {count:>3} ({pct:>5.1f}%)")
        print(f"  [{bar}]{tag}")

    print("\n" + "-" * 64)
    print("  KEY DIAGNOSTIC RATES")
    print("-" * 64)
    print(f"\n  Evidence-Limited Cases With Relevant Evidence Anywhere  : "
          f"{_fmt_pct(metrics.corpus_coverage_rate)}")
    if metrics.avg_best_relevant_rank:
        print(f"  Average Rank of Relevant Evidence (when found)         : "
              f"#{metrics.avg_best_relevant_rank:.0f}")

    print(f"\n  RECOVERABLE FAILURE RATE : {_fmt_pct(metrics.recoverable_failure_rate)}")
    print(f"  (Pipeline failures fixable through engineering)")
    print(f"\n  TRUE KNOWLEDGE GAP RATE  : {_fmt_pct(metrics.true_knowledge_gap_rate)}")
    print(f"  (Require corpus expansion — not fixable by pipeline changes)")
    print(f"\n  INSUFFICIENT INFORMATION : {_fmt_pct(metrics.insufficient_information_rate)}")
    print(f"  (Require better ambiguity gate or customer clarification)")

    print("\n" + "=" * 64)
    print("  ENGINEERING CONCLUSION")
    print("=" * 64)
    print(f"\n  Primary Bottleneck  : {metrics.primary_bottleneck}")
    print(f"\n  Recommended Action  : {metrics.recommended_next_phase}")
    print()

    if metrics.primary_bottleneck in (
        AuditRootCause.RANKING_MISS.value,
        AuditRootCause.RETRIEVAL_MISS.value,
    ):
        print("  OUTCOME A/B: Retrieval/Ranking Improvement")
        print("  ─────────────────────────────────────────────")
        print("  The majority of evidence-limited cases have relevant evidence")
        print("  in the historical corpus — but the production pipeline cannot")
        print("  surface it within top-3 retrieval.")
        print()
        print("  ROOT CAUSE: CaseRetriever uses only 200 golden benchmark records")
        print("  as its corpus instead of the 80,717 available historical conversations.")
        print()
        print("  → Next Phase 10.2: Build proper historical retrieval index.")

    elif metrics.primary_bottleneck == AuditRootCause.TRUE_KNOWLEDGE_GAP.value:
        print("  OUTCOME D: Corpus Coverage Improvement")
        print("  ─────────────────────────────────────────────")
        print("  Genuine knowledge gaps dominate. Pipeline engineering alone is")
        print("  insufficient — new corpus coverage must be acquired.")

    elif metrics.primary_bottleneck == AuditRootCause.VALIDATION_OVER_REJECTION.value:
        print("  OUTCOME C: Evidence Validator Calibration")
        print("  ─────────────────────────────────────────────")
        print("  Evidence is retrieved but rejected by over-strict validation thresholds.")

    print("\n" + "=" * 64 + "\n")


def export_artifacts(metrics: Phase101BenchmarkMetrics, output_dir: Path) -> None:
    """Export structured JSON artifacts to the reports directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Audit summary
    summary = {
        "phase": "10.1",
        "title": "Evidence Coverage Audit",
        "total_evaluated_records": metrics.total_evaluated_records,
        "evidence_limited_count": metrics.evidence_limited_count,
        "audit_corpus_size": metrics.audit_corpus_size,
        "golden_immutability": "PASS" if metrics.dataset_immutability_pass else "FAIL",
        "pre_eval_sha256": metrics.pre_eval_sha256,
        "post_eval_sha256": metrics.post_eval_sha256,
    }
    (output_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2))

    # 2. Root cause distribution
    (output_dir / "root_cause_distribution.json").write_text(
        json.dumps({
            "root_cause_counts": metrics.root_cause_distribution,
            "root_cause_percentages": metrics.root_cause_percentages,
            "recoverable_failure_count": metrics.recoverable_failure_count,
            "recoverable_failure_rate": metrics.recoverable_failure_rate,
            "true_knowledge_gap_count": metrics.true_knowledge_gap_count,
            "true_knowledge_gap_rate": metrics.true_knowledge_gap_rate,
            "primary_bottleneck": metrics.primary_bottleneck,
        }, indent=2)
    )

    # 3. Case-level diagnostics
    (output_dir / "case_level_diagnostics.json").write_text(
        json.dumps(metrics.case_audit_records, indent=2, default=str)
    )

    # 4. Retrieval miss analysis
    retrieval_miss_cases = [
        r for r in metrics.case_audit_records
        if r.get("root_cause") == AuditRootCause.RETRIEVAL_MISS.value
    ]
    (output_dir / "retrieval_miss_analysis.json").write_text(
        json.dumps({
            "count": metrics.retrieval_miss_count,
            "cases": retrieval_miss_cases,
        }, indent=2, default=str)
    )

    # 5. Ranking miss analysis
    ranking_miss_cases = [
        r for r in metrics.case_audit_records
        if r.get("root_cause") == AuditRootCause.RANKING_MISS.value
    ]
    (output_dir / "ranking_miss_analysis.json").write_text(
        json.dumps({
            "count": metrics.ranking_miss_count,
            "cases": ranking_miss_cases,
            "avg_best_relevant_rank": metrics.avg_best_relevant_rank,
        }, indent=2, default=str)
    )

    # 6. Validation rejection analysis
    validation_cases = [
        r for r in metrics.case_audit_records
        if r.get("root_cause") == AuditRootCause.VALIDATION_OVER_REJECTION.value
    ]
    (output_dir / "validation_rejection_analysis.json").write_text(
        json.dumps({
            "count": metrics.validation_over_rejection_count,
            "cases": validation_cases,
        }, indent=2, default=str)
    )

    # 7. True knowledge gaps
    gap_cases = [
        r for r in metrics.case_audit_records
        if r.get("root_cause") == AuditRootCause.TRUE_KNOWLEDGE_GAP.value
    ]
    (output_dir / "true_knowledge_gaps.json").write_text(
        json.dumps({
            "count": metrics.true_knowledge_gap_count,
            "cases": gap_cases,
        }, indent=2, default=str)
    )

    # 8. Recommended next actions
    (output_dir / "recommended_next_actions.json").write_text(
        json.dumps({
            "primary_bottleneck": metrics.primary_bottleneck,
            "recommended_next_phase": metrics.recommended_next_phase,
            "recoverable_failure_rate": metrics.recoverable_failure_rate,
            "true_knowledge_gap_rate": metrics.true_knowledge_gap_rate,
        }, indent=2)
    )

    print(f"  Artifacts exported to: {output_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SupportGraph AI — Phase 10.1 Evidence Coverage Audit"
    )
    parser.add_argument(
        "--corpus-size",
        type=int,
        default=5000,
        help="Maximum number of historical conversations in audit corpus (default: 5000)",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Skip JSON artifact export",
    )
    args = parser.parse_args()

    print("\n" + "=" * 64)
    print("SUPPORTGRAPH AI — EVIDENCE COVERAGE AUDIT")
    print("Phase 10.1 | Evidence Coverage & Retrieval Recall Audit")
    print("=" * 64)

    # Pre-execution SHA-256 verification
    print("\n[1/9] Verifying golden dataset integrity (pre-execution)...")
    _verify_sha256("pre-execution")

    print(f"\n[2/9] Loading protected benchmark from {GOLDEN_CSV}...")

    print(f"\n[3/9] Building leakage-safe audit corpus (max {args.corpus_size:,} entries)...")

    corpus_builder = EvidenceCorpusBuilder(max_sample_size=args.corpus_size)

    print("\n[4/9] Running production pipeline on all benchmark records...")

    evaluator = EvidenceCoverageEvaluator(
        golden_path=GOLDEN_CSV,
        corpus_builder=corpus_builder,
    )

    t0 = time.time()
    metrics = evaluator.run_benchmark()
    elapsed = time.time() - t0

    print(f"\n[5/9] Evidence coverage audit complete ({elapsed:.1f}s).")

    print("\n[6/9] Root cause classification complete.")

    print("\n[7/9] Generating report...")
    print_results(metrics)

    if not args.no_json:
        print("[8/9] Exporting structured JSON artifacts...")
        export_artifacts(metrics, ARTIFACTS_DIR)

    print("\n[9/9] Verifying golden dataset integrity (post-execution)...")
    _verify_sha256("post-execution")

    if not metrics.dataset_immutability_pass:
        print("\n⚠️  GOLDEN DATASET MODIFIED — CRITICAL FAILURE")
        sys.exit(2)

    print("\n✓ Phase 10.1 audit complete. Golden dataset integrity preserved.\n")


if __name__ == "__main__":
    main()
