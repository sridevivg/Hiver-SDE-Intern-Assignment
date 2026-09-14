"""
SupportGraph AI — Analyze Annotation Patterns & Calibrate Taxonomy (Phase 5.9 CLI)

Executes read-only scientific analysis comparing baseline AI suggestions against
human ground-truth annotations across completed records.

Usage:
    python -m backend.scripts.analyze_annotation_patterns \
        --input data/golden/golden_set_human_review.csv \
        --output reports/annotation_pattern_analysis
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.app.core.logging import configure_logging, get_logger
from backend.app.evaluation.annotation_pattern_analysis import (
    export_reports_to_directory,
    generate_markdown_report,
    run_annotation_pattern_analysis,
)

logger = get_logger("analyze_annotation_patterns")

REQUIRED_COLUMNS: list[str] = [
    "golden_id",
    "customer_message",
    "model_suggested_label",
    "annotation_label",
    "annotation_status",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 5.9 Annotation Pattern Analysis & Taxonomy Calibration."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/golden/golden_set_human_review.csv",
        help="Path to golden set human review CSV.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/annotation_pattern_analysis",
        help="Output directory for generated JSON and Markdown reports.",
    )
    parser.add_argument(
        "--total-records",
        type=int,
        default=200,
        help="Total golden benchmark dataset size (default: 200).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debug output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_level = "DEBUG" if args.verbose else "INFO"
    configure_logging(log_level=log_level)

    input_path = Path(args.input)
    output_dir = Path(args.output)

    if not input_path.exists():
        logger.error("Input dataset file does not exist: %s", input_path)
        return 1

    logger.info("Loading golden dataset from: %s", input_path)
    df = pd.read_csv(input_path, dtype=str)

    # Validate required columns
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        logger.error("Dataset is missing required columns: %s", missing)
        return 1

    logger.info(
        "Loaded %d total records. Running Phase 5.9 pattern analysis...",
        len(df),
    )

    report = run_annotation_pattern_analysis(df, total_records=args.total_records)

    # Export structured artifacts
    created_files = export_reports_to_directory(report, output_dir)

    # Also link root phase report
    root_report_path = Path("reports/phase_5_9_annotation_pattern_analysis.md")
    root_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(root_report_path, "w", encoding="utf-8") as f:
        f.write(generate_markdown_report(report))
    created_files["phase_5_9_root_report_md"] = str(root_report_path)

    # Display clean summary
    m = report.metrics
    gds = report.general_device_support_audit

    print("\n" + "=" * 60)
    print("PHASE 5.9 — ANNOTATION PATTERN ANALYSIS & TAXONOMY AUDIT")
    print("=" * 60)
    print(f"Total Golden Records:            {m.total_golden_records}")
    print(f"Completed Human Reviews:         {m.completed_reviews} ({m.sample_coverage_pct:.1f}%)")
    print(f"Pending Human Reviews:           {m.pending_reviews}")
    print("-" * 60)
    print(f"Agreed AI Suggestions:           {m.agreed_count} ({m.observed_agreement_rate:.1%})")
    print(f"Overridden AI Suggestions:       {m.overridden_count} ({1.0 - m.observed_agreement_rate:.1%})")
    print(f"Marked 'unclear_needs_review':   {m.unclear_count}")
    print(f"95% Confidence Interval:         [{m.confidence_interval_95[0]:.1%}, {m.confidence_interval_95[1]:.1%}]")
    print("-" * 60)
    print("TOP AI -> HUMAN OVERRIDE PAIRS:")
    for cp in report.top_confusion_pairs[:5]:
        print(f"  - {cp.ai_suggested_label:<28} -> {cp.human_ground_truth_label:<32} : {cp.count} ({cp.percentage_of_all_overrides:.1f}%)")
    print("-" * 60)
    print("GENERAL_DEVICE_SUPPORT AUDIT:")
    print(f"  - AI Predictions:              {gds.ai_predicted_count}")
    print(f"  - Confirmed by Human:          {gds.human_confirmed_count} ({gds.confirmed_rate:.1%})")
    print(f"  - Overridden by Human:         {gds.human_overridden_count} ({gds.override_rate:.1%})")
    print(f"  - Verdict:                     {gds.audit_verdict}")
    print("-" * 60)
    print("REPORTS GENERATED:")
    for k, p in created_files.items():
        print(f"  - {k:<30} -> {p}")
    print("=" * 60 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
