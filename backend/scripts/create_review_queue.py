"""
SupportGraph AI — Create Risk-Based Review Queue CLI Script (Phase 5.6)

Ingests AI-suggested golden set records and generates a prioritized review queue:
- Computes deterministic risk priority and scores (CRITICAL, HIGH, MEDIUM, LOW)
- Selects a deterministic Quality Control (QC) sample from low-priority records
- Orders pending records so highest-risk edge cases are reviewed first
- Strictly preserves scientific integrity: does not alter human ground truth.

Usage:
    # Generate prioritized review queue with default 15% QC sample:
    python -m backend.scripts.create_review_queue

    # Specify custom input/output and fixed QC sample size:
    python -m backend.scripts.create_review_queue \
        --input data/golden/golden_set_ai_suggestions.csv \
        --output data/golden/golden_set_review_queue.csv \
        --qc-size 15 \
        --seed 42
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import pandas as pd

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from app.core.logging import configure_logging
    from app.evaluation.review_prioritization import (
        DEFAULT_QC_SAMPLE_PERCENTAGE,
        QC_RANDOM_SEED,
        build_prioritized_review_queue,
    )
    from app.nlp.taxonomy_finalization import DEFAULT_CANDIDATE_TAXONOMY_PATH
except ModuleNotFoundError:
    from backend.app.core.logging import configure_logging  # type: ignore[no-redef]
    from backend.app.evaluation.review_prioritization import (  # type: ignore[no-redef]
        DEFAULT_QC_SAMPLE_PERCENTAGE,
        QC_RANDOM_SEED,
        build_prioritized_review_queue,
    )
    from backend.app.nlp.taxonomy_finalization import (  # type: ignore[no-redef]
        DEFAULT_CANDIDATE_TAXONOMY_PATH,
    )

logger = logging.getLogger("create_review_queue")

DEFAULT_INPUT_CSV = "data/golden/golden_set_ai_suggestions.csv"
DEFAULT_OUTPUT_CSV = "data/golden/golden_set_review_queue.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 5.6 Risk-Based Human Review Queue Generator for SupportGraph AI"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT_CSV,
        help=f"Input CSV with AI suggestions (default: {DEFAULT_INPUT_CSV})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Output prioritized review queue CSV (default: {DEFAULT_OUTPUT_CSV})",
    )
    parser.add_argument(
        "--taxonomy-path",
        type=str,
        default=DEFAULT_CANDIDATE_TAXONOMY_PATH,
        help=f"Path to candidate taxonomy JSON (default: {DEFAULT_CANDIDATE_TAXONOMY_PATH})",
    )
    parser.add_argument(
        "--human-review-path",
        type=str,
        default="data/golden/golden_set_human_review.csv",
        help="Path to existing human review CSV to preserve completed annotations (default: data/golden/golden_set_human_review.csv)",
    )
    parser.add_argument(
        "--qc-size",
        type=int,
        default=None,
        help="Explicit number of low-priority records to sample for QC",
    )
    parser.add_argument(
        "--qc-percent",
        type=float,
        default=DEFAULT_QC_SAMPLE_PERCENTAGE,
        help=f"Fraction of low-priority records to sample for QC (default: {DEFAULT_QC_SAMPLE_PERCENTAGE})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=QC_RANDOM_SEED,
        help=f"Fixed random seed for deterministic QC sampling (default: {QC_RANDOM_SEED})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_level = "DEBUG" if args.verbose else "INFO"
    configure_logging(log_level=log_level)

    input_path = Path(args.input)
    output_path = Path(args.output)
    taxonomy_path = Path(args.taxonomy_path)
    human_review_path = Path(args.human_review_path) if args.human_review_path else None

    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        return 1

    # Load input dataframe safely
    df_raw = pd.read_csv(input_path, dtype=str)
    # Sanitize text columns, prevent NaN string coercion
    for col in df_raw.columns:
        df_raw[col] = df_raw[col].fillna("").astype(object)
        df_raw[col] = df_raw[col].apply(
            lambda x: "" if str(x).strip().lower() in ("nan", "none", "null", "undefined") else str(x).strip()
        )

    logger.info("Loaded %d records from: %s", len(df_raw), input_path)

    # If existing human reviews exist, safely merge human annotations
    if human_review_path and human_review_path.exists():
        logger.info("Overlaying completed human review decisions from: %s", human_review_path)
        df_hr = pd.read_csv(human_review_path, dtype=str)
        for col in df_hr.columns:
            df_hr[col] = df_hr[col].fillna("").astype(object)
            df_hr[col] = df_hr[col].apply(
                lambda x: "" if str(x).strip().lower() in ("nan", "none", "null", "undefined") else str(x).strip()
            )
        # Overlay completed annotations
        for _, hr_row in df_hr.iterrows():
            gid = str(hr_row.get("golden_id", "")).strip()
            lbl = str(hr_row.get("annotation_label", "")).strip()
            stat = str(hr_row.get("annotation_status", "")).strip().lower()
            if lbl and stat not in ("", "pending", "pending_human_review"):
                matches = df_raw.index[df_raw["golden_id"] == gid].tolist()
                if matches:
                    m_idx = matches[0]
                    for col in ["annotation_label", "annotation_status", "annotator", "notes", "review_timestamp", "review_duration_seconds"]:
                        if col in df_hr.columns:
                            df_raw.at[m_idx, col] = hr_row.get(col, "")

    # Build prioritized queue
    queue_df, summary = build_prioritized_review_queue(
        df=df_raw,
        taxonomy_path=taxonomy_path,
        sample_size=args.qc_size,
        sample_percentage=args.qc_percent,
        seed=args.seed,
    )

    # Ensure output parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    queue_df.to_csv(output_path, index=False, quoting=csv.QUOTE_NONNUMERIC)

    # Print human-readable summary
    print("\n" + "=" * 60)
    print("PHASE 5.6 — RISK-BASED HUMAN REVIEW QUEUE")
    print("=" * 60)
    print(f"Total records:                         {summary['total_records']}")
    print(f"Already human reviewed:                {summary['already_human_reviewed']}")
    print(f"Pending human review:                  {summary['pending_human_review']}")
    print()
    print(f"CRITICAL priority:                     {summary['critical_priority']}")
    print(f"HIGH priority:                         {summary['high_priority']}")
    print(f"MEDIUM priority:                       {summary['medium_priority']}")
    print(f"LOW priority:                          {summary['low_priority']}")
    print()
    print(f"QC sampled low-priority records:       {summary['qc_sampled_low_priority']} (seed={summary['qc_random_seed']})")
    print()
    print("Human Review Recommendations:")
    print(f"  - Individual Review Required:        {summary.get('individual_review_required', 0)}")
    print(f"  - Mandatory QC Individual Review:    {summary.get('qc_review_required', 0)}")
    print(f"  - Safe for Grouped Review:           {summary.get('safe_for_group_review', 0)}")
    print()
    print(f"Records in review queue:               {summary['total_in_review_queue']}")
    print("=" * 60)
    print(f"Saved prioritized queue to: {output_path}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
