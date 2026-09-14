"""
SupportGraph AI — Validate Golden Set CLI Script (Phase 5)

Validates golden set candidate and annotated datasets against strict criteria:
- Verifies schema and mandatory columns
- Ensures no duplicate tweet_ids or duplicate normalized messages
- Checks label validity against approved operational taxonomy
- Detects empty labels and calculates completion percentage
- Alerts on severe class imbalance
- Fails loudly with non-zero exit code if invalid

Usage:
    # Check in-progress or candidate template:
    python -m backend.scripts.validate_golden_set --input data/golden/golden_set_annotation_template.csv

    # Check completed dataset requiring 100% labeled:
    python -m backend.scripts.validate_golden_set --input data/golden/golden_set_annotated.csv --require-complete
"""
from __future__ import annotations

import argparse
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
    from app.evaluation.label_validation import ValidationReport, validate_golden_set
    from app.nlp.taxonomy_finalization import (
        DEFAULT_CANDIDATE_TAXONOMY_PATH,
        create_initial_candidate_taxonomy,
        load_candidate_taxonomy,
    )
except ModuleNotFoundError:
    from backend.app.core.logging import configure_logging  # type: ignore[no-redef]
    from backend.app.evaluation.label_validation import (  # type: ignore[no-redef]
        ValidationReport,
        validate_golden_set,
    )
    from backend.app.nlp.taxonomy_finalization import (  # type: ignore[no-redef]
        DEFAULT_CANDIDATE_TAXONOMY_PATH,
        create_initial_candidate_taxonomy,
        load_candidate_taxonomy,
    )

logger = logging.getLogger("validate_golden_set")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 5 Golden Set Validation Engine for SupportGraph AI"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the CSV dataset to validate",
    )
    parser.add_argument(
        "--taxonomy-path",
        type=str,
        default=DEFAULT_CANDIDATE_TAXONOMY_PATH,
        help="Path to taxonomy JSON candidate (default: data/interim/final_taxonomy_candidate.json)",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail if any record has an empty annotation_label (for completed/frozen datasets)",
    )
    parser.add_argument(
        "--disallow-unclear",
        action="store_true",
        help="Disallow 'unclear_needs_review' as an acceptable label",
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

    logger.info("=" * 60)
    logger.info("PHASE 5 — GOLDEN SET VALIDATION ENGINE")
    logger.info("=" * 60)

    in_path = Path(args.input)
    if not in_path.exists():
        logger.error(f"Input file does not exist: {in_path}")
        return 1

    # Load taxonomy labels
    tax_path = Path(args.taxonomy_path)
    if tax_path.exists():
        taxonomy = load_candidate_taxonomy(tax_path)
        allowed_labels = set(taxonomy.intent_names)
    else:
        logger.warning(f"Taxonomy file {tax_path} not found. Using default candidate taxonomy.")
        taxonomy = create_initial_candidate_taxonomy()
        allowed_labels = set(taxonomy.intent_names)

    logger.info(f"Loaded {len(allowed_labels)} approved taxonomy categories.")
    logger.info(f"Reading target file: {in_path}")
    df = pd.read_csv(in_path)

    report: ValidationReport = validate_golden_set(
        df=df,
        allowed_labels=allowed_labels,
        require_complete=args.require_complete,
        allow_unclear=not args.disallow_unclear,
    )

    logger.info("-" * 60)
    logger.info(f"Total Records:      {report.total_records}")
    comp = report.completion_stats
    logger.info(f"Annotated Records:  {comp.get('labeled', 0)} ({comp.get('percent_complete', 0.0)}%)")
    logger.info(f"Pending/Empty:      {comp.get('empty', 0)}")
    logger.info("-" * 60)

    if report.label_distribution:
        logger.info("Label Distribution:")
        for lbl, cnt in sorted(report.label_distribution.items()):
            logger.info(f"  {lbl:<35} : {cnt}")
        logger.info("-" * 60)

    if report.warnings:
        logger.warning(f"Validation generated {len(report.warnings)} warning(s):")
        for w in report.warnings:
            logger.warning(f"  [WARN] {w}")

    if not report.is_valid:
        logger.error(f"VALIDATION FAILED with {len(report.errors)} error(s):")
        for err in report.errors:
            logger.error(f"  [ERROR] {err}")
        return 1

    logger.info("VALIDATION PASSED: Dataset conforms to golden specification.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
