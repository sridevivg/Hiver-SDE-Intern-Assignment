"""
SupportGraph AI — Freeze Golden Set CLI Script (Phase 5)

Freezes a completed human-annotated dataset into an immutable versioned golden set:
- Enforces 100% annotation completion (strictly zero empty labels)
- Enforces taxonomy category validity and uniqueness
- Generates data/golden/golden_set_{version}.csv
- Calculates SHA256 checksum for auditability and tamper protection
- Generates data/golden/golden_set_{version}_manifest.json
- Blocks accidental overwriting unless --force-new-version is explicitly passed

Usage:
    python -m backend.scripts.freeze_golden_set \
        --input data/golden/golden_set_completed.csv \
        --version v1
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from app.core.logging import configure_logging
    from app.evaluation.golden_freeze import freeze_golden_dataset
    from app.evaluation.label_validation import GoldenValidationError
    from app.nlp.taxonomy_finalization import DEFAULT_CANDIDATE_TAXONOMY_PATH
except ModuleNotFoundError:
    from backend.app.core.logging import configure_logging  # type: ignore[no-redef]
    from backend.app.evaluation.golden_freeze import freeze_golden_dataset  # type: ignore[no-redef]
    from backend.app.evaluation.label_validation import GoldenValidationError  # type: ignore[no-redef]
    from backend.app.nlp.taxonomy_finalization import (  # type: ignore[no-redef]
        DEFAULT_CANDIDATE_TAXONOMY_PATH,
    )

logger = logging.getLogger("freeze_golden_set")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 5 Golden Set Freeze Workflow for SupportGraph AI"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the completed, human-annotated golden CSV dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/golden",
        help="Directory to store the frozen golden set and manifest (default: data/golden)",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1",
        help="Version identifier for the frozen golden set (e.g. v1, v2; default: v1)",
    )
    parser.add_argument(
        "--taxonomy-path",
        type=str,
        default=DEFAULT_CANDIDATE_TAXONOMY_PATH,
        help="Path to candidate taxonomy JSON file (default: data/interim/final_taxonomy_candidate.json)",
    )
    parser.add_argument(
        "--force-new-version",
        action="store_true",
        help="Explicitly allow overwriting an already existing frozen version file",
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
    logger.info("PHASE 5 — GOLDEN SET FREEZE ENGINE")
    logger.info("=" * 60)

    try:
        csv_path, manifest_path, manifest = freeze_golden_dataset(
            input_csv_path=args.input,
            output_dir=args.output_dir,
            version=args.version,
            taxonomy_path=args.taxonomy_path,
            force_new_version=args.force_new_version,
        )

        logger.info(f"GOLDEN SET SUCCESSFULLY FROZEN: Version {manifest['version']}")
        logger.info(f"Row count:         {manifest['row_count']}")
        logger.info(f"SHA256 Checksum:   {manifest['sha256_checksum']}")
        logger.info(f"Frozen CSV:        {csv_path}")
        logger.info(f"Frozen Manifest:   {manifest_path}")
        logger.info("-" * 60)
        logger.info("Intent Distribution:")
        for intent, cnt in sorted(manifest["intent_distribution"].items()):
            logger.info(f"  {intent:<35} : {cnt}")
        logger.info("-" * 60)
        logger.info("Immutable golden evaluation release successfully published.")
        return 0

    except FileExistsError as exc:
        logger.error(f"Freeze blocked (Immutability Protection): {exc}")
        return 1
    except GoldenValidationError as exc:
        logger.error(f"Freeze blocked (Validation Failed): {exc}")
        return 1
    except Exception as exc:
        logger.error(f"Unexpected error during golden set freeze: {exc}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
