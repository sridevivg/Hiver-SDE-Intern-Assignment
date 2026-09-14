"""
SupportGraph AI — Sample Golden Set Candidates CLI Script (Phase 5)

Extracts a stratified, reproducible golden candidate dataset for human evaluation:
- Loads 20,000 intent discovery corpus and conversation context
- Maps messages to candidate intents based on empirical cluster evidence
- Enforces strict deduplication (unique tweet_id and unique normalized text)
- Stratifies across candidate intents (default target: 200 records)
- Leaves annotation_label strictly empty (no synthetic labels)
- Produces:
    1. data/golden/golden_set_candidates.csv
    2. data/golden/golden_set_annotation_template.csv
    3. data/golden/golden_sampling_manifest.json

Usage:
    python -m backend.scripts.sample_golden_set --target-size 200 --random-seed 42
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
    from app.evaluation.golden_sampler import (
        DEFAULT_CORPUS_PATH,
        DEFAULT_GOLDEN_DIR,
        GoldenSampleResult,
        run_golden_sampling,
    )
except ModuleNotFoundError:
    from backend.app.core.logging import configure_logging  # type: ignore[no-redef]
    from backend.app.evaluation.golden_sampler import (  # type: ignore[no-redef]
        DEFAULT_CORPUS_PATH,
        DEFAULT_GOLDEN_DIR,
        GoldenSampleResult,
        run_golden_sampling,
    )

logger = logging.getLogger("sample_golden_set")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 5 Golden Set Candidate Sampling for SupportGraph AI"
    )
    parser.add_argument(
        "--corpus-path",
        type=str,
        default=DEFAULT_CORPUS_PATH,
        help="Path to intent discovery corpus (default: data/interim/intent_discovery_corpus.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_GOLDEN_DIR,
        help="Directory to save golden evaluation artifacts (default: data/golden)",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=200,
        help="Target number of golden samples (default: 200, allowed: 150-250)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling (default: 42)",
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
    logger.info("PHASE 5 — STRATIFIED GOLDEN SET CANDIDATE SAMPLER")
    logger.info("=" * 60)

    if args.target_size < 150 or args.target_size > 250:
        logger.warning(
            f"Target size {args.target_size} is outside recommended range [150, 250]."
        )

    try:
        result: GoldenSampleResult = run_golden_sampling(
            corpus_path=args.corpus_path,
            output_dir=args.output_dir,
            target_size=args.target_size,
            random_seed=args.random_seed,
        )

        logger.info(f"Successfully sampled {result.sample_size} candidates (seed={args.random_seed}).")
        logger.info("-" * 60)
        logger.info(f"{'Candidate Intent':<35} | {'Count'}")
        logger.info("-" * 60)
        for intent, count in sorted(result.intent_distribution.items()):
            logger.info(f"{intent:<35} | {count}")
        logger.info("-" * 60)
        logger.info(f"Candidates file: {result.candidates_csv}")
        logger.info(f"Annotation template: {result.annotation_template_csv}")
        logger.info(f"Sampling manifest: {result.manifest_path}")
        logger.info("NOTE: All annotation_label values are strictly empty. Awaiting human labeling.")
        return 0

    except Exception as exc:
        logger.error(f"Failed to sample golden candidate set: {exc}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
