"""
SupportGraph AI — Finalize Candidate Taxonomy CLI Script (Phase 5)

Generates the evidence-based candidate operational intent taxonomy:
- Synthesizes findings from Phase 4 and Phase 4.5 audit artifacts
- Implements empirical merge and split decisions (separating account/billing, hardware audio/display)
- Generates structured JSON definition at data/interim/final_taxonomy_candidate.json
- Sets status = 'candidate_human_review_required'

Usage:
    python -m backend.scripts.finalize_taxonomy
    python -m backend.scripts.finalize_taxonomy --output data/interim/final_taxonomy_candidate.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from app.core.logging import configure_logging
    from app.nlp.taxonomy_finalization import (
        DEFAULT_CANDIDATE_TAXONOMY_PATH,
        TaxonomyCandidate,
        finalize_candidate_taxonomy,
    )
except ModuleNotFoundError:
    from backend.app.core.logging import configure_logging  # type: ignore[no-redef]
    from backend.app.nlp.taxonomy_finalization import (  # type: ignore[no-redef]
        DEFAULT_CANDIDATE_TAXONOMY_PATH,
        TaxonomyCandidate,
        finalize_candidate_taxonomy,
    )

logger = logging.getLogger("finalize_taxonomy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 5 Candidate Taxonomy Finalization for SupportGraph AI"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_CANDIDATE_TAXONOMY_PATH,
        help="Path to output taxonomy JSON file (default: data/interim/final_taxonomy_candidate.json)",
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
    logger.info("PHASE 5 — CANDIDATE TAXONOMY FINALIZATION ENGINE")
    logger.info("=" * 60)

    try:
        out_path = Path(args.output)
        taxonomy = finalize_candidate_taxonomy(output_json_path=out_path)

        logger.info(f"Taxonomy Candidate Generated: Version {taxonomy.version}")
        logger.info(f"Review Status: {taxonomy.status}")
        logger.info(f"Total Operational Intents: {len(taxonomy.intents)}")
        logger.info("-" * 60)
        logger.info(f"{'Intent Name':<35} | {'Source Clusters':<18} | {'Status'}")
        logger.info("-" * 60)
        for intent in taxonomy.intents:
            clusters_str = ", ".join(str(c) for c in intent.source_clusters)
            logger.info(
                f"{intent.intent_name:<35} | {clusters_str:<18} | {intent.review_status}"
            )
        logger.info("-" * 60)
        logger.info(f"JSON artifact saved: {out_path}")
        logger.info("Candidate taxonomy successfully generated and ready for human review.")
        return 0

    except Exception as exc:
        logger.error(f"Failed to finalize candidate taxonomy: {exc}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
