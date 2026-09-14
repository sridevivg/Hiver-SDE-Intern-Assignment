"""
SupportGraph AI — Audit Intent Taxonomy CLI Script (Phase 4.5)

Audits Phase 4 candidate clusters to provide empirical evidence for human review:
- Validates Phase 3 & 4 inputs
- Extracts 20 centroid-nearest, 20 random, 10 diverse examples per cluster
- Audits cluster coherence, boundary overlap, entities, and historical event bias
- Audits historical AppleSupport resolution patterns (first brand reply action distribution)
- Identifies merge and split candidate clusters
- Generates human review packet in data/interim/

Usage:
    python -m backend.scripts.audit_intent_taxonomy \
        --random-seed 42 \
        --centroid-examples 20 \
        --random-examples 20 \
        --diverse-examples 10
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
    from app.nlp.taxonomy_audit import TaxonomyAuditResult, run_taxonomy_audit
except ModuleNotFoundError:
    from backend.app.core.logging import configure_logging  # type: ignore[no-redef]
    from backend.app.nlp.taxonomy_audit import TaxonomyAuditResult, run_taxonomy_audit  # type: ignore[no-redef]

logger = logging.getLogger("audit_intent_taxonomy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 4.5 Human Taxonomy Review and Cluster Audit for SupportGraph AI"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling (default: 42)",
    )
    parser.add_argument(
        "--centroid-examples",
        type=int,
        default=20,
        help="Number of centroid-nearest examples per cluster (default: 20)",
    )
    parser.add_argument(
        "--random-examples",
        type=int,
        default=20,
        help="Number of random representative examples per cluster (default: 20)",
    )
    parser.add_argument(
        "--diverse-examples",
        type=int,
        default=10,
        help="Number of max-min diverse examples per cluster (default: 10)",
    )
    parser.add_argument(
        "--corpus-path",
        type=Path,
        default=REPO_ROOT / "data" / "interim" / "intent_discovery_corpus.csv",
        help="Path to intent_discovery_corpus.csv",
    )
    parser.add_argument(
        "--review-path",
        type=Path,
        default=REPO_ROOT / "data" / "interim" / "intent_cluster_review.csv",
        help="Path to intent_cluster_review.csv",
    )
    parser.add_argument(
        "--messages-path",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "conversation_messages.parquet",
        help="Path to conversation_messages.parquet",
    )
    parser.add_argument(
        "--quality-path",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "conversation_quality_summary.csv",
        help="Path to conversation_quality_summary.csv",
    )
    parser.add_argument(
        "--taxonomy-path",
        type=Path,
        default=REPO_ROOT / "data" / "interim" / "provisional_intent_taxonomy.json",
        help="Path to provisional_intent_taxonomy.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "interim",
        help="Directory to save audit artifacts",
    )
    parser.add_argument(
        "--features-cache-path",
        type=Path,
        default=REPO_ROOT / "data" / "interim" / "intent_features_pca.npy",
        help="Path to cached 50-d PCA embeddings (optional, accelerates audit)",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    logger.info("=" * 70)
    logger.info("SUPPORTGRAPH AI — PHASE 4.5: TAXONOMY AUDIT & HUMAN REVIEW")
    logger.info("=" * 70)
    logger.info("Corpus path:        %s", args.corpus_path)
    logger.info("Review path:        %s", args.review_path)
    logger.info("Messages path:      %s", args.messages_path)
    logger.info("Quality path:       %s", args.quality_path)
    logger.info("Taxonomy path:      %s", args.taxonomy_path)
    logger.info("Output dir:         %s", args.output_dir)
    logger.info("Random seed:        %d", args.random_seed)
    logger.info("Centroid examples:  %d", args.centroid_examples)
    logger.info("Random examples:    %d", args.random_examples)
    logger.info("Diverse examples:   %d", args.diverse_examples)

    result = run_taxonomy_audit(
        corpus_path=args.corpus_path,
        review_path=args.review_path,
        messages_path=args.messages_path,
        quality_path=args.quality_path,
        taxonomy_path=args.taxonomy_path,
        output_dir=args.output_dir,
        random_seed=args.random_seed,
        n_centroid=args.centroid_examples,
        n_random=args.random_examples,
        n_diverse=args.diverse_examples,
        features_cache_path=args.features_cache_path,
    )

    # Print summary report
    print("\n" + "=" * 80)
    print("PHASE 4.5 CLUSTER AUDIT SUMMARY (K = 8)")
    print("=" * 80)
    print(f"{'ID':<3} {'Size':<6} {'%':<6} {'Generalization':<16} {'Action Recommendation':<22} {'Current Proposed Label'}")
    print("-" * 80)
    for a in result.cluster_audits:
        print(f"{a.cluster_id:<3} {a.cluster_size:<6} {a.percentage:<6.1f} {a.historical_events.generalization_status:<16} {a.recommended_human_action:<22} {a.current_proposed_label}")

    print("\n" + "-" * 80)
    print("HISTORICAL EVENT DETECTION FINDINGS:")
    print("-" * 80)
    for a in result.cluster_audits:
        print(f"Cluster {a.cluster_id} [{a.historical_events.generalization_status}]: {a.historical_events.explanation}")

    print("\n" + "-" * 80)
    print("RESOLUTION PATTERN AUDIT (FIRST APPLE RESPONSE):")
    print("-" * 80)
    for a in result.cluster_audits:
        print(f"Cluster {a.cluster_id} ({a.current_proposed_label}):")
        print(f"  {a.resolutions.summary_str}")
        if a.resolutions.common_openings:
            openings = ", ".join([f'"{p}" ({c})' for p, c in a.resolutions.common_openings[:2]])
            print(f"  Top Openings: {openings}")

    print("\n" + "-" * 80)
    print("MERGE CANDIDATE PAIRS (REVIEW ONLY — NOT AUTOMATICALLY MERGED):")
    print("-" * 80)
    if result.merge_candidates:
        for m in result.merge_candidates:
            print(f"Pair ({m.cluster_a}, {m.cluster_b}) | Centroid Sim: {m.centroid_similarity:.4f} | Recommendation: {m.recommendation}")
            print(f"  Reason: {m.reason}")
    else:
        print("No strong merge candidates detected.")

    print("\n" + "-" * 80)
    print("SPLIT CANDIDATE CLUSTERS (REVIEW ONLY — NOT AUTOMATICALLY SPLIT):")
    print("-" * 80)
    if result.split_candidates:
        for s in result.split_candidates:
            print(f"Cluster {s.cluster_id} | Signals: {s.coherence_signals}")
            print(f"  Subtopics: {s.possible_subtopics}")
            print(f"  Reason: {s.reason_for_review}")
    else:
        print("No severe split candidates detected.")

    print("\n" + "=" * 80)
    print("ALL AUDIT ARTIFACTS SAVED:")
    print("  - data/interim/cluster_merge_candidates.csv")
    print("  - data/interim/cluster_split_candidates.csv")
    print("  - data/interim/intent_taxonomy_human_review.csv")
    print("Next step: Follow docs/final_intent_taxonomy_guidelines.md for expert review.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
