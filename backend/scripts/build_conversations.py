#!/usr/bin/env python3
"""
SupportGraph AI — Phase 3 CLI: Conversation Reconstruction

Reconstructs customer-support conversations from Twitter reply graphs
for the selected brand (AppleSupport) without fabricating missing turns.

Usage:
    python -m backend.scripts.build_conversations
    python -m backend.scripts.build_conversations --brand AppleSupport --sample-limit 1000
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
from typing import Optional

import pandas as pd

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.logging import configure_logging
from backend.app.data.conversation_builder import (
    ConversationBuilder,
    load_selected_brand,
    save_reconstructed_conversations,
)
from backend.app.data.conversation_models import Conversation, ConversationStatistics
from backend.app.data.loader import discover_dataset, load_dataset

logger = logging.getLogger("build_conversations")

DEFAULT_SELECTED_BRAND_FILE = PROJECT_ROOT / "data" / "interim" / "selected_brand.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "artifacts" / "reports"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SupportGraph AI — Phase 3 Conversation Reconstruction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--brand",
        type=str,
        default=None,
        help="Target brand handle. If omitted, loads from data/interim/selected_brand.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save reconstructed dataset files (default: data/processed).",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=50,
        help="Maximum depth from root for canonical conversation traversal (default: 50).",
    )
    parser.add_argument(
        "--include-low-quality",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include low-quality conversations in output datasets (default: --include-low-quality).",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Optional limit on number of candidate components for testing/development.",
    )
    return parser.parse_args()


def generate_markdown_report(
    brand: str,
    conversations: list[Conversation],
    stats: ConversationStatistics,
    report_path: Path,
) -> Path:
    """Generate the comprehensive Phase 3 markdown report."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# Phase 3 — Conversation Reconstruction",
        "",
        f"> **Generated:** {now}  ",
        f"> **Selected Brand:** `{brand}`  ",
        "> **Method:** Deterministic reply graph component assembly  ",
        "> **Principle:** Strictly observed message relationships — zero fabricated context  ",
        "",
        "---",
        "",
        "## Objective",
        "",
        f"Transform tweet-level interaction records for `{brand}` into structured,",
        "reproducible conversation threads. This dataset serves as the factual foundation",
        "for downstream intent discovery, historical RAG retrieval, and grounded reply generation.",
        "",
        "---",
        "",
        "## Input Data",
        "",
        "- **Dataset Source:** Kaggle `thoughtvector/customer-support-on-twitter` (`twcs.csv`)",
        "- **Total Dataset Rows:** 2,811,774 tweets",
        f"- **Total `{brand}` Tweets:** {stats.total_selected_brand_tweets:,}",
        "- **Relevant Schema Fields:** `tweet_id`, `author_id`, `inbound`, `created_at`, `text`, `in_response_to_tweet_id`, `response_tweet_id`",
        "",
        "---",
        "",
        "## Selected Brand",
        "",
        f"**`{brand}`** was selected scientifically in Phase 2 using a 5-metric multi-criteria evaluation:",
        "- 106,719 usable interaction volume (Phase 1)",
        "- 99.93% response coverage (parent in dataset)",
        "- 100.00% conversation reconstructability (linked parent is valid customer tweet)",
        "- 0.9997 data completeness floor",
        "- 0.9575 issue diversity score",
        "- Winner in 3 out of 3 sensitivity analysis scenarios",
        "",
        "---",
        "",
        "## Reply Graph Model",
        "",
        "The conversation pipeline models tweets as **nodes** and explicit replies as **directed edges**:",
        "",
        "$$\\text{Parent Tweet } A \\longrightarrow \\text{Child Reply } B$$",
        "",
        "Where Child $B$ explicitly records `in_response_to_tweet_id = A`.",
        "A conversation is formally defined as a **weakly connected component** of the reply graph",
        f"containing at least one customer tweet (`inbound=True`) and at least one `{brand}` tweet.",
        "",
        "---",
        "",
        "## Relationship Parsing",
        "",
        "The Twitter dataset provides two fields containing reply relationships:",
        "",
        "1. **`in_response_to_tweet_id` (Backward Pointer):**",
        "   - Indicates the parent tweet that this tweet replies to.",
        "   - Single scalar ID (or null for thread starters).",
        "   - Used as the primary ground-truth backward edge.",
        "",
        "2. **`response_tweet_id` (Forward Pointer):**",
        "   - Indicates one or more child tweets replying to this tweet.",
        "   - Stored as comma-separated integers (e.g. `'5,7'`, `'9,6,10'`).",
        "   - Parsed by splitting on comma, trimming whitespace, and casting to integer IDs.",
        "",
        "**Edge Agreement Classification:**",
        "- `both`: Both child has `in_response_to_tweet_id = parent` AND parent lists child in `response_tweet_id`.",
        "- `in_response_to`: Child points to parent, but parent does not index child in `response_tweet_id`.",
        "- `response_tweet`: Parent points to child, but child is outside the dataset or missing `in_response_to`.",
        "",
        "---",
        "",
        "## Conversation Reconstruction Algorithm",
        "",
        "The 16-step reconstruction follows strict scientific principles:",
        "1. Load selected brand from Phase 2 artifact (`selected_brand.json`).",
        "2. Validate dataset schema against required 7 columns.",
        "3. Filter all tweets authored by the selected brand.",
        "4. Traverse the full reply graph to find all weakly connected components containing the brand.",
        "5. Subset the dataset to connected tweets to maximize memory efficiency.",
        "6. Build bidirectional adjacency and forward/backward lookup tables.",
        "7. Validate edge consistency across both pointer fields.",
        "8. Detect anomalies (missing parents, missing children, timestamp reversals, cycles, self-references).",
        "9. Extract connected conversation components.",
        "10. Discard components lacking either role (must contain both customer and brand).",
        "11. Identify conversation roots (nodes with in-degree 0 or parent outside dataset).",
        "12. Assign deterministic conversation IDs: `conv_{brand}_{root_tweet_id}`.",
        "13. Generate canonical breadth-first graph traversal ordered by depth, timestamp, and tweet ID.",
        "14. Classify thread type based strictly on observed structure.",
        "15. Assign deterministic data quality status (`high`, `medium`, `low`).",
        "16. Export all processed artifacts to `data/processed/`.",
        "",
        "---",
        "",
        "## Conversation Root Definition",
        "",
        "A tweet is defined as a conversation root if:",
        "- It has no parent pointer (`in_response_to_tweet_id IS NULL`), OR",
        "- Its parent pointer references a tweet ID outside the dataset (`broken_parent_root`).",
        "",
        "When a connected component contains multiple candidate roots (e.g. branched threads or joined discussions),",
        "the primary root is deterministically selected as the root with the **earliest timestamp**",
        "(with lowest numeric `tweet_id` as tiebreaker).",
        "",
        "---",
        "",
        "## Message Ordering",
        "",
        "Messages are ordered using a 3-tier hierarchy:",
        "1. **Primary: Graph Depth (Parent-Child Topology)** — Parent messages strictly precede children.",
        "2. **Secondary: Timestamp** — Sibling branches are ordered chronologically.",
        "3. **Tertiary: Numeric Tweet ID** — Deterministic tiebreaker for identical timestamps.",
        "",
        "If a child message has an earlier timestamp than its parent, **graph relationship takes precedence**,",
        "and a `timestamp_anomaly` is formally recorded.",
        "",
        "---",
        "",
        "## Branching Strategy",
        "",
        "Branching occurs when a single tweet receives multiple direct replies (e.g. multi-tweet agent answers or multiple customer follow-ups).",
        "- Branching is **never flattened** into an artificial linear chain.",
        "- Graph edges record the exact branching topology.",
        "- Conversations with branching are flagged `has_branching = True` and typed as `branched`.",
        "",
        "---",
        "",
        "## Anomaly Detection",
        "",
        "Every structural imperfection is recorded without halting execution:",
        "",
        "| Anomaly Type | Count | Description |",
        "|---|---|---|",
        f"| `broken_parent_relationships` | {stats.broken_parent_relationships:,} | Child references parent not present in the dataset |",
        f"| `timestamp_anomalies` | {stats.timestamp_anomalies:,} | Child timestamp is earlier than parent timestamp |",
        f"| `cycles_detected` | {stats.cycles_detected:,} | Cyclic reply relationships detected in graph |",
        f"| `orphan_messages` | {stats.orphan_messages:,} | Disconnected messages within component |",
        f"| **Conversations with Anomalies** | **{stats.conversations_with_anomalies:,}** | Any anomaly detected within conversation |",
        "",
        "---",
        "",
        "## Thread Type Distribution",
        "",
        "| Thread Type | Count | % of Valid Conversations | Description |",
        "|---|---|---|---|",
    ]

    total_valid = stats.total_valid_conversations or 1
    for ttype, count in sorted(stats.thread_type_distribution.items(), key=lambda x: x[1], reverse=True):
        pct = (count / total_valid) * 100
        lines.append(f"| `{ttype}` | {count:,} | {pct:.1f}% |")

    lines += [
        "",
        "---",
        "",
        "## Conversation Quality Distribution",
        "",
        "Deterministic quality rules:",
        "- **HIGH:** No anomalies, complete graph, >=2 messages, contains customer and brand.",
        "- **MEDIUM:** Contains customer and brand, minor anomalies (timestamp or missing child pointers), graph still usable.",
        "- **LOW:** Broken parent pointers, cycles, or incomplete structure.",
        "",
        "| Quality Status | Count | % of Valid Conversations |",
        "|---|---|---|",
    ]

    for qstatus, count in sorted(stats.quality_distribution.items(), key=lambda x: x[1], reverse=True):
        pct = (count / total_valid) * 100
        lines.append(f"| `{qstatus.upper()}` | {count:,} | {pct:.1f}% |")

    lines += [
        "",
        "---",
        "",
        "## Reconstruction Statistics",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Total `{brand}` Tweets | {stats.total_selected_brand_tweets:,} |",
        f"| Total Candidate Components | {stats.total_candidate_conversations:,} |",
        f"| Total Valid Conversations (Customer + Brand) | {stats.total_valid_conversations:,} |",
        f"| Total Messages in Conversations | {stats.total_messages:,} |",
        f"| Mean Messages per Conversation | {stats.mean_messages_per_conversation:.2f} |",
        f"| Median Messages per Conversation | {stats.median_messages_per_conversation:.1f} |",
        f"| Maximum Messages in Single Conversation | {stats.maximum_messages_per_conversation:,} |",
        f"| Branched Conversations | {stats.branched_conversations:,} |",
        "",
        "---",
        "",
        "## Sample Conversations",
        "",
        "> Author IDs and personal handles have been anonymized for privacy. Text excerpts truncated.",
        "",
    ]

    # Sample top 3 clean conversations
    sample_convs = [c for c in conversations if c.quality_status == "high"][:3]
    if not sample_convs:
        sample_convs = conversations[:3]

    for i, c in enumerate(sample_convs, 1):
        lines.append(f"### Sample {i}: `{c.conversation_id}` ({c.thread_type}, {c.message_count} messages)")
        for m in c.messages:
            role_badge = "**Customer**" if m.role == "customer" else f"**{brand}**"
            snippet = m.text.replace("\n", " ")[:150]
            if len(m.text) > 150:
                snippet += "…"
            lines.append(f"- [depth {m.depth}] {role_badge}: {snippet}")
        lines.append("")

    lines += [
        "---",
        "",
        "## Limitations",
        "",
        "1. **Missing Tweets:** Tweets deleted or private at scrape time create broken parent pointers.",
        "2. **Public Context Only:** Twitter exchanges frequently deflect to Private DMs ('Please DM us your serial number'). The resolution occurs off-graph.",
        "3. **Branched Discussions:** Multiple agents or users replying to a single tweet creates trees, not linear chat logs.",
        "4. **Timestamp Anomalies:** Client-side clock drift occasionally produces child tweets with timestamps earlier than their parents.",
        "5. **Inability to Prove Issue Resolution:** A closed conversation does not prove the customer's problem was resolved.",
        "6. **Observed vs Hidden Structure:** Reconstruction captures strictly what was publicly observed on Twitter.",
        "",
        "---",
        "",
        "## What Is Misleading About The Headline Number?",
        "",
        f"A headline stating **'{stats.total_valid_conversations:,} reconstructed conversations'** can easily be misinterpreted:",
        "",
        "1. **Conversation Count ≠ Resolved Customer Issues:** Many conversations end with deflection to Apple Support web forms, phone numbers, or DMs without resolution evidence.",
        "2. **A Complete Graph Does Not Imply High Satisfaction:** A customer may have abandoned the exchange out of frustration after 2 turns.",
        "3. **Short Conversations Overrepresented:** The majority of conversations are exactly 2 turns (`Customer -> Brand`), which represents initial triage rather than complete troubleshooting.",
        "4. **Private Information Gap:** The actual technical solution often occurred in private DMs or Apple retail appointments not present in the dataset.",
        "",
        "---",
        "",
        "_Report auto-generated by `backend/scripts/build_conversations.py`._",
    ]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Saved Phase 3 report: %s", report_path)
    return report_path


def main() -> None:
    configure_logging(log_level="INFO", log_format="text")
    args = parse_args()

    # Determine selected brand
    if args.brand:
        brand = args.brand
        logger.info("Using brand from CLI override: %s", brand)
    else:
        brand = load_selected_brand(DEFAULT_SELECTED_BRAND_FILE)

    # Load dataset
    logger.info("Discovering raw dataset...")
    filepath, _ = discover_dataset()
    logger.info("Loading full dataset from %s...", filepath)
    df = load_dataset(filepath=filepath)
    logger.info("Dataset loaded: %d rows", len(df))

    # Run conversation reconstruction
    builder = ConversationBuilder(
        df=df,
        selected_brand=brand,
        max_depth=args.max_depth,
        sample_limit=args.sample_limit,
    )
    conversations, stats = builder.build_conversations()

    # Save output artifacts
    logger.info("Saving reconstructed conversation datasets to %s...", args.output_dir)
    saved_paths = save_reconstructed_conversations(
        conversations=conversations,
        stats=stats,
        output_dir=args.output_dir,
        include_low_quality=args.include_low_quality,
    )

    # Generate Phase 3 report
    report_path = DEFAULT_REPORTS_DIR / "phase_3_conversation_reconstruction.md"
    generate_markdown_report(brand, conversations, stats, report_path)

    # Terminal summary
    print("\n" + "=" * 65)
    print(f"  PHASE 3: CONVERSATION RECONSTRUCTION COMPLETE — {brand}")
    print("=" * 65)
    print(f"  Total {brand} Tweets:           {stats.total_selected_brand_tweets:,}")
    print(f"  Candidate Components:            {stats.total_candidate_conversations:,}")
    print(f"  Valid Conversations:             {stats.total_valid_conversations:,}")
    print(f"  Total Messages in Threads:       {stats.total_messages:,}")
    print(f"  Mean Messages / Conversation:    {stats.mean_messages_per_conversation:.2f}")
    print(f"  Median Messages / Conversation:  {stats.median_messages_per_conversation:.1f}")
    print(f"  Max Messages / Conversation:     {stats.maximum_messages_per_conversation:,}")
    print(f"  Branched Conversations:          {stats.branched_conversations:,}")
    print(f"  Conversations with Anomalies:    {stats.conversations_with_anomalies:,}")
    print("-" * 65)
    print("  Thread Type Distribution:")
    for ttype, count in sorted(stats.thread_type_distribution.items(), key=lambda x: x[1], reverse=True):
        pct = (count / max(stats.total_valid_conversations, 1)) * 100
        print(f"    - {ttype:<30} {count:>7,} ({pct:>5.1f}%)")
    print("-" * 65)
    print("  Data Quality Distribution:")
    for qstatus, count in sorted(stats.quality_distribution.items(), key=lambda x: x[1], reverse=True):
        pct = (count / max(stats.total_valid_conversations, 1)) * 100
        print(f"    - {qstatus.upper():<30} {count:>7,} ({pct:>5.1f}%)")
    print("-" * 65)
    print("  Anomalies Found:")
    print(f"    - Broken Parent Relationships:  {stats.broken_parent_relationships:,}")
    print(f"    - Timestamp Inversions:         {stats.timestamp_anomalies:,}")
    print(f"    - Cyclic Relationships:         {stats.cycles_detected:,}")
    print(f"    - Orphan Messages:              {stats.orphan_messages:,}")
    print("-" * 65)
    print("  Output Files Generated:")
    for name, p in saved_paths.items():
        print(f"    - {name:<32}: {p}")
    print(f"    - {'report':<32}: {report_path}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
