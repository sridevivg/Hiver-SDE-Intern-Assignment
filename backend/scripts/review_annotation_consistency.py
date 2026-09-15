"""
SupportGraph AI — Human-Guided Annotation Quality Review (Phase 5.11)

Structured CLI workflow for adjudicating flagged semantic consistency conflicts,
evaluating category boundary validity, and capturing human decisions into an
isolated, append-only review log WITHOUT modifying the golden dataset.

Guiding Principles:
1. Strict Immutability: Golden dataset remains 100% byte-for-byte unchanged.
2. Read-Only Consistency Input: Consumes candidates from Phase 5.10.
3. Explicit Human Decision: Captures distinct boundary adjudications:
   - KEEP_BOTH_LABELS / VALID_TAXONOMY_BOUNDARY
   - RECONSIDER_RECORD_A / RECONSIDER_RECORD_B
   - FLAG_TAXONOMY_GUIDELINE_UPDATE
4. Append-Only Persistence: Writes to data/golden/annotation_consistency_human_review.csv.
"""
from __future__ import annotations
import collections
import argparse
import csv
import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from app.core.logging import get_logger
    from app.evaluation.annotation_consistency import (
        AnnotationConsistencyAnalyzer,
        calculate_file_sha256,
        extract_completed_reviews,
    )
except ModuleNotFoundError:
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]
    from backend.app.evaluation.annotation_consistency import (  # type: ignore[no-redef]
        AnnotationConsistencyAnalyzer,
        calculate_file_sha256,
        extract_completed_reviews,
    )

logger = get_logger(__name__)

DEFAULT_GOLDEN_PATH = REPO_ROOT / "data" / "golden" / "golden_set_human_review.csv"
DEFAULT_CANDIDATES_PATH = REPO_ROOT / "reports" / "annotation_consistency" / "cross_label_candidates.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "data" / "golden" / "annotation_consistency_human_review.csv"

REVIEW_CSV_COLUMNS = [
    "conflict_id",
    "record_a_golden_id",
    "record_b_golden_id",
    "record_a_label",
    "record_b_label",
    "priority",
    "similarity_score",
    "conflict_type",
    "human_decision",
    "reviewer",
    "timestamp",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SupportGraph AI — Phase 5.11 Human-Guided Annotation Quality & Boundary Review",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_GOLDEN_PATH),
        help="Path to golden human review dataset CSV (read-only)",
    )
    parser.add_argument(
        "--candidates",
        type=str,
        default=str(DEFAULT_CANDIDATES_PATH),
        help="Path to Phase 5.10 cross-label candidates JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path to append-only consistency human review CSV",
    )
    parser.add_argument(
        "--priority",
        type=str,
        choices=["high", "medium", "low", "all"],
        default="high",
        help="Priority tier of consistency conflicts to review",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of conflict pairs to review in this session",
    )
    parser.add_argument(
        "--annotator",
        type=str,
        default="sridevi",
        help="Human reviewer username / identifier",
    )
    return parser.parse_args()


def load_existing_decisions(output_path: Path) -> Set[str]:
    """Load set of already-reviewed conflict IDs to avoid redundant re-prompting."""
    if not output_path.exists():
        return set()

    reviewed_ids: Set[str] = set()
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cid = row.get("conflict_id", "").strip()
                if cid:
                    reviewed_ids.add(cid)
    except Exception as e:
        logger.warning("Error loading existing consistency review decisions: %s", e)
    return reviewed_ids


def append_decision_to_csv(output_path: Path, decision_row: Dict[str, Any]) -> None:
    """Safely append a single human consistency adjudication decision to the review CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists() and output_path.stat().st_size > 0

    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(decision_row)


def load_candidates_with_fallback(
    candidates_path: Path,
    golden_df: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """Load candidate pairs from JSON, or dynamically compute them if missing."""
    if candidates_path.exists():
        try:
            with open(candidates_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception as e:
            logger.warning("Failed to load candidates JSON: %s. Re-running analyzer in memory.", e)

    analyzer = AnnotationConsistencyAnalyzer()
    completed_df = extract_completed_reviews(golden_df)
    sim_matrix, _ = analyzer.compute_pairwise_similarities(completed_df)
    cands = analyzer.extract_consistency_candidates(completed_df, sim_matrix)
    return [c.to_dict() for c in cands]


def main() -> int:
    args = parse_args()
    golden_path = Path(args.input)
    candidates_path = Path(args.candidates)
    output_path = Path(args.output)

    print("=" * 80)
    print("SUPPORTGRAPH AI — PHASE 5.11 ANNOTATION QUALITY & BOUNDARY REVIEW")
    print("=" * 80)

    # 1. Strict Immutability Check — Pre-run
    if not golden_path.exists():
        print(f"ERROR: Golden dataset not found at: {golden_path}")
        return 1

    sha_pre = calculate_file_sha256(golden_path)
    print(f"\n[1/4] Golden Dataset Verified:")
    print(f"      File:    {golden_path}")
    print(f"      SHA-256: {sha_pre}")

    golden_df = pd.read_csv(golden_path, dtype=str)
    completed_df = extract_completed_reviews(golden_df)
    golden_lookup = {
        str(row["golden_id"]): row
        for _, row in golden_df.iterrows()
    }

    print(f"\n[2/4] Review Queue Configuration:")
    print(f"      Target Priority:     {args.priority.upper()}")
    print(f"      Target Batch Size:   {args.batch_size}")
    print(f"      Human Reviewer:      {args.annotator}")
    print(f"      Completed Ground Truth: {len(completed_df)} / {len(golden_df)} records")

    # 2. Load Candidates & Filter by Priority & Prior Reviews
    raw_candidates = load_candidates_with_fallback(candidates_path, golden_df)
    existing_decisions = load_existing_decisions(output_path)

    # Filter by priority
    if args.priority == "all":
        priority_candidates = raw_candidates
    else:
        priority_candidates = [
            c for c in raw_candidates
            if c.get("priority", "").upper() == args.priority.upper()
        ]

    # Filter out already reviewed conflicts
    pending_queue: List[Dict[str, Any]] = []
    for c in priority_candidates:
        cid = f"conflict_{c.get('golden_id_a')}_vs_{c.get('golden_id_b')}"
        if cid not in existing_decisions:
            pending_queue.append(c)

    print(f"\n[3/4] Queue Status:")
    print(f"      Total Flagged Candidates ({args.priority.upper()}): {len(priority_candidates)}")
    print(f"      Already Reviewed Conflicts:                   {len(existing_decisions)}")
    print(f"      Remaining Pending in Tier:                    {len(pending_queue)}")

    if not pending_queue:
        print(f"\n🎉 All {args.priority.upper()} consistency conflicts have been reviewed!")
        print("To review another tier, specify `--priority medium`, `--priority low`, or `--priority all`.")
        return 0

    session_batch = pending_queue[: args.batch_size]
    print(f"      Starting review session of {len(session_batch)} conflicts...\n")

    # 3. Interactive Review Loop
    reviewed_in_session = 0
    decisions_summary: Dict[str, int] = collections.defaultdict(int)

    for idx, c in enumerate(session_batch, 1):
        gid_a = c.get("golden_id_a", "")
        gid_b = c.get("golden_id_b", "")
        cid = f"conflict_{gid_a}_vs_{gid_b}"
        label_a = c.get("human_label_a", "")
        label_b = c.get("human_label_b", "")
        priority = c.get("priority", "")
        similarity = float(c.get("similarity_score", 0.0))
        signals = c.get("shared_semantic_signals", [])
        explanation = c.get("possible_ambiguity_explanation", "")

        # Lookup rich details
        rec_a = golden_lookup.get(gid_a, {})
        rec_b = golden_lookup.get(gid_b, {})
        ai_label_a = rec_a.get("model_suggested_label", "N/A")
        ai_conf_a = rec_a.get("model_confidence", "N/A")
        ai_label_b = rec_b.get("model_suggested_label", "N/A")
        ai_conf_b = rec_b.get("model_confidence", "N/A")
        msg_a = c.get("message_a", rec_a.get("customer_message", ""))
        msg_b = c.get("message_b", rec_b.get("customer_message", ""))

        conflict_type = f"{label_a}_vs_{label_b}"

        print("=" * 80)
        print(f"ANNOTATION CONSISTENCY REVIEW — CONFLICT {idx}/{len(session_batch)}")
        print("=" * 80)
        print(f"Conflict ID:      \033[1m{cid}\033[0m")
        print(f"Priority:         \033[1;33m{priority}\033[0m")
        print(f"Similarity Score: \033[1;32m{similarity:.4f}\033[0m")
        print(f"Conflict Type:    \033[1;35m{conflict_type}\033[0m")
        print("-" * 80)
        print(f"RECORD A [{gid_a}]")
        print(f"  Customer Message:    \"{msg_a}\"")
        print(f"  Current Human Label: \033[1;36m{label_a}\033[0m")
        print(f"  AI Suggested Label:  {ai_label_a} (conf: {ai_conf_a})")
        print("-" * 80)
        print(f"RECORD B [{gid_b}]")
        print(f"  Customer Message:    \"{msg_b}\"")
        print(f"  Current Human Label: \033[1;36m{label_b}\033[0m")
        print(f"  AI Suggested Label:  {ai_label_b} (conf: {ai_conf_b})")
        print("-" * 80)
        print("WHY THIS WAS FLAGGED:")
        print(f"  Shared Signals: {', '.join(signals) if signals else 'none'}")
        print(f"  Context:        {explanation}")
        print("-" * 80)
        print("HUMAN REVIEW OPTIONS:")
        print("  [A] Keep both existing labels (valid operational distinction)")
        print(f"  [B] Review Record A label (flag {gid_a} for reconsideration)")
        print(f"  [C] Review Record B label (flag {gid_b} for reconsideration)")
        print("  [D] Mark taxonomy boundary as valid")
        print("  [E] Flag for taxonomy guideline update")
        print("  [S] Skip")
        print("  [Q] Save and Quit")
        print("-" * 80)

        choice = input("Enter decision [A/B/C/D/E/S/Q] > ").strip().upper()

        if choice == "Q":
            print("\nSaving progress and quitting review session...")
            break
        elif choice in ("A", "D"):
            human_decision = "VALID_TAXONOMY_BOUNDARY"
        elif choice == "B":
            human_decision = "RECONSIDER_RECORD_A"
        elif choice == "C":
            human_decision = "RECONSIDER_RECORD_B"
        elif choice == "E":
            human_decision = "FLAG_TAXONOMY_GUIDELINE_UPDATE"
        else:
            human_decision = "SKIPPED"

        notes = ""
        if human_decision not in ("SKIPPED", "VALID_TAXONOMY_BOUNDARY"):
            notes = input("Optional adjudication notes (Enter to skip): ").strip()

        row = {
            "conflict_id": cid,
            "record_a_golden_id": gid_a,
            "record_b_golden_id": gid_b,
            "record_a_label": label_a,
            "record_b_label": label_b,
            "priority": priority,
            "similarity_score": round(similarity, 4),
            "conflict_type": conflict_type,
            "human_decision": human_decision,
            "reviewer": args.annotator,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "notes": notes,
        }

        append_decision_to_csv(output_path, row)
        reviewed_in_session += 1
        decisions_summary[human_decision] += 1
        print(f"\n[SUCCESS] Recorded decision: \033[1;32m{human_decision}\033[0m ({reviewed_in_session}/{len(session_batch)})\n")

    # 4. Strict Immutability Check — Post-run
    sha_post = calculate_file_sha256(golden_path)
    print("=" * 80)
    print("SESSION SUMMARY")
    print("=" * 80)
    print(f"Conflicts Reviewed in Session: {reviewed_in_session}")
    for dec, count in decisions_summary.items():
        print(f"  - {dec}: {count}")
    print(f"Decision Log Saved to:         {output_path}")
    print("-" * 80)
    print(f"Source SHA-256 (pre):  {sha_pre}")
    print(f"Source SHA-256 (post): {sha_post}")

    if sha_pre != sha_post:
        print("\n❌ CRITICAL INTEGRITY FAILURE: Golden dataset hash mismatch!")
        return 1

    print("\n✅ Dataset Immutability Verified: golden_set_human_review.csv is byte-for-byte identical.")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
