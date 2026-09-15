"""
SupportGraph AI — Annotation Consistency Analysis CLI (Phase 5.10)

Executes read-only semantic consistency auditing across completed human ground-truth
annotations, computes within-label coherence, evaluates cross-label boundaries,
and generates auditable reports.

Guiding Principles:
1. Strict Immutability: Verifies SHA-256 before and after execution.
2. Read-Only Analysis: Ground-truth annotations are never modified.
3. Controlled Feedback: Human review decisions are saved to an isolated audit log.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from app.core.logging import get_logger
    from app.evaluation.annotation_consistency import (
        AnnotationConsistencyAnalyzer,
        ConsistencyReviewDecision,
        calculate_file_sha256,
        extract_completed_reviews,
        generate_markdown_consistency_report,
    )
except ModuleNotFoundError:
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]
    from backend.app.evaluation.annotation_consistency import (  # type: ignore[no-redef]
        AnnotationConsistencyAnalyzer,
        ConsistencyReviewDecision,
        calculate_file_sha256,
        extract_completed_reviews,
        generate_markdown_consistency_report,
    )

logger = get_logger(__name__)

DEFAULT_INPUT_PATH = REPO_ROOT / "data" / "golden" / "golden_set_human_review.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "annotation_consistency"
DEFAULT_PHASE_REPORT_PATH = REPO_ROOT / "reports" / "phase_5_10_annotation_consistency.md"
DEFAULT_DECISIONS_PATH = REPO_ROOT / "data" / "golden" / "consistency_review_decisions.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SupportGraph AI — Phase 5.10 Annotation Consistency & Taxonomy Boundary Analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=str(DEFAULT_INPUT_PATH),
        help="Path to golden human review dataset CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save JSON and Markdown consistency reports",
    )
    parser.add_argument(
        "--report-file",
        type=str,
        default=str(DEFAULT_PHASE_REPORT_PATH),
        help="Path for main Phase 5.10 summary report",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch interactive terminal review for top consistency candidate pairs",
    )
    parser.add_argument(
        "--top-k-review",
        type=int,
        default=15,
        help="Number of top consistency candidate pairs to include in the review queue",
    )
    parser.add_argument(
        "--annotator",
        type=str,
        default="sridevi",
        help="Annotator identifier for consistency review feedback",
    )
    return parser.parse_args()


def run_interactive_consistency_review(
    candidates: List[Any],
    decisions_path: Path,
    annotator: str,
    top_k: int = 15,
) -> None:
    """
    Present top consistency candidate pairs to human annotator without altering source labels.
    """
    queue = candidates[:top_k]
    if not queue:
        print("\nNo consistency candidates available for interactive review.")
        return

    print("\n" + "=" * 80)
    print("SUPPORTGRAPH AI — HUMAN CONSISTENCY ADJUDICATION QUEUE")
    print("=" * 80)
    print("Reviewing top ambiguous cross-label pairs.")
    print("NOTE: Actions are saved to an isolated audit log. Ground-truth labels are NOT modified.")
    print("=" * 80 + "\n")

    existing_decisions: Dict[str, Dict[str, Any]] = {}
    if decisions_path.exists():
        try:
            with open(decisions_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        existing_decisions[item.get("pair_id", "")] = item
        except Exception as e:
            logger.warning("Could not parse existing consistency decisions: %s", e)

    new_decisions: List[Dict[str, Any]] = list(existing_decisions.values())

    for idx, pair in enumerate(queue, 1):
        pair_id = f"{pair.golden_id_a}_vs_{pair.golden_id_b}"
        print(f"\n[Pair #{idx}/{len(queue)}] Priority: {pair.priority} | Cosine Similarity: {pair.similarity_score:.4f}")
        print("-" * 80)
        print(f"Record A [{pair.golden_id_a}]  -> Label: \033[1;36m{pair.human_label_a}\033[0m")
        print(f"Message A: \"{pair.message_a}\"\n")
        print(f"Record B [{pair.golden_id_b}]  -> Label: \033[1;35m{pair.human_label_b}\033[0m")
        print(f"Message B: \"{pair.message_b}\"\n")
        print(f"Shared Signals: {', '.join(pair.shared_semantic_signals) if pair.shared_semantic_signals else 'none'}")
        print(f"Ambiguity Context: {pair.possible_ambiguity_explanation}\n")

        print("Options:")
        print("  [A] Labels are both appropriate (valid operational distinction)")
        print(f"  [B] Record A [{pair.golden_id_a}] should be reconsidered")
        print(f"  [C] Record B [{pair.golden_id_b}] should be reconsidered")
        print("  [D] Both records should be reconsidered")
        print("  [S] Skip to next pair")
        print("  [Q] Quit review queue")

        choice = input("\nDecision [A/B/C/D/S/Q]: ").strip().upper()

        if choice == "Q":
            print("\nExiting consistency review queue...")
            break
        elif choice == "A":
            decision_code = "LABELS_BOTH_APPROPRIATE"
        elif choice == "B":
            decision_code = "RECONSIDER_A"
        elif choice == "C":
            decision_code = "RECONSIDER_B"
        elif choice == "D":
            decision_code = "RECONSIDER_BOTH"
        else:
            decision_code = "SKIPPED"

        notes = ""
        if decision_code in ("RECONSIDER_A", "RECONSIDER_B", "RECONSIDER_BOTH"):
            notes = input("Optional review notes: ").strip()

        dec = ConsistencyReviewDecision(
            pair_id=pair_id,
            golden_id_a=pair.golden_id_a,
            golden_id_b=pair.golden_id_b,
            decision=decision_code,
            annotator=annotator,
            timestamp=datetime.now(timezone.utc).isoformat(),
            notes=notes,
        )

        # Update decision list
        existing_decisions[pair_id] = dec.to_dict()
        new_decisions = list(existing_decisions.values())

        # Save non-destructively
        decisions_path.parent.mkdir(parents=True, exist_ok=True)
        with open(decisions_path, "w", encoding="utf-8") as f:
            json.dump(new_decisions, f, indent=2)

        print(f"Saved decision '{decision_code}' for pair {pair_id}.")

    print(f"\nReview complete. Total recorded decisions: {len(new_decisions)} -> {decisions_path}")


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir)
    report_path = Path(args.report_file)
    decisions_path = Path(DEFAULT_DECISIONS_PATH)

    print("=" * 80)
    print("SUPPORTGRAPH AI — PHASE 5.10 ANNOTATION CONSISTENCY ANALYSIS")
    print("=" * 80)

    # 1. Immutability Check — Before
    if not input_path.exists():
        print(f"ERROR: Input dataset file not found: {input_path}")
        return 1

    sha_before = calculate_file_sha256(input_path)
    print(f"\n[1/5] Source Dataset Verified:")
    print(f"      File: {input_path}")
    print(f"      SHA-256 (pre-analysis): {sha_before}")

    # 2. Read Dataset in Read-Only Mode
    df = pd.read_csv(input_path, dtype=str)
    completed_df = extract_completed_reviews(df)
    n_total = len(df)
    n_completed = len(completed_df)
    n_pending = n_total - n_completed

    print(f"\n[2/5] Dataset Coverage:")
    print(f"      Total Records:      {n_total}")
    print(f"      Completed Reviews:  {n_completed} ({n_completed / n_total:.1%})")
    print(f"      Pending Reviews:    {n_pending} ({n_pending / n_total:.1%})")

    # 3. Execute Read-Only Consistency Analysis
    print(f"\n[3/5] Executing Semantic Consistency & Boundary Analysis...")
    analyzer = AnnotationConsistencyAnalyzer()
    summary, candidates, within_coherence, boundaries = analyzer.run_full_analysis(
        df=df,
        source_sha256=sha_before,
        total_records=n_total,
    )

    print(f"      Total Evaluated Pairs:       {summary.total_pairs_evaluated:,}")
    print(f"      Same-Label Pairs:            {summary.same_label_pairs:,}")
    print(f"      Cross-Label Pairs:           {summary.cross_label_pairs:,}")
    print(f"      Flagged Consistency Pairs:   {summary.total_candidates_flagged}")
    print(f"        - HIGH Priority:           {summary.high_priority_candidates}")
    print(f"        - MEDIUM Priority:         {summary.medium_priority_candidates}")
    print(f"        - LOW Priority:            {summary.low_priority_candidates}")

    # 4. Serialize Reports & Data Artifacts
    print(f"\n[4/5] Serializing Reports & Data Artifacts...")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # JSON artifacts
    summary_path = output_dir / "consistency_summary.json"
    candidates_path = output_dir / "cross_label_candidates.json"
    coherence_path = output_dir / "within_label_coherence.json"
    boundaries_path = output_dir / "boundary_analysis.json"
    queue_path = output_dir / "consistency_review_queue.json"
    report_dir_md = output_dir / "annotation_consistency_report.md"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary.to_dict(), f, indent=2)

    with open(candidates_path, "w", encoding="utf-8") as f:
        json.dump([c.to_dict() for c in candidates], f, indent=2)

    with open(coherence_path, "w", encoding="utf-8") as f:
        json.dump({k: v.to_dict() for k, v in within_coherence.items()}, f, indent=2)

    with open(boundaries_path, "w", encoding="utf-8") as f:
        json.dump([b.to_dict() for b in boundaries], f, indent=2)

    # Save review queue of top candidates
    top_queue = [c.to_dict() for c in candidates[: args.top_k_review]]
    with open(queue_path, "w", encoding="utf-8") as f:
        json.dump(top_queue, f, indent=2)

    # Markdown Reports
    md_content = generate_markdown_consistency_report(
        summary=summary,
        candidates=candidates,
        within_coherence=within_coherence,
        boundaries=boundaries,
    )

    with open(report_dir_md, "w", encoding="utf-8") as f:
        f.write(md_content)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"      Saved: {summary_path}")
    print(f"      Saved: {candidates_path}")
    print(f"      Saved: {coherence_path}")
    print(f"      Saved: {boundaries_path}")
    print(f"      Saved: {queue_path}")
    print(f"      Saved: {report_dir_md}")
    print(f"      Saved: {report_path}")

    # 5. Interactive Review Mode (if enabled)
    if args.interactive:
        run_interactive_consistency_review(
            candidates=candidates,
            decisions_path=decisions_path,
            annotator=args.annotator,
            top_k=args.top_k_review,
        )

    # 6. Immutability Check — After
    sha_after = calculate_file_sha256(input_path)
    print(f"\n[5/5] Immutability Verification:")
    print(f"      SHA-256 (pre):  {sha_before}")
    print(f"      SHA-256 (post): {sha_after}")

    if sha_before != sha_after:
        print("\n❌ CRITICAL ERROR: Source dataset checksum mismatch! Immutability violated!")
        return 1

    print("\n✅ Immutability Verified: Source dataset is 100% byte-for-byte identical.")
    print("=" * 80)
    print("Phase 5.10 Analysis Completed Successfully.")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
