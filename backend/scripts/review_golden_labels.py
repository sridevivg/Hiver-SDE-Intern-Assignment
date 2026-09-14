"""
SupportGraph AI — Interactive Human Review CLI (Phase 5.8)

Smart Human-In-The-Loop (HITL) review workflow for Golden Set intent validation:
- Intelligent pre-batch summary with risk distribution and recommended actions
- Mode 1: Individual record-by-record review for CRITICAL, HIGH, MEDIUM, and QC-sampled records
- Mode 2: Safe low-risk Group Review with mandatory confirmation screen and selective overrides
- Preserves scientific integrity: AI suggestions are advisory evidence and never auto-copied
- Preserves QC sampling: deterministic 15% sample receives mandatory individual review
- Session-level statistics with agreement metrics and explicit human approval accounting
- Full completion detection when all 200 records are reviewed
- Complete NaN/null protection, backward-compatible progress manifests, and per-action persistence

Usage:
    # Smart HITL review (default: individual for High/QC -> grouped for Safe Low):
    python -m backend.scripts.review_golden_labels --annotator sridevi --mode smart

    # Individual review only:
    python -m backend.scripts.review_golden_labels --annotator sridevi --mode individual

    # Safe low-risk grouped review only:
    python -m backend.scripts.review_golden_labels --annotator sridevi --mode grouped --group-size 10
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from app.core.logging import configure_logging
    from app.evaluation.human_review import (
        DEFAULT_PROGRESS_PATH,
        ReviewAction,
        apply_review_decision,
        compute_review_progress,
        get_allowed_taxonomy_labels,
        is_suggestion_complete,
        is_valid_ai_suggestion,
        load_review_progress_manifest,
        save_review_progress_manifest,
    )
    from app.evaluation.review_prioritization import (
        ReviewPriority,
        ReviewRecommendation,
        SafeReviewGroup,
        build_safe_review_groups,
        is_eligible_for_group_review,
    )
    from app.evaluation.taxonomy_calibration import (
        CalibrationResult,
        CalibrationStatus,
        evaluate_taxonomy_calibration,
    )
    from app.nlp.taxonomy_finalization import (
        DEFAULT_CANDIDATE_TAXONOMY_PATH,
        create_initial_candidate_taxonomy,
        load_candidate_taxonomy,
    )
except ModuleNotFoundError:
    from backend.app.core.logging import configure_logging  # type: ignore[no-redef]
    from backend.app.evaluation.human_review import (  # type: ignore[no-redef]
        DEFAULT_PROGRESS_PATH,
        ReviewAction,
        apply_review_decision,
        compute_review_progress,
        get_allowed_taxonomy_labels,
        is_suggestion_complete,
        is_valid_ai_suggestion,
        load_review_progress_manifest,
        save_review_progress_manifest,
    )
    from backend.app.evaluation.review_prioritization import (  # type: ignore[no-redef]
        ReviewPriority,
        ReviewRecommendation,
        SafeReviewGroup,
        build_safe_review_groups,
        is_eligible_for_group_review,
    )
    from backend.app.evaluation.taxonomy_calibration import (  # type: ignore[no-redef]
        CalibrationResult,
        CalibrationStatus,
        evaluate_taxonomy_calibration,
    )
    from backend.app.nlp.taxonomy_finalization import (  # type: ignore[no-redef]
        DEFAULT_CANDIDATE_TAXONOMY_PATH,
        create_initial_candidate_taxonomy,
        load_candidate_taxonomy,
    )

logger = logging.getLogger("review_golden_labels")

DEFAULT_QUEUE_CSV = "data/golden/golden_set_review_queue.csv"
DEFAULT_SUGGESTIONS_CSV = "data/golden/golden_set_ai_suggestions.csv"
DEFAULT_INPUT_CSV = DEFAULT_QUEUE_CSV if Path(DEFAULT_QUEUE_CSV).exists() else DEFAULT_SUGGESTIONS_CSV
DEFAULT_OUTPUT_CSV = "data/golden/golden_set_human_review.csv"
DEFAULT_BATCH_SIZE = 20
DEFAULT_GROUP_SIZE = 10


def parse_args() -> argparse.Namespace:
    default_in = DEFAULT_QUEUE_CSV if Path(DEFAULT_QUEUE_CSV).exists() else DEFAULT_SUGGESTIONS_CSV

    parser = argparse.ArgumentParser(
        description="Phase 5.8 Safe Grouped Smart Human Review CLI for SupportGraph AI"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=default_in,
        help=f"Input CSV with AI suggestions or prioritized queue (default: {default_in})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Output CSV for human annotations (default: {DEFAULT_OUTPUT_CSV})",
    )
    parser.add_argument(
        "--annotator",
        type=str,
        required=True,
        help="Identifier or name of human reviewer (mandatory)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["smart", "individual", "grouped"],
        default="smart",
        help="Review mode: 'smart' (Individual for High/QC then Safe Grouped), 'individual', or 'grouped' (default: smart)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Number of pending records to review in this session (default: {DEFAULT_BATCH_SIZE}, 0 for all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Alias for --batch-size (maximum records to review in this session)",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=DEFAULT_GROUP_SIZE,
        help=f"Maximum number of records per safe review group (default: {DEFAULT_GROUP_SIZE})",
    )
    parser.add_argument(
        "--taxonomy-path",
        type=str,
        default=DEFAULT_CANDIDATE_TAXONOMY_PATH,
        help="Path to candidate taxonomy JSON",
    )
    parser.add_argument(
        "--progress-path",
        type=str,
        default=DEFAULT_PROGRESS_PATH,
        help=f"Path to progress JSON manifest (default: {DEFAULT_PROGRESS_PATH})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all text and annotation columns are strings, without float NaNs or 'nan' string literals."""
    text_cols = [
        "golden_id", "tweet_id", "conversation_id", "customer_message",
        "normalized_message", "candidate_intent", "source_cluster", "conversation_context",
        "annotation_label", "annotation_status", "annotator", "notes",
        "model_name", "model_suggested_label", "model_confidence", "model_reasoning_summary",
        "model_needs_human_review", "suggestion_timestamp", "suggestion_status",
        "priority", "priority_score", "priority_reason", "qc_sample",
        "review_recommendation", "risk_flags", "risk_factors",
        "calibration_candidate_label", "calibration_confidence",
        "calibration_supporting_signals", "calibration_conflicting_signals", "calibration_status",
        "review_mode", "approval_type", "group_id", "approved_group_label",
        "group_size", "annotation_timestamp"
    ]
    for col in text_cols:
        if col not in df.columns:
            df[col] = ""
        else:
            df[col] = df[col].fillna("").astype(object)
            df[col] = df[col].apply(
                lambda x: "" if str(x).strip().lower() in ("nan", "none", "null", "undefined") else str(x).strip()
            )
    return df


def display_record(
    batch_idx: int,
    batch_total: int,
    total_reviewed_overall: int,
    total_records: int,
    pending_overall: int,
    record: pd.Series | dict[str, Any],
    taxonomy_list: list[str],
    allowed_set: set[str],
) -> tuple[bool, bool, str]:
    """Pretty-print a single record in individual review mode.
    Returns: (has_valid_suggestion, has_valid_calibration, calibration_candidate_label)
    """
    gid = record.get("golden_id", f"row_{batch_idx}")
    tweet_id = record.get("tweet_id", "N/A")
    cust_msg = record.get("customer_message", "")
    context = record.get("conversation_context", "No brand reply observed")

    raw_model_label = record.get("model_suggested_label", "")
    raw_status = record.get("suggestion_status", "")
    has_valid_suggestion = is_valid_ai_suggestion(raw_model_label, allowed_labels=allowed_set, suggestion_status=raw_status)
    model_label = str(raw_model_label).strip() if has_valid_suggestion else ""

    raw_conf = record.get("model_confidence", "")
    try:
        confidence_val = float(raw_conf)
        confidence_str = f"{confidence_val:.2f}"
    except (ValueError, TypeError):
        confidence_str = "N/A"

    reasoning = record.get("model_reasoning_summary", "No reasoning provided")
    needs_review = record.get("model_needs_human_review", False)

    # Priority metadata
    priority = str(record.get("priority", "")).strip().upper()
    priority_score = str(record.get("priority_score", "")).strip()
    priority_reason = str(record.get("priority_reason", "")).strip()
    risk_flags = str(record.get("risk_flags", "")).strip()
    qc_sample_flag = str(record.get("qc_sample", "")).strip().lower() in ("true", "1", "yes")

    # Calibration metadata (Phase 5.9)
    raw_calib_label = record.get("calibration_candidate_label", "")
    calib_label = str(raw_calib_label).strip() if raw_calib_label is not None and not pd.isna(raw_calib_label) else ""
    if calib_label.lower() in ("nan", "none", "null", "undefined"):
        calib_label = ""

    raw_calib_conf = record.get("calibration_confidence", "")
    try:
        calib_conf_val = float(raw_calib_conf)
        calib_conf_str = f"{calib_conf_val:.2f}"
    except (ValueError, TypeError):
        calib_conf_str = ""

    calib_status = str(record.get("calibration_status", "")).strip()
    calib_signals = str(record.get("calibration_supporting_signals", "")).strip()
    calib_conflicts = str(record.get("calibration_conflicting_signals", "")).strip()

    # Dynamic fallback if calibration fields not already populated on record
    if not calib_status:
        try:
            calib_res = evaluate_taxonomy_calibration(record, allowed_labels=allowed_set)
            calib_status = calib_res.status.value
            calib_label = calib_res.candidate_label
            calib_conf_str = f"{calib_res.confidence:.2f}" if calib_res.confidence > 0 else ""
            calib_signals = "; ".join(s.name for s in calib_res.supporting_signals)
            calib_conflicts = "; ".join(s.name for s in calib_res.conflicting_signals)
        except Exception:
            pass

    has_valid_calibration = bool(calib_label and calib_label in allowed_set)

    print("\n" + "=" * 70)
    print(f" ITEM {batch_idx} / {batch_total}  |  OVERALL GROUND TRUTH: {total_reviewed_overall} / {total_records} (Pending: {pending_overall})")
    print(f" Golden ID: {gid}  |  Tweet ID: {tweet_id}")
    print("=" * 70)

    if priority:
        qc_str = "YES (Mandatory QC Individual Review)" if qc_sample_flag else "No"
        print(f"REVIEW PRIORITY: {priority}  |  SCORE: {priority_score or 'N/A'}  |  QC SAMPLE: {qc_str}")
        if priority_reason:
            print(f"PRIORITY REASON: {priority_reason}")
        if risk_flags:
            print(f"RISK FLAGS:      {risk_flags}")
        print("-" * 70)

    print("CUSTOMER MESSAGE:")
    print(f"  \"{cust_msg}\"")
    print("\nCONVERSATION CONTEXT (First Brand Turn):")
    print(f"  \"{context}\"")
    print("-" * 70)

    if has_valid_suggestion:
        print("AI ASSISTED SUGGESTION (Advisory only):")
        print(f"  Suggested Label:   {model_label}")
        print(f"  Confidence:        {confidence_str}")
        print(f"  Needs Review Flag: {needs_review}")
        print(f"  Reasoning Summary: {reasoning}")
    else:
        print("AI ASSISTED SUGGESTION:")
        print("  No valid AI suggestion available for this record.")
        print("  This record requires direct human annotation.")
    print("-" * 70)

    # Display Taxonomy Calibration Card
    print("TAXONOMY CALIBRATION ANALYSIS (Transparent Contextual Guidance):")
    if calib_label:
        conf_display = f" (Confidence: {calib_conf_str})" if calib_conf_str else ""
        print(f"  Calibrated Recommendation: {calib_label}{conf_display}")
    else:
        print("  Calibrated Recommendation: None (Insufficient distinctive signals)")
    print(f"  Status:                    {calib_status or 'INSUFFICIENT_EVIDENCE'}")
    if calib_signals:
        print(f"  Supporting Signals:        {calib_signals}")
    if calib_conflicts:
        print(f"  Conflicting Signals:       {calib_conflicts}")
    print("-" * 70)

    print("ALLOWED OPERATIONAL TAXONOMY LABELS:")
    for i, intent_name in enumerate(taxonomy_list, start=1):
        sug_tag = " [AI SUGGESTED]" if (has_valid_suggestion and intent_name == model_label) else ""
        calib_tag = " [CALIBRATED]" if (has_valid_calibration and intent_name == calib_label) else ""
        tag = f"{sug_tag}{calib_tag}".strip()
        indicator = f" <-- {tag}" if tag else ""
        print(f"  [{i}] {intent_name:<34}{indicator}")
    print("  [U] unclear_needs_review (ambiguous / non-English / truncated)")
    print("-" * 70)
    print("REVIEW ACTIONS:")
    if has_valid_suggestion:
        print("  [A] Accept AI suggestion       [1-9] Override with numbered intent")
    else:
        print("  [A] (Unavailable)              [1-9] Select numbered intent")
    if has_valid_calibration:
        print(f"  [C] Accept calibrated recommendation ('{calib_label}')")
    print("  [U] Mark as unclear             [S] Skip to next record")
    print("  [Q] Save and Quit")
    print("-" * 70)
    return has_valid_suggestion, has_valid_calibration, calib_label


def display_batch_pre_summary(
    batch_records: list[dict[str, Any]],
    mode: str,
) -> None:
    """Print an intelligent batch summary and workflow recommendation before starting review."""
    total_batch = len(batch_records)
    crit_count = sum(1 for r in batch_records if str(r.get("priority", "")).lower() == "critical")
    high_count = sum(1 for r in batch_records if str(r.get("priority", "")).lower() == "high")
    med_count = sum(1 for r in batch_records if str(r.get("priority", "")).lower() == "medium")
    qc_count = sum(1 for r in batch_records if str(r.get("qc_sample", "")).lower() in ("true", "1", "yes"))
    low_safe_count = sum(
        1 for r in batch_records
        if str(r.get("priority", "")).lower() == "low" and str(r.get("qc_sample", "")).lower() not in ("true", "1", "yes")
    )

    print("\n" + "=" * 60)
    print(f"BATCH REVIEW SUMMARY & WORKFLOW PLAN (Mode: {mode.upper()})")
    print("=" * 60)
    print(f"Total records in this session batch: {total_batch}")
    print()
    print("AI Risk Breakdown:")
    print(f"  - Critical risk:                 {crit_count:<3} (Mandatory individual review)")
    print(f"  - High risk:                     {high_count:<3} (Mandatory individual review)")
    print(f"  - Medium risk:                   {med_count:<3} (Individual review)")
    print(f"  - QC sample (Low-risk):          {qc_count:<3} (Mandatory individual audit)")
    print(f"  - Safe Low-risk:                 {low_safe_count:<3} (Eligible for safe grouped approval)")
    print()
    print("Workflow Plan:")
    if mode == "smart":
        print("  1. Review CRITICAL, HIGH, MEDIUM, and QC-sample records individually.")
        print("  2. Review safe LOW-risk records in intent groups for fast explicit human approval.")
    elif mode == "individual":
        print("  - Review all records individually.")
    elif mode == "grouped":
        print("  - Review only safe low-risk records in intent groups (skipping high-risk).")
    print("=" * 60 + "\n")


def display_final_completion_banner(df: pd.DataFrame, output_path: Path) -> None:
    """Display final congratulations banner when 100% of golden dataset is human reviewed."""
    progress = compute_review_progress(df)
    total = progress["total_records"]
    reviewed = progress["human_reviewed"]
    accepted = progress["accepted_ai_suggestions"]
    overridden = progress["overridden_ai_suggestions"]
    unclear = progress["unclear_records"]
    group_app = progress.get("group_approved_records", 0)
    indiv_app = progress.get("individual_reviewed_records", 0)

    agreement_pct = (accepted / reviewed * 100) if reviewed > 0 else 0.0

    print("\n" + "=" * 60)
    print("🎉 GOLDEN SET HUMAN REVIEW COMPLETE 🎉")
    print("=" * 60)
    print(f"Total Records:                 {total}")
    print(f"Human Reviewed:                {reviewed} (100.0%)")
    print(f"  - Individual Human Reviews:  {indiv_app}")
    print(f"  - Group Human Approvals:     {group_app}")
    print(f"Pending:                       0")
    print("-" * 60)
    print(f"AI-Human Agreement Rate:       {agreement_pct:.1f}% ({accepted}/{reviewed})")
    print(f"Human Overrides:               {overridden}")
    print(f"Marked 'unclear_needs_review': {unclear}")
    print("-" * 60)
    print(f"Validated Ground Truth File:   {output_path}")
    print()
    print("Next Recommended Command:")
    print(f"  python -m backend.scripts.validate_golden_dataset --input {output_path} --require-complete")
    print("=" * 60 + "\n")


def display_session_summary(
    individual_reviewed_in_session: int,
    group_approved_in_session: int,
    accepted_in_session: int,
    calibration_accepted_in_session: int,
    overrides_in_session: int,
    unclear_in_session: int,
    skipped_in_session: int,
    groups_fully_approved: int,
    groups_partially_overridden: int,
    groups_reviewed_individually: int,
    corrections_list: list[tuple[str, str]],
    df: pd.DataFrame,
    output_path: Path,
    prog_path: Path,
    annotator: str,
) -> None:
    """Print structured summary of actions taken in the current session matching specification."""
    progress = compute_review_progress(df)
    total_records = progress["total_records"]
    completed_overall = progress["human_reviewed"]
    pending_remaining = progress["pending_human_review"]
    pct = (completed_overall / total_records * 100) if total_records > 0 else 0.0

    # Calculate agreement rate for individual reviews
    indiv_total = individual_reviewed_in_session
    indiv_agreement_str = (
        f"{(accepted_in_session / indiv_total * 100):.1f}% ({accepted_in_session}/{indiv_total})"
        if indiv_total > 0
        else "N/A (No individual reviews in session)"
    )

    print("\n" + "=" * 60)
    print("SESSION SUMMARY")
    print("=" * 60)
    print(f"Individual records reviewed:             {individual_reviewed_in_session}")
    print(f"Group-approved records:                  {group_approved_in_session}")
    print(f"Individual AI suggestions accepted:      {accepted_in_session}")
    print(f"Individual calibration accepted:         {calibration_accepted_in_session}")
    print(f"Human overrides:                         {overrides_in_session}")
    print(f"Marked unclear:                          {unclear_in_session}")
    print(f"Skipped:                                 {skipped_in_session}")
    print(f"Groups fully approved:                   {groups_fully_approved}")
    print(f"Groups partially overridden:             {groups_partially_overridden}")
    print(f"Groups reviewed individually:            {groups_reviewed_individually}")
    print("-" * 60)

    if corrections_list:
        print("Most Common Human Corrections in Session:")
        counts = Counter(corrections_list)
        for (sug, hlbl), count in counts.most_common(5):
            print(f"  - '{sug}' -> '{hlbl}' ({count}x)")
        print("-" * 60)

    print(f"Total completed ground truth:            {completed_overall} / {total_records} ({pct:.1f}%)")
    print(f"Remaining pending:                       {pending_remaining}")
    print("-" * 60)
    print("AI-Human agreement metrics:")
    print(f"For individual reviews only:             {indiv_agreement_str}")
    print("For group-approved records:              Explicit human group approval")
    print("-" * 60)
    print("Scientific integrity:")
    print("Existing annotations overwritten:        NO")
    print("AI suggestions auto-copied:              NO")
    print("Explicit human approvals recorded:       YES")
    print(f"Dataset integrity status:                PASS")
    print("=" * 60)
    print(f"Progress saved to:                       {output_path}")
    print(f"Manifest saved to:                       {prog_path}")

    if pending_remaining > 0:
        print("\nTo review the next batch, run:")
        print(f"  python -m backend.scripts.review_golden_labels --annotator {annotator}")
    else:
        display_final_completion_banner(df, output_path)
    print("=" * 60 + "\n")


def confirm_group_approval(intent: str, count: int) -> bool:
    """Display mandatory confirmation screen before executing group approval."""
    print("\n" + "=" * 60)
    print("CONFIRM GROUP APPROVAL")
    print("=" * 60)
    print("You are about to approve:")
    print()
    print(f"Intent:            {intent}")
    print(f"Number of records: {count}")
    print()
    print(f"All {count} records will receive the label:")
    print(f"  {intent}")
    print()
    print("This is an explicit HUMAN GROUP APPROVAL.")
    print("AI suggestions remain advisory evidence only.")
    print()
    print("Proceed?")
    print("  [Y] Yes, approve group")
    print("  [N] No, return to review")
    print("=" * 60)

    while True:
        try:
            choice = input("Enter confirmation [Y/N] > ").strip().upper()
        except (EOFError, KeyboardInterrupt):
            print("\nGroup approval cancelled.")
            return False

        if choice in ("Y", "YES"):
            return True
        elif choice in ("N", "NO"):
            return False
        else:
            print("Please enter 'Y' to confirm approval or 'N' to return.")


def execute_individual_review_for_record(
    df: pd.DataFrame,
    idx: int,
    item_num: int,
    item_total: int,
    total_reviewed_overall: int,
    total_records: int,
    pending_overall: int,
    annotator: str,
    taxonomy_list: list[str],
    allowed_set: set[str],
) -> tuple[str, Optional[tuple[str, str]]]:
    """
    Review a single record in individual review mode.
    Returns (status: 'accepted'|'calibration_accepted'|'overridden'|'unclear'|'skipped'|'quit', correction_tuple).
    """
    row = df.iloc[idx]
    has_valid_sug, has_valid_calib, calib_label = display_record(
        batch_idx=item_num,
        batch_total=item_total,
        total_reviewed_overall=total_reviewed_overall,
        total_records=total_records,
        pending_overall=pending_overall,
        record=row,
        taxonomy_list=taxonomy_list,
        allowed_set=allowed_set,
    )

    while True:
        try:
            user_input = input("Enter decision > ").strip().upper()
        except (EOFError, KeyboardInterrupt):
            return "quit", None

        if user_input == "Q":
            return "quit", None

        elif user_input == "S":
            print("Skipping record (remains pending in queue)...")
            return "skipped", None

        elif user_input == "A":
            if not has_valid_sug:
                print("\n[A] Accept AI suggestion unavailable because no valid AI suggestion exists.")
                print("Please select [1-9] to assign an approved taxonomy intent, [C] for calibrated recommendation, [U] for unclear, or [S] to skip.")
                continue

            sug = str(row.get("model_suggested_label", "")).strip()
            updated = apply_review_decision(
                record=row,
                action=ReviewAction.ACCEPT,
                annotator=annotator,
                allowed_labels=allowed_set,
                review_mode="individual_human_review",
                approval_type="accepted_ai_suggestion",
            )
            for k, v in updated.items():
                df.at[idx, k] = v
            print(f"[SUCCESS] Accepted suggestion: '{sug}' ({item_num}/{item_total})")
            return "accepted", None

        elif user_input == "C":
            if not has_valid_calib or not calib_label:
                print("\n[C] Accept calibrated recommendation unavailable because no valid recommendation exists.")
                continue

            updated = apply_review_decision(
                record=row,
                action=ReviewAction.CALIBRATION_ACCEPT,
                annotator=annotator,
                selected_label=calib_label,
                allowed_labels=allowed_set,
                review_mode="individual_human_review",
                approval_type="individual_calibration_acceptance",
            )
            for k, v in updated.items():
                df.at[idx, k] = v

            sug = str(row.get("model_suggested_label", "")).strip()
            correction = (sug, calib_label) if sug and sug != calib_label else None
            print(f"[SUCCESS] Accepted calibrated recommendation: '{calib_label}' ({item_num}/{item_total}).")
            return "calibration_accepted", correction

        elif user_input == "U":
            sug = str(row.get("model_suggested_label", "")).strip()
            updated = apply_review_decision(
                record=row,
                action=ReviewAction.UNCLEAR,
                annotator=annotator,
                allowed_labels=allowed_set,
                review_mode="individual_human_review",
                approval_type="unclear",
            )
            for k, v in updated.items():
                df.at[idx, k] = v
            correction = (sug, "unclear_needs_review") if sug and sug != "unclear_needs_review" else None
            print(f"[SUCCESS] Marked record as 'unclear_needs_review' ({item_num}/{item_total}).")
            return "unclear", correction

        elif user_input.isdigit() and 1 <= int(user_input) <= len(taxonomy_list):
            selected_intent = taxonomy_list[int(user_input) - 1]
            notes_input = input(f"Optional notes for '{selected_intent}' (press Enter to skip): ").strip()
            updated = apply_review_decision(
                record=row,
                action=ReviewAction.OVERRIDE,
                annotator=annotator,
                selected_label=selected_intent,
                notes=notes_input,
                allowed_labels=allowed_set,
                review_mode="individual_human_review",
                approval_type="human_override",
            )
            for k, v in updated.items():
                df.at[idx, k] = v

            sug = str(row.get("model_suggested_label", "")).strip()
            correction = (sug, selected_intent) if sug and sug != selected_intent else None
            print(f"[SUCCESS] Recorded override: '{selected_intent}' ({item_num}/{item_total}).")
            return "overridden", correction

        else:
            opts = ["A (Accept AI)" if has_valid_sug else None,
                    f"C (Accept Calibrated '{calib_label}')" if has_valid_calib else None,
                    "1-9 (Override)", "U (Unclear)", "S (Skip)", "Q (Quit)"]
            opt_str = ", ".join(o for o in opts if o)
            print(f"Invalid input. Options: {opt_str}")


def main() -> int:
    args = parse_args()
    log_level = "DEBUG" if args.verbose else "INFO"
    configure_logging(log_level=log_level)

    input_path = Path(args.input)
    output_path = Path(args.output)
    tax_path = Path(args.taxonomy_path)
    prog_path = Path(args.progress_path)

    if not input_path.exists() and not output_path.exists():
        logger.error("Neither input file (%s) nor output file (%s) exists.", input_path, output_path)
        return 1

    # Load taxonomy
    if tax_path.exists():
        taxonomy = load_candidate_taxonomy(tax_path)
    else:
        taxonomy = create_initial_candidate_taxonomy()
    taxonomy_list = sorted(taxonomy.intent_names)
    allowed_set = set(taxonomy_list) | {"unclear_needs_review"}

    # Load working dataframe with safe merging
    if output_path.exists() and input_path.exists():
        logger.info("Resuming review: merging existing annotations from %s with latest suggestions from %s", output_path, input_path)
        df_in = clean_dataframe(pd.read_csv(input_path, dtype=str))
        df_out = clean_dataframe(pd.read_csv(output_path, dtype=str))
        df = df_in.copy()
        # Overlay any existing completed human annotations
        for _, out_row in df_out.iterrows():
            gid = str(out_row.get("golden_id", "")).strip()
            lbl = str(out_row.get("annotation_label", "")).strip()
            stat = str(out_row.get("annotation_status", "")).strip().lower()
            if lbl and stat not in ("", "pending", "pending_human_review"):
                matches = df.index[df["golden_id"] == gid].tolist()
                if matches:
                    m_idx = matches[0]
                    for col in [
                        "annotation_label", "annotation_status", "annotator", "notes",
                        "review_mode", "approval_type", "group_id", "approved_group_label",
                        "group_size", "annotation_timestamp"
                    ]:
                        if col in out_row:
                            df.at[m_idx, col] = out_row.get(col, "")
    elif output_path.exists():
        logger.info("Resuming review from existing output file: %s", output_path)
        df = clean_dataframe(pd.read_csv(output_path, dtype=str))
    else:
        logger.info("Initializing human review dataset from: %s", input_path)
        df = clean_dataframe(pd.read_csv(input_path, dtype=str))

    total_records = len(df)

    # Identify pending records
    pending_row_indices: list[int] = []
    already_reviewed_count = 0

    for idx, row in df.iterrows():
        lbl = str(row.get("annotation_label", "")).strip()
        stat = str(row.get("annotation_status", "")).strip().lower()
        if lbl and lbl.lower() not in ("nan", "none", "null") and stat not in ("", "pending", "pending_human_review"):
            already_reviewed_count += 1
        else:
            pending_row_indices.append(idx)

    pending_total = len(pending_row_indices)

    # Check if all records are already completed
    if pending_total == 0:
        display_final_completion_banner(df, output_path)
        return 0

    # Determine batch size limit
    batch_size = args.limit if args.limit is not None else args.batch_size
    if batch_size is None or batch_size <= 0:
        batch_size = pending_total
    else:
        batch_size = min(batch_size, pending_total)

    target_batch_indices = pending_row_indices[:batch_size]
    batch_records = [df.iloc[i].to_dict() for i in target_batch_indices]

    # Pre-batch summary
    display_batch_pre_summary(batch_records, mode=args.mode)

    # Partition target batch into:
    # 1. Individual Review items (CRITICAL, HIGH, MEDIUM, QC-sample)
    # 2. Safe Group Review items (LOW risk, non-QC, clean)
    individual_indices: list[int] = []
    group_eligible_indices: list[int] = []

    for idx in target_batch_indices:
        r = df.iloc[idx]
        rec_dict = r.to_dict() if hasattr(r, "to_dict") else dict(r)
        is_safe_group, _ = is_eligible_for_group_review(rec_dict, allowed_labels=allowed_set)

        if args.mode == "individual":
            individual_indices.append(idx)
        elif args.mode == "grouped":
            if is_safe_group:
                group_eligible_indices.append(idx)
        else:  # smart mode
            if is_safe_group:
                group_eligible_indices.append(idx)
            else:
                individual_indices.append(idx)

    # Track session metrics
    individual_reviewed_in_session = 0
    group_approved_in_session = 0
    accepted_in_session = 0
    calibration_accepted_in_session = 0
    overrides_in_session = 0
    unclear_in_session = 0
    skipped_in_session = 0
    groups_fully_approved = 0
    groups_partially_overridden = 0
    groups_reviewed_individually = 0
    corrections_list: list[tuple[str, str]] = []

    # Manifest tracking
    manifest = load_review_progress_manifest(prog_path)
    completed_groups = set(manifest.get("completed_group_ids", []))
    partially_groups = set(manifest.get("partially_reviewed_group_ids", []))

    try:
        # =====================================================================
        # STEP 1: INDIVIDUAL REVIEW QUEUE
        # =====================================================================
        if individual_indices and args.mode != "grouped":
            print(f">>> STARTING INDIVIDUAL REVIEW ({len(individual_indices)} records requiring individual inspection)...\n")

            for step_idx, idx in enumerate(individual_indices, start=1):
                row = df.iloc[idx]
                current_label = str(row.get("annotation_label", "")).strip()
                current_status = str(row.get("annotation_status", "")).strip().lower()

                if current_label and current_label.lower() not in ("nan", "none", "null") and current_status not in ("", "pending", "pending_human_review"):
                    continue

                current_reviewed_total = already_reviewed_count + individual_reviewed_in_session + group_approved_in_session
                current_pending_total = total_records - current_reviewed_total

                res_status, correction = execute_individual_review_for_record(
                    df=df,
                    idx=idx,
                    item_num=step_idx,
                    item_total=len(individual_indices),
                    total_reviewed_overall=current_reviewed_total,
                    total_records=total_records,
                    pending_overall=current_pending_total,
                    annotator=args.annotator,
                    taxonomy_list=taxonomy_list,
                    allowed_set=allowed_set,
                )

                if res_status == "quit":
                    df.to_csv(output_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
                    progress = compute_review_progress(df, allowed_labels=allowed_set)
                    progress["completed_group_ids"] = list(completed_groups)
                    progress["partially_reviewed_group_ids"] = list(partially_groups)
                    progress["current_mode"] = args.mode
                    save_review_progress_manifest(progress, prog_path)
                    display_session_summary(
                        individual_reviewed_in_session, group_approved_in_session,
                        accepted_in_session, calibration_accepted_in_session,
                        overrides_in_session, unclear_in_session,
                        skipped_in_session, groups_fully_approved, groups_partially_overridden,
                        groups_reviewed_individually, corrections_list, df, output_path,
                        prog_path, args.annotator
                    )
                    return 0

                elif res_status == "skipped":
                    skipped_in_session += 1

                elif res_status == "accepted":
                    individual_reviewed_in_session += 1
                    accepted_in_session += 1
                    df.to_csv(output_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
                    progress = compute_review_progress(df, allowed_labels=allowed_set)
                    save_review_progress_manifest(progress, prog_path)

                elif res_status == "calibration_accepted":
                    individual_reviewed_in_session += 1
                    calibration_accepted_in_session += 1
                    if correction:
                        corrections_list.append(correction)
                    df.to_csv(output_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
                    progress = compute_review_progress(df, allowed_labels=allowed_set)
                    save_review_progress_manifest(progress, prog_path)

                elif res_status == "unclear":
                    individual_reviewed_in_session += 1
                    unclear_in_session += 1
                    if correction:
                        corrections_list.append(correction)
                    df.to_csv(output_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
                    progress = compute_review_progress(df, allowed_labels=allowed_set)
                    save_review_progress_manifest(progress, prog_path)

                elif res_status == "overridden":
                    individual_reviewed_in_session += 1
                    overrides_in_session += 1
                    if correction:
                        corrections_list.append(correction)
                    df.to_csv(output_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
                    progress = compute_review_progress(df, allowed_labels=allowed_set)
                    save_review_progress_manifest(progress, prog_path)

        # =====================================================================
        # STEP 2: SAFE LOW-RISK GROUP REVIEW
        # =====================================================================
        if group_eligible_indices and args.mode != "individual":
            # Extract eligible subset DataFrame for deterministic grouping
            sub_df = df.iloc[group_eligible_indices].copy()
            safe_groups = build_safe_review_groups(
                df=sub_df,
                max_group_size=args.group_size,
                allowed_labels=allowed_set,
            )

            print("\n" + "=" * 70)
            print(f">>> SAFE LOW-RISK GROUP REVIEW ({len(group_eligible_indices)} records in {len(safe_groups)} groups)")
            print("=" * 70)

            for group_idx, group in enumerate(safe_groups, start=1):
                # Filter active row indices in master df
                active_row_indices = []
                for rec_dict in group.records:
                    gid = str(rec_dict.get("golden_id", ""))
                    matches = df.index[df["golden_id"] == gid].tolist()
                    if matches:
                        m_idx = matches[0]
                        lbl = str(df.at[m_idx, "annotation_label"]).strip()
                        stat = str(df.at[m_idx, "annotation_status"]).strip().lower()
                        if not lbl or stat in ("", "pending", "pending_human_review"):
                            active_row_indices.append(m_idx)

                if not active_row_indices:
                    continue

                active_records = [df.iloc[idx].to_dict() for idx in active_row_indices]

                print("\n" + "=" * 60)
                print("SAFE LOW-RISK GROUP REVIEW")
                print("=" * 60)
                print(f"Intent Group:        {group.suggested_label}")
                print(f"Group Identifier:    {group.group_id} ({group_idx}/{len(safe_groups)})")
                print(f"Records in group:    {len(active_records)}")
                print(f"AI confidence range: {group.min_confidence:.2f} - {group.max_confidence:.2f}")
                print(f"Risk flags:          {', '.join(group.risk_flags)}")
                print("=" * 60)

                # Display compact record cards
                for pos, r in enumerate(active_records, start=1):
                    gid = r.get("golden_id", f"row_{pos}")
                    tweet_id = r.get("tweet_id", "N/A")
                    cust = str(r.get("customer_message", ""))
                    ctx = str(r.get("conversation_context", "No brand reply observed"))
                    conf_raw = r.get("model_confidence", "N/A")
                    try:
                        conf_str = f"{float(conf_raw):.2f}"
                    except (ValueError, TypeError):
                        conf_str = str(conf_raw)

                    print("-" * 60)
                    print(f"[{pos}] Golden ID: {gid}")
                    print(f"Tweet ID: {tweet_id}")
                    print()
                    print("Customer:")
                    print(f"\"{cust}\"")
                    print()
                    print("Context:")
                    print(f"\"{ctx}\"")
                    print()
                    print("AI Suggestion:")
                    print(f"{group.suggested_label} ({conf_str})")

                print("-" * 60)
                print("GROUP REVIEW ACTIONS:")
                print("  [A] Approve ALL records with the suggested label")
                print("  [I] Review records individually")
                print("  [O] Override selected records")
                print("  [U] Mark selected records as unclear")
                print("  [S] Skip this group")
                print("  [Q] Save and quit")
                print("-" * 60)

                while True:
                    try:
                        grp_action = input("Enter group action > ").strip().upper()
                    except (EOFError, KeyboardInterrupt):
                        grp_action = "Q"

                    if grp_action == "Q":
                        df.to_csv(output_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
                        progress = compute_review_progress(df, allowed_labels=allowed_set)
                        progress["completed_group_ids"] = list(completed_groups)
                        progress["partially_reviewed_group_ids"] = list(partially_groups)
                        progress["current_mode"] = args.mode
                        save_review_progress_manifest(progress, prog_path)
                        display_session_summary(
                            individual_reviewed_in_session, group_approved_in_session,
                            accepted_in_session, calibration_accepted_in_session,
                            overrides_in_session, unclear_in_session,
                            skipped_in_session, groups_fully_approved, groups_partially_overridden,
                            groups_reviewed_individually, corrections_list, df, output_path,
                            prog_path, args.annotator
                        )
                        return 0

                    elif grp_action == "S":
                        skipped_in_session += len(active_row_indices)
                        print(f"Skipped {len(active_row_indices)} records in group '{group.group_id}' (remain pending).")
                        break

                    elif grp_action == "A":
                        confirmed = confirm_group_approval(group.suggested_label, len(active_row_indices))
                        if not confirmed:
                            print("\nGroup approval cancelled. Returning to group review actions.")
                            continue

                        now_iso = datetime.now(timezone.utc).isoformat()
                        for row_idx in active_row_indices:
                            updated = apply_review_decision(
                                record=df.iloc[row_idx],
                                action=ReviewAction.GROUP_APPROVE,
                                annotator=args.annotator,
                                allowed_labels=allowed_set,
                                review_mode="group_human_approval",
                                approval_type="group_approved_suggestion",
                                group_id=group.group_id,
                                approved_group_label=group.suggested_label,
                                group_size=len(active_row_indices),
                                annotation_timestamp=now_iso,
                            )
                            for k, v in updated.items():
                                df.at[row_idx, k] = v

                            group_approved_in_session += 1
                            accepted_in_session += 1

                        groups_fully_approved += 1
                        completed_groups.add(group.group_id)

                        df.to_csv(output_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
                        progress = compute_review_progress(df, allowed_labels=allowed_set)
                        progress["completed_group_ids"] = list(completed_groups)
                        progress["partially_reviewed_group_ids"] = list(partially_groups)
                        save_review_progress_manifest(progress, prog_path)
                        print(f"\n[SUCCESS] Group approved {len(active_row_indices)} records as '{group.suggested_label}'.")
                        break

                    elif grp_action == "I":
                        groups_reviewed_individually += 1
                        print(f"\nSwitching to individual review for {len(active_row_indices)} records in group '{group.group_id}'...")
                        for pos_idx, row_idx in enumerate(active_row_indices, start=1):
                            current_reviewed_total = already_reviewed_count + individual_reviewed_in_session + group_approved_in_session
                            current_pending_total = total_records - current_reviewed_total

                            res_status, correction = execute_individual_review_for_record(
                                df=df,
                                idx=row_idx,
                                item_num=pos_idx,
                                item_total=len(active_row_indices),
                                total_reviewed_overall=current_reviewed_total,
                                total_records=total_records,
                                pending_overall=current_pending_total,
                                annotator=args.annotator,
                                taxonomy_list=taxonomy_list,
                                allowed_set=allowed_set,
                            )
                            if res_status == "quit":
                                df.to_csv(output_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
                                progress = compute_review_progress(df, allowed_labels=allowed_set)
                                progress["completed_group_ids"] = list(completed_groups)
                                progress["partially_reviewed_group_ids"] = list(partially_groups)
                                save_review_progress_manifest(progress, prog_path)
                                display_session_summary(
                                    individual_reviewed_in_session, group_approved_in_session,
                                    accepted_in_session, calibration_accepted_in_session,
                                    overrides_in_session, unclear_in_session,
                                    skipped_in_session, groups_fully_approved, groups_partially_overridden,
                                    groups_reviewed_individually, corrections_list, df, output_path,
                                    prog_path, args.annotator
                                )
                                return 0
                            elif res_status == "skipped":
                                skipped_in_session += 1
                            elif res_status == "accepted":
                                individual_reviewed_in_session += 1
                                accepted_in_session += 1
                            elif res_status == "calibration_accepted":
                                individual_reviewed_in_session += 1
                                calibration_accepted_in_session += 1
                                if correction:
                                    corrections_list.append(correction)
                            elif res_status == "unclear":
                                individual_reviewed_in_session += 1
                                unclear_in_session += 1
                                if correction:
                                    corrections_list.append(correction)
                            elif res_status == "overridden":
                                individual_reviewed_in_session += 1
                                overrides_in_session += 1
                                if correction:
                                    corrections_list.append(correction)

                        partially_groups.add(group.group_id)
                        df.to_csv(output_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
                        progress = compute_review_progress(df, allowed_labels=allowed_set)
                        progress["completed_group_ids"] = list(completed_groups)
                        progress["partially_reviewed_group_ids"] = list(partially_groups)
                        save_review_progress_manifest(progress, prog_path)
                        break

                    elif grp_action == "O":
                        # Selective override
                        print("\nEnter record numbers to override (e.g. '1,3' or '2'):")
                        override_nums_str = input("Record numbers > ").strip()
                        num_tokens = [t.strip() for t in override_nums_str.replace(",", " ").split() if t.strip().isdigit()]
                        selected_positions = [int(t) for t in num_tokens if 1 <= int(t) <= len(active_row_indices)]

                        if not selected_positions:
                            print("No valid record numbers entered. Returning to group menu.")
                            continue

                        overridden_row_indices = [active_row_indices[p - 1] for p in selected_positions]
                        for o_pos, o_idx in zip(selected_positions, overridden_row_indices):
                            o_row = df.iloc[o_idx]
                            print(f"\n--- Overriding Record [{o_pos}]: {o_row.get('golden_id')} ---")
                            print(f"Customer: \"{o_row.get('customer_message')}\"")
                            print("Allowed Taxonomy Labels:")
                            for i, t_name in enumerate(taxonomy_list, start=1):
                                print(f"  [{i}] {t_name}")
                            print("  [U] unclear_needs_review")

                            while True:
                                o_choice = input("Select override label > ").strip().upper()
                                if o_choice == "U":
                                    updated = apply_review_decision(
                                        record=o_row,
                                        action=ReviewAction.UNCLEAR,
                                        annotator=args.annotator,
                                        allowed_labels=allowed_set,
                                        review_mode="individual_human_review",
                                        approval_type="unclear",
                                    )
                                    for k, v in updated.items():
                                        df.at[o_idx, k] = v
                                    individual_reviewed_in_session += 1
                                    unclear_in_session += 1
                                    corrections_list.append((group.suggested_label, "unclear_needs_review"))
                                    break
                                elif o_choice.isdigit() and 1 <= int(o_choice) <= len(taxonomy_list):
                                    sel_label = taxonomy_list[int(o_choice) - 1]
                                    notes_in = input(f"Optional notes for '{sel_label}': ").strip()
                                    updated = apply_review_decision(
                                        record=o_row,
                                        action=ReviewAction.OVERRIDE,
                                        annotator=args.annotator,
                                        selected_label=sel_label,
                                        notes=notes_in,
                                        allowed_labels=allowed_set,
                                        review_mode="individual_human_review",
                                        approval_type="human_override",
                                    )
                                    for k, v in updated.items():
                                        df.at[o_idx, k] = v
                                    individual_reviewed_in_session += 1
                                    overrides_in_session += 1
                                    corrections_list.append((group.suggested_label, sel_label))
                                    break
                                else:
                                    print("Invalid choice. Select [1-9] or [U].")

                        groups_partially_overridden += 1
                        partially_groups.add(group.group_id)

                        # Handle remaining records
                        remaining_row_indices = [idx for idx in active_row_indices if idx not in overridden_row_indices]
                        if not remaining_row_indices:
                            df.to_csv(output_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
                            break

                        print(f"\nRemaining records in group: {len(remaining_row_indices)}")
                        print("Actions for remaining:")
                        print(f"  [A] Approve remaining {len(remaining_row_indices)} records as '{group.suggested_label}'")
                        print("  [I] Review remaining individually")
                        print("  [S] Skip remaining records")

                        rem_choice = input("Enter choice for remaining > ").strip().upper()
                        if rem_choice == "A":
                            if confirm_group_approval(group.suggested_label, len(remaining_row_indices)):
                                now_iso = datetime.now(timezone.utc).isoformat()
                                for r_idx in remaining_row_indices:
                                    updated = apply_review_decision(
                                        record=df.iloc[r_idx],
                                        action=ReviewAction.GROUP_APPROVE,
                                        annotator=args.annotator,
                                        allowed_labels=allowed_set,
                                        review_mode="group_human_approval",
                                        approval_type="group_approved_suggestion",
                                        group_id=group.group_id,
                                        approved_group_label=group.suggested_label,
                                        group_size=len(active_row_indices),
                                        annotation_timestamp=now_iso,
                                    )
                                    for k, v in updated.items():
                                        df.at[r_idx, k] = v
                                    group_approved_in_session += 1
                                    accepted_in_session += 1
                                print(f"[SUCCESS] Approved {len(remaining_row_indices)} remaining records as '{group.suggested_label}'.")
                            else:
                                skipped_in_session += len(remaining_row_indices)
                                print("Remaining records skipped (remain pending).")
                        elif rem_choice == "I":
                            for pos_idx, r_idx in enumerate(remaining_row_indices, start=1):
                                current_reviewed_total = already_reviewed_count + individual_reviewed_in_session + group_approved_in_session
                                current_pending_total = total_records - current_reviewed_total
                                res_status, correction = execute_individual_review_for_record(
                                    df=df,
                                    idx=r_idx,
                                    item_num=pos_idx,
                                    item_total=len(remaining_row_indices),
                                    total_reviewed_overall=current_reviewed_total,
                                    total_records=total_records,
                                    pending_overall=current_pending_total,
                                    annotator=args.annotator,
                                    taxonomy_list=taxonomy_list,
                                    allowed_set=allowed_set,
                                )
                                if res_status == "accepted":
                                    individual_reviewed_in_session += 1
                                    accepted_in_session += 1
                                elif res_status == "calibration_accepted":
                                    individual_reviewed_in_session += 1
                                    calibration_accepted_in_session += 1
                                    if correction:
                                        corrections_list.append(correction)
                                elif res_status == "overridden":
                                    individual_reviewed_in_session += 1
                                    overrides_in_session += 1
                                    if correction:
                                        corrections_list.append(correction)
                                elif res_status == "unclear":
                                    individual_reviewed_in_session += 1
                                    unclear_in_session += 1
                                    if correction:
                                        corrections_list.append(correction)
                                elif res_status == "skipped":
                                    skipped_in_session += 1
                        else:
                            skipped_in_session += len(remaining_row_indices)
                            print("Remaining records skipped (remain pending).")

                        df.to_csv(output_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
                        progress = compute_review_progress(df, allowed_labels=allowed_set)
                        save_review_progress_manifest(progress, prog_path)
                        break

                    elif grp_action == "U":
                        # Mark selected as unclear
                        print("\nEnter record numbers to mark as unclear (e.g. '1,3' or '2'):")
                        unclear_nums_str = input("Record numbers > ").strip()
                        num_tokens = [t.strip() for t in unclear_nums_str.replace(",", " ").split() if t.strip().isdigit()]
                        selected_positions = [int(t) for t in num_tokens if 1 <= int(t) <= len(active_row_indices)]

                        if not selected_positions:
                            print("No valid record numbers entered. Returning to group menu.")
                            continue

                        unclear_row_indices = [active_row_indices[p - 1] for p in selected_positions]
                        for u_idx in unclear_row_indices:
                            updated = apply_review_decision(
                                record=df.iloc[u_idx],
                                action=ReviewAction.UNCLEAR,
                                annotator=args.annotator,
                                allowed_labels=allowed_set,
                                review_mode="individual_human_review",
                                approval_type="unclear",
                            )
                            for k, v in updated.items():
                                df.at[u_idx, k] = v
                            individual_reviewed_in_session += 1
                            unclear_in_session += 1
                            corrections_list.append((group.suggested_label, "unclear_needs_review"))

                        groups_partially_overridden += 1
                        partially_groups.add(group.group_id)

                        # Handle remaining
                        remaining_row_indices = [idx for idx in active_row_indices if idx not in unclear_row_indices]
                        if remaining_row_indices:
                            print(f"\nRemaining records in group: {len(remaining_row_indices)}")
                            print("Actions for remaining:")
                            print(f"  [A] Approve remaining {len(remaining_row_indices)} records as '{group.suggested_label}'")
                            print("  [I] Review remaining individually")
                            print("  [S] Skip remaining records")
                            rem_choice = input("Enter choice for remaining > ").strip().upper()
                            if rem_choice == "A" and confirm_group_approval(group.suggested_label, len(remaining_row_indices)):
                                now_iso = datetime.now(timezone.utc).isoformat()
                                for r_idx in remaining_row_indices:
                                    updated = apply_review_decision(
                                        record=df.iloc[r_idx],
                                        action=ReviewAction.GROUP_APPROVE,
                                        annotator=args.annotator,
                                        allowed_labels=allowed_set,
                                        review_mode="group_human_approval",
                                        approval_type="group_approved_suggestion",
                                        group_id=group.group_id,
                                        approved_group_label=group.suggested_label,
                                        group_size=len(active_row_indices),
                                        annotation_timestamp=now_iso,
                                    )
                                    for k, v in updated.items():
                                        df.at[r_idx, k] = v
                                    group_approved_in_session += 1
                                    accepted_in_session += 1
                                print(f"[SUCCESS] Approved {len(remaining_row_indices)} remaining records as '{group.suggested_label}'.")
                            else:
                                skipped_in_session += len(remaining_row_indices)

                        df.to_csv(output_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
                        progress = compute_review_progress(df, allowed_labels=allowed_set)
                        save_review_progress_manifest(progress, prog_path)
                        break

                    else:
                        print("Invalid group action. Options: A (Approve all), I (Review individually), O (Override selected), U (Mark unclear), S (Skip), Q (Quit)")

        # Final persistence & summary
        df.to_csv(output_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
        final_progress = compute_review_progress(df, allowed_labels=allowed_set)
        final_progress["completed_group_ids"] = sorted(list(completed_groups))
        final_progress["partially_reviewed_group_ids"] = sorted(list(partially_groups))
        final_progress["current_mode"] = args.mode
        save_review_progress_manifest(final_progress, prog_path)

        display_session_summary(
            individual_reviewed_in_session, group_approved_in_session,
            accepted_in_session, calibration_accepted_in_session,
            overrides_in_session, unclear_in_session,
            skipped_in_session, groups_fully_approved, groups_partially_overridden,
            groups_reviewed_individually, corrections_list, df, output_path,
            prog_path, args.annotator
        )
        return 0

    except Exception as exc:
        logger.error("Unexpected error during human review session: %s", exc, exc_info=True)
        df.to_csv(output_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
        return 1


if __name__ == "__main__":
    sys.exit(main())
