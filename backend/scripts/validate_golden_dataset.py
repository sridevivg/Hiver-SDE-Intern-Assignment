"""
SupportGraph AI — Golden Dataset Validation & Scientific Audit Engine (Phase 5.6)

Performs comprehensive validation and scientific integrity audits:
1. Dataset Integrity: Record count, unique golden_ids, no duplicate tweet_ids/messages
2. Schema & Taxonomy Conformance: Valid operational taxonomy labels, no 'nan'/'null' strings
3. Scientific Integrity Guarantees:
   - Verifies AI suggestions are NEVER automatically copied to human annotation_label
   - Verifies human ground truth fields (annotation_label, annotator, annotation_status)
     are populated strictly through human review actions
   - Verifies pending records maintain empty human labels and pending status
4. Audit Metrics Breakdown:
   - AI suggestions completed vs failed
   - Human reviewed vs pending
   - Accepted AI suggestions vs Overrides vs Unclear
   - Priority distribution and QC sample counts

Usage:
    # Audit AI suggestions or review queue:
    python -m backend.scripts.validate_golden_dataset --input data/golden/golden_set_review_queue.csv

    # Enforce 100% human annotation completion on final dataset:
    python -m backend.scripts.validate_golden_dataset --input data/golden/golden_set_human_review.csv --require-complete
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from app.core.logging import configure_logging
    from app.evaluation.human_review import (
        get_allowed_taxonomy_labels,
        is_valid_ai_suggestion,
    )
    from app.evaluation.label_validation import (
        detect_class_imbalance,
        validate_no_duplicate_normalized_messages,
        validate_no_duplicate_tweet_ids,
        validate_required_columns,
    )
    from app.nlp.taxonomy_finalization import (
        DEFAULT_CANDIDATE_TAXONOMY_PATH,
        create_initial_candidate_taxonomy,
        load_candidate_taxonomy,
    )
except ModuleNotFoundError:
    from backend.app.core.logging import configure_logging  # type: ignore[no-redef]
    from backend.app.evaluation.human_review import (  # type: ignore[no-redef]
        get_allowed_taxonomy_labels,
        is_valid_ai_suggestion,
    )
    from backend.app.evaluation.label_validation import (  # type: ignore[no-redef]
        detect_class_imbalance,
        validate_no_duplicate_normalized_messages,
        validate_no_duplicate_tweet_ids,
        validate_required_columns,
    )
    from backend.app.nlp.taxonomy_finalization import (  # type: ignore[no-redef]
        DEFAULT_CANDIDATE_TAXONOMY_PATH,
        create_initial_candidate_taxonomy,
        load_candidate_taxonomy,
    )

logger = logging.getLogger("validate_golden_dataset")

DEFAULT_EXPECTED_COUNT = 200


@dataclass
class DatasetAuditReport:
    """Comprehensive validation and audit report."""
    is_valid: bool
    total_records: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    audit_metrics: dict[str, Any] = field(default_factory=dict)
    label_distribution: dict[str, int] = field(default_factory=dict)
    priority_distribution: dict[str, int] = field(default_factory=dict)


def audit_golden_dataset(
    df: pd.DataFrame,
    allowed_labels: set[str],
    expected_count: Optional[int] = DEFAULT_EXPECTED_COUNT,
    require_complete: bool = False,
) -> DatasetAuditReport:
    """
    Execute full dataset integrity and scientific validation checks.
    """
    errors: list[str] = []
    warnings: list[str] = []
    total = len(df)

    # 1. Expected record count
    if expected_count is not None and total != expected_count:
        errors.append(f"Record count mismatch: expected {expected_count} records, found {total}.")

    # 2. Required columns check
    col_errors = validate_required_columns(df)
    errors.extend(col_errors)
    if col_errors:
        return DatasetAuditReport(
            is_valid=False,
            total_records=total,
            errors=errors,
            warnings=warnings,
            audit_metrics={},
            label_distribution={},
            priority_distribution={},
        )

    # 3. Duplicate checks
    errors.extend(validate_no_duplicate_tweet_ids(df))
    errors.extend(validate_no_duplicate_normalized_messages(df))

    # Golden ID uniqueness
    if "golden_id" in df.columns:
        gids = df["golden_id"].fillna("").astype(str).str.strip()
        empty_gids = gids == ""
        if empty_gids.any():
            errors.append(f"Found {int(empty_gids.sum())} row(s) with empty golden_id.")
        dupe_gids = df[df.duplicated(subset=["golden_id"], keep=False)]
        if not dupe_gids.empty:
            errors.append(f"Found duplicate golden_id(s): {dupe_gids['golden_id'].unique().tolist()[:5]}")

    # 4. Check for invalid string representations ('nan', 'none', 'null', 'undefined')
    check_cols = ["golden_id", "annotation_label", "annotation_status", "annotator", "model_suggested_label"]
    for col in check_cols:
        if col in df.columns:
            for idx, val in enumerate(df[col]):
                if val is not None and not pd.isna(val):
                    val_str = str(val).strip().lower()
                    if val_str in ("nan", "none", "null", "undefined"):
                        errors.append(
                            f"Row {idx} ({df.at[idx, 'golden_id']}): Column '{col}' contains illegal literal string '{val}'."
                        )

    # 5. Scientific Integrity & Separation Verification
    ai_suggestions_valid = 0
    ai_suggestions_failed = 0
    human_reviewed = 0
    accepted_ai_suggestions = 0
    calibration_accepted_records = 0
    human_overrides = 0
    unclear_records = 0
    group_approved_records = 0
    individual_reviewed_records = 0
    auto_copy_violations = 0

    human_labels_count: dict[str, int] = {}
    priority_counts: dict[str, int] = {}
    qc_sample_count = 0

    has_priority_col = "priority" in df.columns
    has_qc_col = "qc_sample" in df.columns
    has_review_mode_col = "review_mode" in df.columns
    has_approval_type_col = "approval_type" in df.columns
    has_calib_status_col = "calibration_status" in df.columns
    has_calib_label_col = "calibration_candidate_label" in df.columns

    valid_review_modes = {"individual_human_review", "group_human_approval", ""}
    valid_approval_types = {
        "accepted_ai_suggestion", "human_override", "group_approved_suggestion",
        "individual_calibration_acceptance", "unclear", ""
    }
    valid_calib_statuses = {
        "AGREES_WITH_AI", "DISAGREES_WITH_AI", "INSUFFICIENT_EVIDENCE", "AMBIGUOUS",
        "agrees_with_ai", "disagrees_with_ai", "insufficient_evidence", "ambiguous", ""
    }

    for idx, row in df.iterrows():
        gid = str(row.get("golden_id", f"row_{idx}")).strip()

        # AI suggestion state
        raw_sug = row.get("model_suggested_label")
        raw_status = row.get("suggestion_status")
        sug_label = str(raw_sug).strip() if raw_sug is not None and not pd.isna(raw_sug) else ""
        status = str(raw_status).strip().lower() if raw_status is not None and not pd.isna(raw_status) else ""

        if is_valid_ai_suggestion(sug_label, allowed_labels, status):
            ai_suggestions_valid += 1
        else:
            ai_suggestions_failed += 1

        # Calibration state validation
        if has_calib_status_col:
            raw_cstat = row.get("calibration_status")
            cstat = str(raw_cstat).strip() if raw_cstat is not None and not pd.isna(raw_cstat) else ""
            if cstat and cstat not in valid_calib_statuses:
                errors.append(f"Row {idx} ({gid}): Invalid calibration_status '{cstat}'.")

        if has_calib_label_col:
            raw_clbl = row.get("calibration_candidate_label")
            clbl = str(raw_clbl).strip() if raw_clbl is not None and not pd.isna(raw_clbl) else ""
            if clbl and clbl not in allowed_labels:
                errors.append(f"Row {idx} ({gid}): Calibrated candidate label '{clbl}' not in approved taxonomy.")

        # Human annotation state
        raw_hlbl = row.get("annotation_label")
        raw_hstat = row.get("annotation_status")
        raw_annotator = row.get("annotator")
        raw_rmode = row.get("review_mode")
        raw_apptype = row.get("approval_type")

        h_label = str(raw_hlbl).strip() if raw_hlbl is not None and not pd.isna(raw_hlbl) else ""
        h_status = str(raw_hstat).strip().lower() if raw_hstat is not None and not pd.isna(raw_hstat) else ""
        h_annotator = str(raw_annotator).strip() if raw_annotator is not None and not pd.isna(raw_annotator) else ""
        h_rmode = str(raw_rmode).strip().lower() if raw_rmode is not None and not pd.isna(raw_rmode) else ""
        h_apptype = str(raw_apptype).strip().lower() if raw_apptype is not None and not pd.isna(raw_apptype) else ""

        is_reviewed = bool(h_label and h_status not in ("", "pending", "pending_human_review"))

        if is_reviewed:
            human_reviewed += 1
            # Verify valid human label
            if h_label not in allowed_labels:
                errors.append(f"Row {idx} ({gid}): Human annotation_label '{h_label}' not in approved taxonomy.")

            # Verify annotator is non-empty
            if not h_annotator:
                errors.append(f"Row {idx} ({gid}): Reviewed record has empty annotator field.")

            # Verify review_mode if present
            if has_review_mode_col and h_rmode and h_rmode not in valid_review_modes:
                errors.append(f"Row {idx} ({gid}): Invalid review_mode '{h_rmode}'.")

            # Verify approval_type if present
            if has_approval_type_col and h_apptype and h_apptype not in valid_approval_types:
                errors.append(f"Row {idx} ({gid}): Invalid approval_type '{h_apptype}'.")

            if h_rmode == "group_human_approval" or h_apptype == "group_approved_suggestion":
                group_approved_records += 1
            else:
                individual_reviewed_records += 1

            # Track distribution
            human_labels_count[h_label] = human_labels_count.get(h_label, 0) + 1

            if h_apptype == "individual_calibration_acceptance":
                calibration_accepted_records += 1
            elif h_label == "unclear_needs_review":
                unclear_records += 1
            elif h_status == "overridden_ai_suggestion" or (sug_label and h_label != sug_label):
                human_overrides += 1
            elif h_status == "reviewed" and h_label == sug_label:
                accepted_ai_suggestions += 1
            else:
                accepted_ai_suggestions += 1

        else:
            # Unreviewed record: ensure human fields are strictly empty and status is pending
            if h_label != "":
                auto_copy_violations += 1
                errors.append(
                    f"Row {idx} ({gid}): SCIENTIFIC INTEGRITY VIOLATION — Unreviewed record has non-empty annotation_label '{h_label}' with status '{h_status}'."
                )
            if h_annotator != "":
                errors.append(
                    f"Row {idx} ({gid}): Unreviewed record has non-empty annotator '{h_annotator}'."
                )
            if h_status not in ("pending_human_review", "pending", ""):
                warnings.append(
                    f"Row {idx} ({gid}): Unreviewed record has unusual status '{h_status}'."
                )
            if has_review_mode_col and h_rmode not in ("", "pending"):
                errors.append(
                    f"Row {idx} ({gid}): Unreviewed record has non-empty review_mode '{h_rmode}'."
                )

        # Priority tracking
        if has_priority_col:
            prio = str(row.get("priority", "")).strip().lower()
            if prio:
                priority_counts[prio] = priority_counts.get(prio, 0) + 1

        if has_qc_col:
            qc = str(row.get("qc_sample", "")).strip().lower() in ("true", "1", "yes")
            if qc:
                qc_sample_count += 1

    pending_human_review = total - human_reviewed

    if require_complete and pending_human_review > 0:
        errors.append(f"Dataset completion required: {pending_human_review} record(s) still pending human review.")

    if human_labels_count:
        warnings.extend(detect_class_imbalance(df[df["annotation_label"] != ""]))

    is_valid = len(errors) == 0

    audit_metrics = {
        "total_records": total,
        "valid_ai_suggestions": ai_suggestions_valid,
        "failed_or_missing_ai_suggestions": ai_suggestions_failed,
        "human_reviewed_records": human_reviewed,
        "pending_human_review_records": pending_human_review,
        "accepted_ai_suggestions": accepted_ai_suggestions,
        "calibration_accepted_records": calibration_accepted_records,
        "human_overrides": human_overrides,
        "marked_unclear": unclear_records,
        "group_approved_records": group_approved_records,
        "individual_reviewed_records": individual_reviewed_records,
        "qc_sampled_records": qc_sample_count,
        "scientific_integrity_intact": (auto_copy_violations == 0),
    }

    return DatasetAuditReport(
        is_valid=is_valid,
        total_records=total,
        errors=errors,
        warnings=warnings,
        audit_metrics=audit_metrics,
        label_distribution=human_labels_count,
        priority_distribution=priority_counts,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 5.6 Golden Dataset Validation & Scientific Audit Engine"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the Golden Dataset CSV to validate",
    )
    parser.add_argument(
        "--expected-records",
        type=int,
        default=DEFAULT_EXPECTED_COUNT,
        help=f"Expected total record count (default: {DEFAULT_EXPECTED_COUNT})",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Enforce 100% human annotation completion (fail if any record is pending)",
    )
    parser.add_argument(
        "--taxonomy-path",
        type=str,
        default=DEFAULT_CANDIDATE_TAXONOMY_PATH,
        help=f"Path to candidate taxonomy JSON (default: {DEFAULT_CANDIDATE_TAXONOMY_PATH})",
    )
    parser.add_argument(
        "--json-output",
        type=str,
        default=None,
        help="Optional path to output validation report as JSON",
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
    if not input_path.exists():
        logger.error("Input file does not exist: %s", input_path)
        return 1

    tax_path = Path(args.taxonomy_path)
    allowed_labels = get_allowed_taxonomy_labels(tax_path)

    df = pd.read_csv(input_path, dtype=str)

    report = audit_golden_dataset(
        df=df,
        allowed_labels=allowed_labels,
        expected_count=args.expected_records,
        require_complete=args.require_complete,
    )

    # Print clean formatted summary
    print("\n" + "=" * 60)
    print("PHASE 5.6 — GOLDEN DATASET SCIENTIFIC AUDIT REPORT")
    print("=" * 60)
    print(f"Target File:                           {input_path}")
    print(f"Total Records:                         {report.total_records}")
    print(f"Valid AI Suggestions:                  {report.audit_metrics.get('valid_ai_suggestions', 0)}")
    print(f"Failed/Missing AI Suggestions:         {report.audit_metrics.get('failed_or_missing_ai_suggestions', 0)}")
    print("-" * 60)
    print(f"Human Reviewed Records:                {report.audit_metrics.get('human_reviewed_records', 0)}")
    print(f"  - Individual Human Reviews:          {report.audit_metrics.get('individual_reviewed_records', 0)}")
    print(f"  - Group Human Approvals:             {report.audit_metrics.get('group_approved_records', 0)}")
    print(f"Pending Human Review:                  {report.audit_metrics.get('pending_human_review_records', 0)}")
    print(f"  - Accepted AI Suggestions:           {report.audit_metrics.get('accepted_ai_suggestions', 0)}")
    print(f"  - Accepted Calibrated Labels:        {report.audit_metrics.get('calibration_accepted_records', 0)}")
    print(f"  - Human Overrides:                   {report.audit_metrics.get('human_overrides', 0)}")
    print(f"  - Marked 'unclear_needs_review':     {report.audit_metrics.get('marked_unclear', 0)}")
    print("-" * 60)

    if report.priority_distribution:
        print("Priority Breakdown:")
        for p_name, p_cnt in sorted(report.priority_distribution.items()):
            print(f"  - {p_name.upper():<12}: {p_cnt}")
        print(f"  - QC SAMPLED LOW: {report.audit_metrics.get('qc_sampled_records', 0)}")
        print("-" * 60)

    if report.label_distribution:
        print("Human Ground Truth Distribution:")
        for lbl, cnt in sorted(report.label_distribution.items()):
            print(f"  - {lbl:<34}: {cnt}")
        print("-" * 60)

    print(f"Scientific Integrity Status:           {'[PASS] INTACT' if report.audit_metrics.get('scientific_integrity_intact') else '[FAIL] VIOLATION'}")
    print("=" * 60)

    if report.warnings:
        print(f"\nWarnings ({len(report.warnings)}):")
        for w in report.warnings[:10]:
            print(f"  [WARN] {w}")

    if not report.is_valid:
        print(f"\nVALIDATION FAILED with {len(report.errors)} error(s):")
        for err in report.errors[:15]:
            print(f"  [ERROR] {err}")
        if len(report.errors) > 15:
            print(f"  ... and {len(report.errors) - 15} additional error(s).")
        print("=" * 60 + "\n")
        return 1

    print("\n[PASS] DATASET VALIDATION PASSED: Conforms to scientific integrity and golden specification.\n")

    if args.json_output:
        json_path = Path(args.json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, indent=2)
        print(f"Saved audit report JSON to: {json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
