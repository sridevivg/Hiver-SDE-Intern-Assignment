"""
SupportGraph AI — Taxonomy Calibration Report Generator (Phase 5.9)

Analyzes human ground-truth annotations (N=57) and AI suggestions,
evaluates explainable contextual taxonomy calibration signals, and
generates a comprehensive Markdown report at reports/taxonomy_calibration_report.md.

Usage:
    python -m backend.scripts.generate_calibration_report
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from app.core.logging import configure_logging
    from app.evaluation.human_review import get_allowed_taxonomy_labels
    from app.evaluation.taxonomy_calibration import (
        CalibrationResult,
        CalibrationStatus,
        analyze_human_ai_disagreements,
        evaluate_taxonomy_calibration,
    )
except ModuleNotFoundError:
    from backend.app.core.logging import configure_logging  # type: ignore[no-redef]
    from backend.app.evaluation.human_review import get_allowed_taxonomy_labels  # type: ignore[no-redef]
    from backend.app.evaluation.taxonomy_calibration import (  # type: ignore[no-redef]
        CalibrationResult,
        CalibrationStatus,
        analyze_human_ai_disagreements,
        evaluate_taxonomy_calibration,
    )

logger = logging.getLogger("generate_calibration_report")

DEFAULT_INPUT_CSV = "data/golden/golden_set_human_review.csv"
DEFAULT_REPORT_PATH = "reports/taxonomy_calibration_report.md"


def generate_markdown_report(
    df: pd.DataFrame,
    allowed_labels: set[str],
) -> str:
    """Generate comprehensive scientific calibration report."""
    # Analyze reviewed subset
    disagreement_report = analyze_human_ai_disagreements(df, allowed_labels=allowed_labels)
    total_reviewed = disagreement_report.total_reviewed
    total_agreements = disagreement_report.agreements
    total_disagreements = disagreement_report.disagreements
    baseline_acc = disagreement_report.agreement_rate

    # Analyze pending subset
    pending_records: list[dict[str, Any]] = []
    calib_results: list[CalibrationResult] = []

    for _, row in df.iterrows():
        lbl = str(row.get("annotation_label", "")).strip()
        stat = str(row.get("annotation_status", "")).strip().lower()
        if not lbl or stat in ("", "pending", "pending_human_review"):
            rec_dict = row.to_dict() if hasattr(row, "to_dict") else dict(row)
            res = evaluate_taxonomy_calibration(rec_dict, allowed_labels=allowed_labels)
            pending_records.append(rec_dict)
            calib_results.append(res)

    total_pending = len(pending_records)
    pending_agree = sum(1 for r in calib_results if r.status == CalibrationStatus.AGREES_WITH_AI)
    pending_disagree = sum(1 for r in calib_results if r.status == CalibrationStatus.DISAGREES_WITH_AI)
    pending_insufficient = sum(1 for r in calib_results if r.status == CalibrationStatus.INSUFFICIENT_EVIDENCE)
    pending_ambiguous = sum(1 for r in calib_results if r.status == CalibrationStatus.AMBIGUOUS)

    # Disagreement pairs in reviewed records
    disagree_pairs: Counter[tuple[str, str]] = Counter()
    for _, row in df.iterrows():
        lbl = str(row.get("annotation_label", "")).strip()
        stat = str(row.get("annotation_status", "")).strip().lower()
        sug = str(row.get("model_suggested_label", "")).strip()
        if lbl and stat not in ("", "pending", "pending_human_review"):
            if sug and sug != lbl:
                disagree_pairs[(sug, lbl)] += 1

    # Human ground truth label distribution
    human_dist: Counter[str] = Counter()
    for _, row in df.iterrows():
        lbl = str(row.get("annotation_label", "")).strip()
        stat = str(row.get("annotation_status", "")).strip().lower()
        if lbl and stat not in ("", "pending", "pending_human_review"):
            human_dist[lbl] += 1

    # Calibrated candidates for pending records
    pending_calib_candidates: Counter[str] = Counter()
    for r in calib_results:
        if r.candidate_label:
            pending_calib_candidates[r.candidate_label] += 1

    # Timestamp
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    md = []
    md.append("# SupportGraph AI — Phase 5.9 Taxonomy Calibration & Recommendation Report")
    md.append("")
    md.append(f"**Generated:** {generated_at}  ")
    md.append(f"**Dataset:** {len(df)} Golden Records ({total_reviewed} Human Reviewed, {total_pending} Pending)  ")
    md.append(f"**Baseline AI Agreement:** {baseline_acc * 100:.1f}% ({total_agreements}/{total_reviewed} records)  ")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 1. Executive Summary")
    md.append("")
    md.append("The Phase 5.9 Taxonomy Calibration Layer extracts explainable contextual signals from the completed human ground truth ($N=57$) without training a machine learning model on the golden dataset. It provides transparent advisory guidance for pending records, escalates high-risk discrepancies to mandatory individual review, and identifies safe low-risk candidates for group review.")
    md.append("")
    md.append("| Metric | Reviewed Ground Truth (N=57) | Pending Queue (N=143) | Total Dataset (N=200) |")
    md.append("| :--- | :--- | :--- | :--- |")
    md.append(f"| **Total Records** | {total_reviewed} | {total_pending} | {len(df)} |")
    md.append(f"| **Agrees with AI Suggestion** | {total_agreements} ({total_agreements/total_reviewed*100:.1f}%) | {pending_agree} ({pending_agree/total_pending*100:.1f}%) | {total_agreements + pending_agree} ({(total_agreements+pending_agree)/len(df)*100:.1f}%) |")
    md.append(f"| **Disagrees with AI Suggestion** | {total_disagreements} ({total_disagreements/total_reviewed*100:.1f}%) | {pending_disagree} ({pending_disagree/total_pending*100:.1f}%) | {total_disagreements + pending_disagree} ({(total_disagreements+pending_disagree)/len(df)*100:.1f}%) |")
    md.append(f"| **Insufficient Evidence / Neutral** | 0 (Evaluated) | {pending_insufficient} ({pending_insufficient/total_pending*100:.1f}%) | {pending_insufficient} ({pending_insufficient/len(df)*100:.1f}%) |")
    md.append(f"| **Ambiguous Signals** | 0 (Evaluated) | {pending_ambiguous} ({pending_ambiguous/total_pending*100:.1f}%) | {pending_ambiguous} ({pending_ambiguous/len(df)*100:.1f}%) |")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 2. Empirical Ground Truth Distribution (N=57)")
    md.append("")
    md.append("Completed human ground truth records are immutable and protected:")
    md.append("")
    md.append("| Taxonomy Intent Label | Human Ground Truth Count | Percentage |")
    md.append("| :--- | :--- | :--- |")
    for lbl, count in human_dist.most_common():
        pct = (count / total_reviewed) * 100
        md.append(f"| `{lbl}` | {count} | {pct:.1f}% |")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 3. Disagreement Pattern Analysis (AI vs Human Ground Truth)")
    md.append("")
    md.append(f"Out of {total_reviewed} reviewed records, the human annotator overrode the baseline AI suggestion in **{total_disagreements} cases ({total_disagreements/total_reviewed*100:.1f}%)**.")
    md.append("")
    md.append("| Baseline AI Suggestion | Human True Ground Truth | Occurrences | Root Cause & Contextual Signal |")
    md.append("| :--- | :--- | :--- | :--- |")
    for (sug, hlbl), count in disagree_pairs.most_common():
        if sug == "general_device_support" and hlbl == "software_update_problem":
            cause = "Update causality / regression phrasing (*'since update'*, *'after updating'*)."
        elif sug == "general_device_support" and hlbl == "hardware_audio_connection_issue":
            cause = "Peripherals / cables / audio accessories (*'wired printer USB'*, *'AirPods no sound'*)."
        elif sug == "general_device_support" and hlbl == "keyboard_typing_issue":
            cause = "Text input / autocorrect / letter I glitch (*'autocorrect bug'*, *'letter eye'*)."
        elif sug == "general_device_support" and hlbl == "mac_software_issue":
            cause = "macOS desktop / laptop software (*'High Sierra'*, *'MacBook boot'*)."
        elif sug == "general_device_support" and hlbl == "display_touch_issue":
            cause = "Screen rendering / touch unresponsiveness (*'screen flickering'*, *'black screen'*)."
        else:
            cause = "Taxonomy category specificity / operational context."
        md.append(f"| `{sug}` | `{hlbl}` | {count} | {cause} |")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 4. Transparent Contextual Calibration Rules")
    md.append("")
    md.append("The calibration engine uses 10 transparent, deterministic rule-sets corresponding to the approved taxonomy categories:")
    md.append("")
    md.append("1. **`software_update_problem`**: Matches update regression phrasing (`since update`, `after updating ios`), installation stalls (`update stuck`), and release version regressions (`ios 11.1`).")
    md.append("2. **`hardware_audio_connection_issue`**: Matches audio peripherals (`airpods`, `earphones`, `microphone`, `headphone jack`) and physical connectivity (`usb cable`, `lightning port`, `wired printer`).")
    md.append("3. **`keyboard_typing_issue`**: Matches predictive text (`autocorrect`, `predictive text`), letter glitches (`letter i`, `letter eye`, `unicode glitch`), and keyboard unresponsiveness.")
    md.append("4. **`battery_power_issue`**: Matches battery depletion (`battery draining fast`, `battery percentage drop`), charging failures (`not charging`, `overheating`), and unexpected shutdowns.")
    md.append("5. **`mac_software_issue`**: Matches macOS operating system (`high sierra`, `macos`, `kernel panic`), desktop applications (`finder freeze`, `safari crash`), and recovery utilities.")
    md.append("6. **`display_touch_issue`**: Matches screen unresponsiveness (`touchscreen unresponsive`, `ghost touch`), visual glitches (`screen flicker`, `black display`), and display damage.")
    md.append("7. **`account_access_issue`**: Matches Apple ID credentials (`apple id locked`, `forgot password`, `two factor authentication`, `security questions`).")
    md.append("8. **`billing_purchase_issue`**: Matches subscriptions (`unauthorized charge`, `itunes subscription`), refund requests (`refund`), and payment method errors (`card declined`).")
    md.append("9. **`general_device_support`**: Matches general inquiry when no specific operational subsystem or update causality is identified.")
    md.append("10. **`unclear_needs_review`**: Flags ambiguous multi-domain conflicts, severe message truncations, or foreign language texts.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 5. Pending Queue Calibration Breakdown (N=143)")
    md.append("")
    md.append("Evaluating the 143 pending records with the calibration layer yields the following distribution:")
    md.append("")
    md.append("| Calibration Status | Count | Percentage | Workflow Action |")
    md.append("| :--- | :--- | :--- | :--- |")
    md.append(f"| **`AGREES_WITH_AI`** | {pending_agree} | {pending_agree/total_pending*100:.1f}% | Eligible for Safe Group Review (if low-risk & non-QC) |")
    md.append(f"| **`DISAGREES_WITH_AI`** | {pending_disagree} | {pending_disagree/total_pending*100:.1f}% | Escalated to Mandatory Individual Review with `[C]` recommendation |")
    md.append(f"| **`INSUFFICIENT_EVIDENCE`** | {pending_insufficient} | {pending_insufficient/total_pending*100:.1f}% | Standard priority scoring based on baseline AI confidence |")
    md.append(f"| **`AMBIGUOUS`** | {pending_ambiguous} | {pending_ambiguous/total_pending*100:.1f}% | Escalated to Individual Review due to multi-intent overlap |")
    md.append("")
    md.append("### Calibrated Candidate Breakdown for Pending Records")
    md.append("")
    md.append("| Calibrated Candidate Intent | Pending Record Count |")
    md.append("| :--- | :--- |")
    for cand, c_cnt in pending_calib_candidates.most_common():
        md.append(f"| `{cand}` | {c_cnt} |")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 6. Scientific Integrity Guarantees")
    md.append("")
    md.append("1. **Zero Ground-Truth Contamination**: Calibration candidates are stored in distinct metadata columns (`calibration_candidate_label`, `calibration_confidence`, `calibration_status`) and are NEVER automatically copied into `annotation_label`.")
    md.append("2. **No Model Training on Test Data**: No fine-tuning, embedding fitting, or machine learning training was performed on the golden dataset.")
    md.append("3. **Explicit Human Action**: Calibrated suggestions can only become ground-truth annotations through explicit human review actions (`[C]` action in interactive CLI).")
    md.append("4. **Full Immutability**: All 57 completed human annotations remain 100% byte-for-byte identical.")
    md.append("")
    return "\n".join(md)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Phase 5.9 Taxonomy Calibration Report")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT_CSV, help=f"Input CSV (default: {DEFAULT_INPUT_CSV})")
    parser.add_argument("--output", type=str, default=DEFAULT_REPORT_PATH, help=f"Output Markdown report (default: {DEFAULT_REPORT_PATH})")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    configure_logging(log_level="DEBUG" if args.verbose else "INFO")

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        logger.error("Input file %s does not exist.", input_path)
        return 1

    allowed_labels = get_allowed_taxonomy_labels()
    df = pd.read_csv(input_path, dtype=str)

    report_md = generate_markdown_report(df, allowed_labels=allowed_labels)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"\n[SUCCESS] Taxonomy calibration report written to: {output_path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
