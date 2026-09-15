"""
SupportGraph AI — Data-Driven Routing Calibration CLI (Phase 6.1)

Performs threshold grid search, safety optimization, and baseline comparison
over protected human-reviewed ground truth records.

Usage:
  PYTHONPATH=. python -m backend.scripts.calibrate_routing_thresholds --provider heuristic
  PYTHONPATH=. python -m backend.scripts.calibrate_routing_thresholds --input data/golden/golden_set_human_review.csv --output reports/routing_calibration
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

try:
    from app.core.config import settings
    from app.core.logging import get_logger
    from app.evaluation.routing_calibrator import (
        CalibrationReport,
        RoutingCalibrator,
    )
    from app.intent.classifier import TopKIntentClassifier, compute_normalized_entropy
    from app.schemas.intent_routing import IntentAnalysis
except ModuleNotFoundError:
    from backend.app.core.config import settings  # type: ignore[no-redef]
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]
    from backend.app.evaluation.routing_calibrator import (  # type: ignore[no-redef]
        CalibrationReport,
        RoutingCalibrator,
    )
    from backend.app.intent.classifier import (  # type: ignore[no-redef]
        TopKIntentClassifier,
        compute_normalized_entropy,
    )
    from backend.app.schemas.intent_routing import IntentAnalysis  # type: ignore[no-redef]

logger = get_logger(__name__)


def compute_sha256(filepath: Path | str) -> str:
    """Compute SHA-256 checksum of a file."""
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    sha256 = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def export_calibration_artifacts(report: CalibrationReport, output_dir: Path) -> dict[str, Path]:
    """
    Save all required JSON artifacts for the calibration report.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    # 1. calibration_summary.json
    summary_path = output_dir / "calibration_summary.json"
    summary_payload = {
        "total_evaluated_records": report.total_evaluated_records,
        "total_configurations_evaluated": report.total_configurations_evaluated,
        "unsafe_count": report.unsafe_count,
        "conservative_count": report.conservative_count,
        "balanced_count": report.balanced_count,
        "recommended_thresholds": {
            "confidence_threshold": report.recommended_configuration.conf_threshold,
            "margin_threshold": report.recommended_configuration.margin_threshold,
            "max_entropy": report.recommended_configuration.entropy_threshold,
        },
        "baseline_thresholds": {
            "confidence_threshold": report.baseline_configuration.conf_threshold,
            "margin_threshold": report.baseline_configuration.margin_threshold,
            "max_entropy": report.baseline_configuration.entropy_threshold,
        },
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)
    paths["calibration_summary"] = summary_path

    # 2. threshold_results.json
    results_path = output_dir / "threshold_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in report.all_results], f, indent=2)
    paths["threshold_results"] = results_path

    # 3. recommended_configuration.json
    rec_path = output_dir / "recommended_configuration.json"
    with open(rec_path, "w", encoding="utf-8") as f:
        json.dump(report.recommended_configuration.to_dict(), f, indent=2)
    paths["recommended_configuration"] = rec_path

    # 4. baseline_vs_calibrated.json
    cmp_path = output_dir / "baseline_vs_calibrated.json"
    cmp_payload = {
        "baseline": report.baseline_configuration.to_dict(),
        "recommended": report.recommended_configuration.to_dict(),
        "improvement": {
            "precision_change": round(
                report.recommended_configuration.auto_handle_precision
                - report.baseline_configuration.auto_handle_precision,
                4,
            ),
            "unsafe_auto_handle_reduction": (
                report.baseline_configuration.incorrect_auto_handles
                - report.recommended_configuration.incorrect_auto_handles
            ),
            "error_interception_change": round(
                report.recommended_configuration.error_interception_rate
                - report.baseline_configuration.error_interception_rate,
                4,
            ),
            "safety_score_change": round(
                report.recommended_configuration.safety_score
                - report.baseline_configuration.safety_score,
                4,
            ),
        },
    }
    with open(cmp_path, "w", encoding="utf-8") as f:
        json.dump(cmp_payload, f, indent=2)
    paths["baseline_vs_calibrated"] = cmp_path

    # 5. safety_analysis.json
    safety_path = output_dir / "safety_analysis.json"
    safety_payload = {
        "total_model_errors": (
            report.baseline_configuration.incorrect_auto_handles
            + report.baseline_configuration.intercepted_errors
        ),
        "baseline_unsafe_auto_handles": report.baseline_configuration.incorrect_auto_handles,
        "baseline_error_interception_rate": report.baseline_configuration.error_interception_rate,
        "calibrated_unsafe_auto_handles": report.recommended_configuration.incorrect_auto_handles,
        "calibrated_error_interception_rate": report.recommended_configuration.error_interception_rate,
        "unsafe_configurations_share": round(
            report.unsafe_count / report.total_configurations_evaluated, 4
        ),
    }
    with open(safety_path, "w", encoding="utf-8") as f:
        json.dump(safety_payload, f, indent=2)
    paths["safety_analysis"] = safety_path

    # 6. human_assistance_analysis.json
    assist_path = output_dir / "human_assistance_analysis.json"
    assist_payload = {
        "calibrated_escalations_count": report.recommended_configuration.escalate_count,
        "calibrated_top1_ground_truth_in_escalations": report.recommended_configuration.escalated_top1_correct,
        "calibrated_top2_ground_truth_in_escalations": report.recommended_configuration.escalated_top2_correct,
        "calibrated_top3_ground_truth_in_escalations": report.recommended_configuration.escalated_top3_correct,
        "calibrated_top2_assistance_rate": report.recommended_configuration.human_assistance_top2_rate,
        "calibrated_top3_assistance_rate": report.recommended_configuration.human_assistance_top3_rate,
    }
    with open(assist_path, "w", encoding="utf-8") as f:
        json.dump(assist_payload, f, indent=2)
    paths["human_assistance_analysis"] = assist_path

    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="SupportGraph AI — Routing Threshold Calibration CLI")
    parser.add_argument(
        "--input",
        type=str,
        default="data/golden/golden_set_human_review.csv",
        help="Path to human-reviewed golden dataset CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/routing_calibration",
        help="Directory to save calibration JSON artifacts",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="heuristic",
        choices=["heuristic", "groq", "ollama"],
        help="Classifier backend for pre-caching ('heuristic', 'groq', 'ollama')",
    )

    args = parser.parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output)

    # 1. Pre-analysis SHA-256 verification
    pre_sha = compute_sha256(input_path)
    print(f"\n[INTEGRITY] Pre-analysis SHA-256 for '{input_path}': {pre_sha}")

    # 2. Load dataset
    df = pd.read_csv(input_path)
    completed_mask = df["annotation_status"].isin(["reviewed", "overridden_ai_suggestion"]) & (
        df["annotation_label"].fillna("").astype(str).str.strip() != ""
    )
    completed_count = int(completed_mask.sum())
    print(f"[DATASET] Loaded {len(df)} total records ({completed_count} completed human annotations).")

    # 3. Initialize classifier & calibrator
    if args.provider == "heuristic":
        clf = TopKIntentClassifier(api_key="")
        clf.classify = lambda customer_message, normalized_message="", conversation_context="", top_k=3: (
            clf.classify_heuristic_fallback(customer_message, requested_k=top_k or 3)
            and IntentAnalysis(
                top_predictions=clf.classify_heuristic_fallback(customer_message, requested_k=top_k or 3),
                top_confidence=clf.classify_heuristic_fallback(customer_message, requested_k=top_k or 3)[0].confidence,
                confidence_margin=round(
                    max(
                        0.0,
                        clf.classify_heuristic_fallback(customer_message, requested_k=top_k or 3)[0].confidence
                        - (
                            clf.classify_heuristic_fallback(customer_message, requested_k=top_k or 3)[1].confidence
                            if len(clf.classify_heuristic_fallback(customer_message, requested_k=top_k or 3)) > 1
                            else 0.0
                        ),
                    ),
                    4,
                ),
                normalized_entropy=compute_normalized_entropy([
                    p.confidence
                    for p in clf.classify_heuristic_fallback(customer_message, requested_k=top_k or 3)
                ]),
                model_name="heuristic_baseline",
            )
        )
    else:
        clf = TopKIntentClassifier(provider=args.provider)

    calibrator = RoutingCalibrator(classifier=clf)

    # 4. Run grid search
    print("[CALIBRATION] Executing threshold grid search across 490 parameter configurations...")
    report = calibrator.run_grid_search(
        df,
        baseline_conf=settings.auto_handle_confidence_threshold,
        baseline_margin=settings.min_confidence_margin,
        baseline_entropy=settings.max_uncertainty_entropy,
    )

    # 5. Output comparison table
    border = "=" * 80
    print("\n" + border)
    print(" ROUTING THRESHOLD CALIBRATION & SAFETY COMPARISON")
    print(border + "\n")
    print(report.comparison_table_markdown())
    print("\n" + border)

    rec = report.recommended_configuration
    print(f"\n[RECOMMENDATION] {rec.rationale}")
    print(
        f"  • Recommended Confidence Threshold: {rec.conf_threshold:.2f}\n"
        f"  • Recommended Margin Threshold:     {rec.margin_threshold:.2f}\n"
        f"  • Recommended Max Entropy:          {rec.entropy_threshold:.2f}\n"
        f"  • Auto-Handle Precision:            {rec.auto_handle_precision*100:.1f}%\n"
        f"  • Unsafe Auto-Handles (Errors):     {rec.incorrect_auto_handles} (reduced from {report.baseline_configuration.incorrect_auto_handles})\n"
        f"  • Error Interception Rate:          {rec.error_interception_rate*100:.1f}%\n"
    )

    # 6. Export JSON artifacts
    paths = export_calibration_artifacts(report, output_dir)
    print("[ARTIFACTS] Exported calibration artifacts:")
    for name, p in paths.items():
        print(f"  - {name}: {p}")

    # 7. Post-analysis SHA-256 verification
    post_sha = compute_sha256(input_path)
    print(f"\n[INTEGRITY] Post-analysis SHA-256 for '{input_path}': {post_sha}")
    if pre_sha == post_sha:
        print(" [VERIFIED] Source dataset remained 100% byte-for-byte immutable.\n")
    else:
        raise RuntimeError("FATAL: Source dataset was modified during calibration analysis!")


if __name__ == "__main__":
    main()
