"""
SupportGraph AI — Evaluate Multi-Signal Ambiguity Gate & Strategy Benchmark (Phase 7.1)

Evaluates and compares:
1. Strategy A: Confidence-Only Routing
2. Strategy B: Phase 7 Ambiguity Classification
3. Strategy C: Phase 7.1 Multi-Signal Safe Decision Gate

Usage:
  python -m backend.scripts.evaluate_ambiguity_gate
  python -m backend.scripts.evaluate_ambiguity_gate --provider heuristic
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

try:
    from app.core.config import settings
    from app.core.logging import get_logger
    from app.evaluation.ambiguity_gate_evaluator import (
        AmbiguityGateEvaluationReport,
        AmbiguityGateEvaluator,
    )
    from app.intent.classifier import TopKIntentClassifier
    from app.schemas.intent_routing import IntentAnalysis
except ModuleNotFoundError:
    from backend.app.core.config import settings  # type: ignore[no-redef]
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]
    from backend.app.evaluation.ambiguity_gate_evaluator import (  # type: ignore[no-redef]
        AmbiguityGateEvaluationReport,
        AmbiguityGateEvaluator,
    )
    from backend.app.intent.classifier import TopKIntentClassifier  # type: ignore[no-redef]
    from backend.app.schemas.intent_routing import IntentAnalysis  # type: ignore[no-redef]

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="SupportGraph AI — Phase 7.1 Ambiguity Gate Evaluator")
    parser.add_argument(
        "--golden-path",
        type=str,
        default="data/golden/golden_set_human_review.csv",
        help="Path to golden human review dataset CSV",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="artifacts/reports/routing_strategy_comparison.json",
        help="Path to save strategy comparison JSON",
    )
    parser.add_argument(
        "--signal-json",
        type=str,
        default="artifacts/reports/ambiguity_signal_analysis.json",
        help="Path to save signal analysis JSON",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default="reports/phase_7_1_ambiguity_detection_calibration.md",
        help="Path to save markdown evaluation report",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        choices=["groq", "ollama", "heuristic"],
        help="Classifier provider override ('heuristic', 'groq', 'ollama')",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Limit number of records to evaluate",
    )

    args = parser.parse_args()

    golden_path = Path(args.golden_path)
    if not golden_path.exists():
        raise FileNotFoundError(f"Golden dataset not found: {golden_path}")

    df = pd.read_csv(golden_path)
    if args.max_records:
        df = df.head(args.max_records)

    # Initialize classifier
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
                        - (clf.classify_heuristic_fallback(customer_message, requested_k=top_k or 3)[1].confidence if len(clf.classify_heuristic_fallback(customer_message, requested_k=top_k or 3)) > 1 else 0.0),
                    ),
                    4,
                ),
                normalized_entropy=0.10,
                model_name="heuristic_baseline",
            )
        )
    else:
        clf = TopKIntentClassifier(provider=args.provider)

    evaluator = AmbiguityGateEvaluator(classifier=clf)
    report: AmbiguityGateEvaluationReport = evaluator.evaluate_dataframe(df)

    # Print summary markdown
    print("\n" + report.summary_markdown() + "\n")

    # Save Strategy Comparison JSON
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)
    print(f"Saved Strategy Comparison JSON to: {out_json}")

    # Save Signal Analysis JSON
    sig_json = Path(args.signal_json)
    sig_json.parent.mkdir(parents=True, exist_ok=True)
    with open(sig_json, "w", encoding="utf-8") as f:
        json.dump([asdict(s) if hasattr(s, "__dataclass_fields__") else s for s in report.signal_interception_analysis], f, indent=2)
    print(f"Saved Signal Analysis JSON to: {sig_json}")

    # Save Markdown report
    if args.output_md:
        out_md = Path(args.output_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        with open(out_md, "w", encoding="utf-8") as f:
            f.write(report.summary_markdown() + "\n")
        print(f"Saved Markdown Report to: {out_md}")


if __name__ == "__main__":
    from dataclasses import asdict
    main()
