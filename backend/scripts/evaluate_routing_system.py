"""
SupportGraph AI — Evaluate Routing System Benchmark (Phase 6)

Runs full routing and intent classification evaluation against human ground truth records.
Calculates Top-1/Top-2/Top-3 accuracy, auto-handle precision, error rate, escalation safety,
and human assistance value.

Usage:
  PYTHONPATH=. python -m backend.scripts.evaluate_routing_system
  PYTHONPATH=. python -m backend.scripts.evaluate_routing_system --provider heuristic
  PYTHONPATH=. python -m backend.scripts.evaluate_routing_system --max-records 25
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
    from app.evaluation.routing_evaluator import RoutingSystemEvaluator
    from app.intent.classifier import TopKIntentClassifier
    from app.intent.router import IntentRouter
except ModuleNotFoundError:
    from backend.app.core.config import settings  # type: ignore[no-redef]
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]
    from backend.app.evaluation.routing_evaluator import RoutingSystemEvaluator  # type: ignore[no-redef]
    from backend.app.intent.classifier import TopKIntentClassifier  # type: ignore[no-redef]
    from backend.app.intent.router import IntentRouter  # type: ignore[no-redef]

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="SupportGraph AI — Routing System Evaluator")
    parser.add_argument(
        "--golden-path",
        type=str,
        default="data/golden/golden_set_human_review.csv",
        help="Path to golden human review dataset CSV",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="artifacts/reports/routing_evaluation_report.json",
        help="Path to save output JSON evaluation report",
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
        help="Maximum records to evaluate (useful for quick smoke tests)",
    )

    args = parser.parse_args()

    # Build classifier
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
                normalized_entropy=compute_normalized_entropy([p.confidence for p in clf.classify_heuristic_fallback(customer_message, requested_k=top_k or 3)]),
                model_name="heuristic_baseline",
            )
        )
    else:
        clf = TopKIntentClassifier(provider=args.provider)

    router = IntentRouter()
    evaluator = RoutingSystemEvaluator(classifier=clf, router=router)

    golden_p = Path(args.golden_path)
    if not golden_p.exists():
        raise FileNotFoundError(f"Golden dataset not found: {golden_p}")

    df = pd.read_csv(golden_p)
    if args.max_records:
        df = df.head(args.max_records)

    report = evaluator.evaluate_dataframe(df)

    # Print markdown table summary
    print("\n" + report.summary_markdown() + "\n")

    # Save JSON report
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)

    print(f"Saved complete evaluation report to: {out_path}")


if __name__ == "__main__":
    try:
        from app.intent.classifier import compute_normalized_entropy
        from app.schemas.intent_routing import IntentAnalysis
    except ModuleNotFoundError:
        from backend.app.intent.classifier import compute_normalized_entropy  # type: ignore[no-redef]
        from backend.app.schemas.intent_routing import IntentAnalysis  # type: ignore[no-redef]
    main()
