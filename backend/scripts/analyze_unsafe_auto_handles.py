"""
SupportGraph AI — Analyze Unsafe Auto-Handle Errors (Phase 7.1)

Categorizes and analyzes all remaining false-clarity auto-handled errors:
- TAXONOMY_LIMITATION
- MISSING_CONTEXT
- MULTIPLE_VALID_INTERPRETATIONS
- PROBLEM_EXTRACTION_ERROR
- RETRIEVAL_MISMATCH
- MODEL_CLASSIFICATION_ERROR
- FALSE_CLARITY_DECISION

Usage:
  python -m backend.scripts.analyze_unsafe_auto_handles
  python -m backend.scripts.analyze_unsafe_auto_handles --provider heuristic
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
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
    parser = argparse.ArgumentParser(description="SupportGraph AI — Analyze Unsafe Auto-Handles")
    parser.add_argument(
        "--golden-path",
        type=str,
        default="data/golden/golden_set_human_review.csv",
        help="Path to golden human review dataset CSV",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="artifacts/reports/unsafe_auto_handle_analysis.json",
        help="Path to save unsafe auto-handle analysis JSON",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="heuristic",
        choices=["groq", "ollama", "heuristic"],
        help="Classifier provider override",
    )

    args = parser.parse_args()

    golden_path = Path(args.golden_path)
    if not golden_path.exists():
        raise FileNotFoundError(f"Golden dataset not found: {golden_path}")

    df = pd.read_csv(golden_path)

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

    unsafe_cases = report.unsafe_auto_handles_analysis

    print("\n" + "=" * 80)
    print(" SUPPORTGRAPH AI — UNSAFE AUTO-HANDLE ROOT CAUSE ANALYSIS")
    print("=" * 80)
    print(f"Total Unsafe Auto-Handles Under Strategy C: {len(unsafe_cases)}\n")

    category_counts: dict[str, int] = {}
    for case in unsafe_cases:
        category_counts[case.error_category] = category_counts.get(case.error_category, 0) + 1

    print("Category Breakdown:")
    for cat, cnt in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  • {cat:<32} {cnt:>3} cases ({cnt/len(unsafe_cases)*100:.1f}%)" if unsafe_cases else f"  • {cat}: {cnt}")

    print("\nDetailed Case Log:")
    for idx, case in enumerate(unsafe_cases, 1):
        print(f"\nCase #{idx} [{case.golden_id}] - Category: {case.error_category}")
        print(f"  Message:     \"{case.customer_message[:80]}...\"")
        print(f"  Ground Truth: {case.ground_truth_intent}")
        print(f"  Predicted:    {case.predicted_intent} (Conf: {case.confidence * 100:.1f}%)")
        print(f"  Explanation:  {case.explanation}")

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in unsafe_cases], f, indent=2)
    print(f"\nSaved Unsafe Auto-Handle Analysis JSON to: {out_json}")


if __name__ == "__main__":
    main()
