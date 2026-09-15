"""
SupportGraph AI — Evaluate Evidence-Aware Support Resolution (Phase 7)

Runs the full Phase 7 evaluation benchmark against completed human ground-truth records:
- Problem Understanding & Extraction
- Primary Intent Disambiguation vs Contextual Cause
- Ambiguity Classification & Escalation Safety
- Historical Case Retrieval & Operational Tier Categorization
- Human Assistance Coverage (Ground Truth in Top-2 Candidates)

Usage:
  python -m backend.scripts.evaluate_evidence_routing
  python -m backend.scripts.evaluate_evidence_routing --golden-path data/golden/golden_set_human_review.csv
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
    from app.evaluation.evidence_routing_evaluator import (
        EvidenceRoutingEvaluationReport,
        EvidenceRoutingEvaluator,
    )
    from app.resolution.support_resolution_engine import SupportResolutionEngine
except ModuleNotFoundError:
    from backend.app.core.config import settings  # type: ignore[no-redef]
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]
    from backend.app.evaluation.evidence_routing_evaluator import (  # type: ignore[no-redef]
        EvidenceRoutingEvaluationReport,
        EvidenceRoutingEvaluator,
    )
    from backend.app.resolution.support_resolution_engine import (  # type: ignore[no-redef]
        SupportResolutionEngine,
    )

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="SupportGraph AI — Phase 7 Evidence-Aware Support Resolution Evaluator")
    parser.add_argument(
        "--golden-path",
        type=str,
        default="data/golden/golden_set_human_review.csv",
        help="Path to human-reviewed golden dataset CSV",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="artifacts/reports/phase_7_evidence_routing_evaluation.json",
        help="Path to save output JSON evaluation report",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default="reports/phase_7_evidence_aware_support_resolution.md",
        help="Path to save markdown benchmark report",
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
        help="Limit number of records to evaluate (for testing)",
    )

    args = parser.parse_args()

    golden_path = Path(args.golden_path)
    if not golden_path.exists():
        raise FileNotFoundError(f"Golden dataset not found: {golden_path}")

    df = pd.read_csv(golden_path)
    if args.max_records:
        df = df.head(args.max_records)

    from backend.app.intent.classifier import TopKIntentClassifier
    from backend.app.schemas.intent_routing import IntentAnalysis

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

    engine = SupportResolutionEngine(classifier=clf)
    evaluator = EvidenceRoutingEvaluator(engine=engine)

    report: EvidenceRoutingEvaluationReport = evaluator.evaluate_dataframe(df)

    # Print summary
    print("\n" + report.summary_markdown() + "\n")

    # Save JSON report
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)
    print(f"Saved evaluation JSON report to: {out_json}")

    # Save Markdown report
    if args.output_md:
        out_md = Path(args.output_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        with open(out_md, "w", encoding="utf-8") as f:
            f.write(report.summary_markdown() + "\n")
        print(f"Saved evaluation Markdown report to: {out_md}")


if __name__ == "__main__":
    main()
