"""
SupportGraph AI — Routing System Evaluator (Phase 6)

Evaluates the uncertainty-aware intent classification and routing pipeline
against human-reviewed ground truth records:
- Top-1, Top-2, and Top-3 accuracy
- Macro F1 and per-class precision/recall
- Auto-handle rate vs Escalation rate
- Precision and error rate among auto-handled decisions
- Escalation safety (percentage of misclassifications prevented)
- Human reviewer assistance value (how often true label is in Top-2 when escalated)
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from sklearn.metrics import classification_report, f1_score

try:
    from app.core.config import settings
    from app.core.logging import get_logger
    from app.intent.classifier import TopKIntentClassifier
    from app.intent.router import IntentRouter
    from app.schemas.intent_routing import IntentAnalysis, RoutingDecision, RoutingDecisionType
except ModuleNotFoundError:
    from backend.app.core.config import settings  # type: ignore[no-redef]
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]
    from backend.app.intent.classifier import TopKIntentClassifier  # type: ignore[no-redef]
    from backend.app.intent.router import IntentRouter  # type: ignore[no-redef]
    from backend.app.schemas.intent_routing import (  # type: ignore[no-redef]
        IntentAnalysis,
        RoutingDecision,
        RoutingDecisionType,
    )

logger = get_logger(__name__)


@dataclass
class RecordEvaluationResult:
    """Individual record evaluation result."""
    golden_id: str
    customer_message: str
    ground_truth_label: str
    top_1_intent: str
    top_1_confidence: float
    top_2_intent: str
    top_2_confidence: float
    confidence_margin: float
    normalized_entropy: Optional[float]
    routing_decision: str
    routing_reason: str
    is_top_1_correct: bool
    is_top_2_correct: bool
    is_top_3_correct: bool
    is_safe_decision: bool  # True if auto_handled & correct OR escalated


@dataclass
class RoutingEvaluationReport:
    """Comprehensive evaluation metrics report for the routing system."""
    total_records: int
    top_1_accuracy: float
    top_2_accuracy: float
    top_3_accuracy: float
    macro_f1: float
    auto_handle_count: int
    auto_handle_rate: float
    escalate_count: int
    escalate_rate: float
    auto_handle_accuracy: float
    auto_handle_error_rate: float
    escalation_safety_rate: float
    human_assistance_top2_rate: float
    per_intent_metrics: dict[str, Any] = field(default_factory=dict)
    detailed_results: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary_markdown(self) -> str:
        """Format the report as a clean markdown table summary."""
        lines = [
            "# Routing & Intent Classification Evaluation Summary",
            "",
            "## 1. Primary Performance Metrics",
            "| Metric | Value | Description |",
            "| :--- | :--- | :--- |",
            f"| **Total Evaluated Records** | `{self.total_records}` | Completed human-reviewed benchmark cases |",
            f"| **Top-1 Accuracy** | `{self.top_1_accuracy * 100:.1f}%` | Percentage where Top-1 matches ground truth |",
            f"| **Top-2 Accuracy** | `{self.top_2_accuracy * 100:.1f}%` | Percentage where ground truth is in Top-2 |",
            f"| **Top-3 Accuracy** | `{self.top_3_accuracy * 100:.1f}%` | Percentage where ground truth is in Top-3 |",
            f"| **Macro F1 Score** | `{self.macro_f1:.4f}` | Unweighted mean F1 across operational intents |",
            "",
            "## 2. Uncertainty Routing & Escalation Metrics",
            "| Metric | Value | Target / Description |",
            "| :--- | :--- | :--- |",
            f"| **Auto-Handle Count** | `{self.auto_handle_count}` | Total cases routed to automated handling |",
            f"| **Auto-Handle Rate** | `{self.auto_handle_rate * 100:.1f}%` | Operational workload automated safely |",
            f"| **Auto-Handle Accuracy** | `{self.auto_handle_accuracy * 100:.1f}%` | Precision of auto-handled decisions |",
            f"| **Auto-Handle Error Rate** | `{self.auto_handle_error_rate * 100:.1f}%` | Percentage of wrong auto-handled cases |",
            f"| **Escalation Count** | `{self.escalate_count}` | Total cases escalated to human review |",
            f"| **Escalation Rate** | `{self.escalate_rate * 100:.1f}%` | Workload redirected to human queue |",
            f"| **Escalation Safety Rate** | `{self.escalation_safety_rate * 100:.1f}%` | Percentage of incorrect predictions intercepted |",
            f"| **Human Assistance (Top-2)** | `{self.human_assistance_top2_rate * 100:.1f}%` | When escalated, ground truth is in Top-2 |",
            "",
        ]
        return "\n".join(lines)


class RoutingSystemEvaluator:
    """
    Evaluator that tests TopKIntentClassifier + IntentRouter on ground truth records.
    """

    def __init__(
        self,
        classifier: Optional[TopKIntentClassifier] = None,
        router: Optional[IntentRouter] = None,
    ) -> None:
        self.classifier = classifier or TopKIntentClassifier()
        self.router = router or IntentRouter()

    def evaluate_dataframe(
        self,
        df: pd.DataFrame,
        message_col: str = "customer_message",
        label_col: str = "annotation_label",
        id_col: str = "golden_id",
        status_col: str = "annotation_status",
    ) -> RoutingEvaluationReport:
        """
        Evaluate routing system on a DataFrame of ground-truth labeled records.
        """
        # Filter for valid ground-truth annotations
        clean_df = df.copy()
        if status_col in clean_df.columns:
            clean_df = clean_df[
                clean_df[status_col].isin(["reviewed", "overridden_ai_suggestion"])
            ]
        clean_df = clean_df[clean_df[label_col].fillna("").astype(str).str.strip() != ""]

        if clean_df.empty:
            raise ValueError("No valid completed human annotations found for evaluation.")

        total_recs = len(clean_df)
        logger.info("Evaluating routing engine over %d human-reviewed ground truth records...", total_recs)

        results: list[RecordEvaluationResult] = []
        y_true: list[str] = []
        y_pred_top1: list[str] = []

        for idx, (_, row) in enumerate(clean_df.iterrows(), 1):
            gid = str(row.get(id_col, "unknown"))
            msg = str(row[message_col])
            truth = str(row[label_col]).strip()

            if idx % 10 == 0 or idx == total_recs:
                logger.info("Evaluated %d/%d records...", idx, total_recs)

            analysis: IntentAnalysis = self.classifier.classify(customer_message=msg)
            decision: RoutingDecision = self.router.route(analysis)

            top_1_intent = analysis.top_1_intent
            top_1_conf = analysis.top_confidence
            top_2_intent = analysis.top_2_intent or "none"
            top_2_conf = analysis.top_2_confidence or 0.0

            cand_intents = [p.intent for p in analysis.top_predictions]
            is_top_1 = (truth == top_1_intent)
            is_top_2 = (truth in cand_intents[:2])
            is_top_3 = (truth in cand_intents[:3])

            is_safe = (decision.decision == RoutingDecisionType.ESCALATE_TO_HUMAN) or (
                decision.decision == RoutingDecisionType.AUTO_HANDLE and is_top_1
            )

            res = RecordEvaluationResult(
                golden_id=gid,
                customer_message=msg,
                ground_truth_label=truth,
                top_1_intent=top_1_intent,
                top_1_confidence=top_1_conf,
                top_2_intent=top_2_intent,
                top_2_confidence=top_2_conf,
                confidence_margin=analysis.confidence_margin,
                normalized_entropy=analysis.normalized_entropy,
                routing_decision=decision.decision.value,
                routing_reason=decision.reason,
                is_top_1_correct=is_top_1,
                is_top_2_correct=is_top_2,
                is_top_3_correct=is_top_3,
                is_safe_decision=is_safe,
            )
            results.append(res)
            y_true.append(truth)
            y_pred_top1.append(top_1_intent)

        total = len(results)
        top_1_acc = sum(1 for r in results if r.is_top_1_correct) / total
        top_2_acc = sum(1 for r in results if r.is_top_2_correct) / total
        top_3_acc = sum(1 for r in results if r.is_top_3_correct) / total

        # Compute Macro F1
        macro_f1 = float(f1_score(y_true, y_pred_top1, average="macro", zero_division=0))
        clf_rep = classification_report(y_true, y_pred_top1, output_dict=True, zero_division=0)

        # Routing specific metrics
        auto_handled = [r for r in results if r.routing_decision == RoutingDecisionType.AUTO_HANDLE.value]
        escalated = [r for r in results if r.routing_decision == RoutingDecisionType.ESCALATE_TO_HUMAN.value]

        auto_handle_count = len(auto_handled)
        auto_handle_rate = round(auto_handle_count / total, 4)
        escalate_count = len(escalated)
        escalate_rate = round(escalate_count / total, 4)

        if auto_handle_count > 0:
            auto_handle_correct = sum(1 for r in auto_handled if r.is_top_1_correct)
            auto_handle_acc = round(auto_handle_correct / auto_handle_count, 4)
            auto_handle_err = round(1.0 - auto_handle_acc, 4)
        else:
            auto_handle_acc = 1.0
            auto_handle_err = 0.0

        # Escalation safety: Out of all cases where Top-1 was wrong, how many were escalated?
        wrong_predictions = [r for r in results if not r.is_top_1_correct]
        if wrong_predictions:
            intercepted = sum(1 for r in wrong_predictions if r.routing_decision == RoutingDecisionType.ESCALATE_TO_HUMAN.value)
            escalation_safety = round(intercepted / len(wrong_predictions), 4)
        else:
            escalation_safety = 1.0

        # Human assistance: For escalated cases, how often was ground truth in Top-2?
        if escalated:
            assisted_count = sum(1 for r in escalated if r.is_top_2_correct)
            human_assistance_rate = round(assisted_count / len(escalated), 4)
        else:
            human_assistance_rate = 0.0

        detailed = [asdict(r) for r in results]

        return RoutingEvaluationReport(
            total_records=total,
            top_1_accuracy=round(top_1_acc, 4),
            top_2_accuracy=round(top_2_acc, 4),
            top_3_accuracy=round(top_3_acc, 4),
            macro_f1=round(macro_f1, 4),
            auto_handle_count=auto_handle_count,
            auto_handle_rate=auto_handle_rate,
            escalate_count=escalate_count,
            escalate_rate=escalate_rate,
            auto_handle_accuracy=auto_handle_acc,
            auto_handle_error_rate=auto_handle_err,
            escalation_safety_rate=escalation_safety,
            human_assistance_top2_rate=human_assistance_rate,
            per_intent_metrics=clf_rep,
            detailed_results=detailed,
        )

    def evaluate_golden_dataset(
        self,
        golden_path: Path | str = "data/golden/golden_set_human_review.csv",
    ) -> RoutingEvaluationReport:
        """Evaluate routing against human annotations in the golden benchmark CSV."""
        path = Path(golden_path)
        if not path.exists():
            raise FileNotFoundError(f"Golden dataset file not found: {path}")
        df = pd.read_csv(path)
        return self.evaluate_dataframe(df)
