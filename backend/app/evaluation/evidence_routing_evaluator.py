"""
SupportGraph AI — Evidence-Aware Routing & Support Resolution Evaluator (Phase 7)

Evaluates the end-to-end evidence-aware resolution pipeline against the human-reviewed
golden dataset (77 ground truth records):
- Ambiguity classification distribution (CLEAR_PRIMARY, CAUSE_VS_SYMPTOM, GENUINE_AMBIGUITY, etc.)
- Primary vs Cause disambiguation accuracy
- Auto-Handle rate and Auto-Handle precision
- Escalation rate and Escalation safety (preventing false auto-resolutions)
- Human Assistance Value (ground truth presence in top escalation candidates)
- Historical evidence retrieval match tier distribution
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from sklearn.metrics import classification_report, f1_score

try:
    from app.core.config import settings
    from app.core.logging import get_logger
    from app.intent.ambiguity_analyzer import AmbiguityType
    from app.resolution.support_resolution_engine import (
        SupportResolutionEngine,
        SupportResolutionResult,
    )
    from app.retrieval.evidence_ranker import EvidenceMatchTier
    from app.schemas.intent_routing import RoutingDecisionType
except ModuleNotFoundError:
    from backend.app.core.config import settings  # type: ignore[no-redef]
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]
    from backend.app.intent.ambiguity_analyzer import (  # type: ignore[no-redef]
        AmbiguityType,
    )
    from backend.app.resolution.support_resolution_engine import (  # type: ignore[no-redef]
        SupportResolutionEngine,
        SupportResolutionResult,
    )
    from backend.app.retrieval.evidence_ranker import (  # type: ignore[no-redef]
        EvidenceMatchTier,
    )
    from backend.app.schemas.intent_routing import (  # type: ignore[no-redef]
        RoutingDecisionType,
    )

logger = get_logger(__name__)


@dataclass
class EvidenceRecordEvaluation:
    """Evaluation result for an individual benchmark record."""
    golden_id: str
    customer_message: str
    ground_truth_label: str
    predicted_primary_intent: str
    is_primary_correct: bool
    is_top_2_correct: bool
    is_top_3_correct: bool
    confidence: float
    routing_decision: str
    ambiguity_type: str
    ambiguity_reason: str
    contextual_cause: Optional[str]
    retrieved_tiers: list[str]
    has_direct_evidence_match: bool
    is_safe_resolution: bool  # Auto-handled & correct OR escalated with candidates


@dataclass
class EvidenceRoutingEvaluationReport:
    """Comprehensive evaluation metrics for Phase 7 Evidence-Aware Resolution."""
    total_records: int
    primary_accuracy: float
    top_2_accuracy: float
    top_3_accuracy: float
    macro_f1: float

    auto_handle_count: int
    auto_handle_rate: float
    auto_handle_accuracy: float
    auto_handle_error_rate: float

    escalate_count: int
    escalate_rate: float
    escalation_safety_rate: float
    human_assistance_top2_rate: float

    ambiguity_breakdown: dict[str, int] = field(default_factory=dict)
    evidence_tier_breakdown: dict[str, int] = field(default_factory=dict)
    cause_vs_symptom_count: int = 0
    detailed_records: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary_markdown(self) -> str:
        """Format the report as a clean markdown summary."""
        lines = [
            "# Phase 7 Evidence-Aware Support Resolution Benchmark Report",
            "",
            "## 1. Overall Classification & Disambiguation Performance",
            "| Metric | Value | Description |",
            "| :--- | :--- | :--- |",
            f"| **Evaluated Benchmark Records** | `{self.total_records}` | Completed human-reviewed ground truth cases |",
            f"| **Primary Intent Accuracy** | `{self.primary_accuracy * 100:.1f}%` | Percentage where primary intent matches ground truth |",
            f"| **Top-2 Candidate Coverage** | `{self.top_2_accuracy * 100:.1f}%` | Percentage where ground truth is in Top-2 candidates |",
            f"| **Top-3 Candidate Coverage** | `{self.top_3_accuracy * 100:.1f}%` | Percentage where ground truth is in Top-3 candidates |",
            f"| **Macro F1 Score** | `{self.macro_f1:.4f}` | Unweighted mean F1 across operational intents |",
            "",
            "## 2. Evidence-Aware Routing & Escalation Safety",
            "| Metric | Value | Target / Description |",
            "| :--- | :--- | :--- |",
            f"| **Auto-Handle Count** | `{self.auto_handle_count}` | Total cases routed to automated brand resolution |",
            f"| **Auto-Handle Rate** | `{self.auto_handle_rate * 100:.1f}%` | Workload safely automated without human review |",
            f"| **Auto-Handle Accuracy** | `{self.auto_handle_accuracy * 100:.1f}%` | Precision of auto-handled support responses |",
            f"| **Auto-Handle Error Rate** | `{self.auto_handle_error_rate * 100:.1f}%` | Erroneous auto-handled decisions |",
            f"| **Escalation Count** | `{self.escalate_count}` | Cases packaged for human agent review |",
            f"| **Escalation Rate** | `{self.escalate_rate * 100:.1f}%` | Proportion redirected to human queue |",
            f"| **Escalation Safety Rate** | `{self.escalation_safety_rate * 100:.1f}%` | Percentage of ambiguous/incorrect cases safely escalated |",
            f"| **Human Assistance (Top-2)** | `{self.human_assistance_top2_rate * 100:.1f}%` | True intent in escalation package's Top-2 candidates |",
            "",
            "## 3. Ambiguity Type Distribution",
            "| Ambiguity Type | Count | Percentage |",
            "| :--- | :--- | :--- |",
        ]
        for amb_type, cnt in sorted(self.ambiguity_breakdown.items(), key=lambda x: x[1], reverse=True):
            pct = (cnt / self.total_records) * 100 if self.total_records else 0
            lines.append(f"| `{amb_type}` | `{cnt}` | `{pct:.1f}%` |")

        lines.extend([
            "",
            "## 4. Historical Evidence Match Tiers",
            "| Evidence Match Tier | Count | Percentage |",
            "| :--- | :--- | :--- |",
        ])
        tot_tiers = sum(self.evidence_tier_breakdown.values()) or 1
        for tier_name, cnt in sorted(self.evidence_tier_breakdown.items(), key=lambda x: x[1], reverse=True):
            pct = (cnt / tot_tiers) * 100
            lines.append(f"| `{tier_name}` | `{cnt}` | `{pct:.1f}%` |")

        return "\n".join(lines)


class EvidenceRoutingEvaluator:
    """
    Evaluator that tests SupportResolutionEngine on ground truth records.
    """

    def __init__(self, engine: Optional[SupportResolutionEngine] = None) -> None:
        self.engine = engine or SupportResolutionEngine()

    def evaluate_dataframe(
        self,
        df: pd.DataFrame,
        message_col: str = "customer_message",
        label_col: str = "annotation_label",
        id_col: str = "golden_id",
        status_col: str = "annotation_status",
    ) -> EvidenceRoutingEvaluationReport:
        """
        Evaluate full resolution engine across ground-truth records in a DataFrame.
        """
        clean_df = df.copy()
        if status_col in clean_df.columns:
            clean_df = clean_df[
                clean_df[status_col].isin(["reviewed", "overridden_ai_suggestion"])
            ]
        clean_df = clean_df[clean_df[label_col].fillna("").astype(str).str.strip() != ""]

        if clean_df.empty:
            raise ValueError("No valid completed human annotations found for evaluation.")

        total_recs = len(clean_df)
        logger.info("Evaluating Phase 7 Evidence-Aware Resolution over %d benchmark records...", total_recs)

        results: list[EvidenceRecordEvaluation] = []
        y_true: list[str] = []
        y_pred: list[str] = []
        ambiguity_counts: dict[str, int] = {}
        tier_counts: dict[str, int] = {}
        cause_vs_symptom_cnt = 0

        for idx, (_, row) in enumerate(clean_df.iterrows(), 1):
            gid = str(row.get(id_col, f"rec_{idx}"))
            msg = str(row[message_col]).strip()
            truth = str(row[label_col]).strip()

            res: SupportResolutionResult = self.engine.process_message(customer_message=msg)

            pred_primary = res.primary_intent
            is_primary_correct = (truth == pred_primary)

            # Check candidate coverage
            top_preds = [pred_primary]
            if res.ambiguity_analysis and res.ambiguity_analysis.competing_candidates:
                top_preds.extend(res.ambiguity_analysis.competing_candidates)
            elif res.escalation_package:
                top_preds = [p.intent for p in res.escalation_package.top_candidates]

            is_top_2 = truth in top_preds[:2]
            is_top_3 = truth in top_preds[:3]

            # Ambiguity breakdown
            amb_type_val = res.ambiguity_analysis.ambiguity_type.value
            ambiguity_counts[amb_type_val] = ambiguity_counts.get(amb_type_val, 0) + 1
            if res.ambiguity_analysis.ambiguity_type == AmbiguityType.CAUSE_VS_SYMPTOM:
                cause_vs_symptom_cnt += 1

            # Evidence tiers
            tiers = [c.match_tier.value for c in res.evidence_cases]
            for t in tiers:
                tier_counts[t] = tier_counts.get(t, 0) + 1
            has_direct = any(t == EvidenceMatchTier.DIRECT_PROBLEM_MATCH.value for t in tiers)

            # Safety determination
            is_safe = (res.routing_decision == RoutingDecisionType.ESCALATE_TO_HUMAN) or (
                res.routing_decision == RoutingDecisionType.AUTO_HANDLE and is_primary_correct
            )

            rec_eval = EvidenceRecordEvaluation(
                golden_id=gid,
                customer_message=msg,
                ground_truth_label=truth,
                predicted_primary_intent=pred_primary,
                is_primary_correct=is_primary_correct,
                is_top_2_correct=is_top_2,
                is_top_3_correct=is_top_3,
                confidence=res.confidence,
                routing_decision=res.routing_decision.value,
                ambiguity_type=amb_type_val,
                ambiguity_reason=res.ambiguity_analysis.ambiguity_reason,
                contextual_cause=res.ambiguity_analysis.contextual_cause,
                retrieved_tiers=tiers,
                has_direct_evidence_match=has_direct,
                is_safe_resolution=is_safe,
            )
            results.append(rec_eval)
            y_true.append(truth)
            y_pred.append(pred_primary)

        total = len(results)
        primary_acc = sum(1 for r in results if r.is_primary_correct) / total
        top_2_acc = sum(1 for r in results if r.is_top_2_correct) / total
        top_3_acc = sum(1 for r in results if r.is_top_3_correct) / total

        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

        auto_handled = [r for r in results if r.routing_decision == RoutingDecisionType.AUTO_HANDLE.value]
        escalated = [r for r in results if r.routing_decision == RoutingDecisionType.ESCALATE_TO_HUMAN.value]

        auto_handle_cnt = len(auto_handled)
        auto_handle_rate = round(auto_handle_cnt / total, 4)
        escalate_cnt = len(escalated)
        escalate_rate = round(escalate_cnt / total, 4)

        if auto_handle_cnt > 0:
            auto_handle_correct = sum(1 for r in auto_handled if r.is_primary_correct)
            auto_handle_acc = round(auto_handle_correct / auto_handle_cnt, 4)
            auto_handle_err = round(1.0 - auto_handle_acc, 4)
        else:
            auto_handle_acc = 1.0
            auto_handle_err = 0.0

        wrong_predictions = [r for r in results if not r.is_primary_correct]
        if wrong_predictions:
            intercepted = sum(1 for r in wrong_predictions if r.routing_decision == RoutingDecisionType.ESCALATE_TO_HUMAN.value)
            escalation_safety = round(intercepted / len(wrong_predictions), 4)
        else:
            escalation_safety = 1.0

        if escalated:
            assisted_cnt = sum(1 for r in escalated if r.is_top_2_correct)
            human_assistance_rate = round(assisted_cnt / len(escalated), 4)
        else:
            human_assistance_rate = 0.0

        detailed = [asdict(r) for r in results]

        return EvidenceRoutingEvaluationReport(
            total_records=total,
            primary_accuracy=round(primary_acc, 4),
            top_2_accuracy=round(top_2_acc, 4),
            top_3_accuracy=round(top_3_acc, 4),
            macro_f1=round(macro_f1, 4),
            auto_handle_count=auto_handle_cnt,
            auto_handle_rate=auto_handle_rate,
            auto_handle_accuracy=auto_handle_acc,
            auto_handle_error_rate=auto_handle_err,
            escalate_count=escalate_cnt,
            escalate_rate=escalate_rate,
            escalation_safety_rate=escalation_safety,
            human_assistance_top2_rate=human_assistance_rate,
            ambiguity_breakdown=ambiguity_counts,
            evidence_tier_breakdown=tier_counts,
            cause_vs_symptom_count=cause_vs_symptom_cnt,
            detailed_records=detailed,
        )

    def evaluate_golden_dataset(
        self,
        golden_path: Path | str = "data/golden/golden_set_human_review.csv",
    ) -> EvidenceRoutingEvaluationReport:
        """Evaluate engine against human ground-truth in golden benchmark CSV."""
        path = Path(golden_path)
        if not path.exists():
            raise FileNotFoundError(f"Golden dataset file not found: {path}")
        df = pd.read_csv(path)
        return self.evaluate_dataframe(df)
