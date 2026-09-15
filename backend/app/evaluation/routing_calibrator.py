"""
SupportGraph AI — Routing Threshold Calibrator (Phase 6.1)

Performs data-driven threshold calibration and safety optimization for the
uncertainty-aware intent router:
- Grid search over confidence threshold, margin threshold, and maximum entropy
- Evaluates trade-offs: Auto-Handle Precision vs Automation Coverage vs Escalation Safety
- Safety-first ranking engine penalizing unsafe auto-handles
- Categorizes configurations into UNSAFE, CONSERVATIVE, BALANCED, and RECOMMENDED
- Generates structured evaluation reports and baseline vs calibrated comparisons.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

try:
    from app.core.config import settings
    from app.core.logging import get_logger
    from app.intent.classifier import TopKIntentClassifier
    from app.intent.router import IntentRouter
    from app.schemas.intent_routing import IntentAnalysis, RoutingDecisionType
except ModuleNotFoundError:
    from backend.app.core.config import settings  # type: ignore[no-redef]
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]
    from backend.app.intent.classifier import TopKIntentClassifier  # type: ignore[no-redef]
    from backend.app.intent.router import IntentRouter  # type: ignore[no-redef]
    from backend.app.schemas.intent_routing import (  # type: ignore[no-redef]
        IntentAnalysis,
        RoutingDecisionType,
    )

logger = get_logger(__name__)

# Default Grid Search Space
DEFAULT_CONFIDENCE_CANDIDATES = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
DEFAULT_MARGIN_CANDIDATES = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
DEFAULT_ENTROPY_CANDIDATES = [0.40, 0.50, 0.60, 0.65, 0.70, 0.80, 1.00]


@dataclass
class ThresholdCandidate:
    """Routing threshold parameter combination."""
    conf_threshold: float
    margin_threshold: float
    entropy_threshold: float


@dataclass
class CachedRecordPrediction:
    """Pre-computed classification outputs for a benchmark record."""
    golden_id: str
    customer_message: str
    ground_truth_label: str
    top_1_intent: str
    top_1_confidence: float
    top_2_intent: str
    top_2_confidence: float
    confidence_margin: float
    normalized_entropy: Optional[float]
    top_predictions: list[str]
    is_top_1_correct: bool
    is_top_2_correct: bool
    is_top_3_correct: bool


@dataclass
class ConfigurationEvaluationResult:
    """Comprehensive safety and automation metrics for a threshold configuration."""
    conf_threshold: float
    margin_threshold: float
    entropy_threshold: float
    total_records: int
    auto_handle_count: int
    auto_handle_rate: float
    escalate_count: int
    escalate_rate: float
    correct_auto_handles: int
    incorrect_auto_handles: int  # Unsafe auto-handles
    auto_handle_precision: float
    unsafe_auto_handle_rate: float
    error_share_auto_handled: float
    unnecessary_escalations: int
    intercepted_errors: int  # Successfully prevented errors
    error_interception_rate: float  # Escalation safety rate
    escalated_top1_correct: int
    escalated_top2_correct: int
    escalated_top3_correct: int
    human_assistance_top2_rate: float
    human_assistance_top3_rate: float
    safety_score: float
    category: str  # "UNSAFE", "CONSERVATIVE", "BALANCED", "RECOMMENDED"
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CalibrationReport:
    """Complete summary of routing threshold grid calibration."""
    total_evaluated_records: int
    total_configurations_evaluated: int
    baseline_configuration: ConfigurationEvaluationResult
    recommended_configuration: ConfigurationEvaluationResult
    all_results: list[ConfigurationEvaluationResult] = field(default_factory=list)
    unsafe_count: int = 0
    conservative_count: int = 0
    balanced_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_evaluated_records": self.total_evaluated_records,
            "total_configurations_evaluated": self.total_configurations_evaluated,
            "unsafe_count": self.unsafe_count,
            "conservative_count": self.conservative_count,
            "balanced_count": self.balanced_count,
            "baseline_configuration": self.baseline_configuration.to_dict(),
            "recommended_configuration": self.recommended_configuration.to_dict(),
            "all_results": [r.to_dict() for r in self.all_results],
        }

    def comparison_table_markdown(self) -> str:
        """Generate side-by-side markdown comparison table between Baseline and Calibrated."""
        base = self.baseline_configuration
        rec = self.recommended_configuration

        def diff_pct(val_rec: float, val_base: float) -> str:
            d = (val_rec - val_base) * 100
            sign = "+" if d > 0 else ""
            return f"{sign}{d:.1f}%"

        def diff_count(val_rec: int, val_base: int) -> str:
            d = val_rec - val_base
            sign = "+" if d > 0 else ""
            return f"{sign}{d}"

        lines = [
            "| Metric | Phase 6 Baseline | Calibrated Recommendation | Change |",
            "| :--- | :--- | :--- | :--- |",
            f"| **Confidence Threshold** | `{base.conf_threshold:.2f}` | `{rec.conf_threshold:.2f}` | `{rec.conf_threshold - base.conf_threshold:+.2f}` |",
            f"| **Margin Threshold** | `{base.margin_threshold:.2f}` | `{rec.margin_threshold:.2f}` | `{rec.margin_threshold - base.margin_threshold:+.2f}` |",
            f"| **Max Entropy** | `{base.entropy_threshold:.2f}` | `{rec.entropy_threshold:.2f}` | `{rec.entropy_threshold - base.entropy_threshold:+.2f}` |",
            f"| **Auto-Handle Rate** | `{base.auto_handle_rate * 100:.1f}%` ({base.auto_handle_count}/{base.total_records}) | `{rec.auto_handle_rate * 100:.1f}%` ({rec.auto_handle_count}/{rec.total_records}) | `{diff_pct(rec.auto_handle_rate, base.auto_handle_rate)}` |",
            f"| **Auto-Handle Precision** | `{base.auto_handle_precision * 100:.1f}%` | `{rec.auto_handle_precision * 100:.1f}%` | `{diff_pct(rec.auto_handle_precision, base.auto_handle_precision)}` |",
            f"| **Unsafe Auto-Handles (Errors)** | `{base.incorrect_auto_handles}` | `{rec.incorrect_auto_handles}` | `{diff_count(rec.incorrect_auto_handles, base.incorrect_auto_handles)}` |",
            f"| **Escalation Rate** | `{base.escalate_rate * 100:.1f}%` ({base.escalate_count}/{base.total_records}) | `{rec.escalate_rate * 100:.1f}%` ({rec.escalate_count}/{rec.total_records}) | `{diff_pct(rec.escalate_rate, base.escalate_rate)}` |",
            f"| **Error Interception Rate** | `{base.error_interception_rate * 100:.1f}%` | `{rec.error_interception_rate * 100:.1f}%` | `{diff_pct(rec.error_interception_rate, base.error_interception_rate)}` |",
            f"| **Top-2 Human Assistance** | `{base.human_assistance_top2_rate * 100:.1f}%` | `{rec.human_assistance_top2_rate * 100:.1f}%` | `{diff_pct(rec.human_assistance_top2_rate, base.human_assistance_top2_rate)}` |",
            f"| **Composite Safety Score** | `{base.safety_score:.4f}` | `{rec.safety_score:.4f}` | `{rec.safety_score - base.safety_score:+.4f}` |",
        ]
        return "\n".join(lines)


class RoutingCalibrator:
    """
    Evaluates, ranks, and recommends uncertainty-aware routing threshold configurations.
    """

    def __init__(
        self,
        classifier: Optional[TopKIntentClassifier] = None,
        confidence_candidates: Optional[list[float]] = None,
        margin_candidates: Optional[list[float]] = None,
        entropy_candidates: Optional[list[float]] = None,
    ) -> None:
        self.classifier = classifier or TopKIntentClassifier()
        self.confidence_candidates = confidence_candidates or DEFAULT_CONFIDENCE_CANDIDATES
        self.margin_candidates = margin_candidates or DEFAULT_MARGIN_CANDIDATES
        self.entropy_candidates = entropy_candidates or DEFAULT_ENTROPY_CANDIDATES

    def precache_predictions(
        self,
        df: pd.DataFrame,
        message_col: str = "customer_message",
        label_col: str = "annotation_label",
        id_col: str = "golden_id",
        status_col: str = "annotation_status",
    ) -> list[CachedRecordPrediction]:
        """
        Classify all benchmark records once and cache outputs for high-speed grid evaluation.
        """
        clean_df = df.copy()
        if status_col in clean_df.columns:
            clean_df = clean_df[
                clean_df[status_col].isin(["reviewed", "overridden_ai_suggestion"])
            ]
        clean_df = clean_df[clean_df[label_col].fillna("").astype(str).str.strip() != ""]

        if clean_df.empty:
            raise ValueError("No valid completed human annotations found for calibration.")

        cached: list[CachedRecordPrediction] = []
        for _, row in clean_df.iterrows():
            gid = str(row.get(id_col, "unknown"))
            msg = str(row[message_col])
            truth = str(row[label_col]).strip()

            analysis: IntentAnalysis = self.classifier.classify(customer_message=msg)
            cand_intents = [p.intent for p in analysis.top_predictions]
            top_1_intent = analysis.top_1_intent
            top_1_conf = analysis.top_confidence
            top_2_intent = analysis.top_2_intent or "none"
            top_2_conf = analysis.top_2_confidence or 0.0

            is_top_1 = (truth == top_1_intent)
            is_top_2 = (truth in cand_intents[:2])
            is_top_3 = (truth in cand_intents[:3])

            cached.append(
                CachedRecordPrediction(
                    golden_id=gid,
                    customer_message=msg,
                    ground_truth_label=truth,
                    top_1_intent=top_1_intent,
                    top_1_confidence=top_1_conf,
                    top_2_intent=top_2_intent,
                    top_2_confidence=top_2_conf,
                    confidence_margin=analysis.confidence_margin,
                    normalized_entropy=analysis.normalized_entropy,
                    top_predictions=cand_intents,
                    is_top_1_correct=is_top_1,
                    is_top_2_correct=is_top_2,
                    is_top_3_correct=is_top_3,
                )
            )

        return cached

    def evaluate_configuration(
        self,
        cache: list[CachedRecordPrediction],
        candidate: ThresholdCandidate,
    ) -> ConfigurationEvaluationResult:
        """
        Evaluate a single threshold candidate against pre-cached predictions.
        """
        total = len(cache)
        auto_handles = 0
        correct_auto = 0
        incorrect_auto = 0
        escalates = 0
        unnecessary_esc = 0
        intercepted_errors = 0
        esc_top1_correct = 0
        esc_top2_correct = 0
        esc_top3_correct = 0

        total_model_errors = sum(1 for r in cache if not r.is_top_1_correct)

        for rec in cache:
            # Deterministic Routing Rules
            is_unclear = (rec.top_1_intent == "unclear_needs_review")
            is_low_conf = (rec.top_1_confidence < candidate.conf_threshold)
            is_low_margin = (rec.confidence_margin < candidate.margin_threshold)
            is_high_entropy = (
                rec.normalized_entropy is not None
                and rec.normalized_entropy > candidate.entropy_threshold
            )

            if is_unclear or is_low_conf or is_low_margin or is_high_entropy:
                # Route: ESCALATE_TO_HUMAN
                escalates += 1
                if rec.is_top_1_correct:
                    unnecessary_esc += 1
                    esc_top1_correct += 1
                else:
                    intercepted_errors += 1

                if rec.is_top_2_correct:
                    esc_top2_correct += 1
                if rec.is_top_3_correct:
                    esc_top3_correct += 1
            else:
                # Route: AUTO_HANDLE
                auto_handles += 1
                if rec.is_top_1_correct:
                    correct_auto += 1
                else:
                    incorrect_auto += 1

        auto_handle_rate = round(auto_handles / total, 4) if total > 0 else 0.0
        escalate_rate = round(escalates / total, 4) if total > 0 else 0.0

        auto_handle_precision = round(correct_auto / auto_handles, 4) if auto_handles > 0 else 1.0
        unsafe_auto_rate = round(incorrect_auto / total, 4) if total > 0 else 0.0
        error_share_auto = (
            round(incorrect_auto / total_model_errors, 4) if total_model_errors > 0 else 0.0
        )

        error_interception_rate = (
            round(intercepted_errors / total_model_errors, 4) if total_model_errors > 0 else 1.0
        )

        human_top2_rate = round(esc_top2_correct / escalates, 4) if escalates > 0 else 0.0
        human_top3_rate = round(esc_top3_correct / escalates, 4) if escalates > 0 else 0.0

        # Safety-First Composite Scoring
        # Priority 1: Precision (45%), Priority 2: Error Interception (30%), Priority 3: Human Top-2 Utility (15%), Priority 4: Automation Coverage (10%)
        if auto_handles > 0 and auto_handle_precision < 0.70:
            # Harsh penalty for unsafe auto-handling
            safety_score = round(0.10 * auto_handle_precision, 4)
        else:
            safety_score = round(
                (0.45 * auto_handle_precision)
                + (0.30 * error_interception_rate)
                + (0.15 * human_top2_rate)
                + (0.10 * auto_handle_rate),
                4,
            )

        # Categorization
        if auto_handle_precision < 0.70 or error_share_auto > 0.50:
            category = "UNSAFE"
            rationale = (
                f"Unsafe: auto-handle precision ({auto_handle_precision*100:.1f}%) is below 70% "
                f"or allows too many errors ({incorrect_auto} errors) into automatic handling."
            )
        elif auto_handle_precision >= 0.85 and auto_handle_rate < 0.25:
            category = "CONSERVATIVE"
            rationale = (
                f"Conservative: high precision ({auto_handle_precision*100:.1f}%) but low "
                f"automation coverage ({auto_handle_rate*100:.1f}%)."
            )
        elif auto_handle_precision >= 0.75 and auto_handle_rate >= 0.25 and error_interception_rate >= 0.60:
            category = "BALANCED"
            rationale = (
                f"Balanced: strong precision ({auto_handle_precision*100:.1f}%), good automation "
                f"({auto_handle_rate*100:.1f}%), and high error interception ({error_interception_rate*100:.1f}%)."
            )
        else:
            category = "BALANCED"
            rationale = (
                f"Moderate balance: precision={auto_handle_precision*100:.1f}%, "
                f"auto_handle_rate={auto_handle_rate*100:.1f}%."
            )

        return ConfigurationEvaluationResult(
            conf_threshold=candidate.conf_threshold,
            margin_threshold=candidate.margin_threshold,
            entropy_threshold=candidate.entropy_threshold,
            total_records=total,
            auto_handle_count=auto_handles,
            auto_handle_rate=auto_handle_rate,
            escalate_count=escalates,
            escalate_rate=escalate_rate,
            correct_auto_handles=correct_auto,
            incorrect_auto_handles=incorrect_auto,
            auto_handle_precision=auto_handle_precision,
            unsafe_auto_handle_rate=unsafe_auto_rate,
            error_share_auto_handled=error_share_auto,
            unnecessary_escalations=unnecessary_esc,
            intercepted_errors=intercepted_errors,
            error_interception_rate=error_interception_rate,
            escalated_top1_correct=esc_top1_correct,
            escalated_top2_correct=esc_top2_correct,
            escalated_top3_correct=esc_top3_correct,
            human_assistance_top2_rate=human_top2_rate,
            human_assistance_top3_rate=human_top3_rate,
            safety_score=safety_score,
            category=category,
            rationale=rationale,
        )

    def run_grid_search(
        self,
        df: pd.DataFrame,
        baseline_conf: float = 0.85,
        baseline_margin: float = 0.15,
        baseline_entropy: float = 0.65,
    ) -> CalibrationReport:
        """
        Execute full grid search calibration over benchmark dataset.
        """
        cache = self.precache_predictions(df)
        total_records = len(cache)

        results: list[ConfigurationEvaluationResult] = []

        for conf in self.confidence_candidates:
            for margin in self.margin_candidates:
                for entropy in self.entropy_candidates:
                    candidate = ThresholdCandidate(
                        conf_threshold=conf,
                        margin_threshold=margin,
                        entropy_threshold=entropy,
                    )
                    res = self.evaluate_configuration(cache, candidate)
                    results.append(res)

        # Evaluate current baseline
        baseline_candidate = ThresholdCandidate(
            conf_threshold=baseline_conf,
            margin_threshold=baseline_margin,
            entropy_threshold=baseline_entropy,
        )
        baseline_res = self.evaluate_configuration(cache, baseline_candidate)

        # Sort results by safety_score descending, then precision descending, then automation rate descending
        results.sort(
            key=lambda r: (r.safety_score, r.auto_handle_precision, r.auto_handle_rate),
            reverse=True,
        )

        unsafe_count = sum(1 for r in results if r.category == "UNSAFE")
        conservative_count = sum(1 for r in results if r.category == "CONSERVATIVE")
        balanced_count = sum(1 for r in results if r.category == "BALANCED")

        # Select Recommended Configuration:
        # Filter for candidates with high precision (>= 0.75 if possible, else top precision)
        safe_candidates = [r for r in results if r.category in ["BALANCED", "CONSERVATIVE"] and r.auto_handle_precision >= 0.75]
        if safe_candidates:
            recommended = safe_candidates[0]
        else:
            # Fallback to absolute highest safety score
            recommended = results[0]

        recommended.category = "RECOMMENDED"
        recommended.rationale = (
            f"Recommended: Maximizes auto-handle safety and error interception. "
            f"Achieves {recommended.auto_handle_precision*100:.1f}% auto-handle precision, "
            f"intercepts {recommended.error_interception_rate*100:.1f}% of potential errors, "
            f"and provides {recommended.human_assistance_top2_rate*100:.1f}% Top-2 assistance for escalations."
        )

        return CalibrationReport(
            total_evaluated_records=total_records,
            total_configurations_evaluated=len(results),
            baseline_configuration=baseline_res,
            recommended_configuration=recommended,
            all_results=results,
            unsafe_count=unsafe_count,
            conservative_count=conservative_count,
            balanced_count=balanced_count,
        )
