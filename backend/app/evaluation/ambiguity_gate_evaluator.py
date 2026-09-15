"""
SupportGraph AI — Ambiguity Decision Gate & 3-Strategy Comparative Evaluator (Phase 7.1)

Evaluates and compares 3 routing strategies across the 77 human ground-truth benchmark records:
- Strategy A: Confidence-Only Routing (Baseline)
- Strategy B: Phase 7 Ambiguity Classification
- Strategy C: Phase 7.1 Multi-Signal Safe Decision Gate

Also performs:
- Signal Interception Analysis (errors intercepted vs correct cases escalated per signal)
- Error Categorization of remaining unsafe auto-handled decisions
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from sklearn.metrics import f1_score

try:
    from app.core.config import settings
    from app.core.logging import get_logger
    from app.intent.ambiguity_analyzer import AmbiguityAnalyzer, AmbiguityType
    from app.intent.ambiguity_decision_gate import (
        AmbiguityDecisionGate,
        DecisionGateResult,
    )
    from app.intent.clarity_signals import (
        CandidateConflictType,
        ClaritySignalEvaluator,
        ClaritySignalsProfile,
        EvidenceAgreementLevel,
        MultiSymptomLevel,
        ProblemStrengthLevel,
        SufficiencyLevel,
    )
    from app.intent.classifier import TopKIntentClassifier
    from app.intent.primary_problem_selector import PrimaryProblemSelector
    from app.resolution.support_resolution_engine import (
        SupportResolutionEngine,
        SupportResolutionResult,
    )
    from app.retrieval.case_retriever import CaseRetriever
    from app.schemas.intent_routing import (
        IntentAnalysis,
        RoutingDecisionType,
    )
    from app.understanding.problem_extractor import (
        CustomerProblemProfile,
        ProblemExtractor,
    )
except ModuleNotFoundError:
    from backend.app.core.config import settings  # type: ignore[no-redef]
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]
    from backend.app.intent.ambiguity_analyzer import (  # type: ignore[no-redef]
        AmbiguityAnalyzer,
        AmbiguityType,
    )
    from backend.app.intent.ambiguity_decision_gate import (  # type: ignore[no-redef]
        AmbiguityDecisionGate,
        DecisionGateResult,
    )
    from backend.app.intent.clarity_signals import (  # type: ignore[no-redef]
        CandidateConflictType,
        ClaritySignalEvaluator,
        ClaritySignalsProfile,
        EvidenceAgreementLevel,
        MultiSymptomLevel,
        ProblemStrengthLevel,
        SufficiencyLevel,
    )
    from backend.app.intent.classifier import TopKIntentClassifier  # type: ignore[no-redef]
    from backend.app.intent.primary_problem_selector import (  # type: ignore[no-redef]
        PrimaryProblemSelector,
    )
    from backend.app.resolution.support_resolution_engine import (  # type: ignore[no-redef]
        SupportResolutionEngine,
        SupportResolutionResult,
    )
    from backend.app.retrieval.case_retriever import CaseRetriever  # type: ignore[no-redef]
    from backend.app.schemas.intent_routing import (  # type: ignore[no-redef]
        IntentAnalysis,
        RoutingDecisionType,
    )
    from backend.app.understanding.problem_extractor import (  # type: ignore[no-redef]
        CustomerProblemProfile,
        ProblemExtractor,
    )

logger = get_logger(__name__)


@dataclass
class StrategyMetrics:
    """Performance and safety metrics for a single routing strategy."""
    strategy_name: str
    total_records: int
    auto_handle_count: int
    auto_handle_rate: float
    auto_handle_accuracy: float
    unsafe_auto_handles: int
    escalation_count: int
    escalation_rate: float
    error_interception_rate: float
    correct_cases_escalated: int
    human_assistance_top2: float
    human_assistance_top3: float


@dataclass
class SignalInterceptionMetric:
    """Interception effectiveness of an individual clarity signal."""
    signal_name: str
    records_triggered: int
    errors_intercepted: int
    correct_escalated: int
    interception_precision: float


@dataclass
class UnsafeAutoHandleCase:
    """Details of a false-clarity unsafe auto-handled error."""
    golden_id: str
    customer_message: str
    ground_truth_intent: str
    predicted_intent: str
    confidence: float
    error_category: str
    explanation: str


@dataclass
class AmbiguityGateEvaluationReport:
    """Comprehensive comparative evaluation report for Phase 7.1."""
    total_records: int
    strategy_a_confidence_only: StrategyMetrics
    strategy_b_phase_7_ambiguity: StrategyMetrics
    strategy_c_phase_7_1_gate: StrategyMetrics
    signal_interception_analysis: list[SignalInterceptionMetric] = field(default_factory=list)
    unsafe_auto_handles_analysis: list[UnsafeAutoHandleCase] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_records": self.total_records,
            "strategy_a_confidence_only": asdict(self.strategy_a_confidence_only),
            "strategy_b_phase_7_ambiguity": asdict(self.strategy_b_phase_7_ambiguity),
            "strategy_c_phase_7_1_gate": asdict(self.strategy_c_phase_7_1_gate),
            "signal_interception_analysis": [asdict(s) for s in self.signal_interception_analysis],
            "unsafe_auto_handles_analysis": [asdict(u) for u in self.unsafe_auto_handles_analysis],
        }

    def summary_markdown(self) -> str:
        """Generate a clean markdown summary of the 3-strategy comparison."""
        sa = self.strategy_a_confidence_only
        sb = self.strategy_b_phase_7_ambiguity
        sc = self.strategy_c_phase_7_1_gate

        lines = [
            "# Phase 7.1 — Ambiguity Detection Calibration & Safe Decision Gating Report",
            "",
            "## 1. Three-Strategy Comparative Evaluation",
            "| Metric | Strategy A (Confidence-Only) | Strategy B (Phase 7 Ambiguity) | Strategy C (Phase 7.1 Multi-Signal Gate) | Delta (C vs B) |",
            "| :--- | :--- | :--- | :--- | :--- |",
            f"| **Evaluated Benchmark Records** | `{sa.total_records}` | `{sb.total_records}` | `{sc.total_records}` | — |",
            f"| **Auto-Handle Count** | `{sa.auto_handle_count}` | `{sb.auto_handle_count}` | `{sc.auto_handle_count}` | `{sc.auto_handle_count - sb.auto_handle_count:+d}` |",
            f"| **Auto-Handle Rate** | `{sa.auto_handle_rate * 100:.1f}%` | `{sb.auto_handle_rate * 100:.1f}%` | `{sc.auto_handle_rate * 100:.1f}%` | `{(sc.auto_handle_rate - sb.auto_handle_rate) * 100:+.1f}%` |",
            f"| **Auto-Handle Accuracy (Precision)** | `{sa.auto_handle_accuracy * 100:.1f}%` | `{sb.auto_handle_accuracy * 100:.1f}%` | `{sc.auto_handle_accuracy * 100:.1f}%` | `{(sc.auto_handle_accuracy - sb.auto_handle_accuracy) * 100:+.1f}%` |",
            f"| **Unsafe Auto-Handles (Errors)** | `{sa.unsafe_auto_handles}` | `{sb.unsafe_auto_handles}` | `{sc.unsafe_auto_handles}` | `{sc.unsafe_auto_handles - sb.unsafe_auto_handles:+d}` |",
            f"| **Escalation Count** | `{sa.escalation_count}` | `{sb.escalation_count}` | `{sc.escalation_count}` | `{sc.escalation_count - sb.escalation_count:+d}` |",
            f"| **Escalation Rate** | `{sa.escalation_rate * 100:.1f}%` | `{sb.escalation_rate * 100:.1f}%` | `{sc.escalation_rate * 100:.1f}%` | `{(sc.escalation_rate - sb.escalation_rate) * 100:+.1f}%` |",
            f"| **Error Interception Rate** | `{sa.error_interception_rate * 100:.1f}%` | `{sb.error_interception_rate * 100:.1f}%` | `{sc.error_interception_rate * 100:.1f}%` | `{(sc.error_interception_rate - sb.error_interception_rate) * 100:+.1f}%` |",
            f"| **Correct Cases Escalated** | `{sa.correct_cases_escalated}` | `{sb.correct_cases_escalated}` | `{sc.correct_cases_escalated}` | `{sc.correct_cases_escalated - sb.correct_cases_escalated:+d}` |",
            f"| **Human Assistance (Top-2)** | `{sa.human_assistance_top2 * 100:.1f}%` | `{sb.human_assistance_top2 * 100:.1f}%` | `{sc.human_assistance_top2 * 100:.1f}%` | `{(sc.human_assistance_top2 - sb.human_assistance_top2) * 100:+.1f}%` |",
            f"| **Human Assistance (Top-3)** | `{sa.human_assistance_top3 * 100:.1f}%` | `{sb.human_assistance_top3 * 100:.1f}%` | `{sc.human_assistance_top3 * 100:.1f}%` | `{(sc.human_assistance_top3 - sb.human_assistance_top3) * 100:+.1f}%` |",
            "",
            "## 2. Signal Interception Effectiveness Analysis",
            "| Clarity Signal | Records Triggered | Errors Intercepted | Correct Cases Escalated | Interception Precision |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]
        for sig in self.signal_interception_analysis:
            lines.append(
                f"| `{sig.signal_name}` | `{sig.records_triggered}` | `{sig.errors_intercepted}` | "
                f"`{sig.correct_escalated}` | `{sig.interception_precision * 100:.1f}%` |"
            )

        lines.extend([
            "",
            "## 3. Remaining Unsafe Auto-Handle Error Categorization",
            f"Total remaining unsafe auto-handles: `{len(self.unsafe_auto_handles_analysis)}`",
            "",
        ])
        if self.unsafe_auto_handles_analysis:
            lines.append("| Golden ID | Predicted vs True Intent | Error Category | Explanation |")
            lines.append("| :--- | :--- | :--- | :--- |")
            for u in self.unsafe_auto_handles_analysis:
                lines.append(
                    f"| `{u.golden_id}` | `{u.predicted_intent}` vs `{u.ground_truth_intent}` | "
                    f"`{u.error_category}` | {u.explanation} |"
                )
        else:
            lines.append("No unsafe auto-handles detected under Strategy C.")

        return "\n".join(lines)


def categorize_unsafe_error(
    message: str,
    truth: str,
    pred: str,
    profile: CustomerProblemProfile,
    signals: ClaritySignalsProfile,
) -> tuple[str, str]:
    """Categorize the root cause of an unsafe auto-handle error."""
    msg_lower = message.lower()
    if truth == "general_device_support" and pred in ("battery_power_issue", "software_update_problem"):
        return (
            "TAXONOMY_LIMITATION",
            "Customer describes a general setup or network issue categorized as general_device_support in ground truth.",
        )
    if "mac" in msg_lower or "safari" in msg_lower:
        return (
            "MODEL_CLASSIFICATION_ERROR",
            "Device/service was Mac/Safari, but classifier predicted general iOS/device support.",
        )
    if profile.update_related and truth != pred:
        return (
            "MULTIPLE_VALID_INTERPRETATIONS",
            f"Inquiry mentions software update trigger while primary ground truth intent is '{truth}'.",
        )
    if signals.retrieval_quality == "NO_EVIDENCE":
        return (
            "RETRIEVAL_MISMATCH",
            "Historical evidence cases did not contain an operational match to support the prediction.",
        )
    return (
        "FALSE_CLARITY_DECISION",
        f"Multi-signal gate passed with high confidence but model misclassified '{truth}' as '{pred}'.",
    )


class AmbiguityGateEvaluator:
    """
    Evaluator that executes 3-strategy comparative evaluation across the human ground-truth benchmark.
    """

    def __init__(
        self,
        extractor: Optional[ProblemExtractor] = None,
        classifier: Optional[TopKIntentClassifier] = None,
        problem_selector: Optional[PrimaryProblemSelector] = None,
        ambiguity_analyzer: Optional[AmbiguityAnalyzer] = None,
        retriever: Optional[CaseRetriever] = None,
        decision_gate: Optional[AmbiguityDecisionGate] = None,
    ) -> None:
        self.extractor = extractor or ProblemExtractor()
        self.classifier = classifier or TopKIntentClassifier()
        self.problem_selector = problem_selector or PrimaryProblemSelector()
        self.ambiguity_analyzer = ambiguity_analyzer or AmbiguityAnalyzer()
        self.retriever = retriever or CaseRetriever()
        self.decision_gate = decision_gate or AmbiguityDecisionGate()

    def evaluate_dataframe(
        self,
        df: pd.DataFrame,
        message_col: str = "customer_message",
        label_col: str = "annotation_label",
        id_col: str = "golden_id",
        status_col: str = "annotation_status",
    ) -> AmbiguityGateEvaluationReport:
        """
        Run 3-strategy comparative evaluation on benchmark records.
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
        logger.info("Evaluating Phase 7.1 3-strategy benchmark over %d records...", total_recs)

        # Storage for all evaluated records across strategies
        records_data: list[dict[str, Any]] = []

        for idx, (_, row) in enumerate(clean_df.iterrows(), 1):
            gid = str(row.get(id_col, f"rec_{idx}"))
            msg = str(row[message_col]).strip()
            truth = str(row[label_col]).strip()

            # Pipeline execution
            profile: CustomerProblemProfile = self.extractor.extract(msg)
            analysis: IntentAnalysis = self.classifier.classify(customer_message=msg)
            primary_intent, cause_intent, _ = self.problem_selector.select_primary_problem(msg, profile, analysis)
            evidence_cases = self.retriever.retrieve(query_text=msg, query_intent=primary_intent, query_profile=profile, top_k=3)

            # Strategy A: Confidence-Only (Auto-handle if top_conf >= 0.85 and margin >= 0.15)
            strat_a_auto = (analysis.top_confidence >= 0.85 and analysis.confidence_margin >= 0.15)

            # Strategy B: Phase 7 Ambiguity Routing
            amb_res = self.ambiguity_analyzer.analyze(
                message=msg,
                profile=profile,
                analysis=analysis,
                primary_intent=primary_intent,
                contextual_cause_intent=cause_intent,
            )
            strat_b_auto = (not amb_res.is_ambiguous and amb_res.routing_recommendation == RoutingDecisionType.AUTO_HANDLE)

            # Strategy C: Phase 7.1 Multi-Signal Safe Decision Gate
            gate_res: DecisionGateResult = self.decision_gate.evaluate_gate(
                message=msg,
                profile=profile,
                analysis=analysis,
                primary_intent=primary_intent,
                contextual_cause_intent=cause_intent,
                evidence_cases=evidence_cases,
            )
            strat_c_auto = (gate_res.decision == RoutingDecisionType.AUTO_HANDLE)

            is_correct = (truth == primary_intent)

            # Check candidate coverage
            cand_intents = [p.intent for p in analysis.top_predictions]
            is_in_top2 = truth in cand_intents[:2]
            is_in_top3 = truth in cand_intents[:3]

            records_data.append({
                "golden_id": gid,
                "message": msg,
                "truth": truth,
                "pred": primary_intent,
                "confidence": analysis.top_confidence,
                "is_correct": is_correct,
                "is_in_top2": is_in_top2,
                "is_in_top3": is_in_top3,
                "strat_a_auto": strat_a_auto,
                "strat_b_auto": strat_b_auto,
                "strat_c_auto": strat_c_auto,
                "signals": gate_res.signals,
                "profile": profile,
            })

        # Calculate metrics for a strategy
        def compute_strategy_metrics(name: str, auto_key: str) -> StrategyMetrics:
            total = len(records_data)
            auto_cases = [r for r in records_data if r[auto_key]]
            esc_cases = [r for r in records_data if not r[auto_key]]
            wrong_cases = [r for r in records_data if not r["is_correct"]]
            correct_cases = [r for r in records_data if r["is_correct"]]

            auto_cnt = len(auto_cases)
            auto_rate = round(auto_cnt / total, 4)
            auto_corr = sum(1 for r in auto_cases if r["is_correct"])
            auto_acc = round(auto_corr / auto_cnt, 4) if auto_cnt > 0 else 1.0
            unsafe_auto = auto_cnt - auto_corr

            esc_cnt = len(esc_cases)
            esc_rate = round(esc_cnt / total, 4)

            # Error interception: % of total errors that were escalated
            if wrong_cases:
                intercepted = sum(1 for r in wrong_cases if not r[auto_key])
                err_interception_rate = round(intercepted / len(wrong_cases), 4)
            else:
                err_interception_rate = 1.0

            # Correct cases escalated
            corr_esc = sum(1 for r in correct_cases if not r[auto_key])

            # Human assistance in escalation
            if esc_cases:
                top2_assist = round(sum(1 for r in esc_cases if r["is_in_top2"]) / esc_cnt, 4)
                top3_assist = round(sum(1 for r in esc_cases if r["is_in_top3"]) / esc_cnt, 4)
            else:
                top2_assist = 0.0
                top3_assist = 0.0

            return StrategyMetrics(
                strategy_name=name,
                total_records=total,
                auto_handle_count=auto_cnt,
                auto_handle_rate=auto_rate,
                auto_handle_accuracy=auto_acc,
                unsafe_auto_handles=unsafe_auto,
                escalation_count=esc_cnt,
                escalation_rate=esc_rate,
                error_interception_rate=err_interception_rate,
                correct_cases_escalated=corr_esc,
                human_assistance_top2=top2_assist,
                human_assistance_top3=top3_assist,
            )

        strat_a_metrics = compute_strategy_metrics("Strategy A (Confidence-Only)", "strat_a_auto")
        strat_b_metrics = compute_strategy_metrics("Strategy B (Phase 7 Ambiguity)", "strat_b_auto")
        strat_c_metrics = compute_strategy_metrics("Strategy C (Phase 7.1 Gate)", "strat_c_auto")

        # =========================================================================
        # SIGNAL ABLATION & INTERCEPTION ANALYSIS
        # =========================================================================
        signal_checks = [
            ("SIGNAL_A_INSUFFICIENT_INFO", lambda r: r["signals"].sufficiency != SufficiencyLevel.SUFFICIENT),
            ("SIGNAL_B_WEAK_PROBLEM", lambda r: r["signals"].problem_strength == ProblemStrengthLevel.WEAK),
            ("SIGNAL_C_GENUINE_CONFLICT", lambda r: r["signals"].candidate_conflict == CandidateConflictType.GENUINE_CONFLICT),
            ("SIGNAL_D_EVIDENCE_DISAGREEMENT", lambda r: r["signals"].evidence_agreement == EvidenceAgreementLevel.NO_AGREEMENT),
            ("SIGNAL_F_MULTI_SYMPTOM_COMPLEXITY", lambda r: r["signals"].multi_symptom == MultiSymptomLevel.UNRELATED_MULTI_SYMPTOM),
            ("SAFETY_VETO_COMBINED", lambda r: r["signals"].is_vetoed),
        ]

        signal_interceptions: list[SignalInterceptionMetric] = []
        for sig_name, fn in signal_checks:
            triggered = [r for r in records_data if fn(r)]
            trig_cnt = len(triggered)
            err_cnt = sum(1 for r in triggered if not r["is_correct"])
            corr_cnt = sum(1 for r in triggered if r["is_correct"])
            prec = round(err_cnt / trig_cnt, 4) if trig_cnt > 0 else 1.0
            signal_interceptions.append(
                SignalInterceptionMetric(
                    signal_name=sig_name,
                    records_triggered=trig_cnt,
                    errors_intercepted=err_cnt,
                    correct_escalated=corr_cnt,
                    interception_precision=prec,
                )
            )

        # =========================================================================
        # REMAINING UNSAFE AUTO-HANDLES ANALYSIS (STRATEGY C)
        # =========================================================================
        unsafe_cases: list[UnsafeAutoHandleCase] = []
        for r in records_data:
            if r["strat_c_auto"] and not r["is_correct"]:
                cat, exp = categorize_unsafe_error(
                    message=r["message"],
                    truth=r["truth"],
                    pred=r["pred"],
                    profile=r["profile"],
                    signals=r["signals"],
                )
                unsafe_cases.append(
                    UnsafeAutoHandleCase(
                        golden_id=r["golden_id"],
                        customer_message=r["message"],
                        ground_truth_intent=r["truth"],
                        predicted_intent=r["pred"],
                        confidence=r["confidence"],
                        error_category=cat,
                        explanation=exp,
                    )
                )

        return AmbiguityGateEvaluationReport(
            total_records=total_recs,
            strategy_a_confidence_only=strat_a_metrics,
            strategy_b_phase_7_ambiguity=strat_b_metrics,
            strategy_c_phase_7_1_gate=strat_c_metrics,
            signal_interception_analysis=signal_interceptions,
            unsafe_auto_handles_analysis=unsafe_cases,
        )

    def evaluate_golden_dataset(
        self,
        golden_path: Path | str = "data/golden/golden_set_human_review.csv",
    ) -> AmbiguityGateEvaluationReport:
        """Evaluate 3 strategies on golden human review dataset CSV."""
        path = Path(golden_path)
        if not path.exists():
            raise FileNotFoundError(f"Golden dataset not found: {path}")
        df = pd.read_csv(path)
        return self.evaluate_dataframe(df)
