"""
SupportGraph AI — Escalation Quality Evaluator (Phase 9)

Evaluates Phase 9 escalation categorization and selective recovery against the
protected 77-record human ground-truth benchmark.

CRITICAL SCIENTIFIC PRINCIPLES:
  - Dataset is strictly READ-ONLY.
  - SHA-256 checksums verified before AND after evaluation — zero contamination.
  - Evaluates: escalation category distribution, recovery rates, safety metrics,
    and comparative Phase 6 → 8 → 9 progression.
  - Selective recovery must NOT reduce error interception rate below Phase 8 baseline.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sklearn.metrics import accuracy_score, f1_score

try:
    from app.core.config import settings
    from app.core.logging import get_logger
    from app.resolution.escalation_quality_analyzer import (
        EscalationCategory,
        EscalationQualityAnalyzer,
        EscalationQualityResult,
    )
    from app.resolution.escalation_recovery_engine import (
        EscalationRecoveryEngine,
        EscalationRecoveryResult,
        RecoveryOutcome,
    )
    from app.resolution.evidence_validator import EvidenceVerdict
    from app.resolution.response_verifier import VerificationStatus
    from app.resolution.support_resolution_engine import (
        SupportResolutionEngine,
        SupportResolutionResult,
    )
    from app.schemas.intent_routing import RoutingDecisionType
except ModuleNotFoundError:
    from backend.app.core.config import settings  # type: ignore[no-redef]
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]
    from backend.app.resolution.escalation_quality_analyzer import (  # type: ignore[no-redef]
        EscalationCategory,
        EscalationQualityAnalyzer,
        EscalationQualityResult,
    )
    from backend.app.resolution.escalation_recovery_engine import (  # type: ignore[no-redef]
        EscalationRecoveryEngine,
        EscalationRecoveryResult,
        RecoveryOutcome,
    )
    from backend.app.resolution.evidence_validator import (  # type: ignore[no-redef]
        EvidenceVerdict,
    )
    from backend.app.resolution.response_verifier import (  # type: ignore[no-redef]
        VerificationStatus,
    )
    from backend.app.resolution.support_resolution_engine import (  # type: ignore[no-redef]
        SupportResolutionEngine,
        SupportResolutionResult,
    )
    from backend.app.schemas.intent_routing import (  # type: ignore[no-redef]
        RoutingDecisionType,
    )

logger = get_logger(__name__)

DEFAULT_GOLDEN_CSV = Path("data/golden/golden_set_human_review.csv")
GOLDEN_SHA256 = "1d3e9b3b8bdef3750437e59b17ab7c27167fd1548c2bb29de984151296c5b45a"


def compute_sha256(filepath: Path | str) -> str:
    """Compute SHA-256 checksum of a file."""
    path = Path(filepath)
    if not path.exists():
        return ""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class Phase9BenchmarkMetrics(BaseModel):
    """Structured metrics produced by the Phase 9 benchmark evaluation."""
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    total_evaluated_records: int
    pre_eval_sha256: str
    post_eval_sha256: str
    dataset_immutability_pass: bool

    # 1. Phase 8 Baseline (re-measured from this run)
    phase8_auto_handle_count: int
    phase8_auto_handle_rate: float
    phase8_auto_handle_accuracy: float
    phase8_unsafe_auto_handles: int
    phase8_escalation_count: int
    phase8_escalation_rate: float
    phase8_error_interception_rate: float
    phase8_primary_intent_accuracy: float
    phase8_macro_f1: float

    # 2. Escalation Quality Category Distribution
    escalation_category_distribution: dict[str, int]
    recoverable_escalation_count: int
    recoverable_escalation_rate_of_total: float
    recoverable_escalation_rate_of_escalated: float

    # 3. Selective Recovery Results
    recovery_attempted_count: int
    recovery_success_count: int
    recovery_failed_safety_count: int
    recovery_success_rate: float

    # 4. Phase 9 Post-Recovery Metrics
    phase9_auto_handle_count: int
    phase9_auto_handle_rate: float
    phase9_auto_handle_accuracy: float
    phase9_unsafe_auto_handles: int
    phase9_escalation_count: int
    phase9_escalation_rate: float
    phase9_error_interception_rate: float

    # 5. Safety Delta (Phase 8 → Phase 9)
    auto_handle_rate_delta: float
    auto_handle_accuracy_delta: float
    unsafe_auto_handles_delta: int
    error_interception_rate_delta: float

    # 6. Multi-Phase Progression Comparison
    progression_comparison: dict[str, Any] = Field(default_factory=dict)


class EscalationQualityEvaluator:
    """
    Evaluator for Phase 9 Escalation Quality Analysis & Selective Recovery.

    Runs the full Phase 8 pipeline on the protected golden benchmark, classifies
    all escalated cases through the Phase 9 quality analyzer, attempts recovery
    for eligible cases, and computes comprehensive safety-preserving metrics.
    """

    def __init__(
        self,
        golden_path: Optional[Path | str] = None,
        engine: Optional[SupportResolutionEngine] = None,
        quality_analyzer: Optional[EscalationQualityAnalyzer] = None,
        recovery_engine: Optional[EscalationRecoveryEngine] = None,
    ) -> None:
        self.golden_path = Path(golden_path or DEFAULT_GOLDEN_CSV)
        self.engine = engine or SupportResolutionEngine()
        self.quality_analyzer = quality_analyzer or EscalationQualityAnalyzer()
        self.recovery_engine = recovery_engine or EscalationRecoveryEngine(engine=self.engine)

    def run_benchmark(self) -> Phase9BenchmarkMetrics:
        """
        Execute full Phase 9 benchmark over the protected 77 human ground-truth records.
        """
        pre_hash = compute_sha256(self.golden_path)
        assert pre_hash == GOLDEN_SHA256, (
            f"Golden dataset checksum mismatch BEFORE evaluation!\n"
            f"Expected: {GOLDEN_SHA256}\nFound:    {pre_hash}\n"
            "Evaluation aborted to protect dataset integrity."
        )

        if not self.golden_path.exists():
            raise FileNotFoundError(f"Golden dataset not found at {self.golden_path}")

        df = pd.read_csv(self.golden_path)

        # Filter to completed human reviews only
        human_reviewed = df.copy()
        if "annotation_status" in human_reviewed.columns:
            human_reviewed = human_reviewed[
                human_reviewed["annotation_status"].isin(["reviewed", "overridden_ai_suggestion"])
            ]
        human_reviewed = human_reviewed[
            human_reviewed["annotation_label"].fillna("").astype(str).str.strip() != ""
        ].copy()

        if human_reviewed.empty:
            raise ValueError("No human-reviewed ground truth records found in dataset.")

        total_records = len(human_reviewed)
        logger.info("Running Phase 9 benchmark on %d human ground-truth records.", total_records)

        # ── Step 1: Run Phase 8 Pipeline ──────────────────────────────────────
        y_true: list[str] = []
        y_pred: list[str] = []
        phase8_results: list[tuple[str, str, SupportResolutionResult]] = []

        for _, row in human_reviewed.iterrows():
            gold_id = str(row.get("golden_id", ""))
            msg = str(row["customer_message"])
            true_label = str(row["annotation_label"]).strip()
            y_true.append(true_label)

            res: SupportResolutionResult = self.engine.process_message(
                customer_message=msg,
                case_id=gold_id,
                log_audit=False,
            )
            y_pred.append(res.primary_intent)
            phase8_results.append((msg, true_label, res))

        # ── Step 2: Compute Phase 8 Baseline Metrics ──────────────────────────
        p8_accuracy = round(float(accuracy_score(y_true, y_pred)), 4)
        p8_macro_f1 = round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4)

        p8_auto_total = 0
        p8_auto_correct = 0
        p8_unsafe = 0
        p8_esc_total = 0
        p8_total_errors = 0
        p8_intercepted = 0

        for msg, true_label, res in phase8_results:
            is_correct = (res.primary_intent == true_label)
            if not is_correct:
                p8_total_errors += 1
            if res.routing_decision == RoutingDecisionType.AUTO_HANDLE:
                p8_auto_total += 1
                if is_correct:
                    p8_auto_correct += 1
                else:
                    p8_unsafe += 1
            else:
                p8_esc_total += 1
                if not is_correct:
                    p8_intercepted += 1

        p8_auto_rate = round(p8_auto_total / total_records, 4)
        p8_auto_acc = round(p8_auto_correct / p8_auto_total, 4) if p8_auto_total > 0 else 0.0
        p8_esc_rate = round(p8_esc_total / total_records, 4)
        p8_interception = round(p8_intercepted / p8_total_errors, 4) if p8_total_errors > 0 else 1.0

        # ── Step 3: Phase 9 Escalation Quality Analysis ───────────────────────
        category_distribution: dict[str, int] = {cat.value: 0 for cat in EscalationCategory}
        escalated_cases_with_quality: list[tuple[str, str, SupportResolutionResult, EscalationQualityResult]] = []

        for msg, true_label, res in phase8_results:
            if res.routing_decision == RoutingDecisionType.ESCALATE_TO_HUMAN:
                quality = self.quality_analyzer.analyze_escalation(res)
                category_distribution[quality.escalation_category.value] += 1
                escalated_cases_with_quality.append((msg, true_label, res, quality))

        recoverable_count = category_distribution.get(EscalationCategory.RECOVERABLE_ESCALATION.value, 0)
        recoverable_rate_of_total = round(recoverable_count / total_records, 4)
        recoverable_rate_of_esc = round(recoverable_count / p8_esc_total, 4) if p8_esc_total > 0 else 0.0

        # ── Step 4: Selective Recovery ─────────────────────────────────────────
        recovery_attempted = 0
        recovery_success = 0
        recovery_failed_safety = 0

        # Lookup true labels for recovered cases for metric computation
        recovered_correct = 0
        recovered_unsafe = 0

        for msg, true_label, res, quality in escalated_cases_with_quality:
            if quality.escalation_category != EscalationCategory.RECOVERABLE_ESCALATION:
                continue

            recovery_attempted += 1
            recovery_result: EscalationRecoveryResult = self.recovery_engine.attempt_recovery(
                customer_message=msg,
                quality_result=quality,
            )

            if recovery_result.recovery_outcome == RecoveryOutcome.RECOVERED_AUTO_HANDLE:
                recovery_success += 1
                rec_res = recovery_result.recovered_resolution
                if rec_res and rec_res.primary_intent == true_label:
                    recovered_correct += 1
                else:
                    recovered_unsafe += 1
            else:
                recovery_failed_safety += 1

        recovery_success_rate = round(recovery_success / recovery_attempted, 4) if recovery_attempted > 0 else 0.0

        # ── Step 5: Phase 9 Post-Recovery Metrics ─────────────────────────────
        # Post-recovery: originally auto-handled cases + successfully recovered cases
        p9_auto_total = p8_auto_total + recovery_success
        p9_auto_correct = p8_auto_correct + recovered_correct
        p9_unsafe = p8_unsafe + recovered_unsafe
        p9_esc_total = p8_esc_total - recovery_success

        # Re-compute error interception: errors_intercepted / total_errors
        # Errors that went to human review = p8_intercepted - (recovered cases that were wrong)
        # Note: recovered_unsafe = newly uncovered errors from recovered cases
        p9_intercepted = p8_intercepted - recovered_unsafe
        p9_total_errors = p8_total_errors  # Total errors don't change — just routing of them

        p9_auto_rate = round(p9_auto_total / total_records, 4)
        p9_auto_acc = round(p9_auto_correct / p9_auto_total, 4) if p9_auto_total > 0 else 0.0
        p9_esc_rate = round(p9_esc_total / total_records, 4)
        p9_interception = round(p9_intercepted / p9_total_errors, 4) if p9_total_errors > 0 else 1.0

        # ── Step 6: Safety Deltas ──────────────────────────────────────────────
        auto_rate_delta = round(p9_auto_rate - p8_auto_rate, 4)
        auto_acc_delta = round(p9_auto_acc - p8_auto_acc, 4)
        unsafe_delta = p9_unsafe - p8_unsafe
        interception_delta = round(p9_interception - p8_interception, 4)

        # ── Step 7: Verify Dataset Immutability ───────────────────────────────
        post_hash = compute_sha256(self.golden_path)
        immutability_pass = (pre_hash == post_hash) and (post_hash == GOLDEN_SHA256)
        if not immutability_pass:
            logger.error(
                "CRITICAL: Golden dataset checksum mismatch AFTER evaluation! "
                "Pre: %s Post: %s", pre_hash, post_hash
            )

        # ── Step 8: Progression Comparison ────────────────────────────────────
        progression = {
            "Phase_6_Uncertainty_Baseline": {
                "auto_handle_rate": 0.831,
                "auto_handle_precision": 0.531,
                "unsafe_auto_handles": 30,
                "error_interception_rate": 0.211,
            },
            "Phase_6_1_Calibrated_Ranking": {
                "auto_handle_rate": 0.156,
                "auto_handle_precision": 0.750,
                "unsafe_auto_handles": 5,
                "error_interception_rate": 0.868,
            },
            "Phase_7_1_Safe_Gate": {
                "auto_handle_rate": 0.532,
                "auto_handle_precision": 0.634,
                "unsafe_auto_handles": 15,
                "error_interception_rate": 0.559,
            },
            "Phase_8_Evidence_Grounded_Resolution": {
                "auto_handle_rate": p8_auto_rate,
                "auto_handle_precision": p8_auto_acc,
                "unsafe_auto_handles": p8_unsafe,
                "error_interception_rate": p8_interception,
            },
            "Phase_9_Selective_Recovery": {
                "auto_handle_rate": p9_auto_rate,
                "auto_handle_precision": p9_auto_acc,
                "unsafe_auto_handles": p9_unsafe,
                "error_interception_rate": p9_interception,
                "recoverable_escalations_identified": recoverable_count,
                "recoveries_attempted": recovery_attempted,
                "recoveries_succeeded": recovery_success,
                "recoveries_blocked_by_safety": recovery_failed_safety,
            },
        }

        return Phase9BenchmarkMetrics(
            total_evaluated_records=total_records,
            pre_eval_sha256=pre_hash,
            post_eval_sha256=post_hash,
            dataset_immutability_pass=immutability_pass,
            phase8_auto_handle_count=p8_auto_total,
            phase8_auto_handle_rate=p8_auto_rate,
            phase8_auto_handle_accuracy=p8_auto_acc,
            phase8_unsafe_auto_handles=p8_unsafe,
            phase8_escalation_count=p8_esc_total,
            phase8_escalation_rate=p8_esc_rate,
            phase8_error_interception_rate=p8_interception,
            phase8_primary_intent_accuracy=p8_accuracy,
            phase8_macro_f1=p8_macro_f1,
            escalation_category_distribution=category_distribution,
            recoverable_escalation_count=recoverable_count,
            recoverable_escalation_rate_of_total=recoverable_rate_of_total,
            recoverable_escalation_rate_of_escalated=recoverable_rate_of_esc,
            recovery_attempted_count=recovery_attempted,
            recovery_success_count=recovery_success,
            recovery_failed_safety_count=recovery_failed_safety,
            recovery_success_rate=recovery_success_rate,
            phase9_auto_handle_count=p9_auto_total,
            phase9_auto_handle_rate=p9_auto_rate,
            phase9_auto_handle_accuracy=p9_auto_acc,
            phase9_unsafe_auto_handles=p9_unsafe,
            phase9_escalation_count=p9_esc_total,
            phase9_escalation_rate=p9_esc_rate,
            phase9_error_interception_rate=p9_interception,
            auto_handle_rate_delta=auto_rate_delta,
            auto_handle_accuracy_delta=auto_acc_delta,
            unsafe_auto_handles_delta=unsafe_delta,
            error_interception_rate_delta=interception_delta,
            progression_comparison=progression,
        )
