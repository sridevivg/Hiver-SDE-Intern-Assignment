"""
SupportGraph AI — Evidence-Grounded Resolution Evaluator (Phase 8)

Evaluates the full Phase 8 resolution pipeline against the protected 77-record
human ground-truth benchmark (`data/golden/golden_set_human_review.csv`).

CRITICAL SCIENTIFIC PRINCIPLES:
- Dataset is strictly READ-ONLY.
- SHA-256 checksums are verified before and after evaluation to guarantee zero contamination.
- Metrics evaluate classification coverage, evidence quality, routing precision, and response grounding.
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
    from app.resolution.evidence_validator import (
        EvidenceValidationResult,
        EvidenceVerdict,
    )
    from app.resolution.response_verifier import VerificationStatus
    from app.resolution.support_resolution_engine import (
        SupportResolutionEngine,
        SupportResolutionResult,
    )
    from app.retrieval.evidence_ranker import EvidenceMatchTier
    from app.schemas.intent_routing import RoutingDecisionType
except ModuleNotFoundError:
    from backend.app.core.config import settings  # type: ignore[no-redef]
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]
    from backend.app.resolution.evidence_validator import (  # type: ignore[no-redef]
        EvidenceValidationResult,
        EvidenceVerdict,
    )
    from backend.app.resolution.response_verifier import (  # type: ignore[no-redef]
        VerificationStatus,
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

DEFAULT_GOLDEN_CSV = Path("data/golden/golden_set_human_review.csv")


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


class Phase8BenchmarkMetrics(BaseModel):
    """Structured metrics produced by the Phase 8 benchmark evaluation."""
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    total_evaluated_records: int
    completed_human_ground_truth_records: int
    pre_eval_sha256: str
    post_eval_sha256: str
    dataset_immutability_pass: bool

    # 1. Intent Classification
    primary_intent_accuracy: float
    top_2_coverage: float
    top_3_coverage: float
    macro_f1: float

    # 2. Operational Evidence Metrics
    direct_evidence_availability_rate: float
    average_intent_agreement: float
    average_symptom_agreement: float
    average_evidence_consistency: float
    verdict_distribution: dict[str, int]
    match_tier_counts: dict[str, int]

    # 3. Routing & Safety Metrics
    auto_handle_count: int
    auto_handle_rate: float
    auto_handle_accuracy: float
    unsafe_auto_handles_count: int
    escalation_count: int
    escalation_rate: float
    error_interception_rate: float
    human_assistance_top_2_rate: float

    # 4. Response Grounding & Verification
    response_verification_pass_count: int
    response_verification_pass_rate: float
    evidence_grounded_response_rate: float
    unsupported_response_rate: float
    average_grounding_score: float

    # 5. Multi-Phase Progression Matrix
    progression_comparison: dict[str, Any] = Field(default_factory=dict)


class EvidenceResolutionEvaluator:
    """
    Evaluator for Phase 8 Evidence-Grounded Support Resolution.
    """

    def __init__(
        self,
        golden_path: Optional[Path | str] = None,
        engine: Optional[SupportResolutionEngine] = None,
    ) -> None:
        self.golden_path = Path(golden_path or DEFAULT_GOLDEN_CSV)
        self.engine = engine or SupportResolutionEngine()

    def run_benchmark(self) -> Phase8BenchmarkMetrics:
        """
        Execute full benchmark evaluation over completed human ground truth records.
        """
        pre_hash = compute_sha256(self.golden_path)

        if not self.golden_path.exists():
            raise FileNotFoundError(f"Golden dataset not found at {self.golden_path}")

        df = pd.read_csv(self.golden_path)

        # Filter completed human reviews
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
        y_true: list[str] = []
        y_pred: list[str] = []
        top2_matches: list[bool] = []
        top3_matches: list[bool] = []

        auto_handle_correct = 0
        auto_handle_total = 0
        unsafe_auto_handles = 0
        escalated_total = 0
        escalated_top2_correct = 0
        total_errors = 0
        intercepted_errors = 0

        verdict_counts: dict[str, int] = {}
        tier_counts = {
            EvidenceMatchTier.DIRECT_PROBLEM_MATCH.value: 0,
            EvidenceMatchTier.RELATED_SYMPTOM.value: 0,
            EvidenceMatchTier.RELATED_CONTEXT.value: 0,
            EvidenceMatchTier.WEAK_SEMANTIC_MATCH.value: 0,
        }

        intent_agreements: list[float] = []
        symptom_agreements: list[float] = []
        consistencies: list[float] = []
        direct_evidence_cases = 0

        grounding_pass_count = 0
        grounding_scores: list[float] = []

        for _, row in human_reviewed.iterrows():
            gold_id = str(row.get("golden_id", ""))
            msg = str(row["customer_message"])
            true_label = str(row["annotation_label"]).strip()

            y_true.append(true_label)

            # Execute Phase 8 Engine without logging to runtime audit
            res: SupportResolutionResult = self.engine.process_message(
                customer_message=msg,
                case_id=gold_id,
                log_audit=False,
            )

            pred_label = res.primary_intent
            y_pred.append(pred_label)

            # Extract candidates list
            if res.escalation_package and res.escalation_package.top_candidates:
                candidates = [p.intent for p in res.escalation_package.top_candidates]
            else:
                candidates = [pred_label]
                for c in res.ambiguity_analysis.competing_candidates:
                    if c not in candidates:
                        candidates.append(c)

            in_top2 = true_label in candidates[:2]
            in_top3 = true_label in candidates[:3]
            top2_matches.append(in_top2)
            top3_matches.append(in_top3)

            is_correct = (pred_label == true_label)
            if not is_correct:
                total_errors += 1

            # Evidence Analysis
            if res.evidence_validation:
                v = res.evidence_validation.evidence_verdict.value
                verdict_counts[v] = verdict_counts.get(v, 0) + 1
                intent_agreements.append(res.evidence_validation.intent_agreement)
                symptom_agreements.append(res.evidence_validation.symptom_agreement)
                consistencies.append(res.evidence_validation.evidence_consistency)

                if res.evidence_validation.direct_problem_matches > 0:
                    direct_evidence_cases += 1

                for dm in res.evidence_validation.dimension_matches:
                    tier_counts[dm.match_tier.value] = tier_counts.get(dm.match_tier.value, 0) + 1

            # Grounding Analysis
            if res.response_grounding:
                grounding_scores.append(res.response_grounding.support_score)
                if res.response_grounding.verification_status == VerificationStatus.PASS:
                    grounding_pass_count += 1

            # Routing & Safety Analysis
            if res.routing_decision == RoutingDecisionType.AUTO_HANDLE:
                auto_handle_total += 1
                if is_correct:
                    auto_handle_correct += 1
                else:
                    unsafe_auto_handles += 1
            else:
                escalated_total += 1
                if not is_correct:
                    intercepted_errors += 1
                if in_top2:
                    escalated_top2_correct += 1

        post_hash = compute_sha256(self.golden_path)
        immutability_pass = (pre_hash == post_hash)

        # Classification Metrics
        accuracy = round(float(accuracy_score(y_true, y_pred)), 4)
        top_2_cov = round(float(sum(top2_matches) / total_records), 4)
        top_3_cov = round(float(sum(top3_matches) / total_records), 4)
        macro_f1 = round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4)

        # Evidence Metrics
        direct_avail = round(direct_evidence_cases / total_records, 4)
        avg_intent_agree = round(float(np.mean(intent_agreements)), 4) if intent_agreements else 0.0
        avg_sym_agree = round(float(np.mean(symptom_agreements)), 4) if symptom_agreements else 0.0
        avg_consistency = round(float(np.mean(consistencies)), 4) if consistencies else 0.0

        # Routing Metrics
        auto_rate = round(auto_handle_total / total_records, 4)
        auto_acc = round(auto_handle_correct / auto_handle_total, 4) if auto_handle_total > 0 else 0.0
        esc_rate = round(escalated_total / total_records, 4)
        interception_rate = round(intercepted_errors / total_errors, 4) if total_errors > 0 else 1.0
        human_top2_rate = round(escalated_top2_correct / escalated_total, 4) if escalated_total > 0 else 0.0

        # Grounding Metrics
        ground_pass_rate = round(grounding_pass_count / total_records, 4)
        avg_ground_score = round(float(np.mean(grounding_scores)), 4) if grounding_scores else 0.0
        grounded_resp_rate = round(auto_handle_correct / total_records, 4)
        unsupported_resp_rate = round(unsafe_auto_handles / total_records, 4)

        # Progression Comparison Matrix across Phases
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
            "Phase_7_Evidence_Aware": {
                "auto_handle_rate": 0.844,
                "auto_handle_precision": 0.569,
                "unsafe_auto_handles": 28,
                "error_interception_rate": 0.176,
            },
            "Phase_7_1_Safe_Gate": {
                "auto_handle_rate": 0.532,
                "auto_handle_precision": 0.634,
                "unsafe_auto_handles": 15,
                "error_interception_rate": 0.559,
            },
            "Phase_8_Evidence_Grounded_Resolution": {
                "auto_handle_rate": auto_rate,
                "auto_handle_precision": auto_acc,
                "unsafe_auto_handles": unsafe_auto_handles,
                "error_interception_rate": interception_rate,
                "human_assistance_top_2": human_top2_rate,
                "grounding_pass_rate": ground_pass_rate,
            },
        }

        return Phase8BenchmarkMetrics(
            total_evaluated_records=total_records,
            completed_human_ground_truth_records=total_records,
            pre_eval_sha256=pre_hash,
            post_eval_sha256=post_hash,
            dataset_immutability_pass=immutability_pass,
            primary_intent_accuracy=accuracy,
            top_2_coverage=top_2_cov,
            top_3_coverage=top_3_cov,
            macro_f1=macro_f1,
            direct_evidence_availability_rate=direct_avail,
            average_intent_agreement=avg_intent_agree,
            average_symptom_agreement=avg_sym_agree,
            average_evidence_consistency=avg_consistency,
            verdict_distribution=verdict_counts,
            match_tier_counts=tier_counts,
            auto_handle_count=auto_handle_total,
            auto_handle_rate=auto_rate,
            auto_handle_accuracy=auto_acc,
            unsafe_auto_handles_count=unsafe_auto_handles,
            escalation_count=escalated_total,
            escalation_rate=esc_rate,
            error_interception_rate=interception_rate,
            human_assistance_top_2_rate=human_top2_rate,
            response_verification_pass_count=grounding_pass_count,
            response_verification_pass_rate=ground_pass_rate,
            evidence_grounded_response_rate=grounded_resp_rate,
            unsupported_response_rate=unsupported_resp_rate,
            average_grounding_score=avg_ground_score,
            progression_comparison=progression,
        )
