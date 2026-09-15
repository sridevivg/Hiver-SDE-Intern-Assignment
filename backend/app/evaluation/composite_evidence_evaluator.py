"""
SupportGraph AI — Composite Evidence Evaluator (Phase 10)

Evaluates Multi-Case Evidence Synthesis against the protected 77-record
human ground-truth benchmark (`data/golden/golden_set_human_review.csv`).

CRITICAL SCIENTIFIC PRINCIPLES:
  - Dataset is strictly READ-ONLY.
  - SHA-256 checksums verified before AND after evaluation — zero contamination.
  - Evaluates: evidence coverage expansion, composite verdict distribution,
    dimension coverage, conflict detection, auto-handle precision, and safety.
  - Safety requirement: Unsafe auto-handles must not materially increase vs Phase 9 baseline.
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
    from app.resolution.evidence_conflict_detector import (
        ConflictDetectionResult,
        ConflictType,
        EvidenceConflictDetector,
    )
    from app.resolution.evidence_synthesizer import (
        CompositeEvidencePackage,
        MultiCaseEvidenceSynthesizer,
    )
    from app.resolution.evidence_validator import (
        CompositeEvidenceVerdict,
        EvidenceVerdict,
        ResolutionEvidenceValidator,
    )
    from app.resolution.response_verifier import VerificationStatus
    from app.resolution.support_resolution_engine import (
        SupportResolutionEngine,
        SupportResolutionResult,
    )
    from app.schemas.intent_routing import RoutingDecisionType
except ModuleNotFoundError:
    from backend.app.core.config import settings  # type: ignore[no-redef]
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]
    from backend.app.resolution.evidence_conflict_detector import (  # type: ignore[no-redef]
        ConflictDetectionResult,
        ConflictType,
        EvidenceConflictDetector,
    )
    from backend.app.resolution.evidence_synthesizer import (  # type: ignore[no-redef]
        CompositeEvidencePackage,
        MultiCaseEvidenceSynthesizer,
    )
    from backend.app.resolution.evidence_validator import (  # type: ignore[no-redef]
        CompositeEvidenceVerdict,
        EvidenceVerdict,
        ResolutionEvidenceValidator,
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


class Phase10BenchmarkMetrics(BaseModel):
    """Structured metrics produced by the Phase 10 benchmark evaluation."""
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    total_evaluated_records: int
    pre_eval_sha256: str
    post_eval_sha256: str
    dataset_immutability_pass: bool

    # 1. Intent Classification
    primary_intent_accuracy: float
    top_2_coverage: float
    top_3_coverage: float
    macro_f1: float

    # 2. Composite Evidence Verdict Distribution
    composite_verdict_distribution: dict[str, int]
    direct_strong_evidence_count: int
    composite_strong_evidence_count: int
    moderate_composite_count: int
    weak_composite_count: int
    conflicting_composite_count: int
    insufficient_composite_count: int

    # 3. Evidence Coverage Expansion
    usable_evidence_count: int
    usable_evidence_coverage_rate: float
    coverage_expansion_delta: float  # difference in usable evidence vs single-case baseline

    # 4. Dimension Coverage
    symptom_coverage_rate: float
    context_coverage_rate: float
    device_coverage_rate: float
    resolution_pattern_coverage_rate: float
    average_intent_consistency: float
    average_composite_support_score: float

    # 5. Conflict Detection
    conflict_detected_count: int
    intent_conflict_count: int
    symptom_contradiction_count: int
    total_excluded_cases_count: int
    deduplicated_cases_count: int

    # 6. Routing & Safety
    auto_handle_count: int
    auto_handle_rate: float
    auto_handle_accuracy: float
    unsafe_auto_handles_count: int
    escalation_count: int
    escalation_rate: float
    error_interception_rate: float

    # 7. Multi-Phase Progression Comparison
    progression_comparison: dict[str, Any] = Field(default_factory=dict)


class CompositeEvidenceEvaluator:
    """
    Evaluator for Phase 10 Evidence Coverage Expansion & Multi-Case Evidence Synthesis.
    """

    def __init__(
        self,
        golden_path: Optional[Path | str] = None,
        engine: Optional[SupportResolutionEngine] = None,
    ) -> None:
        self.golden_path = Path(golden_path or DEFAULT_GOLDEN_CSV)
        self.engine = engine or SupportResolutionEngine()

    def run_benchmark(self) -> Phase10BenchmarkMetrics:
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
        top_2_hits: int = 0
        top_3_hits: int = 0

        # Composite verdict counts
        verdict_counts: dict[str, int] = {
            CompositeEvidenceVerdict.DIRECT_STRONG_EVIDENCE.value: 0,
            CompositeEvidenceVerdict.COMPOSITE_STRONG_EVIDENCE.value: 0,
            CompositeEvidenceVerdict.MODERATE_COMPOSITE_EVIDENCE.value: 0,
            CompositeEvidenceVerdict.WEAK_COMPOSITE_EVIDENCE.value: 0,
            CompositeEvidenceVerdict.CONFLICTING_COMPOSITE_EVIDENCE.value: 0,
            CompositeEvidenceVerdict.INSUFFICIENT_COMPOSITE_EVIDENCE.value: 0,
        }

        # Dimension metrics
        symptom_covered_count = 0
        context_covered_count = 0
        device_covered_count = 0
        res_pattern_covered_count = 0
        intent_consistencies: list[float] = []
        composite_scores: list[float] = []

        # Conflict metrics
        conflicts_count = 0
        intent_conflicts = 0
        symptom_contradictions = 0
        excluded_cases_total = 0
        dedup_cases_total = 0

        # Routing & Safety
        auto_handles: list[dict[str, Any]] = []
        escalations: list[dict[str, Any]] = []
        single_case_strong_count = 0

        for _, row in human_reviewed.iterrows():
            cid = str(row.get("conversation_id", ""))
            text = str(row.get("customer_message", ""))
            gold_label = str(row.get("annotation_label", "")).strip()

            res: SupportResolutionResult = self.engine.process_message(
                customer_message=text,
                case_id=cid,
                log_audit=False,
            )

            pred_intent = res.primary_intent
            y_true.append(gold_label)
            y_pred.append(pred_intent)

            # Top-K candidate coverage
            if res.escalation_package and res.escalation_package.top_candidates:
                cand_intents = [c.intent for c in res.escalation_package.top_candidates]
            else:
                cand_intents = [pred_intent]

            if gold_label in cand_intents[:2]:
                top_2_hits += 1
            if gold_label in cand_intents[:3]:
                top_3_hits += 1

            # Single-case baseline check
            if res.evidence_validation and res.evidence_validation.evidence_verdict == EvidenceVerdict.STRONG_EVIDENCE:
                single_case_strong_count += 1

            # Composite evidence metrics
            pkg = res.composite_evidence
            if pkg:
                v_str = pkg.composite_verdict.value
                verdict_counts[v_str] = verdict_counts.get(v_str, 0) + 1

                if pkg.symptom_covered:
                    symptom_covered_count += 1
                if "context" in pkg.covered_dimensions:
                    context_covered_count += 1
                if "device" in pkg.covered_dimensions:
                    device_covered_count += 1
                if "resolution_pattern" in pkg.covered_dimensions:
                    res_pattern_covered_count += 1

                intent_consistencies.append(pkg.intent_consistency)
                composite_scores.append(pkg.composite_support_score)
                dedup_cases_total += len(pkg.duplicate_case_ids)

                if pkg.conflict_result and pkg.conflict_result.has_conflict:
                    conflicts_count += 1
                    if pkg.conflict_result.conflict_type in (ConflictType.INTENT_CONFLICT, ConflictType.MIXED_CONFLICT):
                        intent_conflicts += 1
                    if pkg.conflict_result.conflict_type in (ConflictType.SYMPTOM_CONTRADICTION, ConflictType.MIXED_CONFLICT):
                        symptom_contradictions += 1
                    excluded_cases_total += len(pkg.conflict_result.contradicting_case_ids)

            # Routing outcome
            is_auto = (res.routing_decision == RoutingDecisionType.AUTO_HANDLE)
            is_correct = (pred_intent == gold_label)

            record_info = {
                "conversation_id": cid,
                "gold_label": gold_label,
                "pred_intent": pred_intent,
                "is_correct": is_correct,
                "composite_verdict": pkg.composite_verdict.value if pkg else "N/A",
                "gate_passed": res.gate_result.decision == RoutingDecisionType.AUTO_HANDLE if res.gate_result else False,
            }

            if is_auto:
                auto_handles.append(record_info)
            else:
                escalations.append(record_info)

        # Classification metrics
        intent_acc = round(accuracy_score(y_true, y_pred), 4)
        top_2_cov = round(top_2_hits / total_records, 4)
        top_3_cov = round(top_3_hits / total_records, 4)
        macro_f1 = round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4)

        # Usable evidence coverage
        direct_strong = verdict_counts[CompositeEvidenceVerdict.DIRECT_STRONG_EVIDENCE.value]
        comp_strong = verdict_counts[CompositeEvidenceVerdict.COMPOSITE_STRONG_EVIDENCE.value]
        usable_evidence_count = direct_strong + comp_strong
        usable_coverage_rate = round(usable_evidence_count / total_records, 4)
        single_case_rate = round(single_case_strong_count / total_records, 4)
        coverage_expansion_delta = round(usable_coverage_rate - single_case_rate, 4)

        # Routing & Safety metrics
        auto_count = len(auto_handles)
        auto_rate = round(auto_count / total_records, 4)
        auto_correct = sum(1 for a in auto_handles if a["is_correct"])
        auto_acc = round(auto_correct / auto_count, 4) if auto_count > 0 else 0.0
        unsafe_auto = auto_count - auto_correct

        esc_count = len(escalations)
        esc_rate = round(esc_count / total_records, 4)
        total_errors = sum(1 for yt, yp in zip(y_true, y_pred) if yt != yp)
        intercepted_errors = sum(1 for e in escalations if not e["is_correct"])
        interception_rate = round(intercepted_errors / total_errors, 4) if total_errors > 0 else 1.0

        post_hash = compute_sha256(self.golden_path)
        immutability_pass = (pre_hash == post_hash)

        # Multi-Phase Progression Matrix
        progression = {
            "Phase 6 (Uncertainty Routing)": {
                "auto_handle_rate": 0.3636,
                "auto_handle_acc": 0.8214,
                "unsafe_auto_handles": 5,
                "error_interception_rate": 0.8529,
            },
            "Phase 7.1 (Ambiguity Gating)": {
                "auto_handle_rate": 0.3117,
                "auto_handle_acc": 0.8750,
                "unsafe_auto_handles": 3,
                "error_interception_rate": 0.9118,
            },
            "Phase 8 (Evidence Resolution)": {
                "auto_handle_rate": 0.2338,
                "auto_handle_acc": 0.6111,
                "unsafe_auto_handles": 7,
                "error_interception_rate": 0.7941,
            },
            "Phase 9 (Escalation Recovery)": {
                "auto_handle_rate": 0.2338,
                "auto_handle_acc": 0.6111,
                "unsafe_auto_handles": 7,
                "error_interception_rate": 0.7941,
            },
            "Phase 10 (Composite Evidence)": {
                "auto_handle_rate": auto_rate,
                "auto_handle_acc": auto_acc,
                "unsafe_auto_handles": unsafe_auto,
                "error_interception_rate": interception_rate,
                "usable_evidence_coverage_rate": usable_coverage_rate,
            },
        }

        return Phase10BenchmarkMetrics(
            total_evaluated_records=total_records,
            pre_eval_sha256=pre_hash,
            post_eval_sha256=post_hash,
            dataset_immutability_pass=immutability_pass,
            primary_intent_accuracy=intent_acc,
            top_2_coverage=top_2_cov,
            top_3_coverage=top_3_cov,
            macro_f1=macro_f1,
            composite_verdict_distribution=verdict_counts,
            direct_strong_evidence_count=direct_strong,
            composite_strong_evidence_count=comp_strong,
            moderate_composite_count=verdict_counts[CompositeEvidenceVerdict.MODERATE_COMPOSITE_EVIDENCE.value],
            weak_composite_count=verdict_counts[CompositeEvidenceVerdict.WEAK_COMPOSITE_EVIDENCE.value],
            conflicting_composite_count=verdict_counts[CompositeEvidenceVerdict.CONFLICTING_COMPOSITE_EVIDENCE.value],
            insufficient_composite_count=verdict_counts[CompositeEvidenceVerdict.INSUFFICIENT_COMPOSITE_EVIDENCE.value],
            usable_evidence_count=usable_evidence_count,
            usable_evidence_coverage_rate=usable_coverage_rate,
            coverage_expansion_delta=coverage_expansion_delta,
            symptom_coverage_rate=round(symptom_covered_count / total_records, 4),
            context_coverage_rate=round(context_covered_count / total_records, 4),
            device_coverage_rate=round(device_covered_count / total_records, 4),
            resolution_pattern_coverage_rate=round(res_pattern_covered_count / total_records, 4),
            average_intent_consistency=round(float(np.mean(intent_consistencies)), 4) if intent_consistencies else 0.0,
            average_composite_support_score=round(float(np.mean(composite_scores)), 4) if composite_scores else 0.0,
            conflict_detected_count=conflicts_count,
            intent_conflict_count=intent_conflicts,
            symptom_contradiction_count=symptom_contradictions,
            total_excluded_cases_count=excluded_cases_total,
            deduplicated_cases_count=dedup_cases_total,
            auto_handle_count=auto_count,
            auto_handle_rate=auto_rate,
            auto_handle_accuracy=auto_acc,
            unsafe_auto_handles_count=unsafe_auto,
            escalation_count=esc_count,
            escalation_rate=esc_rate,
            error_interception_rate=interception_rate,
            progression_comparison=progression,
        )
