"""
SupportGraph AI — Evidence Coverage Evaluator (Phase 10.1)

Runs the Evidence Coverage Audit over all evidence-limited cases in the
protected 77-record human ground-truth benchmark.

CRITICAL SCIENTIFIC RULES:
  - Dataset is strictly READ-ONLY. SHA-256 verified before AND after.
  - Retrieval corpus uses historical_parquet ONLY — NOT the golden benchmark.
  - All 200 golden conversation IDs are excluded from the audit corpus.
  - This evaluator produces diagnostic measurements ONLY — no pipeline changes.

KEY METRICS:
  - RECOVERABLE FAILURE RATE: proportion of evidence-limited cases that can
    potentially be fixed through engineering (retrieval/ranking/validation).
  - TRUE KNOWLEDGE GAP RATE: proportion that require new corpus coverage.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

try:
    from app.core.logging import get_logger
    from app.evaluation.evidence_corpus_builder import (
        EvidenceCorpusBuilder,
        AuditCorpus,
        GOLDEN_SHA256,
        compute_sha256,
    )
    from app.evaluation.evidence_coverage_auditor import (
        AuditRootCause,
        EvidenceCoverageAuditRecord,
        EvidenceCoverageAuditor,
        RECOVERABLE_CAUSES,
    )
    from app.resolution.escalation_quality_analyzer import (
        EscalationCategory,
        EscalationQualityAnalyzer,
    )
    from app.resolution.support_resolution_engine import SupportResolutionEngine
    from app.schemas.intent_routing import RoutingDecisionType
except ModuleNotFoundError:
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]
    from backend.app.evaluation.evidence_corpus_builder import (  # type: ignore[no-redef]
        EvidenceCorpusBuilder,
        AuditCorpus,
        GOLDEN_SHA256,
        compute_sha256,
    )
    from backend.app.evaluation.evidence_coverage_auditor import (  # type: ignore[no-redef]
        AuditRootCause,
        EvidenceCoverageAuditRecord,
        EvidenceCoverageAuditor,
        RECOVERABLE_CAUSES,
    )
    from backend.app.resolution.escalation_quality_analyzer import (  # type: ignore[no-redef]
        EscalationCategory,
        EscalationQualityAnalyzer,
    )
    from backend.app.resolution.support_resolution_engine import (  # type: ignore[no-redef]
        SupportResolutionEngine,
    )
    from backend.app.schemas.intent_routing import RoutingDecisionType  # type: ignore[no-redef]

logger = get_logger(__name__)

DEFAULT_GOLDEN_CSV = Path("data/golden/golden_set_human_review.csv")



class Phase101BenchmarkMetrics(BaseModel):
    """Structured metrics produced by the Phase 10.1 evidence coverage audit."""
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    total_evaluated_records: int
    pre_eval_sha256: str
    post_eval_sha256: str
    dataset_immutability_pass: bool

    # Audit corpus
    audit_corpus_size: int
    audit_corpus_excluded_golden_ids: int
    audit_extended_search_depth: int

    # Escalation breakdown
    total_escalated: int
    evidence_limited_count: int
    other_escalation_count: int
    auto_handled_count: int

    # Root cause distribution (evidence-limited cases only)
    root_cause_distribution: dict[str, int]
    root_cause_percentages: dict[str, float]

    # Key diagnostic rates
    recoverable_failure_count: int
    recoverable_failure_rate: float   # recoverable / evidence-limited
    true_knowledge_gap_count: int
    true_knowledge_gap_rate: float    # knowledge gaps / evidence-limited
    insufficient_information_count: int
    insufficient_information_rate: float

    # Breakdown by recoverable type
    retrieval_miss_count: int
    ranking_miss_count: int
    validation_over_rejection_count: int
    representation_limitation_count: int

    # Supporting metrics
    avg_best_relevant_rank: Optional[float]  # avg rank where relevant evidence found
    corpus_coverage_rate: float  # % of evidence-limited cases that have relevant evidence somewhere

    # Engineering conclusion
    primary_bottleneck: str
    recommended_next_phase: str

    # Per-case records (for detailed reporting)
    case_audit_records: list[dict[str, Any]] = Field(default_factory=list)


class EvidenceCoverageEvaluator:
    """
    Evaluates Phase 10.1 Evidence Coverage Audit over the protected benchmark.

    Runs the full production pipeline, identifies evidence-limited cases,
    builds a leakage-safe audit corpus, and classifies each failure's root cause.
    """

    def __init__(
        self,
        golden_path: Optional[Path | str] = None,
        engine: Optional[SupportResolutionEngine] = None,
        quality_analyzer: Optional[EscalationQualityAnalyzer] = None,
        corpus_builder: Optional[EvidenceCorpusBuilder] = None,
        max_audit_corpus_size: int = 5000,
    ) -> None:
        self.golden_path = Path(golden_path or DEFAULT_GOLDEN_CSV)
        self.engine = engine or SupportResolutionEngine()
        self.quality_analyzer = quality_analyzer or EscalationQualityAnalyzer()
        self.corpus_builder = corpus_builder or EvidenceCorpusBuilder(
            max_sample_size=max_audit_corpus_size
        )
        self._audit_corpus: Optional[AuditCorpus] = None

    def _ensure_corpus(self) -> AuditCorpus:
        """Build the audit corpus if not already built."""
        if self._audit_corpus is None:
            logger.info("Building leakage-safe audit corpus...")
            self._audit_corpus = self.corpus_builder.build()
            logger.info(
                "Audit corpus ready: %d entries.", self._audit_corpus.size
            )
        return self._audit_corpus

    def run_benchmark(self) -> Phase101BenchmarkMetrics:
        """
        Execute Phase 10.1 evidence coverage audit over benchmark records.
        """
        pre_hash = compute_sha256(self.golden_path)

        if not self.golden_path.exists():
            raise FileNotFoundError(f"Golden dataset not found at {self.golden_path}")

        df = pd.read_csv(self.golden_path)

        # Filter human-reviewed records (the protected 77)
        human_reviewed = df[
            df["annotation_status"].isin(["reviewed", "overridden_ai_suggestion"])
        ].copy()
        human_reviewed = human_reviewed[
            human_reviewed["annotation_label"].fillna("").astype(str).str.strip() != ""
        ].copy()

        if human_reviewed.empty:
            raise ValueError("No human-reviewed records found.")

        total_records = len(human_reviewed)

        # Build audit corpus (leakage-safe)
        audit_corpus = self._ensure_corpus()
        auditor = EvidenceCoverageAuditor(audit_corpus=audit_corpus)

        # ── Phase 1: Run production pipeline on all records ───────────────────
        logger.info("Running production pipeline on %d benchmark records...", total_records)

        auto_handled: list[dict] = []
        escalated: list[dict] = []
        evidence_limited_records: list[dict] = []

        for _, row in human_reviewed.iterrows():
            cid = str(row.get("conversation_id", ""))
            text = str(row.get("customer_message", ""))
            gold_label = str(row.get("annotation_label", "")).strip()

            res = self.engine.process_message(
                customer_message=text, case_id=cid, log_audit=False
            )

            record_info = {
                "case_id": cid,
                "customer_message": text,
                "gold_label": gold_label,
                "pred_intent": res.primary_intent,
                "result": res,
            }

            if res.routing_decision == RoutingDecisionType.ESCALATE_TO_HUMAN:
                eq = self.quality_analyzer.analyze_escalation(res)
                record_info["escalation_category"] = eq.escalation_category.value
                escalated.append(record_info)

                if eq.escalation_category == EscalationCategory.EVIDENCE_LIMITED:
                    evidence_limited_records.append(record_info)
            else:
                record_info["escalation_category"] = "AUTO_HANDLED"
                auto_handled.append(record_info)

        logger.info(
            "Production run complete: %d auto-handled, %d escalated, %d evidence-limited.",
            len(auto_handled), len(escalated), len(evidence_limited_records),
        )

        # ── Phase 2: Audit evidence-limited cases ─────────────────────────────
        logger.info("Running evidence coverage audit on %d evidence-limited cases...",
                    len(evidence_limited_records))

        audit_records: list[EvidenceCoverageAuditRecord] = []
        root_cause_counts: dict[str, int] = {rc.value: 0 for rc in AuditRootCause}
        best_relevant_ranks: list[int] = []
        corpus_coverage_hits = 0

        for i, rec in enumerate(evidence_limited_records):
            cid = rec["case_id"]
            text = rec["customer_message"]
            res = rec["result"]
            esc_cat = rec.get("escalation_category", "EVIDENCE_LIMITED")

            logger.debug("Auditing case %d/%d: %s", i + 1, len(evidence_limited_records), cid)

            ev_verdict = (
                res.evidence_validation.evidence_verdict.value
                if res.evidence_validation
                else "UNKNOWN"
            )

            audit_rec = auditor.audit_case(
                case_id=cid,
                customer_message=text,
                primary_intent=res.primary_intent,
                evidence_verdict=ev_verdict,
                escalation_category=esc_cat,
                production_evidence_cases=res.evidence_cases,
                evidence_validation=res.evidence_validation,
            )

            audit_records.append(audit_rec)
            root_cause_counts[audit_rec.root_cause.value] += 1

            if audit_rec.corpus_relevant_evidence_exists:
                corpus_coverage_hits += 1
            if audit_rec.best_relevant_rank is not None:
                best_relevant_ranks.append(audit_rec.best_relevant_rank)

        # ── Phase 3: Compute aggregate metrics ────────────────────────────────
        ev_lim_count = len(evidence_limited_records)

        def safe_rate(num: int, den: int) -> float:
            return round(num / den, 4) if den > 0 else 0.0

        def root_cause_pct(cause: str) -> float:
            return safe_rate(root_cause_counts.get(cause, 0), ev_lim_count)

        recoverable_count = sum(
            root_cause_counts.get(rc.value, 0) for rc in RECOVERABLE_CAUSES
        )
        recoverable_rate = safe_rate(recoverable_count, ev_lim_count)

        tkg_count = root_cause_counts.get(AuditRootCause.TRUE_KNOWLEDGE_GAP.value, 0)
        tkg_rate = safe_rate(tkg_count, ev_lim_count)

        insuf_count = root_cause_counts.get(AuditRootCause.INSUFFICIENT_INFORMATION.value, 0)
        insuf_rate = safe_rate(insuf_count, ev_lim_count)

        corpus_cov_rate = safe_rate(corpus_coverage_hits, ev_lim_count)
        avg_rank = (
            round(sum(best_relevant_ranks) / len(best_relevant_ranks), 1)
            if best_relevant_ranks else None
        )

        # Root cause percentages
        root_cause_pcts = {
            cause: round(safe_rate(count, ev_lim_count) * 100, 1)
            for cause, count in root_cause_counts.items()
        }

        # Determine primary bottleneck
        sorted_causes = sorted(
            [(v, k) for k, v in root_cause_counts.items() if v > 0],
            reverse=True,
        )
        primary_bottleneck = sorted_causes[0][1] if sorted_causes else "UNKNOWN"

        # Recommended next phase
        bottleneck_to_next = {
            AuditRootCause.RETRIEVAL_MISS.value: (
                "Build a proper historical retrieval index from "
                "data/processed/conversation_messages.parquet "
                "with golden benchmark exclusion."
            ),
            AuditRootCause.RANKING_MISS.value: (
                "Operational Evidence Ranking Improvement — "
                "increase retrieval depth or improve operational re-ranking."
            ),
            AuditRootCause.VALIDATION_OVER_REJECTION.value: (
                "Evidence Validator Calibration — "
                "review MODERATE_EVIDENCE thresholds for over-rejection."
            ),
            AuditRootCause.REPRESENTATION_LIMITATION.value: (
                "Representation/Indexing Improvement — "
                "consider dense embedding retrieval or improved text chunking."
            ),
            AuditRootCause.TRUE_KNOWLEDGE_GAP.value: (
                "Knowledge Gap Discovery & Human-Validated Knowledge Acquisition — "
                "corpus expansion for uncovered problem types."
            ),
            AuditRootCause.INSUFFICIENT_INFORMATION.value: (
                "Ambiguity Gate Improvement — "
                "strengthen message sufficiency detection."
            ),
        }
        recommended_next = bottleneck_to_next.get(
            primary_bottleneck,
            "Investigate pipeline further."
        )

        post_hash = compute_sha256(self.golden_path)
        immutability_pass = (pre_hash == post_hash)

        return Phase101BenchmarkMetrics(
            total_evaluated_records=total_records,
            pre_eval_sha256=pre_hash,
            post_eval_sha256=post_hash,
            dataset_immutability_pass=immutability_pass,
            audit_corpus_size=audit_corpus.size,
            audit_corpus_excluded_golden_ids=audit_corpus.excluded_golden_ids,
            audit_extended_search_depth=100,
            total_escalated=len(escalated),
            evidence_limited_count=ev_lim_count,
            other_escalation_count=len(escalated) - ev_lim_count,
            auto_handled_count=len(auto_handled),
            root_cause_distribution=dict(root_cause_counts),
            root_cause_percentages=root_cause_pcts,
            recoverable_failure_count=recoverable_count,
            recoverable_failure_rate=recoverable_rate,
            true_knowledge_gap_count=tkg_count,
            true_knowledge_gap_rate=tkg_rate,
            insufficient_information_count=insuf_count,
            insufficient_information_rate=insuf_rate,
            retrieval_miss_count=root_cause_counts.get(AuditRootCause.RETRIEVAL_MISS.value, 0),
            ranking_miss_count=root_cause_counts.get(AuditRootCause.RANKING_MISS.value, 0),
            validation_over_rejection_count=root_cause_counts.get(AuditRootCause.VALIDATION_OVER_REJECTION.value, 0),
            representation_limitation_count=root_cause_counts.get(AuditRootCause.REPRESENTATION_LIMITATION.value, 0),
            avg_best_relevant_rank=avg_rank,
            corpus_coverage_rate=corpus_cov_rate,
            primary_bottleneck=primary_bottleneck,
            recommended_next_phase=recommended_next,
            case_audit_records=[r.to_dict() for r in audit_records],
        )
