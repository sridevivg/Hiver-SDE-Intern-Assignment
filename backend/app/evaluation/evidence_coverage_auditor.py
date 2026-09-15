"""
SupportGraph AI — Evidence Coverage Auditor (Phase 10.1)

Diagnoses WHY a specific evidence-limited case failed to find usable evidence.

DIAGNOSTIC ROOT CAUSE TAXONOMY (6 categories):
  RETRIEVAL_MISS              — Relevant case exists in corpus; retriever never returned it
  RANKING_MISS                — Relevant case retrieved but ranked below production cutoff
  VALIDATION_OVER_REJECTION   — Relevant case retrieved & selected; validator rejected it
  REPRESENTATION_LIMITATION   — Relevant information exists but representation prevents matching
  TRUE_KNOWLEDGE_GAP          — No operationally relevant case exists anywhere in corpus
  INSUFFICIENT_INFORMATION    — Customer message lacks minimum actionable content

SCIENTIFIC RULES:
  - Operational relevance = DIRECT_PROBLEM_MATCH or RELATED_SYMPTOM tier only.
    RELATED_CONTEXT and WEAK_SEMANTIC_MATCH are NOT considered relevant.
  - Relevance is evaluated using the existing EvidenceRanker, NOT raw lexical similarity.
  - This module NEVER changes production configuration, thresholds, or routing behavior.
  - Extended corpus search is audit-only; production top_k is not modified.
"""
from __future__ import annotations

import re
from enum import Enum
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from app.core.logging import get_logger
    from app.evaluation.evidence_corpus_builder import AuditCorpus, CorpusEntry
    from app.resolution.evidence_validator import (
        EvidenceValidationResult,
        EvidenceVerdict,
    )
    from app.retrieval.evidence_ranker import (
        EvidenceMatchTier,
        EvidenceRanker,
        RetrievedEvidenceCase,
    )
    from app.resolution.support_resolution_engine import SupportResolutionResult
    from app.understanding.problem_extractor import CustomerProblemProfile, ProblemExtractor
except ModuleNotFoundError:
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]
    from backend.app.evaluation.evidence_corpus_builder import (  # type: ignore[no-redef]
        AuditCorpus,
        CorpusEntry,
    )
    from backend.app.resolution.evidence_validator import (  # type: ignore[no-redef]
        EvidenceValidationResult,
        EvidenceVerdict,
    )
    from backend.app.retrieval.evidence_ranker import (  # type: ignore[no-redef]
        EvidenceMatchTier,
        EvidenceRanker,
        RetrievedEvidenceCase,
    )
    from backend.app.resolution.support_resolution_engine import (  # type: ignore[no-redef]
        SupportResolutionResult,
    )
    from backend.app.understanding.problem_extractor import (  # type: ignore[no-redef]
        CustomerProblemProfile,
        ProblemExtractor,
    )

logger = get_logger(__name__)

# Production retrieval top-K (used for RANKING_MISS detection)
PRODUCTION_TOP_K = 3

# Extended audit search depth
AUDIT_EXTENDED_TOP_K = 100

# Minimum TF-IDF similarity to consider a candidate for operational ranking
MIN_RAW_SIMILARITY_THRESHOLD = 0.05

# Information sufficiency: message is too short/vague if below this token count
MIN_INFORMATIVE_TOKEN_COUNT = 4


class AuditRootCause(str, Enum):
    """Root cause taxonomy for evidence coverage failures."""
    RETRIEVAL_MISS = "RETRIEVAL_MISS"
    RANKING_MISS = "RANKING_MISS"
    VALIDATION_OVER_REJECTION = "VALIDATION_OVER_REJECTION"
    REPRESENTATION_LIMITATION = "REPRESENTATION_LIMITATION"
    TRUE_KNOWLEDGE_GAP = "TRUE_KNOWLEDGE_GAP"
    INSUFFICIENT_INFORMATION = "INSUFFICIENT_INFORMATION"


# Root causes fixable through pipeline engineering (vs knowledge/information gaps)
RECOVERABLE_CAUSES: frozenset[AuditRootCause] = frozenset({
    AuditRootCause.RETRIEVAL_MISS,
    AuditRootCause.RANKING_MISS,
    AuditRootCause.VALIDATION_OVER_REJECTION,
    AuditRootCause.REPRESENTATION_LIMITATION,
})


class AuditCandidateResult(BaseModel):
    """A single candidate inspected during extended audit search."""
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    case_id: str
    rank_in_extended_pool: int
    raw_similarity: float
    operational_similarity: float
    match_tier: str
    is_operationally_relevant: bool
    customer_message_snippet: str = ""
    inferred_intent: str = ""


class EvidenceCoverageAuditRecord(BaseModel):
    """
    Complete per-case diagnostic output from the Evidence Coverage Auditor.

    Documents exactly why a single evidence-limited case failed to find
    usable operational evidence.
    """
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    case_id: str
    customer_message: str
    current_primary_intent: str
    current_evidence_verdict: str
    current_escalation_category: str = "EVIDENCE_LIMITED"

    # Information sufficiency check
    information_sufficient: bool = True
    information_sufficiency_notes: str = ""

    # Extended corpus search results
    audit_corpus_size: int = 0
    extended_search_depth: int = AUDIT_EXTENDED_TOP_K
    corpus_relevant_evidence_exists: bool = False
    relevant_evidence_count_in_extended: int = 0

    # Production retrieval check
    production_top_k: int = PRODUCTION_TOP_K
    retrieval_found_relevant_evidence: bool = False
    best_relevant_rank: Optional[int] = None  # 1-indexed rank in extended pool
    production_cutoff_rank: int = PRODUCTION_TOP_K

    # Top candidates from extended search
    top_audit_candidates: list[AuditCandidateResult] = Field(default_factory=list)

    # Validation check (for cases where retrieval succeeded)
    validation_attempted: bool = False
    validation_verdict: Optional[str] = None

    # Root cause
    root_cause: AuditRootCause = AuditRootCause.TRUE_KNOWLEDGE_GAP
    diagnostic_explanation: str = ""
    recommended_improvement_area: str = ""

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class EvidenceCoverageAuditor:
    """
    Diagnoses why a specific evidence-limited case failed to find usable evidence.

    The auditor uses an extended corpus search (AUDIT_EXTENDED_TOP_K candidates)
    to determine whether the failure is a pipeline problem or a knowledge gap.

    DOES NOT modify any production component.
    """

    def __init__(
        self,
        audit_corpus: Optional[AuditCorpus] = None,
        ranker: Optional[EvidenceRanker] = None,
        extractor: Optional[ProblemExtractor] = None,
    ) -> None:
        self.audit_corpus = audit_corpus
        self.ranker = ranker or EvidenceRanker()
        self.extractor = extractor or ProblemExtractor()
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._tfidf_matrix: Optional[Any] = None
        self._corpus_entries: list[CorpusEntry] = []

        if audit_corpus and audit_corpus.entries:
            self._build_index(audit_corpus.entries)

    def _build_index(self, entries: list[CorpusEntry]) -> None:
        """Build TF-IDF index over audit corpus entries."""
        self._corpus_entries = entries
        messages = [e.customer_message for e in entries]
        try:
            self._vectorizer = TfidfVectorizer(
                ngram_range=(1, 2), min_df=1, max_features=10000
            )
            self._tfidf_matrix = self._vectorizer.fit_transform(messages)
            logger.info("Audit corpus TF-IDF index built: %d entries.", len(entries))
        except Exception as exc:
            logger.warning("Failed to build audit corpus index: %s", exc)
            self._vectorizer = None
            self._tfidf_matrix = None

    def load_corpus(self, audit_corpus: AuditCorpus) -> None:
        """Load a new corpus and rebuild the index."""
        self.audit_corpus = audit_corpus
        if audit_corpus.entries:
            self._build_index(audit_corpus.entries)

    def audit_case(
        self,
        case_id: str,
        customer_message: str,
        primary_intent: str,
        evidence_verdict: str,
        escalation_category: str = "EVIDENCE_LIMITED",
        production_evidence_cases: Optional[list[RetrievedEvidenceCase]] = None,
        evidence_validation: Optional[EvidenceValidationResult] = None,
    ) -> EvidenceCoverageAuditRecord:
        """
        Diagnose why a single evidence-limited case failed.

        Args:
            case_id: Identifier of the benchmark case.
            customer_message: Raw customer message text.
            primary_intent: Intent predicted by production pipeline.
            evidence_verdict: EvidenceVerdict from production ResolutionEvidenceValidator.
            escalation_category: Phase 9 category (expected: EVIDENCE_LIMITED).
            production_evidence_cases: Cases retrieved by production pipeline (top-3).
            evidence_validation: Full EvidenceValidationResult from production run.

        Returns:
            EvidenceCoverageAuditRecord with root_cause classification and explanation.
        """
        record = EvidenceCoverageAuditRecord(
            case_id=case_id,
            customer_message=customer_message,
            current_primary_intent=primary_intent,
            current_evidence_verdict=evidence_verdict,
            current_escalation_category=escalation_category,
            audit_corpus_size=len(self._corpus_entries),
            extended_search_depth=AUDIT_EXTENDED_TOP_K,
            production_top_k=PRODUCTION_TOP_K,
            production_cutoff_rank=PRODUCTION_TOP_K,
        )

        # Extract problem profile for operational relevance evaluation
        profile = self.extractor.extract(customer_message)

        # ── Step 1: Information sufficiency check ─────────────────────────────
        tokens = [t for t in re.split(r'\W+', customer_message.lower()) if len(t) > 2]
        meaningful_tokens = [t for t in tokens if t not in {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'any', 'can',
            'her', 'was', 'one', 'our', 'out', 'had', 'him', 'his', 'has',
            'have', 'with', 'that', 'this', 'they', 'from', 'been', 'more',
            'will', 'when', 'what', 'make', 'like', 'into', 'than', 'its',
            'who', 'also', 'just', 'very', 'about', 'which', 'because',
            'applesupport', 'apple', 'support', 'help', 'please', 'thanks'
        }]

        is_sufficient = len(meaningful_tokens) >= MIN_INFORMATIVE_TOKEN_COUNT
        record.information_sufficient = is_sufficient
        if not is_sufficient:
            record.root_cause = AuditRootCause.INSUFFICIENT_INFORMATION
            record.information_sufficiency_notes = (
                f"Message contains only {len(meaningful_tokens)} meaningful tokens "
                f"(minimum required: {MIN_INFORMATIVE_TOKEN_COUNT}). "
                f"Too vague for operational evidence matching."
            )
            record.diagnostic_explanation = (
                f"Customer message is operationally insufficient: '{customer_message[:80]}'. "
                f"Only {len(meaningful_tokens)} meaningful tokens detected. "
                "No retrieval or ranking improvement can compensate for missing problem context."
            )
            record.recommended_improvement_area = (
                "Request additional context from customer. "
                "Improve message sufficiency detection at the ambiguity gate layer."
            )
            return record

        # ── Step 2: Extended corpus search ────────────────────────────────────
        if self._vectorizer is None or self._tfidf_matrix is None or not self._corpus_entries:
            record.corpus_relevant_evidence_exists = False
            record.root_cause = AuditRootCause.TRUE_KNOWLEDGE_GAP
            record.diagnostic_explanation = (
                "No audit corpus available for extended search. "
                "Cannot determine if evidence exists in the historical corpus."
            )
            record.recommended_improvement_area = (
                "Build and index the full historical corpus from "
                "data/processed/conversation_messages.parquet."
            )
            return record

        # Compute similarities over extended audit corpus
        query_vec = self._vectorizer.transform([customer_message])
        sims = cosine_similarity(query_vec, self._tfidf_matrix).flatten()

        # Get top candidates up to AUDIT_EXTENDED_TOP_K
        n_candidates = min(AUDIT_EXTENDED_TOP_K, len(self._corpus_entries))
        top_indices = np.argsort(sims)[::-1][:n_candidates]

        # Evaluate operational relevance for each candidate
        audit_candidates: list[AuditCandidateResult] = []
        relevant_cases: list[tuple[int, AuditCandidateResult]] = []  # (rank_1indexed, candidate)

        for rank_0, idx in enumerate(top_indices):
            entry = self._corpus_entries[idx]
            raw_sim = float(sims[idx])

            if raw_sim < MIN_RAW_SIMILARITY_THRESHOLD:
                # Skip clearly irrelevant candidates (saves time)
                continue

            tier, op_sim, _ = self.ranker.classify_evidence_tier(
                query_intent=primary_intent,
                query_profile=profile,
                hist_intent=entry.inferred_intent,
                hist_text=entry.customer_message,
                base_similarity=raw_sim,
            )

            is_relevant = tier in (
                EvidenceMatchTier.DIRECT_PROBLEM_MATCH,
                EvidenceMatchTier.RELATED_SYMPTOM,
            )

            candidate = AuditCandidateResult(
                case_id=entry.case_id,
                rank_in_extended_pool=rank_0 + 1,
                raw_similarity=round(raw_sim, 4),
                operational_similarity=round(op_sim, 4),
                match_tier=tier.value,
                is_operationally_relevant=is_relevant,
                customer_message_snippet=entry.customer_message[:100],
                inferred_intent=entry.inferred_intent,
            )
            audit_candidates.append(candidate)

            if is_relevant:
                relevant_cases.append((rank_0 + 1, candidate))

        record.top_audit_candidates = audit_candidates[:20]  # Store top 20 for reporting
        record.corpus_relevant_evidence_exists = len(relevant_cases) > 0
        record.relevant_evidence_count_in_extended = len(relevant_cases)

        if relevant_cases:
            best_rank = relevant_cases[0][0]  # Rank of first relevant case (1-indexed)
            record.best_relevant_rank = best_rank
            # Was it within production top-K?
            record.retrieval_found_relevant_evidence = (best_rank <= PRODUCTION_TOP_K)

        # ── Step 3: Root cause classification ─────────────────────────────────

        if not record.corpus_relevant_evidence_exists:
            # No operationally relevant evidence anywhere in extended corpus
            record.root_cause = AuditRootCause.TRUE_KNOWLEDGE_GAP
            record.diagnostic_explanation = (
                f"Extended search across {n_candidates} audit corpus entries found no "
                f"operationally relevant evidence (DIRECT_PROBLEM_MATCH or RELATED_SYMPTOM tier) "
                f"for intent '{primary_intent}'. "
                f"The symptom '{profile.primary_symptom}' is not represented in the "
                f"available historical corpus."
            )
            record.recommended_improvement_area = (
                "True knowledge gap. Corpus expansion required: "
                "acquire additional historical AppleSupport cases covering "
                f"'{primary_intent}' problems."
            )

        elif not record.retrieval_found_relevant_evidence:
            best_rank = record.best_relevant_rank
            # Relevant evidence exists deeper in the pool
            if best_rank is not None and best_rank <= AUDIT_EXTENDED_TOP_K:
                record.root_cause = AuditRootCause.RANKING_MISS
                record.diagnostic_explanation = (
                    f"Operationally relevant evidence found at extended rank #{best_rank} "
                    f"({record.relevant_evidence_count_in_extended} relevant cases total in top-{n_candidates}). "
                    f"Production pipeline only exposes top-{PRODUCTION_TOP_K} candidates. "
                    f"The relevant case was present in the retrieval pool but ranked below "
                    f"the production cutoff of {PRODUCTION_TOP_K}."
                )
                record.recommended_improvement_area = (
                    f"Ranking improvement: relevant evidence exists at rank #{best_rank}. "
                    "Investigate operational re-ranking quality or increase production top-K "
                    "after confirming safety metrics are preserved."
                )
            else:
                record.root_cause = AuditRootCause.RETRIEVAL_MISS
                record.diagnostic_explanation = (
                    f"Relevant evidence exists in corpus but was not found in top-{n_candidates} "
                    "extended search. This suggests a representation mismatch between "
                    "the query and the relevant corpus entries — the TF-IDF vocabulary "
                    "does not capture the operational similarity."
                )
                record.recommended_improvement_area = (
                    "Representation/retrieval improvement: "
                    "consider semantic embedding retrieval (dense vectors) to capture "
                    "operational similarity beyond vocabulary overlap."
                )

        else:
            # Relevant evidence was in production top-K but validation rejected it
            record.validation_attempted = True

            # Check production evidence validation verdict
            if evidence_validation and not evidence_validation.auto_resolution_allowed:
                # Look at dimension matches to understand why
                dimension_issues = []
                for d in evidence_validation.dimension_matches:
                    if d.symptom_match in ("unrelated", "contradictory"):
                        dimension_issues.append(f"symptom_match={d.symptom_match}")
                    if d.intent_match == "conflicting":
                        dimension_issues.append(f"intent_match=conflicting")

                if evidence_validation.evidence_verdict == EvidenceVerdict.WEAK_EVIDENCE:
                    record.root_cause = AuditRootCause.VALIDATION_OVER_REJECTION
                    record.validation_verdict = evidence_validation.evidence_verdict.value
                    record.diagnostic_explanation = (
                        f"Relevant evidence was retrieved (rank #{record.best_relevant_rank}) "
                        f"but the ResolutionEvidenceValidator classified it as "
                        f"'{evidence_validation.evidence_verdict.value}'. "
                        f"Resolution support strength: {evidence_validation.resolution_support_strength:.2f}. "
                        f"Dimension issues: {'; '.join(dimension_issues) if dimension_issues else 'minor metric shortfall'}. "
                        "Validation thresholds may be calibrated too strictly for this case type."
                    )
                    record.recommended_improvement_area = (
                        "Evidence validator calibration: "
                        f"relevant evidence exists but scored below auto-resolution threshold "
                        f"({evidence_validation.resolution_support_strength:.2f}). "
                        "Review MODERATE_EVIDENCE threshold parameters."
                    )
                else:
                    record.root_cause = AuditRootCause.REPRESENTATION_LIMITATION
                    record.validation_verdict = evidence_validation.evidence_verdict.value
                    record.diagnostic_explanation = (
                        f"Evidence retrieved and in top-{PRODUCTION_TOP_K} but failed "
                        f"operational validation (verdict: {evidence_validation.evidence_verdict.value}). "
                        f"Dimension issues: {'; '.join(dimension_issues) if dimension_issues else 'unknown'}. "
                        "The corpus entry may not have sufficient operational detail."
                    )
                    record.recommended_improvement_area = (
                        "Representation improvement: "
                        "corpus entries lack sufficient operational metadata "
                        "(symptom keywords, device context, troubleshooting steps) "
                        "for accurate multi-dimensional validation."
                    )
            else:
                # Validation passed but somehow still EVIDENCE_LIMITED — edge case
                record.root_cause = AuditRootCause.REPRESENTATION_LIMITATION
                record.diagnostic_explanation = (
                    f"Relevant evidence retrieved at rank #{record.best_relevant_rank} "
                    "but could not be confirmed as properly utilized by synthesis layer. "
                    "Possible representation or indexing limitation."
                )
                record.recommended_improvement_area = (
                    "Investigate evidence synthesis pipeline: "
                    "relevant cases may be retrieved but failing synthesis gates."
                )

        return record

    def audit_case_from_result(
        self,
        result: SupportResolutionResult,
        case_id: str,
        escalation_category: str = "EVIDENCE_LIMITED",
    ) -> EvidenceCoverageAuditRecord:
        """
        Convenience wrapper: audit a case from a SupportResolutionResult.
        """
        evidence_verdict = (
            result.evidence_validation.evidence_verdict.value
            if result.evidence_validation
            else "UNKNOWN"
        )
        return self.audit_case(
            case_id=case_id,
            customer_message=result.escalation_package.customer_message
            if result.escalation_package
            else "",
            primary_intent=result.primary_intent,
            evidence_verdict=evidence_verdict,
            escalation_category=escalation_category,
            production_evidence_cases=result.evidence_cases,
            evidence_validation=result.evidence_validation,
        )
