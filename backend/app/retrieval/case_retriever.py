"""
SupportGraph AI — Historical Case Retriever (Phase 10.2)

Retrieves semantically and operationally relevant historical AppleSupport cases:
- Backed by the 80,487-conversation leakage-safe HistoricalCorpusIndex
- Controlled Top-K candidate retrieval strategy (Top-25 candidate pool -> 5-dim operational ranking -> deduplication -> Top-3)
- Evaluates operational evidence across 20 problem families and 5 evidence tiers
- Attaches official AppleSupport brand resolution turn to retrieved evidence cases
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from app.core.config import settings
    from app.core.logging import get_logger
    from app.feedback.approved_store import ApprovedEvidenceStore
    from app.retrieval.evidence_ranker import (
        EvidenceMatchTier,
        EvidenceRanker,
        RetrievedEvidenceCase,
    )
    from app.retrieval.historical_corpus_index import HistoricalCorpusIndex
    from app.understanding.problem_extractor import CustomerProblemProfile
except ModuleNotFoundError:
    from backend.app.core.config import settings  # type: ignore[no-redef]
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]
    from backend.app.feedback.approved_store import (  # type: ignore[no-redef]
        ApprovedEvidenceStore,
    )
    from backend.app.retrieval.evidence_ranker import (  # type: ignore[no-redef]
        EvidenceMatchTier,
        EvidenceRanker,
        RetrievedEvidenceCase,
    )
    from backend.app.retrieval.historical_corpus_index import (  # type: ignore[no-redef]
        HistoricalCorpusIndex,
    )
    from backend.app.understanding.problem_extractor import (  # type: ignore[no-redef]
        CustomerProblemProfile,
    )

logger = get_logger(__name__)


class CaseRetriever:
    """
    Indexes and retrieves historical AppleSupport cases with operational evidence tiers.
    Defaults to HistoricalCorpusIndex (80,487 non-golden historical conversations)
    and queries ApprovedEvidenceStore for verified human-derived support evidence.
    """

    def __init__(
        self,
        dataset_path: Optional[Path | str] = None,
        ranker: Optional[EvidenceRanker] = None,
        corpus_index: Optional[HistoricalCorpusIndex] = None,
        approved_store: Optional[ApprovedEvidenceStore] = None,
    ) -> None:
        self.ranker = ranker or EvidenceRanker()
        self.dataset_path = Path(dataset_path) if dataset_path else None
        self.corpus_index = corpus_index
        self.approved_store = approved_store or ApprovedEvidenceStore()

        self._legacy_df: Optional[pd.DataFrame] = None
        self._legacy_vectorizer: Optional[TfidfVectorizer] = None
        self._legacy_tfidf: Optional[Any] = None

        if self.dataset_path is not None:
            self._load_legacy_corpus()
        else:
            if self.corpus_index is None:
                self.corpus_index = HistoricalCorpusIndex()

    def _load_legacy_corpus(self) -> None:
        """Load from a custom legacy CSV path (used primarily for custom test datasets)."""
        if self.dataset_path is None or not self.dataset_path.exists():
            logger.warning("Legacy dataset not found at %s.", self.dataset_path)
            self._legacy_df = pd.DataFrame()
            return

        try:
            df = pd.read_csv(self.dataset_path)
            df = df[df["customer_message"].fillna("").astype(str).str.strip() != ""].copy()
            df["intent"] = df["annotation_label"].fillna(df.get("candidate_intent", "general_device_support"))
            df["intent"] = df["intent"].replace("", "general_device_support")
            df["brand_reply"] = df.get(
                "conversation_context",
                "Thanks for reaching out to AppleSupport. Please send us a DM with your device details.",
            ).fillna("Thanks for reaching out to AppleSupport. Please DM us so we can assist further.")

            self._legacy_df = df.reset_index(drop=True)
            self._legacy_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=5000)
            self._legacy_tfidf = self._legacy_vectorizer.fit_transform(self._legacy_df["customer_message"].astype(str))
            logger.info("Indexed %d legacy support cases for retrieval.", len(self._legacy_df))
        except Exception as exc:
            logger.warning("Failed to load legacy dataset: %s", exc)
            self._legacy_df = pd.DataFrame()

    def retrieve(
        self,
        query_text: str,
        query_intent: str = "general_device_support",
        query_profile: Optional[CustomerProblemProfile] = None,
        top_k: int = 3,
    ) -> list[RetrievedEvidenceCase]:
        """
        Retrieve top historical cases and evaluate their operational evidence tier across 5 dimensions.
        """
        if query_profile is None:
            try:
                from app.understanding.problem_extractor import ProblemExtractor
            except ModuleNotFoundError:
                from backend.app.understanding.problem_extractor import (  # type: ignore[no-redef]
                    ProblemExtractor,
                )
            query_profile = ProblemExtractor().extract(query_text)

        # ── Pathway A: Full Historical Corpus Index (Default Production) ──────
        if self.corpus_index is not None and self.corpus_index.cases:
            query_family = getattr(query_profile, "primary_problem_family", None)
            candidates = self.corpus_index.search_candidates(
                query_text=query_text,
                query_family=query_family,
                top_n=max(top_k * 8, 25),
            )

            candidate_cases: list[RetrievedEvidenceCase] = []
            seen_prefixes: set[str] = set()

            for case_rec, raw_sim, boosted_sim in candidates:
                # Deduplication: suppress near-identical message prefixes
                norm_prefix = " ".join(case_rec.customer_message.lower().split()[:10])
                if norm_prefix in seen_prefixes:
                    continue
                seen_prefixes.add(norm_prefix)

                tier, op_sim, expl = self.ranker.classify_evidence_tier(
                    query_intent=query_intent,
                    query_profile=query_profile,
                    hist_intent=case_rec.inferred_intent,
                    hist_text=case_rec.customer_message,
                    base_similarity=raw_sim,
                    hist_family=case_rec.primary_problem_family,
                    hist_brand_response=case_rec.brand_response,
                )

                candidate_cases.append(
                    RetrievedEvidenceCase(
                        case_id=case_rec.case_id,
                        similarity_score=raw_sim,
                        operational_similarity=op_sim,
                        match_tier=tier,
                        historical_customer_message=case_rec.customer_message,
                        historical_brand_response=case_rec.brand_response,
                        historical_intent=case_rec.inferred_intent,
                        historical_problem_family=case_rec.primary_problem_family.value,
                        retrieval_explanation=expl,
                        source_type="HISTORICAL_CORPUS_EVIDENCE",
                        provenance="historical_apple_support_corpus",
                    )
                )

            # Query Approved Human-Validated Evidence Store (Phase 13)
            if self.approved_store is not None:
                fam_val = query_family.value if hasattr(query_family, "value") else (str(query_family) if query_family else None)
                human_cand_results = self.approved_store.search_evidence(
                    query_text=query_text,
                    query_family=fam_val,
                    top_k=max(top_k * 2, 5),
                )
                for app_item, app_sim in human_cand_results:
                    tier, op_sim, expl = self.ranker.classify_evidence_tier(
                        query_intent=query_intent,
                        query_profile=query_profile,
                        hist_intent=app_item.primary_intent,
                        hist_text=app_item.customer_problem_summary,
                        base_similarity=app_sim,
                        hist_family=app_item.problem_family,
                        hist_brand_response=app_item.brand_guidance,
                    )
                    candidate_cases.append(
                        RetrievedEvidenceCase(
                            case_id=app_item.evidence_id,
                            similarity_score=app_sim,
                            operational_similarity=op_sim,
                            match_tier=tier,
                            historical_customer_message=app_item.customer_problem_summary,
                            historical_brand_response=app_item.brand_guidance,
                            historical_intent=app_item.primary_intent,
                            historical_problem_family=app_item.problem_family,
                            retrieval_explanation=f"[HUMAN_VALIDATED_EVIDENCE v{app_item.version}] {expl}",
                            source_type="HUMAN_VALIDATED_EVIDENCE",
                            provenance=app_item.provenance,
                            evidence_quality_score=app_item.evidence_quality_score,
                            version=app_item.version,
                        )
                    )

            # Sort by tier hierarchy, then operational similarity, then raw similarity
            tier_priority = {
                EvidenceMatchTier.DIRECT_PROBLEM_MATCH: 1,
                EvidenceMatchTier.RELATED_PROBLEM_MATCH: 2,
                EvidenceMatchTier.RELATED_SYMPTOM: 3,
                EvidenceMatchTier.RELATED_CONTEXT: 4,
                EvidenceMatchTier.WEAK_SEMANTIC_MATCH: 5,
            }

            candidate_cases.sort(
                key=lambda c: (
                    tier_priority.get(c.match_tier, 5),
                    0 if c.source_type == "HUMAN_VALIDATED_EVIDENCE" else 1,
                    -c.operational_similarity,
                    -c.similarity_score,
                    -(c.evidence_quality_score or 0.0),
                )
            )

            return candidate_cases[:top_k]

        # ── Pathway B: Custom Legacy Dataset (Test Fallback) ───────────────────
        if self._legacy_df is None or self._legacy_df.empty or self._legacy_vectorizer is None:
            return []

        query_vec = self._legacy_vectorizer.transform([query_text])
        sims = cosine_similarity(query_vec, self._legacy_tfidf).flatten()
        top_indices = np.argsort(sims)[::-1][: max(top_k * 3, 10)]

        candidate_cases = []
        for idx in top_indices:
            row = self._legacy_df.iloc[idx]
            case_id = str(row.get("golden_id", row.get("conversation_id", f"case_{idx}")))
            hist_msg = str(row["customer_message"])
            hist_reply = str(row.get("brand_reply", "Please reach out to AppleSupport in DM for assistance."))
            hist_intent = str(row.get("intent", "general_device_support"))
            raw_sim = round(float(sims[idx]), 4)

            tier, op_sim, expl = self.ranker.classify_evidence_tier(
                query_intent=query_intent,
                query_profile=query_profile,
                hist_intent=hist_intent,
                hist_text=hist_msg,
                base_similarity=raw_sim,
            )

            candidate_cases.append(
                RetrievedEvidenceCase(
                    case_id=case_id,
                    similarity_score=raw_sim,
                    operational_similarity=op_sim,
                    match_tier=tier,
                    historical_customer_message=hist_msg,
                    historical_brand_response=hist_reply,
                    historical_intent=hist_intent,
                    retrieval_explanation=expl,
                )
            )

        tier_priority = {
            EvidenceMatchTier.DIRECT_PROBLEM_MATCH: 1,
            EvidenceMatchTier.RELATED_PROBLEM_MATCH: 2,
            EvidenceMatchTier.RELATED_SYMPTOM: 3,
            EvidenceMatchTier.RELATED_CONTEXT: 4,
            EvidenceMatchTier.WEAK_SEMANTIC_MATCH: 5,
        }

        candidate_cases.sort(
            key=lambda c: (
                tier_priority.get(c.match_tier, 5),
                -c.operational_similarity,
                -c.similarity_score,
            )
        )

        return candidate_cases[:top_k]
