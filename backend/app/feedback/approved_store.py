"""
SupportGraph AI — Approved Evidence Store (Phase 13).

Stores versioned, promoted human-derived support evidence in data/evidence_approved/.
Only items in this store can be accessed by the CaseRetriever to contribute
as HUMAN_VALIDATED_EVIDENCE in production support resolution.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from app.core.logging import get_logger
    from app.feedback.schemas import ApprovedEvidenceItem
except ModuleNotFoundError:
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]
    from backend.app.feedback.schemas import (  # type: ignore[no-redef]
        ApprovedEvidenceItem,
    )

logger = get_logger(__name__)

DEFAULT_APPROVED_DIR = Path("data/evidence_approved")


class ApprovedEvidenceStore:
    """
    Versioned, file-backed store for approved and promoted operational support evidence.
    Integrates seamlessly with the CaseRetriever.
    """

    def __init__(self, base_dir: Optional[Path | str] = None) -> None:
        self.base_dir = Path(base_dir or DEFAULT_APPROVED_DIR)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._items_cache: List[ApprovedEvidenceItem] = []
        self._reload_cache()

    def _file_path(self, evidence_id: str) -> Path:
        return self.base_dir / f"{evidence_id}.json"

    def _reload_cache(self) -> None:
        """Reload in-memory cache from disk."""
        items: List[ApprovedEvidenceItem] = []
        for file in sorted(self.base_dir.glob("*.json")):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                items.append(ApprovedEvidenceItem.model_validate(data))
            except Exception as exc:
                logger.warning("Failed reading approved evidence file %s: %s", file, exc)
        self._items_cache = items

    def save_evidence(self, item: ApprovedEvidenceItem) -> str:
        """Persist approved evidence item to disk and update cache."""
        path = self._file_path(item.evidence_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(item.model_dump(mode="json"), f, indent=2, default=str)
        self._reload_cache()
        logger.info("Saved approved evidence item %s (v%s, %s)", item.evidence_id, item.version, item.problem_family)
        return item.evidence_id

    def get_evidence(self, evidence_id: str) -> Optional[ApprovedEvidenceItem]:
        """Retrieve approved evidence by ID."""
        for item in self._items_cache:
            if item.evidence_id == evidence_id:
                return item
        path = self._file_path(evidence_id)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return ApprovedEvidenceItem.model_validate(data)
        return None

    def list_evidence(self, problem_family: Optional[str] = None) -> List[ApprovedEvidenceItem]:
        """List all approved evidence items with optional family filter."""
        if not problem_family:
            return list(self._items_cache)
        return [it for it in self._items_cache if it.problem_family == problem_family]

    def search_evidence(
        self,
        query_text: str,
        query_family: Optional[str] = None,
        top_k: int = 3,
    ) -> List[Tuple[ApprovedEvidenceItem, float]]:
        """
        Search approved human evidence items using TF-IDF cosine similarity + family boost.
        Returns list of (ApprovedEvidenceItem, similarity_score) sorted descending.
        """
        if not self._items_cache:
            return []

        corpus_texts = [f"{it.customer_problem_summary} {it.brand_guidance}" for it in self._items_cache]
        try:
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
            tfidf_mat = vec.fit_transform(corpus_texts)
            q_vec = vec.transform([query_text])
            sims = cosine_similarity(q_vec, tfidf_mat).flatten()

            results: List[Tuple[ApprovedEvidenceItem, float]] = []
            for idx, item in enumerate(self._items_cache):
                base_sim = float(sims[idx])
                if base_sim < 0.05:
                    continue
                boost = 1.0
                if query_family and item.problem_family == query_family:
                    boost = 1.30
                score = round(min(1.0, base_sim * boost), 4)
                results.append((item, score))

            results.sort(key=lambda x: -x[1])
            return results[:top_k]
        except Exception as exc:
            logger.warning("Error searching approved evidence store: %s", exc)
            return []
