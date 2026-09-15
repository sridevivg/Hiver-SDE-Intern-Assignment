"""
SupportGraph AI — Retrieval Package (Phase 7–10.2)

Exposes:
- CaseRetriever: Historical support conversation retriever
- EvidenceRanker: Operational match tier classifier across 5 dimensions
- RetrievedEvidenceCase: Data model for retrieved historical evidence
- EvidenceMatchTier: Operational match tier enum
- HistoricalCorpusIndex: 80,487-conversation leakage-safe retrieval index
- OperationalProblemFamily: 20 corpus-informed operational problem families
"""
from __future__ import annotations

try:
    from app.retrieval.case_retriever import CaseRetriever
    from app.retrieval.evidence_ranker import (
        EvidenceMatchTier,
        EvidenceRanker,
        RetrievedEvidenceCase,
    )
    from app.retrieval.historical_corpus_index import HistoricalCorpusIndex
    from app.understanding.problem_family_registry import (
        OperationalProblemFamily,
        ProblemFamilyDefinition,
        ProblemFamilyDetector,
    )
except ModuleNotFoundError:
    from backend.app.retrieval.case_retriever import CaseRetriever  # type: ignore[no-redef]
    from backend.app.retrieval.evidence_ranker import (  # type: ignore[no-redef]
        EvidenceMatchTier,
        EvidenceRanker,
        RetrievedEvidenceCase,
    )
    from backend.app.retrieval.historical_corpus_index import (  # type: ignore[no-redef]
        HistoricalCorpusIndex,
    )
    from backend.app.understanding.problem_family_registry import (  # type: ignore[no-redef]
        OperationalProblemFamily,
        ProblemFamilyDefinition,
        ProblemFamilyDetector,
    )

__all__ = [
    "CaseRetriever",
    "EvidenceRanker",
    "EvidenceMatchTier",
    "RetrievedEvidenceCase",
    "HistoricalCorpusIndex",
    "OperationalProblemFamily",
    "ProblemFamilyDefinition",
    "ProblemFamilyDetector",
]
