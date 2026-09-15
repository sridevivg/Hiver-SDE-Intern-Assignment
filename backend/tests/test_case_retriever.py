"""
Tests for CaseRetriever and EvidenceRanker (Phase 7 Evidence Retrieval Layer)
"""
import pytest

from backend.app.retrieval.case_retriever import CaseRetriever
from backend.app.retrieval.evidence_ranker import (
    EvidenceMatchTier,
    EvidenceRanker,
    RetrievedEvidenceCase,
)
from backend.app.understanding.problem_extractor import CustomerProblemProfile


@pytest.fixture
def ranker() -> EvidenceRanker:
    return EvidenceRanker()


@pytest.fixture
def retriever() -> CaseRetriever:
    return CaseRetriever()


def test_evidence_ranker_direct_problem_match(ranker: EvidenceRanker) -> None:
    query_profile = CustomerProblemProfile(
        device="iPhone",
        primary_symptom="battery draining fast",
        information_sufficiency="sufficient",
    )
    hist_text = "my battery is draining so fast since morning"

    tier, op_sim, expl = ranker.classify_evidence_tier(
        query_intent="battery_power_issue",
        query_profile=query_profile,
        hist_intent="battery_power_issue",
        hist_text=hist_text,
        base_similarity=0.85,
    )

    assert tier == EvidenceMatchTier.DIRECT_PROBLEM_MATCH
    assert op_sim >= 0.85
    assert "battery" in expl.lower()


def test_evidence_ranker_weak_semantic_match_different_intent(ranker: EvidenceRanker) -> None:
    query_profile = CustomerProblemProfile(
        device="iPhone",
        primary_symptom="general inquiries",
        information_sufficiency="partial",
    )
    hist_text = "random query about something else"

    tier, op_sim, expl = ranker.classify_evidence_tier(
        query_intent="general_device_support",
        query_profile=query_profile,
        hist_intent="billing_purchase_issue",
        hist_text=hist_text,
        base_similarity=0.35,
    )

    assert tier == EvidenceMatchTier.WEAK_SEMANTIC_MATCH
    assert op_sim < 0.35


def test_case_retriever_retrieve(retriever: CaseRetriever) -> None:
    query_profile = CustomerProblemProfile(
        device="iPhone",
        primary_symptom="battery drain",
        information_sufficiency="sufficient",
    )
    cases = retriever.retrieve(
        query_text="battery dying fast",
        query_intent="battery_power_issue",
        query_profile=query_profile,
        top_k=3,
    )
    assert isinstance(cases, list)
    assert len(cases) <= 3
    if cases:
        assert isinstance(cases[0], RetrievedEvidenceCase)
