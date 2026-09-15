"""
SupportGraph AI — Adversarial & Leakage Tests: Historical Case Retriever (Phase 10.2)

Implements all 6 mandatory adversarial tests:
  - Test A: Same Words, Different Problem (MacBook + update + WiFi vs MacBook + update + battery)
  - Test B: Different Words, Same Operational Problem ("cannot connect to WiFi" vs "wireless network disconnecting")
  - Test C: New Operational Family (WiFi, software crash, Bluetooth outside the original 5 categories)
  - Test D: Semantic False Positive (superficially similar words, but operationally different)
  - Test E: Golden Dataset Leakage (intersection == empty & SHA-256 validation)
  - Test F: Safety Regression Check (ensuring weak/ambiguous cases are never unsafely promoted)
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest

from backend.app.retrieval.case_retriever import CaseRetriever
from backend.app.retrieval.evidence_ranker import (
    EvidenceMatchTier,
    EvidenceRanker,
    RetrievedEvidenceCase,
)
from backend.app.retrieval.historical_corpus_index import (
    GOLDEN_CSV_PATH,
    HistoricalCorpusIndex,
    compute_file_sha256,
)
from backend.app.retrieval.problem_family_registry import (
    OperationalProblemFamily,
    ProblemFamilyDetector,
)
from backend.app.understanding.problem_extractor import ProblemExtractor


@pytest.fixture(scope="module")
def shared_index():
    """Load or initialize HistoricalCorpusIndex for tests."""
    index = HistoricalCorpusIndex()
    return index


@pytest.fixture(scope="module")
def retriever(shared_index):
    """CaseRetriever with loaded HistoricalCorpusIndex."""
    return CaseRetriever(corpus_index=shared_index)


# ── Test A: Same Words, Different Problem ─────────────────────────────────────
def test_a_same_words_different_problem():
    """
    Test A: Two cases sharing context words ('MacBook', 'update') but with different
    problems (WiFi vs Battery) must NOT become DIRECT_PROBLEM_MATCH.
    """
    ranker = EvidenceRanker()
    extractor = ProblemExtractor()

    # Query: MacBook + update + WiFi
    query_text = "After the latest macOS update on my MacBook Pro, my WiFi keeps dropping completely."
    query_profile = extractor.extract(query_text)

    # Historical candidate: MacBook + update + Battery
    hist_battery_msg = "Updated my MacBook Pro to the latest OS and now the battery drains in under an hour."

    tier, op_sim, expl = ranker.classify_evidence_tier(
        query_intent="general_device_support",
        query_profile=query_profile,
        hist_intent="battery_power_issue",
        hist_text=hist_battery_msg,
        base_similarity=0.68,
        hist_family=OperationalProblemFamily.POWER_BATTERY,
    )

    # Must NOT be DIRECT_PROBLEM_MATCH or RELATED_PROBLEM_MATCH
    assert tier != EvidenceMatchTier.DIRECT_PROBLEM_MATCH
    assert tier != EvidenceMatchTier.RELATED_PROBLEM_MATCH
    # Must be RELATED_CONTEXT or WEAK_SEMANTIC_MATCH because problems are completely different
    assert tier in (EvidenceMatchTier.RELATED_CONTEXT, EvidenceMatchTier.WEAK_SEMANTIC_MATCH)


# ── Test B: Different Words, Same Operational Problem ────────────────────────
def test_b_different_words_same_operational_problem():
    """
    Test B: Two cases using completely different lexical vocabulary to describe
    the exact same operational problem (WiFi dropping) must be recognized as DIRECT or RELATED.
    """
    ranker = EvidenceRanker()
    extractor = ProblemExtractor()

    # Query: "My Mac cannot connect to WiFi"
    query_text = "My Mac cannot connect to WiFi at all"
    query_profile = extractor.extract(query_text)

    # Historical: "Wireless network keeps disconnecting on my Mac"
    hist_msg = "Wireless network keeps disconnecting on my Mac notebook"

    tier, op_sim, expl = ranker.classify_evidence_tier(
        query_intent="general_device_support",
        query_profile=query_profile,
        hist_intent="general_device_support",
        hist_text=hist_msg,
        base_similarity=0.45,
        hist_family=OperationalProblemFamily.CONNECTIVITY_WIFI,
    )

    # Should be DIRECT_PROBLEM_MATCH or RELATED_PROBLEM_MATCH, definitely NOT WEAK_SEMANTIC_MATCH
    assert tier in (EvidenceMatchTier.DIRECT_PROBLEM_MATCH, EvidenceMatchTier.RELATED_PROBLEM_MATCH)
    assert op_sim >= 0.50


# ── Test C: New Operational Family ───────────────────────────────────────────
def test_c_new_operational_family(retriever):
    """
    Test C: Query outside original 5 categories (e.g. WiFi turning on automatically)
    must retrieve relevant evidence and NOT collapse into WEAK_SEMANTIC_MATCH.
    """
    query = "why does WiFi turn on by itself when I turn it off manually"
    results = retriever.retrieve(query_text=query, top_k=3)

    assert len(results) > 0
    top_tier = results[0].match_tier
    # Must find strong or related evidence in the 80k historical corpus
    assert top_tier in (EvidenceMatchTier.DIRECT_PROBLEM_MATCH, EvidenceMatchTier.RELATED_PROBLEM_MATCH)
    assert results[0].historical_problem_family == "CONNECTIVITY_WIFI"


# ── Test D: Semantic False Positive ──────────────────────────────────────────
def test_d_semantic_false_positive():
    """
    Test D: Lexical / token overlap without operational problem alignment
    must remain WEAK_SEMANTIC_MATCH.
    """
    ranker = EvidenceRanker()
    extractor = ProblemExtractor()

    # Query: Password account lock
    query_text = "I forgot my Apple ID password and my account is locked out"
    query_profile = extractor.extract(query_text)

    # Historical: Mentions account and id in billing context
    hist_msg = "I was billed twice on my Apple account for an in-app purchase"

    tier, op_sim, expl = ranker.classify_evidence_tier(
        query_intent="account_access_issue",
        query_profile=query_profile,
        hist_intent="billing_purchase_issue",
        hist_text=hist_msg,
        base_similarity=0.55,
        hist_family=OperationalProblemFamily.BILLING_PAYMENT,
    )

    # Account access vs Billing is an operational conflict
    assert tier == EvidenceMatchTier.WEAK_SEMANTIC_MATCH
    assert op_sim < 0.35


# ── Test E: Golden Dataset Leakage Prevention ─────────────────────────────────
def test_e_golden_dataset_leakage(shared_index):
    """
    Test E: Zero overlap between historical corpus IDs and golden benchmark IDs.
    SHA-256 of protected golden dataset must match immutable ground truth.
    """
    expected_sha256 = "1d3e9b3b8bdef3750437e59b17ab7c27167fd1548c2bb29de984151296c5b45a"
    actual_sha256 = compute_file_sha256(GOLDEN_CSV_PATH)
    assert actual_sha256 == expected_sha256, f"Golden SHA-256 changed: {actual_sha256}"

    golden_df = pd.read_csv(GOLDEN_CSV_PATH)
    golden_conv_ids = set(golden_df["conversation_id"].dropna().astype(str).unique())

    index_case_ids = {c.case_id for c in shared_index.cases}
    overlap = index_case_ids.intersection(golden_conv_ids)
    assert len(overlap) == 0, f"Found {len(overlap)} golden IDs leaking into historical index!"


# ── Test F: Safety Regression Check ──────────────────────────────────────────
def test_f_safety_regression_check():
    """
    Test F: Verify that vague messages without clear operational content
    are NOT unsafely promoted to DIRECT_PROBLEM_MATCH.
    """
    ranker = EvidenceRanker()
    extractor = ProblemExtractor()

    # Vague message with no clear operational problem
    vague_query = "Hey @AppleSupport I need you to fix this right now please"
    vague_profile = extractor.extract(vague_query)

    hist_msg = "Help me fix this battery issue on my iPhone"

    tier, op_sim, expl = ranker.classify_evidence_tier(
        query_intent="general_device_support",
        query_profile=vague_profile,
        hist_intent="battery_power_issue",
        hist_text=hist_msg,
        base_similarity=0.45,
        hist_family=OperationalProblemFamily.POWER_BATTERY,
    )

    # Vague message must NOT be promoted to direct or related problem match
    assert tier in (EvidenceMatchTier.WEAK_SEMANTIC_MATCH, EvidenceMatchTier.RELATED_SYMPTOM)
    assert tier != EvidenceMatchTier.DIRECT_PROBLEM_MATCH
