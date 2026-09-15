"""
SupportGraph AI — Tests for Phase 5.10 Annotation Consistency & Taxonomy Boundary Validation

Validates:
1. Completed records filtering & pending record exclusion
2. Pairwise similarity matrix computation
3. Cross-label consistency candidate detection & priority assignment
4. Within-label coherence & small sample caveat handling
5. Cross-label boundary overlap analysis & soft guideline generation
6. Strict dataset immutability verification
7. Missing fields & edge case handling
8. Deterministic reproducibility
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from backend.app.evaluation.annotation_consistency import (
        HIGH_PRIORITY_SIMILARITY_THRESHOLD,
        LOW_PRIORITY_SIMILARITY_THRESHOLD,
        MEDIUM_PRIORITY_SIMILARITY_THRESHOLD,
        AnnotationConsistencyAnalyzer,
        ConsistencyCandidatePair,
        ConsistencySummary,
        WithinLabelCoherence,
        calculate_file_sha256,
        clean_text_for_similarity,
        extract_completed_reviews,
        extract_shared_signals,
        generate_ambiguity_explanation,
        generate_markdown_consistency_report,
    )
except ModuleNotFoundError:
    from app.evaluation.annotation_consistency import (  # type: ignore[no-redef]
        HIGH_PRIORITY_SIMILARITY_THRESHOLD,
        LOW_PRIORITY_SIMILARITY_THRESHOLD,
        MEDIUM_PRIORITY_SIMILARITY_THRESHOLD,
        AnnotationConsistencyAnalyzer,
        ConsistencyCandidatePair,
        ConsistencySummary,
        WithinLabelCoherence,
        calculate_file_sha256,
        clean_text_for_similarity,
        extract_completed_reviews,
        extract_shared_signals,
        generate_ambiguity_explanation,
        generate_markdown_consistency_report,
    )



@pytest.fixture
def sample_dataset() -> pd.DataFrame:
    """Fixture with a mix of completed, pending, and edge-case records."""
    return pd.DataFrame([
        {
            "golden_id": "gold_001",
            "customer_message": "My iPhone battery dies in two hours after updating to iOS 11.",
            "annotation_label": "battery_power_issue",
            "annotation_status": "reviewed",
        },
        {
            "golden_id": "gold_002",
            "customer_message": "Battery life is terrible on my iPhone after the recent iOS 11 update.",
            "annotation_label": "software_update_problem",
            "annotation_status": "overridden_ai_suggestion",
        },
        {
            "golden_id": "gold_003",
            "customer_message": "I cannot turn off Wi-Fi in the control center on my iPhone 7.",
            "annotation_label": "hardware_audio_connection_issue",
            "annotation_status": "completed",
        },
        {
            "golden_id": "gold_004",
            "customer_message": "Wi-Fi control center toggle is not working on iOS 11.",
            "annotation_label": "general_device_support",
            "annotation_status": "reviewed",
        },
        {
            "golden_id": "gold_005",
            "customer_message": "My screen is cracked and unresponsive to touch.",
            "annotation_label": "display_touch_issue",
            "annotation_status": "reviewed",
        },
        {
            "golden_id": "gold_006",
            "customer_message": "Typing letter I produces strange symbols on my keyboard.",
            "annotation_label": "keyboard_typing_issue",
            "annotation_status": "reviewed",
        },
        # Pending records that MUST be excluded
        {
            "golden_id": "gold_007",
            "customer_message": "Pending message that has no label.",
            "annotation_label": "",
            "annotation_status": "pending_human_review",
        },
        {
            "golden_id": "gold_008",
            "customer_message": "Another pending message.",
            "annotation_label": "pending_human_review",
            "annotation_status": "pending",
        },
    ])


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

class TestExtractionAndFiltering:
    def test_extract_completed_reviews_filters_pending_and_empty(self, sample_dataset: pd.DataFrame) -> None:
        """Verify that extract_completed_reviews extracts only completed items."""
        completed = extract_completed_reviews(sample_dataset)
        assert len(completed) == 6
        assert "gold_007" not in completed["golden_id"].values
        assert "gold_008" not in completed["golden_id"].values
        assert not completed["annotation_label"].str.contains("pending", case=False).any()
        assert (completed["annotation_label"] != "").all()

    def test_extract_completed_reviews_empty_dataframe(self) -> None:
        """Verify behavior on an empty DataFrame."""
        empty_df = pd.DataFrame(columns=["golden_id", "customer_message", "annotation_label", "annotation_status"])
        completed = extract_completed_reviews(empty_df)
        assert completed.empty


class TestTextPreprocessingAndSignals:
    def test_clean_text_normalizes_properly(self) -> None:
        """Verify text normalization strips URLs, user mentions, and punctuation."""
        raw = "@AppleSupport Hey! My <USER> iPhone battery is dying fast... Check https://t.co/xyz123."
        cleaned = clean_text_for_similarity(raw)
        assert "applesupport" not in cleaned
        assert "https" not in cleaned
        assert "xyz123" not in cleaned
        assert "battery is dying fast" in cleaned

    def test_extract_shared_signals_finds_overlap(self) -> None:
        """Verify shared semantic words and bigrams are extracted."""
        text_a = "iPhone battery life is draining after the update"
        text_b = "My phone battery life died after latest iOS update"
        signals = extract_shared_signals(text_a, text_b)
        assert any("battery" in s for s in signals)
        assert any("update" in s for s in signals)


class TestPairwiseSimilarityAndCandidates:
    def test_pairwise_similarity_properties(self, sample_dataset: pd.DataFrame) -> None:
        """Verify mathematical properties of the similarity matrix."""
        completed = extract_completed_reviews(sample_dataset)
        analyzer = AnnotationConsistencyAnalyzer()
        sim_matrix, _ = analyzer.compute_pairwise_similarities(completed)

        n = len(completed)
        assert sim_matrix.shape == (n, n)
        # Diagonal must be 1.0
        np.testing.assert_allclose(np.diag(sim_matrix), np.ones(n), atol=1e-5)
        # Matrix must be symmetric
        np.testing.assert_allclose(sim_matrix, sim_matrix.T, atol=1e-5)
        # Values must be bounded in [0, 1]
        assert (sim_matrix >= 0.0).all() and (sim_matrix <= 1.0 + 1e-5).all()

    def test_consistency_candidate_flagging(self, sample_dataset: pd.DataFrame) -> None:
        """Verify that high-similarity pairs with different labels are flagged."""
        completed = extract_completed_reviews(sample_dataset)
        analyzer = AnnotationConsistencyAnalyzer(
            high_sim_threshold=0.40,
            med_sim_threshold=0.25,
            low_sim_threshold=0.15,
        )
        sim_matrix, _ = analyzer.compute_pairwise_similarities(completed)
        candidates = analyzer.extract_consistency_candidates(completed, sim_matrix)

        assert len(candidates) > 0
        for c in candidates:
            assert c.human_label_a != c.human_label_b
            assert c.similarity_score >= 0.15
            assert c.priority in ("HIGH", "MEDIUM", "LOW")
            assert len(c.possible_ambiguity_explanation) > 0


class TestWithinLabelCoherence:
    def test_within_label_coherence_computation(self, sample_dataset: pd.DataFrame) -> None:
        """Verify intra-class coherence metrics and small-sample caveats."""
        completed = extract_completed_reviews(sample_dataset)
        analyzer = AnnotationConsistencyAnalyzer()
        sim_matrix, _ = analyzer.compute_pairwise_similarities(completed)
        coherence = analyzer.compute_within_label_coherence(completed, sim_matrix)

        assert "battery_power_issue" in coherence
        assert "software_update_problem" in coherence
        assert "general_device_support" in coherence

        for lbl, coh in coherence.items():
            assert coh.reviewed_count >= 1
            assert 0.0 <= coh.average_pairwise_similarity <= 1.0
            assert 0.0 <= coh.semantic_diversity_indicator <= 1.0
            # Since sample dataset has 1 item per class, small sample flag should be True
            assert coh.insufficient_sample_flag is True
            assert "N=" in coh.sample_size_caveat


class TestCrossLabelBoundaryAndSoftGuidelines:
    def test_boundary_analysis_and_soft_guidelines(self, sample_dataset: pd.DataFrame) -> None:
        """Verify cross-boundary analysis generates non-prescriptive soft guidelines."""
        completed = extract_completed_reviews(sample_dataset)
        analyzer = AnnotationConsistencyAnalyzer(low_sim_threshold=0.10)
        sim_matrix, _ = analyzer.compute_pairwise_similarities(completed)
        candidates = analyzer.extract_consistency_candidates(completed, sim_matrix)
        boundaries = analyzer.compute_cross_label_boundaries(completed, sim_matrix, candidates)

        assert len(boundaries) >= 4

        # Verify soft guidelines do NOT contain hard coercive phrases
        for b in boundaries:
            guideline = b.soft_annotation_guideline.lower()
            assert "always" not in guideline
            assert "must automatically" not in guideline
            assert "never" not in guideline
            # Verify guideline contains recommended soft language
            assert any(word in guideline for word in ["consider", "usually", "evidence favoring", "when"])


class TestImmutabilityAndEdgeCases:
    def test_source_dataset_immutability(self, sample_dataset: pd.DataFrame) -> None:
        """Verify that running analysis does not alter the underlying file."""
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".csv", delete=False) as tf:
            sample_dataset.to_csv(tf.name, index=False)
            temp_path = tf.name

        try:
            sha_before = calculate_file_sha256(temp_path)
            df = pd.read_csv(temp_path, dtype=str)

            analyzer = AnnotationConsistencyAnalyzer()
            summary, candidates, within_coherence, boundaries = analyzer.run_full_analysis(
                df=df,
                source_sha256=sha_before,
                total_records=len(df),
            )

            sha_after = calculate_file_sha256(temp_path)
            assert sha_before == sha_after, "Dataset SHA-256 changed during analysis!"
            assert summary.source_sha256 == sha_before
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_deterministic_reproducibility(self, sample_dataset: pd.DataFrame) -> None:
        """Verify that running analysis multiple times yields identical outputs."""
        completed = extract_completed_reviews(sample_dataset)
        analyzer = AnnotationConsistencyAnalyzer()

        sim_1, _ = analyzer.compute_pairwise_similarities(completed)
        sim_2, _ = analyzer.compute_pairwise_similarities(completed)
        np.testing.assert_array_equal(sim_1, sim_2)

        cand_1 = analyzer.extract_consistency_candidates(completed, sim_1)
        cand_2 = analyzer.extract_consistency_candidates(completed, sim_2)
        assert len(cand_1) == len(cand_2)
        for c1, c2 in zip(cand_1, cand_2):
            assert c1.to_dict() == c2.to_dict()

    def test_missing_and_malformed_fields(self) -> None:
        """Verify handling of records with missing, null, or unusual characters."""
        malformed_df = pd.DataFrame([
            {
                "golden_id": "gold_bad_1",
                "customer_message": None,
                "annotation_label": "general_device_support",
                "annotation_status": "reviewed",
            },
            {
                "golden_id": "gold_bad_2",
                "customer_message": "!!! ??? 12345 漢字 🚀",
                "annotation_label": "software_update_problem",
                "annotation_status": "reviewed",
            },
        ])

        analyzer = AnnotationConsistencyAnalyzer()
        summary, candidates, within_coherence, boundaries = analyzer.run_full_analysis(
            df=malformed_df,
            source_sha256="fake_sha",
            total_records=2,
        )

        assert summary.completed_reviews == 2
        assert isinstance(summary.total_pairs_evaluated, int)

    def test_markdown_report_generation(self, sample_dataset: pd.DataFrame) -> None:
        """Verify Markdown report generation compiles properly without errors."""
        completed = extract_completed_reviews(sample_dataset)
        analyzer = AnnotationConsistencyAnalyzer()
        summary, candidates, within_coherence, boundaries = analyzer.run_full_analysis(
            df=sample_dataset,
            source_sha256="dummy_sha256",
            total_records=len(sample_dataset),
        )

        report = generate_markdown_consistency_report(
            summary=summary,
            candidates=candidates,
            within_coherence=within_coherence,
            boundaries=boundaries,
        )

        assert "# Phase 5.10: Annotation Consistency & Taxonomy Boundary Validation" in report
        assert "SECTION 1 — OBSERVED DATA" in report
        assert "SECTION 2 — PRELIMINARY INTERPRETATION" in report
        assert "SECTION 3 — PROPOSED SOFT ANNOTATION GUIDELINES" in report
        assert "SECTION 4 — HUMAN CONSISTENCY REVIEW QUEUE" in report
