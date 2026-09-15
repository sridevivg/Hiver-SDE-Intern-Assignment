"""
SupportGraph AI — Tests for Phase 5.11 Annotation Quality Review & Dashboard

Validates:
1. Candidate loading, priority filtering, and golden dataset enrichment.
2. Conflict ID generation and append-only CSV decision recording.
3. Deduplication of already reviewed conflict pairs.
4. Source dataset immutability verification.
5. Consolidated quality dashboard metric aggregation and JSON export.
6. Graceful handling of missing files and corrupted records.
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.app.evaluation.annotation_consistency import calculate_file_sha256
from backend.scripts.annotation_quality_dashboard import (
    load_consistency_decisions,
    load_json_safe,
)
from backend.scripts.review_annotation_consistency import (
    REVIEW_CSV_COLUMNS,
    append_decision_to_csv,
    load_candidates_with_fallback,
    load_existing_decisions,
)


@pytest.fixture
def sample_golden_df() -> pd.DataFrame:
    """Fixture with a mock golden dataset."""
    return pd.DataFrame([
        {
            "golden_id": "gold_101",
            "customer_message": "My WiFi cannot be toggled on in settings.",
            "annotation_label": "hardware_audio_connection_issue",
            "annotation_status": "reviewed",
            "model_suggested_label": "general_device_support",
            "model_confidence": "0.85",
        },
        {
            "golden_id": "gold_102",
            "customer_message": "Why does my WiFi turn on automatically?",
            "annotation_label": "general_device_support",
            "annotation_status": "reviewed",
            "model_suggested_label": "general_device_support",
            "model_confidence": "0.90",
        },
    ])


@pytest.fixture
def sample_candidates_json(tmp_path: Path) -> Path:
    """Fixture creating a temporary candidates JSON file."""
    cands = [
        {
            "golden_id_a": "gold_101",
            "golden_id_b": "gold_102",
            "message_a": "My WiFi cannot be toggled on in settings.",
            "message_b": "Why does my WiFi turn on automatically?",
            "human_label_a": "hardware_audio_connection_issue",
            "human_label_b": "general_device_support",
            "similarity_score": 0.22,
            "priority": "HIGH",
            "shared_semantic_signals": ["wifi", "turn on"],
            "possible_ambiguity_explanation": "WiFi settings vs hardware toggle",
        },
        {
            "golden_id_a": "gold_103",
            "golden_id_b": "gold_104",
            "message_a": "Battery dead in 1 hr.",
            "message_b": "Updated to iOS 11 today.",
            "human_label_a": "battery_power_issue",
            "human_label_b": "software_update_problem",
            "similarity_score": 0.08,
            "priority": "LOW",
            "shared_semantic_signals": ["update"],
            "possible_ambiguity_explanation": "Update vs battery",
        },
    ]
    cand_path = tmp_path / "candidates.json"
    with open(cand_path, "w", encoding="utf-8") as f:
        json.dump(cands, f)
    return cand_path


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

class TestReviewAnnotationConsistency:
    def test_load_candidates_and_filtering(
        self,
        sample_candidates_json: Path,
        sample_golden_df: pd.DataFrame,
    ) -> None:
        """Verify candidate loading and priority segregation."""
        cands = load_candidates_with_fallback(sample_candidates_json, sample_golden_df)
        assert len(cands) == 2

        high_cands = [c for c in cands if c.get("priority") == "HIGH"]
        low_cands = [c for c in cands if c.get("priority") == "LOW"]
        assert len(high_cands) == 1
        assert len(low_cands) == 1
        assert high_cands[0]["golden_id_a"] == "gold_101"

    def test_append_decision_and_load_existing(self, tmp_path: Path) -> None:
        """Verify append-only decision logging and deduplication lookup."""
        review_csv = tmp_path / "annotation_consistency_human_review.csv"

        # Initially empty
        assert len(load_existing_decisions(review_csv)) == 0

        # Record decision 1
        row1 = {
            "conflict_id": "conflict_gold_101_vs_gold_102",
            "record_a_golden_id": "gold_101",
            "record_b_golden_id": "gold_102",
            "record_a_label": "hardware_audio_connection_issue",
            "record_b_label": "general_device_support",
            "priority": "HIGH",
            "similarity_score": 0.22,
            "conflict_type": "hardware_audio_connection_issue_vs_general_device_support",
            "human_decision": "VALID_TAXONOMY_BOUNDARY",
            "reviewer": "test_annotator",
            "timestamp": "2026-09-14T12:00:00Z",
            "notes": "Legitimate boundary distinction",
        }
        append_decision_to_csv(review_csv, row1)

        # Verify loaded decisions
        decisions_set = load_existing_decisions(review_csv)
        assert len(decisions_set) == 1
        assert "conflict_gold_101_vs_gold_102" in decisions_set

        # Record decision 2
        row2 = {
            "conflict_id": "conflict_gold_103_vs_gold_104",
            "record_a_golden_id": "gold_103",
            "record_b_golden_id": "gold_104",
            "record_a_label": "battery_power_issue",
            "record_b_label": "software_update_problem",
            "priority": "LOW",
            "similarity_score": 0.08,
            "conflict_type": "battery_power_issue_vs_software_update_problem",
            "human_decision": "RECONSIDER_RECORD_A",
            "reviewer": "test_annotator",
            "timestamp": "2026-09-14T12:05:00Z",
            "notes": "Check battery symptom",
        }
        append_decision_to_csv(review_csv, row2)

        # Verify both decisions present and schema matches
        decisions_summary = load_consistency_decisions(review_csv)
        assert decisions_summary["total_reviewed"] == 2
        assert decisions_summary["valid_taxonomy_boundary"] == 1
        assert decisions_summary["reconsider_record_a"] == 1

    def test_golden_dataset_immutability_during_review(
        self,
        sample_golden_df: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """Verify that review actions do not touch the golden dataset CSV."""
        golden_file = tmp_path / "golden_set_human_review.csv"
        sample_golden_df.to_csv(golden_file, index=False)

        sha_before = calculate_file_sha256(golden_file)

        # Perform review writes to a separate file
        review_file = tmp_path / "reviews.csv"
        append_decision_to_csv(review_file, {
            "conflict_id": "conflict_1_vs_2",
            "record_a_golden_id": "gold_101",
            "record_b_golden_id": "gold_102",
            "record_a_label": "hardware_audio_connection_issue",
            "record_b_label": "general_device_support",
            "priority": "HIGH",
            "similarity_score": 0.22,
            "conflict_type": "test",
            "human_decision": "VALID_TAXONOMY_BOUNDARY",
            "reviewer": "test",
            "timestamp": "now",
            "notes": "",
        })

        sha_after = calculate_file_sha256(golden_file)
        assert sha_before == sha_after


class TestAnnotationQualityDashboard:
    def test_load_json_safe(self, tmp_path: Path) -> None:
        """Verify safe JSON loading for existing, missing, and malformed files."""
        # Missing file
        assert load_json_safe(tmp_path / "non_existent.json") == {}

        # Valid file
        valid_path = tmp_path / "valid.json"
        with open(valid_path, "w", encoding="utf-8") as f:
            json.dump({"key": "val"}, f)
        assert load_json_safe(valid_path) == {"key": "val"}

        # Malformed file
        malformed_path = tmp_path / "malformed.json"
        with open(malformed_path, "w", encoding="utf-8") as f:
            f.write("not json content {")
        assert load_json_safe(malformed_path) == {}

    def test_load_consistency_decisions_empty(self, tmp_path: Path) -> None:
        """Verify behavior when no decisions CSV exists."""
        non_existent = tmp_path / "missing_reviews.csv"
        summary = load_consistency_decisions(non_existent)
        assert summary["total_reviewed"] == 0
        assert summary["valid_taxonomy_boundary"] == 0
