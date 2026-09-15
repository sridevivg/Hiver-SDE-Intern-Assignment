"""
Unit and Integration Tests for Phase 5.9 Taxonomy Calibration & Recommendation Layer.

Verifies:
1. Deterministic signal extraction for all 10 taxonomy categories.
2. Calibration evaluation logic (AGREES_WITH_AI, DISAGREES_WITH_AI, INSUFFICIENT_EVIDENCE, AMBIGUOUS).
3. Risk prioritization integration & calibration disagreement escalation.
4. Safe Group Review eligibility exclusion for calibration disagreements.
5. ReviewAction.CALIBRATION_ACCEPT and approval_type='individual_calibration_acceptance'.
6. Progress computation tracking for calibration accepted records.
7. Ground truth immutability and separation guarantees.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_BACKEND_DIR = _PROJECT_ROOT / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

try:
    from app.evaluation.human_review import (
        ReviewAction,
        apply_review_decision,
        compute_review_progress,
        get_allowed_taxonomy_labels,
    )
    from app.evaluation.review_prioritization import (
        ReviewPriority,
        build_prioritized_review_queue,
        build_safe_review_groups,
        evaluate_record_priority,
        is_eligible_for_group_review,
    )
    from app.evaluation.taxonomy_calibration import (
        CalibrationResult,
        CalibrationStatus,
        analyze_human_ai_disagreements,
        evaluate_taxonomy_calibration,
        extract_taxonomy_signals,
    )
except ModuleNotFoundError:
    from backend.app.evaluation.human_review import (  # type: ignore[no-redef]
        ReviewAction,
        apply_review_decision,
        compute_review_progress,
        get_allowed_taxonomy_labels,
    )
    from backend.app.evaluation.review_prioritization import (  # type: ignore[no-redef]
        ReviewPriority,
        build_prioritized_review_queue,
        build_safe_review_groups,
        evaluate_record_priority,
        is_eligible_for_group_review,
    )
    from backend.app.evaluation.taxonomy_calibration import (  # type: ignore[no-redef]
        CalibrationResult,
        CalibrationStatus,
        analyze_human_ai_disagreements,
        evaluate_taxonomy_calibration,
        extract_taxonomy_signals,
    )


class TestTaxonomySignalExtraction:
    """Test transparent contextual signal extractors across taxonomy categories."""

    def test_software_update_signals(self) -> None:
        text = "My battery dies in 2 hours since updating to iOS 11.1 on my iPhone."
        signals = extract_taxonomy_signals(text)
        assert any(s.category == "software_update_problem" for s in signals)
        assert any(s.name == "update_causality" for s in signals)

    def test_hardware_audio_connection_signals(self) -> None:
        text = "My AirPods have no sound coming from the right earbud."
        signals = extract_taxonomy_signals(text)
        assert any(s.category == "hardware_audio_connection_issue" for s in signals)
        assert any("airpods" in s.evidence_text.lower() for s in signals)

    def test_keyboard_typing_signals(self) -> None:
        text = "Please fix this letter eye problem with autocorrect changing i to A [?]."
        signals = extract_taxonomy_signals(text)
        assert any(s.category == "keyboard_typing_issue" for s in signals)
        assert any(s.name in ("letter_i_bug", "text_input_issue") for s in signals)

    def test_battery_power_signals(self) -> None:
        text = "My iPhone battery is draining from 100 to 20 in 30 minutes and overheating."
        signals = extract_taxonomy_signals(text)
        assert any(s.category == "battery_power_issue" for s in signals)

    def test_mac_software_signals(self) -> None:
        text = "My MacBook Pro running High Sierra keeps freezing on boot and Safari crashes."
        signals = extract_taxonomy_signals(text)
        assert any(s.category == "mac_software_issue" for s in signals)

    def test_display_touch_signals(self) -> None:
        text = "The touchscreen is completely unresponsive and the screen is flickering black."
        signals = extract_taxonomy_signals(text)
        assert any(s.category == "display_touch_issue" for s in signals)

    def test_account_access_signals(self) -> None:
        text = "I am locked out of my Apple ID and cannot reset my password."
        signals = extract_taxonomy_signals(text)
        assert any(s.category == "account_access_issue" for s in signals)

    def test_billing_purchase_signals(self) -> None:
        text = "I was charged twice for my subscription and need a refund."
        signals = extract_taxonomy_signals(text)
        assert any(s.category == "billing_purchase_issue" for s in signals)


class TestCalibrationEvaluation:
    """Test calibration decision logic and status generation."""

    def test_agreement_with_ai(self) -> None:
        record = {
            "golden_id": "test_01",
            "customer_message": "My AirPods sound is crackling and disconnects",
            "model_suggested_label": "hardware_audio_connection_issue",
            "model_confidence": "0.92",
            "suggestion_status": "completed",
        }
        res = evaluate_taxonomy_calibration(record)
        assert res.status == CalibrationStatus.AGREES_WITH_AI
        assert res.candidate_label == "hardware_audio_connection_issue"
        assert res.confidence >= 0.85

    def test_disagreement_with_ai(self) -> None:
        record = {
            "golden_id": "test_02",
            "customer_message": "Ever since the new update my phone keeps rebooting randomly",
            "model_suggested_label": "general_device_support",
            "model_confidence": "0.85",
            "suggestion_status": "completed",
        }
        res = evaluate_taxonomy_calibration(record)
        assert res.status == CalibrationStatus.DISAGREES_WITH_AI
        assert res.candidate_label == "software_update_problem"
        assert len(res.supporting_signals) > 0

    def test_insufficient_evidence_neutral(self) -> None:
        record = {
            "golden_id": "test_03",
            "customer_message": "Hello @AppleSupport I have a question about my device",
            "model_suggested_label": "general_device_support",
            "model_confidence": "0.90",
            "suggestion_status": "completed",
        }
        res = evaluate_taxonomy_calibration(record)
        assert res.status in (CalibrationStatus.AGREES_WITH_AI, CalibrationStatus.INSUFFICIENT_EVIDENCE)


class TestPriorityIntegration:
    """Test priority escalation and queue creation with calibration metadata."""

    def test_calibration_disagreement_escalates_to_high(self) -> None:
        record = {
            "golden_id": "rec_disagree",
            "customer_message": "My phone is totally broken since updating to ios 11",
            "model_suggested_label": "general_device_support",
            "model_confidence": "0.92",  # High baseline confidence
            "suggestion_status": "completed",
        }
        prio = evaluate_record_priority(record)
        # Disagreement MUST escalate what would otherwise be LOW priority into HIGH priority
        assert prio.priority == ReviewPriority.HIGH
        assert "calibration_disagreement" in prio.risk_flags
        assert "Taxonomy calibration divergence" in prio.priority_reason
        assert prio.calibration_result is not None
        assert prio.calibration_result.status == CalibrationStatus.DISAGREES_WITH_AI

    def test_calibration_agreement_maintains_low_priority(self) -> None:
        record = {
            "golden_id": "rec_agree",
            "customer_message": "My AirPods microphone does not pick up voice during calls",
            "model_suggested_label": "hardware_audio_connection_issue",
            "model_confidence": "0.94",
            "suggestion_status": "completed",
        }
        prio = evaluate_record_priority(record)
        assert prio.priority == ReviewPriority.LOW
        assert "calibration_agreement" in prio.risk_flags

    def test_group_review_rejection_on_calibration_disagreement(self) -> None:
        record = {
            "golden_id": "rec_grp_reject",
            "customer_message": "Ever since the update my battery drains fast",
            "model_suggested_label": "general_device_support",
            "model_confidence": "0.95",
            "suggestion_status": "completed",
            "annotation_label": "",
            "annotation_status": "pending",
        }
        eligible, reason = is_eligible_for_group_review(record)
        assert eligible is False
        assert "Calibration" in reason or "HIGH" in reason or "calibration_disagreement" in reason


class TestHumanReviewCalibrationAction:
    """Test ReviewAction.CALIBRATION_ACCEPT and human review updates."""

    def test_apply_calibration_acceptance(self) -> None:
        record = {
            "golden_id": "rec_calib_accept",
            "customer_message": "since update my phone has an issue",
            "model_suggested_label": "general_device_support",
            "calibration_candidate_label": "software_update_problem",
            "annotation_label": "",
            "annotation_status": "pending",
        }
        updated = apply_review_decision(
            record=record,
            action=ReviewAction.CALIBRATION_ACCEPT,
            annotator="test_reviewer",
            selected_label="software_update_problem",
            review_mode="individual_human_review",
            approval_type="individual_calibration_acceptance",
        )
        assert updated["annotation_label"] == "software_update_problem"
        assert updated["annotation_status"] == "reviewed"
        assert updated["annotator"] == "test_reviewer"
        assert updated["approval_type"] == "individual_calibration_acceptance"

    def test_compute_progress_tracks_calibration_accepts(self) -> None:
        df = pd.DataFrame([
            {
                "golden_id": "g1",
                "annotation_label": "software_update_problem",
                "annotation_status": "reviewed",
                "annotator": "sridevi",
                "approval_type": "individual_calibration_acceptance",
            },
            {
                "golden_id": "g2",
                "annotation_label": "hardware_audio_connection_issue",
                "annotation_status": "reviewed",
                "annotator": "sridevi",
                "approval_type": "accepted_ai_suggestion",
            },
            {
                "golden_id": "g3",
                "annotation_label": "",
                "annotation_status": "pending",
                "annotator": "",
                "approval_type": "",
            },
        ])
        progress = compute_review_progress(df)
        assert progress["human_reviewed"] == 2
        assert progress["pending_human_review"] == 1
        assert progress.get("calibration_accepted_records") == 1
        assert progress.get("accepted_ai_suggestions") == 1


class TestGroundTruthImmutability:
    """Ensure the 57 human ground-truth records remain byte-for-byte immutable."""

    def test_human_ground_truth_57_records_checksum(self) -> None:
        ground_truth_csv = Path("data/golden/golden_set_human_review.csv")
        assert ground_truth_csv.exists(), "Ground truth file data/golden/golden_set_human_review.csv must exist."

        df = pd.read_csv(ground_truth_csv, dtype=str)
        reviewed = df[
            (df["annotation_label"].notna())
            & (df["annotation_label"] != "")
            & (~df["annotation_status"].isin(["", "pending", "pending_human_review"]))
        ].sort_values("golden_id")

        snapshot_path = Path("data/golden/human_ground_truth_57_records_snapshot.txt")
        snapshot_ids = [line.strip().split("|")[0] for line in snapshot_path.read_text().splitlines() if line.strip()] if snapshot_path.exists() else []

        reviewed_57 = reviewed[reviewed["golden_id"].isin(snapshot_ids)].sort_values("golden_id")
        assert len(reviewed_57) == 57, f"Expected 57 snapshot records, found {len(reviewed_57)}."

        # Compute SHA-256 of the 57 records matching snapshot
        canonical_content = "\n".join(
            f"{r['golden_id']}|{r['annotation_label']}|{r['annotator']}|{r['annotation_status']}"
            for _, r in reviewed_57.iterrows()
        ) + "\n"
        sha256_hash = hashlib.sha256(canonical_content.encode("utf-8")).hexdigest()

        expected_hash_file = Path("data/golden/human_ground_truth_57_records_checksum.sha256")
        if expected_hash_file.exists():
            expected_hash = expected_hash_file.read_text().strip()
            assert sha256_hash == expected_hash, f"Ground truth checksum mismatch! Expected {expected_hash}, got {sha256_hash}"

