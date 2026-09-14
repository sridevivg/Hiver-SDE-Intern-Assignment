"""
SupportGraph AI — Unit Tests for Human Review Logic (Phase 5.5)

Tests:
1. Human decision ACCEPT populates label and sets status to 'reviewed'
2. Human decision OVERRIDE populates label and sets status to 'overridden_ai_suggestion'
3. Human decision OVERRIDE rejects unauthorized intent labels
4. Human decision UNCLEAR marks record as 'unclear_needs_review'
5. Human decision SKIP leaves label empty and status pending
6. Mandatory annotator requirement enforcement
7. Accurate progress metrics calculation
8. Manifest creation and serialization
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from app.evaluation.human_review import (
        InvalidReviewActionError,
        ReviewAction,
        apply_review_decision,
        compute_review_progress,
        is_suggestion_complete,
        is_valid_ai_suggestion,
        save_review_progress_manifest,
    )
except ModuleNotFoundError:
    from backend.app.evaluation.human_review import (  # type: ignore[no-redef]
        InvalidReviewActionError,
        ReviewAction,
        apply_review_decision,
        compute_review_progress,
        is_suggestion_complete,
        is_valid_ai_suggestion,
        save_review_progress_manifest,
    )


@pytest.fixture
def sample_suggested_record() -> dict[str, str]:
    return {
        "golden_id": "gold_001",
        "tweet_id": "1141068",
        "customer_message": "Battery is draining rapidly since update",
        "model_suggested_label": "battery_power_issue",
        "model_confidence": "0.90",
        "annotation_label": "",
        "annotation_status": "pending_human_review",
        "annotator": "",
        "notes": "",
    }


def test_human_accept_decision(sample_suggested_record: dict[str, str]) -> None:
    """Verify ACCEPT sets label, annotator, and reviewed status."""
    updated = apply_review_decision(
        record=sample_suggested_record,
        action=ReviewAction.ACCEPT,
        annotator="Alice",
    )
    assert updated["annotation_label"] == "battery_power_issue"
    assert updated["annotator"] == "Alice"
    assert updated["annotation_status"] == "reviewed"


def test_human_override_decision(sample_suggested_record: dict[str, str]) -> None:
    """Verify OVERRIDE sets custom taxonomy label and overridden status."""
    updated = apply_review_decision(
        record=sample_suggested_record,
        action=ReviewAction.OVERRIDE,
        annotator="Bob",
        selected_label="software_update_problem",
        notes="Customer explicitly connects drain to update process.",
    )
    assert updated["annotation_label"] == "software_update_problem"
    assert updated["annotator"] == "Bob"
    assert updated["annotation_status"] == "overridden_ai_suggestion"
    assert "update process" in updated["notes"]


def test_human_override_rejects_invalid_label(sample_suggested_record: dict[str, str]) -> None:
    """Verify OVERRIDE raises error when label does not exist in taxonomy."""
    with pytest.raises(InvalidReviewActionError):
        apply_review_decision(
            record=sample_suggested_record,
            action=ReviewAction.OVERRIDE,
            annotator="Bob",
            selected_label="invented_nonexistent_category",
        )


def test_human_unclear_decision(sample_suggested_record: dict[str, str]) -> None:
    """Verify UNCLEAR sets unclear_needs_review label."""
    updated = apply_review_decision(
        record=sample_suggested_record,
        action=ReviewAction.UNCLEAR,
        annotator="Charlie",
    )
    assert updated["annotation_label"] == "unclear_needs_review"
    assert updated["annotator"] == "Charlie"
    assert updated["annotation_status"] == "reviewed"


def test_human_skip_decision(sample_suggested_record: dict[str, str]) -> None:
    """Verify SKIP leaves annotation_label strictly empty."""
    updated = apply_review_decision(
        record=sample_suggested_record,
        action=ReviewAction.SKIP,
        annotator="",
    )
    assert updated["annotation_label"] == ""
    assert updated["annotation_status"] == "pending_human_review"


def test_missing_annotator_raises_value_error(sample_suggested_record: dict[str, str]) -> None:
    """Verify non-SKIP decision without annotator raises ValueError."""
    with pytest.raises(ValueError):
        apply_review_decision(
            record=sample_suggested_record,
            action=ReviewAction.ACCEPT,
            annotator="",
        )


def test_compute_review_progress() -> None:
    """Verify progress statistics calculation across varied records."""
    df = pd.DataFrame([
        # Record 1: Accepted suggestion
        {
            "model_suggested_label": "battery_power_issue",
            "annotation_label": "battery_power_issue",
            "annotation_status": "reviewed",
        },
        # Record 2: Overridden suggestion
        {
            "model_suggested_label": "software_update_problem",
            "annotation_label": "display_touch_issue",
            "annotation_status": "overridden_ai_suggestion",
        },
        # Record 3: Marked unclear
        {
            "model_suggested_label": "general_device_support",
            "annotation_label": "unclear_needs_review",
            "annotation_status": "reviewed",
        },
        # Record 4: Untouched / Pending
        {
            "model_suggested_label": "mac_software_issue",
            "annotation_label": "",
            "annotation_status": "pending_human_review",
        },
    ])

    stats = compute_review_progress(df)
    assert stats["total_records"] == 4
    assert stats["ai_suggestions_completed"] == 4
    assert stats["human_reviewed"] == 3
    assert stats["pending_human_review"] == 1
    assert stats["accepted_ai_suggestions"] == 1
    assert stats["overridden_ai_suggestions"] == 1
    assert stats["unclear_records"] == 1


def test_save_review_progress_manifest(tmp_path: Path) -> None:
    """Verify progress manifest writing to JSON."""
    stats = {"total_records": 10, "human_reviewed": 5}
    out_file = tmp_path / "progress.json"
    p = save_review_progress_manifest(stats, out_file)
    assert p.exists()
    with open(p, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["total_records"] == 10
    assert loaded["human_reviewed"] == 5


def test_is_valid_ai_suggestion_matrix() -> None:
    """Verify is_valid_ai_suggestion correctly validates and rejects various inputs."""
    allowed = {"battery_power_issue", "display_touch_issue", "unclear_needs_review"}

    # Valid inputs
    assert is_valid_ai_suggestion("battery_power_issue", allowed, "success") is True
    assert is_valid_ai_suggestion("display_touch_issue", allowed, None) is True
    assert is_valid_ai_suggestion("unclear_needs_review", allowed, "success") is True

    # Empty / whitespace
    assert is_valid_ai_suggestion("", allowed, "success") is False
    assert is_valid_ai_suggestion("   ", allowed, "success") is False

    # NaN / None / float
    assert is_valid_ai_suggestion(None, allowed, "success") is False
    assert is_valid_ai_suggestion(float("nan"), allowed, "success") is False

    # Literal 'nan', 'none', 'null' strings
    assert is_valid_ai_suggestion("nan", allowed, "success") is False
    assert is_valid_ai_suggestion("NaN", allowed, "success") is False
    assert is_valid_ai_suggestion("none", allowed, "success") is False
    assert is_valid_ai_suggestion("null", allowed, "success") is False

    # Unauthorized / invented labels
    assert is_valid_ai_suggestion("invented_label", allowed, "success") is False

    # Failed or invalid statuses
    assert is_valid_ai_suggestion("battery_power_issue", allowed, "failed") is False
    assert is_valid_ai_suggestion("battery_power_issue", allowed, "invalid_model_output") is False
    assert is_valid_ai_suggestion("battery_power_issue", allowed, "error") is False


def test_accept_nan_or_invalid_suggestion_raises_error(sample_suggested_record: dict[str, str]) -> None:
    """Verify ACCEPT raises InvalidReviewActionError on NaN, 'nan', empty, or failed suggestions."""
    # Test with 'nan' string
    rec_nan = dict(sample_suggested_record)
    rec_nan["model_suggested_label"] = "nan"
    with pytest.raises(InvalidReviewActionError, match="Cannot ACCEPT suggestion"):
        apply_review_decision(record=rec_nan, action=ReviewAction.ACCEPT, annotator="Alice")

    # Test with float NaN
    rec_float_nan = dict(sample_suggested_record)
    rec_float_nan["model_suggested_label"] = float("nan")
    with pytest.raises(InvalidReviewActionError, match="Cannot ACCEPT suggestion"):
        apply_review_decision(record=rec_float_nan, action=ReviewAction.ACCEPT, annotator="Alice")

    # Test with empty string
    rec_empty = dict(sample_suggested_record)
    rec_empty["model_suggested_label"] = ""
    with pytest.raises(InvalidReviewActionError, match="Cannot ACCEPT suggestion"):
        apply_review_decision(record=rec_empty, action=ReviewAction.ACCEPT, annotator="Alice")

    # Test with failed status
    rec_failed = dict(sample_suggested_record)
    rec_failed["model_suggested_label"] = "battery_power_issue"
    rec_failed["suggestion_status"] = "failed"
    with pytest.raises(InvalidReviewActionError, match="Cannot ACCEPT suggestion"):
        apply_review_decision(record=rec_failed, action=ReviewAction.ACCEPT, annotator="Alice")


def test_compute_review_progress_with_nans_and_float_columns() -> None:
    """Verify compute_review_progress does not crash on float/NaN columns and counts accurately."""
    df = pd.DataFrame({
        "golden_id": ["g1", "g2", "g3", "g4"],
        "model_suggested_label": ["battery_power_issue", float("nan"), "nan", "display_touch_issue"],
        "suggestion_status": ["success", float("nan"), "failed", "success"],
        "annotation_label": ["battery_power_issue", float("nan"), float("nan"), ""],
        "annotation_status": ["reviewed", "pending_human_review", "pending_human_review", "pending_human_review"],
        "annotator": ["Alice", float("nan"), float("nan"), ""],
    })

    stats = compute_review_progress(df)
    assert stats["total_records"] == 4
    assert stats["ai_suggestions_completed"] == 2  # Only g1 and g4 are valid
    assert stats["human_reviewed"] == 1           # Only g1
    assert stats["pending_human_review"] == 3
    assert stats["accepted_ai_suggestions"] == 1


def test_batch_review_session_and_auto_resume() -> None:
    """Verify that batched review accurately processes a slice of pending items and resumes subsequent batches."""
    initial_rows = [
        {"golden_id": f"gold_{i:03d}", "model_suggested_label": "battery_power_issue", "annotation_label": "", "annotation_status": "pending_human_review", "annotator": "", "suggestion_status": "success"}
        for i in range(1, 6)
    ]
    df = pd.DataFrame(initial_rows)

    # Session 1: Batch size 2
    pending_1 = [idx for idx, r in df.iterrows() if not r["annotation_label"]]
    assert len(pending_1) == 5
    batch_1 = pending_1[:2]

    for idx in batch_1:
        updated = apply_review_decision(df.iloc[idx], action=ReviewAction.ACCEPT, annotator="ReviewerA")
        for k, v in updated.items():
            df.at[idx, k] = v

    stats_1 = compute_review_progress(df)
    assert stats_1["human_reviewed"] == 2
    assert stats_1["pending_human_review"] == 3

    # Session 2: Auto-resume with next batch of 2
    pending_2 = [idx for idx, r in df.iterrows() if not r["annotation_label"]]
    assert len(pending_2) == 3
    assert pending_2 == [2, 3, 4]  # gold_003, gold_004, gold_005
    batch_2 = pending_2[:2]

    for idx in batch_2:
        updated = apply_review_decision(df.iloc[idx], action=ReviewAction.OVERRIDE, annotator="ReviewerB", selected_label="display_touch_issue")
        for k, v in updated.items():
            df.at[idx, k] = v

    stats_2 = compute_review_progress(df)
    assert stats_2["human_reviewed"] == 4
    assert stats_2["pending_human_review"] == 1
    assert stats_2["accepted_ai_suggestions"] == 2
    assert stats_2["overridden_ai_suggestions"] == 2

    # Verify immutability of Session 1 records
    assert df.at[0, "annotator"] == "ReviewerA"
    assert df.at[0, "annotation_label"] == "battery_power_issue"
    assert df.at[1, "annotator"] == "ReviewerA"
    assert df.at[2, "annotator"] == "ReviewerB"
    assert df.at[2, "annotation_label"] == "display_touch_issue"


