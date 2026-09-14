"""
SupportGraph AI — Unit Tests for Golden Set Label Validation (Phase 5)

Tests:
1. Schema and required column enforcement
2. Duplicate tweet_id and duplicate normalized text detection
3. Annotation completion metrics computation
4. Strict empty label detection
5. Taxonomy allowed labels verification (including 'unclear_needs_review')
6. Class imbalance alert detection
7. End-to-end golden set validation
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from app.evaluation.label_validation import (
        GoldenValidationError,
        ValidationReport,
        compute_annotation_completion,
        compute_label_distribution,
        detect_class_imbalance,
        validate_allowed_labels,
        validate_golden_set,
        validate_no_duplicate_normalized_messages,
        validate_no_duplicate_tweet_ids,
        validate_no_empty_labels,
        validate_required_columns,
    )
except ModuleNotFoundError:
    from backend.app.evaluation.label_validation import (  # type: ignore[no-redef]
        GoldenValidationError,
        ValidationReport,
        compute_annotation_completion,
        compute_label_distribution,
        detect_class_imbalance,
        validate_allowed_labels,
        validate_golden_set,
        validate_no_duplicate_normalized_messages,
        validate_no_duplicate_tweet_ids,
        validate_no_empty_labels,
        validate_required_columns,
    )


@pytest.fixture
def valid_labeled_df() -> pd.DataFrame:
    """Fixture providing a fully valid, labeled synthetic golden set."""
    return pd.DataFrame([
        {
            "golden_id": f"gold_{i:03d}",
            "tweet_id": f"tw_{i}",
            "conversation_id": f"conv_{i}",
            "customer_message": f"Message {i}",
            "normalized_message": f"message {i}",
            "candidate_intent": "battery_power_issue",
            "source_cluster": 1,
            "conversation_context": "Context",
            "annotation_label": "battery_power_issue" if i % 2 == 0 else "software_update_problem",
            "annotation_status": "completed",
            "annotator": "Annotator A",
            "notes": "",
        }
        for i in range(20)
    ])


def test_validate_required_columns() -> None:
    """Verify missing required columns are flagged."""
    df_incomplete = pd.DataFrame({"golden_id": ["1"], "tweet_id": ["10"]})
    errors = validate_required_columns(df_incomplete)
    assert len(errors) == 1
    assert "Missing required golden columns" in errors[0]


def test_validate_duplicates() -> None:
    """Verify duplicate tweet IDs and normalized texts are flagged."""
    df_dupes = pd.DataFrame({
        "tweet_id": ["101", "101", "102"],
        "normalized_message": ["text a", "text b", "text a"],
    })
    id_errors = validate_no_duplicate_tweet_ids(df_dupes)
    assert len(id_errors) == 1
    assert "duplicate tweet_id" in id_errors[0]

    text_errors = validate_no_duplicate_normalized_messages(df_dupes)
    assert len(text_errors) == 1
    assert "duplicate normalized text" in text_errors[0]


def test_compute_annotation_completion() -> None:
    """Verify calculation of annotation completion metrics."""
    df = pd.DataFrame({
        "annotation_label": ["battery_power_issue", "", "  ", "software_update_problem"]
    })
    stats = compute_annotation_completion(df)
    assert stats["total"] == 4
    assert stats["labeled"] == 2
    assert stats["empty"] == 2
    assert stats["percent_complete"] == 50.0


def test_validate_no_empty_labels() -> None:
    """Verify detection of empty and whitespace-only labels."""
    df = pd.DataFrame({
        "golden_id": ["g1", "g2", "g3"],
        "annotation_label": ["battery_power_issue", "", "   "],
    })
    errors = validate_no_empty_labels(df)
    assert len(errors) == 1
    assert "contains 2 unlabelled record(s)" in errors[0]


def test_validate_allowed_labels() -> None:
    """Verify detection of unauthorized labels and acceptance of unclear_needs_review."""
    allowed = ["battery_power_issue", "software_update_problem"]
    df = pd.DataFrame({
        "golden_id": ["g1", "g2", "g3"],
        "annotation_label": [
            "battery_power_issue",
            "fake_invented_label",
            "unclear_needs_review",
        ],
    })

    errors = validate_allowed_labels(df, allowed_labels=allowed, allow_unclear=True)
    assert len(errors) == 1
    assert "invalid label 'fake_invented_label'" in errors[0]


def test_detect_class_imbalance() -> None:
    """Verify detection of severe class imbalance under threshold."""
    # 98 battery, 2 software_update
    labels = ["battery_power_issue"] * 98 + ["software_update_problem"] * 2
    df = pd.DataFrame({"annotation_label": labels})
    warnings = detect_class_imbalance(df, min_class_ratio=0.05)
    assert len(warnings) == 1
    assert "software_update_problem" in warnings[0]


def test_validate_golden_set_full_pipeline(valid_labeled_df: pd.DataFrame) -> None:
    """Verify valid dataset passes complete validation pipeline."""
    allowed = ["battery_power_issue", "software_update_problem"]
    report: ValidationReport = validate_golden_set(
        valid_labeled_df,
        allowed_labels=allowed,
        require_complete=True,
    )
    assert report.is_valid is True
    assert len(report.errors) == 0
    assert report.total_records == 20
    assert report.completion_stats["percent_complete"] == 100.0
