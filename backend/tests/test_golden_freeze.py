"""
SupportGraph AI — Unit Tests for Golden Set Freeze Workflow (Phase 5)

Tests:
1. Valid freeze process generating frozen CSV and JSON manifest
2. SHA256 checksum generation and verification
3. Immutability protection: blocks overwrite without explicit force_new_version
4. Validation rejection: blocks freezing unlabelled or invalid category datasets
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from app.evaluation.golden_freeze import compute_file_sha256, freeze_golden_dataset
    from app.evaluation.label_validation import GoldenValidationError
except ModuleNotFoundError:
    from backend.app.evaluation.golden_freeze import (  # type: ignore[no-redef]
        compute_file_sha256,
        freeze_golden_dataset,
    )
    from backend.app.evaluation.label_validation import GoldenValidationError  # type: ignore[no-redef]


@pytest.fixture
def valid_completed_annotation_file(tmp_path: Path) -> Path:
    """Create a temporary completed annotation CSV file."""
    intents = [
        "software_update_problem",
        "battery_power_issue",
        "display_touch_issue",
        "account_access_issue",
        "billing_purchase_issue",
        "keyboard_typing_issue",
        "mac_software_issue",
        "hardware_audio_connection_issue",
        "general_device_support",
    ]
    records = []
    for i in range(27):
        intent = intents[i % len(intents)]
        records.append({
            "golden_id": f"gold_{i+1:03d}",
            "tweet_id": str(2000 + i),
            "conversation_id": f"conv_{2000 + i}",
            "customer_message": f"Customer problem message {i}",
            "normalized_message": f"customer problem message {i}",
            "candidate_intent": intent,
            "source_cluster": 1,
            "conversation_context": "AppleSupport reply context",
            "annotation_label": intent,
            "annotation_status": "completed",
            "annotator": "Annotator Alice",
            "notes": "",
        })
    df = pd.DataFrame(records)
    csv_path = tmp_path / "completed_annotations.csv"
    df.to_csv(csv_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
    return csv_path


def test_freeze_golden_dataset_success(
    valid_completed_annotation_file: Path, tmp_path: Path
) -> None:
    """Verify that a valid completed annotation dataset is frozen successfully."""
    out_dir = tmp_path / "golden_output"
    csv_path, manifest_path, manifest = freeze_golden_dataset(
        input_csv_path=valid_completed_annotation_file,
        output_dir=out_dir,
        version="v1",
    )

    assert csv_path.exists()
    assert manifest_path.exists()
    assert manifest["version"] == "v1"
    assert manifest["row_count"] == 27
    assert manifest["sha256_checksum"] == compute_file_sha256(csv_path)
    assert len(manifest["intent_distribution"]) == 9
    assert manifest["validation_results"]["is_valid"] is True


def test_freeze_immutability_protection(
    valid_completed_annotation_file: Path, tmp_path: Path
) -> None:
    """Verify that attempting to overwrite an existing version raises FileExistsError."""
    out_dir = tmp_path / "golden_output"
    freeze_golden_dataset(
        input_csv_path=valid_completed_annotation_file,
        output_dir=out_dir,
        version="v1",
    )

    # Attempting to freeze the same version again without force flag must fail
    with pytest.raises(FileExistsError) as exc_info:
        freeze_golden_dataset(
            input_csv_path=valid_completed_annotation_file,
            output_dir=out_dir,
            version="v1",
            force_new_version=False,
        )
    assert "already exists" in str(exc_info.value)

    # With force_new_version=True it should succeed
    csv_path, _, _ = freeze_golden_dataset(
        input_csv_path=valid_completed_annotation_file,
        output_dir=out_dir,
        version="v1",
        force_new_version=True,
    )
    assert csv_path.exists()


def test_freeze_rejects_empty_labels(tmp_path: Path) -> None:
    """Verify freeze process loudly fails when input has unannotated/empty rows."""
    records = [
        {
            "golden_id": "gold_001",
            "tweet_id": "2001",
            "conversation_id": "conv_2001",
            "customer_message": "Customer msg",
            "normalized_message": "customer msg",
            "candidate_intent": "battery_power_issue",
            "source_cluster": 1,
            "conversation_context": "Reply",
            "annotation_label": "",  # Empty!
            "annotation_status": "pending",
            "annotator": "",
            "notes": "",
        }
    ]
    df = pd.DataFrame(records)
    csv_path = tmp_path / "pending_annotations.csv"
    df.to_csv(csv_path, index=False)

    out_dir = tmp_path / "golden_output"
    with pytest.raises(GoldenValidationError) as exc_info:
        freeze_golden_dataset(
            input_csv_path=csv_path,
            output_dir=out_dir,
            version="v1",
        )
    assert "Golden set validation failed" in str(exc_info.value)
