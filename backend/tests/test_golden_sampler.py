"""
SupportGraph AI — Unit Tests for Golden Set Sampler (Phase 5)

Tests:
1. Stratified quota calculation summing precisely to target size
2. Duplicate tweet_id and duplicate normalized text elimination
3. Deterministic sampling reproducibility with random seed
4. Empty initial annotation labels (no synthetic labeling)
5. Manifest creation, metadata recording, and SHA256 hashing
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from app.evaluation.golden_sampler import (
        GoldenSamplingResult,
        compute_sha256,
        compute_stratified_quotas,
        sample_golden_candidates,
    )
except ModuleNotFoundError:
    from backend.app.evaluation.golden_sampler import (  # type: ignore[no-redef]
        GoldenSamplingResult,
        compute_sha256,
        compute_stratified_quotas,
        sample_golden_candidates,
    )


@pytest.fixture
def synthetic_corpus_and_messages(tmp_path: Path) -> tuple[Path, Path]:
    """Create small synthetic corpus and parquet messages fixtures."""
    # Create synthetic corpus
    records = []
    # 9 intents, generate 30 records each = 270 records
    cluster_mapping = {
        "software_update_problem": 3,
        "battery_power_issue": 1,
        "display_touch_issue": 2,
        "account_access_issue": 4,
        "billing_purchase_issue": 4,
        "keyboard_typing_issue": 0,
        "mac_software_issue": 5,
        "hardware_audio_connection_issue": 2,
        "general_device_support": 2,
    }

    tweet_counter = 1000
    for intent, cluster_id in cluster_mapping.items():
        for i in range(30):
            tweet_counter += 1
            # Add keywords to trigger regex
            kw = intent.replace("_issue", "").replace("_problem", "").replace("_", " ")
            records.append({
                "tweet_id": str(tweet_counter),
                "conversation_id": f"conv_{tweet_counter}",
                "text": f"Customer message about {kw} {i}",
                "normalized_text": f"customer message about {kw} {i}",
                "cluster_id": cluster_id,
                "author_id": f"user_{tweet_counter}",
                "clean_char_length": 40,
                "clean_word_count": 6,
            })

    # Add deliberate duplicate tweet_id and duplicate normalized text
    records.append({
        "tweet_id": "1001",  # duplicate ID
        "conversation_id": "conv_dupe_1",
        "text": "Duplicate tweet id record",
        "normalized_text": "duplicate tweet id record unique text",
        "cluster_id": 1,
        "author_id": "user_dupe",
        "clean_char_length": 35,
        "clean_word_count": 6,
    })
    records.append({
        "tweet_id": "9999",  # unique ID, but duplicate text
        "conversation_id": "conv_dupe_2",
        "text": "Customer message about software update 0",  # dupe text
        "normalized_text": "customer message about software update 0",
        "cluster_id": 3,
        "author_id": "user_dupe_2",
        "clean_char_length": 40,
        "clean_word_count": 6,
    })

    df_corpus = pd.DataFrame(records)
    corpus_file = tmp_path / "test_corpus.csv"
    df_corpus.to_csv(corpus_file, index=False)

    # Create dummy messages parquet for conversation context
    messages = []
    for r in records:
        messages.append({
            "conversation_id": r["conversation_id"],
            "role": "brand",
            "depth": 1,
            "text": f"AppleSupport reply for {r['conversation_id']}",
        })
    df_msgs = pd.DataFrame(messages)
    messages_file = tmp_path / "test_messages.parquet"
    df_msgs.to_parquet(messages_file, index=False)

    return corpus_file, messages_file


def test_compute_stratified_quotas() -> None:
    """Verify quotas distribute evenly across intents to match target sum exactly."""
    intent_counts = {
        "intent_a": 50,
        "intent_b": 50,
        "intent_c": 50,
        "intent_d": 50,
    }
    quotas = compute_stratified_quotas(intent_counts, target_size=100)
    assert sum(quotas.values()) == 100
    assert quotas == {"intent_a": 25, "intent_b": 25, "intent_c": 25, "intent_d": 25}

    # Test uneven division
    quotas_uneven = compute_stratified_quotas(intent_counts, target_size=102)
    assert sum(quotas_uneven.values()) == 102
    assert set(quotas_uneven.values()) == {25, 26}


def test_sample_golden_candidates_deduplication(
    synthetic_corpus_and_messages: tuple[Path, Path], tmp_path: Path
) -> None:
    """Verify deduplication removes duplicate tweet_ids and normalized texts."""
    corpus_file, messages_file = synthetic_corpus_and_messages
    out_dir = tmp_path / "golden_out"

    result = sample_golden_candidates(
        corpus_path=corpus_file,
        messages_path=messages_file,
        output_dir=out_dir,
        target_size=50,
        random_seed=42,
    )

    assert result.sample_size == 50
    df = result.df_candidates

    # Assert no duplicates
    assert df["tweet_id"].duplicated().sum() == 0
    assert df["normalized_message"].duplicated().sum() == 0

    # Assert all annotation fields are initialized empty
    assert (df["annotation_label"] == "").all()
    assert (df["annotation_status"] == "pending").all()
    assert (df["annotator"] == "").all()


def test_sampling_determinism(
    synthetic_corpus_and_messages: tuple[Path, Path], tmp_path: Path
) -> None:
    """Verify identical random seed produces identical sampled records."""
    corpus_file, messages_file = synthetic_corpus_and_messages

    out_1 = tmp_path / "golden_out_1"
    out_2 = tmp_path / "golden_out_2"

    res_1 = sample_golden_candidates(
        corpus_path=corpus_file,
        messages_path=messages_file,
        output_dir=out_1,
        target_size=40,
        random_seed=42,
    )
    res_2 = sample_golden_candidates(
        corpus_path=corpus_file,
        messages_path=messages_file,
        output_dir=out_2,
        target_size=40,
        random_seed=42,
    )

    assert res_1.df_candidates["tweet_id"].tolist() == res_2.df_candidates["tweet_id"].tolist()


def test_sampling_manifest_generation(
    synthetic_corpus_and_messages: tuple[Path, Path], tmp_path: Path
) -> None:
    """Verify sampling manifest contains all required metadata and hashes."""
    corpus_file, messages_file = synthetic_corpus_and_messages
    out_dir = tmp_path / "golden_out_manifest"

    res = sample_golden_candidates(
        corpus_path=corpus_file,
        messages_path=messages_file,
        output_dir=out_dir,
        target_size=45,
        random_seed=42,
    )

    assert res.manifest_path.exists()
    with open(res.manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["brand"] == "AppleSupport"
    assert manifest["random_seed"] == 42
    assert manifest["actual_sample_size"] == 45
    assert "corpus_source_sha256" in manifest
    assert "candidates_csv_sha256" in manifest
    assert manifest["duplicate_removal_statistics"]["duplicate_tweet_ids_removed"] == 1
    assert manifest["duplicate_removal_statistics"]["duplicate_texts_removed"] == 1
