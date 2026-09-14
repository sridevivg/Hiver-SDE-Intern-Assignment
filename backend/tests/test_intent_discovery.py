"""
SupportGraph AI — Unit Tests for Intent Discovery (Phase 4)

Uses synthetic data and mocked embeddings to run fast, deterministic unit tests
without downloading heavy external models during test runs.
"""
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
    from app.nlp.intent_discovery import (
        ClusteringExperimentRow,
        IntentDiscoveryResult,
        build_discovery_corpus,
        extract_cluster_evidence,
        propose_cluster_label,
        reduce_dimensions_pca,
        run_clustering_experiments,
        sample_corpus,
        save_discovery_artifacts,
        select_candidate_k,
        validate_and_load_inputs,
    )
except ModuleNotFoundError:
    from backend.app.nlp.intent_discovery import (
        ClusteringExperimentRow,
        IntentDiscoveryResult,
        build_discovery_corpus,
        extract_cluster_evidence,
        propose_cluster_label,
        reduce_dimensions_pca,
        run_clustering_experiments,
        sample_corpus,
        save_discovery_artifacts,
        select_candidate_k,
        validate_and_load_inputs,
    )


@pytest.fixture
def sample_raw_data():
    """Create synthetic messages and quality DataFrames."""
    messages_data = {
        "conversation_id": [
            "conv_1", "conv_1", "conv_1",
            "conv_2", "conv_2",
            "conv_3", "conv_3",
            "conv_4", "conv_4",
        ],
        "tweet_id": [101, 102, 103, 201, 202, 301, 302, 401, 402],
        "author_id": ["u1", "AppleSupport", "u1", "u2", "AppleSupport", "u3", "AppleSupport", "u4", "AppleSupport"],
        "role": ["customer", "brand", "customer", "customer", "brand", "customer", "brand", "customer", "brand"],
        "depth": [0, 1, 2, 0, 1, 0, 1, 0, 1],
        "text": [
            "@AppleSupport my battery dies in 2 hours after iOS update",
            "We can help with that. DM us.",
            "Thanks DM sent",
            "@AppleSupport forgot my apple id password and cannot login",
            "Follow these steps to reset",
            "   ",  # unusable empty text
            "Hello",
            "@AppleSupport screen is cracked and touch is not responding",
            "Visit an Apple Store",
        ],
    }
    df_messages = pd.DataFrame(messages_data)

    quality_data = {
        "conversation_id": ["conv_1", "conv_2", "conv_3", "conv_4"],
        "quality_status": ["high", "high", "high", "low"],  # conv_4 is low quality
    }
    df_quality = pd.DataFrame(quality_data)

    return df_messages, df_quality


def test_build_discovery_corpus(sample_raw_data):
    df_messages, df_quality = sample_raw_data
    result = build_discovery_corpus(df_messages, df_quality, min_text_length=3)

    # conv_4 is low quality -> excluded
    # conv_3 depth 0 has empty text -> excluded
    # conv_1 depth 0 and conv_2 depth 0 are valid high quality customer opening messages
    assert result.candidate_customer_opening_count == 3  # conv_1, conv_2, conv_3
    assert result.excluded_unusable_count == 1          # conv_3 empty
    assert result.final_eligible_count == 2             # conv_1 and conv_2
    assert len(result.df) == 2

    # Check columns
    assert "conversation_id" in result.df.columns
    assert "tweet_id" in result.df.columns
    assert "original_text" in result.df.columns
    assert "normalized_text" in result.df.columns

    # Check normalization applied: user mention replaced
    first_norm = result.df.loc[0, "normalized_text"]
    assert "<USER>" in first_norm
    assert "@AppleSupport" not in first_norm


def test_sample_corpus_deterministic():
    data = {
        "conversation_id": [f"c_{i}" for i in range(50)],
        "tweet_id": list(range(50)),
        "original_text": [f"Message {i}" for i in range(50)],
        "normalized_text": [f"Message {i}" for i in range(50)],
    }
    df = pd.DataFrame(data)

    # When max_samples >= len(df), returns all
    sampled_all, meta_all = sample_corpus(df, max_samples=100, random_seed=42)
    assert len(sampled_all) == 50
    assert meta_all["sampling_strategy"] == "ALL_MESSAGES"

    # When max_samples < len(df), samples deterministically
    sampled_1, _ = sample_corpus(df, max_samples=20, random_seed=42)
    sampled_2, _ = sample_corpus(df, max_samples=20, random_seed=42)
    assert len(sampled_1) == 20
    assert sampled_1["tweet_id"].tolist() == sampled_2["tweet_id"].tolist()


def test_reduce_dimensions_pca():
    rng = np.random.RandomState(42)
    synth_embeddings = rng.randn(40, 128).astype(np.float32)

    reduced, pca = reduce_dimensions_pca(synth_embeddings, n_components=10, random_state=42)
    assert reduced.shape == (40, 10)
    assert 0.0 < np.sum(pca.explained_variance_ratio_) <= 1.0


def test_run_clustering_experiments():
    rng = np.random.RandomState(42)
    # Generate 3 distinct synthetic clusters
    c1 = rng.randn(30, 8) + np.array([5, 0, 0, 0, 0, 0, 0, 0])
    c2 = rng.randn(30, 8) + np.array([0, 5, 0, 0, 0, 0, 0, 0])
    c3 = rng.randn(30, 8) + np.array([0, 0, 5, 0, 0, 0, 0, 0])
    features = np.vstack([c1, c2, c3])

    experiments = run_clustering_experiments(features, k_min=2, k_max=4, random_state=42)
    assert len(experiments) == 3
    for exp in experiments:
        assert 2 <= exp.k <= 4
        assert -1.0 <= exp.silhouette_score <= 1.0
        assert exp.davies_bouldin_score >= 0.0
        assert exp.calinski_harabasz_score >= 0.0
        assert exp.smallest_cluster_size > 0
        assert exp.largest_cluster_size >= exp.smallest_cluster_size


def test_select_candidate_k():
    experiments = [
        ClusteringExperimentRow(k=4, silhouette_score=0.10, calinski_harabasz_score=100.0, davies_bouldin_score=2.5, smallest_cluster_size=50, largest_cluster_size=200, tiny_cluster_count=0, selection_notes=""),
        ClusteringExperimentRow(k=6, silhouette_score=0.25, calinski_harabasz_score=250.0, davies_bouldin_score=1.8, smallest_cluster_size=80, largest_cluster_size=150, tiny_cluster_count=0, selection_notes=""),
        ClusteringExperimentRow(k=8, silhouette_score=0.20, calinski_harabasz_score=210.0, davies_bouldin_score=2.0, smallest_cluster_size=5, largest_cluster_size=160, tiny_cluster_count=2, selection_notes=""),
    ]
    # k=6 has highest silhouette, best DB, high CH, and no tiny clusters
    selected = select_candidate_k(experiments, target_k_range=(4, 8))
    assert selected == 6


def test_propose_cluster_label():
    # Battery terms
    assert propose_cluster_label(["battery", "drain", "charge"], cluster_id=0) == "battery_power_issue"
    # Update terms
    assert propose_cluster_label(["update", "ios", "version"], cluster_id=1) == "software_update_issue"
    # Account terms
    assert propose_cluster_label(["appleid", "password", "login"], cluster_id=2) == "account_access_issue"
    # Ambiguous terms fallback
    fallback = propose_cluster_label(["foo", "bar"], cluster_id=3)
    assert "cluster_3" in fallback
    assert "unreviewed" in fallback


def test_extract_cluster_evidence():
    texts = [
        "my battery is draining fast after update",
        "battery percentage drops suddenly",
        "cannot login to apple id password reset",
        "appleid locked how to recover account",
    ]
    features = np.array([
        [1.0, 0.0],
        [1.1, 0.1],
        [0.0, 1.0],
        [0.1, 1.1],
    ])
    labels = np.array([0, 0, 1, 1])
    centroids = np.array([
        [1.05, 0.05],
        [0.05, 1.05],
    ])

    clusters = extract_cluster_evidence(texts, features, labels, k=2, centroids=centroids, top_n_terms=3, top_n_examples=2)
    assert len(clusters) == 2
    assert clusters[0].cluster_id == 0
    assert clusters[0].cluster_size == 2
    assert len(clusters[0].representative_examples) == 2
    assert clusters[0].review_status == "pending"
    assert clusters[0].final_intent_label is None


def test_validate_and_load_inputs_missing_file(tmp_path):
    brand_file = tmp_path / "brand.json"
    messages_file = tmp_path / "msg.parquet"
    qual_file = tmp_path / "qual.csv"

    with pytest.raises(FileNotFoundError):
        validate_and_load_inputs(messages_file, qual_file, brand_file)


def test_save_discovery_artifacts(tmp_path):
    corpus_df = pd.DataFrame({
        "conversation_id": ["c1", "c2"],
        "tweet_id": [1, 2],
        "original_text": ["t1", "t2"],
        "normalized_text": ["n1", "n2"],
        "cluster_id": [0, 1],
    })
    experiments = [
        ClusteringExperimentRow(k=2, silhouette_score=0.3, calinski_harabasz_score=150.0, davies_bouldin_score=1.2, smallest_cluster_size=1, largest_cluster_size=1, tiny_cluster_count=0, selection_notes="k=2"),
    ]
    clusters = [
        extract_cluster_evidence(
            texts=["battery drain", "password reset"],
            features=np.array([[1.0], [2.0]]),
            cluster_labels=np.array([0, 1]),
            k=2,
            centroids=np.array([[1.0], [2.0]]),
        )[0]
    ]

    result = IntentDiscoveryResult(
        corpus_df=corpus_df,
        experiments=experiments,
        selected_k=2,
        clusters=clusters,
        explained_variance_ratio=0.85,
        metadata={"random_seed": 42},
    )

    paths = save_discovery_artifacts(
        result=result,
        interim_dir=tmp_path,
        selected_brand="AppleSupport",
        embedding_model_name="test-model",
    )

    assert paths["corpus_csv"].exists()
    assert paths["experiments_csv"].exists()
    assert paths["review_csv"].exists()
    assert paths["taxonomy_json"].exists()

    # Verify JSON structure
    with open(paths["taxonomy_json"]) as f:
        tax = json.load(f)
    assert tax["status"] == "PROVISIONAL_HUMAN_REVIEW_REQUIRED"
    assert tax["selected_brand"] == "AppleSupport"
    assert tax["cluster_review_required"] is True
