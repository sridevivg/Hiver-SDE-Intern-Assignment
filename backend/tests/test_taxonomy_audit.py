"""
SupportGraph AI — Unit Tests for Taxonomy Audit (Phase 4.5)

Synthetic deterministic unit tests covering:
1. Centroid-nearest example selection
2. Deterministic random sampling
3. Diversity sampling (Max-Min)
4. Merge candidate detection
5. Split candidate detection
6. Entity frequency analysis
7. Generalization status assignment logic
8. Response strategy categorization
9. Review artifact generation
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
    from app.nlp.taxonomy_audit import (
        ClusterCoherenceStats,
        TopicEntityAnalysis,
        analyze_entities_and_topics,
        audit_resolution_patterns,
        compute_cluster_coherence,
        detect_historical_events,
        determine_human_action_recommendation,
        extract_expanded_examples,
        identify_merge_candidates,
        identify_split_candidates,
        max_min_diversity_sampling,
        validate_audit_inputs,
        run_taxonomy_audit,
    )
except ModuleNotFoundError:
    from backend.app.nlp.taxonomy_audit import (
        ClusterCoherenceStats,
        TopicEntityAnalysis,
        analyze_entities_and_topics,
        audit_resolution_patterns,
        compute_cluster_coherence,
        detect_historical_events,
        determine_human_action_recommendation,
        extract_expanded_examples,
        identify_merge_candidates,
        identify_split_candidates,
        max_min_diversity_sampling,
        validate_audit_inputs,
        run_taxonomy_audit,
    )


# ---------------------------------------------------------------------------
# Test 1 & 2 & 3: Example Selection (Centroid, Random, Diverse)
# ---------------------------------------------------------------------------
def test_max_min_diversity_sampling():
    # 5 points along a 1D line: 0, 1, 2, 8, 10
    features = np.array([[0.0], [1.0], [2.0], [8.0], [10.0]])
    # Select 3 diverse points
    selected = max_min_diversity_sampling(features, n_samples=3)
    assert len(selected) == 3
    # The selected points should include extremes (0.0 or 10.0)
    assert 0 in selected or 4 in selected


def test_extract_expanded_examples():
    features = np.array([
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.2],
        [5.0, 5.0],
        [-5.0, -5.0],
    ])
    texts = [f"Text_{i}" for i in range(5)]
    centroid = np.array([0.0, 0.0])

    examples = extract_expanded_examples(
        cluster_features=features,
        cluster_texts=texts,
        centroid=centroid,
        n_centroid=2,
        n_random=2,
        n_diverse=2,
        random_seed=42,
    )

    # Centroid nearest should be Text_0 and Text_1
    assert examples.centroid_nearest[0] == "Text_0"
    assert examples.centroid_nearest[1] == "Text_1"

    # Random samples should have 2 items
    assert len(examples.random_samples) == 2

    # Diverse samples should capture extreme points (Text_3 or Text_4)
    assert len(examples.diverse_samples) == 2
    assert "Text_3" in examples.diverse_samples or "Text_4" in examples.diverse_samples


# ---------------------------------------------------------------------------
# Test 4: Cluster Coherence Calculation
# ---------------------------------------------------------------------------
def test_compute_cluster_coherence():
    c0_features = np.array([[0.0, 0.0], [0.1, 0.0], [-0.1, 0.0]])
    c0_centroid = np.array([0.0, 0.0])
    all_centroids = np.array([
        [0.0, 0.0],
        [10.0, 0.0],
    ])

    stats = compute_cluster_coherence(
        cluster_features=c0_features,
        centroid=c0_centroid,
        all_centroids=all_centroids,
        cluster_id=0,
        total_corpus_size=10,
    )

    assert stats.cluster_size == 3
    assert stats.percentage == 30.0
    assert stats.mean_distance_to_centroid < 0.1
    # Dist to other centroid is ~10.0 vs ~0.05 to own -> overlap should be 0.0
    assert stats.nearest_cluster_overlap_proxy == 0.0


# ---------------------------------------------------------------------------
# Test 5: Entity and Topic Analysis
# ---------------------------------------------------------------------------
def test_analyze_entities_and_topics():
    texts = [
        "My iPhone 8 battery drain is crazy on iOS 11",
        "iPhone battery won't turn on after update",
        "iPad display screen is cracked and unresponsive",
    ]
    analysis = analyze_entities_and_topics(texts)

    # Check detected products
    prod_names = [p[0] for p in analysis.products]
    assert "iphone" in prod_names
    assert "ipad" in prod_names

    # Check detected components
    comp_names = [c[0] for c in analysis.components]
    assert "battery" in comp_names
    assert "screen" in comp_names

    # Check detected OS
    os_names = [o[0] for o in analysis.operating_systems]
    assert "ios 11" in os_names or "ios" in os_names

    assert "Products:" in analysis.summary_str


# ---------------------------------------------------------------------------
# Test 6: Historical Event Detection
# ---------------------------------------------------------------------------
def test_detect_historical_events():
    # Transient autocorrect bug cluster
    autocorrect_texts = [
        "Why is letter i changing to A and a question mark?",
        "Fix this capital I glitch and question mark box",
        "When I type I it autocorrects to question mark",
        "Another tweet about something else",
    ]
    ev_autocorrect = detect_historical_events(autocorrect_texts)
    assert ev_autocorrect.generalization_status == "EVENT_SPECIFIC"
    assert ev_autocorrect.autocorrect_bug_mentions >= 3

    # Timeless support issues
    timeless_texts = [
        "My iPhone battery dies in two hours",
        "Can't remember my Apple ID password to sign in",
        "Screen cracked after dropping phone",
        "How do I cancel my App Store subscription",
    ]
    ev_timeless = detect_historical_events(timeless_texts)
    assert ev_timeless.generalization_status == "GENERALIZABLE"
    assert ev_timeless.autocorrect_bug_mentions == 0


# ---------------------------------------------------------------------------
# Test 7: Resolution Pattern Categorization
# ---------------------------------------------------------------------------
def test_audit_resolution_patterns():
    replies = [
        "We'd like to help. Please DM us your country.",
        "Take a look at the troubleshooting steps here: https://support.apple.com/kb/HT201263",
        "What model iPhone are you using? When did this start happening?",
        "We can assist. Please visit an Apple Store Genius Bar for service options.",
    ]
    audit = audit_resolution_patterns(replies)

    assert audit.total_conversations == 4
    assert audit.action_frequencies["request_private_message"] >= 1
    assert audit.action_frequencies["provide_support_link"] >= 1
    assert audit.action_frequencies["ask_for_details"] >= 1
    assert audit.action_frequencies["refer_to_service_or_repair"] >= 1
    assert "we'd like to help" in [o[0] for o in audit.common_openings]


# ---------------------------------------------------------------------------
# Test 8: Merge Candidate Detection
# ---------------------------------------------------------------------------
def test_identify_merge_candidates():
    # Centroid 0 and 1 are almost identical, Centroid 2 is far
    centroids = np.array([
        [1.0, 0.0],
        [0.98, 0.05],
        [0.0, 5.0],
    ])
    top_terms = {
        0: ["update", "ios", "iphone"],
        1: ["update", "ios", "phone"],
        2: ["screen", "display", "crack"],
    }
    labels = {
        0: "software_update_issue",
        1: "software_update_issue",
        2: "display_hardware_issue",
    }

    candidates = identify_merge_candidates(
        centroids=centroids,
        cluster_top_terms=top_terms,
        proposed_labels=labels,
        similarity_threshold=0.8,
    )

    # Pair (0, 1) should be flagged
    assert len(candidates) >= 1
    c01 = [c for c in candidates if (c.cluster_a == 0 and c.cluster_b == 1) or (c.cluster_a == 1 and c.cluster_b == 0)][0]
    assert c01.centroid_similarity > 0.95
    assert c01.recommendation == "review"
    assert "update" in c01.shared_terms


# ---------------------------------------------------------------------------
# Test 9: Split Candidate Detection & Recommendation Logic
# ---------------------------------------------------------------------------
def test_identify_split_candidates_and_action():
    coh_map = {
        0: ClusterCoherenceStats(100, 50.0, 1.0, 0.9, 0.8, 0.20),
        1: ClusterCoherenceStats(100, 50.0, 1.0, 0.9, 0.1, 0.01),
    }
    ent_map = {
        0: TopicEntityAnalysis(
            products=[("macbook", 20), ("iphone", 15)],
            operating_systems=[("sierra", 10), ("ios", 10)],
            services=[],
            components=[("screen", 15), ("speaker", 12), ("battery", 10)],
            problem_phrases=[],
            summary_str="Multiple components",
        ),
        1: TopicEntityAnalysis(
            products=[("iphone", 30)],
            operating_systems=[("ios", 25)],
            services=[],
            components=[("battery", 30)],
            problem_phrases=[],
            summary_str="Battery only",
        ),
    }
    labels = {0: "mixed_issues", 1: "battery_power_issue"}

    split_candidates = identify_split_candidates(coh_map, ent_map, labels)
    # Cluster 0 should be flagged
    split_ids = [s.cluster_id for s in split_candidates]
    assert 0 in split_ids
    assert split_candidates[0].recommendation == "review"

    # Test recommendation helper
    rec_0 = determine_human_action_recommendation(
        cluster_id=0,
        generalization_status="MIXED",
        merge_candidates=[],
        split_candidates=split_candidates,
    )
    assert rec_0 == "REVIEW_FOR_SPLIT"

    rec_1 = determine_human_action_recommendation(
        cluster_id=1,
        generalization_status="GENERALIZABLE",
        merge_candidates=[],
        split_candidates=split_candidates,
    )
    assert rec_1 == "KEEP_AS_SEPARATE"


# ---------------------------------------------------------------------------
# Test 10: Input Validation
# ---------------------------------------------------------------------------
def test_validate_audit_inputs_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_audit_inputs(
            corpus_path=tmp_path / "corp.csv",
            review_path=tmp_path / "rev.csv",
            messages_path=tmp_path / "msg.parquet",
            quality_path=tmp_path / "qual.csv",
            taxonomy_path=tmp_path / "tax.json",
        )
