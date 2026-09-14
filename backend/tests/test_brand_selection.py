"""
SupportGraph AI — Phase 2 Tests: Brand Selection

Tests cover:
  1. Metric normalization (min-max correctness)
  2. Weighted score calculation
  3. Weights sum validation
  4. Ranking behavior (higher score = lower rank number)
  5. Edge case: all metrics identical → normalized = 0
  6. Sensitivity analysis produces correct number of scenarios
  7. BrandSelectionConfig default weights sum to 1.0
  8. Config validation catches bad inputs
  9. Invalid scenario name raises ValueError
  10. compute_issue_diversity with synthetic messages

All tests use SYNTHETIC data only — never the real 492 MB dataset.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.app.data.brand_selection import (
    WEIGHT_SCENARIOS,
    BrandMetrics,
    BrandSelectionConfig,
    _minmax_normalize,
    compute_data_completeness,
    compute_interaction_volume,
    compute_issue_diversity,
    compute_reconstructability,
    compute_response_coverage,
    compute_weighted_scores,
    normalize_metrics,
    sensitivity_analysis,
)


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def simple_metrics() -> list[BrandMetrics]:
    """Three brands with distinct raw scores for normalization testing."""
    return [
        BrandMetrics(
            brand="BrandA",
            raw_interaction_volume=1000.0,
            raw_response_coverage=0.9,
            raw_reconstructability=0.8,
            raw_completeness=0.95,
            raw_diversity=0.7,
        ),
        BrandMetrics(
            brand="BrandB",
            raw_interaction_volume=500.0,
            raw_response_coverage=0.6,
            raw_reconstructability=0.5,
            raw_completeness=0.8,
            raw_diversity=0.5,
        ),
        BrandMetrics(
            brand="BrandC",
            raw_interaction_volume=100.0,
            raw_response_coverage=0.3,
            raw_reconstructability=0.2,
            raw_completeness=0.6,
            raw_diversity=0.2,
        ),
    ]


@pytest.fixture
def equal_metrics() -> list[BrandMetrics]:
    """All brands with identical raw scores — tests degenerate normalization."""
    return [
        BrandMetrics(brand="X", raw_interaction_volume=500.0, raw_response_coverage=0.5,
                     raw_reconstructability=0.5, raw_completeness=0.5, raw_diversity=0.5),
        BrandMetrics(brand="Y", raw_interaction_volume=500.0, raw_response_coverage=0.5,
                     raw_reconstructability=0.5, raw_completeness=0.5, raw_diversity=0.5),
    ]


@pytest.fixture
def brand_candidates_df() -> pd.DataFrame:
    """Synthetic brand_candidates DataFrame matching Phase 1 format."""
    return pd.DataFrame({
        "author_id": ["BrandA", "BrandB", "BrandC"],
        "total_messages": [1200, 600, 150],
        "outbound_count": [1000, 500, 100],
        "inbound_count": [0, 0, 0],
        "response_ratio": [1.0, 1.0, 1.0],
        "usable_interaction_count": [1000, 500, 100],
    })


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """
    Minimal synthetic dataset DataFrame.
    tweet_id: unique IDs
    author_id: brand or customer
    inbound: True=customer, False=brand
    text: simple message strings
    in_response_to_tweet_id: reply chain
    response_tweet_id: not used in tests
    """
    rows = [
        # Customer tweets (inbound=True)
        {"tweet_id": 1001, "author_id": "cust1", "inbound": True,
         "text": "My order is missing, please help!",
         "in_response_to_tweet_id": None, "response_tweet_id": None},
        {"tweet_id": 1002, "author_id": "cust2", "inbound": True,
         "text": "Your app keeps crashing on my phone.",
         "in_response_to_tweet_id": None, "response_tweet_id": None},
        {"tweet_id": 1003, "author_id": "cust3", "inbound": True,
         "text": "I was charged twice for the same item!",
         "in_response_to_tweet_id": None, "response_tweet_id": None},
        {"tweet_id": 1004, "author_id": "cust4", "inbound": True,
         "text": "Can you please refund my subscription?",
         "in_response_to_tweet_id": None, "response_tweet_id": None},
        {"tweet_id": 1005, "author_id": "cust5", "inbound": True,
         "text": "The website is down right now!",
         "in_response_to_tweet_id": None, "response_tweet_id": None},

        # BrandA replies (inbound=False) — all with valid inbound parents
        {"tweet_id": 2001, "author_id": "BrandA", "inbound": False,
         "text": "We're sorry! We'll look into your order immediately.",
         "in_response_to_tweet_id": 1001.0, "response_tweet_id": None},
        {"tweet_id": 2002, "author_id": "BrandA", "inbound": False,
         "text": "Please DM us your device details and we'll help.",
         "in_response_to_tweet_id": 1002.0, "response_tweet_id": None},
        {"tweet_id": 2003, "author_id": "BrandA", "inbound": False,
         "text": "We apologize for the double charge. Refund issued.",
         "in_response_to_tweet_id": 1003.0, "response_tweet_id": None},

        # BrandB replies — one with missing parent (not in dataset)
        {"tweet_id": 3001, "author_id": "BrandB", "inbound": False,
         "text": "Thanks for reaching out! We'll investigate.",
         "in_response_to_tweet_id": 1004.0, "response_tweet_id": None},
        {"tweet_id": 3002, "author_id": "BrandB", "inbound": False,
         "text": "Sorry for the inconvenience. Site is restored.",
         "in_response_to_tweet_id": 9999.0,   # missing parent!
         "response_tweet_id": None},
    ]
    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# Test 1: Min-max normalization correctness
# ---------------------------------------------------------------------------
class TestMinMaxNormalize:
    def test_basic_normalization(self) -> None:
        """Normalized min should be 0.0, max should be 1.0."""
        result = _minmax_normalize([100.0, 500.0, 1000.0])
        assert result[0] == pytest.approx(0.0)
        assert result[-1] == pytest.approx(1.0)
        # Middle value should be between 0 and 1
        assert 0.0 < result[1] < 1.0

    def test_identical_values_return_zeros(self) -> None:
        """All-equal values → all normalized to 0.0 (no discrimination)."""
        result = _minmax_normalize([5.0, 5.0, 5.0])
        assert all(v == 0.0 for v in result)

    def test_two_values(self) -> None:
        """Two distinct values → [0.0, 1.0]."""
        result = _minmax_normalize([10.0, 20.0])
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(1.0)

    def test_single_value(self) -> None:
        """Single value → [0.0] (no range)."""
        result = _minmax_normalize([42.0])
        assert result == [0.0]

    def test_all_in_zero_to_one_range(self) -> None:
        """All normalized values must be in [0, 1]."""
        values = [1.0, 3.0, 2.5, 7.0, 0.5, 10.0]
        result = _minmax_normalize(values)
        for v in result:
            assert 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# Test 2: normalize_metrics applies correctly to BrandMetrics list
# ---------------------------------------------------------------------------
class TestNormalizeMetrics:
    def test_highest_raw_gets_norm_one(self, simple_metrics: list[BrandMetrics]) -> None:
        """Brand with highest raw score should get normalized = 1.0."""
        normalized = normalize_metrics(simple_metrics)
        top = max(normalized, key=lambda m: m.raw_interaction_volume)
        assert top.norm_interaction_volume == pytest.approx(1.0)

    def test_lowest_raw_gets_norm_zero(self, simple_metrics: list[BrandMetrics]) -> None:
        """Brand with lowest raw score should get normalized = 0.0."""
        normalized = normalize_metrics(simple_metrics)
        bottom = min(normalized, key=lambda m: m.raw_interaction_volume)
        assert bottom.norm_interaction_volume == pytest.approx(0.0)

    def test_all_norms_in_range(self, simple_metrics: list[BrandMetrics]) -> None:
        """All normalized values must be in [0, 1]."""
        normalized = normalize_metrics(simple_metrics)
        for m in normalized:
            for attr in [
                "norm_interaction_volume", "norm_response_coverage",
                "norm_reconstructability", "norm_completeness", "norm_diversity",
            ]:
                v = getattr(m, attr)
                assert 0.0 <= v <= 1.0, f"{m.brand}.{attr} = {v} out of [0,1]"

    def test_equal_metrics_all_zero(self, equal_metrics: list[BrandMetrics]) -> None:
        """When all brands have identical raw scores, all norms = 0.0."""
        normalized = normalize_metrics(equal_metrics)
        for m in normalized:
            assert m.norm_interaction_volume == pytest.approx(0.0)
            assert m.norm_response_coverage == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test 3: Weighted score calculation
# ---------------------------------------------------------------------------
class TestWeightedScores:
    def test_scores_computed_correctly(self, simple_metrics: list[BrandMetrics]) -> None:
        """Final score should equal sum of (weight × norm_value) for all metrics."""
        metrics = normalize_metrics(simple_metrics)
        weights = WEIGHT_SCENARIOS["default"]
        scored = compute_weighted_scores(metrics, weights)

        for m in scored:
            expected = (
                weights["interaction_volume"]              * m.norm_interaction_volume
                + weights["response_coverage"]             * m.norm_response_coverage
                + weights["conversation_reconstructability"] * m.norm_reconstructability
                + weights["data_completeness"]             * m.norm_completeness
                + weights["issue_diversity"]               * m.norm_diversity
            )
            assert m.final_score == pytest.approx(expected, abs=1e-5)

    def test_scores_in_zero_one_range(self, simple_metrics: list[BrandMetrics]) -> None:
        """All final scores must be in [0, 1] when norms are in [0, 1]."""
        metrics = normalize_metrics(simple_metrics)
        weights = WEIGHT_SCENARIOS["default"]
        scored = compute_weighted_scores(metrics, weights)
        for m in scored:
            assert 0.0 <= m.final_score <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# Test 4: Ranking behavior
# ---------------------------------------------------------------------------
class TestRanking:
    def test_rank_one_has_highest_score(self, simple_metrics: list[BrandMetrics]) -> None:
        """Rank #1 must have the highest final_score."""
        metrics = normalize_metrics(simple_metrics)
        scored = compute_weighted_scores(metrics, WEIGHT_SCENARIOS["default"])
        rank1 = next(m for m in scored if m.rank == 1)
        assert rank1.final_score == max(m.final_score for m in scored)

    def test_ranks_are_unique_and_sequential(self, simple_metrics: list[BrandMetrics]) -> None:
        """Ranks must be 1, 2, 3, ... without gaps."""
        metrics = normalize_metrics(simple_metrics)
        scored = compute_weighted_scores(metrics, WEIGHT_SCENARIOS["default"])
        ranks = sorted(m.rank for m in scored)
        assert ranks == list(range(1, len(scored) + 1))

    def test_brand_a_wins_on_volume(self, simple_metrics: list[BrandMetrics]) -> None:
        """BrandA has highest raw values — should win under volume-focused weights."""
        metrics = normalize_metrics(simple_metrics)
        scored = compute_weighted_scores(metrics, WEIGHT_SCENARIOS["volume_focused"])
        winner = next(m for m in scored if m.rank == 1)
        assert winner.brand == "BrandA"


# ---------------------------------------------------------------------------
# Test 5: Weights sum validation
# ---------------------------------------------------------------------------
class TestWeightSums:
    def test_all_scenarios_sum_to_one(self) -> None:
        """Every weight scenario must sum to exactly 1.0."""
        for scenario_name, weights in WEIGHT_SCENARIOS.items():
            total = sum(weights.values())
            assert abs(total - 1.0) < 1e-9, (
                f"Scenario '{scenario_name}' weights sum to {total}, expected 1.0"
            )

    def test_default_config_weights_sum_to_one(self) -> None:
        """BrandSelectionConfig default weights must sum to 1.0."""
        config = BrandSelectionConfig()
        total = sum(config.weights.values())
        assert abs(total - 1.0) < 1e-9

    def test_quality_focused_weights_sum_to_one(self) -> None:
        config = BrandSelectionConfig(scenario="quality_focused")
        assert abs(sum(config.weights.values()) - 1.0) < 1e-9

    def test_volume_focused_weights_sum_to_one(self) -> None:
        config = BrandSelectionConfig(scenario="volume_focused")
        assert abs(sum(config.weights.values()) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Test 6: Config validation
# ---------------------------------------------------------------------------
class TestConfigValidation:
    def test_invalid_top_n_raises(self) -> None:
        config = BrandSelectionConfig(top_n=0)
        with pytest.raises(ValueError, match="top_n"):
            config.validate()

    def test_invalid_sample_size_raises(self) -> None:
        config = BrandSelectionConfig(sample_size=5)
        with pytest.raises(ValueError, match="sample_size"):
            config.validate()

    def test_invalid_scenario_raises(self) -> None:
        config = BrandSelectionConfig(scenario="nonexistent_scenario")
        with pytest.raises(ValueError, match="Unknown scenario"):
            _ = config.weights

    def test_valid_config_passes(self) -> None:
        config = BrandSelectionConfig(top_n=5, sample_size=100, random_seed=0)
        config.validate()  # Should not raise


# ---------------------------------------------------------------------------
# Test 7: Sensitivity analysis produces correct scenarios
# ---------------------------------------------------------------------------
class TestSensitivityAnalysis:
    def test_produces_all_scenarios(self, simple_metrics: list[BrandMetrics]) -> None:
        """Sensitivity analysis must produce one result per weight scenario."""
        metrics = normalize_metrics(simple_metrics)
        compute_weighted_scores(metrics, WEIGHT_SCENARIOS["default"])
        results = sensitivity_analysis(metrics)
        assert len(results) == len(WEIGHT_SCENARIOS)

    def test_each_scenario_has_a_winner(self, simple_metrics: list[BrandMetrics]) -> None:
        """Every scenario result must have a non-empty winner."""
        metrics = normalize_metrics(simple_metrics)
        compute_weighted_scores(metrics, WEIGHT_SCENARIOS["default"])
        results = sensitivity_analysis(metrics)
        for r in results:
            assert r.winner != "", f"Scenario '{r.scenario_name}' has no winner"

    def test_scenario_names_match(self, simple_metrics: list[BrandMetrics]) -> None:
        """Scenario names in results must match WEIGHT_SCENARIOS keys."""
        metrics = normalize_metrics(simple_metrics)
        compute_weighted_scores(metrics, WEIGHT_SCENARIOS["default"])
        results = sensitivity_analysis(metrics)
        result_names = {r.scenario_name for r in results}
        assert result_names == set(WEIGHT_SCENARIOS.keys())


# ---------------------------------------------------------------------------
# Test 8: compute_interaction_volume reads from DataFrame
# ---------------------------------------------------------------------------
class TestComputeInteractionVolume:
    def test_returns_correct_volume(self, brand_candidates_df: pd.DataFrame) -> None:
        vol = compute_interaction_volume("BrandA", brand_candidates_df)
        assert vol == 1000.0

    def test_missing_brand_returns_zero(self, brand_candidates_df: pd.DataFrame) -> None:
        vol = compute_interaction_volume("NonExistentBrand", brand_candidates_df)
        assert vol == 0.0


# ---------------------------------------------------------------------------
# Test 9: compute_response_coverage
# ---------------------------------------------------------------------------
class TestComputeResponseCoverage:
    def test_full_coverage(self, synthetic_df: pd.DataFrame) -> None:
        """BrandA: 3 replies, all parents (1001, 1002, 1003) exist in dataset."""
        tweet_id_set = set(synthetic_df["tweet_id"].tolist())
        brand_df = synthetic_df[synthetic_df["author_id"] == "BrandA"]
        coverage = compute_response_coverage("BrandA", brand_df, tweet_id_set)
        assert coverage == pytest.approx(1.0)

    def test_partial_coverage(self, synthetic_df: pd.DataFrame) -> None:
        """BrandB: 2 replies, one parent (9999) is NOT in dataset → coverage = 0.5."""
        tweet_id_set = set(synthetic_df["tweet_id"].tolist())
        brand_df = synthetic_df[synthetic_df["author_id"] == "BrandB"]
        coverage = compute_response_coverage("BrandB", brand_df, tweet_id_set)
        assert coverage == pytest.approx(0.5)

    def test_no_replies_returns_zero(self, synthetic_df: pd.DataFrame) -> None:
        """Brand with no reply rows → coverage = 0."""
        tweet_id_set = set(synthetic_df["tweet_id"].tolist())
        # Create a brand with no replies
        empty_df = synthetic_df[synthetic_df["author_id"] == "nobody"]
        coverage = compute_response_coverage("nobody", empty_df, tweet_id_set)
        assert coverage == 0.0


# ---------------------------------------------------------------------------
# Test 10: compute_issue_diversity with synthetic messages
# ---------------------------------------------------------------------------
class TestComputeIssueDiversity:
    def test_returns_score_in_range(self) -> None:
        """Diversity score must be in [0, 1]."""
        config = BrandSelectionConfig(
            sample_size=100, random_seed=42, n_clusters=3, tfidf_max_features=50
        )
        messages = [
            f"My order {i} is delayed by {i} days" for i in range(50)
        ] + [
            f"App crash on device {i} model {i}" for i in range(50)
        ]
        score, _ = compute_issue_diversity("TestBrand", messages, config)
        assert 0.0 <= score <= 1.0

    def test_empty_messages_returns_zero(self) -> None:
        """Empty message list → diversity = 0.0."""
        config = BrandSelectionConfig()
        score, sample_size = compute_issue_diversity("TestBrand", [], config)
        assert score == 0.0
        assert sample_size == 0

    def test_too_few_messages_returns_zero(self) -> None:
        """Fewer messages than n_clusters → diversity = 0.0."""
        config = BrandSelectionConfig(n_clusters=10, sample_size=100)
        messages = ["hello", "world"]
        score, _ = compute_issue_diversity("TestBrand", messages, config)
        assert score == 0.0

    def test_reproducible_with_same_seed(self) -> None:
        """Same config and messages must produce identical diversity scores."""
        config = BrandSelectionConfig(sample_size=50, random_seed=99, n_clusters=3)
        messages = [f"issue number {i} is critical and needs attention" for i in range(100)]
        score1, _ = compute_issue_diversity("BrandX", messages, config)
        score2, _ = compute_issue_diversity("BrandX", messages, config)
        assert score1 == pytest.approx(score2)

    def test_varied_messages_score_higher_than_identical(self) -> None:
        """Varied messages should produce higher diversity than identical messages."""
        config = BrandSelectionConfig(
            sample_size=200, random_seed=42, n_clusters=5, tfidf_max_features=100
        )
        topics = [
            "order missing tracking number delayed",
            "app crash error phone device not working",
            "billing charge refund double payment",
            "login password account locked reset",
            "delivery address shipping wrong location",
        ]
        varied = [f"{topics[i % len(topics)]} case {i}" for i in range(100)]
        identical = ["the exact same complaint repeated again and again" for _ in range(100)]

        score_varied, _ = compute_issue_diversity("VarBrand", varied, config)
        score_identical, _ = compute_issue_diversity("IdBrand", identical, config)
        # Varied should score strictly higher than identical (which has 1 cluster)
        assert score_varied > score_identical


# ---------------------------------------------------------------------------
# Test 12: Invalid metric values & edge case handling
# ---------------------------------------------------------------------------
class TestInvalidMetricValues:
    def test_weights_not_summing_to_one_raises(self, simple_metrics: list[BrandMetrics]) -> None:
        """compute_weighted_scores must raise ValueError if weights do not sum to 1.0."""
        invalid_weights = {
            "interaction_volume": 0.50,
            "response_coverage": 0.10,
            "conversation_reconstructability": 0.10,
            "data_completeness": 0.10,
            "issue_diversity": 0.05,  # Sum = 0.85
        }
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            compute_weighted_scores(simple_metrics, invalid_weights)

    def test_negative_weight_raises(self, simple_metrics: list[BrandMetrics]) -> None:
        """compute_weighted_scores must raise ValueError if any weight is negative."""
        invalid_weights = {
            "interaction_volume": 1.20,
            "response_coverage": -0.20,
            "conversation_reconstructability": 0.0,
            "data_completeness": 0.0,
            "issue_diversity": 0.0,
        }
        with pytest.raises(ValueError, match="cannot be negative"):
            compute_weighted_scores(simple_metrics, invalid_weights)

    def test_missing_weight_key_raises(self, simple_metrics: list[BrandMetrics]) -> None:
        """compute_weighted_scores must raise ValueError if any required weight is missing."""
        incomplete_weights = {
            "interaction_volume": 0.50,
            "response_coverage": 0.50,
        }
        with pytest.raises(ValueError, match="Missing required weights"):
            compute_weighted_scores(simple_metrics, incomplete_weights)

    def test_empty_metrics_list_returns_empty(self) -> None:
        """Empty metrics list returns empty list without error."""
        assert compute_weighted_scores([], WEIGHT_SCENARIOS["default"]) == []
        assert normalize_metrics([]) == []
        assert _minmax_normalize([]) == []

    def test_nan_values_in_normalization(self) -> None:
        """NaN values are converted to 0.0 safely in min-max normalization."""
        values = [float("nan"), 10.0, 20.0]
        norm = _minmax_normalize(values)
        assert len(norm) == 3
        assert not any(math.isnan(x) for x in norm)
        assert norm[0] == 0.0  # nan converted to 0.0
        assert norm[2] == 1.0  # max value
