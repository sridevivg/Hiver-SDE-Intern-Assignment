"""
SupportGraph AI — Phase 2: Brand Selection

Scientific multi-criteria brand selection for the SupportGraph AI project.

This module computes five metrics for each brand candidate:
  A. Usable Interaction Volume     (weight 0.30)
  B. Response Coverage             (weight 0.20)
  C. Conversation Reconstructability (weight 0.20)
  D. Data Completeness             (weight 0.15)
  E. Issue / Message Diversity     (weight 0.15)

All metric definitions, formulas, and computation decisions are documented
inline and in artifacts/reports/phase_2_brand_selection.md.

Raw data is NEVER modified by this module.
Results depend only on the actual dataset — nothing is fabricated.
"""
from __future__ import annotations

import logging
import math
import re
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

try:
    from app.core.logging import get_logger
except ModuleNotFoundError:
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Weight scenarios — must each sum to 1.0
# ---------------------------------------------------------------------------
WEIGHT_SCENARIOS: dict[str, dict[str, float]] = {
    "default": {
        "interaction_volume":             0.30,
        "response_coverage":              0.20,
        "conversation_reconstructability": 0.20,
        "data_completeness":              0.15,
        "issue_diversity":                0.15,
    },
    "volume_focused": {
        "interaction_volume":             0.50,
        "response_coverage":              0.15,
        "conversation_reconstructability": 0.15,
        "data_completeness":              0.10,
        "issue_diversity":                0.10,
    },
    "quality_focused": {
        "interaction_volume":             0.15,
        "response_coverage":              0.25,
        "conversation_reconstructability": 0.30,
        "data_completeness":              0.15,
        "issue_diversity":                0.15,
    },
}

# Validate all scenarios sum to 1.0
for _name, _w in WEIGHT_SCENARIOS.items():
    _total = sum(_w.values())
    assert abs(_total - 1.0) < 1e-9, f"Weights for scenario '{_name}' sum to {_total}, not 1.0"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class BrandSelectionConfig:
    """
    Configuration for the brand selection pipeline.

    All parameters are explicit — no hidden defaults.
    """
    top_n: int = 10
    """Number of candidate brands to evaluate (by Phase 1 usable interaction volume)."""

    sample_size: int = 5000
    """Maximum customer messages to sample per brand for diversity analysis."""

    random_seed: int = 42
    """Fixed seed for all random operations — ensures reproducibility."""

    n_clusters: int = 10
    """Number of KMeans clusters for diversity analysis."""

    tfidf_max_features: int = 500
    """Maximum TF-IDF vocabulary size."""

    tfidf_min_df: int = 2
    """Minimum document frequency for TF-IDF terms."""

    scenario: str = "default"
    """Weight scenario name — must be a key in WEIGHT_SCENARIOS."""

    @property
    def weights(self) -> dict[str, float]:
        if self.scenario not in WEIGHT_SCENARIOS:
            raise ValueError(
                f"Unknown scenario '{self.scenario}'. "
                f"Valid options: {list(WEIGHT_SCENARIOS.keys())}"
            )
        return WEIGHT_SCENARIOS[self.scenario]

    def validate(self) -> None:
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Weights must sum to 1.0, got {total:.6f}")
        if self.top_n < 1:
            raise ValueError(f"top_n must be >= 1, got {self.top_n}")
        if self.sample_size < 10:
            raise ValueError(f"sample_size must be >= 10, got {self.sample_size}")
        if self.n_clusters < 2:
            raise ValueError(f"n_clusters must be >= 2, got {self.n_clusters}")


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------
@dataclass
class BrandMetrics:
    """
    All raw and normalized scores for a single brand candidate.

    Raw scores are in their natural units.
    Normalized scores are in [0, 1] via min-max normalization across candidates.
    """
    brand: str

    # Raw metric values
    raw_interaction_volume: float = 0.0
    raw_response_coverage: float = 0.0
    raw_reconstructability: float = 0.0
    raw_completeness: float = 0.0
    raw_diversity: float = 0.0

    # Normalized values (set by normalize_metrics())
    norm_interaction_volume: float = 0.0
    norm_response_coverage: float = 0.0
    norm_reconstructability: float = 0.0
    norm_completeness: float = 0.0
    norm_diversity: float = 0.0

    # Final weighted score (set by compute_weighted_score())
    final_score: float = 0.0
    rank: int = 0

    # Weighted contributions
    contrib_interaction_volume: float = 0.0
    contrib_response_coverage: float = 0.0
    contrib_reconstructability: float = 0.0
    contrib_completeness: float = 0.0
    contrib_diversity: float = 0.0

    # Metadata
    sample_size_used: int = 0
    notes: str = ""


@dataclass
class SensitivityResult:
    """Result for one weight scenario."""
    scenario_name: str
    weights: dict[str, float]
    rankings: list[dict[str, Any]] = field(default_factory=list)
    winner: str = ""


@dataclass
class QualitativeSample:
    """A sampled interaction for qualitative review."""
    brand: str
    customer_text: str
    brand_reply: str
    customer_tweet_id: int
    brand_tweet_id: int


@dataclass
class BrandSelectionResult:
    """Complete result of the brand selection pipeline."""
    config: BrandSelectionConfig
    metrics: list[BrandMetrics]
    sensitivity: list[SensitivityResult]
    qualitative_samples: list[QualitativeSample]
    selected_brand: str
    selection_rationale: str


# ---------------------------------------------------------------------------
# Metric computation helpers
# ---------------------------------------------------------------------------
def _clean_text_for_diversity(text: str) -> str:
    """
    Normalize tweet text for TF-IDF diversity analysis.

    Removes @mentions, URLs, and extra whitespace.
    Does NOT remove stopwords (TF-IDF handles weighting).
    """
    # Remove @mentions
    text = re.sub(r"@\w+", "", text)
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # Remove hashtags (keep word)
    text = re.sub(r"#(\w+)", r"\1", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def _minmax_normalize(values: list[float]) -> list[float]:
    """
    Min-max normalize a list of values to [0, 1].

    If all values are identical, returns all zeros (no discrimination possible).
    This is clearly documented — it means the metric does not differentiate candidates.

    Args:
        values: List of raw metric values.

    Returns:
        List of normalized values in [0, 1].
    """
    if not values:
        return []
    arr = np.array(values, dtype=float)
    if np.isnan(arr).any() or np.isinf(arr).any():
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    min_v, max_v = arr.min(), arr.max()
    if abs(max_v - min_v) < 1e-12:
        logger.warning(
            "All values identical (%.6f) — min-max normalization returns all zeros. "
            "This metric does not discriminate candidates.",
            min_v,
        )
        return [0.0] * len(values)
    return list((arr - min_v) / (max_v - min_v))


# ---------------------------------------------------------------------------
# Core metric computations
# ---------------------------------------------------------------------------

def compute_interaction_volume(
    brand: str,
    brand_candidates_df: pd.DataFrame,
) -> float:
    """
    Metric A — Usable Interaction Volume.

    Definition:
      Count of brand outbound tweets where in_response_to_tweet_id is not null.
      This means the brand was directly replying to a customer tweet.

    Source:
      Directly read from Phase 1 brand_candidates.csv (usable_interaction_count).
      No recomputation needed unless the file is missing.

    Returns:
      Raw count (float).
    """
    row = brand_candidates_df[brand_candidates_df["author_id"] == brand]
    if row.empty:
        logger.warning("Brand '%s' not found in brand_candidates_df", brand)
        return 0.0
    return float(row["usable_interaction_count"].iloc[0])


def compute_response_coverage(
    brand: str,
    brand_df: pd.DataFrame,
    tweet_id_set: set[int],
) -> float:
    """
    Metric B — Response Coverage.

    Definition:
      Of all brand outbound tweets that claim to be replies
      (in_response_to_tweet_id is not null), what fraction have their
      claimed parent tweet ID actually present in the full dataset?

    Formula:
      replies_with_parent_in_dataset / total_brand_replies

    Rationale:
      If a brand reply's parent is missing from the dataset, we cannot
      reconstruct the context for that interaction. High coverage means
      we can actually use those interactions for RAG.

    Args:
      brand:           Brand author_id string.
      brand_df:        All rows from the full df where author_id == brand.
      tweet_id_set:    Set of all tweet_ids in the full dataset (for O(1) lookup).

    Returns:
      Coverage ratio in [0, 1].
    """
    reply_rows = brand_df[brand_df["in_response_to_tweet_id"].notna()]
    total_replies = len(reply_rows)

    if total_replies == 0:
        logger.warning("Brand '%s' has no reply rows — coverage = 0.0", brand)
        return 0.0

    # Check how many parent IDs exist in the full dataset
    parent_ids = reply_rows["in_response_to_tweet_id"].astype(int)
    covered = parent_ids.isin(tweet_id_set).sum()

    coverage = float(covered) / float(total_replies)
    logger.debug(
        "Brand '%s': %d/%d reply parents found in dataset (coverage=%.4f)",
        brand, covered, total_replies, coverage,
    )
    return coverage


def compute_reconstructability(
    brand: str,
    brand_df: pd.DataFrame,
    full_df: pd.DataFrame,
    tweet_id_set: set[int],
) -> float:
    """
    Metric C — Conversation Reconstructability.

    Definition:
      Of all brand replies whose parent tweet exists in the dataset,
      what fraction have that parent as a valid INBOUND (customer) tweet?

    Formula:
      valid_customer_parents / replies_with_parent_in_dataset

    Rationale:
      A brand reply is only useful for historical grounding if we can
      identify the original customer complaint/question. If the parent
      tweet is another brand tweet (e.g., a thread continuation), the
      interaction is less useful as a standalone customer→brand pair.

    Args:
      brand:         Brand author_id string.
      brand_df:      Rows from full df where author_id == brand.
      full_df:       Full dataset DataFrame.
      tweet_id_set:  Set of all tweet_ids.

    Returns:
      Reconstructability ratio in [0, 1].
    """
    reply_rows = brand_df[brand_df["in_response_to_tweet_id"].notna()].copy()
    total_replies = len(reply_rows)

    if total_replies == 0:
        return 0.0

    # Keep only replies whose parent is in the dataset
    parent_ids = reply_rows["in_response_to_tweet_id"].astype(int)
    has_parent = parent_ids.isin(tweet_id_set)
    reply_rows = reply_rows[has_parent]
    replies_with_parent = len(reply_rows)

    if replies_with_parent == 0:
        return 0.0

    # Check that the parent is an inbound (customer) tweet
    parent_id_list = reply_rows["in_response_to_tweet_id"].astype(int).tolist()
    # Build a lookup: tweet_id → inbound value
    parent_lookup = full_df[full_df["tweet_id"].isin(parent_id_list)][
        ["tweet_id", "inbound"]
    ].set_index("tweet_id")["inbound"].to_dict()

    # Count parents that are inbound=True
    valid_customer_parents = sum(
        1
        for pid in parent_id_list
        if parent_lookup.get(pid, False) is True
        or parent_lookup.get(pid, False) == True  # noqa: E712 — handles bool/object mix
    )

    reconstructability = float(valid_customer_parents) / float(replies_with_parent)
    logger.debug(
        "Brand '%s': %d/%d replies have valid inbound parents (recon=%.4f)",
        brand, valid_customer_parents, replies_with_parent, reconstructability,
    )
    return reconstructability


def compute_data_completeness(
    brand: str,
    brand_df: pd.DataFrame,
    tweet_id_set: set[int],
) -> float:
    """
    Metric D — Data Completeness.

    Definition:
      Composite score of two sub-metrics weighted equally (0.5 each):

      1. Text completeness:
         1 - (null_or_empty_text_count / total_brand_tweets)
         Measures whether brand tweets have actual text content.

      2. Reply chain integrity:
         1 - (broken_chain_rate)
         Where broken_chain_rate = % of brand reply tweets whose
         in_response_to_tweet_id does NOT exist in the full tweet_id set.
         (Only counted over tweets that have a non-null in_response_to_tweet_id.)

    Formula:
      0.5 × text_completeness + 0.5 × chain_integrity

    Returns:
      Completeness score in [0, 1].
    """
    total = len(brand_df)
    if total == 0:
        return 0.0

    # Sub-metric 1: text completeness
    null_text = brand_df["text"].isna().sum()
    empty_text = (brand_df["text"].astype(str).str.strip() == "").sum()
    bad_text = int(null_text) + int(empty_text)
    text_completeness = 1.0 - (bad_text / total)

    # Sub-metric 2: reply chain integrity
    reply_rows = brand_df[brand_df["in_response_to_tweet_id"].notna()]
    n_replies = len(reply_rows)
    if n_replies == 0:
        chain_integrity = 1.0  # no replies claimed, nothing is broken
    else:
        parent_ids = reply_rows["in_response_to_tweet_id"].astype(int)
        broken = (~parent_ids.isin(tweet_id_set)).sum()
        chain_integrity = 1.0 - (float(broken) / float(n_replies))

    completeness = 0.5 * text_completeness + 0.5 * chain_integrity
    logger.debug(
        "Brand '%s': text_completeness=%.4f, chain_integrity=%.4f, composite=%.4f",
        brand, text_completeness, chain_integrity, completeness,
    )
    return completeness


def compute_issue_diversity(
    brand: str,
    customer_messages: list[str],
    config: BrandSelectionConfig,
) -> tuple[float, int]:
    """
    Metric E — Issue / Message Diversity.

    Algorithm (fixed seed for reproducibility):
      1. Take up to config.sample_size customer messages.
      2. Normalize: lowercase, remove @mentions and URLs.
      3. TF-IDF vectorization (max_features, min_df from config).
      4. KMeans clustering into config.n_clusters clusters.
      5. Compute Shannon entropy of cluster size distribution.
      6. Normalize entropy to [0,1] by dividing by log(n_clusters).

    Rationale:
      High entropy = cluster sizes are roughly equal = messages cover many
      distinct topics. Low entropy = a few dominant clusters (repetitive issues).

    Note on imperfection:
      TF-IDF + KMeans is a proxy, not ground-truth topic modeling.
      Results are meaningful for relative comparison but not absolute claims.
      The same seed and parameters must be used across all brands.

    Args:
      brand:             Brand name (for logging).
      customer_messages: List of customer message strings to analyze.
      config:            BrandSelectionConfig with fixed seed + parameters.

    Returns:
      Tuple of (diversity_score in [0,1], actual_sample_size_used).
    """
    if not customer_messages:
        logger.warning("Brand '%s': no customer messages for diversity analysis", brand)
        return 0.0, 0

    # Sample
    rng = np.random.RandomState(config.random_seed)
    msgs = list(customer_messages)
    if len(msgs) > config.sample_size:
        indices = rng.choice(len(msgs), size=config.sample_size, replace=False)
        msgs = [msgs[i] for i in indices]

    sample_size_used = len(msgs)

    # Normalize text
    cleaned = [_clean_text_for_diversity(m) for m in msgs]
    cleaned = [m for m in cleaned if len(m.strip()) >= 3]

    if len(cleaned) < config.n_clusters:
        logger.warning(
            "Brand '%s': only %d valid messages after cleaning (need >= %d clusters). "
            "Returning diversity=0.0",
            brand, len(cleaned), config.n_clusters,
        )
        return 0.0, sample_size_used

    # TF-IDF
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vectorizer = TfidfVectorizer(
            max_features=config.tfidf_max_features,
            min_df=min(config.tfidf_min_df, max(1, len(cleaned) // 50)),
            sublinear_tf=True,
        )
        try:
            tfidf_matrix = vectorizer.fit_transform(cleaned)
        except ValueError as e:
            logger.warning("Brand '%s': TF-IDF failed: %s — diversity=0.0", brand, e)
            return 0.0, sample_size_used

    if tfidf_matrix.shape[1] == 0:
        logger.warning("Brand '%s': TF-IDF produced empty vocabulary — diversity=0.0", brand)
        return 0.0, sample_size_used

    # KMeans
    n_clusters = min(config.n_clusters, len(cleaned))
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=config.random_seed,
        n_init=10,
        max_iter=100,
    )
    labels = kmeans.fit_predict(tfidf_matrix)

    # Shannon entropy of cluster distribution
    cluster_sizes = np.bincount(labels, minlength=n_clusters)
    cluster_probs = cluster_sizes / cluster_sizes.sum()
    # Filter zero-probability clusters
    cluster_probs = cluster_probs[cluster_probs > 0]
    entropy = -np.sum(cluster_probs * np.log(cluster_probs))

    # Normalize by maximum possible entropy = log(n_clusters)
    max_entropy = math.log(n_clusters)
    diversity_score = float(entropy / max_entropy) if max_entropy > 0 else 0.0

    logger.debug(
        "Brand '%s': diversity=%.4f (entropy=%.4f, max=%.4f, sample=%d, clusters=%d)",
        brand, diversity_score, entropy, max_entropy, sample_size_used, n_clusters,
    )
    return diversity_score, sample_size_used


# ---------------------------------------------------------------------------
# Normalization and scoring
# ---------------------------------------------------------------------------
def normalize_metrics(metrics_list: list[BrandMetrics]) -> list[BrandMetrics]:
    """
    Apply min-max normalization to all five raw metrics across the candidate set.

    Normalization is relative to the evaluated candidate set, NOT to all 108 brands.
    This is documented in the report and limitations section.

    Modifies metrics in-place and returns the same list.
    """
    keys = [
        ("raw_interaction_volume", "norm_interaction_volume"),
        ("raw_response_coverage", "norm_response_coverage"),
        ("raw_reconstructability", "norm_reconstructability"),
        ("raw_completeness", "norm_completeness"),
        ("raw_diversity", "norm_diversity"),
    ]

    for raw_attr, norm_attr in keys:
        raw_values = [getattr(m, raw_attr) for m in metrics_list]
        norm_values = _minmax_normalize(raw_values)
        for m, nv in zip(metrics_list, norm_values):
            setattr(m, norm_attr, round(nv, 6))

    return metrics_list


def compute_weighted_scores(
    metrics_list: list[BrandMetrics],
    weights: dict[str, float],
) -> list[BrandMetrics]:
    """
    Compute final weighted score for each brand.

    Formula:
      final_score =
        weights["interaction_volume"]             × norm_interaction_volume
      + weights["response_coverage"]              × norm_response_coverage
      + weights["conversation_reconstructability"]× norm_reconstructability
      + weights["data_completeness"]              × norm_completeness
      + weights["issue_diversity"]                × norm_diversity

    Also saves individual weighted contributions for transparency.
    Sets rank based on descending final_score.

    Modifies in-place, returns the same list sorted by rank.
    """
    if not metrics_list:
        return []

    required_keys = {
        "interaction_volume",
        "response_coverage",
        "conversation_reconstructability",
        "data_completeness",
        "issue_diversity",
    }
    missing = required_keys - set(weights.keys())
    if missing:
        raise ValueError(f"Missing required weights: {missing}")

    for k in required_keys:
        if weights[k] < 0:
            raise ValueError(f"Weight '{k}' cannot be negative: {weights[k]}")

    total_weight = sum(weights[k] for k in required_keys)
    if abs(total_weight - 1.0) > 1e-6:
        raise ValueError(
            f"Weights must sum to 1.0 within floating-point tolerance. Actual sum: {total_weight}"
        )

    w_vol   = weights["interaction_volume"]
    w_cov   = weights["response_coverage"]
    w_rec   = weights["conversation_reconstructability"]
    w_comp  = weights["data_completeness"]
    w_div   = weights["issue_diversity"]

    for m in metrics_list:
        m.contrib_interaction_volume  = round(w_vol  * m.norm_interaction_volume, 6)
        m.contrib_response_coverage   = round(w_cov  * m.norm_response_coverage, 6)
        m.contrib_reconstructability  = round(w_rec  * m.norm_reconstructability, 6)
        m.contrib_completeness        = round(w_comp * m.norm_completeness, 6)
        m.contrib_diversity           = round(w_div  * m.norm_diversity, 6)
        m.final_score = round(
            m.contrib_interaction_volume
            + m.contrib_response_coverage
            + m.contrib_reconstructability
            + m.contrib_completeness
            + m.contrib_diversity,
            6,
        )

    # Sort by descending final_score and assign ranks
    metrics_list.sort(key=lambda x: x.final_score, reverse=True)
    for i, m in enumerate(metrics_list):
        m.rank = i + 1

    return metrics_list


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------
def sensitivity_analysis(
    metrics_list: list[BrandMetrics],
) -> list[SensitivityResult]:
    """
    Run brand ranking under all three weight scenarios.

    Returns a list of SensitivityResult, one per scenario.
    Does NOT modify the original metrics_list normalized scores.
    """
    results = []

    for scenario_name, weights in WEIGHT_SCENARIOS.items():
        # Deep copy metric objects to avoid mutation
        import copy
        scenario_metrics = copy.deepcopy(metrics_list)

        # Recompute scores with scenario weights
        scenario_metrics = compute_weighted_scores(scenario_metrics, weights)

        rankings = [
            {"rank": m.rank, "brand": m.brand, "score": round(m.final_score, 6)}
            for m in scenario_metrics
        ]
        winner = scenario_metrics[0].brand if scenario_metrics else ""

        results.append(SensitivityResult(
            scenario_name=scenario_name,
            weights=weights,
            rankings=rankings,
            winner=winner,
        ))

        logger.info(
            "Sensitivity [%s]: winner=%s (score=%.4f)",
            scenario_name, winner, scenario_metrics[0].final_score if scenario_metrics else 0,
        )

    return results


# ---------------------------------------------------------------------------
# Qualitative sanity check
# ---------------------------------------------------------------------------
def qualitative_sanity_check(
    top_brands: list[str],
    full_df: pd.DataFrame,
    tweet_id_to_idx: dict[int, int],
    config: BrandSelectionConfig,
    n_samples: int = 5,
) -> list[QualitativeSample]:
    """
    Sample real customer→brand interaction pairs for qualitative inspection.

    For each of the top_brands, takes a deterministic random sample of
    n_samples interaction pairs (customer tweet + brand reply).

    Purpose:
      Verify that the data actually contains usable customer support exchanges.
      This is a SANITY CHECK — it cannot and should not override quantitative scores.

    Privacy note:
      Customer messages may contain personal information. We log them only
      to local files and truncate to 200 characters for the report.

    Args:
      top_brands:        Brand names to sample from (top 3 recommended).
      full_df:           Full dataset DataFrame.
      tweet_id_to_idx:   Dict mapping tweet_id → row index for fast lookup.
      config:            BrandSelectionConfig (for random seed).
      n_samples:         Interactions to sample per brand.

    Returns:
      List of QualitativeSample objects.
    """
    samples: list[QualitativeSample] = []
    rng = np.random.RandomState(config.random_seed)

    for brand in top_brands:
        brand_df = full_df[
            (full_df["author_id"] == brand)
            & (full_df["in_response_to_tweet_id"].notna())
        ]
        if brand_df.empty:
            logger.warning("No reply rows found for brand '%s' in qualitative check", brand)
            continue

        # Sample up to n_samples rows
        n = min(n_samples, len(brand_df))
        sampled = brand_df.sample(n=n, random_state=rng.randint(0, 2**31))

        for _, brand_row in sampled.iterrows():
            parent_id = int(brand_row["in_response_to_tweet_id"])
            if parent_id not in tweet_id_to_idx:
                continue
            parent_idx = tweet_id_to_idx[parent_id]
            parent_row = full_df.iloc[parent_idx]

            # Only include if parent is a customer (inbound) tweet
            if not bool(parent_row["inbound"]):
                continue

            customer_text = str(parent_row["text"])[:200]
            brand_text = str(brand_row["text"])[:200]

            samples.append(QualitativeSample(
                brand=brand,
                customer_text=customer_text,
                brand_reply=brand_text,
                customer_tweet_id=int(parent_row["tweet_id"]),
                brand_tweet_id=int(brand_row["tweet_id"]),
            ))

    logger.info("Collected %d qualitative samples across %d brands", len(samples), len(top_brands))
    return samples


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_brand_selection(
    full_df: pd.DataFrame,
    brand_candidates_df: pd.DataFrame,
    config: BrandSelectionConfig,
) -> BrandSelectionResult:
    """
    Run the complete brand selection pipeline.

    Steps:
      1. Select top-N candidates from brand_candidates_df.
      2. Build tweet_id lookup index (one-time O(n) build).
      3. For each candidate brand, compute all 5 metrics.
      4. Normalize metrics across candidates.
      5. Compute weighted scores under default scenario.
      6. Run sensitivity analysis across all scenarios.
      7. Run qualitative sanity check on top-3.

    Args:
      full_df:              Complete dataset DataFrame (2.8M rows).
      brand_candidates_df:  Phase 1 brand_candidates.csv as DataFrame.
      config:               BrandSelectionConfig instance.

    Returns:
      BrandSelectionResult with all findings.
    """
    config.validate()
    logger.info("=" * 60)
    logger.info("Phase 2 — Brand Selection Pipeline")
    logger.info("Config: top_n=%d, sample_size=%d, seed=%d, scenario=%s",
                config.top_n, config.sample_size, config.random_seed, config.scenario)
    logger.info("=" * 60)

    # Step 1: Select top-N candidates
    top_candidates = brand_candidates_df.nlargest(config.top_n, "usable_interaction_count")
    candidate_brands = top_candidates["author_id"].tolist()
    logger.info("Top-%d candidates: %s", config.top_n, candidate_brands)

    # Step 2: Build tweet_id lookup index (one-time O(n) — avoids repeated scans)
    logger.info("Building tweet_id lookup index (%d rows)...", len(full_df))
    tweet_id_array = full_df["tweet_id"].values
    tweet_id_set = set(tweet_id_array.tolist())
    # For qualitative check we need index positions
    tweet_id_to_idx = {int(tid): idx for idx, tid in enumerate(tweet_id_array)}
    logger.info("Index built: %d unique tweet IDs", len(tweet_id_set))

    # Step 3: Compute metrics per brand
    metrics_list: list[BrandMetrics] = []

    for brand in candidate_brands:
        logger.info("--- Computing metrics for: %s ---", brand)
        brand_df = full_df[full_df["author_id"] == brand]

        # A. Interaction Volume (from Phase 1 results)
        vol = compute_interaction_volume(brand, brand_candidates_df)
        logger.info("  A. Interaction volume: %.0f", vol)

        # B. Response Coverage
        cov = compute_response_coverage(brand, brand_df, tweet_id_set)
        logger.info("  B. Response coverage: %.4f", cov)

        # C. Reconstructability
        rec = compute_reconstructability(brand, brand_df, full_df, tweet_id_set)
        logger.info("  C. Reconstructability: %.4f", rec)

        # D. Data Completeness
        comp = compute_data_completeness(brand, brand_df, tweet_id_set)
        logger.info("  D. Data completeness: %.4f", comp)

        # E. Diversity — need customer messages directed at this brand
        # Customer messages are parent tweets of brand's reply tweets
        reply_rows = brand_df[brand_df["in_response_to_tweet_id"].notna()]
        parent_ids = set(reply_rows["in_response_to_tweet_id"].dropna().astype(int).tolist())

        # Get customer messages (inbound parent tweets)
        customer_msgs_df = full_df[
            (full_df["tweet_id"].isin(parent_ids))
            & (full_df["inbound"] == True)  # noqa: E712
        ]["text"].dropna()
        customer_msgs = customer_msgs_df.tolist()

        div, sample_used = compute_issue_diversity(brand, customer_msgs, config)
        logger.info("  E. Diversity: %.4f (sample_size=%d)", div, sample_used)

        metrics = BrandMetrics(
            brand=brand,
            raw_interaction_volume=vol,
            raw_response_coverage=cov,
            raw_reconstructability=rec,
            raw_completeness=comp,
            raw_diversity=div,
            sample_size_used=sample_used,
        )
        metrics_list.append(metrics)

    # Step 4: Normalize
    logger.info("Normalizing metrics across %d candidates...", len(metrics_list))
    metrics_list = normalize_metrics(metrics_list)

    # Step 5: Weighted scores (default scenario)
    logger.info("Computing weighted scores [scenario=%s]...", config.scenario)
    metrics_list = compute_weighted_scores(metrics_list, config.weights)

    # Step 6: Sensitivity analysis
    logger.info("Running sensitivity analysis across %d scenarios...", len(WEIGHT_SCENARIOS))
    sensitivity = sensitivity_analysis(metrics_list)

    # Step 7: Qualitative sanity check (top 3)
    top3 = [m.brand for m in metrics_list[:3]]
    logger.info("Running qualitative sanity check for top-3: %s", top3)
    qual_samples = qualitative_sanity_check(top3, full_df, tweet_id_to_idx, config, n_samples=5)

    # Final selection
    selected = metrics_list[0].brand
    winner_counts = {s.winner: 0 for s in sensitivity}
    for s in sensitivity:
        winner_counts[s.winner] = winner_counts.get(s.winner, 0) + 1
    dominant_winner = max(winner_counts, key=winner_counts.get)

    rationale = (
        f"'{selected}' ranked #1 in the default (balanced) scenario. "
        f"In sensitivity analysis, '{dominant_winner}' won "
        f"{winner_counts[dominant_winner]}/{len(sensitivity)} scenarios."
    )
    if selected != dominant_winner:
        rationale += (
            f" NOTE: Default winner ({selected}) differs from sensitivity "
            f"dominant winner ({dominant_winner}). Selection uses default weights."
        )

    logger.info("=" * 60)
    logger.info("SELECTED BRAND: %s", selected)
    logger.info("Rationale: %s", rationale)
    logger.info("=" * 60)

    return BrandSelectionResult(
        config=config,
        metrics=metrics_list,
        sensitivity=sensitivity,
        qualitative_samples=qual_samples,
        selected_brand=selected,
        selection_rationale=rationale,
    )
