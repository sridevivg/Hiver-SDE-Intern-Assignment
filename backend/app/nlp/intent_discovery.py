"""
SupportGraph AI — Intent Discovery Module (Phase 4)

Discovers candidate customer intent clusters from historical AppleSupport customer opening messages.

Pipeline:
1. Load & validate Phase 3 outputs (conversation_messages.parquet, conversation_quality_summary.csv, selected_brand.json)
2. Filter for HIGH quality conversations, customer role, depth == 0 (opening messages)
3. Normalize customer text conservatively (preserving product names and error codes)
4. Deterministically sample candidate messages (default: 20,000)
5. Generate semantic sentence embeddings locally (sentence-transformers/all-MiniLM-L6-v2)
6. Apply PCA dimensionality reduction (default: 50 components)
7. Run clustering experiments across K in [k_min, k_max] (default: 4 to 15) using MiniBatchKMeans
8. Compute multi-metric evaluation (Silhouette, Calinski-Harabasz, Davies-Bouldin, cluster distributions)
9. Select candidate cluster count via documented multi-metric balanced scoring rule
10. Generate cluster interpretation (centroid-nearest representative examples, TF-IDF top terms)
11. Generate heuristic proposed labels while maintaining final_intent_label = null and review_status = 'pending'
12. Save discovery corpus, clustering experiments, review CSV, and provisional JSON taxonomy.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

try:
    from app.core.logging import get_logger
    from app.nlp.embeddings import SentenceEmbeddingGenerator
    from app.nlp.text_preprocessing import is_usable_text, normalize_text
except ModuleNotFoundError:
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]
    from backend.app.nlp.embeddings import SentenceEmbeddingGenerator  # type: ignore[no-redef]
    from backend.app.nlp.text_preprocessing import is_usable_text, normalize_text  # type: ignore[no-redef]

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data Transfer Objects / Results
# ---------------------------------------------------------------------------
@dataclass
class CorpusBuildResult:
    """Statistics and DataFrame from corpus construction."""
    df: pd.DataFrame
    total_messages_loaded: int
    total_conversations_in_summary: int
    high_quality_conversations_count: int
    candidate_customer_opening_count: int
    excluded_unusable_count: int
    final_eligible_count: int


@dataclass
class ClusteringExperimentRow:
    """Clustering metrics for a specific K."""
    k: int
    silhouette_score: float
    calinski_harabasz_score: float
    davies_bouldin_score: float
    smallest_cluster_size: int
    largest_cluster_size: int
    tiny_cluster_count: int
    selection_notes: str


@dataclass
class ClusterEvidence:
    """Interpretable evidence for an individual cluster."""
    cluster_id: int
    cluster_size: int
    percentage: float
    top_terms: list[str]
    representative_examples: list[str]
    proposed_label: str
    final_intent_label: Optional[str] = None
    review_status: str = "pending"


@dataclass
class IntentDiscoveryResult:
    """Complete output of Phase 4 Intent Discovery."""
    corpus_df: pd.DataFrame
    experiments: list[ClusteringExperimentRow]
    selected_k: int
    clusters: list[ClusterEvidence]
    explained_variance_ratio: float
    metadata: dict[str, Any]


# ---------------------------------------------------------------------------
# Step 1: Input Validation and Loading
# ---------------------------------------------------------------------------
def validate_and_load_inputs(
    messages_path: Path,
    quality_path: Path,
    brand_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Validate presence and schemas of Phase 3 outputs.

    Raises:
        FileNotFoundError: If any required file is missing.
        ValueError: If required columns are absent or dataset cannot be joined.
    """
    if not brand_path.exists():
        raise FileNotFoundError(f"Selected brand file not found at: {brand_path}")
    if not messages_path.exists():
        raise FileNotFoundError(f"Conversation messages file not found at: {messages_path}")
    if not quality_path.exists():
        raise FileNotFoundError(f"Quality summary file not found at: {quality_path}")

    with open(brand_path, "r", encoding="utf-8") as f:
        brand_data = json.load(f)
    selected_brand = brand_data.get("selected_brand")
    if not selected_brand:
        raise ValueError("selected_brand.json is missing 'selected_brand' key.")

    logger.info("Validated brand file: Selected brand is '%s'", selected_brand)

    # Load messages
    df_messages = pd.read_parquet(messages_path)
    required_msg_cols = {"conversation_id", "tweet_id", "role", "depth", "text"}
    missing_msg_cols = required_msg_cols - set(df_messages.columns)
    if missing_msg_cols:
        raise ValueError(f"conversation_messages.parquet missing columns: {missing_msg_cols}")

    # Load quality summary
    df_quality = pd.read_csv(quality_path)
    required_qual_cols = {"conversation_id", "quality_status"}
    missing_qual_cols = required_qual_cols - set(df_quality.columns)
    if missing_qual_cols:
        raise ValueError(f"conversation_quality_summary.csv missing columns: {missing_qual_cols}")

    logger.info(
        "Successfully loaded %d messages and %d quality records.",
        len(df_messages),
        len(df_quality),
    )
    return df_messages, df_quality, selected_brand


# ---------------------------------------------------------------------------
# Step 2: Build Discovery Corpus
# ---------------------------------------------------------------------------
def build_discovery_corpus(
    df_messages: pd.DataFrame,
    df_quality: pd.DataFrame,
    min_text_length: int = 3,
) -> CorpusBuildResult:
    """
    Filter candidate customer opening messages from HIGH-quality conversations.
    Applies conservative normalization while preserving original text.
    """
    # 1. Identify HIGH quality conversations
    high_qual_mask = df_quality["quality_status"].astype(str).str.upper() == "HIGH"
    high_qual_conv_ids = set(df_quality.loc[high_qual_mask, "conversation_id"].dropna())

    # 2. Filter messages: in high quality conv, customer role, depth 0
    candidate_mask = (
        df_messages["conversation_id"].isin(high_qual_conv_ids)
        & (df_messages["role"] == "customer")
        & (df_messages["depth"] == 0)
    )
    candidates = df_messages.loc[candidate_mask].copy()
    candidate_count = len(candidates)

    # 3. Clean and normalize text
    candidates["original_text"] = candidates["text"].fillna("").astype(str)
    candidates["normalized_text"] = candidates["original_text"].apply(normalize_text)

    # 4. Filter unusable text (empty or whitespace only)
    usable_mask = candidates["normalized_text"].apply(
        lambda t: is_usable_text(t, min_length=min_text_length)
    )
    excluded_count = int((~usable_mask).sum())
    usable_df = candidates.loc[usable_mask].copy()

    # Select and order final columns
    final_df = usable_df[[
        "conversation_id",
        "tweet_id",
        "original_text",
        "normalized_text",
    ]].reset_index(drop=True)

    logger.info(
        "Corpus built: %d high-quality convs, %d candidate opening messages, "
        "%d excluded unusable, %d eligible messages.",
        len(high_qual_conv_ids),
        candidate_count,
        excluded_count,
        len(final_df),
    )

    return CorpusBuildResult(
        df=final_df,
        total_messages_loaded=len(df_messages),
        total_conversations_in_summary=len(df_quality),
        high_quality_conversations_count=len(high_qual_conv_ids),
        candidate_customer_opening_count=candidate_count,
        excluded_unusable_count=excluded_count,
        final_eligible_count=len(final_df),
    )


# ---------------------------------------------------------------------------
# Step 3: Dataset Sampling
# ---------------------------------------------------------------------------
def sample_corpus(
    df: pd.DataFrame,
    max_samples: int = 20000,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Deterministically sample messages if corpus size exceeds max_samples.
    """
    total_eligible = len(df)
    if total_eligible <= max_samples:
        sampled_df = df.copy().reset_index(drop=True)
        strategy = "ALL_MESSAGES"
    else:
        sampled_df = df.sample(
            n=max_samples,
            random_state=random_seed,
            replace=False,
        ).reset_index(drop=True)
        strategy = "RANDOM_REPRODUCIBLE_SAMPLE"

    metadata = {
        "total_eligible_messages": total_eligible,
        "sampled_messages": len(sampled_df),
        "max_samples_limit": max_samples,
        "random_seed": random_seed,
        "sampling_strategy": strategy,
    }
    logger.info(
        "Sampled corpus: %d / %d messages (strategy=%s, seed=%d)",
        len(sampled_df),
        total_eligible,
        strategy,
        random_seed,
    )
    return sampled_df, metadata


# ---------------------------------------------------------------------------
# Step 5: Dimensionality Reduction
# ---------------------------------------------------------------------------
def reduce_dimensions_pca(
    embeddings: np.ndarray,
    n_components: int = 50,
    random_state: int = 42,
) -> tuple[np.ndarray, PCA]:
    """
    Reduce embedding dimensionality using reproducible PCA.

    Args:
        embeddings: 2D numpy array of embeddings (N, D).
        n_components: Target dimensionality (capped at min(N, D)).
        random_state: Random state for deterministic solver.

    Returns:
        reduced_embeddings, fitted_pca_object
    """
    n_samples, n_features = embeddings.shape
    actual_components = min(n_components, n_samples - 1, n_features)
    logger.info(
        "Applying PCA: %d features -> %d components (samples=%d)...",
        n_features,
        actual_components,
        n_samples,
    )
    pca = PCA(n_components=actual_components, random_state=random_state)
    reduced = pca.fit_transform(embeddings)
    explained_variance = float(np.sum(pca.explained_variance_ratio_))
    logger.info(
        "PCA completed. Total explained variance ratio: %.4f (%.2f%%)",
        explained_variance,
        explained_variance * 100.0,
    )
    return reduced, pca


# ---------------------------------------------------------------------------
# Step 6: Clustering Experiments
# ---------------------------------------------------------------------------
def run_clustering_experiments(
    reduced_features: np.ndarray,
    k_min: int = 4,
    k_max: int = 15,
    random_state: int = 42,
    tiny_cluster_threshold_pct: float = 0.01,
) -> list[ClusteringExperimentRow]:
    """
    Run MiniBatchKMeans across range [k_min, k_max] and compute evaluation metrics.

    Metrics:
    - Silhouette Score (higher is better, [-1, 1])
    - Calinski-Harabasz Score (higher is better, >= 0)
    - Davies-Bouldin Score (lower is better, >= 0)
    - Cluster size distribution
    """
    n_samples = len(reduced_features)
    min_cluster_size_threshold = max(5, int(n_samples * tiny_cluster_threshold_pct))

    experiments: list[ClusteringExperimentRow] = []

    for k in range(k_min, k_max + 1):
        mbk = MiniBatchKMeans(
            n_clusters=k,
            random_state=random_state,
            batch_size=min(1024, n_samples),
            n_init=3,
        )
        cluster_labels = mbk.fit_predict(reduced_features)

        # Metrics computation
        sil = float(silhouette_score(reduced_features, cluster_labels, sample_size=min(5000, n_samples), random_state=random_state))
        ch = float(calinski_harabasz_score(reduced_features, cluster_labels))
        db = float(davies_bouldin_score(reduced_features, cluster_labels))

        # Size distribution
        _, counts = np.unique(cluster_labels, return_counts=True)
        smallest = int(np.min(counts))
        largest = int(np.max(counts))
        tiny_count = int(np.sum(counts < min_cluster_size_threshold))

        notes = (
            f"k={k}: Sil={sil:.4f}, CH={ch:.1f}, DB={db:.4f}, "
            f"min_size={smallest}, max_size={largest}, tiny={tiny_count}"
        )

        row = ClusteringExperimentRow(
            k=k,
            silhouette_score=sil,
            calinski_harabasz_score=ch,
            davies_bouldin_score=db,
            smallest_cluster_size=smallest,
            largest_cluster_size=largest,
            tiny_cluster_count=tiny_count,
            selection_notes=notes,
        )
        experiments.append(row)
        logger.info(notes)

    return experiments


# ---------------------------------------------------------------------------
# Step 7: Select Candidate Cluster Count
# ---------------------------------------------------------------------------
def select_candidate_k(
    experiments: list[ClusteringExperimentRow],
    target_k_range: tuple[int, int] = (6, 12),
) -> int:
    """
    Select candidate K using a documented multi-metric balanced scoring rule.

    Principle:
    - Normalizes Silhouette (higher better), Davies-Bouldin (lower better),
      and Calinski-Harabasz (higher better) onto [0, 1].
    - Composite Score:
        Score(K) = 0.45 * Silhouette_norm + 0.35 * DaviesBouldin_norm + 0.20 * CalinskiHarabasz_norm
                   - (0.25 * tiny_cluster_count)
    - Prefers values within practical taxonomy range (target_k_range, default 6-12)
      by applying a minor penalty if outside target range.
    """
    if not experiments:
        raise ValueError("No clustering experiments provided.")

    sil_vals = [e.silhouette_score for e in experiments]
    db_vals = [e.davies_bouldin_score for e in experiments]
    ch_vals = [e.calinski_harabasz_score for e in experiments]

    sil_min, sil_max = min(sil_vals), max(sil_vals)
    db_min, db_max = min(db_vals), max(db_vals)
    ch_min, ch_max = min(ch_vals), max(ch_vals)

    def normalize(val: float, low: float, high: float) -> float:
        if abs(high - low) < 1e-9:
            return 0.5
        return (val - low) / (high - low)

    best_k = experiments[0].k
    best_score = -float("inf")

    for exp in experiments:
        sil_norm = normalize(exp.silhouette_score, sil_min, sil_max)
        # Davies-Bouldin: lower is better -> invert
        db_norm = 1.0 - normalize(exp.davies_bouldin_score, db_min, db_max)
        ch_norm = normalize(exp.calinski_harabasz_score, ch_min, ch_max)

        # Composite metric
        base_score = 0.45 * sil_norm + 0.35 * db_norm + 0.20 * ch_norm

        # Penalize tiny degenerate clusters
        penalty = 0.25 * exp.tiny_cluster_count

        # Small penalty if outside target practical taxonomy range (6-12)
        if not (target_k_range[0] <= exp.k <= target_k_range[1]):
            penalty += 0.15

        composite = base_score - penalty

        logger.debug(
            "k=%d composite_score=%.4f (sil_norm=%.3f, db_norm=%.3f, ch_norm=%.3f, penalty=%.3f)",
            exp.k,
            composite,
            sil_norm,
            db_norm,
            ch_norm,
            penalty,
        )

        if composite > best_score:
            best_score = composite
            best_k = exp.k

    logger.info("Selected candidate K=%d with best composite score=%.4f", best_k, best_score)
    return best_k


# ---------------------------------------------------------------------------
# Step 8: Cluster Interpretation (TF-IDF + Centroid Proximity)
# ---------------------------------------------------------------------------
DOMAIN_PATTERNS: list[tuple[str, list[str]]] = [
    ("battery_power_issue", ["battery", "drain", "draining", "charge", "charging", "charger", "dies", "die", "power", "percentage"]),
    ("software_update_issue", ["update", "updated", "updating", "ios", "install", "download", "version", "upgrade", "11", "software"]),
    ("account_access_issue", ["account", "appleid", "apple id", "password", "login", "locked", "verification", "verify", "icloud", "id"]),
    ("display_hardware_issue", ["screen", "display", "touch", "cracked", "black", "speaker", "audio", "sound", "button", "camera"]),
    ("network_connectivity_issue", ["wifi", "wi-fi", "bluetooth", "service", "signal", "cellular", "carrier", "network", "connect", "internet"]),
    ("app_store_billing_issue", ["app", "apps", "store", "itunes", "purchase", "purchased", "subscription", "billed", "payment", "card", "refund"]),
    ("keyboard_typing_glitch", ["keyboard", "typing", "autocorrect", "type", "letter", "glitch", "predictive", "i"]),
    ("order_reservation_inquiry", ["order", "shipping", "delivery", "delivered", "reservation", "reserved", "pickup", "store", "appointment", "repair"]),
]


def propose_cluster_label(top_terms: list[str], cluster_id: int) -> str:
    """
    Generate an evidence-based proposed intent label using deterministic keyword patterns.
    Does NOT use external LLMs. Fallback is 'cluster_{id}_{top_term}_unreviewed'.
    """
    terms_set = set(t.lower() for t in top_terms)

    best_match_label = ""
    max_matches = 0

    for label, keywords in DOMAIN_PATTERNS:
        match_count = sum(1 for kw in keywords if kw in terms_set or any(kw in t for t in terms_set))
        if match_count > max_matches:
            max_matches = match_count
            best_match_label = label

    if max_matches >= 2:
        return best_match_label
    elif max_matches == 1:
        return f"possible_{best_match_label}"
    else:
        top_term = top_terms[0].replace(" ", "_") if top_terms else "generic"
        return f"cluster_{cluster_id}_{top_term}_unreviewed"


def extract_cluster_evidence(
    texts: list[str],
    features: np.ndarray,
    cluster_labels: np.ndarray,
    k: int,
    centroids: np.ndarray,
    top_n_terms: int = 10,
    top_n_examples: int = 4,
) -> list[ClusterEvidence]:
    """
    Compute cluster sizes, centroid-nearest representative examples,
    top TF-IDF terms, and heuristic proposed labels.
    """
    n_total = len(texts)

    # TF-IDF vectorization for term extraction
    tfidf = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[A-Za-z][A-Za-z0-9_]{2,}\b",
    )
    try:
        tfidf_matrix = tfidf.fit_transform(texts)
        feature_names = np.array(tfidf.get_feature_names_out())
    except Exception as exc:
        logger.warning("TF-IDF extraction failed: %s", exc)
        tfidf_matrix = None
        feature_names = np.array([])

    filtered_stopwords = {"url", "user", "applesupport", "apple", "thanks", "hello", "help", "please", "hi"}

    clusters: list[ClusterEvidence] = []

    for cluster_id in range(k):
        member_indices = np.where(cluster_labels == cluster_id)[0]
        size = len(member_indices)
        pct = (size / n_total) * 100.0 if n_total > 0 else 0.0

        if size == 0:
            clusters.append(ClusterEvidence(
                cluster_id=cluster_id,
                cluster_size=0,
                percentage=0.0,
                top_terms=[],
                representative_examples=[],
                proposed_label=f"cluster_{cluster_id}_empty",
            ))
            continue

        # 1. Representative examples closest to centroid
        cluster_features = features[member_indices]
        centroid = centroids[cluster_id]
        distances = np.linalg.norm(cluster_features - centroid, axis=1)
        closest_order = np.argsort(distances)
        rep_indices = member_indices[closest_order[:top_n_examples]]
        rep_examples = [texts[idx] for idx in rep_indices]

        # 2. Top TF-IDF terms
        top_terms: list[str] = []
        if tfidf_matrix is not None and len(feature_names) > 0:
            cluster_tfidf_mean = np.asarray(tfidf_matrix[member_indices].mean(axis=0)).flatten()
            top_term_indices = np.argsort(cluster_tfidf_mean)[::-1]

            for term_idx in top_term_indices:
                term = str(feature_names[term_idx]).lower()
                if term not in filtered_stopwords and not any(term == s for s in filtered_stopwords):
                    top_terms.append(term)
                if len(top_terms) >= top_n_terms:
                    break

        # 3. Proposed heuristic label
        proposed = propose_cluster_label(top_terms, cluster_id)

        clusters.append(ClusterEvidence(
            cluster_id=cluster_id,
            cluster_size=size,
            percentage=round(pct, 2),
            top_terms=top_terms,
            representative_examples=rep_examples,
            proposed_label=proposed,
            final_intent_label=None,
            review_status="pending",
        ))

    return clusters


# ---------------------------------------------------------------------------
# Step 12: Save Discovery Outputs
# ---------------------------------------------------------------------------
def save_discovery_artifacts(
    result: IntentDiscoveryResult,
    interim_dir: Path,
    selected_brand: str,
    embedding_model_name: str,
    clustering_algorithm: str = "MiniBatchKMeans",
) -> dict[str, Path]:
    """
    Save all Phase 4 artifacts:
    - data/interim/intent_discovery_corpus.csv
    - data/interim/intent_clustering_experiments.csv
    - data/interim/intent_cluster_review.csv
    - data/interim/provisional_intent_taxonomy.json
    """
    interim_dir.mkdir(parents=True, exist_ok=True)

    import csv
    # 1. Corpus CSV
    corpus_csv_path = interim_dir / "intent_discovery_corpus.csv"
    result.corpus_df.to_csv(corpus_csv_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
    logger.info("Saved intent discovery corpus to: %s", corpus_csv_path)

    # 2. Experiments CSV
    exp_csv_path = interim_dir / "intent_clustering_experiments.csv"
    df_exp = pd.DataFrame([asdict(e) for e in result.experiments])
    df_exp.to_csv(exp_csv_path, index=False)
    logger.info("Saved clustering experiments to: %s", exp_csv_path)

    # 3. Cluster Review CSV
    review_csv_path = interim_dir / "intent_cluster_review.csv"
    review_rows = []
    for c in result.clusters:
        review_rows.append({
            "cluster_id": c.cluster_id,
            "cluster_size": c.cluster_size,
            "percentage": c.percentage,
            "top_terms": "; ".join(c.top_terms),
            "representative_examples": " || ".join(c.representative_examples),
            "proposed_label": c.proposed_label,
            "final_intent_label": c.final_intent_label if c.final_intent_label is not None else "",
            "review_status": c.review_status,
        })
    df_review = pd.DataFrame(review_rows)
    df_review.to_csv(review_csv_path, index=False)
    logger.info("Saved cluster review sheet to: %s", review_csv_path)

    # 4. Provisional Taxonomy JSON
    taxonomy_json_path = interim_dir / "provisional_intent_taxonomy.json"
    taxonomy_payload = {
        "status": "PROVISIONAL_HUMAN_REVIEW_REQUIRED",
        "selected_brand": selected_brand,
        "discovery_method": {
            "embedding_model": embedding_model_name,
            "dimension_reduction": "PCA",
            "pca_components": result.metadata.get("pca_components", 50),
            "explained_variance_ratio": round(result.explained_variance_ratio, 4),
            "clustering_algorithm": clustering_algorithm,
            "random_seed": result.metadata.get("random_seed", 42),
        },
        "selected_cluster_count": result.selected_k,
        "cluster_review_required": True,
        "clusters": [
            {
                "cluster_id": c.cluster_id,
                "cluster_size": c.cluster_size,
                "percentage": c.percentage,
                "top_terms": c.top_terms,
                "representative_examples": c.representative_examples,
                "proposed_label": c.proposed_label,
                "final_intent_label": c.final_intent_label,
                "review_status": c.review_status,
            }
            for c in result.clusters
        ],
    }
    with open(taxonomy_json_path, "w", encoding="utf-8") as f:
        json.dump(taxonomy_payload, f, indent=2)
    logger.info("Saved provisional intent taxonomy to: %s", taxonomy_json_path)

    return {
        "corpus_csv": corpus_csv_path,
        "experiments_csv": exp_csv_path,
        "review_csv": review_csv_path,
        "taxonomy_json": taxonomy_json_path,
    }


# ---------------------------------------------------------------------------
# High-level Orchestrator
# ---------------------------------------------------------------------------
def run_intent_discovery(
    messages_path: Path,
    quality_path: Path,
    brand_path: Path,
    output_dir: Path,
    max_samples: int = 20000,
    k_min: int = 4,
    k_max: int = 15,
    random_seed: int = 42,
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
    pca_components: int = 50,
    embedding_generator: Optional[SentenceEmbeddingGenerator] = None,
) -> IntentDiscoveryResult:
    """
    Execute full Phase 4 Intent Discovery Pipeline.
    """
    logger.info("Starting Phase 4 Intent Discovery pipeline...")

    # 1. Validate & load inputs
    df_messages, df_quality, selected_brand = validate_and_load_inputs(
        messages_path, quality_path, brand_path
    )

    # 2. Build corpus
    corpus_result = build_discovery_corpus(df_messages, df_quality)
    if corpus_result.final_eligible_count == 0:
        raise ValueError("Corpus is empty after filtering for high-quality customer opening messages.")

    # 3. Sample corpus
    sampled_df, sample_meta = sample_corpus(
        corpus_result.df,
        max_samples=max_samples,
        random_seed=random_seed,
    )

    # 4. Generate embeddings
    if embedding_generator is None:
        embedding_generator = SentenceEmbeddingGenerator(
            model_name=embedding_model_name,
            normalize_embeddings=True,
        )
    logger.info(
        "Generating embeddings for %d messages (batch_size=%d)...",
        len(sampled_df),
        batch_size,
    )
    embeddings = embedding_generator.encode(
        sampled_df["normalized_text"].tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
    )

    # 5. Dimensionality reduction (PCA)
    reduced_features, pca_model = reduce_dimensions_pca(
        embeddings,
        n_components=pca_components,
        random_state=random_seed,
    )
    explained_var = float(np.sum(pca_model.explained_variance_ratio_))

    # 6. Run clustering experiments across [k_min, k_max]
    logger.info("Running clustering experiments for K=%d to %d...", k_min, k_max)
    experiments = run_clustering_experiments(
        reduced_features,
        k_min=k_min,
        k_max=k_max,
        random_state=random_seed,
    )

    # 7. Select candidate cluster count
    selected_k = select_candidate_k(experiments)
    logger.info("Optimal candidate cluster count selected: K=%d", selected_k)

    # 8. Fit selected model
    final_mbk = MiniBatchKMeans(
        n_clusters=selected_k,
        random_state=random_seed,
        batch_size=min(1024, len(reduced_features)),
        n_init=5,
    )
    final_labels = final_mbk.fit_predict(reduced_features)
    centroids = final_mbk.cluster_centers_

    # Assign cluster_id to corpus
    sampled_df["cluster_id"] = final_labels

    # 9. Extract cluster evidence
    clusters = extract_cluster_evidence(
        texts=sampled_df["normalized_text"].tolist(),
        features=reduced_features,
        cluster_labels=final_labels,
        k=selected_k,
        centroids=centroids,
    )

    # Compile result
    result = IntentDiscoveryResult(
        corpus_df=sampled_df,
        experiments=experiments,
        selected_k=selected_k,
        clusters=clusters,
        explained_variance_ratio=explained_var,
        metadata={
            "selected_brand": selected_brand,
            "max_samples": max_samples,
            "random_seed": random_seed,
            "embedding_model": embedding_model_name,
            "pca_components": pca_components,
            "total_high_quality_convs": corpus_result.high_quality_conversations_count,
            "candidate_customer_opening_count": corpus_result.candidate_customer_opening_count,
            "excluded_unusable_count": corpus_result.excluded_unusable_count,
            "eligible_corpus_size": corpus_result.final_eligible_count,
            "sampled_corpus_size": len(sampled_df),
        },
    )

    # 10. Save artifacts
    save_discovery_artifacts(
        result=result,
        interim_dir=output_dir,
        selected_brand=selected_brand,
        embedding_model_name=embedding_model_name,
    )

    logger.info("Phase 4 Intent Discovery completed successfully.")
    return result
