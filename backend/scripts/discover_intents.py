"""
SupportGraph AI — Discover Intents CLI Script (Phase 4)

Executes Phase 4 Data-Driven Intent Discovery on customer opening messages:
1. Filters high-quality customer opening messages
2. Encodes messages with local sentence-transformers
3. Reduces dimensionality via PCA
4. Runs clustering experiments from k_min to k_max
5. Selects candidate K using multi-metric balance
6. Generates interpretable cluster evidence and heuristic proposed labels
7. Saves all discovery artifacts and figures

Usage:
    python -m backend.scripts.discover_intents \
        --max-samples 20000 \
        --k-min 4 \
        --k-max 15 \
        --random-seed 42
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add repo root to path if needed
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from app.core.logging import configure_logging
    from app.nlp.intent_discovery import IntentDiscoveryResult, run_intent_discovery
except ModuleNotFoundError:
    from backend.app.core.logging import configure_logging  # type: ignore[no-redef]
    from backend.app.nlp.intent_discovery import IntentDiscoveryResult, run_intent_discovery  # type: ignore[no-redef]

logger = logging.getLogger("discover_intents")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 4 Data-Driven Intent Discovery for SupportGraph AI"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=20000,
        help="Maximum customer opening messages to sample (default: 20000)",
    )
    parser.add_argument(
        "--k-min",
        type=int,
        default=4,
        help="Minimum number of clusters to test (default: 4)",
    )
    parser.add_argument(
        "--k-max",
        type=int,
        default=15,
        help="Maximum number of clusters to test (default: 15)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling and clustering (default: 42)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HuggingFace model ID for local sentence-transformers (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding generation (default: 64)",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=50,
        help="Number of PCA components for dimensionality reduction (default: 50)",
    )
    parser.add_argument(
        "--messages-path",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "conversation_messages.parquet",
        help="Path to conversation_messages.parquet",
    )
    parser.add_argument(
        "--quality-path",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "conversation_quality_summary.csv",
        help="Path to conversation_quality_summary.csv",
    )
    parser.add_argument(
        "--brand-path",
        type=Path,
        default=REPO_ROOT / "data" / "interim" / "selected_brand.json",
        help="Path to selected_brand.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "interim",
        help="Directory to save discovery artifacts",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "figures",
        help="Directory to save generated figures",
    )
    return parser.parse_args()


def generate_figures(result: IntentDiscoveryResult, figures_dir: Path) -> None:
    """
    Generate and save Phase 4 visualization figures:
    1. intent_clustering_metrics.png
    2. intent_cluster_sizes.png
    3. intent_embedding_projection.png (exploratory 2D PCA)
    """
    figures_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Figure 1: Clustering Experiment Metrics across K
    # -----------------------------------------------------------------------
    exp_df = pd.DataFrame([e.__dict__ for e in result.experiments])
    k_vals = exp_df["k"].values
    sil_vals = exp_df["silhouette_score"].values
    ch_vals = exp_df["calinski_harabasz_score"].values
    db_vals = exp_df["davies_bouldin_score"].values

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Phase 4 — Clustering Evaluation Across K (Selected K={result.selected_k})",
        fontsize=14,
        fontweight="bold",
    )

    # Silhouette
    axes[0].plot(k_vals, sil_vals, marker="o", color="#2563EB", linewidth=2)
    axes[0].axvline(result.selected_k, color="#DC2626", linestyle="--", label=f"Selected K={result.selected_k}")
    axes[0].set_title("Silhouette Score (Higher is Better)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Number of Clusters (K)")
    axes[0].set_ylabel("Score")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend()

    # Davies-Bouldin
    axes[1].plot(k_vals, db_vals, marker="s", color="#D97706", linewidth=2)
    axes[1].axvline(result.selected_k, color="#DC2626", linestyle="--", label=f"Selected K={result.selected_k}")
    axes[1].set_title("Davies-Bouldin Index (Lower is Better)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Number of Clusters (K)")
    axes[1].set_ylabel("Score")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend()

    # Calinski-Harabasz
    axes[2].plot(k_vals, ch_vals, marker="^", color="#059669", linewidth=2)
    axes[2].axvline(result.selected_k, color="#DC2626", linestyle="--", label=f"Selected K={result.selected_k}")
    axes[2].set_title("Calinski-Harabasz Score (Higher is Better)", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Number of Clusters (K)")
    axes[2].set_ylabel("Score")
    axes[2].grid(True, linestyle=":", alpha=0.6)
    axes[2].legend()

    plt.tight_layout()
    metrics_path = figures_dir / "intent_clustering_metrics.png"
    plt.savefig(metrics_path, dpi=200)
    plt.close()
    logger.info("Saved clustering metrics plot to: %s", metrics_path)

    # -----------------------------------------------------------------------
    # Figure 2: Cluster Size Distribution
    # -----------------------------------------------------------------------
    cluster_ids = [c.cluster_id for c in result.clusters]
    cluster_sizes = [c.cluster_size for c in result.clusters]
    cluster_pcts = [c.percentage for c in result.clusters]
    proposed_labels = [c.proposed_label for c in result.clusters]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(
        range(len(cluster_ids)),
        cluster_sizes,
        color="#3B82F6",
        edgecolor="#1E3A8A",
        alpha=0.85,
    )

    # Add count and percentage labels above bars
    for i, (bar, size, pct) in enumerate(zip(bars, cluster_sizes, cluster_pcts)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (max(cluster_sizes) * 0.015),
            f"{size:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(range(len(cluster_ids)))
    ax.set_xticklabels(
        [f"C{cid}\n{lbl}" for cid, lbl in zip(cluster_ids, proposed_labels)],
        rotation=30,
        ha="right",
        fontsize=9,
    )
    ax.set_title(
        f"Phase 4 — Discovered Cluster Sizes & Percentages (K={result.selected_k})",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Cluster ID & Heuristic Proposed Label")
    ax.set_ylabel("Message Count")
    ax.set_ylim(0, max(cluster_sizes) * 1.18)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    sizes_path = figures_dir / "intent_cluster_sizes.png"
    plt.savefig(sizes_path, dpi=200)
    plt.close()
    logger.info("Saved cluster size distribution plot to: %s", sizes_path)

    # -----------------------------------------------------------------------
    # Figure 3: Exploratory 2D Projection (PCA)
    # -----------------------------------------------------------------------
    try:
        from sklearn.decomposition import PCA
        sample_size = min(3000, len(result.corpus_df))
        sub_df = result.corpus_df.sample(n=sample_size, random_state=42)
        # Load embeddings or fit 2D PCA from text tfidf or corpus subset
        from sklearn.feature_extraction.text import TfidfVectorizer
        sub_tfidf = TfidfVectorizer(max_features=1000, stop_words="english").fit_transform(sub_df["normalized_text"])
        pca2 = PCA(n_components=2, random_state=42)
        coords = pca2.fit_transform(sub_tfidf.toarray())

        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=sub_df["cluster_id"].values,
            cmap="tab10",
            alpha=0.6,
            s=20,
        )
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Cluster ID")
        ax.set_title(
            "Phase 4 — Exploratory 2D Projection of Sampled Customer Messages\n"
            "(Exploratory visualization only — NOT proof of cluster validity)",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.grid(True, linestyle=":", alpha=0.4)

        plt.tight_layout()
        proj_path = figures_dir / "intent_embedding_projection.png"
        plt.savefig(proj_path, dpi=200)
        plt.close()
        logger.info("Saved exploratory projection plot to: %s", proj_path)
    except Exception as exc:
        logger.warning("Could not generate 2D projection figure: %s", exc)


def main() -> None:
    configure_logging()
    args = parse_args()

    logger.info("=" * 70)
    logger.info("SUPPORTGRAPH AI — PHASE 4: DATA-DRIVEN INTENT DISCOVERY")
    logger.info("=" * 70)
    logger.info("Messages path:       %s", args.messages_path)
    logger.info("Quality path:        %s", args.quality_path)
    logger.info("Brand path:          %s", args.brand_path)
    logger.info("Output directory:    %s", args.output_dir)
    logger.info("Max samples:         %d", args.max_samples)
    logger.info("K range:             [%d, %d]", args.k_min, args.k_max)
    logger.info("Random seed:         %d", args.random_seed)
    logger.info("Embedding model:     %s", args.embedding_model)
    logger.info("PCA components:      %d", args.pca_components)
    logger.info("Batch size:          %d", args.batch_size)

    result = run_intent_discovery(
        messages_path=args.messages_path,
        quality_path=args.quality_path,
        brand_path=args.brand_path,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        k_min=args.k_min,
        k_max=args.k_max,
        random_seed=args.random_seed,
        embedding_model_name=args.embedding_model,
        batch_size=args.batch_size,
        pca_components=args.pca_components,
    )

    logger.info("Generating visualization figures...")
    generate_figures(result, args.figures_dir)

    # Print clean summary table
    print("\n" + "=" * 70)
    print("PHASE 4 DISCOVERY COMPLETE — SUMMARY REPORT")
    print("=" * 70)
    print(f"Eligible High-Quality Customer Opening Messages: {result.metadata['eligible_corpus_size']:,}")
    print(f"Sampled Messages for Discovery:                 {result.metadata['sampled_corpus_size']:,}")
    print(f"Embedding Model:                                 {result.metadata['embedding_model']}")
    print(f"PCA Dimensionality:                              {result.metadata['pca_components']} components ({result.explained_variance_ratio * 100:.1f}% explained variance)")
    print(f"Selected Candidate Cluster Count:                K = {result.selected_k}")
    print("-" * 70)
    print("CLUSTERING EXPERIMENTS ACROSS TESTED K:")
    print(f"{'K':<4} {'Silhouette':<12} {'Davies-Bouldin':<15} {'Calinski-Harabasz':<18} {'Smallest':<10} {'Largest':<10} {'Tiny Count'}")
    for exp in result.experiments:
        marker = " <-- SELECTED" if exp.k == result.selected_k else ""
        print(f"{exp.k:<4} {exp.silhouette_score:<12.4f} {exp.davies_bouldin_score:<15.4f} {exp.calinski_harabasz_score:<18.1f} {exp.smallest_cluster_size:<10} {exp.largest_cluster_size:<10} {exp.tiny_cluster_count}{marker}")

    print("-" * 70)
    print("DISCOVERED CANDIDATE CLUSTERS (PROVISIONAL — HUMAN REVIEW REQUIRED):")
    for c in result.clusters:
        top_kws = ", ".join(c.top_terms[:5])
        print(f"\n[Cluster {c.cluster_id}] {c.proposed_label.upper()}")
        print(f"  Size: {c.cluster_size:,} ({c.percentage:.1f}%) | Review Status: {c.review_status}")
        print(f"  Top terms: {top_kws}")
        print("  Representative Examples:")
        for ex in c.representative_examples[:2]:
            print(f"    * {ex[:110]}...")

    print("\n" + "=" * 70)
    print("All artifacts saved to data/interim/ and artifacts/figures/.")
    print("Next step: Perform human review using docs/intent_review_guide.md.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
