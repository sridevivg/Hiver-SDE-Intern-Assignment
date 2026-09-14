"""
SupportGraph AI — Golden Evaluation Set Sampler (Phase 5)

Generates a reproducible, stratified candidate evaluation dataset (target: 200 examples)
drawn from high-quality customer opening messages.

Sampling Rules:
- Stratified across all candidate operational intents
- Deterministic random sampling (random_seed = 42)
- Zero duplicate tweet_ids
- Zero duplicate normalized messages
- Maintains balanced representation of minority intents (~22 samples per intent)
- Includes conversation context (first AppleSupport response) to aid human disambiguation
- Initializes annotation_label to empty string (human annotation pending)
- Generates golden_sampling_manifest.json recording metadata, SHA256 hashes, and quotas.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from app.core.logging import get_logger
    from app.nlp.taxonomy_finalization import assign_candidate_intent
except ModuleNotFoundError:
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]
    from backend.app.nlp.taxonomy_finalization import assign_candidate_intent  # type: ignore[no-redef]

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------
@dataclass
class GoldenRecord:
    """A single candidate record in the golden evaluation dataset."""
    golden_id: str
    tweet_id: str
    conversation_id: str
    customer_message: str
    normalized_message: str
    candidate_intent: str
    source_cluster: int
    conversation_context: str
    annotation_label: str = ""
    annotation_status: str = "pending"
    annotator: str = ""
    notes: str = ""


DEFAULT_CORPUS_PATH = Path("data/interim/intent_discovery_corpus.csv")
DEFAULT_MESSAGES_PATH = Path("data/processed/conversation_messages.parquet")
DEFAULT_GOLDEN_DIR = Path("data/golden")


@dataclass
class GoldenSamplingResult:
    """Complete output of the golden sampling process."""
    df_candidates: pd.DataFrame
    manifest: dict[str, Any]
    candidates_csv_path: Path
    template_csv_path: Path
    manifest_json_path: Path

    @property
    def sample_size(self) -> int:
        return len(self.df_candidates)

    @property
    def intent_distribution(self) -> dict[str, int]:
        return self.manifest.get("candidate_class_distribution", {})

    @property
    def candidates_csv(self) -> Path:
        return self.candidates_csv_path

    @property
    def annotation_template_csv(self) -> Path:
        return self.template_csv_path

    @property
    def manifest_path(self) -> Path:
        return self.manifest_json_path


GoldenSampleResult = GoldenSamplingResult


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 checksum of a file."""
    if not file_path.exists():
        return ""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_stratified_quotas(
    intent_counts: dict[str, int],
    target_size: int = 200,
) -> dict[str, int]:
    """
    Compute integer quotas per intent so that total equals target_size,
    with balanced distribution across all intents.
    """
    num_intents = len(intent_counts)
    if num_intents == 0:
        return {}

    base_quota = target_size // num_intents
    remainder = target_size % num_intents

    # Sort intents by available count descending to allocate remainder to largest pools
    sorted_intents = sorted(intent_counts.keys(), key=lambda k: intent_counts[k], reverse=True)

    quotas: dict[str, int] = {}
    for i, intent in enumerate(sorted_intents):
        q = base_quota + (1 if i < remainder else 0)
        # Cannot exceed available count
        quotas[intent] = min(q, intent_counts[intent])

    # Re-distribute any deficit if an intent had fewer than quota
    total_allocated = sum(quotas.values())
    shortfall = target_size - total_allocated

    if shortfall > 0:
        for intent in sorted_intents:
            available = intent_counts[intent] - quotas[intent]
            add = min(shortfall, available)
            quotas[intent] += add
            shortfall -= add
            if shortfall == 0:
                break

    return quotas


# ---------------------------------------------------------------------------
# Main Sampling Engine
# ---------------------------------------------------------------------------
def sample_golden_candidates(
    corpus_path: Path,
    messages_path: Path,
    output_dir: Path,
    target_size: int = 200,
    random_seed: int = 42,
    taxonomy_version: str = "v1.0-candidate",
) -> GoldenSamplingResult:
    """
    Execute stratified sampling of golden candidate messages from Phase 4 corpus.
    """
    logger.info("=" * 70)
    logger.info("STARTING PHASE 5: GOLDEN EVALUATION SET SAMPLING")
    logger.info("Target size: %d | Random seed: %d", target_size, random_seed)
    logger.info("=" * 70)

    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    # 1. Load corpus
    df_corpus = pd.read_csv(corpus_path)
    initial_count = len(df_corpus)
    logger.info("Loaded %d corpus records from %s", initial_count, corpus_path)

    # 2. Assign candidate intents
    logger.info("Assigning candidate intents to corpus messages...")
    df_corpus["candidate_intent"] = df_corpus.apply(
        lambda row: assign_candidate_intent(row["normalized_text"], row["cluster_id"]),
        axis=1,
    )

    # 3. Deduplication: drop duplicate tweet_ids and normalized texts
    df_unique = df_corpus.drop_duplicates(subset=["tweet_id"]).copy()
    tweet_dupes_removed = initial_count - len(df_unique)

    before_text_dedupe = len(df_unique)
    df_unique = df_unique.drop_duplicates(subset=["normalized_text"]).copy()
    text_dupes_removed = before_text_dedupe - len(df_unique)

    logger.info(
        "Deduplication: removed %d duplicate tweet IDs and %d duplicate normalized texts. Remaining pool: %d",
        tweet_dupes_removed,
        text_dupes_removed,
        len(df_unique),
    )

    # 4. Load conversation context from messages parquet
    conversation_context_map: dict[str, str] = {}
    if messages_path.exists():
        logger.info("Loading conversation context (first brand reply) from %s...", messages_path)
        try:
            df_msgs = pd.read_parquet(
                messages_path,
                columns=["conversation_id", "role", "depth", "text"],
            )
            brand_replies = df_msgs[df_msgs["role"] == "brand"].sort_values(
                ["conversation_id", "depth"]
            )
            first_brand = brand_replies.groupby("conversation_id").first().reset_index()
            for _, r in first_brand.iterrows():
                conversation_context_map[str(r["conversation_id"])] = str(r["text"]).strip()
            logger.info("Loaded brand turn context for %d conversations.", len(conversation_context_map))
        except Exception as exc:
            logger.warning("Could not load brand context: %s", exc)

    # 5. Compute class distribution and quotas
    intent_counts = df_unique["candidate_intent"].value_counts().to_dict()
    quotas = compute_stratified_quotas(intent_counts, target_size=target_size)
    logger.info("Calculated stratified quotas across %d intents: %s", len(quotas), quotas)

    # 6. Stratified deterministic sampling
    sampled_dfs: list[pd.DataFrame] = []
    rng = np.random.RandomState(random_seed)

    for intent_name, quota in quotas.items():
        intent_pool = df_unique[df_unique["candidate_intent"] == intent_name]
        if len(intent_pool) <= quota:
            sampled_intent = intent_pool.copy()
        else:
            sampled_indices = rng.choice(len(intent_pool), size=quota, replace=False)
            sampled_intent = intent_pool.iloc[sampled_indices].copy()
        sampled_dfs.append(sampled_intent)

    df_sampled = pd.concat(sampled_dfs, ignore_index=True)

    # Shuffle the final sampled candidates deterministically so they are not sorted by intent
    shuffle_indices = rng.permutation(len(df_sampled))
    df_sampled = df_sampled.iloc[shuffle_indices].reset_index(drop=True)

    # 7. Construct Golden Records
    golden_records: list[GoldenRecord] = []
    for idx, row in df_sampled.iterrows():
        g_id = f"gold_{idx + 1:03d}"
        conv_id = str(row["conversation_id"])
        context = conversation_context_map.get(conv_id, "No brand response observed")

        msg_text = row.get("original_text") if "original_text" in row and pd.notna(row["original_text"]) else row.get("text", "")
        norm_text = row.get("normalized_text") if "normalized_text" in row and pd.notna(row["normalized_text"]) else row.get("normalized_message", "")

        record = GoldenRecord(
            golden_id=g_id,
            tweet_id=str(row["tweet_id"]),
            conversation_id=conv_id,
            customer_message=str(msg_text).strip(),
            normalized_message=str(norm_text).strip(),
            candidate_intent=str(row["candidate_intent"]),
            source_cluster=int(row["cluster_id"]),
            conversation_context=context[:300],  # clean context limit
            annotation_label="",                # MUST REMAIN EMPTY
            annotation_status="pending",
            annotator="",
            notes="",
        )
        golden_records.append(record)

    df_final = pd.DataFrame([asdict(r) for r in golden_records])

    # 8. Save Artifacts
    output_dir.mkdir(parents=True, exist_ok=True)

    # Candidates CSV
    candidates_csv_path = output_dir / "golden_set_candidates.csv"
    df_final.to_csv(candidates_csv_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
    logger.info("Saved golden candidates to: %s", candidates_csv_path)

    # Annotation Template CSV (exact copy intended for annotators to fill)
    template_csv_path = output_dir / "golden_set_annotation_template.csv"
    df_final.to_csv(template_csv_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
    logger.info("Saved golden annotation template to: %s", template_csv_path)

    # Sampling Manifest JSON
    manifest_json_path = output_dir / "golden_sampling_manifest.json"
    actual_dist = df_final["candidate_intent"].value_counts().to_dict()

    manifest = {
        "manifest_version": "1.0",
        "dataset_source": "Kaggle Customer Support on Twitter (AppleSupport subset)",
        "brand": "AppleSupport",
        "sampling_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": random_seed,
        "target_sample_size": target_size,
        "actual_sample_size": len(df_final),
        "taxonomy_version": taxonomy_version,
        "corpus_source_sha256": compute_sha256(corpus_path),
        "candidates_csv_sha256": compute_sha256(candidates_csv_path),
        "sampling_strategy": "STRATIFIED_RANDOM_WITHOUT_REPLACEMENT",
        "candidate_class_distribution": actual_dist,
        "duplicate_removal_statistics": {
            "initial_corpus_count": initial_count,
            "duplicate_tweet_ids_removed": tweet_dupes_removed,
            "duplicate_texts_removed": text_dupes_removed,
            "eligible_unique_pool": len(df_unique),
        },
        "annotation_status": "PENDING_HUMAN_ANNOTATION",
    }

    with open(manifest_json_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Saved golden sampling manifest to: %s", manifest_json_path)

    return GoldenSamplingResult(
        df_candidates=df_final,
        manifest=manifest,
        candidates_csv_path=candidates_csv_path,
        template_csv_path=template_csv_path,
        manifest_json_path=manifest_json_path,
    )


def run_golden_sampling(
    corpus_path: Path | str = DEFAULT_CORPUS_PATH,
    messages_path: Optional[Path | str] = None,
    output_dir: Path | str = DEFAULT_GOLDEN_DIR,
    target_size: int = 200,
    random_seed: int = 42,
    taxonomy_version: str = "v1.0-candidate",
) -> GoldenSamplingResult:
    """
    High-level entry point to sample golden candidate records.
    """
    c_path = Path(corpus_path)
    if messages_path is None:
        messages_path = DEFAULT_MESSAGES_PATH
    m_path = Path(messages_path)
    out_path = Path(output_dir)

    return sample_golden_candidates(
        corpus_path=c_path,
        messages_path=m_path,
        output_dir=out_path,
        target_size=target_size,
        random_seed=random_seed,
        taxonomy_version=taxonomy_version,
    )
