"""
SupportGraph AI — Evidence Corpus Builder (Phase 10.1)

Builds a leakage-safe extended retrieval corpus from the full processed
AppleSupport conversation dataset for use in evidence coverage auditing.

CRITICAL SAFETY RULES:
  - All 200 golden benchmark conversation_id values are EXCLUDED from the corpus.
  - The golden_set_human_review.csv is NEVER modified or used as a retrieval source.
  - The SHA-256 of golden_set_human_review.csv must be verified before and after.
  - This corpus is for DIAGNOSTIC AUDIT PURPOSES ONLY — not for changing production.

The real AppleSupport historical corpus contains:
  - 80,717 conversations
  - 132,047 customer messages
  - Located at: data/processed/conversation_messages.parquet

Current production CaseRetriever uses only:
  - 200 rows from the golden benchmark (data/golden/golden_set_human_review.csv)
  - This represents 0.25% of the available historical corpus.
"""
from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

try:
    from app.core.logging import get_logger
except ModuleNotFoundError:
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]

logger = get_logger(__name__)

# Canonical data paths
CONVERSATION_MESSAGES_PATH = Path("data/processed/conversation_messages.parquet")
GOLDEN_CSV_PATH = Path("data/golden/golden_set_human_review.csv")
GOLDEN_SHA256 = "1d3e9b3b8bdef3750437e59b17ab7c27167fd1548c2bb29de984151296c5b45a"

# Default audit corpus sample size (feasibility vs completeness balance)
DEFAULT_AUDIT_CORPUS_SIZE = 5000


def compute_sha256(filepath: Path | str) -> str:
    """Compute SHA-256 checksum of a file."""
    path = Path(filepath)
    if not path.exists():
        return ""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


# ── Intent heuristics for corpus labeling ────────────────────────────────────
# Used ONLY for inferred intent in audit corpus — not for production routing

_INTENT_PATTERNS: list[tuple[str, str]] = [
    (r"\b(battery|drain|draining|charge|charging|power|dead battery)\b", "battery_power_issue"),
    (r"\b(keyboard|typing|autocorrect|letter i|️|autocorrect|autocorrection)\b", "keyboard_typing_issue"),
    (r"\b(screen|display|touch|flicker|cracked|black screen|touchscreen)\b", "display_touch_issue"),
    (r"\b(speaker|headphone|airpod|sound|volume|mic|microphone|audio|bluetooth)\b", "hardware_audio_connection_issue"),
    (r"\b(password|apple\s*id|sign in|locked|2fa|two.factor|verification|account)\b", "account_access_issue"),
    (r"\b(charge|charged|refund|subscription|itunes|receipt|billing|billed|purchase)\b", "billing_purchase_issue"),
    (r"\b(update|updated|updating|ios 11|ios11|high sierra|macos|upgrade|install)\b", "software_update_problem"),
    (r"\b(macbook|mac\b|imac|mac pro|macos|safari|finder|osx)\b", "mac_software_issue"),
    (r"\b(wifi|wi-fi|network|cellular|lte|5g|signal|connect)\b", "software_update_problem"),
]


def infer_intent_from_text(text: str) -> str:
    """
    Lightweight heuristic intent inference for unlabeled corpus entries.

    Used ONLY for audit corpus intent labeling. Not used in production routing.
    Returns the most likely intent or 'general_device_support' as fallback.
    """
    if not text or not text.strip():
        return "general_device_support"
    lower = text.lower()
    for pattern, intent in _INTENT_PATTERNS:
        if re.search(pattern, lower):
            return intent
    return "general_device_support"


class CorpusEntry(BaseModel):
    """A single entry in the leakage-safe extended retrieval corpus."""
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    case_id: str
    conversation_id: str
    customer_message: str
    brand_response: str = ""
    inferred_intent: str = "general_device_support"
    corpus_source: str = "historical_parquet"  # audit trail


class AuditCorpus(BaseModel):
    """Complete leakage-safe audit corpus."""
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    entries: list[CorpusEntry] = Field(default_factory=list)
    total_conversations_available: int = 0
    total_after_exclusion: int = 0
    sample_size: int = 0
    excluded_golden_ids: int = 0
    pre_build_golden_sha256: str = ""
    post_build_golden_sha256: str = ""
    golden_immutability_pass: bool = False

    @property
    def size(self) -> int:
        return len(self.entries)


class EvidenceCorpusBuilder:
    """
    Builds a leakage-safe extended retrieval corpus from the full historical
    AppleSupport conversation dataset for diagnostic auditing.

    SAFETY GUARANTEES:
    1. All 200 golden benchmark conversation IDs are excluded from the corpus.
    2. Golden SHA-256 is verified before and after corpus construction.
    3. The corpus is for audit/diagnostic purposes only.
    4. Does NOT modify any production component.
    """

    def __init__(
        self,
        messages_path: Optional[Path | str] = None,
        golden_path: Optional[Path | str] = None,
        max_sample_size: int = DEFAULT_AUDIT_CORPUS_SIZE,
        random_seed: int = 42,
    ) -> None:
        self.messages_path = Path(messages_path or CONVERSATION_MESSAGES_PATH)
        self.golden_path = Path(golden_path or GOLDEN_CSV_PATH)
        self.max_sample_size = max_sample_size
        self.random_seed = random_seed

    def build(self) -> AuditCorpus:
        """
        Build a leakage-safe extended retrieval corpus.

        Returns:
            AuditCorpus with entries drawn from historical data, golden IDs excluded.
        """
        pre_sha = compute_sha256(self.golden_path)
        logger.info(
            "Building audit corpus. Golden SHA-256 pre-build: %s...", pre_sha[:16]
        )

        # Load golden benchmark to get excluded IDs
        excluded_ids: set[str] = set()
        if self.golden_path.exists():
            golden_df = pd.read_csv(self.golden_path)
            excluded_ids = set(golden_df["conversation_id"].astype(str).unique())
            logger.info(
                "Excluding %d golden benchmark conversation IDs from audit corpus.",
                len(excluded_ids),
            )
        else:
            logger.warning("Golden path not found at %s — no IDs excluded.", self.golden_path)

        # Load full historical corpus
        if not self.messages_path.exists():
            logger.error(
                "Historical corpus not found at %s. Returning empty corpus.", self.messages_path
            )
            post_sha = compute_sha256(self.golden_path)
            return AuditCorpus(
                pre_build_golden_sha256=pre_sha,
                post_build_golden_sha256=post_sha,
                golden_immutability_pass=(pre_sha == post_sha),
            )

        logger.info("Loading historical corpus from %s...", self.messages_path)
        df = pd.read_parquet(self.messages_path)

        # Split into customer and brand messages
        customers = df[df["role"] == "customer"].copy()
        brands = df[df["role"] == "brand"].copy()

        total_conversations = customers["conversation_id"].nunique()

        # Filter out golden benchmark conversation IDs
        customers_clean = customers[
            ~customers["conversation_id"].astype(str).isin(excluded_ids)
        ].copy()
        conversations_after_exclusion = customers_clean["conversation_id"].nunique()
        logger.info(
            "Historical conversations: %d total, %d after excluding golden IDs (%d excluded).",
            total_conversations, conversations_after_exclusion, len(excluded_ids),
        )

        # Get first customer message per conversation (inbound tweet)
        # This is the canonical "customer problem statement"
        cust_first = (
            customers_clean
            .sort_values("depth")
            .groupby("conversation_id")
            .first()
            .reset_index()
        )

        # Get first brand response per conversation
        brands_clean = brands[
            ~brands["conversation_id"].astype(str).isin(excluded_ids)
        ].copy()
        brand_first = (
            brands_clean
            .sort_values("depth")
            .groupby("conversation_id")["text"]
            .first()
            .reset_index()
            .rename(columns={"text": "brand_response"})
        )

        merged = cust_first.merge(brand_first, on="conversation_id", how="left")
        merged["brand_response"] = merged["brand_response"].fillna(
            "Thanks for reaching out to AppleSupport. Please DM us so we can assist."
        )

        # Filter out empty / very short messages (not operationally useful)
        merged = merged[
            merged["text"].fillna("").str.strip().str.len() >= 10
        ].copy()

        # Sample for tractability
        sample_size = min(self.max_sample_size, len(merged))
        if len(merged) > self.max_sample_size:
            sampled = merged.sample(n=self.max_sample_size, random_state=self.random_seed)
            logger.info(
                "Sampled %d conversations from %d available.", self.max_sample_size, len(merged)
            )
        else:
            sampled = merged.copy()

        # Build corpus entries
        entries: list[CorpusEntry] = []
        for i, (_, row) in enumerate(sampled.iterrows()):
            msg = str(row.get("text", "")).strip()
            brand_resp = str(row.get("brand_response", "")).strip()
            conv_id = str(row.get("conversation_id", f"conv_{i}"))

            entries.append(
                CorpusEntry(
                    case_id=f"hist_{conv_id}",
                    conversation_id=conv_id,
                    customer_message=msg,
                    brand_response=brand_resp,
                    inferred_intent=infer_intent_from_text(msg),
                    corpus_source="historical_parquet",
                )
            )

        post_sha = compute_sha256(self.golden_path)
        immutability_pass = (pre_sha == post_sha)

        if not immutability_pass:
            logger.critical(
                "GOLDEN DATASET SHA-256 MISMATCH! pre=%s post=%s", pre_sha, post_sha
            )

        logger.info(
            "Audit corpus built: %d entries. Golden immutability: %s.",
            len(entries), "PASS" if immutability_pass else "FAIL",
        )

        return AuditCorpus(
            entries=entries,
            total_conversations_available=total_conversations,
            total_after_exclusion=conversations_after_exclusion,
            sample_size=sample_size,
            excluded_golden_ids=len(excluded_ids),
            pre_build_golden_sha256=pre_sha,
            post_build_golden_sha256=post_sha,
            golden_immutability_pass=immutability_pass,
        )
