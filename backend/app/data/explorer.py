"""
SupportGraph AI — Dataset Explorer

Orchestrates the full Phase 1 exploration pipeline.

Sections:
  A. Dataset Structure      — shape, columns, types, missing, duplicates
  B. Message Analysis       — text length stats, empty messages (dynamic detection)
  C. Account Analysis       — unique authors, top accounts, inbound/outbound split
  D. Conversation Structure — reply chains, thread indicators
  E. Brand Candidate Prep   — aggregate stats per potential brand account

All column detection is DYNAMIC — no column names are hardcoded as required.
The explorer inspects the actual schema and adapts its analysis accordingly.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd

try:
    from app.core.logging import get_logger
    from app.data.cleaner import CleaningReport, analyze_cleaning_needs
    from app.data.loader import DatasetMetadata, get_dataset_metadata, load_dataset
except ModuleNotFoundError:
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]
    from backend.app.data.cleaner import CleaningReport, analyze_cleaning_needs  # type: ignore[no-redef]
    from backend.app.data.loader import DatasetMetadata, get_dataset_metadata, load_dataset  # type: ignore[no-redef]

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------
@dataclass
class StructureAnalysis:
    """A. Dataset structure findings."""
    row_count: int
    column_count: int
    columns: list[str]
    dtypes: dict[str, str]
    missing_by_column: dict[str, dict[str, Any]]  # col → {count, pct}
    duplicate_count: int
    duplicate_pct: float
    file_size_mb: float
    filename: str


@dataclass
class MessageAnalysis:
    """B. Text/message column analysis."""
    text_column_found: bool
    text_column_name: Optional[str]
    total_messages: int
    empty_message_count: int
    length_stats: dict[str, float]   # min, max, mean, median, std, p25, p75, p95
    sample_messages: list[str] = field(default_factory=list)


@dataclass
class AccountAnalysis:
    """C. Author/account analysis."""
    author_column_found: bool
    author_column_name: Optional[str]
    inbound_column_found: bool
    inbound_column_name: Optional[str]
    total_unique_authors: int
    inbound_count: int         # customer tweets
    outbound_count: int        # brand tweets
    top_accounts: list[dict[str, Any]]   # [{author, count, inbound, outbound}]


@dataclass
class ConversationAnalysis:
    """D. Conversation / thread structure analysis."""
    reply_to_column_found: bool
    reply_to_column_name: Optional[str]
    response_ids_column_found: bool
    response_ids_column_name: Optional[str]
    thread_id_column_found: bool
    thread_id_column_name: Optional[str]
    messages_with_parent: int      # tweets that are replies
    messages_without_parent: int   # thread starters
    messages_with_child: int       # tweets that received responses


@dataclass
class BrandCandidateStats:
    """Stats for one potential brand account."""
    author_id: str
    total_messages: int
    outbound_count: int    # brand responses
    inbound_count: int     # messages received (approximated)
    response_ratio: float  # outbound / total
    usable_interaction_count: int  # outbound replies in a thread


@dataclass
class ExplorationResult:
    """Complete Phase 1 exploration result."""
    structure: StructureAnalysis
    message: MessageAnalysis
    account: AccountAnalysis
    conversation: ConversationAnalysis
    brand_candidates: list[BrandCandidateStats]
    cleaning_report: CleaningReport
    metadata: DatasetMetadata


# ---------------------------------------------------------------------------
# Column detection helpers
# ---------------------------------------------------------------------------
def _detect_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """
    Return the first column name from candidates that exists in df (case-insensitive).

    Args:
        df:         DataFrame to inspect.
        candidates: Ordered list of likely column name variants.

    Returns:
        The matched column name, or None if none found.
    """
    lower_to_actual = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in lower_to_actual:
            return lower_to_actual[c.lower()]
    return None


# ---------------------------------------------------------------------------
# Section analyzers
# ---------------------------------------------------------------------------
def _analyze_structure(
    df: pd.DataFrame,
    metadata: DatasetMetadata,
    cleaning: CleaningReport,
) -> StructureAnalysis:
    """A. Build structure analysis from metadata and cleaning report."""
    logger.info("A. Analyzing dataset structure...")

    missing_by_column: dict[str, dict[str, Any]] = {}
    for col in df.columns:
        null_count = int(df[col].isna().sum())
        if null_count > 0:
            missing_by_column[col] = {
                "count": null_count,
                "pct": round((null_count / len(df)) * 100, 2),
            }

    return StructureAnalysis(
        row_count=metadata.row_count,
        column_count=metadata.column_count,
        columns=metadata.columns,
        dtypes=metadata.dtypes,
        missing_by_column=missing_by_column,
        duplicate_count=cleaning.duplicate_row_count,
        duplicate_pct=cleaning.duplicate_row_pct,
        file_size_mb=metadata.file_size_mb,
        filename=metadata.filename,
    )


def _analyze_messages(df: pd.DataFrame) -> MessageAnalysis:
    """B. Detect and analyze the primary text/message column."""
    logger.info("B. Analyzing message content...")

    text_col = _detect_column(df, ["text", "message", "body", "content", "tweet_text"])

    if text_col is None:
        logger.warning("No text/message column detected in the dataset.")
        return MessageAnalysis(
            text_column_found=False,
            text_column_name=None,
            total_messages=0,
            empty_message_count=0,
            length_stats={},
        )

    logger.info("  Text column detected: '%s'", text_col)
    series = df[text_col].astype(str)

    empty_count = int(
        (series.isna() | (series.str.strip() == "") | (series == "nan")).sum()
    )
    lengths = series.str.len()

    stats = {
        "min": float(lengths.min()),
        "max": float(lengths.max()),
        "mean": round(float(lengths.mean()), 2),
        "median": round(float(lengths.median()), 2),
        "std": round(float(lengths.std()), 2),
        "p25": round(float(lengths.quantile(0.25)), 2),
        "p75": round(float(lengths.quantile(0.75)), 2),
        "p95": round(float(lengths.quantile(0.95)), 2),
    }

    logger.info("  Message length — mean: %.1f, max: %.0f", stats["mean"], stats["max"])

    # Sample 3 non-empty messages for the report
    non_empty = df[text_col].dropna()
    samples = non_empty[non_empty.str.strip() != ""].head(3).tolist()

    return MessageAnalysis(
        text_column_found=True,
        text_column_name=text_col,
        total_messages=len(df),
        empty_message_count=empty_count,
        length_stats=stats,
        sample_messages=[str(s)[:200] for s in samples],
    )


def _analyze_accounts(df: pd.DataFrame) -> AccountAnalysis:
    """C. Analyze author / account distribution."""
    logger.info("C. Analyzing account distribution...")

    author_col = _detect_column(
        df, ["author_id", "author", "user_id", "username", "screen_name", "from"]
    )
    inbound_col = _detect_column(
        df, ["inbound", "is_inbound", "direction", "type"]
    )

    if author_col is None:
        logger.warning("No author/account column detected.")
        return AccountAnalysis(
            author_column_found=False,
            author_column_name=None,
            inbound_column_found=False,
            inbound_column_name=None,
            total_unique_authors=0,
            inbound_count=0,
            outbound_count=0,
            top_accounts=[],
        )

    logger.info("  Author column: '%s'", author_col)

    unique_authors = df[author_col].nunique()
    logger.info("  Unique authors: %d", unique_authors)

    # Inbound / outbound split
    inbound_count = 0
    outbound_count = 0
    if inbound_col is not None:
        logger.info("  Inbound column: '%s'", inbound_col)
        # Handle both boolean True/False and string "True"/"False"
        inbound_series = df[inbound_col]
        if inbound_series.dtype == object:
            inbound_series = inbound_series.map(
                lambda x: str(x).strip().lower() in ("true", "1", "yes")
            )
        inbound_count = int(inbound_series.sum())
        outbound_count = len(df) - inbound_count
        logger.info(
            "  Inbound (customer): %d | Outbound (brand): %d",
            inbound_count,
            outbound_count,
        )

    # Top accounts by message volume
    top_n = 20
    account_counts = df[author_col].value_counts().head(top_n)
    top_accounts: list[dict[str, Any]] = []

    for author, total in account_counts.items():
        row: dict[str, Any] = {"author": str(author), "total_messages": int(total)}

        if inbound_col is not None:
            author_mask = df[author_col] == author
            inbound_series = df.loc[author_mask, inbound_col]
            if inbound_series.dtype == object:
                inbound_series = inbound_series.map(
                    lambda x: str(x).strip().lower() in ("true", "1", "yes")
                )
            row["inbound"] = int(inbound_series.sum())
            row["outbound"] = int(total) - row["inbound"]

        top_accounts.append(row)

    return AccountAnalysis(
        author_column_found=True,
        author_column_name=author_col,
        inbound_column_found=inbound_col is not None,
        inbound_column_name=inbound_col,
        total_unique_authors=int(unique_authors),
        inbound_count=inbound_count,
        outbound_count=outbound_count,
        top_accounts=top_accounts,
    )


def _analyze_conversations(df: pd.DataFrame) -> ConversationAnalysis:
    """D. Analyze conversation / reply thread structure."""
    logger.info("D. Analyzing conversation structure...")

    reply_to_col = _detect_column(
        df,
        [
            "in_response_to_tweet_id",
            "in_reply_to_tweet_id",
            "parent_id",
            "reply_to",
            "response_to",
        ],
    )
    response_ids_col = _detect_column(
        df,
        [
            "response_tweet_id",
            "child_id",
            "replies",
            "response_ids",
        ],
    )
    thread_id_col = _detect_column(
        df,
        [
            "conversation_id",
            "thread_id",
            "session_id",
        ],
    )

    messages_with_parent = 0
    messages_without_parent = 0
    messages_with_child = 0

    if reply_to_col is not None:
        logger.info("  Reply-to column: '%s'", reply_to_col)
        has_parent = df[reply_to_col].notna()
        messages_with_parent = int(has_parent.sum())
        messages_without_parent = len(df) - messages_with_parent
        logger.info(
            "  Messages with parent: %d | Thread starters: %d",
            messages_with_parent,
            messages_without_parent,
        )

    if response_ids_col is not None:
        logger.info("  Response-ids column: '%s'", response_ids_col)
        has_child = df[response_ids_col].notna() & (df[response_ids_col].astype(str).str.strip() != "")
        messages_with_child = int(has_child.sum())
        logger.info("  Messages with responses: %d", messages_with_child)

    if thread_id_col is not None:
        logger.info("  Thread-ID column: '%s'", thread_id_col)

    return ConversationAnalysis(
        reply_to_column_found=reply_to_col is not None,
        reply_to_column_name=reply_to_col,
        response_ids_column_found=response_ids_col is not None,
        response_ids_column_name=response_ids_col,
        thread_id_column_found=thread_id_col is not None,
        thread_id_column_name=thread_id_col,
        messages_with_parent=messages_with_parent,
        messages_without_parent=messages_without_parent,
        messages_with_child=messages_with_child,
    )


def _prepare_brand_candidates(
    df: pd.DataFrame,
    account_analysis: AccountAnalysis,
) -> list[BrandCandidateStats]:
    """
    E. Identify potential brand/company accounts and compute interaction stats.

    Brand accounts are identified as outbound (inbound=False) accounts.
    Anonymous customer accounts are typically numeric IDs.
    Brand handles are typically alphabetic strings.

    Args:
        df:               Full DataFrame.
        account_analysis: Result from _analyze_accounts.

    Returns:
        List of BrandCandidateStats, sorted by usable_interaction_count descending.
    """
    logger.info("E. Preparing brand candidates...")

    if not account_analysis.author_column_found or not account_analysis.inbound_column_found:
        logger.warning(
            "Cannot identify brand candidates without both author and inbound columns."
        )
        return []

    author_col = account_analysis.author_column_name
    inbound_col = account_analysis.inbound_column_name

    # Normalize inbound column to boolean
    inbound_series = df[inbound_col]
    if inbound_series.dtype == object:
        df = df.copy()
        df["_inbound_bool"] = inbound_series.map(
            lambda x: str(x).strip().lower() in ("true", "1", "yes")
        )
        inbound_bool_col = "_inbound_bool"
    else:
        inbound_bool_col = inbound_col

    # Brand accounts: authors who appear in outbound (inbound=False) rows
    outbound_df = df[~df[inbound_bool_col]]
    brand_authors = outbound_df[author_col].unique()

    logger.info("  Potential brand accounts identified: %d", len(brand_authors))

    # Filter to alphabetic handles (brand names, not numeric customer IDs)
    brand_authors_filtered = [
        a for a in brand_authors
        if not str(a).strip().isdigit()
    ]
    logger.info(
        "  Non-numeric brand handles: %d",
        len(brand_authors_filtered),
    )

    candidates: list[BrandCandidateStats] = []
    reply_to_col = None
    for col_candidate in ["in_response_to_tweet_id", "in_reply_to_tweet_id", "response_to"]:
        if col_candidate in df.columns:
            reply_to_col = col_candidate
            break

    for author in brand_authors_filtered:
        author_mask = df[author_col] == author
        author_df = df[author_mask]
        total = len(author_df)

        outbound_count = int((~author_df[inbound_bool_col]).sum())
        inbound_received = 0  # messages by customers to this brand

        # Approximate inbound: rows where inbound=True and this brand was mentioned or replied to
        # We use: outbound rows that have a parent (i.e., they responded to something)
        if reply_to_col:
            outbound_with_reply = author_df[
                (~author_df[inbound_bool_col]) & (author_df[reply_to_col].notna())
            ]
            usable = len(outbound_with_reply)
        else:
            usable = outbound_count

        response_ratio = round(outbound_count / total, 4) if total > 0 else 0.0

        candidates.append(
            BrandCandidateStats(
                author_id=str(author),
                total_messages=total,
                outbound_count=outbound_count,
                inbound_count=inbound_received,
                response_ratio=response_ratio,
                usable_interaction_count=usable,
            )
        )

    # Sort by usable interaction count (most useful for RAG / training)
    candidates.sort(key=lambda c: c.usable_interaction_count, reverse=True)

    logger.info(
        "Top 5 brand candidates by usable interactions: %s",
        [c.author_id for c in candidates[:5]],
    )

    return candidates


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def run_exploration(
    filepath: Optional[Path] = None,
    nrows: Optional[int] = None,
) -> ExplorationResult:
    """
    Run the complete Phase 1 dataset exploration pipeline.

    Args:
        filepath: Optional explicit path to CSV. If None, auto-discovers.
        nrows:    Optional row limit (useful for quick testing).

    Returns:
        ExplorationResult with all analysis sections populated.
    """
    logger.info("=" * 60)
    logger.info("SupportGraph AI — Phase 1 Dataset Exploration")
    logger.info("=" * 60)

    # Load data
    df = load_dataset(filepath=filepath, nrows=nrows)

    # Extract metadata
    try:
        from app.data.loader import discover_dataset
    except ModuleNotFoundError:
        from backend.app.data.loader import discover_dataset  # type: ignore[no-redef]
    if filepath is None:
        filepath, _ = discover_dataset()
    metadata = get_dataset_metadata(df, filepath)

    # Non-destructive cleaning analysis
    cleaning = analyze_cleaning_needs(df)

    # Run all sections
    structure = _analyze_structure(df, metadata, cleaning)
    message = _analyze_messages(df)
    account = _analyze_accounts(df)
    conversation = _analyze_conversations(df)
    brand_candidates = _prepare_brand_candidates(df, account)

    result = ExplorationResult(
        structure=structure,
        message=message,
        account=account,
        conversation=conversation,
        brand_candidates=brand_candidates,
        cleaning_report=cleaning,
        metadata=metadata,
    )

    logger.info("Exploration complete.")
    return result
