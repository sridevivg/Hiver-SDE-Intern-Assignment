"""
SupportGraph AI — Data Cleaning Analysis

Performs NON-DESTRUCTIVE analysis of the raw dataset.

Responsibilities:
  - Identify missing values per column.
  - Identify empty/whitespace-only strings.
  - Identify duplicate rows.
  - Record all findings in a typed report.
  - NEVER modify or delete rows from the raw DataFrame.

All cleaning decisions are documented. Actual transformations (if any)
will produce NEW DataFrames in downstream processing steps — never in-place
on the raw data.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

try:
    from app.core.logging import get_logger
except ModuleNotFoundError:
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class ColumnCleaningReport:
    """Cleaning findings for a single column."""
    column_name: str
    dtype: str
    total_values: int
    null_count: int
    null_pct: float
    empty_string_count: int      # only meaningful for string columns
    whitespace_only_count: int   # only meaningful for string columns
    is_string_type: bool


@dataclass
class CleaningReport:
    """Aggregate non-destructive cleaning analysis for the entire dataset."""
    total_rows: int
    total_columns: int
    duplicate_row_count: int
    duplicate_row_pct: float
    fully_null_rows: int
    column_reports: list[ColumnCleaningReport] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)

    # Convenience accessors
    @property
    def columns_with_nulls(self) -> list[ColumnCleaningReport]:
        return [r for r in self.column_reports if r.null_count > 0]

    @property
    def string_columns_with_empty(self) -> list[ColumnCleaningReport]:
        return [
            r for r in self.column_reports
            if r.is_string_type and (r.empty_string_count > 0 or r.whitespace_only_count > 0)
        ]

    @property
    def total_null_cells(self) -> int:
        return sum(r.null_count for r in self.column_reports)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _analyze_column(series: pd.Series) -> ColumnCleaningReport:
    """
    Analyze a single column for quality issues.

    Args:
        series: A pandas Series (one column of the DataFrame).

    Returns:
        ColumnCleaningReport with all findings for that column.
    """
    col_name = str(series.name)
    dtype_str = str(series.dtype)
    total = len(series)
    null_count = int(series.isna().sum())
    null_pct = round((null_count / total) * 100, 2) if total > 0 else 0.0

    # String-specific analysis
    is_string = dtype_str in ("object", "string") or "string" in dtype_str.lower()
    empty_count = 0
    whitespace_count = 0

    if is_string:
        non_null = series.dropna()
        empty_count = int((non_null == "").sum())
        whitespace_count = int(non_null.str.strip().eq("").sum()) - empty_count
        whitespace_count = max(0, whitespace_count)

    return ColumnCleaningReport(
        column_name=col_name,
        dtype=dtype_str,
        total_values=total,
        null_count=null_count,
        null_pct=null_pct,
        empty_string_count=empty_count,
        whitespace_only_count=whitespace_count,
        is_string_type=is_string,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def analyze_cleaning_needs(df: pd.DataFrame) -> CleaningReport:
    """
    Perform a full non-destructive cleaning analysis on the DataFrame.

    This function ONLY reads data — it never modifies df.

    Args:
        df: The raw DataFrame to analyze.

    Returns:
        CleaningReport with complete quality findings and documented decisions.
    """
    logger.info("Starting non-destructive cleaning analysis...")
    logger.info("Input shape: %d rows × %d columns", len(df), len(df.columns))

    # -----------------------------------------------------------------------
    # Duplicate analysis
    # -----------------------------------------------------------------------
    total_rows = len(df)
    duplicate_mask = df.duplicated(keep="first")
    duplicate_count = int(duplicate_mask.sum())
    duplicate_pct = round((duplicate_count / total_rows) * 100, 2) if total_rows > 0 else 0.0

    logger.info(
        "Duplicate rows: %d (%.2f%%) — keeping first occurrence in analysis",
        duplicate_count,
        duplicate_pct,
    )

    # -----------------------------------------------------------------------
    # Fully null rows
    # -----------------------------------------------------------------------
    fully_null_count = int(df.isna().all(axis=1).sum())
    logger.info("Fully null rows: %d", fully_null_count)

    # -----------------------------------------------------------------------
    # Per-column analysis
    # -----------------------------------------------------------------------
    column_reports: list[ColumnCleaningReport] = []
    for col in df.columns:
        report = _analyze_column(df[col])
        column_reports.append(report)
        if report.null_count > 0:
            logger.info(
                "  Column '%s': %d nulls (%.1f%%)",
                col,
                report.null_count,
                report.null_pct,
            )
        if report.empty_string_count > 0:
            logger.info(
                "  Column '%s': %d empty strings",
                col,
                report.empty_string_count,
            )

    # -----------------------------------------------------------------------
    # Document cleaning decisions
    # -----------------------------------------------------------------------
    decisions: list[str] = [
        "DECISION: Raw DataFrame is NOT modified by this analysis.",
        "DECISION: Duplicate rows are identified but NOT removed here; "
        "removal (if decided) will produce a new DataFrame in data/interim/.",
        "DECISION: Null values are reported per-column. Imputation or row-drop "
        "decisions are deferred to the processing phase.",
        "DECISION: Empty strings are distinguished from null values. "
        "Both are counted separately to preserve granularity.",
    ]

    if duplicate_count > 0:
        decisions.append(
            f"FINDING: {duplicate_count} duplicate rows detected. "
            "Recommend deduplication before model training."
        )
    if fully_null_count > 0:
        decisions.append(
            f"FINDING: {fully_null_count} fully-null rows detected. "
            "These should be dropped before analysis."
        )

    # Columns with high null rate
    high_null_cols = [r for r in column_reports if r.null_pct > 50]
    for r in high_null_cols:
        decisions.append(
            f"FINDING: Column '{r.column_name}' has {r.null_pct:.1f}% null values. "
            "Consider whether this column is useful for modeling."
        )

    cleaning_report = CleaningReport(
        total_rows=total_rows,
        total_columns=len(df.columns),
        duplicate_row_count=duplicate_count,
        duplicate_row_pct=duplicate_pct,
        fully_null_rows=fully_null_count,
        column_reports=column_reports,
        decisions=decisions,
    )

    logger.info(
        "Cleaning analysis complete. Total null cells: %d. Decisions documented: %d.",
        cleaning_report.total_null_cells,
        len(decisions),
    )

    return cleaning_report
