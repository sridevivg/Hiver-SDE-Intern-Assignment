"""
SupportGraph AI — Dataset Loader

Responsibilities:
  - Auto-detect CSV files inside the configured raw data directory.
  - Support multiple fallback candidate paths.
  - Raise clear, actionable errors when no dataset is found.
  - Load large CSVs efficiently (chunked reading option).
  - Return typed metadata without hardcoding any column names.

Raw data is NEVER modified by this module.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    from app.core.config import settings
    from app.core.logging import get_logger
except ModuleNotFoundError:
    from backend.app.core.config import settings  # type: ignore[no-redef]
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Candidate raw data directories (searched in order)
# ---------------------------------------------------------------------------
_RAW_DIR_CANDIDATES: list[Path] = [
    settings.raw_data_path,
    settings.raw_data_path.parent,          # one level up
    settings.raw_data_path.parent.parent,   # two levels up
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class DatasetMetadata:
    """Metadata about a loaded dataset — schema only, no data."""
    filepath: Path
    filename: str
    row_count: int
    column_count: int
    columns: list[str]
    dtypes: dict[str, str]
    file_size_mb: float
    missing_counts: dict[str, int] = field(default_factory=dict)
    sample_rows: int = 0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _find_csv_files(directory: Path) -> list[Path]:
    """
    Recursively find all CSV files within a directory.

    Args:
        directory: Path to search within.

    Returns:
        Sorted list of CSV file paths found.
    """
    if not directory.exists():
        return []
    if not directory.is_dir():
        return []
    return sorted(directory.rglob("*.csv"))


def _resolve_raw_directory() -> Path:
    """
    Find the first candidate directory that contains at least one CSV file.

    Returns:
        Path to the raw data directory.

    Raises:
        FileNotFoundError: If no directory with CSV files is found.
    """
    for candidate in _RAW_DIR_CANDIDATES:
        logger.debug("Checking candidate raw directory: %s", candidate)
        csv_files = _find_csv_files(candidate)
        if csv_files:
            logger.info("Raw data directory resolved: %s", candidate)
            return candidate

    searched = "\n  ".join(str(c) for c in _RAW_DIR_CANDIDATES)
    raise FileNotFoundError(
        "No CSV files found in any of the candidate raw data directories.\n"
        f"Searched:\n  {searched}\n\n"
        "Resolution:\n"
        "  1. Download the dataset from Kaggle:\n"
        "     https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter\n"
        "  2. Place twcs.csv inside: Dataset/Raw/twcs/twcs.csv\n"
        "  3. Or set DATA_RAW_DIR in your .env to the correct path."
    )


def _select_primary_csv(csv_files: list[Path]) -> Path:
    """
    Choose the primary CSV from a list. Prefers 'twcs.csv' if present.
    Falls back to the largest CSV file.

    Args:
        csv_files: List of discovered CSV paths.

    Returns:
        The selected primary CSV path.
    """
    if not csv_files:
        raise ValueError("csv_files list must not be empty.")

    # Prefer the canonical dataset file
    for f in csv_files:
        if f.name.lower() == "twcs.csv":
            logger.info("Selected primary CSV: %s", f)
            return f

    # Fall back to largest file
    largest = max(csv_files, key=lambda p: p.stat().st_size)
    logger.warning(
        "twcs.csv not found. Falling back to largest CSV: %s (%.1f MB)",
        largest,
        largest.stat().st_size / (1024 ** 2),
    )
    return largest


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def discover_dataset() -> tuple[Path, list[Path]]:
    """
    Discover available CSV files in the raw data directory.

    Returns:
        Tuple of (primary_csv_path, all_csv_paths_found).

    Raises:
        FileNotFoundError: If no CSVs are found anywhere.
    """
    raw_dir = _resolve_raw_directory()
    all_csvs = _find_csv_files(raw_dir)

    logger.info("Found %d CSV file(s) in %s:", len(all_csvs), raw_dir)
    for csv in all_csvs:
        size_mb = csv.stat().st_size / (1024 ** 2)
        logger.info("  %s (%.1f MB)", csv.name, size_mb)

    primary = _select_primary_csv(all_csvs)
    return primary, all_csvs


def load_dataset(
    filepath: Optional[Path] = None,
    nrows: Optional[int] = None,
    chunksize: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load a CSV dataset into a pandas DataFrame.

    If filepath is None, auto-discovers the dataset.

    Args:
        filepath: Explicit path to CSV. If None, uses auto-discovery.
        nrows:    Number of rows to load (None = all rows).
        chunksize: If set, returns a TextFileReader for chunked iteration.
                   Note: when chunksize is set, returns concatenated df after
                   loading to keep the API consistent.

    Returns:
        A pandas DataFrame with the loaded data.

    Raises:
        FileNotFoundError: If the file cannot be found.
        ValueError: If the file is empty or cannot be parsed.
    """
    if filepath is None:
        filepath, _ = discover_dataset()

    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    size_mb = filepath.stat().st_size / (1024 ** 2)
    logger.info("Loading dataset: %s (%.1f MB)", filepath.name, size_mb)

    if nrows is not None:
        logger.info("Loading first %d rows", nrows)

    try:
        if chunksize and nrows is None:
            # For large files: read in chunks, concatenate
            logger.info("Using chunked reading (chunksize=%d)", chunksize)
            chunks = []
            total_rows = 0
            reader = pd.read_csv(
                filepath,
                chunksize=chunksize,
                low_memory=False,
                on_bad_lines="warn",
            )
            for chunk in reader:
                chunks.append(chunk)
                total_rows += len(chunk)
                logger.debug("Loaded %d rows so far...", total_rows)
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.read_csv(
                filepath,
                nrows=nrows,
                low_memory=False,
                on_bad_lines="warn",
            )
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Dataset file is empty: {filepath}") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"Failed to parse CSV file {filepath}: {exc}") from exc

    if df.empty:
        raise ValueError(f"Dataset loaded but contains no rows: {filepath}")

    logger.info(
        "Dataset loaded: %d rows × %d columns",
        len(df),
        len(df.columns),
    )
    return df


def get_dataset_metadata(
    df: pd.DataFrame,
    filepath: Path,
) -> DatasetMetadata:
    """
    Extract metadata from a loaded DataFrame.

    Args:
        df:       The loaded DataFrame.
        filepath: Path to the source CSV file.

    Returns:
        A DatasetMetadata dataclass instance.
    """
    missing_counts = {
        col: int(df[col].isna().sum())
        for col in df.columns
        if df[col].isna().sum() > 0
    }

    metadata = DatasetMetadata(
        filepath=filepath,
        filename=filepath.name,
        row_count=len(df),
        column_count=len(df.columns),
        columns=list(df.columns),
        dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
        file_size_mb=round(filepath.stat().st_size / (1024 ** 2), 2),
        missing_counts=missing_counts,
        sample_rows=min(5, len(df)),
    )

    logger.info("Metadata extracted for %s", metadata.filename)
    logger.info("  Rows: %d", metadata.row_count)
    logger.info("  Columns: %d — %s", metadata.column_count, metadata.columns)
    logger.info(
        "  Missing values in columns: %s",
        list(missing_counts.keys()) if missing_counts else "none",
    )

    return metadata
