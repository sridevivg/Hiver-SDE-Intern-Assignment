"""
SupportGraph AI — Unit Tests for the Data Loader

Tests:
  1. test_missing_raw_directory    — FileNotFoundError when raw dir doesn't exist
  2. test_empty_raw_directory      — FileNotFoundError when dir exists but has no CSVs
  3. test_csv_file_detection       — Detects CSV files inside a temp directory
  4. test_load_synthetic_csv       — Loads a small synthetic CSV fixture correctly
  5. test_get_dataset_metadata     — Metadata extracted correctly from a small fixture

These tests use temporary directories and synthetic fixtures.
They do NOT use the real Kaggle dataset.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.app.data.loader import (
    _find_csv_files,
    get_dataset_metadata,
    load_dataset,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
SYNTHETIC_ROWS = [
    {
        "tweet_id": 1001,
        "author_id": "AppleSupport",
        "inbound": False,
        "created_at": "Wed Oct 11 13:30:00 +0000 2017",
        "text": "@user123 We'd love to help. Please DM us your issue.",
        "response_tweet_id": "1002",
        "in_response_to_tweet_id": 1002,
    },
    {
        "tweet_id": 1002,
        "author_id": "105001",
        "inbound": True,
        "created_at": "Wed Oct 11 12:00:00 +0000 2017",
        "text": "@AppleSupport my battery drains so fast after the update!",
        "response_tweet_id": None,
        "in_response_to_tweet_id": None,
    },
    {
        "tweet_id": 1003,
        "author_id": "SpotifyCares",
        "inbound": False,
        "created_at": "Wed Oct 11 14:00:00 +0000 2017",
        "text": "@user456 Hi! What device are you using?",
        "response_tweet_id": "1004",
        "in_response_to_tweet_id": 1004,
    },
    {
        "tweet_id": 1004,
        "author_id": "105002",
        "inbound": True,
        "created_at": "Wed Oct 11 13:45:00 +0000 2017",
        "text": "@SpotifyCares songs keep skipping on my Android.",
        "response_tweet_id": None,
        "in_response_to_tweet_id": None,
    },
    {
        "tweet_id": 1005,
        "author_id": "AppleSupport",
        "inbound": False,
        "created_at": "Wed Oct 11 15:00:00 +0000 2017",
        "text": "@user789 Which iOS version are you running?",
        "response_tweet_id": None,
        "in_response_to_tweet_id": 1006,
    },
]


@pytest.fixture
def synthetic_csv(tmp_path: Path) -> Path:
    """Write a small synthetic CSV to a temp directory and return its path."""
    df = pd.DataFrame(SYNTHETIC_ROWS)
    csv_path = tmp_path / "test_dataset.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def empty_dir(tmp_path: Path) -> Path:
    """Return a temp directory with no CSV files."""
    subdir = tmp_path / "empty_raw"
    subdir.mkdir()
    return subdir


@pytest.fixture
def nonexistent_dir(tmp_path: Path) -> Path:
    """Return a path that does NOT exist."""
    return tmp_path / "does_not_exist"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestFindCsvFiles:
    """Tests for _find_csv_files internal helper."""

    def test_missing_directory_returns_empty_list(self, nonexistent_dir: Path) -> None:
        """
        _find_csv_files must return an empty list for a non-existent directory
        (it is the caller's responsibility to raise FileNotFoundError).
        """
        result = _find_csv_files(nonexistent_dir)
        assert result == []

    def test_empty_directory_returns_empty_list(self, empty_dir: Path) -> None:
        """_find_csv_files must return [] when no CSV files are present."""
        result = _find_csv_files(empty_dir)
        assert result == []

    def test_detects_csv_file(self, tmp_path: Path) -> None:
        """_find_csv_files must find CSV files inside a directory."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col_a,col_b\n1,2\n")
        result = _find_csv_files(tmp_path)
        assert len(result) == 1
        assert result[0].name == "data.csv"

    def test_detects_nested_csv_file(self, tmp_path: Path) -> None:
        """_find_csv_files must detect CSV files in subdirectories."""
        subdir = tmp_path / "nested" / "deep"
        subdir.mkdir(parents=True)
        csv_file = subdir / "nested.csv"
        csv_file.write_text("a,b\n1,2\n")
        result = _find_csv_files(tmp_path)
        assert len(result) == 1
        assert result[0] == csv_file

    def test_ignores_non_csv_files(self, tmp_path: Path) -> None:
        """_find_csv_files must not return non-CSV files."""
        (tmp_path / "readme.txt").write_text("hello")
        (tmp_path / "data.json").write_text("{}")
        result = _find_csv_files(tmp_path)
        assert result == []


class TestLoadDataset:
    """Tests for load_dataset public function."""

    def test_load_synthetic_csv(self, synthetic_csv: Path) -> None:
        """load_dataset must successfully load a small synthetic CSV."""
        df = load_dataset(filepath=synthetic_csv)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(SYNTHETIC_ROWS)
        assert "tweet_id" in df.columns
        assert "author_id" in df.columns
        assert "text" in df.columns

    def test_load_respects_nrows(self, synthetic_csv: Path) -> None:
        """load_dataset with nrows must return only that many rows."""
        df = load_dataset(filepath=synthetic_csv, nrows=2)
        assert len(df) == 2

    def test_load_nonexistent_file_raises(self, nonexistent_dir: Path) -> None:
        """load_dataset must raise FileNotFoundError for a missing file."""
        with pytest.raises(FileNotFoundError):
            load_dataset(filepath=nonexistent_dir / "missing.csv")

    def test_load_empty_csv_raises(self, tmp_path: Path) -> None:
        """load_dataset must raise ValueError for an empty CSV file."""
        empty_csv = tmp_path / "empty.csv"
        empty_csv.write_text("")  # Completely empty file
        with pytest.raises(ValueError):
            load_dataset(filepath=empty_csv)

    def test_loaded_df_not_empty(self, synthetic_csv: Path) -> None:
        """Loaded DataFrame must not be empty."""
        df = load_dataset(filepath=synthetic_csv)
        assert not df.empty


class TestGetDatasetMetadata:
    """Tests for get_dataset_metadata."""

    def test_metadata_row_count(self, synthetic_csv: Path) -> None:
        """Metadata row_count must match the DataFrame length."""
        df = load_dataset(filepath=synthetic_csv)
        meta = get_dataset_metadata(df, synthetic_csv)
        assert meta.row_count == len(SYNTHETIC_ROWS)

    def test_metadata_column_count(self, synthetic_csv: Path) -> None:
        """Metadata column_count must match the number of columns."""
        df = load_dataset(filepath=synthetic_csv)
        meta = get_dataset_metadata(df, synthetic_csv)
        expected_cols = len(pd.DataFrame(SYNTHETIC_ROWS).columns)
        assert meta.column_count == expected_cols

    def test_metadata_filename(self, synthetic_csv: Path) -> None:
        """Metadata filename must match the file name."""
        df = load_dataset(filepath=synthetic_csv)
        meta = get_dataset_metadata(df, synthetic_csv)
        assert meta.filename == "test_dataset.csv"

    def test_metadata_columns_list(self, synthetic_csv: Path) -> None:
        """Metadata columns list must contain all expected column names."""
        df = load_dataset(filepath=synthetic_csv)
        meta = get_dataset_metadata(df, synthetic_csv)
        for col in ["tweet_id", "author_id", "inbound", "text"]:
            assert col in meta.columns

    def test_metadata_dtypes_dict(self, synthetic_csv: Path) -> None:
        """Metadata dtypes must be a non-empty dict."""
        df = load_dataset(filepath=synthetic_csv)
        meta = get_dataset_metadata(df, synthetic_csv)
        assert isinstance(meta.dtypes, dict)
        assert len(meta.dtypes) == meta.column_count

    def test_metadata_file_size(self, synthetic_csv: Path) -> None:
        """Metadata file_size_mb must be a non-negative float, and the file must be non-empty."""
        df = load_dataset(filepath=synthetic_csv)
        meta = get_dataset_metadata(df, synthetic_csv)
        # file_size_mb may round to 0.0 for very small files; check raw file size instead
        assert isinstance(meta.file_size_mb, float)
        assert meta.file_size_mb >= 0.0
        assert synthetic_csv.stat().st_size > 0

    def test_metadata_missing_counts(self, tmp_path: Path) -> None:
        """Metadata missing_counts must identify columns with NaN values."""
        rows = [
            {"col_a": "hello", "col_b": None},
            {"col_a": "world", "col_b": "value"},
        ]
        csv_path = tmp_path / "with_nulls.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        df = load_dataset(filepath=csv_path)
        meta = get_dataset_metadata(df, csv_path)
        assert "col_b" in meta.missing_counts
        assert meta.missing_counts["col_b"] == 1
