"""
SupportGraph AI — Golden Set Freeze Workflow (Phase 5)

Freezes a validated human-annotated dataset into an immutable versioned release:
- Strict validation: 100% completion (zero empty labels), valid taxonomy labels, zero duplicates.
- Generates versioned CSV artifact: data/golden/golden_set_{version}.csv
- Generates cryptographic SHA256 checksum for tamper evidence.
- Writes metadata manifest: data/golden/golden_set_{version}_manifest.json
- Protects against accidental overwrites: requires explicit --force-new-version.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

try:
    from app.core.logging import get_logger
    from app.evaluation.label_validation import (
        GoldenValidationError,
        ValidationReport,
        validate_golden_set,
    )
    from app.nlp.taxonomy_finalization import (
        DEFAULT_CANDIDATE_TAXONOMY_PATH,
        load_candidate_taxonomy,
    )
except ModuleNotFoundError:
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]
    from backend.app.evaluation.label_validation import (  # type: ignore[no-redef]
        GoldenValidationError,
        ValidationReport,
        validate_golden_set,
    )
    from backend.app.nlp.taxonomy_finalization import (  # type: ignore[no-redef]
        DEFAULT_CANDIDATE_TAXONOMY_PATH,
        load_candidate_taxonomy,
    )

logger = get_logger(__name__)


def compute_file_sha256(file_path: Path | str) -> str:
    """Calculate the cryptographic SHA256 hex digest of a file."""
    path = Path(file_path)
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def freeze_golden_dataset(
    input_csv_path: Path | str,
    output_dir: Path | str = "data/golden",
    version: str = "v1",
    taxonomy_path: Optional[Path | str] = None,
    force_new_version: bool = False,
    allowed_labels: Optional[list[str] | set[str]] = None,
) -> tuple[Path, Path, dict[str, Any]]:
    """
    Freeze a human-annotated dataset into an immutable versioned golden evaluation release.

    Args:
        input_csv_path: Path to completed annotation CSV.
        output_dir: Output directory for golden artifacts.
        version: Version string (e.g. 'v1', 'v2').
        taxonomy_path: Optional path to final taxonomy JSON candidate.
        force_new_version: If True, allows overwriting an existing version file.
        allowed_labels: Optional explicit list of allowed labels (overrides taxonomy_path).

    Returns:
        Tuple of (frozen_csv_path, manifest_json_path, manifest_dict).

    Raises:
        FileExistsError: If versioned file already exists and force_new_version is False.
        GoldenValidationError: If validation fails (empty labels, invalid categories, dupes).
    """
    in_path = Path(input_csv_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Input annotation file not found: {in_path}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_filename = f"golden_set_{version}.csv"
    manifest_filename = f"golden_set_{version}_manifest.json"
    target_csv = out_dir / csv_filename
    target_manifest = out_dir / manifest_filename

    # Immutability check
    if target_csv.exists() and not force_new_version:
        raise FileExistsError(
            f"Golden set '{version}' already exists at {target_csv}. "
            "To prevent accidental tampering with evaluation sets, overwriting is blocked. "
            "Use a new version (e.g. '--version v2') or supply '--force-new-version' explicitly."
        )

    # Resolve allowed taxonomy labels
    tax_version = "candidate_v1.0"
    if allowed_labels is None:
        tax_file = Path(taxonomy_path) if taxonomy_path else Path(DEFAULT_CANDIDATE_TAXONOMY_PATH)
        if tax_file.exists():
            tax_obj = load_candidate_taxonomy(tax_file)
            allowed_labels = set(tax_obj.intent_names)
            tax_version = tax_obj.version
        else:
            # Fallback to default candidate taxonomy
            from app.nlp.taxonomy_finalization import create_initial_candidate_taxonomy
            fallback_tax = create_initial_candidate_taxonomy()
            allowed_labels = set(fallback_tax.intent_names)
            tax_version = fallback_tax.version

    # Load input data
    df = pd.read_csv(in_path)
    logger.info(f"Loaded {len(df)} rows from {in_path} for freezing.")

    # Strict validation: require 100% completion (zero empty labels)
    report: ValidationReport = validate_golden_set(
        df=df,
        allowed_labels=allowed_labels,
        require_complete=True,
        allow_unclear=True,
    )

    if not report.is_valid:
        raise GoldenValidationError(report.errors)

    # Write frozen CSV with quoting
    df.to_csv(target_csv, index=False, quoting=csv.QUOTE_NONNUMERIC)
    logger.info(f"Frozen golden set saved to {target_csv}")

    # Compute SHA256
    file_checksum = compute_file_sha256(target_csv)

    # Assemble manifest
    manifest_data: dict[str, Any] = {
        "version": version,
        "creation_timestamp": datetime.now(timezone.utc).isoformat(),
        "row_count": len(df),
        "source_file": str(in_path),
        "sha256_checksum": file_checksum,
        "taxonomy_version": tax_version,
        "allowed_intents": sorted(list(allowed_labels)),
        "intent_distribution": report.label_distribution,
        "validation_results": {
            "is_valid": report.is_valid,
            "errors": report.errors,
            "warnings": report.warnings,
            "completion_stats": report.completion_stats,
        },
    }

    with open(target_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    logger.info(f"Golden set manifest written to {target_manifest}")

    return target_csv, target_manifest, manifest_data
