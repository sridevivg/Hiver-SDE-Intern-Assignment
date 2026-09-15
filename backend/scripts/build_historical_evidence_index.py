"""
SupportGraph AI — Build Historical Evidence Index (Phase 10.2)

Builds and serializes the leakage-safe historical evidence index,
verifies golden benchmark SHA-256 and zero-leakage guarantee,
and outputs reports/phase_10_2/leakage_verification.json.
"""
from __future__ import annotations

import json
from pathlib import Path

from backend.app.retrieval.historical_corpus_index import (
    GOLDEN_CSV_PATH,
    HistoricalCorpusIndex,
)

OUTPUT_DIR = Path("reports/phase_10_2")


def build_and_verify_index() -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Building full historical evidence index...")
    index = HistoricalCorpusIndex(auto_load=False)
    index.load_or_build(force_rebuild=True)

    print(f"Indexed {len(index.cases)} clean historical support cases.")

    print("Verifying golden benchmark leakage safety...")
    leakage_report = index.verify_leakage(GOLDEN_CSV_PATH)

    output_path = OUTPUT_DIR / "leakage_verification.json"
    with open(output_path, "w") as f:
        json.dump(leakage_report, f, indent=2)

    print("Verification Report:")
    print(f"  Leakage Free: {leakage_report['is_leakage_free']}")
    print(f"  Overlap Count: {leakage_report['overlap_count']}")
    print(f"  Total Indexed Cases: {leakage_report['total_indexed_cases']}")
    print(f"  Golden SHA-256: {leakage_report['golden_sha256_pre']}")
    print(f"  Golden Immutable: {leakage_report['golden_dataset_immutable']}")
    print(f"Report saved to: {output_path}")

    return leakage_report


if __name__ == "__main__":
    build_and_verify_index()
