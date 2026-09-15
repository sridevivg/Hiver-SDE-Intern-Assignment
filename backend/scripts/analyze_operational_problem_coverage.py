"""
SupportGraph AI — Corpus Operational Problem Coverage Analysis (Phase 10.2)

Analyzes the full 80,517-conversation non-golden historical dataset and the
77-record human-reviewed benchmark to produce reproducible distribution artifacts.

Outputs:
  - reports/phase_10_2/corpus_problem_family_distribution.json
  - reports/phase_10_2/benchmark_problem_family_coverage.json
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from backend.app.retrieval.problem_family_registry import (
    PROBLEM_FAMILY_DEFINITIONS,
    OperationalProblemFamily,
    ProblemFamilyDetector,
)

CONVERSATION_MESSAGES_PATH = Path("data/processed/conversation_messages.parquet")
GOLDEN_CSV_PATH = Path("data/golden/golden_set_human_review.csv")
OUTPUT_DIR = Path("reports/phase_10_2")


def analyze_corpus_and_benchmark() -> tuple[dict[str, Any], dict[str, Any]]:
    """Run full reproducible analysis across corpus and benchmark."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detector = ProblemFamilyDetector()

    # 1. Load Golden Benchmark to exclude IDs and extract 77 benchmark records
    golden_df = pd.read_csv(GOLDEN_CSV_PATH)
    all_golden_ids = set(golden_df["conversation_id"].dropna().astype(str).unique())

    # 77 human-reviewed benchmark records
    hr_bench = golden_df[
        golden_df["annotation_status"].isin(["reviewed", "overridden_ai_suggestion"])
        & (golden_df["annotation_label"].fillna("").astype(str).str.strip() != "")
    ].copy()

    # 2. Load and Filter Historical Corpus (Excluding all golden IDs)
    raw_df = pd.read_parquet(CONVERSATION_MESSAGES_PATH)
    cust_df = raw_df[
        (raw_df["role"] == "customer")
        & (~raw_df["conversation_id"].astype(str).isin(all_golden_ids))
    ].copy()

    # Get first customer message per conversation
    cust_starters = (
        cust_df.sort_values("depth")
        .groupby("conversation_id")
        .first()
        .reset_index()
    )
    cust_starters = cust_starters[cust_starters["text"].fillna("").str.strip().str.len() >= 10].copy()

    total_historical = len(cust_starters)

    # 3. Classify Historical Corpus by Operational Problem Family
    corpus_counts: dict[str, int] = defaultdict(int)
    corpus_examples: dict[str, list[str]] = defaultdict(list)

    for _, row in cust_starters.iterrows():
        text = str(row["text"]).strip()
        fam, _, _ = detector.detect_family(text)
        fam_str = fam.value
        corpus_counts[fam_str] += 1
        if len(corpus_examples[fam_str]) < 3:
            corpus_examples[fam_str].append(text[:120])

    corpus_dist: dict[str, Any] = {
        "total_historical_cases": total_historical,
        "golden_conversations_excluded": len(all_golden_ids),
        "leakage_count": len(set(cust_starters["conversation_id"].astype(str)).intersection(all_golden_ids)),
        "family_breakdown": {},
    }

    for fam in OperationalProblemFamily:
        cnt = corpus_counts.get(fam.value, 0)
        pct = round(cnt / total_historical * 100, 2) if total_historical else 0.0
        defn = PROBLEM_FAMILY_DEFINITIONS[fam]
        corpus_dist["family_breakdown"][fam.value] = {
            "display_name": defn.display_name,
            "count": cnt,
            "percentage": pct,
            "example_patterns": defn.patterns[:2],
            "sample_cases": corpus_examples.get(fam.value, []),
        }

    # 4. Classify Benchmark (77 Records) by Operational Problem Family
    bench_counts: dict[str, int] = defaultdict(int)
    bench_records: list[dict[str, Any]] = []

    for _, row in hr_bench.iterrows():
        text = str(row["customer_message"]).strip()
        gid = str(row["golden_id"])
        annot_label = str(row["annotation_label"])
        fam, sec, conf = detector.detect_family(text, candidate_intent=annot_label)
        fam_str = fam.value
        bench_counts[fam_str] += 1
        bench_records.append({
            "golden_id": gid,
            "annotation_label": annot_label,
            "primary_problem_family": fam_str,
            "family_confidence": conf,
            "customer_message": text[:100],
        })

    benchmark_dist: dict[str, Any] = {
        "total_benchmark_records": len(hr_bench),
        "family_breakdown": {},
        "records": bench_records,
    }

    for fam in OperationalProblemFamily:
        cnt = bench_counts.get(fam.value, 0)
        pct = round(cnt / len(hr_bench) * 100, 2) if len(hr_bench) else 0.0
        benchmark_dist["family_breakdown"][fam.value] = {
            "count": cnt,
            "percentage": pct,
        }

    # Save artifacts
    corpus_file = OUTPUT_DIR / "corpus_problem_family_distribution.json"
    bench_file = OUTPUT_DIR / "benchmark_problem_family_coverage.json"

    with open(corpus_file, "w") as f:
        json.dump(corpus_dist, f, indent=2)
    with open(bench_file, "w") as f:
        json.dump(benchmark_dist, f, indent=2)

    return corpus_dist, benchmark_dist


if __name__ == "__main__":
    c_dist, b_dist = analyze_corpus_and_benchmark()
    print("Corpus analysis complete!")
    print(f"Total historical conversations: {c_dist['total_historical_cases']}")
    print(f"Total benchmark records: {b_dist['total_benchmark_records']}")
    print("\nTop Historical Problem Families:")
    sorted_fams = sorted(c_dist["family_breakdown"].items(), key=lambda x: x[1]["count"], reverse=True)
    for fam, stats in sorted_fams[:10]:
        print(f"  {fam:<30} {stats['count']:>6} ({stats['percentage']:>5.2f}%)")
