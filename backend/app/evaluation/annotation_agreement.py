"""
SupportGraph AI — Inter-Annotator Agreement Module (Phase 5)

Computes inter-rater reliability metrics between two human annotators:
- Raw percentage agreement
- Cohen's Kappa coefficient (with Landis & Koch interpretation)
- Disagreement confusion matrix
- Detailed disagreement example extraction

Scientific Honesty Rule:
If only one annotator dataset exists, this module does NOT fabricate a second
annotator or simulate random ratings. It explicitly reports:
`INTER_ANNOTATOR_AGREEMENT_NOT_AVAILABLE`.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

try:
    from app.core.logging import get_logger
except ModuleNotFoundError:
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]

logger = get_logger(__name__)

STATUS_AVAILABLE = "AVAILABLE"
STATUS_NOT_AVAILABLE = "INTER_ANNOTATOR_AGREEMENT_NOT_AVAILABLE"


@dataclass
class AgreementReport:
    """Structured report on inter-annotator agreement."""
    status: str
    total_compared: int = 0
    agreed_count: int = 0
    raw_agreement: float = 0.0
    cohen_kappa: Optional[float] = None
    kappa_interpretation: str = ""
    disagreement_count: int = 0
    disagreement_matrix: dict[str, dict[str, int]] = field(default_factory=dict)
    disagreements: list[dict[str, Any]] = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def interpret_cohen_kappa(kappa: Optional[float]) -> str:
    """
    Interpret Cohen's Kappa score according to the Landis & Koch (1977) benchmark.
    """
    if kappa is None:
        return "N/A"
    if kappa < 0.0:
        return "Poor (less than chance agreement)"
    elif kappa <= 0.20:
        return "Slight agreement"
    elif kappa <= 0.40:
        return "Fair agreement"
    elif kappa <= 0.60:
        return "Moderate agreement"
    elif kappa <= 0.80:
        return "Substantial agreement"
    else:
        return "Almost perfect agreement"


def compute_cohen_kappa(
    labels_a: list[str],
    labels_b: list[str],
) -> float:
    """
    Compute Cohen's Kappa for two equal-length lists of categorical labels.
    Formula: kappa = (Po - Pe) / (1 - Pe)
    """
    if len(labels_a) != len(labels_b):
        raise ValueError(
            f"Label list lengths do not match: {len(labels_a)} vs {len(labels_b)}"
        )
    n = len(labels_a)
    if n == 0:
        return 0.0

    # All unique categories across both raters
    categories = sorted(list(set(labels_a) | set(labels_b)))
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    num_cats = len(categories)

    # Contingency matrix
    matrix = [[0] * num_cats for _ in range(num_cats)]
    for a, b in zip(labels_a, labels_b):
        matrix[cat_to_idx[a]][cat_to_idx[b]] += 1

    # Observed agreement Po
    observed_agreed = sum(matrix[i][i] for i in range(num_cats))
    p_o = observed_agreed / n

    # Expected agreement Pe
    row_sums = [sum(matrix[i][j] for j in range(num_cats)) for i in range(num_cats)]
    col_sums = [sum(matrix[i][j] for i in range(num_cats)) for j in range(num_cats)]
    p_e = sum((row_sums[i] * col_sums[i]) for i in range(num_cats)) / (n * n)

    if abs(1.0 - p_e) < 1e-9:
        # Perfect agreement by chance or degenerate distribution
        return 1.0 if abs(p_o - 1.0) < 1e-9 else 0.0

    kappa = (p_o - p_e) / (1.0 - p_e)
    return float(round(kappa, 4))


def compute_inter_annotator_agreement(
    df_a: Optional[pd.DataFrame],
    df_b: Optional[pd.DataFrame],
    match_key: str = "golden_id",
    label_col: str = "annotation_label",
) -> AgreementReport:
    """
    Compute inter-annotator agreement metrics between two annotated dataframes.

    If df_b is None, empty, or lacks records, returns
    `INTER_ANNOTATOR_AGREEMENT_NOT_AVAILABLE` without fabricating data.
    """
    if df_a is None or df_b is None:
        logger.info("One or both annotation sources are missing. Reporting agreement unavailable.")
        return AgreementReport(
            status=STATUS_NOT_AVAILABLE,
            message="INTER_ANNOTATOR_AGREEMENT_NOT_AVAILABLE: Second annotator dataset was not provided.",
        )

    if df_a.empty or df_b.empty:
        logger.info("One or both annotation sources are empty. Reporting agreement unavailable.")
        return AgreementReport(
            status=STATUS_NOT_AVAILABLE,
            message="INTER_ANNOTATOR_AGREEMENT_NOT_AVAILABLE: One or both annotation datasets are empty.",
        )

    if match_key not in df_a.columns or match_key not in df_b.columns:
        raise ValueError(f"Match key '{match_key}' not found in both dataframes.")

    if label_col not in df_a.columns or label_col not in df_b.columns:
        raise ValueError(f"Label column '{label_col}' not found in both dataframes.")

    # Merge by match_key
    merged = pd.merge(
        df_a,
        df_b,
        on=match_key,
        suffixes=("_annotator_a", "_annotator_b"),
        how="inner",
    )

    if merged.empty:
        return AgreementReport(
            status=STATUS_NOT_AVAILABLE,
            message="INTER_ANNOTATOR_AGREEMENT_NOT_AVAILABLE: Zero overlapping records found between annotators.",
        )

    labels_a = (
        merged[f"{label_col}_annotator_a"]
        .fillna("")
        .astype(str)
        .str.strip()
        .tolist()
    )
    labels_b = (
        merged[f"{label_col}_annotator_b"]
        .fillna("")
        .astype(str)
        .str.strip()
        .tolist()
    )

    # Filter out records where either annotator left the label blank
    valid_pairs = [
        (idx, la, lb)
        for idx, (la, lb) in enumerate(zip(labels_a, labels_b))
        if la != "" and lb != ""
    ]

    if not valid_pairs:
        return AgreementReport(
            status=STATUS_NOT_AVAILABLE,
            message="INTER_ANNOTATOR_AGREEMENT_NOT_AVAILABLE: No records have complete labels from both annotators.",
        )

    total_valid = len(valid_pairs)
    filtered_labels_a = [p[1] for p in valid_pairs]
    filtered_labels_b = [p[2] for p in valid_pairs]

    # Agreement counts
    agreed_count = sum(1 for la, lb in zip(filtered_labels_a, filtered_labels_b) if la == lb)
    raw_agreement = round((agreed_count / total_valid) * 100.0, 2)
    disagreement_count = total_valid - agreed_count

    # Cohen's Kappa
    kappa = compute_cohen_kappa(filtered_labels_a, filtered_labels_b)
    interpretation = interpret_cohen_kappa(kappa)

    # Disagreement matrix (crosstab)
    categories = sorted(list(set(filtered_labels_a) | set(filtered_labels_b)))
    ct = pd.crosstab(
        pd.Series(filtered_labels_a, name="Annotator_A"),
        pd.Series(filtered_labels_b, name="Annotator_B"),
    )
    # Reindex for consistent dimensions
    ct = ct.reindex(index=categories, columns=categories, fill_value=0)
    matrix_dict = {cat: {col: int(ct.loc[cat, col]) for col in categories} for cat in categories}

    # Extract disagreement examples
    disagreements: list[dict[str, Any]] = []
    for orig_idx, la, lb in valid_pairs:
        if la != lb:
            row = merged.iloc[orig_idx]
            customer_msg = (
                row.get("customer_message_annotator_a")
                or row.get("customer_message_annotator_b")
                or row.get("customer_message", "")
            )
            annotator_a_name = str(row.get("annotator_annotator_a", "Annotator A"))
            annotator_b_name = str(row.get("annotator_annotator_b", "Annotator B"))
            notes_a = str(row.get("notes_annotator_a", ""))
            notes_b = str(row.get("notes_annotator_b", ""))

            disagreements.append({
                "golden_id": row[match_key],
                "tweet_id": row.get("tweet_id_annotator_a", row.get("tweet_id", "")),
                "customer_message": customer_msg,
                "label_annotator_a": la,
                "label_annotator_b": lb,
                "annotator_a": annotator_a_name,
                "annotator_b": annotator_b_name,
                "notes_a": notes_a,
                "notes_b": notes_b,
            })

    return AgreementReport(
        status=STATUS_AVAILABLE,
        total_compared=total_valid,
        agreed_count=agreed_count,
        raw_agreement=raw_agreement,
        cohen_kappa=kappa,
        kappa_interpretation=interpretation,
        disagreement_count=disagreement_count,
        disagreement_matrix=matrix_dict,
        disagreements=disagreements,
        message=f"Agreement calculated over {total_valid} items: raw={raw_agreement}%, kappa={kappa} ({interpretation}).",
    )
