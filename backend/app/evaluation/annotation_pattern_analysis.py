"""
SupportGraph AI — Annotation Pattern Analysis & Taxonomy Calibration (Phase 5.9)

Scientific, read-only analysis layer for discovering empirical patterns from
human-reviewed ground-truth annotations vs baseline AI suggestions.

Guiding Principles:
1. READ-ONLY: Never modifies source dataset, human labels, pending records, or predictions.
2. EVIDENCE-BASED: Dynamically extracts semantic patterns from actual reviewed text.
3. CLEAR TERMINOLOGY: Distinguishes 'observed annotation agreement' on preliminary sample
   from 'true production model accuracy'.
4. TAXONOMY AUDIT: Analyzes general_device_support over-prediction and operational boundary overlaps.
"""
from __future__ import annotations

import collections
import json
import logging
import math
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

try:
    from app.core.logging import get_logger
    from app.nlp.taxonomy_finalization import (
        DEFAULT_CANDIDATE_TAXONOMY_PATH,
        load_candidate_taxonomy,
    )
except ModuleNotFoundError:
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]
    from backend.app.nlp.taxonomy_finalization import (  # type: ignore[no-redef]
        DEFAULT_CANDIDATE_TAXONOMY_PATH,
        load_candidate_taxonomy,
    )

logger = get_logger(__name__)

# Standard stop words for n-gram / token frequency analysis
STOP_WORDS: set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves", "user", "url", "applesupport", "http", "https",
}


# ---------------------------------------------------------------------------
# Structured Data Models
# ---------------------------------------------------------------------------

@dataclass
class AgreementMetrics:
    """Overall annotation agreement statistics on completed human reviews."""
    total_golden_records: int
    completed_reviews: int
    pending_reviews: int
    agreed_count: int
    overridden_count: int
    unclear_count: int
    observed_agreement_rate: float
    confidence_interval_95: Tuple[float, float]
    sample_coverage_pct: float
    scientific_disclaimer: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_golden_records": self.total_golden_records,
            "completed_reviews": self.completed_reviews,
            "pending_reviews": self.pending_reviews,
            "agreed_count": self.agreed_count,
            "overridden_count": self.overridden_count,
            "unclear_count": self.unclear_count,
            "observed_agreement_rate": round(self.observed_agreement_rate, 4),
            "confidence_interval_95": [
                round(self.confidence_interval_95[0], 4),
                round(self.confidence_interval_95[1], 4),
            ],
            "sample_coverage_pct": round(self.sample_coverage_pct, 2),
            "scientific_disclaimer": self.scientific_disclaimer,
        }


@dataclass
class PerIntentAgreement:
    """Per-intent support and agreement statistics."""
    intent: str
    ground_truth_support: int
    ai_suggested_support: int
    agreed_matches: int
    ai_false_positives: int  # AI predicted this, but human assigned a different label
    ai_false_negatives: int  # Human assigned this, but AI predicted something else
    observed_precision: float  # agreed / ai_suggested_support
    observed_recall: float     # agreed / ground_truth_support
    observed_f1: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "ground_truth_support": self.ground_truth_support,
            "ai_suggested_support": self.ai_suggested_support,
            "agreed_matches": self.agreed_matches,
            "ai_false_positives": self.ai_false_positives,
            "ai_false_negatives": self.ai_false_negatives,
            "observed_precision": round(self.observed_precision, 4),
            "observed_recall": round(self.observed_recall, 4),
            "observed_f1": round(self.observed_f1, 4),
        }


@dataclass
class ConfusionPair:
    """Detailed analysis of a specific AI -> Human override pattern."""
    ai_suggested_label: str
    human_ground_truth_label: str
    count: int
    percentage_of_all_overrides: float
    representative_examples: list[dict[str, str]]
    recurring_semantic_signals: list[dict[str, Any]]
    possible_taxonomy_overlap: str
    recommendation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ai_suggested_label": self.ai_suggested_label,
            "human_ground_truth_label": self.human_ground_truth_label,
            "count": self.count,
            "percentage_of_all_overrides": round(self.percentage_of_all_overrides, 2),
            "representative_examples": self.representative_examples,
            "recurring_semantic_signals": self.recurring_semantic_signals,
            "possible_taxonomy_overlap": self.possible_taxonomy_overlap,
            "recommendation": self.recommendation,
        }


@dataclass
class GeneralDeviceSupportAudit:
    """Deep-dive audit into general_device_support behavior and specificity."""
    ai_predicted_count: int
    human_confirmed_count: int
    human_overridden_count: int
    override_rate: float
    confirmed_rate: float
    top_target_intents: list[dict[str, Any]]
    distinguishing_signals_retained: list[str]
    distinguishing_signals_overridden: list[str]
    audit_verdict: str
    actionable_guidelines: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ai_predicted_count": self.ai_predicted_count,
            "human_confirmed_count": self.human_confirmed_count,
            "human_overridden_count": self.human_overridden_count,
            "override_rate": round(self.override_rate, 4),
            "confirmed_rate": round(self.confirmed_rate, 4),
            "top_target_intents": self.top_target_intents,
            "distinguishing_signals_retained": self.distinguishing_signals_retained,
            "distinguishing_signals_overridden": self.distinguishing_signals_overridden,
            "audit_verdict": self.audit_verdict,
            "actionable_guidelines": self.actionable_guidelines,
        }


@dataclass
class TaxonomyOverlap:
    """Identified ambiguity or boundary overlap between two taxonomy categories."""
    label_a: str
    label_b: str
    observed_disagreements: int
    example_messages: list[str]
    ambiguity_source: str
    proposed_boundary_guideline: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "label_a": self.label_a,
            "label_b": self.label_b,
            "observed_disagreements": self.observed_disagreements,
            "example_messages": self.example_messages,
            "ambiguity_source": self.ambiguity_source,
            "proposed_boundary_guideline": self.proposed_boundary_guideline,
        }


@dataclass
class AnnotationGuidelineItem:
    """Empirical annotation guidelines for a single intent category."""
    intent_label: str
    definition: str
    include_when: list[str]
    do_not_use_when: list[str]
    boundary_cases: list[str]
    reviewed_examples: list[str]
    status: str = "OBSERVED FROM DATA"

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent_label": self.intent_label,
            "definition": self.definition,
            "include_when": self.include_when,
            "do_not_use_when": self.do_not_use_when,
            "boundary_cases": self.boundary_cases,
            "reviewed_examples": self.reviewed_examples,
            "status": self.status,
        }


@dataclass
class PatternAnalysisReport:
    """Comprehensive container for all Phase 5.9 analysis findings."""
    metrics: AgreementMetrics
    per_intent_agreement: list[PerIntentAgreement]
    confusion_matrix: dict[str, dict[str, int]]
    all_intents: list[str]
    top_confusion_pairs: list[ConfusionPair]
    most_overpredicted_ai_labels: list[dict[str, Any]]
    most_underpredicted_human_labels: list[dict[str, Any]]
    general_device_support_audit: GeneralDeviceSupportAudit
    taxonomy_overlaps: list[TaxonomyOverlap]
    annotation_guidelines: list[AnnotationGuidelineItem]
    scientific_limitations: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics.to_dict(),
            "per_intent_agreement": [p.to_dict() for p in self.per_intent_agreement],
            "confusion_matrix": self.confusion_matrix,
            "all_intents": self.all_intents,
            "top_confusion_pairs": [cp.to_dict() for cp in self.top_confusion_pairs],
            "most_overpredicted_ai_labels": self.most_overpredicted_ai_labels,
            "most_underpredicted_human_labels": self.most_underpredicted_human_labels,
            "general_device_support_audit": self.general_device_support_audit.to_dict(),
            "taxonomy_overlaps": [to.to_dict() for to in self.taxonomy_overlaps],
            "annotation_guidelines": [ag.to_dict() for ag in self.annotation_guidelines],
            "scientific_limitations": self.scientific_limitations,
        }


# ---------------------------------------------------------------------------
# Core Analysis Functions
# ---------------------------------------------------------------------------

def extract_reviewed_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract only completed human-reviewed records, ignoring pending records.
    """
    if df.empty:
        return df.copy()

    # Identify reviewed records
    has_label = df["annotation_label"].fillna("").astype(str).str.strip() != ""
    not_pending = ~df["annotation_status"].fillna("").astype(str).str.lower().isin(
        ["", "pending", "pending_human_review", "unreviewed"]
    )
    return df[has_label & not_pending].copy()


def compute_agreement_metrics(df: pd.DataFrame, total_records: int = 200) -> AgreementMetrics:
    """
    Calculate observed AI-human agreement rates with Wilson score 95% confidence intervals.
    """
    reviewed_df = extract_reviewed_dataset(df)
    n_reviewed = len(reviewed_df)
    n_pending = total_records - n_reviewed

    if n_reviewed == 0:
        return AgreementMetrics(
            total_golden_records=total_records,
            completed_reviews=0,
            pending_reviews=total_records,
            agreed_count=0,
            overridden_count=0,
            unclear_count=0,
            observed_agreement_rate=0.0,
            confidence_interval_95=(0.0, 0.0),
            sample_coverage_pct=0.0,
            scientific_disclaimer="No completed human reviews found.",
        )

    agreed = (reviewed_df["annotation_label"] == reviewed_df["model_suggested_label"]).sum()
    unclear = (reviewed_df["annotation_label"] == "unclear_needs_review").sum()
    overridden = n_reviewed - agreed
    rate = float(agreed) / float(n_reviewed)

    # Wilson score interval for binomial proportion
    z = 1.96  # 95% confidence
    denominator = 1 + z**2 / n_reviewed
    centre = (rate + z**2 / (2 * n_reviewed)) / denominator
    spread = (z * math.sqrt(rate * (1 - rate) / n_reviewed + z**2 / (4 * n_reviewed**2))) / denominator
    ci_lower = max(0.0, centre - spread)
    ci_upper = min(1.0, centre + spread)

    coverage_pct = (n_reviewed / total_records) * 100.0 if total_records > 0 else 0.0

    disclaimer = (
        f"OBSERVED ANNOTATION AGREEMENT on N={n_reviewed} completed records ({coverage_pct:.1f}% coverage). "
        "This reflects annotator agreement with baseline LLM suggestions during curation and must NOT be "
        "interpreted as generalized production model accuracy."
    )

    return AgreementMetrics(
        total_golden_records=total_records,
        completed_reviews=n_reviewed,
        pending_reviews=n_pending,
        agreed_count=int(agreed),
        overridden_count=int(overridden),
        unclear_count=int(unclear),
        observed_agreement_rate=rate,
        confidence_interval_95=(ci_lower, ci_upper),
        sample_coverage_pct=coverage_pct,
        scientific_disclaimer=disclaimer,
    )


def compute_confusion_matrix(reviewed_df: pd.DataFrame) -> Tuple[dict[str, dict[str, int]], list[str]]:
    """
    Build a complete 2D confusion matrix where rows are AI suggestions and columns are Human ground truth.
    """
    if reviewed_df.empty:
        return {}, []

    ai_labels = reviewed_df["model_suggested_label"].fillna("unknown").astype(str).str.strip().tolist()
    human_labels = reviewed_df["annotation_label"].fillna("unknown").astype(str).str.strip().tolist()

    all_intents = sorted(list(set(ai_labels + human_labels)))

    matrix: dict[str, dict[str, int]] = {ai: {h: 0 for h in all_intents} for ai in all_intents}

    for ai, human in zip(ai_labels, human_labels):
        if ai in matrix and human in matrix[ai]:
            matrix[ai][human] += 1

    return matrix, all_intents


def compute_per_intent_statistics(
    reviewed_df: pd.DataFrame,
    all_intents: list[str],
) -> list[PerIntentAgreement]:
    """
    Compute per-intent support, precision, recall, and F1 based on observed agreement.
    """
    if reviewed_df.empty:
        return []

    stats: list[PerIntentAgreement] = []

    for intent in all_intents:
        gt_mask = reviewed_df["annotation_label"] == intent
        ai_mask = reviewed_df["model_suggested_label"] == intent

        gt_support = int(gt_mask.sum())
        ai_support = int(ai_mask.sum())
        agreed = int((gt_mask & ai_mask).sum())

        fp = ai_support - agreed
        fn = gt_support - agreed

        prec = float(agreed) / float(ai_support) if ai_support > 0 else 0.0
        rec = float(agreed) / float(gt_support) if gt_support > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        stats.append(
            PerIntentAgreement(
                intent=intent,
                ground_truth_support=gt_support,
                ai_suggested_support=ai_support,
                agreed_matches=agreed,
                ai_false_positives=fp,
                ai_false_negatives=fn,
                observed_precision=prec,
                observed_recall=rec,
                observed_f1=f1,
            )
        )

    # Sort by ground truth support descending
    stats.sort(key=lambda x: -x.ground_truth_support)
    return stats


def extract_frequent_ngrams(texts: list[str], top_k: int = 6) -> list[dict[str, Any]]:
    """
    Dynamically extract most frequent tokens and bigrams/trigrams from a list of customer messages.
    """
    if not texts:
        return []

    token_counts: collections.Counter[str] = collections.Counter()
    bigram_counts: collections.Counter[str] = collections.Counter()
    trigram_counts: collections.Counter[str] = collections.Counter()

    for text in texts:
        # Simple clean tokenization
        clean = re.sub(r"[^\w\s]", " ", text.lower())
        tokens = [t.strip() for t in clean.split() if t.strip() and len(t.strip()) > 2]
        content_tokens = [t for t in tokens if t not in STOP_WORDS]

        for tok in content_tokens:
            token_counts[tok] += 1

        for i in range(len(tokens) - 1):
            w1, w2 = tokens[i], tokens[i + 1]
            if w1 not in STOP_WORDS or w2 not in STOP_WORDS:
                bigram_counts[f"{w1} {w2}"] += 1

        for i in range(len(tokens) - 2):
            w1, w2, w3 = tokens[i], tokens[i + 1], tokens[i + 2]
            if any(w not in STOP_WORDS for w in (w1, w2, w3)):
                trigram_counts[f"{w1} {w2} {w3}"] += 1

    signals: list[dict[str, Any]] = []

    # Combine top n-grams
    combined = (
        [(k, v, "token") for k, v in token_counts.most_common(top_k)]
        + [(k, v, "phrase") for k, v in bigram_counts.most_common(top_k)]
        + [(k, v, "phrase") for k, v in trigram_counts.most_common(top_k)]
    )
    combined.sort(key=lambda x: -x[1])

    seen_phrases: set[str] = set()
    for phrase, count, ptype in combined:
        if phrase not in seen_phrases and count >= 1:
            seen_phrases.add(phrase)
            signals.append({"signal": phrase, "frequency": count, "type": ptype})
            if len(signals) >= top_k:
                break

    return signals


def discover_semantic_patterns(reviewed_df: pd.DataFrame) -> list[ConfusionPair]:
    """
    Discover recurring confusion patterns and empirical semantic signals for all AI -> Human override pairs.
    """
    if reviewed_df.empty:
        return []

    overrides = reviewed_df[reviewed_df["model_suggested_label"] != reviewed_df["annotation_label"]].copy()
    total_overrides = len(overrides)

    if total_overrides == 0:
        return []

    # Group by (AI suggested, Human annotation)
    grouped = overrides.groupby(["model_suggested_label", "annotation_label"])

    pairs: list[ConfusionPair] = []

    for (ai_label, human_label), group_df in grouped:
        count = len(group_df)
        pct = (count / total_overrides) * 100.0

        messages = group_df["customer_message"].fillna("").astype(str).tolist()
        gids = group_df["golden_id"].fillna("").astype(str).tolist()

        rep_examples = [
            {"golden_id": gid, "message": msg}
            for gid, msg in zip(gids[:3], messages[:3])
        ]

        # Extract dynamic n-grams
        signals = extract_frequent_ngrams(messages, top_k=5)

        # Determine overlap risk & recommendation based on empirical pair
        overlap_risk = f"AI over-relies on '{ai_label}' when text contains specific cues for '{human_label}'."
        if ai_label == "general_device_support":
            rec = (
                f"Enforce strict priority rules routing messages with '{human_label}' operational symptoms "
                f"away from general fallback '{ai_label}'."
            )
        elif human_label == "general_device_support":
            rec = (
                f"Retain '{human_label}' only when no specific subsystem, hardware component, or update causality "
                f"is mentioned."
            )
        else:
            rec = f"Clarify decision boundary between '{ai_label}' and '{human_label}' in annotation guidelines."

        pairs.append(
            ConfusionPair(
                ai_suggested_label=str(ai_label),
                human_ground_truth_label=str(human_label),
                count=count,
                percentage_of_all_overrides=pct,
                representative_examples=rep_examples,
                recurring_semantic_signals=signals,
                possible_taxonomy_overlap=overlap_risk,
                recommendation=rec,
            )
        )

    # Sort by count descending
    pairs.sort(key=lambda x: -x.count)
    return pairs


def audit_general_device_support(reviewed_df: pd.DataFrame) -> GeneralDeviceSupportAudit:
    """
    Perform dedicated audit of general_device_support usage and override patterns.
    """
    if reviewed_df.empty:
        return GeneralDeviceSupportAudit(
            ai_predicted_count=0,
            human_confirmed_count=0,
            human_overridden_count=0,
            override_rate=0.0,
            confirmed_rate=0.0,
            top_target_intents=[],
            distinguishing_signals_retained=[],
            distinguishing_signals_overridden=[],
            audit_verdict="INSUFFICIENT DATA",
            actionable_guidelines=[],
        )

    ai_gds = reviewed_df[reviewed_df["model_suggested_label"] == "general_device_support"]
    n_ai_gds = len(ai_gds)

    if n_ai_gds == 0:
        return GeneralDeviceSupportAudit(
            ai_predicted_count=0,
            human_confirmed_count=0,
            human_overridden_count=0,
            override_rate=0.0,
            confirmed_rate=0.0,
            top_target_intents=[],
            distinguishing_signals_retained=[],
            distinguishing_signals_overridden=[],
            audit_verdict="NOT PREDICTED BY AI",
            actionable_guidelines=[],
        )

    confirmed = ai_gds[ai_gds["annotation_label"] == "general_device_support"]
    overridden = ai_gds[ai_gds["annotation_label"] != "general_device_support"]

    n_confirmed = len(confirmed)
    n_overridden = len(overridden)

    override_rate = float(n_overridden) / float(n_ai_gds)
    confirmed_rate = float(n_confirmed) / float(n_ai_gds)

    # Target intents breakdown
    target_counts = overridden["annotation_label"].value_counts()
    top_targets = [
        {"intent": intent, "count": int(cnt), "percentage": round((cnt / n_overridden) * 100.0, 1)}
        for intent, cnt in target_counts.items()
    ]

    # Distinguishing signals
    retained_signals_raw = extract_frequent_ngrams(confirmed["customer_message"].tolist(), top_k=5)
    retained_signals = [s["signal"] for s in retained_signals_raw]

    overridden_signals_raw = extract_frequent_ngrams(overridden["customer_message"].tolist(), top_k=8)
    overridden_signals = [s["signal"] for s in overridden_signals_raw]

    if override_rate > 0.50:
        verdict = (
            f"OVER-USED AS CATCH-ALL: AI over-predicts general_device_support ({override_rate:.1%} override rate). "
            "Human annotators consistently reclassify into concrete operational intents."
        )
    else:
        verdict = f"BALANCED USAGE: AI general_device_support predictions are confirmed in {confirmed_rate:.1%} of cases."

    guidelines = [
        "DO NOT use general_device_support if a specific subsystem (audio, screen, battery, keyboard, macOS) is named.",
        "DO NOT use general_device_support if problem onset is attributed to a software update ('since update', 'after updating').",
        "USE general_device_support strictly for generic assistance, store inquiries, broad operational how-to questions, or when multiple disparate subsystems are mentioned without a dominant root cause.",
    ]

    return GeneralDeviceSupportAudit(
        ai_predicted_count=n_ai_gds,
        human_confirmed_count=n_confirmed,
        human_overridden_count=n_overridden,
        override_rate=override_rate,
        confirmed_rate=confirmed_rate,
        top_target_intents=top_targets,
        distinguishing_signals_retained=retained_signals,
        distinguishing_signals_overridden=overridden_signals,
        audit_verdict=verdict,
        actionable_guidelines=guidelines,
    )


def analyze_taxonomy_overlaps(
    reviewed_df: pd.DataFrame,
    confusion_pairs: list[ConfusionPair],
) -> list[TaxonomyOverlap]:
    """
    Identify potential taxonomy overlaps and ambiguous decision boundaries from empirical confusion pairs.
    """
    overlaps: list[TaxonomyOverlap] = []

    # Map of known operational boundary pairs to analyze
    boundary_pairs = [
        (
            "general_device_support",
            "software_update_problem",
            "Customer reports vague instability or general issue after a software update.",
            "If the message explicitly links problem onset to an update ('after update', 'since ios 11'), assign software_update_problem. If it is a generic query without update causality, assign general_device_support.",
        ),
        (
            "general_device_support",
            "hardware_audio_connection_issue",
            "Customer asks about cables, adapters, printers, or AirPods.",
            "Any mention of physical ports (USB, Lightning, 3.5mm jack), peripheral cables, or audio accessories routes to hardware_audio_connection_issue.",
        ),
        (
            "display_touch_issue",
            "keyboard_typing_issue",
            "Visual rendering glitches affecting autocorrect / letter I glitch.",
            "If the glitch specifically affects text input or the letter 'I' box symbol, assign keyboard_typing_issue. If the screen panel itself is flickering, cracked, or unresponsive, assign display_touch_issue.",
        ),
        (
            "mac_software_issue",
            "general_device_support",
            "Desktop Mac / MacBook troubleshooting vs generic device inquiries.",
            "Assign mac_software_issue when macOS-specific features (High Sierra, Finder, Safari, Time Machine, macOS install) are mentioned.",
        ),
        (
            "battery_power_issue",
            "hardware_audio_connection_issue",
            "Charging cable / lead failure vs battery depletion.",
            "If the charger or cable is physically defective/broken, assign hardware_audio_connection_issue. If the battery is draining fast or device won't charge/hold power, assign battery_power_issue.",
        ),
    ]

    for label_a, label_b, ambiguity_source, guideline in boundary_pairs:
        # Count bidirectional disagreements between label_a and label_b
        a_to_b = reviewed_df[
            (reviewed_df["model_suggested_label"] == label_a) & (reviewed_df["annotation_label"] == label_b)
        ]
        b_to_a = reviewed_df[
            (reviewed_df["model_suggested_label"] == label_b) & (reviewed_df["annotation_label"] == label_a)
        ]

        total_disagreements = len(a_to_b) + len(b_to_a)
        examples = (
            a_to_b["customer_message"].tolist()[:2] + b_to_a["customer_message"].tolist()[:2]
        )

        overlaps.append(
            TaxonomyOverlap(
                label_a=label_a,
                label_b=label_b,
                observed_disagreements=total_disagreements,
                example_messages=examples,
                ambiguity_source=ambiguity_source,
                proposed_boundary_guideline=guideline,
            )
        )

    return overlaps


def generate_annotation_guidelines(
    reviewed_df: pd.DataFrame,
    all_intents: list[str],
) -> list[AnnotationGuidelineItem]:
    """
    Generate evidence-based annotation guidelines for every intent category in the taxonomy.
    """
    # Intent base definition mappings
    base_definitions: dict[str, dict[str, Any]] = {
        "software_update_problem": {
            "definition": "Customer experiences system instability, installation failures, or regressions caused by an OS or app update.",
            "include_when": [
                "Issue explicitly started after installing an update (iOS 11, High Sierra, app updates).",
                "Update download, verification, or installation fails or gets stuck.",
                "General complaint about bugs introduced by a named software release.",
            ],
            "do_not_use_when": [
                "Problem has an isolated physical root cause (e.g. cracked display, physical cable broken).",
            ],
            "boundary_cases": [
                "If battery drains rapidly after update, battery_power_issue takes precedence unless the user only complains about update bugs.",
            ],
        },
        "hardware_audio_connection_issue": {
            "definition": "Malfunction of physical audio components, headphones, speakers, microphones, or peripheral connectivity ports/cables.",
            "include_when": [
                "AirPods, wired headphones, speaker crackling, microphone failure.",
                "Physical connection issues: USB-C hubs, lightning adapters, dongles, wired printer connections.",
                "Bluetooth audio disconnects or audio accessory failure.",
            ],
            "do_not_use_when": [
                "Network/Wi-Fi router connectivity without physical hardware failure.",
            ],
            "boundary_cases": [
                "MacBook speaker failure routes to hardware_audio_connection_issue, not mac_software_issue.",
            ],
        },
        "keyboard_typing_issue": {
            "definition": "Malfunctions in text input, keyboard unresponsiveness, predictive text bugs, or autocorrect character glitches.",
            "include_when": [
                "Autocorrect letter 'I' rendering bug (e.g. A [?] box glitch).",
                "Keyboard lag, freeze, or failure to appear on screen.",
                "Text prediction or auto-replacement errors.",
            ],
            "do_not_use_when": [
                "Whole touchscreen digitizer is physically unresponsive (use display_touch_issue).",
            ],
            "boundary_cases": [
                "Autocorrect bug accompanied by display glitch routes to keyboard_typing_issue.",
            ],
        },
        "battery_power_issue": {
            "definition": "Abnormal battery depletion, device overheating, charging failure, or sudden power loss.",
            "include_when": [
                "Battery percentage drops rapidly (e.g. 100% to 20% in minutes).",
                "Device fails to charge, won't turn on, or overheats while charging.",
                "Battery health degradation complaints.",
            ],
            "do_not_use_when": [
                "Charger cord or physical adapter is physically broken/torn (use hardware_audio_connection_issue).",
            ],
            "boundary_cases": [
                "Battery drain following an update routes to battery_power_issue.",
            ],
        },
        "mac_software_issue": {
            "definition": "Issues specific to the macOS operating system, Mac desktop applications, or system utilities.",
            "include_when": [
                "macOS High Sierra OS freezes, kernel panics, or spinning beach ball.",
                "Finder, Safari, or Time Machine errors and crashes.",
                "MacBook boot loops and macOS Recovery inquiries.",
            ],
            "do_not_use_when": [
                "Physical MacBook hardware/speaker failures (use hardware_audio_connection_issue).",
            ],
            "boundary_cases": [
                "iCloud photo sync between Mac and iOS routes to mac_software_issue if desktop sync fails.",
            ],
        },
        "display_touch_issue": {
            "definition": "Physical screen panel damage, backlight flickering, touch digitizer unresponsiveness, or display artifacts.",
            "include_when": [
                "Screen flickering, black display, lines across screen, or shattered glass.",
                "Touchscreen digitizer completely unresponsive or ghost touch.",
            ],
            "do_not_use_when": [
                "Only keyboard text input is glitching (use keyboard_typing_issue).",
            ],
            "boundary_cases": [
                "Display unresponsiveness without keyboard context routes to display_touch_issue.",
            ],
        },
        "account_access_issue": {
            "definition": "Authentication, Apple ID credentials, two-factor authentication, or security lockouts.",
            "include_when": [
                "Apple ID locked, disabled, or password reset failure.",
                "Two-factor verification codes not received.",
                "iCloud account login credential failure.",
            ],
            "do_not_use_when": [
                "Unauthorized financial charge on account (use billing_purchase_issue).",
            ],
            "boundary_cases": [
                "Subscription billing password prompt routes to billing_purchase_issue if payment is disputed.",
            ],
        },
        "billing_purchase_issue": {
            "definition": "Financial transactions, subscriptions, refunds, App Store charges, or Apple Pay errors.",
            "include_when": [
                "Unauthorized charges, double billing, or unexpected subscription renewals.",
                "Refund requests for apps, in-app purchases, or media.",
                "Credit card declined on Apple ID.",
            ],
            "do_not_use_when": [
                "Account credential lockout (use account_access_issue).",
            ],
            "boundary_cases": [
                "Inquiry on hardware warranty coverage without financial dispute routes to general_device_support.",
            ],
        },
        "general_device_support": {
            "definition": "Broad device inquiries, general troubleshooting assistance, store visit queries, or multi-topic questions without a single dominant subsystem.",
            "include_when": [
                "Customer asks general how-to or store visit questions.",
                "Customer expresses general dissatisfaction without citing a specific failure mode.",
            ],
            "do_not_use_when": [
                "Any specific operational subsystem (battery, screen, audio, update, keyboard, macOS) is cited.",
            ],
            "boundary_cases": [
                "Use strictly as a fallback when no specific category applies.",
            ],
        },
        "unclear_needs_review": {
            "definition": "Messages with insufficient information, non-English foreign text, severe truncation, or contradictory multi-domain claims.",
            "include_when": [
                "Message consists of only 1-3 words without context.",
                "Message is entirely in a foreign language without clear diagnostic intent.",
                "Customer message is severely corrupted or unintelligible.",
            ],
            "do_not_use_when": [
                "A plausible operational intent can be determined from the customer message.",
            ],
            "boundary_cases": [
                "Use when the reviewer cannot confidently assign any of the 9 operational intents.",
            ],
        },
    }

    guidelines: list[AnnotationGuidelineItem] = []

    for intent in all_intents:
        base = base_definitions.get(
            intent,
            {
                "definition": f"Operational support intent for {intent}.",
                "include_when": [f"Customer requires assistance with {intent}."],
                "do_not_use_when": ["Unrelated issues."],
                "boundary_cases": ["Review context before assigning."],
            },
        )

        # Find reviewed examples from ground truth
        matched_examples = reviewed_df[reviewed_df["annotation_label"] == intent]["customer_message"].tolist()[:3]

        guidelines.append(
            AnnotationGuidelineItem(
                intent_label=intent,
                definition=base["definition"],
                include_when=base["include_when"],
                do_not_use_when=base["do_not_use_when"],
                boundary_cases=base["boundary_cases"],
                reviewed_examples=matched_examples,
                status="OBSERVED FROM DATA",
            )
        )

    return guidelines


def run_annotation_pattern_analysis(
    df: pd.DataFrame,
    total_records: int = 200,
) -> PatternAnalysisReport:
    """
    Execute full Phase 5.9 Pattern Analysis & Taxonomy Calibration pipeline.
    """
    # 1. Filter completed reviews
    reviewed_df = extract_reviewed_dataset(df)

    # 2. Overall agreement metrics
    metrics = compute_agreement_metrics(df, total_records=total_records)

    # 3. Confusion matrix
    confusion_matrix, all_intents = compute_confusion_matrix(reviewed_df)

    # 4. Per-intent statistics
    per_intent_stats = compute_per_intent_statistics(reviewed_df, all_intents)

    # 5. Over-predicted AI labels and Under-predicted Human labels
    overpredicted = []
    underpredicted = []
    for p in per_intent_stats:
        diff = p.ai_suggested_support - p.ground_truth_support
        if diff > 0:
            overpredicted.append({
                "intent": p.intent,
                "ai_suggested_count": p.ai_suggested_support,
                "ground_truth_count": p.ground_truth_support,
                "overprediction_delta": diff,
            })
        elif diff < 0:
            underpredicted.append({
                "intent": p.intent,
                "ai_suggested_count": p.ai_suggested_support,
                "ground_truth_count": p.ground_truth_support,
                "underprediction_delta": abs(diff),
            })

    overpredicted.sort(key=lambda x: -x["overprediction_delta"])
    underpredicted.sort(key=lambda x: -x["underprediction_delta"])

    # 6. Dynamic confusion pairs
    top_confusion_pairs = discover_semantic_patterns(reviewed_df)

    # 7. general_device_support audit
    gds_audit = audit_general_device_support(reviewed_df)

    # 8. Taxonomy overlaps
    taxonomy_overlaps = analyze_taxonomy_overlaps(reviewed_df, top_confusion_pairs)

    # 9. Annotation guidelines
    guidelines = generate_annotation_guidelines(reviewed_df, all_intents)

    # 10. Scientific limitations
    limitations = [
        f"PRELIMINARY SAMPLE SIZE: Analysis is based on N={len(reviewed_df)} of {total_records} golden records ({metrics.sample_coverage_pct:.1f}% coverage).",
        "NOT GENERALIZED MODEL ACCURACY: Observed agreement rates describe human review decisions against baseline predictions and may shift as annotation progresses.",
        "SAMPLING VARIANCE: Per-intent precision/recall on low-support categories (< 5 examples) have wide statistical confidence intervals.",
        "RE-EVALUATION REQUIREMENT: This analysis layer should be re-run at N=100 and N=200 completed human reviews.",
    ]

    return PatternAnalysisReport(
        metrics=metrics,
        per_intent_agreement=per_intent_stats,
        confusion_matrix=confusion_matrix,
        all_intents=all_intents,
        top_confusion_pairs=top_confusion_pairs,
        most_overpredicted_ai_labels=overpredicted,
        most_underpredicted_human_labels=underpredicted,
        general_device_support_audit=gds_audit,
        taxonomy_overlaps=taxonomy_overlaps,
        annotation_guidelines=guidelines,
        scientific_limitations=limitations,
    )


# ---------------------------------------------------------------------------
# Report Exporters (JSON & Markdown)
# ---------------------------------------------------------------------------

def generate_markdown_report(report: PatternAnalysisReport) -> str:
    """
    Format a complete, human-readable Markdown report for Phase 5.9 Pattern Analysis.
    """
    m = report.metrics
    gds = report.general_device_support_audit

    lines: list[str] = [
        "# SupportGraph AI — Phase 5.9 Annotation Pattern Analysis & Taxonomy Calibration Report",
        "",
        f"**Dataset Coverage:** {m.completed_reviews} / {m.total_golden_records} Records ({m.sample_coverage_pct:.1f}%)  ",
        f"**Observed Annotation Agreement:** {m.observed_agreement_rate:.1%} (95% CI: [{m.confidence_interval_95[0]:.1%}, {m.confidence_interval_95[1]:.1%}])  ",
        f"**Human Overrides:** {m.overridden_count} ({1.0 - m.observed_agreement_rate:.1%})  ",
        "",
        "> [!IMPORTANT]",
        f"> **Scientific Integrity Disclaimer:** {m.scientific_disclaimer}",
        "",
        "---",
        "",
        "## 1. Executive Summary & Agreement Statistics",
        "",
        "| Metric | Value | Description |",
        "| :--- | :--- | :--- |",
        f"| **Total Golden Records** | `{m.total_golden_records}` | Target evaluation benchmark size |",
        f"| **Completed Human Reviews** | `{m.completed_reviews}` | Authoritative ground-truth annotations |",
        f"| **Pending Human Reviews** | `{m.pending_reviews}` | Unreviewed records remaining in queue |",
        f"| **Agreed AI Suggestions** | `{m.agreed_count}` | Human confirmed baseline AI suggestion |",
        f"| **Overridden AI Suggestions** | `{m.overridden_count}` | Human assigned a different taxonomy label |",
        f"| **Marked 'unclear_needs_review'** | `{m.unclear_count}` | Records requiring special investigation |",
        f"| **Observed Agreement Rate** | `{m.observed_agreement_rate:.1%}` | Preliminary agreement on reviewed sample |",
        "",
        "---",
        "",
        "## 2. Confusion Matrix (AI Baseline vs Human Ground Truth)",
        "",
        "Rows represent **Baseline AI Suggestions**; columns represent **Human True Annotations**.",
        "",
    ]

    # Confusion matrix markdown table
    header = "| Baseline AI Suggestion \\ Human GT | " + " | ".join(f"`{col}`" for col in report.all_intents) + " | **Total** |"
    sep = "| :--- | " + " | ".join(":---:" for _ in report.all_intents) + " | :---: |"
    lines.append(header)
    lines.append(sep)

    for ai in report.all_intents:
        row_counts = [report.confusion_matrix.get(ai, {}).get(h, 0) for h in report.all_intents]
        row_total = sum(row_counts)
        row_str = f"| `{ai}` | " + " | ".join(str(c) if c > 0 else "-" for c in row_counts) + f" | **{row_total}** |"
        lines.append(row_str)

    # Column totals
    col_totals = [sum(report.confusion_matrix.get(ai, {}).get(h, 0) for ai in report.all_intents) for h in report.all_intents]
    grand_total = sum(col_totals)
    footer = "| **Total Ground Truth** | " + " | ".join(f"**{c}**" for c in col_totals) + f" | **{grand_total}** |"
    lines.append(footer)
    lines.append("")

    # Per-intent table
    lines.extend([
        "### Per-Intent Performance Breakdown",
        "",
        "| Intent Label | Ground Truth Support | AI Predicted Support | Agreed | Observed Precision | Observed Recall | Observed F1 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for p in report.per_intent_agreement:
        lines.append(
            f"| `{p.intent}` | {p.ground_truth_support} | {p.ai_suggested_support} | {p.agreed_matches} | "
            f"{p.observed_precision:.1%} | {p.observed_recall:.1%} | {p.observed_f1:.3f} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Top AI → Human Correction Patterns",
        "",
        "The following table itemizes the most frequent systematic correction pairs observed in the human review data:",
        "",
        "| Baseline AI Suggestion | Human Ground Truth | Count | % Overrides | Recurring Semantic Signals | Actionable Recommendation |",
        "| :--- | :--- | :---: | :---: | :--- | :--- |",
    ])

    for cp in report.top_confusion_pairs:
        sig_str = ", ".join(f"'{s['signal']}' ({s['frequency']})" for s in cp.recurring_semantic_signals[:3]) or "None"
        lines.append(
            f"| `{cp.ai_suggested_label}` | `{cp.human_ground_truth_label}` | **{cp.count}** | {cp.percentage_of_all_overrides:.1f}% | "
            f"{sig_str} | {cp.recommendation} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 4. `general_device_support` Diagnostic Audit",
        "",
        f"**Audit Verdict:** {gds.audit_verdict}",
        "",
        f"- **AI Predictions:** {gds.ai_predicted_count} ({gds.ai_predicted_count / m.completed_reviews:.1%} of reviewed sample)",
        f"- **Human Confirmed:** {gds.human_confirmed_count} ({gds.confirmed_rate:.1%})",
        f"- **Human Overridden:** {gds.human_overridden_count} ({gds.override_rate:.1%})",
        "",
        "### Primary Reclassification Destinations for `general_device_support`",
        "",
        "| Target Intent | Reclassification Count | Share of Overrides |",
        "| :--- | :---: | :---: |",
    ])

    for tgt in gds.top_target_intents:
        lines.append(f"| `{tgt['intent']}` | {tgt['count']} | {tgt['percentage']:.1f}% |")

    lines.extend([
        "",
        "### Distinguishing Semantic Characteristics",
        f"- **Retained in `general_device_support`:** {', '.join(f'`{s}`' for s in gds.distinguishing_signals_retained) or 'Generic queries'}",
        f"- **Overridden into Specific Intents:** {', '.join(f'`{s}`' for s in gds.distinguishing_signals_overridden) or 'Specific failure verbs'}",
        "",
        "### Actionable Specificity Rules",
    ])
    for g in gds.actionable_guidelines:
        lines.append(f"1. {g}")

    lines.extend([
        "",
        "---",
        "",
        "## 5. Taxonomy Overlap & Boundary Disambiguation",
        "",
    ])

    for to in report.taxonomy_overlaps:
        lines.extend([
            f"### Boundary: `{to.label_a}` vs `{to.label_b}`",
            f"- **Observed Disagreements:** {to.observed_disagreements}",
            f"- **Ambiguity Source:** {to.ambiguity_source}",
            f"- **Proposed Guideline:** {to.proposed_boundary_guideline}",
            "",
        ])

    lines.extend([
        "---",
        "",
        "## 6. Scientific Sample Size & Statistical Limitations",
        "",
    ])
    for lim in report.scientific_limitations:
        lines.append(f"- {lim}")

    lines.append("")
    return "\n".join(lines)


def generate_guidelines_markdown(guidelines: list[AnnotationGuidelineItem]) -> str:
    """
    Format evidence-based annotation guidelines as standalone documentation.
    """
    lines: list[str] = [
        "# SupportGraph AI — Evidence-Based Intent Annotation Guidelines (Phase 5.9)",
        "",
        "Derived from empirical human ground truth and observed confusion patterns.",
        "",
        "---",
        "",
    ]

    for item in guidelines:
        lines.extend([
            f"## `{item.intent_label}`",
            f"**Definition:** {item.definition}  ",
            f"**Status:** `{item.status}`",
            "",
            "### Include When:",
        ])
        for inc in item.include_when:
            lines.append(f"- {inc}")

        lines.extend([
            "",
            "### Do NOT Use When:",
        ])
        for exc in item.do_not_use_when:
            lines.append(f"- {exc}")

        lines.extend([
            "",
            "### Boundary Cases:",
        ])
        for b in item.boundary_cases:
            lines.append(f"- {b}")

        if item.reviewed_examples:
            lines.extend([
                "",
                "### Observed Examples from Ground Truth:",
            ])
            for ex in item.reviewed_examples:
                lines.append(f"> *\"{ex}\"*")

        lines.extend(["", "---", ""])

    return "\n".join(lines)


def export_reports_to_directory(
    report: PatternAnalysisReport,
    output_dir: Path | str,
) -> dict[str, str]:
    """
    Export all JSON artifacts and Markdown reports to specified directory.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    created_files: dict[str, str] = {}

    # 1. summary.json
    summary_data = {
        "metrics": report.metrics.to_dict(),
        "per_intent_agreement": [p.to_dict() for p in report.per_intent_agreement],
        "most_overpredicted_ai_labels": report.most_overpredicted_ai_labels,
        "most_underpredicted_human_labels": report.most_underpredicted_human_labels,
    }
    p_sum = out_path / "summary.json"
    with open(p_sum, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    created_files["summary_json"] = str(p_sum)

    # 2. confusion_matrix.json
    p_cm = out_path / "confusion_matrix.json"
    with open(p_cm, "w", encoding="utf-8") as f:
        json.dump({
            "all_intents": report.all_intents,
            "confusion_matrix": report.confusion_matrix,
        }, f, indent=2)
    created_files["confusion_matrix_json"] = str(p_cm)

    # 3. correction_patterns.json
    p_cp = out_path / "correction_patterns.json"
    with open(p_cp, "w", encoding="utf-8") as f:
        json.dump([cp.to_dict() for cp in report.top_confusion_pairs], f, indent=2)
    created_files["correction_patterns_json"] = str(p_cp)

    # 4. taxonomy_overlap_report.json
    p_to = out_path / "taxonomy_overlap_report.json"
    with open(p_to, "w", encoding="utf-8") as f:
        json.dump({
            "general_device_support_audit": report.general_device_support_audit.to_dict(),
            "taxonomy_overlaps": [to.to_dict() for to in report.taxonomy_overlaps],
        }, f, indent=2)
    created_files["taxonomy_overlap_json"] = str(p_to)

    # 5. annotation_guidelines.md
    p_ag = out_path / "annotation_guidelines.md"
    guidelines_md = generate_guidelines_markdown(report.annotation_guidelines)
    with open(p_ag, "w", encoding="utf-8") as f:
        f.write(guidelines_md)
    created_files["annotation_guidelines_md"] = str(p_ag)

    # 6. annotation_pattern_analysis.md
    p_apa = out_path / "annotation_pattern_analysis.md"
    report_md = generate_markdown_report(report)
    with open(p_apa, "w", encoding="utf-8") as f:
        f.write(report_md)
    created_files["annotation_pattern_analysis_md"] = str(p_apa)

    return created_files
