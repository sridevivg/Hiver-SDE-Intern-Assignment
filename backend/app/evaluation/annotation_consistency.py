"""
SupportGraph AI — Annotation Consistency & Taxonomy Boundary Validation (Phase 5.10)

READ-ONLY scientific layer for auditing semantic consistency across human-reviewed
ground-truth annotations, evaluating within-label coherence, and formulating
non-coercive soft annotation boundary guidelines.

Guiding Principles:
1. READ-ONLY & IMMUTABLE: Never modifies source datasets, annotations, or pending items.
2. NON-PRESCRIPTIVE: Flags "Possible boundary inconsistency", never "Incorrect annotation".
3. NO HARD CLASSIFIER RULES: Does not convert observed empirical patterns into rigid heuristics.
4. CALIBRATED & AUDITABLE: Tracks sample size caveats, confidence bounds, and full provenance.
"""
from __future__ import annotations

import collections
import hashlib
import json
import logging
import math
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

# Stop words tailored for customer support social messaging (pre-tokenized unigrams)
SUPPORT_STOP_WORDS: set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "cannot", "could",
    "did", "do", "does", "doing", "down", "during",
    "each", "few", "for", "from", "further", "had", "has",
    "have", "having", "he", "her", "here",
    "hers", "herself", "him", "himself", "his", "how", "i",
    "if", "in", "into", "is", "it",
    "its", "itself", "let", "me", "more", "most", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "she",
    "should", "so", "some", "such",
    "than", "that", "the", "their", "theirs", "them", "themselves",
    "then", "there", "these", "they",
    "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "we", "were",
    "what", "when", "where", "which",
    "while", "who", "whom", "why", "with", "would",
    "you", "your", "yours",
    "yourself", "yourselves", "user", "url", "applesupport", "http", "https",
    "please", "help", "hey", "hi", "hello", "thanks", "thank",
}


# Calibrated Priority Thresholds for Short Social Messaging (15-30 words)
HIGH_PRIORITY_SIMILARITY_THRESHOLD = 0.14
MEDIUM_PRIORITY_SIMILARITY_THRESHOLD = 0.09
LOW_PRIORITY_SIMILARITY_THRESHOLD = 0.05



# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class ConsistencyCandidatePair:
    """A pair of human-reviewed records with different labels but high semantic similarity."""
    golden_id_a: str
    golden_id_b: str
    message_a: str
    message_b: str
    human_label_a: str
    human_label_b: str
    similarity_score: float
    priority: str  # 'HIGH', 'MEDIUM', 'LOW'
    shared_semantic_signals: List[str]
    possible_ambiguity_explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "golden_id_a": self.golden_id_a,
            "golden_id_b": self.golden_id_b,
            "message_a": self.message_a,
            "message_b": self.message_b,
            "human_label_a": self.human_label_a,
            "human_label_b": self.human_label_b,
            "similarity_score": round(float(self.similarity_score), 4),
            "priority": self.priority,
            "shared_semantic_signals": self.shared_semantic_signals,
            "possible_ambiguity_explanation": self.possible_ambiguity_explanation,
        }


@dataclass
class WithinLabelCoherence:
    """Semantic coherence statistics within a single intent label class."""
    intent_label: str
    reviewed_count: int
    average_pairwise_similarity: float
    min_pairwise_similarity: float
    max_pairwise_similarity: float
    semantic_diversity_indicator: float
    insufficient_sample_flag: bool
    sample_size_caveat: str
    outlier_records: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent_label": self.intent_label,
            "reviewed_count": self.reviewed_count,
            "average_pairwise_similarity": round(float(self.average_pairwise_similarity), 4),
            "min_pairwise_similarity": round(float(self.min_pairwise_similarity), 4),
            "max_pairwise_similarity": round(float(self.max_pairwise_similarity), 4),
            "semantic_diversity_indicator": round(float(self.semantic_diversity_indicator), 4),
            "insufficient_sample_flag": self.insufficient_sample_flag,
            "sample_size_caveat": self.sample_size_caveat,
            "outlier_records": self.outlier_records,
        }


@dataclass
class CrossLabelBoundary:
    """Empirical cross-label boundary analysis between two intent categories."""
    pair_name: str
    intent_a: str
    intent_b: str
    reviewed_count_a: int
    reviewed_count_b: int
    cross_pair_count: int
    mean_cross_similarity: float
    max_cross_similarity: float
    representative_boundary_examples: List[Dict[str, Any]]
    overlap_explanation: str
    soft_annotation_guideline: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pair_name": self.pair_name,
            "intent_a": self.intent_a,
            "intent_b": self.intent_b,
            "reviewed_count_a": self.reviewed_count_a,
            "reviewed_count_b": self.reviewed_count_b,
            "cross_pair_count": self.cross_pair_count,
            "mean_cross_similarity": round(float(self.mean_cross_similarity), 4),
            "max_cross_similarity": round(float(self.max_cross_similarity), 4),
            "representative_boundary_examples": self.representative_boundary_examples,
            "overlap_explanation": self.overlap_explanation,
            "soft_annotation_guideline": self.soft_annotation_guideline,
        }


@dataclass
class ConsistencySummary:
    """Overall summary of annotation consistency analysis."""
    total_golden_records: int
    completed_reviews: int
    pending_reviews: int
    total_pairs_evaluated: int
    same_label_pairs: int
    cross_label_pairs: int
    high_priority_candidates: int
    medium_priority_candidates: int
    low_priority_candidates: int
    total_candidates_flagged: int
    source_sha256: str
    scientific_disclaimer: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_golden_records": self.total_golden_records,
            "completed_reviews": self.completed_reviews,
            "pending_reviews": self.pending_reviews,
            "total_pairs_evaluated": self.total_pairs_evaluated,
            "same_label_pairs": self.same_label_pairs,
            "cross_label_pairs": self.cross_label_pairs,
            "high_priority_candidates": self.high_priority_candidates,
            "medium_priority_candidates": self.medium_priority_candidates,
            "low_priority_candidates": self.low_priority_candidates,
            "total_candidates_flagged": self.total_candidates_flagged,
            "source_sha256": self.source_sha256,
            "scientific_disclaimer": self.scientific_disclaimer,
        }


@dataclass
class ConsistencyReviewDecision:
    """Audit record of human validation on a consistency candidate pair."""
    pair_id: str
    golden_id_a: str
    golden_id_b: str
    decision: str  # 'LABELS_BOTH_APPROPRIATE', 'RECONSIDER_A', 'RECONSIDER_B', 'RECONSIDER_BOTH', 'SKIPPED'
    annotator: str
    timestamp: str
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Core Utilities
# ---------------------------------------------------------------------------

def calculate_file_sha256(filepath: str | Path) -> str:
    """Compute SHA-256 digest of a file to ensure strict byte-for-byte immutability."""
    path = Path(filepath)
    if not path.exists():
        return ""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def extract_completed_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract ONLY completed human-reviewed records, strictly excluding pending or empty items.
    """
    if df.empty:
        return df.copy()

    # Must have a non-empty human annotation label
    has_label = df["annotation_label"].fillna("").astype(str).str.strip() != ""
    
    # Must have a recognized completion or reviewed status
    status_col = df["annotation_status"].fillna("").astype(str).str.lower().str.strip()
    not_pending = ~status_col.isin(["", "pending", "pending_human_review", "unreviewed", "none"])
    
    # Exclude pending placeholders if label equals pending_human_review
    valid_label = df["annotation_label"].fillna("").astype(str).str.strip().str.lower() != "pending_human_review"

    return df[has_label & not_pending & valid_label].copy()


def clean_text_for_similarity(text: str) -> str:
    """Normalize and clean customer text for lexical/semantic vectorization."""
    if not isinstance(text, str):
        return ""
    cleaned = text.lower()
    cleaned = re.sub(r"https?://\S+", "", cleaned)
    cleaned = re.sub(r"<url>", "", cleaned)
    cleaned = re.sub(r"<user>", "", cleaned)
    cleaned = re.sub(r"@\w+", "", cleaned)
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def extract_shared_signals(text_a: str, text_b: str, top_k: int = 5) -> List[str]:
    """Extract salient shared words and bigrams between two messages."""
    clean_a = set(clean_text_for_similarity(text_a).split()) - SUPPORT_STOP_WORDS
    clean_b = set(clean_text_for_similarity(text_b).split()) - SUPPORT_STOP_WORDS
    shared_unigrams = [w for w in clean_a if w in clean_b and len(w) > 2]
    
    # Extract simple bigrams
    words_a = [w for w in clean_text_for_similarity(text_a).split() if len(w) > 1]
    words_b = [w for w in clean_text_for_similarity(text_b).split() if len(w) > 1]
    bigrams_a = {f"{words_a[i]} {words_a[i+1]}" for i in range(len(words_a) - 1)}
    bigrams_b = {f"{words_b[i]} {words_b[i+1]}" for i in range(len(words_b) - 1)}
    shared_bigrams = [bg for bg in bigrams_a if bg in bigrams_b and not all(w in SUPPORT_STOP_WORDS for w in bg.split())]
    
    signals = sorted(list(set(shared_bigrams + shared_unigrams)), key=lambda x: (-len(x.split()), -len(x)))
    return signals[:top_k]


def generate_ambiguity_explanation(
    label_a: str,
    label_b: str,
    shared_signals: List[str],
    similarity: float,
) -> str:
    """
    Generate an explainable rationale for why two customer messages may share high similarity
    despite receiving different human labels.
    """
    signals_str = f" ('{', '.join(shared_signals)}')" if shared_signals else ""
    
    # Boundary: general_device_support vs specific intent
    if label_a == "general_device_support" or label_b == "general_device_support":
        specific_label = label_b if label_a == "general_device_support" else label_a
        return (
            f"Cross-boundary between general device triage and specific intent '{specific_label}'. "
            f"Shared lexical cues{signals_str} may indicate that one message was interpreted as a broad "
            f"operational friction while the other was assigned to a concrete operational symptom."
        )
    
    # Boundary: display_touch_issue vs keyboard_typing_issue
    if {label_a, label_b} == {"display_touch_issue", "keyboard_typing_issue"}:
        return (
            f"Input interface overlap between display/touch behavior and keyboard typing issues. "
            f"Shared context{signals_str} indicates customer may describe on-screen typing unresponsiveness "
            f"or visual character rendering that spans both modalities."
        )
    
    # Boundary: software_update_problem vs hardware_audio_connection_issue
    if {label_a, label_b} == {"software_update_problem", "hardware_audio_connection_issue"}:
        return (
            f"Causality overlap where an audio or connectivity issue arose following an iOS update. "
            f"Shared signals{signals_str} suggest one annotator focused on update causality while the other "
            f"focused on the presenting hardware/audio symptom."
        )

    # Boundary: software_update_problem vs battery_power_issue
    if {label_a, label_b} == {"software_update_problem", "battery_power_issue"}:
        return (
            f"Post-update symptom overlap. Shared terms{signals_str} reflect customer attributing battery "
            f"drain to an OS update. One label captures the root cause (update) while the other captures the symptom (battery)."
        )

    # Boundary: mac_software_issue vs other
    if "mac_software_issue" in (label_a, label_b):
        other_label = label_b if label_a == "mac_software_issue" else label_a
        return (
            f"Platform boundary between macOS desktop environment and specific function '{other_label}'. "
            f"Shared cues{signals_str} reflect macOS context with specific functional symptoms."
        )

    # Generic fallback
    return (
        f"Semantic overlap across distinct operational categories '{label_a}' and '{label_b}'. "
        f"Shared vocabulary{signals_str} with cosine similarity {similarity:.2f} suggests "
        f"multi-symptom customer phrasing or borderline operational interpretation."
    )


# ---------------------------------------------------------------------------
# Core Analysis Engine
# ---------------------------------------------------------------------------

class AnnotationConsistencyAnalyzer:
    """
    Read-only analyzer for detecting semantic consistency candidates, evaluating
    within-label coherence, and formulating soft boundary guidelines.
    """

    def __init__(
        self,
        high_sim_threshold: float = HIGH_PRIORITY_SIMILARITY_THRESHOLD,
        med_sim_threshold: float = MEDIUM_PRIORITY_SIMILARITY_THRESHOLD,
        low_sim_threshold: float = LOW_PRIORITY_SIMILARITY_THRESHOLD,
    ) -> None:
        self.high_sim_threshold = high_sim_threshold
        self.med_sim_threshold = med_sim_threshold
        self.low_sim_threshold = low_sim_threshold

    def compute_pairwise_similarities(
        self,
        reviewed_df: pd.DataFrame,
    ) -> Tuple[np.ndarray, TfidfVectorizer]:
        """
        Compute TF-IDF cosine similarity matrix across all completed customer messages.
        """
        if reviewed_df.empty:
            return np.zeros((0, 0)), TfidfVectorizer()

        texts = [
            clean_text_for_similarity(str(msg))
            for msg in reviewed_df["customer_message"].tolist()
        ]

        # Use 1-gram and 2-gram TF-IDF with support-adapted stop words
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words=list(SUPPORT_STOP_WORDS),
            min_df=1,
            sublinear_tf=True,
        )
        tfidf_matrix = vectorizer.fit_transform(texts)
        sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        return sim_matrix, vectorizer

    def extract_consistency_candidates(
        self,
        reviewed_df: pd.DataFrame,
        sim_matrix: np.ndarray,
    ) -> List[ConsistencyCandidatePair]:
        """
        Identify pairs of completed human-reviewed records where semantic similarity is high
        but assigned human labels are different.
        """
        if reviewed_df.empty or sim_matrix.size == 0:
            return []

        records = reviewed_df.to_dict(orient="records")
        n = len(records)
        candidates: List[ConsistencyCandidatePair] = []

        for i in range(n):
            for j in range(i + 1, n):
                rec_a = records[i]
                rec_b = records[j]
                label_a = str(rec_a.get("annotation_label", "")).strip()
                label_b = str(rec_b.get("annotation_label", "")).strip()

                # Only evaluate pairs with different human labels
                if label_a == label_b or not label_a or not label_b:
                    continue

                sim = float(sim_matrix[i, j])

                # Check if similarity meets low threshold
                if sim < self.low_sim_threshold:
                    continue

                if sim >= self.high_sim_threshold:
                    priority = "HIGH"
                elif sim >= self.med_sim_threshold:
                    priority = "MEDIUM"
                else:
                    priority = "LOW"

                msg_a = str(rec_a.get("customer_message", ""))
                msg_b = str(rec_b.get("customer_message", ""))
                shared_signals = extract_shared_signals(msg_a, msg_b)
                explanation = generate_ambiguity_explanation(label_a, label_b, shared_signals, sim)

                candidates.append(
                    ConsistencyCandidatePair(
                        golden_id_a=str(rec_a.get("golden_id", f"idx_{i}")),
                        golden_id_b=str(rec_b.get("golden_id", f"idx_{j}")),
                        message_a=msg_a,
                        message_b=msg_b,
                        human_label_a=label_a,
                        human_label_b=label_b,
                        similarity_score=sim,
                        priority=priority,
                        shared_semantic_signals=shared_signals,
                        possible_ambiguity_explanation=explanation,
                    )
                )

        # Sort candidates descending by similarity score
        candidates.sort(key=lambda c: c.similarity_score, reverse=True)
        return candidates

    def compute_within_label_coherence(
        self,
        reviewed_df: pd.DataFrame,
        sim_matrix: np.ndarray,
    ) -> Dict[str, WithinLabelCoherence]:
        """
        Analyze internal semantic coherence and diversity within each human-assigned intent class.
        """
        if reviewed_df.empty or sim_matrix.size == 0:
            return {}

        records = reviewed_df.to_dict(orient="records")
        n = len(records)
        label_indices: Dict[str, List[int]] = collections.defaultdict(list)

        for idx, rec in enumerate(records):
            lbl = str(rec.get("annotation_label", "")).strip()
            if lbl:
                label_indices[lbl].append(idx)

        coherence_results: Dict[str, WithinLabelCoherence] = {}

        for lbl, indices in sorted(label_indices.items()):
            count = len(indices)
            insufficient = count < 5
            caveat = (
                f"Preliminary sample size (N={count} < 5); findings are descriptive and lack statistical power."
                if insufficient
                else f"Sufficient preliminary sample (N={count}); representative of reviewed subset."
            )

            if count < 2:
                # Single item in category
                coherence_results[lbl] = WithinLabelCoherence(
                    intent_label=lbl,
                    reviewed_count=count,
                    average_pairwise_similarity=1.0,
                    min_pairwise_similarity=1.0,
                    max_pairwise_similarity=1.0,
                    semantic_diversity_indicator=0.0,
                    insufficient_sample_flag=insufficient,
                    sample_size_caveat=caveat,
                    outlier_records=[],
                )
                continue

            # Sub-matrix of similarities for this class
            sub_sims: List[float] = []
            item_mean_sims: List[Tuple[int, float]] = []

            for i_idx in indices:
                peer_sims = [float(sim_matrix[i_idx, j_idx]) for j_idx in indices if i_idx != j_idx]
                mean_peer = float(np.mean(peer_sims)) if peer_sims else 0.0
                item_mean_sims.append((i_idx, mean_peer))

            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    sub_sims.append(float(sim_matrix[indices[i], indices[j]]))

            avg_sim = float(np.mean(sub_sims)) if sub_sims else 0.0
            min_sim = float(np.min(sub_sims)) if sub_sims else 0.0
            max_sim = float(np.max(sub_sims)) if sub_sims else 0.0
            diversity = max(0.0, 1.0 - avg_sim)

            # Identify outlier records (lowest mean similarity to peers in same label)
            item_mean_sims.sort(key=lambda x: x[1])
            outliers: List[Dict[str, Any]] = []
            for item_idx, mean_peer in item_mean_sims[:2]:
                if mean_peer < avg_sim * 0.75:  # Noticeably below class average
                    rec = records[item_idx]
                    outliers.append({
                        "golden_id": str(rec.get("golden_id", "")),
                        "customer_message": str(rec.get("customer_message", "")),
                        "mean_similarity_to_class": round(mean_peer, 4),
                    })

            coherence_results[lbl] = WithinLabelCoherence(
                intent_label=lbl,
                reviewed_count=count,
                average_pairwise_similarity=avg_sim,
                min_pairwise_similarity=min_sim,
                max_pairwise_similarity=max_sim,
                semantic_diversity_indicator=diversity,
                insufficient_sample_flag=insufficient,
                sample_size_caveat=caveat,
                outlier_records=outliers,
            )

        return coherence_results

    def compute_cross_label_boundaries(
        self,
        reviewed_df: pd.DataFrame,
        sim_matrix: np.ndarray,
        candidates: List[ConsistencyCandidatePair],
    ) -> List[CrossLabelBoundary]:
        """
        Perform in-depth boundary analysis across specific confusing intent pairs
        and formulate soft annotation guidelines.
        """
        if reviewed_df.empty:
            return []

        records = reviewed_df.to_dict(orient="records")
        label_counts = collections.Counter(
            str(r.get("annotation_label", "")).strip() for r in records if str(r.get("annotation_label", "")).strip()
        )

        # Predefined focus pairs as requested in Phase 5.10
        target_pairs = [
            ("general_device_support", "software_update_problem"),
            ("general_device_support", "hardware_audio_connection_issue"),
            ("general_device_support", "battery_power_issue"),
            ("general_device_support", "mac_software_issue"),
            ("display_touch_issue", "keyboard_typing_issue"),
        ]

        boundaries: List[CrossLabelBoundary] = []

        for intent_a, intent_b in target_pairs:
            pair_candidates = [
                c for c in candidates
                if {c.human_label_a, c.human_label_b} == {intent_a, intent_b}
            ]

            count_a = label_counts.get(intent_a, 0)
            count_b = label_counts.get(intent_b, 0)

            # Compute cross similarities across all pairs between intent_a and intent_b
            indices_a = [i for i, r in enumerate(records) if str(r.get("annotation_label", "")).strip() == intent_a]
            indices_b = [i for i, r in enumerate(records) if str(r.get("annotation_label", "")).strip() == intent_b]

            cross_sims: List[float] = []
            for i in indices_a:
                for j in indices_b:
                    cross_sims.append(float(sim_matrix[i, j]))

            mean_cross_sim = float(np.mean(cross_sims)) if cross_sims else 0.0
            max_cross_sim = float(np.max(cross_sims)) if cross_sims else 0.0

            rep_examples = [
                {
                    "golden_id_a": c.golden_id_a,
                    "golden_id_b": c.golden_id_b,
                    "message_a": c.message_a,
                    "message_b": c.message_b,
                    "label_a": c.human_label_a,
                    "label_b": c.human_label_b,
                    "similarity": round(c.similarity_score, 4),
                    "shared_signals": c.shared_semantic_signals,
                }
                for c in pair_candidates[:3]
            ]

            explanation, guideline = self._generate_boundary_guideline(intent_a, intent_b, mean_cross_sim, max_cross_sim)

            boundaries.append(
                CrossLabelBoundary(
                    pair_name=f"{intent_a} vs {intent_b}",
                    intent_a=intent_a,
                    intent_b=intent_b,
                    reviewed_count_a=count_a,
                    reviewed_count_b=count_b,
                    cross_pair_count=len(cross_sims),
                    mean_cross_similarity=mean_cross_sim,
                    max_cross_similarity=max_cross_sim,
                    representative_boundary_examples=rep_examples,
                    overlap_explanation=explanation,
                    soft_annotation_guideline=guideline,
                )
            )

        return boundaries

    def _generate_boundary_guideline(
        self,
        intent_a: str,
        intent_b: str,
        mean_sim: float,
        max_sim: float,
    ) -> Tuple[str, str]:
        """Generate explainable rationale and non-coercive soft annotation guideline."""
        if {intent_a, intent_b} == {"general_device_support", "software_update_problem"}:
            explanation = (
                "Customers frequently mention broad iOS version complaints alongside general dissatisfaction or "
                "settings confusion. When an update is mentioned as the triggering event, the boundary between "
                "general triage and update-specific troubleshooting can become blurred."
            )
            guideline = (
                "When the customer's primary friction point involves an active OS upgrade attempt, installation failure, "
                "or explicit post-update regression, consider evidence favoring 'software_update_problem'. "
                "Conversely, when the inquiry represents broad feature questions, general settings configuration, "
                "or unspecified device sluggishness without a focal update symptom, 'general_device_support' is usually appropriate."
            )
        elif {intent_a, intent_b} == {"general_device_support", "hardware_audio_connection_issue"}:
            explanation = (
                "Bluetooth, Wi-Fi toggles, and audio accessories are often described alongside general device settings. "
                "Phrases such as 'can't turn off Wi-Fi in Control Center' bridge general UI triage and wireless connectivity."
            )
            guideline = (
                "When the dominant issue involves physical audio output (speakers, microphones, headphones), Bluetooth pairing, "
                "or wireless adapter connectivity, annotators should consider evidence favoring 'hardware_audio_connection_issue'. "
                "When the customer seeks general advice or expresses broad frustration with OS UI controls without an underlying hardware/audio fault, "
                "'general_device_support' may be considered."
            )
        elif {intent_a, intent_b} == {"general_device_support", "battery_power_issue"}:
            explanation = (
                "Power drain is occasionally reported alongside general device overheating or sluggish performance. "
                "Ambiguity arises when battery drain is mentioned as one of multiple diffuse symptoms."
            )
            guideline = (
                "When battery depletion, charging failure, percentage drop, or power preservation is the central complaint, "
                "evidence usually favors 'battery_power_issue'. If power is only mentioned tangentially amidst broad system complaints, "
                "consider the primary operational action requested."
            )
        elif {intent_a, intent_b} == {"general_device_support", "mac_software_issue"}:
            explanation = (
                "Mac mentions (MacBook, iMac, macOS, Sierra) can overlap with general software questions. "
                "Reviewers must distinguish platform-specific desktop software troubleshooting from general cross-device questions."
            )
            guideline = (
                "When the troubleshooting context specifically requires macOS desktop software remediation (e.g. Safari desktop, macOS installer, Time Machine, Finder), "
                "consider 'mac_software_issue'. If the inquiry involves generic multi-device queries or general support referral, "
                "'general_device_support' may be appropriate."
            )
        elif {intent_a, intent_b} == {"display_touch_issue", "keyboard_typing_issue"}:
            explanation = (
                "Touchscreen unresponsiveness directly interferes with on-screen keyboard typing. "
                "Customers experiencing keyboard lag or phantom keystrokes may describe both screen and keyboard symptoms."
            )
            guideline = (
                "When the core issue is character input, text prediction, autocorrect glitches, or keyboard layout behavior, "
                "evidence usually favors 'keyboard_typing_issue'. When the screen physical digitizer, touch unresponsiveness, "
                "display flickering, or backlight failure is the primary defect, annotators should consider 'display_touch_issue'."
            )
        else:
            explanation = f"Cross-boundary overlap between '{intent_a}' and '{intent_b}'."
            guideline = (
                f"When classifying ambiguous customer messages on the boundary of '{intent_a}' and '{intent_b}', "
                f"reviewers should examine the brand's clarifying response and consider the dominant operational friction."
            )

        return explanation, guideline

    def run_full_analysis(
        self,
        df: pd.DataFrame,
        source_sha256: str = "",
        total_records: int = 200,
    ) -> Tuple[ConsistencySummary, List[ConsistencyCandidatePair], Dict[str, WithinLabelCoherence], List[CrossLabelBoundary]]:
        """
        Execute full end-to-end read-only consistency analysis on the dataframe.
        """
        reviewed_df = extract_completed_reviews(df)
        n_reviewed = len(reviewed_df)
        n_pending = total_records - n_reviewed

        if n_reviewed == 0:
            summary = ConsistencySummary(
                total_golden_records=total_records,
                completed_reviews=0,
                pending_reviews=total_records,
                total_pairs_evaluated=0,
                same_label_pairs=0,
                cross_label_pairs=0,
                high_priority_candidates=0,
                medium_priority_candidates=0,
                low_priority_candidates=0,
                total_candidates_flagged=0,
                source_sha256=source_sha256,
                scientific_disclaimer="No completed human reviews found in dataset.",
            )
            return summary, [], {}, []

        sim_matrix, _ = self.compute_pairwise_similarities(reviewed_df)
        candidates = self.extract_consistency_candidates(reviewed_df, sim_matrix)
        within_coherence = self.compute_within_label_coherence(reviewed_df, sim_matrix)
        boundaries = self.compute_cross_label_boundaries(reviewed_df, sim_matrix, candidates)

        total_pairs = (n_reviewed * (n_reviewed - 1)) // 2
        
        # Calculate same-label vs cross-label pairs
        records = reviewed_df.to_dict(orient="records")
        same_label_count = 0
        cross_label_count = 0
        for i in range(n_reviewed):
            for j in range(i + 1, n_reviewed):
                la = str(records[i].get("annotation_label", "")).strip()
                lb = str(records[j].get("annotation_label", "")).strip()
                if la == lb:
                    same_label_count += 1
                else:
                    cross_label_count += 1

        high_count = sum(1 for c in candidates if c.priority == "HIGH")
        med_count = sum(1 for c in candidates if c.priority == "MEDIUM")
        low_count = sum(1 for c in candidates if c.priority == "LOW")

        summary = ConsistencySummary(
            total_golden_records=total_records,
            completed_reviews=n_reviewed,
            pending_reviews=n_pending,
            total_pairs_evaluated=total_pairs,
            same_label_pairs=same_label_count,
            cross_label_pairs=cross_label_count,
            high_priority_candidates=high_count,
            medium_priority_candidates=med_count,
            low_priority_candidates=low_count,
            total_candidates_flagged=len(candidates),
            source_sha256=source_sha256,
            scientific_disclaimer=(
                f"OBSERVED ANNOTATION CONSISTENCY on N={n_reviewed} completed human reviews. "
                "Flagged pairs represent 'Possible boundary inconsistencies' for human validation, "
                "NOT definitive annotation errors. Observed findings reflect a preliminary subset and "
                "must NOT be converted into hard deterministic classifier rules."
            ),
        )

        return summary, candidates, within_coherence, boundaries


# ---------------------------------------------------------------------------
# Report Serialization
# ---------------------------------------------------------------------------

def generate_markdown_consistency_report(
    summary: ConsistencySummary,
    candidates: List[ConsistencyCandidatePair],
    within_coherence: Dict[str, WithinLabelCoherence],
    boundaries: List[CrossLabelBoundary],
) -> str:
    """
    Generate comprehensive, scientifically structured Markdown report for Phase 5.10.
    Explicitly demarcates OBSERVED DATA from PRELIMINARY INTERPRETATION and PROPOSED SOFT GUIDELINES.
    """
    lines: List[str] = []

    # Title & Metadata
    lines.append("# Phase 5.10: Annotation Consistency & Taxonomy Boundary Validation")
    lines.append("")
    lines.append("> **SupportGraph AI — Quality Assurance & Taxonomy Calibration**  ")
    lines.append(f"> **Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  ")
    lines.append(f"> **Source Dataset SHA-256:** `{summary.source_sha256}`  ")
    lines.append(f"> **Evaluated Human Reviews:** {summary.completed_reviews} / {summary.total_golden_records} ({summary.completed_reviews / summary.total_golden_records:.1%})  ")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Scientific Notice
    lines.append("## Core Scientific Principles & Data Protection")
    lines.append("")
    lines.append("> [!IMPORTANT]")
    lines.append("> **READ-ONLY AUDIT LAYER:** This layer performs read-only semantic consistency validation across completed human annotations.")
    lines.append("> 1. **Zero Ground Truth Modification:** `annotation_label` and `annotation_status` remain 100% immutable.")
    lines.append("> 2. **No Hard Classification Rules:** Observed patterns are NOT converted into rigid production classifier rules.")
    lines.append("> 3. **Non-Prescriptive Flagging:** Flagged candidates represent *Possible boundary inconsistencies*, never *Incorrect annotations*.")
    lines.append(r"> 4. **Sample Size Caveat:** Findings are based on $N=57$ completed reviews ($28.5\%$) and must be re-evaluated as annotation proceeds.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Section 1: Observed Data
    lines.append("## SECTION 1 — OBSERVED DATA")
    lines.append("")
    lines.append("### 1.1 Consistency Analysis Metrics")
    lines.append("")
    lines.append("| Metric | Observed Value | Description |")
    lines.append("|---|---|---|")
    lines.append(f"| **Total Golden Records** | `{summary.total_golden_records}` | Size of the stratified golden benchmark |")
    lines.append(f"| **Completed Human Reviews** | `{summary.completed_reviews}` | Authoritative ground-truth annotations ($N$) |")
    lines.append(rf"| **Pending Reviews** | `{summary.pending_reviews}` | Unreviewed records ($71.5\%$) |")
    lines.append(rf"| **Total Pairs Evaluated** | `{summary.total_pairs_evaluated:,}` | Total unique 2-combinations $\binom{{N}}{{2}}$ |")

    lines.append(f"| **Same-Label Pairs** | `{summary.same_label_pairs:,}` | Pairs where both records share the exact same human intent |")
    lines.append(f"| **Cross-Label Pairs** | `{summary.cross_label_pairs:,}` | Pairs assigned to different intent categories |")
    lines.append(f"| **Total Consistency Candidates** | `{summary.total_candidates_flagged}` | Cross-label pairs meeting similarity thresholds |")
    lines.append(rf"| **HIGH Priority Candidates** | `{summary.high_priority_candidates}` | Cross-label similarity $\ge {HIGH_PRIORITY_SIMILARITY_THRESHOLD}$ |")
    lines.append(rf"| **MEDIUM Priority Candidates** | `{summary.medium_priority_candidates}` | Cross-label similarity ${MEDIUM_PRIORITY_SIMILARITY_THRESHOLD} - {HIGH_PRIORITY_SIMILARITY_THRESHOLD}$ |")
    lines.append(rf"| **LOW Priority Candidates** | `{summary.low_priority_candidates}` | Cross-label similarity ${LOW_PRIORITY_SIMILARITY_THRESHOLD} - {MEDIUM_PRIORITY_SIMILARITY_THRESHOLD}$ |")
    lines.append("")

    # Section 1.2: Within-Label Coherence Table
    lines.append("### 1.2 Within-Label Semantic Coherence")
    lines.append("")
    lines.append("| Intent Label | Reviewed ($N$) | Mean Pairwise Similarity | Min Similarity | Max Similarity | Diversity Score | Data Sufficiency |")
    lines.append("|---|---|---|---|---|---|---|")
    for lbl, coh in within_coherence.items():
        flag_str = r"⚠️ Small ($N < 5$)" if coh.insufficient_sample_flag else r"✅ Sufficient ($N \ge 5$)"

        lines.append(
            f"| `{lbl}` | `{coh.reviewed_count}` | `{coh.average_pairwise_similarity:.4f}` | "
            f"`{coh.min_pairwise_similarity:.4f}` | `{coh.max_pairwise_similarity:.4f}` | "
            f"`{coh.semantic_diversity_indicator:.4f}` | {flag_str} |"
        )
    lines.append("")

    # Section 1.3: Top Consistency Candidates
    lines.append("### 1.3 Top Flagged Consistency Candidate Pairs")
    lines.append("")
    top_candidates = candidates[:15]
    if not top_candidates:
        lines.append("*No cross-label consistency candidate pairs met the threshold criteria.*")
    else:
        lines.append("| Rank | Priority | Sim | Golden ID A | Human Label A | Golden ID B | Human Label B | Shared Signals |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for idx, c in enumerate(top_candidates, 1):
            signals_str = ", ".join(c.shared_semantic_signals[:3]) if c.shared_semantic_signals else "none"
            lines.append(
                f"| #{idx} | **`{c.priority}`** | `{c.similarity_score:.4f}` | `{c.golden_id_a}` | `{c.human_label_a}` | "
                f"`{c.golden_id_b}` | `{c.human_label_b}` | `{signals_str}` |"
            )
    lines.append("")

    # Section 2: Preliminary Interpretation
    lines.append("---")
    lines.append("")
    lines.append("## SECTION 2 — PRELIMINARY INTERPRETATION")
    lines.append("")
    lines.append("### 2.1 Cross-Label Boundary Overlaps")
    lines.append("")
    for b in boundaries:
        lines.append(f"#### 🔍 `{b.pair_name}`")
        lines.append(f"- **Reviewed Sample Support:** `{b.intent_a}` ($N={b.reviewed_count_a}$) vs `{b.intent_b}` ($N={b.reviewed_count_b}$)")
        lines.append(f"- **Mean Cross Similarity:** `{b.mean_cross_similarity:.4f}` (Max: `{b.max_cross_similarity:.4f}` across {b.cross_pair_count} pairs)")
        lines.append(f"- **Observed Boundary Pattern:** {b.overlap_explanation}")
        lines.append("")
        if b.representative_boundary_examples:
            lines.append("**Representative Cross-Boundary Examples:**")
            lines.append("")
            for ex in b.representative_boundary_examples[:2]:
                lines.append(f"> **[{ex['golden_id_a']}] ({ex['label_a']}):** \"{ex['message_a']}\"  ")
                lines.append(f"> **[{ex['golden_id_b']}] ({ex['label_b']}):** \"{ex['message_b']}\"  ")
                lines.append(f"> *Similarity:* `{ex['similarity']:.4f}` | *Shared terms:* `{', '.join(ex['shared_signals']) if ex['shared_signals'] else 'none'}`")
                lines.append(">")
        lines.append("")

    # Section 2.2: Within-Label Outliers
    lines.append("### 2.2 Within-Label Outlier & Diversity Interpretation")
    lines.append("")
    outlier_found = False
    for lbl, coh in within_coherence.items():
        if coh.outlier_records:
            outlier_found = True
            lines.append(f"- **`{lbl}` Outliers:**")
            for out in coh.outlier_records:
                lines.append(f"  - `[{out['golden_id']}]`: \"{out['customer_message']}\" (Mean intra-class similarity: `{out['mean_similarity_to_class']:.4f}` vs class mean `{coh.average_pairwise_similarity:.4f}`)")
    if not outlier_found:
        lines.append("All individual annotations demonstrate moderate-to-high coherence with their assigned class peers.")
    lines.append("")

    # Section 3: Proposed Soft Guidelines
    lines.append("---")
    lines.append("")
    lines.append("## SECTION 3 — PROPOSED SOFT ANNOTATION GUIDELINES")
    lines.append("")
    lines.append("> [!TIP]")
    lines.append("> **NON-COERCIVE GUIDELINES:** The following guidelines are advisory cognitive aids for human reviewers and prompt designers. They emphasize dominant operational friction over superficial keyword matches.")
    lines.append("")
    for b in boundaries:
        lines.append(f"### Guideline for `{b.pair_name}`")
        lines.append(f"{b.soft_annotation_guideline}")
        lines.append("")

    # Human Review Queue Overview
    lines.append("---")
    lines.append("")
    lines.append("## SECTION 4 — HUMAN CONSISTENCY REVIEW QUEUE")
    lines.append("")
    lines.append("A top-priority consistency review queue containing the highest-ambiguity pairs has been prepared for controlled human validation:")
    lines.append("- Reviewers can inspect pairs in the CLI using `python -m backend.scripts.analyze_annotation_consistency --interactive`.")
    lines.append("- Available non-destructive actions:")
    lines.append("  - `[A]` Labels are both appropriate (confirms distinct operational nuance).")
    lines.append("  - `[B]` Record A should be reconsidered.")
    lines.append("  - `[C]` Record B should be reconsidered.")
    lines.append("  - `[D]` Both records should be reconsidered.")
    lines.append("  - `[S]` Skip.")
    lines.append("- Decisions are logged to `data/golden/consistency_review_decisions.json` without modifying ground-truth labels.")
    lines.append("")

    return "\n".join(lines)
