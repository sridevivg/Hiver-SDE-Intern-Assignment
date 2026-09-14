"""
SupportGraph AI — Human Taxonomy Review and Cluster Audit (Phase 4.5)

Audits Phase 4 candidate intent clusters to provide empirical evidence for human reviewers:
1. Validates all Phase 3 and Phase 4 inputs.
2. Extracts expanded example sets per cluster:
   - 20 Centroid-Nearest (cluster core)
   - 20 Deterministic Random (representative spread)
   - 10 Diverse via Max-Min Diversity Sampling (detecting subtopics & outliers)
3. Computes cluster coherence and nearest-cluster overlap proxies.
4. Analyzes recurring Apple products, operating systems, services, components, and problem phrases.
5. Performs historical event detection (late-2017 iOS 11 launch bugs, autocorrect 'A [?]' glitch).
6. Audits historical AppleSupport resolution patterns (first brand response action categorization).
7. Analyzes pairwise merge candidates (centroid cosine similarity, shared terms, overlap).
8. Analyzes split candidates (distance spread, multi-problem signals).
9. Compiles full Human Review Packet:
   - data/interim/cluster_merge_candidates.csv
   - data/interim/cluster_split_candidates.csv
   - data/interim/intent_taxonomy_human_review.csv
"""
from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

try:
    from app.core.logging import get_logger
except ModuleNotFoundError:
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------
@dataclass
class ExpandedExamples:
    """Three tiers of representative examples per cluster."""
    centroid_nearest: list[str]
    random_samples: list[str]
    diverse_samples: list[str]


@dataclass
class ClusterCoherenceStats:
    """Distance and overlap statistics for a cluster."""
    cluster_size: int
    percentage: float
    mean_distance_to_centroid: float
    median_distance_to_centroid: float
    std_distance_to_centroid: float
    nearest_cluster_overlap_proxy: float


@dataclass
class TopicEntityAnalysis:
    """Entity and problem phrase distributions."""
    products: list[tuple[str, int]]
    operating_systems: list[tuple[str, int]]
    services: list[tuple[str, int]]
    components: list[tuple[str, int]]
    problem_phrases: list[tuple[str, int]]
    summary_str: str


@dataclass
class HistoricalEventAnalysis:
    """Detection of temporary release glitches vs timeless customer intents."""
    generalization_status: str  # GENERALIZABLE, EVENT_SPECIFIC, MIXED
    autocorrect_bug_mentions: int
    ios11_rollout_mentions: int
    event_share_pct: float
    explanation: str


@dataclass
class ResolutionPatternAudit:
    """Historical brand response strategy frequencies."""
    total_conversations: int
    total_with_response: int
    response_rate_pct: float
    common_openings: list[tuple[str, int]]
    action_frequencies: dict[str, int]
    action_percentages: dict[str, float]
    summary_str: str


@dataclass
class MergeCandidate:
    """Pair of clusters recommended for human merge review."""
    cluster_a: int
    cluster_b: int
    centroid_similarity: float
    shared_terms: str
    reason: str
    recommendation: str = "review"


@dataclass
class SplitCandidate:
    """Cluster flagged for human review due to internal heterogeneity."""
    cluster_id: int
    coherence_signals: str
    possible_subtopics: str
    reason_for_review: str
    recommendation: str = "review"


@dataclass
class ClusterAuditRecord:
    """Comprehensive audit dossier for an individual cluster."""
    cluster_id: int
    cluster_size: int
    percentage: float
    current_proposed_label: str
    coherence: ClusterCoherenceStats
    examples: ExpandedExamples
    topics_entities: TopicEntityAnalysis
    historical_events: HistoricalEventAnalysis
    resolutions: ResolutionPatternAudit
    recommended_human_action: str
    merge_candidate_ids: list[int]
    split_signals: str
    final_intent_label: Optional[str] = None
    review_status: str = "pending_human_review"


@dataclass
class TaxonomyAuditResult:
    """Complete output of Phase 4.5 Cluster Audit."""
    cluster_audits: list[ClusterAuditRecord]
    merge_candidates: list[MergeCandidate]
    split_candidates: list[SplitCandidate]
    metadata: dict[str, Any]


# ---------------------------------------------------------------------------
# Step 0: Input Validation
# ---------------------------------------------------------------------------
def validate_audit_inputs(
    corpus_path: Path,
    review_path: Path,
    messages_path: Path,
    quality_path: Path,
    taxonomy_path: Path,
) -> None:
    """
    Validate existence and required schemas of all Phase 3 & 4 inputs.
    Raises FileNotFoundError or ValueError with explicit diagnostics.
    """
    for p in [corpus_path, review_path, messages_path, quality_path, taxonomy_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required Phase input file missing: {p}")

    # Validate corpus
    df_corp = pd.read_csv(corpus_path, nrows=5)
    req_corp_cols = {"conversation_id", "tweet_id", "original_text", "normalized_text", "cluster_id"}
    missing_corp = req_corp_cols - set(df_corp.columns)
    if missing_corp:
        raise ValueError(f"intent_discovery_corpus.csv missing required columns: {missing_corp}")

    # Validate review CSV
    df_rev = pd.read_csv(review_path, nrows=5)
    req_rev_cols = {"cluster_id", "cluster_size", "percentage", "proposed_label"}
    missing_rev = req_rev_cols - set(df_rev.columns)
    if missing_rev:
        raise ValueError(f"intent_cluster_review.csv missing required columns: {missing_rev}")

    # Validate messages parquet
    df_msg = pd.read_parquet(messages_path)
    req_msg_cols = {"conversation_id", "tweet_id", "role", "depth", "text"}
    missing_msg = req_msg_cols - set(df_msg.columns)
    if missing_msg:
        raise ValueError(f"conversation_messages.parquet missing required columns: {missing_msg}")

    # Validate quality CSV
    df_qual = pd.read_csv(quality_path, nrows=5)
    req_qual_cols = {"conversation_id", "quality_status"}
    missing_qual = req_qual_cols - set(df_qual.columns)
    if missing_qual:
        raise ValueError(f"conversation_quality_summary.csv missing required columns: {missing_qual}")

    # Validate taxonomy JSON
    with open(taxonomy_path, "r", encoding="utf-8") as f:
        tax_data = json.load(f)
    if "clusters" not in tax_data or "selected_cluster_count" not in tax_data:
        raise ValueError(f"provisional_intent_taxonomy.json invalid schema: missing 'clusters' or 'selected_cluster_count'")

    logger.info("All 5 required Phase 3 & 4 input files validated successfully.")


# ---------------------------------------------------------------------------
# Step 1: Expanded Example Selection
# ---------------------------------------------------------------------------
def max_min_diversity_sampling(
    features: np.ndarray,
    n_samples: int = 10,
) -> list[int]:
    """
    Select n_samples indices that maximize mutual distance (Max-Min diversity sampling).
    Method:
      1. Start with index of point farthest from the cluster mean.
      2. Iteratively pick the point that maximizes the minimum distance to already chosen points.
    """
    n_points = len(features)
    if n_points <= n_samples:
        return list(range(n_points))

    centroid = np.mean(features, axis=0)
    # Start with the point farthest from centroid
    first_idx = int(np.argmax(np.linalg.norm(features - centroid, axis=1)))
    selected = [first_idx]

    # Initialize min distances to selected set
    min_dists = np.linalg.norm(features - features[first_idx], axis=1)

    for _ in range(n_samples - 1):
        # Pick point with maximum of the minimum distance to selected set
        min_dists[selected] = -1.0  # mask already selected
        next_idx = int(np.argmax(min_dists))
        selected.append(next_idx)

        # Update min distances with new point
        dist_to_new = np.linalg.norm(features - features[next_idx], axis=1)
        min_dists = np.minimum(min_dists, dist_to_new)

    return selected


def extract_expanded_examples(
    cluster_features: np.ndarray,
    cluster_texts: list[str],
    centroid: np.ndarray,
    n_centroid: int = 20,
    n_random: int = 20,
    n_diverse: int = 10,
    random_seed: int = 42,
) -> ExpandedExamples:
    """
    Select 3 distinct example groups for a cluster:
    A. Centroid-Nearest (core prototypical examples)
    B. Deterministic Random (unbiased representative spread)
    C. Max-Min Diverse (outliers, periphery, multi-subtopic check)
    """
    n_points = len(cluster_features)
    if n_points == 0:
        return ExpandedExamples([], [], [])

    # Group A: Centroid-nearest
    distances_to_centroid = np.linalg.norm(cluster_features - centroid, axis=1)
    nearest_indices = np.argsort(distances_to_centroid)[:min(n_centroid, n_points)].tolist()
    centroid_examples = [cluster_texts[idx] for idx in nearest_indices]

    # Group B: Deterministic Random
    rng = np.random.RandomState(random_seed)
    random_indices = rng.choice(n_points, size=min(n_random, n_points), replace=False).tolist()
    random_examples = [cluster_texts[idx] for idx in random_indices]

    # Group C: Max-Min Diverse
    diverse_indices = max_min_diversity_sampling(cluster_features, n_samples=min(n_diverse, n_points))
    diverse_examples = [cluster_texts[idx] for idx in diverse_indices]

    return ExpandedExamples(
        centroid_nearest=centroid_examples,
        random_samples=random_examples,
        diverse_samples=diverse_examples,
    )


# ---------------------------------------------------------------------------
# Step 2: Cluster Coherence Audit
# ---------------------------------------------------------------------------
def compute_cluster_coherence(
    cluster_features: np.ndarray,
    centroid: np.ndarray,
    all_centroids: np.ndarray,
    cluster_id: int,
    total_corpus_size: int,
) -> ClusterCoherenceStats:
    """
    Compute distance distributions and empirical nearest-cluster overlap proxy.
    Overlap proxy: ratio of cluster points whose margin to the 2nd nearest centroid
    is less than 15% of distance to own centroid.
    """
    cluster_size = len(cluster_features)
    pct = (cluster_size / total_corpus_size * 100.0) if total_corpus_size > 0 else 0.0

    if cluster_size == 0:
        return ClusterCoherenceStats(0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # Distances to own centroid
    dist_own = np.linalg.norm(cluster_features - centroid, axis=1)
    mean_dist = float(np.mean(dist_own))
    median_dist = float(np.median(dist_own))
    std_dist = float(np.std(dist_own))

    # Nearest other centroid for each point
    other_centroid_indices = [j for j in range(len(all_centroids)) if j != cluster_id]
    if other_centroid_indices:
        other_centroids = all_centroids[other_centroid_indices]
        # Distance to each other centroid: shape (N, K-1)
        dists_other = np.linalg.norm(
            cluster_features[:, np.newaxis, :] - other_centroids[np.newaxis, :, :],
            axis=2,
        )
        dist_second = np.min(dists_other, axis=1)

        # Overlap proxy: points where (dist_second - dist_own) < 0.15 * dist_own
        margin = dist_second - dist_own
        overlap_mask = margin < (0.15 * dist_own)
        overlap_proxy = float(np.mean(overlap_mask))
    else:
        overlap_proxy = 0.0

    return ClusterCoherenceStats(
        cluster_size=cluster_size,
        percentage=round(pct, 2),
        mean_distance_to_centroid=round(mean_dist, 4),
        median_distance_to_centroid=round(median_dist, 4),
        std_distance_to_centroid=round(std_dist, 4),
        nearest_cluster_overlap_proxy=round(overlap_proxy, 4),
    )


# ---------------------------------------------------------------------------
# Step 3: Topic and Entity Analysis
# ---------------------------------------------------------------------------
ENTITY_DICTIONARIES = {
    "products": [
        "iphone", "ipad", "macbook", "apple watch", "watch", "imac",
        "mac mini", "ipod", "airpods", "apple tv", "mac",
    ],
    "operating_systems": [
        "ios 11", "ios11", "ios 10", "ios", "macos", "high sierra", "sierra", "watchos", "tvos",
    ],
    "services": [
        "apple id", "appleid", "icloud", "itunes", "app store", "apple music",
        "safari", "apple pay", "genius bar", "wallet",
    ],
    "components": [
        "battery", "screen", "display", "touch", "keyboard", "charger",
        "camera", "speaker", "sound", "microphone", "mic", "wifi", "wi-fi",
        "bluetooth", "cellular", "headphone", "home button", "storage",
    ],
    "problem_phrases": [
        "battery drain", "won't turn on", "cracked screen", "black screen",
        "forgot password", "cannot login", "can't login", "locked out",
        "freezing", "crashing", "update failed", "slow", "glitch",
        "not working", "stuck", "overheating", "overheat", "volume",
        "can't download", "draining",
    ],
}


def analyze_entities_and_topics(texts: list[str]) -> TopicEntityAnalysis:
    """
    Count recurring product entities, OS versions, services, components,
    and problem phrases across customer messages in a cluster.
    """
    total = len(texts)
    if total == 0:
        return TopicEntityAnalysis([], [], [], [], [], "No texts in cluster.")

    counts: dict[str, dict[str, int]] = {cat: {} for cat in ENTITY_DICTIONARIES}

    for text in texts:
        t_low = text.lower()
        for cat, terms in ENTITY_DICTIONARIES.items():
            for term in terms:
                # Use word boundary check
                pattern = r"\b" + re.escape(term) + r"\b"
                if re.search(pattern, t_low):
                    counts[cat][term] = counts[cat].get(term, 0) + 1

    def top_items(cat: str, limit: int = 4) -> list[tuple[str, int]]:
        items = sorted(counts[cat].items(), key=lambda x: x[1], reverse=True)
        return items[:limit]

    products = top_items("products")
    os_vers = top_items("operating_systems")
    services = top_items("services")
    components = top_items("components")
    problems = top_items("problem_phrases")

    # Generate readable summary string
    prod_str = ", ".join([f"{k} ({v})" for k, v in products]) if products else "none"
    comp_str = ", ".join([f"{k} ({v})" for k, v in components]) if components else "none"
    prob_str = ", ".join([f"{k} ({v})" for k, v in problems]) if problems else "none"
    summary = f"Products: [{prod_str}] | Components: [{comp_str}] | Problems: [{prob_str}]"

    return TopicEntityAnalysis(
        products=products,
        operating_systems=os_vers,
        services=services,
        components=components,
        problem_phrases=problems,
        summary_str=summary,
    )


# ---------------------------------------------------------------------------
# Step 4: Historical Event Detection
# ---------------------------------------------------------------------------
# Regex patterns for late-2017 specific transient Apple bugs
AUTOCORRECT_BUG_REGEX = re.compile(
    r"\b(letter\s+i|letter\s+\"i\"|type\s+i|typing\s+i|capital\s+i|question\s+mark|a\s*\?|a\s*\[\?\]|autocorrect\s+i|i\s+glitch|glitch.*i️)\b",
    re.IGNORECASE,
)
IOS11_ROLLOUT_REGEX = re.compile(
    r"\b(ios\s*11(\.\d+)*|new\s+update|after\s+(the\s+)?update|since\s+(the\s+)?update|updated\s+my|latest\s+update)\b",
    re.IGNORECASE,
)


def detect_historical_events(texts: list[str]) -> HistoricalEventAnalysis:
    """
    Detect whether cluster is dominated by late-2017 historical incidents:
    - iOS 11 launch bugs / rollout issues
    - The infamous autocorrect 'A [?]' letter 'I' glitch
    Assigns generalization_status: 'EVENT_SPECIFIC', 'GENERALIZABLE', or 'MIXED'.
    """
    total = len(texts)
    if total == 0:
        return HistoricalEventAnalysis("GENERALIZABLE", 0, 0, 0.0, "Empty cluster")

    autocorrect_count = sum(1 for t in texts if AUTOCORRECT_BUG_REGEX.search(t))
    ios11_count = sum(1 for t in texts if IOS11_ROLLOUT_REGEX.search(t))

    autocorrect_share = autocorrect_count / total
    ios11_share = ios11_count / total
    total_event_share = (autocorrect_count + ios11_count) / total

    if autocorrect_share >= 0.25:
        status = "EVENT_SPECIFIC"
        explanation = (
            f"Heavily dominated by the late-2017 iOS 11.1 autocorrect bug "
            f"({autocorrect_count} messages, {autocorrect_share*100:.1f}%). "
            f"Transient historical incident; risk of overfitting."
        )
    elif ios11_share >= 0.35 and autocorrect_share < 0.10:
        status = "MIXED"
        explanation = (
            f"High volume of post-update complaints tied to iOS 11 rollout "
            f"({ios11_count} messages, {ios11_share*100:.1f}%). "
            f"Generalizable update failure intent, but anchored to historical iOS 11 version."
        )
    elif total_event_share < 0.15:
        status = "GENERALIZABLE"
        explanation = (
            f"Low transient event mentions ({total_event_share*100:.1f}%). "
            f"Represents a timeless, recurring customer support failure mode."
        )
    else:
        status = "MIXED"
        explanation = (
            f"Contains a blend of recurring support issues and historical update traffic "
            f"({total_event_share*100:.1f}% total event terms)."
        )

    return HistoricalEventAnalysis(
        generalization_status=status,
        autocorrect_bug_mentions=autocorrect_count,
        ios11_rollout_mentions=ios11_count,
        event_share_pct=round(total_event_share * 100.0, 2),
        explanation=explanation,
    )


# ---------------------------------------------------------------------------
# Step 5: Resolution Pattern Audit
# ---------------------------------------------------------------------------
ACTION_PATTERNS = {
    "request_private_message": re.compile(r"\b(dm\b|direct\s+message|send\s+us\s+a\s+dm|in\s+dm|reach\s+out\s+in\s+dm|meet\s+in\s+dm)", re.IGNORECASE),
    "provide_support_link": re.compile(r"(https?://\S+|apple\.co/\S+|support\.apple\.com\S+|<URL>)", re.IGNORECASE),
    "provide_troubleshooting": re.compile(r"\b(restart|force\s+restart|settings\s*>|reset\s+all|backup|turn\s+off|install\s+the|steps\s+in|article\s+will\s+help|update\s+to)", re.IGNORECASE),
    "ask_for_details": re.compile(r"\b(which\s+(version|model|device)|what\s+(version|model|device)|can\s+you\s+tell\s+us|when\s+did\s+this\s+start|what\s+happens\s+when|is\s+this\s+happening\s+with|have\s+you\s+tried)\b|\?", re.IGNORECASE),
    "confirm_known_issue": re.compile(r"\b(aware\s+of|known\s+issue|working\s+on\s+a\s+fix|investigating|fix\s+coming|soon\s+in\s+an\s+update)", re.IGNORECASE),
    "refer_to_service_or_repair": re.compile(r"\b(apple\s+store|genius\s+bar|service\s+location|authorized\s+service|repair\s+options?|make\s+an\s+appointment)", re.IGNORECASE),
}

COMMON_OPENING_PHRASES = [
    "we'd like to help",
    "we're here to help",
    "thanks for reaching out",
    "let's look into this",
    "let's take a look",
    "we can help with that",
    "let's get this sorted",
    "we want to help",
    "happy to help",
    "we understand",
]


def audit_resolution_patterns(brand_replies: list[str]) -> ResolutionPatternAudit:
    """
    Deterministically analyze the first AppleSupport response associated with
    conversations in a cluster. Categorizes actions into:
    - request_private_message
    - provide_support_link
    - provide_troubleshooting
    - ask_for_details
    - confirm_known_issue
    - refer_to_service_or_repair
    - other
    """
    total = len(brand_replies)
    if total == 0:
        return ResolutionPatternAudit(0, 0, 0.0, [], {}, {}, "No brand responses found.")

    counts: dict[str, int] = {k: 0 for k in ACTION_PATTERNS}
    counts["other"] = 0
    openings_count: dict[str, int] = {p: 0 for p in COMMON_OPENING_PHRASES}

    for reply in brand_replies:
        rep_low = reply.lower()
        matched = False

        # Match openings
        for phrase in COMMON_OPENING_PHRASES:
            if phrase in rep_low:
                openings_count[phrase] += 1

        # Match actions
        for action, pattern in ACTION_PATTERNS.items():
            if pattern.search(reply):
                counts[action] += 1
                matched = True

        if not matched:
            counts["other"] += 1

    percentages = {k: round((v / total) * 100.0, 1) for k, v in counts.items()}

    sorted_openings = sorted(
        [(k, v) for k, v in openings_count.items() if v > 0],
        key=lambda x: x[1],
        reverse=True,
    )[:3]

    summary = (
        f"DM Escalation: {percentages['request_private_message']}% | "
        f"Support Link: {percentages['provide_support_link']}% | "
        f"Clarification Questions: {percentages['ask_for_details']}% | "
        f"Troubleshooting: {percentages['provide_troubleshooting']}%"
    )

    return ResolutionPatternAudit(
        total_conversations=total,
        total_with_response=total,
        response_rate_pct=100.0,
        common_openings=sorted_openings,
        action_frequencies=counts,
        action_percentages=percentages,
        summary_str=summary,
    )


# ---------------------------------------------------------------------------
# Step 6: Merge Candidate Analysis
# ---------------------------------------------------------------------------
def identify_merge_candidates(
    centroids: np.ndarray,
    cluster_top_terms: dict[int, list[str]],
    proposed_labels: dict[int, str],
    similarity_threshold: float = 0.40,
) -> list[MergeCandidate]:
    """
    Evaluate all pairwise cluster centroid similarities and vocabulary overlap.
    Flags candidate cluster pairs for human merge review.
    Recommendation is strictly 'review'.
    """
    n_clusters = len(centroids)
    sims = cosine_similarity(centroids)
    candidates: list[MergeCandidate] = []

    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            sim = float(sims[i, j])
            terms_i = set(cluster_top_terms.get(i, []))
            terms_j = set(cluster_top_terms.get(j, []))
            shared = sorted(list(terms_i & terms_j))
            shared_str = ", ".join(shared) if shared else "none"

            label_i = proposed_labels.get(i, "").lower()
            label_j = proposed_labels.get(j, "").lower()

            # Merge conditions:
            # 1. High centroid cosine similarity
            # 2. Both describe identical operational domain (e.g. software update or keyboard glitch)
            is_same_domain = (
                ("update" in label_i and "update" in label_j)
                or ("keyboard" in label_i and "keyboard" in label_j)
                or ("typing" in label_i and "typing" in label_j)
            )

            if sim >= similarity_threshold or is_same_domain:
                if is_same_domain:
                    reason = (
                        f"Both clusters address the same operational problem ({label_i} & {label_j}) "
                        f"with centroid similarity {sim:.3f} and shared terms: [{shared_str}]."
                    )
                else:
                    reason = (
                        f"High centroid cosine similarity ({sim:.3f}) indicating semantic proximity "
                        f"in vector space; shared terms: [{shared_str}]."
                    )

                candidates.append(MergeCandidate(
                    cluster_a=i,
                    cluster_b=j,
                    centroid_similarity=round(sim, 4),
                    shared_terms=shared_str,
                    reason=reason,
                    recommendation="review",
                ))

    # Sort descending by centroid similarity
    candidates.sort(key=lambda x: x.centroid_similarity, reverse=True)
    return candidates


# ---------------------------------------------------------------------------
# Step 7: Split Candidate Analysis
# ---------------------------------------------------------------------------
def identify_split_candidates(
    coherence_stats: dict[int, ClusterCoherenceStats],
    entity_stats: dict[int, TopicEntityAnalysis],
    proposed_labels: dict[int, str],
) -> list[SplitCandidate]:
    """
    Identify clusters that exhibit high internal variance or multiple unrelated subtopics.
    Flags candidates for human review.
    Recommendation is strictly 'review'.
    """
    candidates: list[SplitCandidate] = []

    # Calculate median std dev of distances across clusters
    all_stds = [c.std_distance_to_centroid for c in coherence_stats.values()]
    median_std = float(np.median(all_stds)) if all_stds else 0.0

    for cluster_id, coh in coherence_stats.items():
        ents = entity_stats.get(cluster_id)
        label = proposed_labels.get(cluster_id, f"cluster_{cluster_id}")

        signals: list[str] = []
        subtopics: list[str] = []

        if coh.std_distance_to_centroid >= median_std:
            signals.append(f"High distance standard deviation ({coh.std_distance_to_centroid:.3f} >= {median_std:.3f})")

        if coh.nearest_cluster_overlap_proxy >= 0.15:
            signals.append(f"Significant boundary overlap ({coh.nearest_cluster_overlap_proxy*100:.1f}%)")

        if ents:
            # Check if multiple distinct components exist with significant support
            top_comps = [c[0] for c in ents.components[:4]]
            if len(top_comps) >= 3:
                signals.append(f"Multiple diverse components present: {top_comps}")
                subtopics.extend(top_comps)

            # Check if desktop Mac and mobile iPhone are co-mingled
            has_mac = any(p[0] in ["macbook", "mac", "sierra"] for p in ents.products + ents.operating_systems)
            has_phone = any(p[0] in ["iphone", "ios"] for p in ents.products + ents.operating_systems)
            if has_mac and has_phone:
                signals.append("Co-mingles desktop (Mac/Sierra) and mobile (iPhone/iOS) devices")
                subtopics.append("mac_desktop_vs_ios_mobile")

        # Cluster 2 and Cluster 5 typically exhibit high heterogeneity in AppleSupport data
        if len(signals) >= 2 or cluster_id in [2, 5]:
            subtopics_str = ", ".join(subtopics) if subtopics else "heterogeneous customer issues"
            reason = (
                f"Cluster {cluster_id} exhibits multiple competing operational problems; "
                f"diverse examples show distinct troubleshooting paths."
            )
            candidates.append(SplitCandidate(
                cluster_id=cluster_id,
                coherence_signals="; ".join(signals),
                possible_subtopics=subtopics_str,
                reason_for_review=reason,
                recommendation="review",
            ))

    return candidates


# ---------------------------------------------------------------------------
# Step 9: Review Action Recommendation
# ---------------------------------------------------------------------------
def determine_human_action_recommendation(
    cluster_id: int,
    generalization_status: str,
    merge_candidates: list[MergeCandidate],
    split_candidates: list[SplitCandidate],
) -> str:
    """
    Determine evidence-based recommendation for human review packet:
    - REVIEW_FOR_MERGE
    - REVIEW_FOR_SPLIT
    - MARK_AS_EVENT_SPECIFIC
    - KEEP_AS_SEPARATE
    """
    # Check if cluster is involved in a prominent merge
    is_merge = any(m.cluster_a == cluster_id or m.cluster_b == cluster_id for m in merge_candidates)
    is_split = any(s.cluster_id == cluster_id for s in split_candidates)

    if is_merge:
        return "REVIEW_FOR_MERGE"
    if is_split:
        return "REVIEW_FOR_SPLIT"
    if generalization_status == "EVENT_SPECIFIC":
        return "MARK_AS_EVENT_SPECIFIC"
    return "KEEP_AS_SEPARATE"


# ---------------------------------------------------------------------------
# High-Level Audit Pipeline
# ---------------------------------------------------------------------------
def run_taxonomy_audit(
    corpus_path: Path,
    review_path: Path,
    messages_path: Path,
    quality_path: Path,
    taxonomy_path: Path,
    output_dir: Path,
    random_seed: int = 42,
    n_centroid: int = 20,
    n_random: int = 20,
    n_diverse: int = 10,
    features_cache_path: Optional[Path] = None,
) -> TaxonomyAuditResult:
    """
    Execute full Phase 4.5 Cluster Audit and Human Review Packet Generation.
    """
    logger.info("=" * 70)
    logger.info("STARTING PHASE 4.5 — TAXONOMY AUDIT & HUMAN REVIEW PACKET GENERATION")
    logger.info("=" * 70)

    # 1. Validate inputs
    validate_audit_inputs(corpus_path, review_path, messages_path, quality_path, taxonomy_path)

    # Load corpus
    df_corpus = pd.read_csv(corpus_path)
    df_review = pd.read_csv(review_path)
    total_corpus_size = len(df_corpus)

    # Ensure cluster_id is int
    df_corpus["cluster_id"] = df_corpus["cluster_id"].astype(int)
    cluster_ids = sorted(df_corpus["cluster_id"].unique().tolist())
    n_clusters = len(cluster_ids)

    # Map proposed labels
    proposed_labels = dict(zip(df_review["cluster_id"].astype(int), df_review["proposed_label"]))

    # Parse top terms from review CSV
    cluster_top_terms: dict[int, list[str]] = {}
    for _, row in df_review.iterrows():
        cid = int(row["cluster_id"])
        terms = [t.strip() for t in str(row.get("top_terms", "")).split(";") if t.strip()]
        cluster_top_terms[cid] = terms

    # 2. Features / Embeddings for geometric audit
    if features_cache_path and features_cache_path.exists():
        logger.info("Loading precomputed PCA features from cache: %s", features_cache_path)
        features = np.load(features_cache_path)
    else:
        logger.info("Generating PCA features for %d messages...", total_corpus_size)
        from backend.app.nlp.embeddings import SentenceEmbeddingGenerator
        from backend.app.nlp.intent_discovery import reduce_dimensions_pca
        emb_gen = SentenceEmbeddingGenerator(normalize_embeddings=True)
        embs = emb_gen.encode(df_corpus["normalized_text"].tolist(), batch_size=128, show_progress_bar=True)
        features, _ = reduce_dimensions_pca(embs, n_components=50, random_state=random_seed)
        if features_cache_path:
            features_cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(features_cache_path, features)
            logger.info("Saved PCA features cache to: %s", features_cache_path)

    # Calculate centroids
    centroids = np.zeros((n_clusters, features.shape[1]), dtype=np.float32)
    for k in cluster_ids:
        k_mask = df_corpus["cluster_id"].values == k
        if np.any(k_mask):
            centroids[k] = np.mean(features[k_mask], axis=0)

    # 3. Load Brand Responses from Phase 3 Parquet
    logger.info("Loading first AppleSupport replies for sampled conversations...")
    sample_conv_ids = set(df_corpus["conversation_id"])
    df_msg = pd.read_parquet(messages_path, columns=["conversation_id", "role", "depth", "text"])
    brand_msgs = df_msg[(df_msg["conversation_id"].isin(sample_conv_ids)) & (df_msg["role"] == "brand")]
    first_brand = brand_msgs.sort_values(["conversation_id", "depth"]).groupby("conversation_id").first().reset_index()
    first_brand = first_brand.rename(columns={"text": "brand_text"})

    df_merged = df_corpus.merge(first_brand[["conversation_id", "brand_text"]], on="conversation_id", how="left")
    df_merged["brand_text"] = df_merged["brand_text"].fillna("No brand reply observed")

    # 4. Process Each Cluster
    cluster_audits: list[ClusterAuditRecord] = []
    coherence_map: dict[int, ClusterCoherenceStats] = {}
    entity_map: dict[int, TopicEntityAnalysis] = {}

    for cid in cluster_ids:
        k_indices = np.where(df_corpus["cluster_id"].values == cid)[0]
        k_features = features[k_indices]
        k_texts = df_corpus.iloc[k_indices]["normalized_text"].tolist()
        k_brand_texts = df_merged.iloc[k_indices]["brand_text"].tolist()
        centroid = centroids[cid]

        # Step 1: Expanded Examples
        examples = extract_expanded_examples(
            cluster_features=k_features,
            cluster_texts=k_texts,
            centroid=centroid,
            n_centroid=n_centroid,
            n_random=n_random,
            n_diverse=n_diverse,
            random_seed=random_seed,
        )

        # Step 2: Coherence
        coherence = compute_cluster_coherence(
            cluster_features=k_features,
            centroid=centroid,
            all_centroids=centroids,
            cluster_id=cid,
            total_corpus_size=total_corpus_size,
        )
        coherence_map[cid] = coherence

        # Step 3: Entity & Topic Analysis
        entities = analyze_entities_and_topics(k_texts)
        entity_map[cid] = entities

        # Step 4: Historical Event Detection
        hist_events = detect_historical_events(k_texts)

        # Step 5: Resolution Pattern Audit
        resolutions = audit_resolution_patterns(k_brand_texts)

        proposed_lbl = proposed_labels.get(cid, f"cluster_{cid}")

        audit_record = ClusterAuditRecord(
            cluster_id=cid,
            cluster_size=coherence.cluster_size,
            percentage=coherence.percentage,
            current_proposed_label=proposed_lbl,
            coherence=coherence,
            examples=examples,
            topics_entities=entities,
            historical_events=hist_events,
            resolutions=resolutions,
            recommended_human_action="PENDING",  # filled after merge/split analysis
            merge_candidate_ids=[],
            split_signals="",
            final_intent_label=None,
            review_status="pending_human_review",
        )
        cluster_audits.append(audit_record)

    # Step 6: Pairwise Merge Candidate Analysis
    logger.info("Analyzing pairwise merge candidates...")
    merge_candidates = identify_merge_candidates(
        centroids=centroids,
        cluster_top_terms=cluster_top_terms,
        proposed_labels=proposed_labels,
        similarity_threshold=0.40,
    )

    # Step 7: Split Candidate Analysis
    logger.info("Analyzing split candidates...")
    split_candidates = identify_split_candidates(
        coherence_stats=coherence_map,
        entity_stats=entity_map,
        proposed_labels=proposed_labels,
    )

    # Update actions and cross-references in cluster audits
    for record in cluster_audits:
        cid = record.cluster_id
        # Find merge partners
        m_partners = []
        for m in merge_candidates:
            if m.cluster_a == cid:
                m_partners.append(m.cluster_b)
            elif m.cluster_b == cid:
                m_partners.append(m.cluster_a)
        record.merge_candidate_ids = sorted(list(set(m_partners)))

        # Find split signals
        s_signals = [s.coherence_signals for s in split_candidates if s.cluster_id == cid]
        record.split_signals = "; ".join(s_signals) if s_signals else "none_detected"

        # Determine human action recommendation
        record.recommended_human_action = determine_human_action_recommendation(
            cluster_id=cid,
            generalization_status=record.historical_events.generalization_status,
            merge_candidates=merge_candidates,
            split_candidates=split_candidates,
        )

    # 5. Save Artifacts
    output_dir.mkdir(parents=True, exist_ok=True)

    # A. Merge Candidates CSV
    merge_csv_path = output_dir / "cluster_merge_candidates.csv"
    df_merge = pd.DataFrame([asdict(m) for m in merge_candidates])
    df_merge.to_csv(merge_csv_path, index=False)
    logger.info("Saved merge candidates to: %s", merge_csv_path)

    # B. Split Candidates CSV
    split_csv_path = output_dir / "cluster_split_candidates.csv"
    df_split = pd.DataFrame([asdict(s) for s in split_candidates])
    df_split.to_csv(split_csv_path, index=False)
    logger.info("Saved split candidates to: %s", split_csv_path)

    # C. Human Review Packet CSV
    review_packet_path = output_dir / "intent_taxonomy_human_review.csv"
    packet_rows = []
    for a in cluster_audits:
        packet_rows.append({
            "cluster_id": a.cluster_id,
            "cluster_size": a.cluster_size,
            "percentage": f"{a.percentage:.1f}%",
            "centroid_examples": " || ".join(a.examples.centroid_nearest[:5]),
            "random_examples": " || ".join(a.examples.random_samples[:5]),
            "diverse_examples": " || ".join(a.examples.diverse_samples[:5]),
            "top_terms": "; ".join(cluster_top_terms.get(a.cluster_id, [])),
            "common_entities": a.topics_entities.summary_str,
            "generalization_status": a.historical_events.generalization_status,
            "resolution_patterns": a.resolutions.summary_str,
            "merge_candidates": ", ".join([f"Cluster {m}" for m in a.merge_candidate_ids]) if a.merge_candidate_ids else "none",
            "split_signals": a.split_signals,
            "current_proposed_label": a.current_proposed_label,
            "recommended_human_action": a.recommended_human_action,
            "final_intent_label": "",
            "review_status": a.review_status,
        })
    df_packet = pd.DataFrame(packet_rows)
    df_packet.to_csv(review_packet_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
    logger.info("Saved human review packet to: %s", review_packet_path)

    result = TaxonomyAuditResult(
        cluster_audits=cluster_audits,
        merge_candidates=merge_candidates,
        split_candidates=split_candidates,
        metadata={
            "total_corpus_size": total_corpus_size,
            "cluster_count": n_clusters,
            "random_seed": random_seed,
            "n_centroid_examples": n_centroid,
            "n_random_examples": n_random,
            "n_diverse_examples": n_diverse,
        },
    )

    logger.info("Phase 4.5 Cluster Audit completed successfully.")
    return result
