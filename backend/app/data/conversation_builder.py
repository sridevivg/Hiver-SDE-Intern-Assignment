"""
SupportGraph AI — Phase 3 Conversation Reconstruction Pipeline

Transforms tweet-level records into structured, ordered, typed conversation threads
grounded in historical support interactions for the selected brand.
"""
from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from backend.app.data.conversation_models import (
    Conversation,
    ConversationAnomaly,
    ConversationEdge,
    ConversationMessage,
    ConversationStatistics,
)

logger = logging.getLogger(__name__)

# Required columns from Phase 1 schema
REQUIRED_COLUMNS = {
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
}


def load_selected_brand(selected_brand_file: Path | str) -> str:
    """
    Load the selected brand from the Phase 2 artifact.

    Args:
        selected_brand_file: Path to data/interim/selected_brand.json.

    Returns:
        Selected brand name (e.g. 'AppleSupport').

    Raises:
        FileNotFoundError: If the artifact does not exist.
        ValueError: If the file does not contain a valid 'selected_brand' field.
    """
    p = Path(selected_brand_file)
    if not p.is_file():
        raise FileNotFoundError(
            f"Selected brand artifact not found: {p}\n"
            "Please run Phase 2 brand selection first:\n"
            "  python -m backend.scripts.select_brand"
        )
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    brand = data.get("selected_brand")
    if not brand or not isinstance(brand, str):
        raise ValueError(f"Invalid selected_brand in {p}: {brand}")
    logger.info("Loaded selected brand from %s: %s", p, brand)
    return brand


def validate_dataset_schema(df: pd.DataFrame) -> None:
    """
    Validate that the DataFrame contains all expected columns.

    Raises:
        ValueError: If any required column is missing.
    """
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset schema missing required columns: {sorted(missing)}")
    logger.debug("Dataset schema validated successfully against required columns.")


def parse_response_tweet_ids(val: Any) -> list[int]:
    """
    Parse the response_tweet_id field into a list of integer tweet IDs.
    Handles None/NaN, single integers, floats, comma-separated strings.
    """
    if val is None or pd.isna(val):
        return []
    val_str = str(val).strip()
    if not val_str:
        return []
    parts = val_str.split(",")
    res = []
    for part in parts:
        part = part.strip()
        if part.endswith(".0"):
            part = part[:-2]
        if part.isdigit():
            res.append(int(part))
    return res


def parse_twitter_timestamp(ts: str) -> Optional[datetime]:
    """Parse Twitter created_at timestamp string (e.g. 'Tue Oct 31 22:10:47 +0000 2017')."""
    if not ts or pd.isna(ts):
        return None
    try:
        return datetime.strptime(str(ts).strip(), "%a %b %d %H:%M:%S %z %Y")
    except Exception:
        return None


def build_deterministic_conversation_id(selected_brand: str, root_tweet_id: int) -> str:
    """
    Deterministic conversation ID based on brand and root tweet ID.
    Example: conv_AppleSupport_12345
    """
    return f"conv_{selected_brand}_{root_tweet_id}"


def classify_thread_type(
    messages: list[ConversationMessage],
    has_branching: bool,
    has_breaking_anomalies: bool,
) -> str:
    """
    Classify observed conversation structure.

    Categories:
      - customer_brand: Exactly Customer -> Brand
      - customer_brand_customer: Exactly Customer -> Brand -> Customer
      - customer_brand_customer_brand: Exactly Customer -> Brand -> Customer -> Brand
      - multi_turn: Longer alternating sequence (>4 messages) or other multi-turn
      - branched: Tree structure with branching (>1 child for at least one message)
      - incomplete: Broken relationships or cycles prevent full linear reconstruction
    """
    if has_breaking_anomalies:
        return "incomplete"
    if has_branching:
        return "branched"

    roles = [m.role for m in messages]
    if roles == ["customer", "brand"]:
        return "customer_brand"
    elif roles == ["customer", "brand", "customer"]:
        return "customer_brand_customer"
    elif roles == ["customer", "brand", "customer", "brand"]:
        return "customer_brand_customer_brand"
    else:
        return "multi_turn"


def assign_quality_status(
    conversation_id: str,
    message_count: int,
    roles: set[str],
    has_breaking_anomalies: bool,
    has_minor_anomalies: bool,
) -> str:
    """
    Deterministic quality classification:
      - HIGH: No anomalies, unbroken parent chains, at least 2 messages, contains customer and brand.
      - MEDIUM: Contains customer and brand, minor anomalies (timestamp or missing child pointers), graph still usable.
      - LOW: Missing parent tweet, cycle, fewer than 2 messages, or missing either role.
    """
    if message_count < 2 or "customer" not in roles or "brand" not in roles:
        return "low"
    if has_breaking_anomalies:
        return "low"
    if has_minor_anomalies:
        return "medium"
    return "high"


class ConversationBuilder:
    """
    High-performance, memory-efficient conversation reconstruction builder.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        selected_brand: str,
        max_depth: int = 50,
        sample_limit: Optional[int] = None,
    ) -> None:
        self.df = df
        self.selected_brand = selected_brand
        self.max_depth = max_depth
        self.sample_limit = sample_limit

        # Validate schema
        validate_dataset_schema(df)

        # Quick lookup for all tweet IDs in the full dataset
        self.all_dataset_tweet_ids = set(df["tweet_id"].astype(int))

        # Filter brand tweets
        brand_mask = df["author_id"] == selected_brand
        self.brand_df = df[brand_mask]
        self.brand_tweet_ids = set(self.brand_df["tweet_id"].astype(int))
        logger.info(
            "Initialized ConversationBuilder for brand '%s': %d total brand tweets",
            selected_brand,
            len(self.brand_tweet_ids),
        )

    def extract_brand_connected_subgraph(self) -> dict[int, dict[str, Any]]:
        """
        Extract all tweets that are part of connected components containing
        at least one selected brand tweet.

        Returns:
            Dictionary of tweet_id -> row dictionary with parsed fields.
        """
        logger.info("Indexing parent-child reply relationships across dataset...")
        has_parent = self.df[self.df["in_response_to_tweet_id"].notna()]
        parents = has_parent["in_response_to_tweet_id"].astype(int).values
        children = has_parent["tweet_id"].astype(int).values

        # Build bidirectional adjacency for component discovery
        # Only connect edges where parent exists in the dataset
        adj: dict[int, list[int]] = defaultdict(list)
        for p, c in zip(parents, children):
            if p in self.all_dataset_tweet_ids:
                adj[p].append(c)
                adj[c].append(p)

        logger.info("Traversing connected components for %d brand tweets...", len(self.brand_tweet_ids))
        visited: set[int] = set()
        connected_tweet_ids: set[int] = set()

        for bid in self.brand_tweet_ids:
            if bid in visited:
                continue
            comp = []
            q = deque([bid])
            visited.add(bid)
            while q:
                curr = q.popleft()
                comp.append(curr)
                for neighbor in adj.get(curr, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        q.append(neighbor)
            connected_tweet_ids.update(comp)

        logger.info(
            "Found %d total tweets connected to brand '%s' across components",
            len(connected_tweet_ids),
            self.selected_brand,
        )

        # Subset dataframe to connected tweets only
        subset_df = self.df[self.df["tweet_id"].isin(connected_tweet_ids)]
        logger.info("Building metadata dictionary for %d subset tweets...", len(subset_df))

        lookup: dict[int, dict[str, Any]] = {}
        for row in subset_df.itertuples(index=False):
            tid = int(row.tweet_id)
            pid = int(row.in_response_to_tweet_id) if pd.notna(row.in_response_to_tweet_id) else None
            r_ids = parse_response_tweet_ids(row.response_tweet_id)
            ts_dt = parse_twitter_timestamp(row.created_at)

            is_brand = (row.author_id == self.selected_brand)
            role = "brand" if is_brand else "customer"

            lookup[tid] = {
                "tweet_id": tid,
                "author_id": str(row.author_id),
                "role": role,
                "inbound": bool(row.inbound),
                "created_at": str(row.created_at),
                "created_at_dt": ts_dt,
                "text": str(row.text) if pd.notna(row.text) else "",
                "parent_tweet_id": pid,
                "response_tweet_ids": r_ids,
            }

        return lookup

    def build_conversations(self) -> tuple[list[Conversation], ConversationStatistics]:
        """
        Execute full conversation reconstruction pipeline.
        """
        tweet_lookup = self.extract_brand_connected_subgraph()
        stats = ConversationStatistics()
        stats.total_selected_brand_tweets = len(self.brand_tweet_ids)

        # Build component groups within the subset
        # An edge exists if child has parent in tweet_lookup
        children_map: dict[int, list[int]] = defaultdict(list)
        undirected_adj: dict[int, list[int]] = defaultdict(list)

        # Global forward response lookup for edge source verification
        for tid, data in tweet_lookup.items():
            pid = data["parent_tweet_id"]
            if pid is not None and pid in tweet_lookup:
                children_map[pid].append(tid)
                undirected_adj[pid].append(tid)
                undirected_adj[tid].append(pid)

        # Find connected components
        visited: set[int] = set()
        components: list[list[int]] = []

        for tid in tweet_lookup:
            if tid in visited:
                continue
            comp = []
            q = deque([tid])
            visited.add(tid)
            while q:
                curr = q.popleft()
                comp.append(curr)
                for neighbor in undirected_adj.get(curr, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        q.append(neighbor)
            components.append(comp)

        stats.total_candidate_conversations = len(components)
        logger.info("Formed %d candidate components", len(components))

        # Optional sample limit for testing
        if self.sample_limit is not None and self.sample_limit > 0:
            components = components[: self.sample_limit]
            logger.info("Applied sample_limit: evaluating first %d components", len(components))

        reconstructed_conversations: list[Conversation] = []
        thread_type_counts: dict[str, int] = defaultdict(int)
        quality_counts: dict[str, int] = defaultdict(int)

        for comp_tids in components:
            comp_set = set(comp_tids)
            roles_present = {tweet_lookup[tid]["role"] for tid in comp_tids}

            # STEP 10: Filter components: must contain both customer and selected brand
            if "brand" not in roles_present or "customer" not in roles_present:
                continue

            # Identify roots: nodes in component with no parent in the component
            candidate_roots: list[int] = []
            broken_parent_roots: list[int] = []

            for tid in comp_tids:
                pid = tweet_lookup[tid]["parent_tweet_id"]
                if pid is None:
                    candidate_roots.append(tid)
                elif pid not in comp_set:
                    # Parent points outside the component or outside the dataset
                    candidate_roots.append(tid)
                    broken_parent_roots.append(tid)

            # Choose primary root: earliest timestamp, then lowest tweet_id
            if not candidate_roots:
                # Component is purely cyclic with no in-degree 0 node
                primary_root = min(comp_tids, key=lambda t: (tweet_lookup[t]["created_at_dt"] or datetime.max, t))
            else:
                primary_root = min(candidate_roots, key=lambda t: (tweet_lookup[t]["created_at_dt"] or datetime.max, t))

            conv_id = build_deterministic_conversation_id(self.selected_brand, primary_root)
            anomalies: list[ConversationAnomaly] = []
            breaking_anomaly = False
            minor_anomaly = False

            # Check broken parents
            for tid in broken_parent_roots:
                pid = tweet_lookup[tid]["parent_tweet_id"]
                stats.broken_parent_relationships += 1
                breaking_anomaly = True
                anomalies.append(
                    ConversationAnomaly(
                        conversation_id=conv_id,
                        tweet_id=tid,
                        anomaly_type="missing_parent_tweet",
                        details=f"Tweet {tid} references parent {pid} which is not present in the dataset.",
                    )
                )

            # Check self-references & invalid response IDs
            for tid in comp_tids:
                data = tweet_lookup[tid]
                if data["parent_tweet_id"] == tid:
                    breaking_anomaly = True
                    anomalies.append(
                        ConversationAnomaly(
                            conversation_id=conv_id,
                            tweet_id=tid,
                            anomaly_type="self_reference",
                            details=f"Tweet {tid} references itself as parent.",
                        )
                    )
                # Check response_tweet_id targets that don't exist in dataset
                for r_id in data["response_tweet_ids"]:
                    if r_id not in self.all_dataset_tweet_ids:
                        minor_anomaly = True
                        anomalies.append(
                            ConversationAnomaly(
                                conversation_id=conv_id,
                                tweet_id=tid,
                                anomaly_type="missing_child_tweet",
                                details=f"Tweet {tid} lists response {r_id} not present in dataset.",
                            )
                        )

            # Cycle detection and BFS traversal from primary_root
            visited_traversal: set[int] = set()
            depth_map: dict[int, int] = {primary_root: 0}
            ordered_messages: list[ConversationMessage] = []
            has_branching = False

            q_traverse = deque([primary_root])
            visited_traversal.add(primary_root)

            # Detect branching within component
            for tid in comp_tids:
                valid_children = [c for c in children_map.get(tid, []) if c in comp_set]
                if len(valid_children) > 1:
                    has_branching = True

            # BFS canonical traversal
            while q_traverse:
                curr = q_traverse.popleft()
                curr_depth = depth_map[curr]
                c_data = tweet_lookup[curr]

                ordered_messages.append(
                    ConversationMessage(
                        tweet_id=curr,
                        author_id=c_data["author_id"],
                        role=c_data["role"],
                        inbound=c_data["inbound"],
                        created_at=c_data["created_at"],
                        text=c_data["text"],
                        parent_tweet_id=c_data["parent_tweet_id"],
                        response_tweet_ids=c_data["response_tweet_ids"],
                        depth=curr_depth,
                    )
                )

                # Order children: secondary by timestamp, tertiary by tweet_id
                children_of_curr = [c for c in children_map.get(curr, []) if c in comp_set]
                children_of_curr.sort(
                    key=lambda c: (tweet_lookup[c]["created_at_dt"] or datetime.max, c)
                )

                for ch in children_of_curr:
                    # Check timestamp anomaly
                    p_dt = c_data["created_at_dt"]
                    ch_dt = tweet_lookup[ch]["created_at_dt"]
                    if p_dt and ch_dt and ch_dt < p_dt:
                        stats.timestamp_anomalies += 1
                        minor_anomaly = True
                        anomalies.append(
                            ConversationAnomaly(
                                conversation_id=conv_id,
                                tweet_id=ch,
                                anomaly_type="timestamp_anomaly",
                                details=f"Child {ch} timestamp ({ch_dt}) is earlier than parent {curr} timestamp ({p_dt}).",
                            )
                        )

                    if ch in visited_traversal:
                        # Cycle detected!
                        stats.cycles_detected += 1
                        breaking_anomaly = True
                        anomalies.append(
                            ConversationAnomaly(
                                conversation_id=conv_id,
                                tweet_id=ch,
                                anomaly_type="cyclic_relationship",
                                details=f"Cycle detected: child {ch} was already visited in conversation traversal.",
                            )
                        )
                        continue

                    if curr_depth + 1 <= self.max_depth:
                        depth_map[ch] = curr_depth + 1
                        visited_traversal.add(ch)
                        q_traverse.append(ch)

            # Include any disconnected / unvisited nodes in component (orphans in sub-graph)
            unvisited_nodes = comp_set - visited_traversal
            if unvisited_nodes:
                minor_anomaly = True
                stats.orphan_messages += len(unvisited_nodes)
                for orphan_id in sorted(unvisited_nodes):
                    o_data = tweet_lookup[orphan_id]
                    depth_map[orphan_id] = 99
                    ordered_messages.append(
                        ConversationMessage(
                            tweet_id=orphan_id,
                            author_id=o_data["author_id"],
                            role=o_data["role"],
                            inbound=o_data["inbound"],
                            created_at=o_data["created_at"],
                            text=o_data["text"],
                            parent_tweet_id=o_data["parent_tweet_id"],
                            response_tweet_ids=o_data["response_tweet_ids"],
                            depth=99,
                        )
                    )

            # Construct Edges
            edges: list[ConversationEdge] = []
            seen_edges: set[tuple[int, int]] = set()

            for tid in comp_tids:
                pid = tweet_lookup[tid]["parent_tweet_id"]
                if pid is not None and pid in comp_set:
                    edge_key = (pid, tid)
                    if edge_key in seen_edges:
                        anomalies.append(
                            ConversationAnomaly(
                                conversation_id=conv_id,
                                tweet_id=tid,
                                anomaly_type="duplicate_relationship",
                                details=f"Duplicate edge {pid} -> {tid}.",
                            )
                        )
                        continue
                    seen_edges.add(edge_key)

                    # Determine relationship_source
                    p_responses = tweet_lookup[pid]["response_tweet_ids"]
                    in_resp_flag = True
                    resp_flag = (tid in p_responses)

                    if in_resp_flag and resp_flag:
                        rel_source = "both"
                    elif in_resp_flag:
                        rel_source = "in_response_to"
                    else:
                        rel_source = "response_tweet"

                    edges.append(
                        ConversationEdge(
                            conversation_id=conv_id,
                            parent_tweet_id=pid,
                            child_tweet_id=tid,
                            relationship_source=rel_source,
                            relationship_valid=True,
                        )
                    )

            has_anomalies = len(anomalies) > 0
            if has_anomalies:
                stats.conversations_with_anomalies += 1
            if has_branching:
                stats.branched_conversations += 1

            # Classify thread type
            thread_type = classify_thread_type(
                ordered_messages,
                has_branching=has_branching,
                has_breaking_anomalies=breaking_anomaly,
            )
            thread_type_counts[thread_type] += 1

            # Quality status
            quality = assign_quality_status(
                conv_id,
                message_count=len(ordered_messages),
                roles=roles_present,
                has_breaking_anomalies=breaking_anomaly,
                has_minor_anomalies=minor_anomaly,
            )
            quality_counts[quality] += 1

            conv = Conversation(
                conversation_id=conv_id,
                selected_brand=self.selected_brand,
                root_tweet_id=primary_root,
                message_count=len(ordered_messages),
                thread_type=thread_type,
                has_branching=has_branching,
                has_anomalies=has_anomalies,
                quality_status=quality,
                messages=ordered_messages,
                edges=edges,
                anomalies=anomalies,
            )
            reconstructed_conversations.append(conv)

        stats.total_valid_conversations = len(reconstructed_conversations)
        stats.total_messages = sum(c.message_count for c in reconstructed_conversations)
        counts = [c.message_count for c in reconstructed_conversations]
        if counts:
            stats.mean_messages_per_conversation = round(float(np.mean(counts)), 2)
            stats.median_messages_per_conversation = round(float(np.median(counts)), 1)
            stats.maximum_messages_per_conversation = int(max(counts))

        stats.thread_type_distribution = dict(thread_type_counts)
        stats.quality_distribution = dict(quality_counts)

        logger.info(
            "Reconstruction complete: %d valid conversations, %d total messages (mean=%.2f, max=%d)",
            stats.total_valid_conversations,
            stats.total_messages,
            stats.mean_messages_per_conversation,
            stats.maximum_messages_per_conversation,
        )

        return reconstructed_conversations, stats


def save_reconstructed_conversations(
    conversations: list[Conversation],
    stats: ConversationStatistics,
    output_dir: Path | str,
    include_low_quality: bool = True,
) -> dict[str, Path]:
    """
    Save all Phase 3 outputs:
      1. conversations.jsonl
      2. conversation_messages.parquet
      3. conversation_edges.csv
      4. conversation_anomalies.csv
      5. conversation_statistics.json
      6. conversation_quality_summary.csv
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: dict[str, Path] = {}

    target_conversations = conversations
    if not include_low_quality:
        target_conversations = [c for c in conversations if c.quality_status != "low"]
        logger.info("Filtered out low quality: %d remaining conversations", len(target_conversations))

    # 1. conversations.jsonl
    jsonl_path = out_dir / "conversations.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for c in target_conversations:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    saved_paths["conversations_jsonl"] = jsonl_path
    logger.info("Saved: %s", jsonl_path)

    # 2. conversation_messages.parquet
    message_rows = []
    for c in target_conversations:
        for m in c.messages:
            message_rows.append({
                "conversation_id": c.conversation_id,
                "tweet_id": m.tweet_id,
                "author_id": m.author_id,
                "role": m.role,
                "inbound": m.inbound,
                "created_at": m.created_at,
                "text": m.text,
                "parent_tweet_id": m.parent_tweet_id,
                "depth": m.depth,
                "thread_type": c.thread_type,
                "has_branching": c.has_branching,
                "has_anomalies": c.has_anomalies,
            })
    parquet_path = out_dir / "conversation_messages.parquet"
    msg_df = pd.DataFrame(message_rows)
    if not msg_df.empty:
        msg_df["parent_tweet_id"] = msg_df["parent_tweet_id"].astype("Int64")
        msg_df["tweet_id"] = msg_df["tweet_id"].astype("int64")
        msg_df["depth"] = msg_df["depth"].astype("int32")
    msg_df.to_parquet(parquet_path, index=False, engine="pyarrow")
    saved_paths["conversation_messages_parquet"] = parquet_path
    logger.info("Saved: %s", parquet_path)

    # 3. conversation_edges.csv
    edge_rows = []
    for c in target_conversations:
        for e in c.edges:
            edge_rows.append(e.to_dict())
    edges_path = out_dir / "conversation_edges.csv"
    edge_df = pd.DataFrame(edge_rows)
    if edge_df.empty:
        edge_df = pd.DataFrame(columns=[
            "conversation_id", "parent_tweet_id", "child_tweet_id",
            "relationship_source", "relationship_valid"
        ])
    edge_df.to_csv(edges_path, index=False)
    saved_paths["conversation_edges_csv"] = edges_path
    logger.info("Saved: %s", edges_path)

    # 4. conversation_anomalies.csv
    anomaly_rows = []
    for c in target_conversations:
        for a in c.anomalies:
            anomaly_rows.append(a.to_dict())
    anomalies_path = out_dir / "conversation_anomalies.csv"
    anomaly_df = pd.DataFrame(anomaly_rows)
    if anomaly_df.empty:
        anomaly_df = pd.DataFrame(columns=["conversation_id", "tweet_id", "anomaly_type", "details"])
    anomaly_df.to_csv(anomalies_path, index=False)
    saved_paths["conversation_anomalies_csv"] = anomalies_path
    logger.info("Saved: %s", anomalies_path)

    # 5. conversation_statistics.json
    stats_path = out_dir / "conversation_statistics.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats.to_dict(), f, indent=2)
    saved_paths["conversation_statistics_json"] = stats_path
    logger.info("Saved: %s", stats_path)

    # 6. conversation_quality_summary.csv
    quality_rows = [
        {
            "conversation_id": c.conversation_id,
            "message_count": c.message_count,
            "thread_type": c.thread_type,
            "has_branching": c.has_branching,
            "has_anomalies": c.has_anomalies,
            "quality_status": c.quality_status,
        }
        for c in target_conversations
    ]
    quality_path = out_dir / "conversation_quality_summary.csv"
    quality_df = pd.DataFrame(quality_rows)
    quality_df.to_csv(quality_path, index=False)
    saved_paths["conversation_quality_summary_csv"] = quality_path
    logger.info("Saved: %s", quality_path)

    return saved_paths
