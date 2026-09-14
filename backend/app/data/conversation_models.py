"""
SupportGraph AI — Phase 3 Conversation Models

Data structures for reconstructed customer-support conversations, messages,
edges, anomalies, and aggregate statistics.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class ConversationMessage:
    """Individual tweet message within a reconstructed conversation thread."""
    tweet_id: int
    author_id: str
    role: str  # "customer" | "brand"
    inbound: bool
    created_at: str
    text: str
    parent_tweet_id: Optional[int] = None
    response_tweet_ids: list[int] = field(default_factory=list)
    depth: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tweet_id": self.tweet_id,
            "author_id": self.author_id,
            "role": self.role,
            "inbound": self.inbound,
            "created_at": self.created_at,
            "text": self.text,
            "parent_tweet_id": self.parent_tweet_id,
            "response_tweet_ids": list(self.response_tweet_ids),
            "depth": self.depth,
        }


@dataclass
class ConversationEdge:
    """Directed parent -> child reply edge in the conversation graph."""
    conversation_id: str
    parent_tweet_id: int
    child_tweet_id: int
    relationship_source: str  # "in_response_to" | "response_tweet" | "both"
    relationship_valid: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "parent_tweet_id": self.parent_tweet_id,
            "child_tweet_id": self.child_tweet_id,
            "relationship_source": self.relationship_source,
            "relationship_valid": self.relationship_valid,
        }


@dataclass
class ConversationAnomaly:
    """Anomalous or broken relationship observed during graph reconstruction."""
    conversation_id: str
    tweet_id: Optional[int]
    anomaly_type: str
    details: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "tweet_id": self.tweet_id,
            "anomaly_type": self.anomaly_type,
            "details": self.details,
        }


@dataclass
class Conversation:
    """A fully reconstructed conversation component."""
    conversation_id: str
    selected_brand: str
    root_tweet_id: int
    message_count: int
    thread_type: str
    has_branching: bool
    has_anomalies: bool
    quality_status: str  # "high" | "medium" | "low"
    messages: list[ConversationMessage] = field(default_factory=list)
    edges: list[ConversationEdge] = field(default_factory=list)
    anomalies: list[ConversationAnomaly] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "selected_brand": self.selected_brand,
            "root_tweet_id": self.root_tweet_id,
            "message_count": self.message_count,
            "thread_type": self.thread_type,
            "has_branching": self.has_branching,
            "has_anomalies": self.has_anomalies,
            "quality_status": self.quality_status,
            "messages": [m.to_dict() for m in self.messages],
            "edges": [{"parent": e.parent_tweet_id, "child": e.child_tweet_id} for e in self.edges],
            "anomalies": [a.to_dict() for a in self.anomalies],
        }


@dataclass
class ConversationStatistics:
    """Aggregate statistics for Phase 3 conversation reconstruction."""
    total_selected_brand_tweets: int = 0
    total_candidate_conversations: int = 0
    total_valid_conversations: int = 0
    total_messages: int = 0
    mean_messages_per_conversation: float = 0.0
    median_messages_per_conversation: float = 0.0
    maximum_messages_per_conversation: int = 0
    thread_type_distribution: dict[str, int] = field(default_factory=dict)
    quality_distribution: dict[str, int] = field(default_factory=dict)
    branched_conversations: int = 0
    conversations_with_anomalies: int = 0
    orphan_messages: int = 0
    broken_parent_relationships: int = 0
    cycles_detected: int = 0
    timestamp_anomalies: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
