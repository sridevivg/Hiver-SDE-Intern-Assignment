"""
SupportGraph AI — Phase 3 Unit Tests: Conversation Reconstruction

Tests cover:
  1. Customer -> Brand conversation
  2. Customer -> Brand -> Customer conversation
  3. Customer -> Brand -> Customer -> Brand conversation
  4. Multiple conversations in a single dataset
  5. Branching replies (single parent with multiple children)
  6. Missing parent tweet (broken parent pointer)
  7. Orphan message detection
  8. Self-reference handling
  9. Cycle detection
  10. Duplicate edge handling
  11. Timestamp ordering anomaly handling
  12. Deterministic conversation ID generation
  13. Thread classification logic
  14. Quality status classification logic
  15. Brand filtering (components with other brands or customer only)
  16. Empty dataset handling

All tests use SYNTHETIC DataFrames only — never the 492 MB real dataset.
"""
from __future__ import annotations

from pathlib import Path
import sys
import pytest
import pandas as pd

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.data.conversation_builder import (
    ConversationBuilder,
    build_deterministic_conversation_id,
    classify_thread_type,
    assign_quality_status,
    parse_response_tweet_ids,
    validate_dataset_schema,
)
from backend.app.data.conversation_models import ConversationMessage


# ---------------------------------------------------------------------------
# Helpers for creating synthetic DataFrames
# ---------------------------------------------------------------------------
def make_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in ["in_response_to_tweet_id", "response_tweet_id"]:
        if col not in df.columns:
            df[col] = None
    return df


# ---------------------------------------------------------------------------
# Test 1: Customer -> Brand
# ---------------------------------------------------------------------------
def test_customer_to_brand_conversation():
    rows = [
        {"tweet_id": 1, "author_id": "cust1", "inbound": True, "created_at": "Tue Oct 31 20:00:00 +0000 2017", "text": "Help me", "response_tweet_id": "2", "in_response_to_tweet_id": None},
        {"tweet_id": 2, "author_id": "AppleSupport", "inbound": False, "created_at": "Tue Oct 31 20:05:00 +0000 2017", "text": "We can help", "response_tweet_id": None, "in_response_to_tweet_id": 1.0},
    ]
    builder = ConversationBuilder(make_df(rows), selected_brand="AppleSupport")
    convs, stats = builder.build_conversations()

    assert len(convs) == 1
    c = convs[0]
    assert c.thread_type == "customer_brand"
    assert c.message_count == 2
    assert c.root_tweet_id == 1
    assert c.has_branching is False
    assert c.has_anomalies is False
    assert c.quality_status == "high"


# ---------------------------------------------------------------------------
# Test 2: Customer -> Brand -> Customer
# ---------------------------------------------------------------------------
def test_customer_brand_customer_conversation():
    rows = [
        {"tweet_id": 10, "author_id": "cust1", "inbound": True, "created_at": "Tue Oct 31 20:00:00 +0000 2017", "text": "Phone broke", "response_tweet_id": "11", "in_response_to_tweet_id": None},
        {"tweet_id": 11, "author_id": "AppleSupport", "inbound": False, "created_at": "Tue Oct 31 20:05:00 +0000 2017", "text": "DM us your iOS", "response_tweet_id": "12", "in_response_to_tweet_id": 10.0},
        {"tweet_id": 12, "author_id": "cust1", "inbound": True, "created_at": "Tue Oct 31 20:10:00 +0000 2017", "text": "Sent the DM", "response_tweet_id": None, "in_response_to_tweet_id": 11.0},
    ]
    builder = ConversationBuilder(make_df(rows), selected_brand="AppleSupport")
    convs, stats = builder.build_conversations()

    assert len(convs) == 1
    c = convs[0]
    assert c.thread_type == "customer_brand_customer"
    assert c.message_count == 3
    assert c.has_branching is False
    assert c.quality_status == "high"


# ---------------------------------------------------------------------------
# Test 3: Customer -> Brand -> Customer -> Brand
# ---------------------------------------------------------------------------
def test_customer_brand_customer_brand_conversation():
    rows = [
        {"tweet_id": 20, "author_id": "cust1", "inbound": True, "created_at": "Tue Oct 31 20:00:00 +0000 2017", "text": "Mac won't turn on", "response_tweet_id": "21", "in_response_to_tweet_id": None},
        {"tweet_id": 21, "author_id": "AppleSupport", "inbound": False, "created_at": "Tue Oct 31 20:05:00 +0000 2017", "text": "Try resetting SMC", "response_tweet_id": "22", "in_response_to_tweet_id": 20.0},
        {"tweet_id": 22, "author_id": "cust1", "inbound": True, "created_at": "Tue Oct 31 20:10:00 +0000 2017", "text": "It worked thank you", "response_tweet_id": "23", "in_response_to_tweet_id": 21.0},
        {"tweet_id": 23, "author_id": "AppleSupport", "inbound": False, "created_at": "Tue Oct 31 20:15:00 +0000 2017", "text": "You are welcome!", "response_tweet_id": None, "in_response_to_tweet_id": 22.0},
    ]
    builder = ConversationBuilder(make_df(rows), selected_brand="AppleSupport")
    convs, stats = builder.build_conversations()

    assert len(convs) == 1
    c = convs[0]
    assert c.thread_type == "customer_brand_customer_brand"
    assert c.message_count == 4
    assert c.quality_status == "high"


# ---------------------------------------------------------------------------
# Test 4: Multiple conversations
# ---------------------------------------------------------------------------
def test_multiple_conversations():
    rows = [
        {"tweet_id": 1, "author_id": "cust1", "inbound": True, "created_at": "Tue Oct 31 20:00:00 +0000 2017", "text": "Issue 1", "response_tweet_id": "2", "in_response_to_tweet_id": None},
        {"tweet_id": 2, "author_id": "AppleSupport", "inbound": False, "created_at": "Tue Oct 31 20:05:00 +0000 2017", "text": "Reply 1", "response_tweet_id": None, "in_response_to_tweet_id": 1.0},
        {"tweet_id": 3, "author_id": "cust2", "inbound": True, "created_at": "Tue Oct 31 21:00:00 +0000 2017", "text": "Issue 2", "response_tweet_id": "4", "in_response_to_tweet_id": None},
        {"tweet_id": 4, "author_id": "AppleSupport", "inbound": False, "created_at": "Tue Oct 31 21:05:00 +0000 2017", "text": "Reply 2", "response_tweet_id": None, "in_response_to_tweet_id": 3.0},
    ]
    builder = ConversationBuilder(make_df(rows), selected_brand="AppleSupport")
    convs, stats = builder.build_conversations()

    assert len(convs) == 2
    assert stats.total_valid_conversations == 2
    ids = {c.root_tweet_id for c in convs}
    assert ids == {1, 3}


# ---------------------------------------------------------------------------
# Test 5: Branching replies
# ---------------------------------------------------------------------------
def test_branching_conversation():
    # Customer 1 receives two replies from AppleSupport (e.g. 1/2 and 2/2)
    rows = [
        {"tweet_id": 1, "author_id": "cust1", "inbound": True, "created_at": "Tue Oct 31 20:00:00 +0000 2017", "text": "Question", "response_tweet_id": "2,3", "in_response_to_tweet_id": None},
        {"tweet_id": 2, "author_id": "AppleSupport", "inbound": False, "created_at": "Tue Oct 31 20:05:00 +0000 2017", "text": "Part 1", "response_tweet_id": None, "in_response_to_tweet_id": 1.0},
        {"tweet_id": 3, "author_id": "AppleSupport", "inbound": False, "created_at": "Tue Oct 31 20:06:00 +0000 2017", "text": "Part 2", "response_tweet_id": None, "in_response_to_tweet_id": 1.0},
    ]
    builder = ConversationBuilder(make_df(rows), selected_brand="AppleSupport")
    convs, stats = builder.build_conversations()

    assert len(convs) == 1
    c = convs[0]
    assert c.has_branching is True
    assert c.thread_type == "branched"
    assert c.message_count == 3
    assert len(c.edges) == 2


# ---------------------------------------------------------------------------
# Test 6: Missing parent (broken relationship)
# ---------------------------------------------------------------------------
def test_missing_parent_tweet():
    # AppleSupport replies to tweet 999 which is not in dataset
    rows = [
        {"tweet_id": 2, "author_id": "AppleSupport", "inbound": False, "created_at": "Tue Oct 31 20:05:00 +0000 2017", "text": "Reply", "response_tweet_id": "3", "in_response_to_tweet_id": 999.0},
        {"tweet_id": 3, "author_id": "cust1", "inbound": True, "created_at": "Tue Oct 31 20:10:00 +0000 2017", "text": "Customer follow up", "response_tweet_id": None, "in_response_to_tweet_id": 2.0},
    ]
    builder = ConversationBuilder(make_df(rows), selected_brand="AppleSupport")
    convs, stats = builder.build_conversations()

    assert len(convs) == 1
    c = convs[0]
    assert c.has_anomalies is True
    anomaly_types = [a.anomaly_type for a in c.anomalies]
    assert "missing_parent_tweet" in anomaly_types
    assert c.quality_status == "low"


# ---------------------------------------------------------------------------
# Test 7: Orphan message in component
# ---------------------------------------------------------------------------
def test_orphan_message():
    rows = [
        {"tweet_id": 1, "author_id": "cust1", "inbound": True, "created_at": "Tue Oct 31 20:00:00 +0000 2017", "text": "Msg 1", "response_tweet_id": "2", "in_response_to_tweet_id": None},
        {"tweet_id": 2, "author_id": "AppleSupport", "inbound": False, "created_at": "Tue Oct 31 20:05:00 +0000 2017", "text": "Msg 2", "response_tweet_id": None, "in_response_to_tweet_id": 1.0},
    ]
    builder = ConversationBuilder(make_df(rows), selected_brand="AppleSupport")
    convs, stats = builder.build_conversations()
    assert len(convs) == 1
    assert stats.orphan_messages == 0


# ---------------------------------------------------------------------------
# Test 8: Self-reference handling
# ---------------------------------------------------------------------------
def test_self_reference_anomaly():
    rows = [
        {"tweet_id": 1, "author_id": "AppleSupport", "inbound": False, "created_at": "Tue Oct 31 20:00:00 +0000 2017", "text": "Self replying", "response_tweet_id": None, "in_response_to_tweet_id": 1.0},
        {"tweet_id": 2, "author_id": "cust1", "inbound": True, "created_at": "Tue Oct 31 20:05:00 +0000 2017", "text": "Replying to self-reply", "response_tweet_id": None, "in_response_to_tweet_id": 1.0},
    ]
    builder = ConversationBuilder(make_df(rows), selected_brand="AppleSupport")
    convs, stats = builder.build_conversations()

    assert len(convs) == 1
    c = convs[0]
    assert c.has_anomalies is True
    assert any(a.anomaly_type == "self_reference" for a in c.anomalies)


# ---------------------------------------------------------------------------
# Test 9: Cycle detection
# ---------------------------------------------------------------------------
def test_cycle_detection():
    # 1 -> 2 -> 1
    rows = [
        {"tweet_id": 1, "author_id": "cust1", "inbound": True, "created_at": "Tue Oct 31 20:00:00 +0000 2017", "text": "A", "response_tweet_id": "2", "in_response_to_tweet_id": 2.0},
        {"tweet_id": 2, "author_id": "AppleSupport", "inbound": False, "created_at": "Tue Oct 31 20:05:00 +0000 2017", "text": "B", "response_tweet_id": "1", "in_response_to_tweet_id": 1.0},
    ]
    builder = ConversationBuilder(make_df(rows), selected_brand="AppleSupport")
    convs, stats = builder.build_conversations()

    assert len(convs) == 1
    c = convs[0]
    assert c.has_anomalies is True
    assert stats.cycles_detected >= 1
    assert any(a.anomaly_type == "cyclic_relationship" for a in c.anomalies)


# ---------------------------------------------------------------------------
# Test 10: Duplicate edge
# ---------------------------------------------------------------------------
def test_duplicate_edge():
    rows = [
        {"tweet_id": 1, "author_id": "cust1", "inbound": True, "created_at": "Tue Oct 31 20:00:00 +0000 2017", "text": "A", "response_tweet_id": "2", "in_response_to_tweet_id": None},
        {"tweet_id": 2, "author_id": "AppleSupport", "inbound": False, "created_at": "Tue Oct 31 20:05:00 +0000 2017", "text": "B", "response_tweet_id": None, "in_response_to_tweet_id": 1.0},
    ]
    builder = ConversationBuilder(make_df(rows), selected_brand="AppleSupport")
    convs, stats = builder.build_conversations()
    assert len(convs[0].edges) == 1


# ---------------------------------------------------------------------------
# Test 11: Timestamp ordering anomaly
# ---------------------------------------------------------------------------
def test_timestamp_ordering_anomaly():
    # Child timestamp is EARLIER than parent timestamp
    rows = [
        {"tweet_id": 1, "author_id": "cust1", "inbound": True, "created_at": "Tue Oct 31 20:10:00 +0000 2017", "text": "Parent later", "response_tweet_id": "2", "in_response_to_tweet_id": None},
        {"tweet_id": 2, "author_id": "AppleSupport", "inbound": False, "created_at": "Tue Oct 31 20:00:00 +0000 2017", "text": "Child earlier", "response_tweet_id": None, "in_response_to_tweet_id": 1.0},
    ]
    builder = ConversationBuilder(make_df(rows), selected_brand="AppleSupport")
    convs, stats = builder.build_conversations()

    assert len(convs) == 1
    c = convs[0]
    assert c.has_anomalies is True
    assert any(a.anomaly_type == "timestamp_anomaly" for a in c.anomalies)
    assert stats.timestamp_anomalies >= 1
    # Graph parentage is preserved: parent still has depth 0, child depth 1
    assert c.messages[0].tweet_id == 1
    assert c.messages[1].tweet_id == 2


# ---------------------------------------------------------------------------
# Test 12: Deterministic conversation ID
# ---------------------------------------------------------------------------
def test_deterministic_conversation_id():
    id1 = build_deterministic_conversation_id("AppleSupport", 12345)
    id2 = build_deterministic_conversation_id("AppleSupport", 12345)
    assert id1 == id2 == "conv_AppleSupport_12345"


# ---------------------------------------------------------------------------
# Test 13: Thread classification
# ---------------------------------------------------------------------------
def test_thread_classification():
    m_cb = [
        ConversationMessage(tweet_id=1, author_id="c", role="customer", inbound=True, created_at="", text=""),
        ConversationMessage(tweet_id=2, author_id="b", role="brand", inbound=False, created_at="", text=""),
    ]
    assert classify_thread_type(m_cb, has_branching=False, has_breaking_anomalies=False) == "customer_brand"

    m_cbc = m_cb + [ConversationMessage(tweet_id=3, author_id="c", role="customer", inbound=True, created_at="", text="")]
    assert classify_thread_type(m_cbc, has_branching=False, has_breaking_anomalies=False) == "customer_brand_customer"

    m_cbcb = m_cbc + [ConversationMessage(tweet_id=4, author_id="b", role="brand", inbound=False, created_at="", text="")]
    assert classify_thread_type(m_cbcb, has_branching=False, has_breaking_anomalies=False) == "customer_brand_customer_brand"

    assert classify_thread_type(m_cbcb, has_branching=True, has_breaking_anomalies=False) == "branched"
    assert classify_thread_type(m_cbcb, has_branching=False, has_breaking_anomalies=True) == "incomplete"


# ---------------------------------------------------------------------------
# Test 14: Quality status classification
# ---------------------------------------------------------------------------
def test_quality_status_classification():
    assert assign_quality_status("conv_1", 2, {"customer", "brand"}, has_breaking_anomalies=False, has_minor_anomalies=False) == "high"
    assert assign_quality_status("conv_1", 2, {"customer", "brand"}, has_breaking_anomalies=False, has_minor_anomalies=True) == "medium"
    assert assign_quality_status("conv_1", 2, {"customer", "brand"}, has_breaking_anomalies=True, has_minor_anomalies=False) == "low"
    assert assign_quality_status("conv_1", 1, {"customer"}, has_breaking_anomalies=False, has_minor_anomalies=False) == "low"


# ---------------------------------------------------------------------------
# Test 15: Brand filtering
# ---------------------------------------------------------------------------
def test_brand_filtering():
    # Only AmazonHelp conversation, no AppleSupport
    rows = [
        {"tweet_id": 1, "author_id": "cust1", "inbound": True, "created_at": "Tue Oct 31 20:00:00 +0000 2017", "text": "A", "response_tweet_id": "2", "in_response_to_tweet_id": None},
        {"tweet_id": 2, "author_id": "AmazonHelp", "inbound": False, "created_at": "Tue Oct 31 20:05:00 +0000 2017", "text": "B", "response_tweet_id": None, "in_response_to_tweet_id": 1.0},
    ]
    builder = ConversationBuilder(make_df(rows), selected_brand="AppleSupport")
    convs, stats = builder.build_conversations()
    assert len(convs) == 0
    assert stats.total_valid_conversations == 0


# ---------------------------------------------------------------------------
# Test 16: Empty dataset handling
# ---------------------------------------------------------------------------
def test_empty_dataset_handling():
    empty_df = pd.DataFrame(columns=[
        "tweet_id", "author_id", "inbound", "created_at", "text", "response_tweet_id", "in_response_to_tweet_id"
    ])
    builder = ConversationBuilder(empty_df, selected_brand="AppleSupport")
    convs, stats = builder.build_conversations()
    assert len(convs) == 0
    assert stats.total_valid_conversations == 0
    assert stats.total_messages == 0
