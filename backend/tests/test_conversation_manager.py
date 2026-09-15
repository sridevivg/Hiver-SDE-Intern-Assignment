"""
SupportGraph AI — Conversation Manager & API Route Tests (Phase 11).

Tests conversation persistence, file-backed state saving, atomic replacement,
append-only audit logs, and FastAPI route responses.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

try:
    from app.conversation.conversation_manager import ConversationManager
    from app.conversation.conversation_state import (
        ConversationState,
        ConversationStatus,
        ResolutionStage,
    )
    from app.main import app
except ModuleNotFoundError:
    from backend.app.conversation.conversation_manager import (  # type: ignore[no-redef]
        ConversationManager,
    )
    from backend.app.conversation.conversation_state import (  # type: ignore[no-redef]
        ConversationState,
        ConversationStatus,
        ResolutionStage,
    )
    from backend.app.main import app  # type: ignore[no-redef]


@pytest.fixture
def client():
    return TestClient(app)


class TestConversationPersistence:
    """Test disk persistence, reload, and audit trail."""

    def test_state_persists_to_disk_and_reloads(self, tmp_path):
        mgr = ConversationManager(base_dir=tmp_path)
        state, turn = mgr.create_conversation(
            customer_id="cust_123",
            initial_message="My iPhone battery drains within 2 hours.",
        )
        conv_id = state.conversation_id

        # Verify disk file exists
        state_file = tmp_path / f"{conv_id}.json"
        assert state_file.exists()

        # Load fresh manager pointing to same directory
        mgr_fresh = ConversationManager(base_dir=tmp_path)
        loaded = mgr_fresh.get_conversation(conv_id)
        assert loaded is not None
        assert loaded.conversation_id == conv_id
        assert loaded.customer_id == "cust_123"
        assert len(loaded.turns) == 2  # 1 customer turn, 1 agent turn

    def test_audit_trail_recorded(self, tmp_path):
        mgr = ConversationManager(base_dir=tmp_path)
        state, turn = mgr.create_conversation(
            initial_message="My MacBook won't turn on.",
        )
        mgr.process_message(state.conversation_id, "I plugged it in and nothing happens.")

        audit_entries = mgr.get_audit_trail(state.conversation_id)
        assert len(audit_entries) >= 2  # CONVERSATION_CREATED + TURN_PROCESSED
        assert audit_entries[0]["event"] == "CONVERSATION_CREATED"
        assert audit_entries[1]["event"] == "TURN_PROCESSED"


class TestConversationAPIEndpoints:
    """Test the 7 FastAPI endpoints."""

    def test_start_conversation_endpoint(self, client):
        resp = client.post(
            "/api/v1/conversations/start",
            json={"customer_id": "cust_test_1", "initial_message": "My iPad screen is frozen."},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert "conversation_id" in data
        assert data["status"] == "ACTIVE"
        assert data["initial_agent_response"] is not None
        assert data["turns_count"] == 2

    def test_send_message_endpoint(self, client):
        start_resp = client.post(
            "/api/v1/conversations/start",
            json={"initial_message": "My iPhone won't connect to Bluetooth."},
        )
        conv_id = start_resp.json()["conversation_id"]

        msg_resp = client.post(
            f"/api/v1/conversations/{conv_id}/message",
            json={"message": "I turned Bluetooth off and on, but it still fails to pair."},
        )
        assert msg_resp.status_code == 200
        data = msg_resp.json()
        assert data["conversation_id"] == conv_id
        assert "agent_response" in data
        assert data["turn_index"] >= 2

    def test_get_state_endpoint(self, client):
        start_resp = client.post(
            "/api/v1/conversations/start",
            json={"initial_message": "Need help with my Apple ID password."},
        )
        conv_id = start_resp.json()["conversation_id"]

        state_resp = client.get(f"/api/v1/conversations/{conv_id}/state")
        assert state_resp.status_code == 200
        state = state_resp.json()
        assert state["conversation_id"] == conv_id
        assert "problem_family" in state

    def test_get_history_endpoint(self, client):
        start_resp = client.post(
            "/api/v1/conversations/start",
            json={"initial_message": "My iPhone camera is completely black."},
        )
        conv_id = start_resp.json()["conversation_id"]

        hist_resp = client.get(f"/api/v1/conversations/{conv_id}/history")
        assert hist_resp.status_code == 200
        history = hist_resp.json()
        assert len(history) == 2
        assert history[0]["speaker"] == "customer"
        assert history[1]["speaker"] == "agent"

    def test_resolution_summary_and_resolve_endpoints(self, client):
        start_resp = client.post(
            "/api/v1/conversations/start",
            json={"initial_message": "My iPhone volume buttons are stuck."},
        )
        conv_id = start_resp.json()["conversation_id"]

        # Resolve
        res_resp = client.post(
            f"/api/v1/conversations/{conv_id}/resolve",
            json={"summary": "Customer cleaned debris and buttons now click normally."},
        )
        assert res_resp.status_code == 200
        assert res_resp.json()["status"] == "RESOLVED"

        # Check summary endpoint
        sum_resp = client.get(f"/api/v1/conversations/{conv_id}/resolution-summary")
        assert sum_resp.status_code == 200
        assert sum_resp.json()["type"] == "RESOLVED"

    def test_audit_trail_endpoint(self, client):
        start_resp = client.post(
            "/api/v1/conversations/start",
            json={"initial_message": "My iPhone Wi-Fi disconnects often."},
        )
        conv_id = start_resp.json()["conversation_id"]

        audit_resp = client.get(f"/api/v1/conversations/{conv_id}/audit")
        assert audit_resp.status_code == 200
        audit = audit_resp.json()
        assert len(audit) >= 1
