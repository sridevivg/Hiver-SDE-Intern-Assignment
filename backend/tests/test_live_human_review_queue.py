"""
Comprehensive Unit & Integration Test Suite for Live Human Review Queue
Phase 24: Real-time Live Escalations & Queue Lifecycle
"""
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.schemas.human_review import (
    CaseSource,
    ReviewCaseStatus,
    LiveReviewCase,
    ReviewActionRequest,
)
from backend.app.feedback.live_queue_manager import (
    LiveQueueManager,
    get_live_queue_manager,
)
from backend.app.api.routes import human_review as human_review_module
from backend.app.api.routes import support_resolution as support_resolution_module


@pytest.fixture
def temp_queue_mgr(tmp_path: Path) -> LiveQueueManager:
    test_file = tmp_path / "test_live_queue.jsonl"
    return LiveQueueManager(storage_path=test_file)


@pytest.fixture
def isolated_client(temp_queue_mgr: LiveQueueManager, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    app.dependency_overrides[get_live_queue_manager] = lambda: temp_queue_mgr
    monkeypatch.setattr(support_resolution_module, "get_live_queue_manager", lambda: temp_queue_mgr)
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


def test_empty_queue_on_clean_start(temp_queue_mgr: LiveQueueManager) -> None:
    """Queue should start completely empty."""
    active_cases = temp_queue_mgr.get_active_cases()
    assert active_cases == []
    assert len(active_cases) == 0


def test_isolation_rejects_test_cases(temp_queue_mgr: LiveQueueManager) -> None:
    """Test, benchmark, demo, synthetic cases must be rejected from live queue."""
    for rejected_source in [
        CaseSource.TEST,
        CaseSource.BENCHMARK,
        CaseSource.DEMO,
        CaseSource.SYNTHETIC,
        CaseSource.EVALUATION,
    ]:
        case = temp_queue_mgr.create_case(
            source=rejected_source,
            customer_query="Test benchmark inquiry",
            decision="ESCALATE_TO_HUMAN",
            escalation_reason="Test reason",
        )
        assert case is None

    assert len(temp_queue_mgr.get_active_cases()) == 0


def test_create_live_support_case(temp_queue_mgr: LiveQueueManager) -> None:
    """Live support escalation must create a LiveReviewCase."""
    case = temp_queue_mgr.create_case(
        source=CaseSource.LIVE_SUPPORT,
        customer_query="My iPhone is overheating and smoking near the charger",
        decision="ESCALATE_TO_HUMAN",
        escalation_reason="Urgent thermal hazard detected",
        priority="URGENT",
        intent="hardware_safety",
        problem_family="thermal_hazard",
    )
    assert case is not None
    assert case.source == CaseSource.LIVE_SUPPORT
    assert case.status == ReviewCaseStatus.NEW
    assert case.priority == "URGENT"
    assert case.case_id.startswith("live_rev_")
    assert len(temp_queue_mgr.get_active_cases()) == 1


def test_newest_first_ordering(temp_queue_mgr: LiveQueueManager) -> None:
    """Active queue must return cases newest-first."""
    case1 = temp_queue_mgr.create_case(
        source=CaseSource.LIVE_SUPPORT,
        customer_query="First customer question",
        decision="ESCALATE_TO_HUMAN",
        escalation_reason="First reason",
    )
    case2 = temp_queue_mgr.create_case(
        source=CaseSource.LIVE_SUPPORT,
        customer_query="Second customer question",
        decision="ESCALATE_TO_HUMAN",
        escalation_reason="Second reason",
    )
    assert case1 is not None and case2 is not None

    active = temp_queue_mgr.get_active_cases()
    assert len(active) == 2
    assert active[0].case_id == case2.case_id
    assert active[1].case_id == case1.case_id


def test_approve_case_removes_from_active_queue(temp_queue_mgr: LiveQueueManager) -> None:
    """Approving a case marks it APPROVED and removes it from the active queue."""
    case = temp_queue_mgr.create_case(
        source=CaseSource.LIVE_SUPPORT,
        customer_query="Customer needs confirmation",
        decision="ESCALATE_TO_HUMAN",
        escalation_reason="Confidence gate",
    )
    assert case is not None

    updated = temp_queue_mgr.approve_case(
        case.case_id,
        reviewer_id="lead_reviewer_1",
        notes="Approved for customer delivery",
    )
    assert updated is not None
    assert updated.status == ReviewCaseStatus.APPROVED
    assert updated.reviewer_id == "lead_reviewer_1"
    assert updated.completed_at is not None

    # Verify no longer in active queue
    active = temp_queue_mgr.get_active_cases()
    assert len(active) == 0

    # Verify historical case still retrievable
    retrieved = temp_queue_mgr.get_case(case.case_id)
    assert retrieved is not None
    assert retrieved.status == ReviewCaseStatus.APPROVED


def test_edit_case_removes_from_active_queue(temp_queue_mgr: LiveQueueManager) -> None:
    """Editing a case records edited_response, marks EDITED, and removes from active queue."""
    case = temp_queue_mgr.create_case(
        source=CaseSource.LIVE_SUPPORT,
        customer_query="AirPods sound crackling",
        decision="ESCALATE_TO_HUMAN",
        escalation_reason="Ambiguous symptom",
        ai_suggested_response="Try cleaning the mesh.",
    )
    assert case is not None

    edited = temp_queue_mgr.edit_case(
        case.case_id,
        edited_response="Please reset your AirPods by holding the setup button for 15 seconds.",
        reviewer_id="audio_specialist",
        notes="Provided exact reset instructions",
    )
    assert edited is not None
    assert edited.status == ReviewCaseStatus.EDITED
    assert edited.edited_response == "Please reset your AirPods by holding the setup button for 15 seconds."
    assert edited.reviewer_id == "audio_specialist"

    # Verify removed from active queue
    active = temp_queue_mgr.get_active_cases()
    assert len(active) == 0


def test_escalate_case_removes_from_active_queue(temp_queue_mgr: LiveQueueManager) -> None:
    """Escalating a case to Tier 2 marks it ESCALATED and removes from active queue."""
    case = temp_queue_mgr.create_case(
        source=CaseSource.LIVE_SUPPORT,
        customer_query="Kernel panic during macOS boot",
        decision="ESCALATE_TO_HUMAN",
        escalation_reason="Deep OS crash",
    )
    assert case is not None

    escalated = temp_queue_mgr.escalate_case(
        case.case_id,
        reviewer_id="triage_specialist",
        notes="Escalated to Tier 2 CoreOS engineering team",
    )
    assert escalated is not None
    assert escalated.status == ReviewCaseStatus.ESCALATED
    assert "Escalated to Tier 2 CoreOS engineering team" in str(escalated.reviewer_notes)

    active = temp_queue_mgr.get_active_cases()
    assert len(active) == 0


def test_api_get_queue_empty(isolated_client: TestClient) -> None:
    """GET /api/v1/human-review/queue returns empty list on clean start."""
    response = isolated_client.get("/api/v1/human-review/queue")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 0
    assert data["cases"] == []


def test_api_resolve_inquiry_auto_handle_no_review_case(isolated_client: TestClient) -> None:
    """Clear solvable inquiry is AUTO-HANDLED and creates NO review cases in queue."""
    payload = {
        "customer_message": "My iPhone battery is draining very quickly after updating to iOS 16.",
        "top_k_evidence": 3,
        "source": "LIVE_SUPPORT",
    }
    res = isolated_client.post("/api/v1/resolution/resolve", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["routing_decision"] == "AUTO_HANDLE"

    # Verify review queue is still empty
    queue_res = isolated_client.get("/api/v1/human-review/queue")
    assert queue_res.status_code == 200
    assert queue_res.json()["count"] == 0


def test_api_resolve_inquiry_escalation_creates_live_case(isolated_client: TestClient) -> None:
    """Hazard inquiry triggers safety gate, escalates, and creates live review case."""
    payload = {
        "customer_message": "My iPhone is burning hot, battery is swollen, and smoking!",
        "top_k_evidence": 3,
        "source": "LIVE_SUPPORT",
    }
    res = isolated_client.post("/api/v1/resolution/resolve", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["routing_decision"] == "ESCALATE_TO_HUMAN"

    # Verify review queue now contains 1 urgent case
    queue_res = isolated_client.get("/api/v1/human-review/queue")
    assert queue_res.status_code == 200
    queue_data = queue_res.json()
    assert queue_data["count"] == 1
    case = queue_data["cases"][0]
    assert case["priority"] == "URGENT"
    assert case["source"] == "LIVE_SUPPORT"
    assert "burning" in case["customer_query"] or "smoking" in case["customer_query"]


def test_api_actions_lifecycle(isolated_client: TestClient, temp_queue_mgr: LiveQueueManager) -> None:
    """Test approve, edit, and escalate endpoints through the API."""
    case = temp_queue_mgr.create_case(
        source=CaseSource.LIVE_SUPPORT,
        customer_query="Need human review for account billing issue",
        decision="ESCALATE_TO_HUMAN",
        escalation_reason="Billing ambiguity",
    )
    assert case is not None

    # Get by ID
    get_res = isolated_client.get(f"/api/v1/human-review/{case.case_id}")
    assert get_res.status_code == 200
    assert get_res.json()["case_id"] == case.case_id

    # Approve
    approve_res = isolated_client.post(
        f"/api/v1/human-review/{case.case_id}/approve",
        json={"reviewer_id": "auditor_1", "notes": "Approved"},
    )
    assert approve_res.status_code == 200
    assert approve_res.json()["status"] == "APPROVED"

    # Queue is now empty
    q_res = isolated_client.get("/api/v1/human-review/queue")
    assert q_res.json()["count"] == 0


def test_api_case_not_found(isolated_client: TestClient) -> None:
    """GET /api/v1/human-review/nonexistent returns 404."""
    res = isolated_client.get("/api/v1/human-review/nonexistent_id")
    assert res.status_code == 404
