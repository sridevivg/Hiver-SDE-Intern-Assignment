"""
Tests for SupportResolutionEngine & FastAPI Resolution Endpoints (Phase 8)
"""
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.resolution.support_resolution_engine import (
    SupportResolutionEngine,
    SupportResolutionResult,
)
from backend.app.schemas.intent_routing import RoutingDecisionType
from backend.app.understanding.problem_extractor import CustomerProblemProfile


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def engine() -> SupportResolutionEngine:
    return SupportResolutionEngine()


def test_process_message_battery_issue_resolution(engine: SupportResolutionEngine) -> None:
    message = "My battery is draining completely within 2 hours of use on my iPhone 7."
    result: SupportResolutionResult = engine.process_message(customer_message=message)

    assert result.primary_intent == "battery_power_issue"
    assert result.problem_summary != ""
    assert isinstance(result.evidence_cases, list)

    if result.routing_decision == RoutingDecisionType.AUTO_HANDLE:
        assert result.grounded_response is not None
        assert "battery" in result.grounded_response.lower()
    else:
        assert result.escalation_package is not None
        assert len(result.escalation_package.top_candidates) > 0


def test_process_message_vague_escalation(engine: SupportResolutionEngine) -> None:
    message = "phone broken please help"
    result: SupportResolutionResult = engine.process_message(customer_message=message)

    assert result.routing_decision == RoutingDecisionType.ESCALATE_TO_HUMAN
    assert result.escalation_package is not None
    assert result.grounded_response is None
    assert result.ambiguity_analysis.is_ambiguous is True


def test_api_resolve_endpoint(client: TestClient) -> None:
    payload = {
        "customer_message": "How do I reset my Apple ID password?",
        "top_k_evidence": 2,
    }
    response = client.post("/api/v1/resolution/resolve", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "primary_intent" in data
    assert "routing_decision" in data
    assert "ambiguity_analysis" in data
    assert "evidence_cases" in data


def test_api_understand_endpoint(client: TestClient) -> None:
    payload = {
        "customer_message": "My iPhone 8 microphone does not record sound after iOS 11 update.",
    }
    response = client.post("/api/v1/resolution/understand", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["device"] == "iPhone 8"
    assert data["update_related"] is True
    assert "audio" in data["primary_symptom"] or "microphone" in data["primary_symptom"] or "sound" in data["primary_symptom"]


def test_api_retrieve_evidence_endpoint(client: TestClient) -> None:
    payload = {
        "query_text": "battery dying fast",
        "query_intent": "battery_power_issue",
        "top_k": 2,
    }
    response = client.post("/api/v1/resolution/retrieve-evidence", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_api_validate_evidence_endpoint(client: TestClient) -> None:
    payload = {
        "customer_message": "My battery is dying in 1 hour on iPhone 7.",
        "primary_intent": "battery_power_issue",
        "top_k_evidence": 2,
    }
    response = client.post("/api/v1/resolution/validate-evidence", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "evidence_verdict" in data
    assert "auto_resolution_allowed" in data
    assert "symptom_agreement" in data


def test_api_verify_response_endpoint(client: TestClient) -> None:
    payload = {
        "customer_message": "My battery is dying fast on iPhone 7.",
        "primary_intent": "battery_power_issue",
        "candidate_response": "We know how important battery is on iPhone 7. Check Settings > Battery > Battery Health. DM us: https://apple.co/DM",
    }
    response = client.post("/api/v1/resolution/verify-response", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "verification_status" in data
    assert "grounded" in data
    assert "support_score" in data


def test_api_audit_stats_endpoint(client: TestClient) -> None:
    response = client.get("/api/v1/resolution/audit/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_logged_decisions" in data
    assert "auto_handle_count" in data
