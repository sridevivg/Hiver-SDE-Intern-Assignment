"""
SupportGraph AI — Test Suite for Intent Routing & HITL Escalation (Phase 6)

Covers:
- Normalized entropy and uncertainty calculations
- Top-K intent classifier JSON parsing and probability normalization
- Heuristic fallback classification
- Deterministic routing engine (AUTO_HANDLE vs ESCALATE_TO_HUMAN)
- Threshold boundary conditions and close-margin cases
- Runtime escalation manager logging and append-only audit trail
- Routing system evaluator metrics computation
- FastAPI REST endpoints via TestClient
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import settings
from backend.app.core.llm_factory import LLMCompletionResponse
from backend.app.evaluation.routing_evaluator import RoutingSystemEvaluator
from backend.app.intent.classifier import (
    TopKIntentClassifier,
    compute_normalized_entropy,
)
from backend.app.intent.escalation import RuntimeEscalationManager
from backend.app.intent.router import IntentRouter
from backend.app.main import app
from backend.app.schemas.intent_routing import (
    CustomerMessageRequest,
    HumanReviewActionRequest,
    HumanReviewActionType,
    IntentAnalysis,
    IntentPrediction,
    RoutingDecision,
    RoutingDecisionType,
)


# ---------------------------------------------------------------------------
# Test Uncertainty Calculations
# ---------------------------------------------------------------------------
class TestUncertaintyCalculations:
    def test_entropy_pure_certainty(self):
        """A single dominant prediction has 0 entropy."""
        assert compute_normalized_entropy([1.0, 0.0, 0.0]) == 0.0
        assert compute_normalized_entropy([1.0]) == 0.0

    def test_entropy_uniform_distribution(self):
        """Uniform distribution across candidates has maximum normalized entropy (1.0)."""
        probs = [1 / 3, 1 / 3, 1 / 3]
        entropy = compute_normalized_entropy(probs)
        assert pytest.approx(entropy, abs=0.01) == 1.0

    def test_entropy_intermediate_distribution(self):
        """Intermediate probabilities have entropy strictly between 0 and 1."""
        entropy = compute_normalized_entropy([0.70, 0.20, 0.10])
        assert 0.0 < entropy < 1.0


# ---------------------------------------------------------------------------
# Test Top-K Intent Classifier
# ---------------------------------------------------------------------------
class TestTopKIntentClassifier:
    def test_parse_predictions_json_valid(self):
        classifier = TopKIntentClassifier()
        raw_json = json.dumps({
            "predictions": [
                {"intent": "software_update_problem", "confidence": 0.60, "reasoning": "Update bug"},
                {"intent": "hardware_audio_connection_issue", "confidence": 0.40, "reasoning": "Audio crackle"},
            ]
        })
        preds = classifier.parse_predictions_json(raw_json, requested_k=2)
        assert len(preds) == 2
        assert preds[0].intent == "software_update_problem"
        assert preds[1].intent == "hardware_audio_connection_issue"
        assert pytest.approx(sum(p.confidence for p in preds), abs=0.001) == 1.0

    def test_parse_predictions_with_markdown_wrapper(self):
        classifier = TopKIntentClassifier()
        raw = "```json\n{\"predictions\": [{\"intent\": \"battery_power_issue\", \"confidence\": 0.95, \"reasoning\": \"Battery drain\"}]}\n```"
        preds = classifier.parse_predictions_json(raw, requested_k=1)
        assert len(preds) == 1
        assert preds[0].intent == "battery_power_issue"
        assert preds[0].confidence == 1.0  # normalized

    def test_parse_predictions_invalid_label_filtered(self):
        classifier = TopKIntentClassifier()
        raw = json.dumps({
            "predictions": [
                {"intent": "made_up_nonexistent_intent", "confidence": 0.80},
                {"intent": "display_touch_issue", "confidence": 0.20},
            ]
        })
        preds = classifier.parse_predictions_json(raw, requested_k=2)
        assert len(preds) == 1
        assert preds[0].intent == "display_touch_issue"

    def test_heuristic_fallback_classification(self):
        classifier = TopKIntentClassifier()
        preds = classifier.classify_heuristic_fallback(
            "My iPhone battery dies within 30 minutes and won't charge",
            requested_k=3,
        )
        assert len(preds) == 3
        assert preds[0].intent == "battery_power_issue"
        assert preds[0].confidence > preds[1].confidence
        assert pytest.approx(sum(p.confidence for p in preds), abs=0.01) == 1.0

    def test_classify_with_mock_llm(self):
        mock_llm = MagicMock()
        mock_llm.provider = "groq"
        mock_llm.model = "mock-model"
        mock_llm.chat_completion.return_value = LLMCompletionResponse(
            content=json.dumps({
                "predictions": [
                    {"intent": "keyboard_typing_issue", "confidence": 0.88, "reasoning": "Letter I autocorrect bug"},
                    {"intent": "general_device_support", "confidence": 0.12, "reasoning": "Fallback"},
                ]
            }),
            model="mock-model",
            provider="groq",
        )
        classifier = TopKIntentClassifier(llm_client=mock_llm)
        analysis = classifier.classify("Letter I is turning into exclamation mark")

        assert analysis.top_1_intent == "keyboard_typing_issue"
        assert analysis.top_2_intent == "general_device_support"
        assert analysis.top_confidence == 0.88
        assert analysis.confidence_margin == round(0.88 - 0.12, 4)


# ---------------------------------------------------------------------------
# Test Intent Router
# ---------------------------------------------------------------------------
class TestIntentRouter:
    @pytest.fixture
    def router(self):
        return IntentRouter(
            auto_handle_confidence_threshold=0.85,
            min_confidence_margin=0.15,
            max_uncertainty_entropy=0.65,
        )

    def test_high_confidence_decisive_margin_auto_handles(self, router):
        analysis = IntentAnalysis(
            top_predictions=[
                IntentPrediction(intent="battery_power_issue", confidence=0.92, reasoning="Clear battery drain"),
                IntentPrediction(intent="general_device_support", confidence=0.08, reasoning="Low likelihood"),
            ],
            top_confidence=0.92,
            confidence_margin=0.84,
            normalized_entropy=0.40,
            model_name="test_model",
        )
        decision = router.route(analysis)
        assert decision.decision == RoutingDecisionType.AUTO_HANDLE
        assert "High confidence" in decision.reason
        assert "battery_power_issue" in decision.reason

    def test_low_confidence_escalates(self, router):
        analysis = IntentAnalysis(
            top_predictions=[
                IntentPrediction(intent="software_update_problem", confidence=0.70, reasoning="Moderate update signal"),
                IntentPrediction(intent="general_device_support", confidence=0.30, reasoning="Generic support"),
            ],
            top_confidence=0.70,
            confidence_margin=0.40,
            normalized_entropy=0.60,
            model_name="test_model",
        )
        decision = router.route(analysis)
        assert decision.decision == RoutingDecisionType.ESCALATE_TO_HUMAN
        assert "below the minimum auto-handle threshold" in decision.reason

    def test_close_competing_margin_escalates(self, router):
        """Top-1 confidence may be moderate/high, but small margin between Top-1 and Top-2 must trigger escalation."""
        analysis = IntentAnalysis(
            top_predictions=[
                IntentPrediction(intent="software_update_problem", confidence=0.86, reasoning="Update issue"),
                IntentPrediction(intent="hardware_audio_connection_issue", confidence=0.78, reasoning="Audio issue"),
            ],
            top_confidence=0.86,
            confidence_margin=0.08,  # Below 0.15 threshold!
            normalized_entropy=0.60,
            model_name="test_model",
        )
        decision = router.route(analysis)
        assert decision.decision == RoutingDecisionType.ESCALATE_TO_HUMAN
        assert "Ambiguous intent competition" in decision.reason
        assert "0.08" in decision.reason

    def test_unclear_intent_always_escalates(self, router):
        analysis = IntentAnalysis(
            top_predictions=[
                IntentPrediction(intent="unclear_needs_review", confidence=0.99, reasoning="Gibberish text"),
            ],
            top_confidence=0.99,
            confidence_margin=0.99,
            normalized_entropy=0.0,
            model_name="test_model",
        )
        decision = router.route(analysis)
        assert decision.decision == RoutingDecisionType.ESCALATE_TO_HUMAN
        assert "unclear_needs_review" in decision.reason


# ---------------------------------------------------------------------------
# Test Runtime Escalation Manager
# ---------------------------------------------------------------------------
class TestRuntimeEscalationManager:
    @pytest.fixture
    def temp_log(self, tmp_path):
        return tmp_path / "runtime_escalation_reviews.csv"

    def test_initialization_creates_file_with_headers(self, temp_log):
        mgr = RuntimeEscalationManager(log_path=temp_log)
        assert temp_log.exists()
        with open(temp_log, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            assert "case_id" in header
            assert "human_final_intent" in header
            assert "is_ai_accepted" in header

    def test_record_accept_top_1(self, temp_log):
        mgr = RuntimeEscalationManager(log_path=temp_log)
        req = HumanReviewActionRequest(
            case_id="case_001",
            customer_message="My battery is dead",
            action=HumanReviewActionType.ACCEPT_TOP_1,
            top_predictions=[
                IntentPrediction(intent="battery_power_issue", confidence=0.60),
                IntentPrediction(intent="general_device_support", confidence=0.40),
            ],
            confidence_margin=0.20,
        )
        record = mgr.record_human_decision(req)
        assert record.human_final_intent == "battery_power_issue"
        assert record.is_ai_accepted is True

        df = mgr.load_review_history()
        assert len(df) == 1
        assert df.iloc[0]["case_id"] == "case_001"
        assert df.iloc[0]["human_final_intent"] == "battery_power_issue"

    def test_record_select_top_2(self, temp_log):
        mgr = RuntimeEscalationManager(log_path=temp_log)
        req = HumanReviewActionRequest(
            case_id="case_002",
            customer_message="Screen flickers and sound is weird",
            action=HumanReviewActionType.SELECT_TOP_2,
            top_predictions=[
                IntentPrediction(intent="display_touch_issue", confidence=0.52),
                IntentPrediction(intent="hardware_audio_connection_issue", confidence=0.48),
            ],
        )
        record = mgr.record_human_decision(req)
        assert record.human_final_intent == "hardware_audio_connection_issue"
        assert record.is_ai_accepted is False

    def test_record_override_intent(self, temp_log):
        mgr = RuntimeEscalationManager(log_path=temp_log)
        req = HumanReviewActionRequest(
            case_id="case_003",
            customer_message="Can't log into iCloud",
            action=HumanReviewActionType.OVERRIDE_INTENT,
            selected_intent="account_access_issue",
            top_predictions=[
                IntentPrediction(intent="general_device_support", confidence=0.70),
            ],
        )
        record = mgr.record_human_decision(req)
        assert record.human_final_intent == "account_access_issue"
        assert record.is_ai_accepted is False

    def test_summary_stats(self, temp_log):
        mgr = RuntimeEscalationManager(log_path=temp_log)
        # Add 1 accept, 1 override
        mgr.record_human_decision(
            HumanReviewActionRequest(
                case_id="c1",
                customer_message="msg1",
                action=HumanReviewActionType.ACCEPT_TOP_1,
                top_predictions=[IntentPrediction(intent="battery_power_issue", confidence=0.9)],
            )
        )
        mgr.record_human_decision(
            HumanReviewActionRequest(
                case_id="c2",
                customer_message="msg2",
                action=HumanReviewActionType.OVERRIDE_INTENT,
                selected_intent="display_touch_issue",
                top_predictions=[IntentPrediction(intent="battery_power_issue", confidence=0.5)],
            )
        )
        stats = mgr.get_summary_stats()
        assert stats["total_reviews"] == 2
        assert stats["ai_accepted_count"] == 1
        assert stats["ai_accepted_rate"] == 0.5
        assert stats["overridden_count"] == 1


# ---------------------------------------------------------------------------
# Test Routing System Evaluator
# ---------------------------------------------------------------------------
class TestRoutingSystemEvaluator:
    def test_evaluator_on_mock_dataframe(self):
        mock_llm = MagicMock()
        mock_llm.provider = "mock"
        mock_llm.model = "mock-model"
        classifier = TopKIntentClassifier(llm_client=mock_llm)
        classifier.classify = lambda customer_message, normalized_message="", conversation_context="", top_k=3: (
            IntentAnalysis(
                top_predictions=classifier.classify_heuristic_fallback(customer_message, requested_k=top_k or 3),
                top_confidence=0.90,
                confidence_margin=0.80,
                normalized_entropy=0.25,
                model_name="mock_classifier",
            )
        )
        evaluator = RoutingSystemEvaluator(classifier=classifier)
        mock_data = pd.DataFrame([
            {
                "golden_id": "g1",
                "customer_message": "My battery is completely dead and drains quickly",
                "annotation_label": "battery_power_issue",
                "annotation_status": "reviewed",
            },
            {
                "golden_id": "g2",
                "customer_message": "My screen is shattered and touch not working",
                "annotation_label": "display_touch_issue",
                "annotation_status": "reviewed",
            },
            {
                "golden_id": "g3",
                "customer_message": "I was charged twice on my credit card for music",
                "annotation_label": "billing_purchase_issue",
                "annotation_status": "reviewed",
            },
        ])
        report = evaluator.evaluate_dataframe(mock_data)
        assert report.total_records == 3
        assert report.top_1_accuracy > 0.0
        assert report.top_2_accuracy >= report.top_1_accuracy
        assert report.auto_handle_rate >= 0.0
        assert "Primary Performance Metrics" in report.summary_markdown()


# ---------------------------------------------------------------------------
# Test FastAPI Endpoints
# ---------------------------------------------------------------------------
class TestAPIEndpoints:
    @pytest.fixture
    def client(self):
        from backend.app.api.routes.intent import get_classifier
        mock_llm = MagicMock()
        mock_llm.provider = "mock"
        mock_llm.model = "mock-model"
        mock_classifier = TopKIntentClassifier(llm_client=mock_llm)
        mock_classifier.classify = lambda customer_message, normalized_message="", conversation_context="", top_k=3: (
            mock_classifier.classify_heuristic_fallback(customer_message, requested_k=top_k or 3)
            and IntentAnalysis(
                top_predictions=mock_classifier.classify_heuristic_fallback(customer_message, requested_k=top_k or 3),
                top_confidence=0.90,
                confidence_margin=0.80,
                normalized_entropy=0.25,
                model_name="mock_classifier",
            )
        )
        app.dependency_overrides[get_classifier] = lambda: mock_classifier
        test_client = TestClient(app)
        yield test_client
        app.dependency_overrides.clear()

    def test_health_check(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    def test_get_routing_config(self, client):
        resp = client.get("/api/v1/intent/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "auto_handle_confidence_threshold" in data
        assert "min_confidence_margin" in data

    def test_classify_and_route_endpoint(self, client):
        payload = {"customer_message": "My battery is draining rapidly since updating iOS."}
        resp = client.post("/api/v1/intent/classify-and-route", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "intent_analysis" in data
        assert "top_predictions" in data["intent_analysis"]
        assert len(data["intent_analysis"]["top_predictions"]) >= 1
        assert "routing" in data
        assert data["routing"]["decision"] in ["AUTO_HANDLE", "ESCALATE_TO_HUMAN"]
        assert "reason" in data["routing"]

    def test_classify_endpoint_only(self, client):
        payload = {"customer_message": "I forgot my Apple ID password and account is locked."}
        resp = client.post("/api/v1/intent/classify", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "top_predictions" in data
        assert "top_confidence" in data
        assert "confidence_margin" in data

    def test_route_endpoint_only(self, client):
        analysis_payload = {
            "top_predictions": [
                {"intent": "battery_power_issue", "confidence": 0.95, "reasoning": "High battery confidence"},
                {"intent": "general_device_support", "confidence": 0.05, "reasoning": "Fallback"},
            ],
            "top_confidence": 0.95,
            "confidence_margin": 0.90,
            "normalized_entropy": 0.20,
            "model_name": "test_model",
        }
        resp = client.post("/api/v1/intent/route", json=analysis_payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["decision"] == "AUTO_HANDLE"

    def test_submit_human_review_endpoint(self, client, tmp_path):
        custom_log = tmp_path / "test_escalation_reviews.csv"
        with patch.object(settings, "runtime_escalations_log_file", str(custom_log.name)):
            payload = {
                "case_id": "api_case_001",
                "customer_message": "Need help with my account",
                "action": "ACCEPT_TOP_1",
                "selected_intent": "account_access_issue",
                "top_predictions": [
                    {"intent": "account_access_issue", "confidence": 0.60},
                    {"intent": "general_device_support", "confidence": 0.40},
                ],
                "confidence_margin": 0.20,
            }
            resp = client.post("/api/v1/intent/review", json=payload)
            assert resp.status_code == 200
            data = resp.json()
            assert data["case_id"] == "api_case_001"
            assert data["human_final_intent"] == "account_access_issue"
            assert data["is_ai_accepted"] is True
