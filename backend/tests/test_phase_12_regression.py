"""
SupportGraph AI — Phase 12.1 Regression & Generalization Test Suite.

Verifies:
1. CLEAR_02 Fix: AirPods / Bluetooth pairing inquiry resolves safely with grounded evidence.
2. ADVERSARIAL_02 Fix: Thermal / physical hazard during multi-turn triggers immediate urgent human escalation.
3. ADVERSARIAL_03 Safe Escalation: Conflicting multi-symptom inquiries safely escalate without ungrounded auto-handle.
4. Generalization: Audio connection variations classify and resolve cleanly.
5. Generalization: Thermal/hazard variations halt automated actions and escalate immediately.
6. Dataset & Integrity: Golden set remains unchanged and generalization scenarios are well-formed.
"""

import json
from pathlib import Path
import pytest

from backend.app.conversation.conversation_manager import ConversationManager
from backend.app.conversation.conversation_state import (
    ConversationStatus,
    MessageRoleType,
    ResolutionStage,
)
from backend.app.conversation.turn_classifier import TurnClassifier
from backend.app.evaluation.phase_12_evaluator import (
    GOLDEN_CSV_PATH,
    GOLDEN_SHA256,
    compute_sha256,
)
from backend.app.intent.clarity_signals import EvidenceAgreementLevel
from backend.app.resolution.support_resolution_engine import SupportResolutionEngine
from backend.app.schemas.decision_explanation import EndToEndOutcome
from backend.app.schemas.intent_routing import RoutingDecisionType


@pytest.fixture
def resolution_engine() -> SupportResolutionEngine:
    return SupportResolutionEngine()


@pytest.fixture
def conversation_manager(tmp_path: Path) -> ConversationManager:
    return ConversationManager(base_dir=tmp_path)


def test_clear_02_airpods_pairing_resolved(resolution_engine: SupportResolutionEngine):
    """
    Verify CLEAR_02: AirPods Pro Bluetooth pairing resolves with grounded evidence
    and does not trigger a false symptom-intent contradiction veto.
    """
    msg = "My AirPods Pro will not pair with my iPhone 14."
    res = resolution_engine.process_message(msg, case_id="test-clear-02")

    assert res.primary_intent == "hardware_audio_connection_issue"
    if res.gate_result and res.gate_result.signals:
        assert res.gate_result.signals.evidence_agreement in (
            EvidenceAgreementLevel.STRONG_AGREEMENT,
            EvidenceAgreementLevel.PARTIAL_AGREEMENT,
        )
        assert res.gate_result.signals.is_vetoed is False
    assert res.routing_decision == RoutingDecisionType.AUTO_HANDLE
    assert res.outcome in (
        EndToEndOutcome.SAFE_AUTO_HANDLED,
        EndToEndOutcome.SUCCESSFULLY_RESOLVED,
    )
    assert len(res.evidence_cases) > 0
    assert res.grounded_response is not None


def test_adversarial_02_thermal_hazard_urgent_escalation(conversation_manager: ConversationManager):
    """
    Verify ADVERSARIAL_02: Urgent thermal hazard worsening immediately halts automated
    troubleshooting and escalates urgently to human support.
    """
    # Turn 1: Normal battery drain complaint
    t1_msg = "My iPhone battery is draining quickly."
    state, t1_turn = conversation_manager.create_conversation(initial_message=t1_msg)
    assert state.status == ConversationStatus.ACTIVE

    # Turn 2: Hazardous worsening event
    t2_msg = "Now the device is smoking, extremely hot to touch, and the back glass is cracking!"
    state, t2_turn = conversation_manager.process_message(state.conversation_id, t2_msg)

    assert state.status == ConversationStatus.ESCALATED
    assert state.escalation_package is not None
    assert state.escalation_package.recommended_specialist_tier == "TIER_2_TECHNICAL_URGENT"
    explanation = state.get_decision_explanation()
    assert explanation.outcome == EndToEndOutcome.ESCALATED_HUMAN_REQUIRED


def test_adversarial_03_conflicting_charge_and_lockout_safe_escalation(resolution_engine: SupportResolutionEngine):
    """
    Verify ADVERSARIAL_03: Multi-symptom conflict (charge dispute + password lockout)
    is safely escalated rather than issuing an ungrounded or contradictory auto-handle.
    """
    msg = "I need a refund because I was charged twice for my subscription but cannot enter my billing password."
    res = resolution_engine.process_message(msg, case_id="test-adv-03")

    assert res.routing_decision == RoutingDecisionType.ESCALATE_TO_HUMAN
    assert res.outcome in (
        EndToEndOutcome.ESCALATED_VERIFICATION_FAILURE,
        EndToEndOutcome.ESCALATED_HUMAN_REQUIRED,
        EndToEndOutcome.ESCALATED_CONFLICT,
        EndToEndOutcome.ESCALATED_EVIDENCE_LIMITED,
    )


def test_audio_connection_variations_generalization(resolution_engine: SupportResolutionEngine):
    """
    Verify that general audio connection and bluetooth pairing variations are cleanly handled.
    """
    test_cases = [
        "My Bluetooth headphones won't connect to my iPhone.",
        "My AirPods keep disconnecting during phone calls.",
        "Unable to pair my wireless earbuds with my iPad.",
    ]
    for text in test_cases:
        res = resolution_engine.process_message(text, case_id="test-audio-gen")
        assert res.primary_intent == "hardware_audio_connection_issue"
        if res.gate_result and res.gate_result.signals:
            assert res.gate_result.signals.evidence_agreement != EvidenceAgreementLevel.NO_AGREEMENT


def test_thermal_hazard_turn_classification_generalization():
    """
    Verify that various thermal and physical hazard statements are classified as PROBLEM_WORSENING.
    """
    hazard_phrases = [
        "The battery is swelling and the screen is popping off.",
        "Device is smoking and feels extremely hot to touch.",
        "It is sparking from the charging port and burning!",
        "The phone is burning hot and I smell smoke.",
    ]
    for phrase in hazard_phrases:
        role, facts, act = TurnClassifier.classify_turn(phrase, None)
        assert role == MessageRoleType.PROBLEM_WORSENING, f"Failed on phrase: {phrase}"


def test_generalization_scenarios_dataset_validity():
    """
    Verify that phase_12_1_generalization_scenarios.json contains 20 valid scenarios across 5 groups.
    """
    gen_path = Path("data/evaluation/phase_12_1_generalization_scenarios.json")
    assert gen_path.exists()

    with open(gen_path, "r", encoding="utf-8") as f:
        scenarios = json.load(f)

    assert len(scenarios) == 20
    groups = {s["group"] for s in scenarios}
    assert groups == {
        "CLEAR_SOLVABLE",
        "AMBIGUOUS_CLARIFY",
        "EVIDENCE_LIMITED",
        "MULTI_TURN",
        "ADVERSARIAL_CONFLICT",
    }
    for s in scenarios:
        assert "scenario_id" in s
        assert "turns" in s and len(s["turns"]) > 0
        assert "expected_decision" in s
        assert "expected_outcome" in s


def test_golden_dataset_immutability():
    """
    Verify that the protected golden dataset remains byte-for-byte unchanged.
    """
    sha = compute_sha256(GOLDEN_CSV_PATH)
    assert sha == GOLDEN_SHA256
