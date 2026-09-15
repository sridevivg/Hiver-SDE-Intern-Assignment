"""
SupportGraph AI — Multi-Turn Resolution & Conversation State Adversarial Tests (Phase 11).

Tests all 10 mandatory adversarial scenarios (A through J):
- Scenario A: Customer already attempted suggested action (no repeat)
- Scenario B: Progressive troubleshooting across 3-4 turns without repetition
- Scenario C: Clarification loop prevention (max 1 crisp question, no loop)
- Scenario D: Negative confirmation handling (clean progression)
- Scenario E: Resolution confirmation (immediate resolution and summary)
- Scenario F: Problem worsening / urgent safety escalation
- Scenario G: Topic drift / multiple problems cleanly distinguished
- Scenario H: Context retention across multi-turn dialogue (confirmed facts preserved)
- Scenario I: Semantic equivalence repeat prevention ('power cycle' == 'reboot' == 'restart')
- Scenario J: Safe human escalation with full context package
"""

from __future__ import annotations

import pytest

try:
    from app.conversation.action_catalog import ActionCatalog
    from app.conversation.action_tracker import ResolutionActionTracker
    from app.conversation.conversation_manager import ConversationManager
    from app.conversation.conversation_state import (
        ActionStatus,
        ConversationState,
        ConversationStatus,
        MessageRoleType,
        ResolutionStage,
    )
    from app.conversation.resolution_progress_engine import ResolutionProgressEngine
    from app.conversation.turn_classifier import TurnClassifier
except ModuleNotFoundError:
    from backend.app.conversation.action_catalog import ActionCatalog  # type: ignore[no-redef]
    from backend.app.conversation.action_tracker import (  # type: ignore[no-redef]
        ResolutionActionTracker,
    )
    from backend.app.conversation.conversation_manager import (  # type: ignore[no-redef]
        ConversationManager,
    )
    from backend.app.conversation.conversation_state import (  # type: ignore[no-redef]
        ActionStatus,
        ConversationState,
        ConversationStatus,
        MessageRoleType,
        ResolutionStage,
    )
    from backend.app.conversation.resolution_progress_engine import (  # type: ignore[no-redef]
        ResolutionProgressEngine,
    )
    from backend.app.conversation.turn_classifier import (  # type: ignore[no-redef]
        TurnClassifier,
    )


class TestScenarioA_CustomerAlreadyAttemptedAction:
    """Scenario A: Customer explicitly mentions they already restarted device."""

    def test_customer_already_attempted_restart_not_repeated(self, tmp_path):
        mgr = ConversationManager(base_dir=tmp_path)
        state, agent_turn = mgr.create_conversation(
            initial_message="My iPhone Wi-Fi keeps dropping, and I already restarted my phone twice."
        )

        assert state.status == ConversationStatus.ACTIVE
        assert state.is_action_attempted("restart_device")
        # System should NOT recommend restart_device
        assert state.current_action is not None
        assert state.current_action.canonical_name != "restart_device"
        assert "restart" not in agent_turn.content.lower() or "already" in agent_turn.content.lower()


class TestScenarioB_ProgressiveTroubleshootingProgression:
    """Scenario B: Progressive sequence across multiple turns without loops."""

    def test_progressive_non_repeating_troubleshooting(self, tmp_path):
        mgr = ConversationManager(base_dir=tmp_path)
        state, turn1 = mgr.create_conversation(
            initial_message="My iPhone 14 won't connect to Wi-Fi at all."
        )
        action_1 = state.current_action.canonical_name

        # Turn 2: Customer reports failure
        state, turn2 = mgr.process_message(state.conversation_id, "I tried that and it didn't help.")
        action_2 = state.current_action.canonical_name
        assert action_2 != action_1

        # Turn 3: Customer reports failure again
        state, turn3 = mgr.process_message(state.conversation_id, "Still not working after that step.")
        action_3 = state.current_action.canonical_name
        assert action_3 not in (action_1, action_2)

        # Confirm all attempted actions are unique
        attempted = [a.canonical_name for a in state.attempted_actions]
        assert len(attempted) == len(set(attempted))


class TestScenarioC_ClarificationLoopPrevention:
    """Scenario C: System must never loop on clarification questions."""

    def test_clarification_does_not_loop_infinitely(self, tmp_path):
        mgr = ConversationManager(base_dir=tmp_path)
        # Ambiguous initial message
        state, turn1 = mgr.create_conversation(initial_message="help it's broken")
        assert state.stage == ResolutionStage.CLARIFYING or state.stage == ResolutionStage.AWAITING_RESULT

        # Customer replies vaguely
        state, turn2 = mgr.process_message(state.conversation_id, "my phone")
        # System should not ask the same question; should proceed to troubleshooting or escalate
        assert len(state.clarification_questions_asked) <= 2
        # Consecutive unclear limit escalates if still unclear
        state, turn3 = mgr.process_message(state.conversation_id, "idk")
        # After 2 consecutive unclear turns, it must escalate safely
        assert state.status == ConversationStatus.ESCALATED


class TestScenarioD_NegativeConfirmationResult:
    """Scenario D: Customer reports action failed, pipeline continues cleanly."""

    def test_negative_confirmation_advances_troubleshooting(self, tmp_path):
        mgr = ConversationManager(base_dir=tmp_path)
        state, turn1 = mgr.create_conversation(initial_message="My AirPods won't connect to my iPhone.")
        first_action = state.current_action.canonical_name

        # Customer reports it didn't work
        state, turn2 = mgr.process_message(state.conversation_id, "No, it's still doing the exact same thing.")
        assert state.status == ConversationStatus.ACTIVE
        assert state.is_action_attempted(first_action)
        # A new, different action is recommended
        assert state.current_action.canonical_name != first_action


class TestScenarioE_ResolutionConfirmation:
    """Scenario E: Customer confirms fix, conversation marks resolved."""

    def test_resolution_confirmation_closes_conversation(self, tmp_path):
        mgr = ConversationManager(base_dir=tmp_path)
        state, turn1 = mgr.create_conversation(initial_message="My iPhone is stuck on a black screen.")
        current_action = state.current_action.canonical_name

        # Customer confirms success
        state, turn2 = mgr.process_message(state.conversation_id, "That worked! Thanks so much, it powered right back on!")
        assert state.status == ConversationStatus.RESOLVED
        assert state.stage == ResolutionStage.RESOLVED
        assert state.resolution_summary is not None
        assert state.resolution_summary["resolved_by_action"] == current_action
        assert state.is_action_attempted(current_action)
        # Action status is SUCCESS
        successful_actions = [a for a in state.attempted_actions if a.canonical_name == current_action]
        assert len(successful_actions) == 1
        assert successful_actions[0].status == ActionStatus.SUCCESS


class TestScenarioF_ProblemWorseningUrgentEscalation:
    """Scenario F: Customer reports worsening, triggers immediate urgent escalation."""

    def test_problem_worsening_triggers_immediate_escalation(self, tmp_path):
        mgr = ConversationManager(base_dir=tmp_path)
        state, turn1 = mgr.create_conversation(initial_message="My iPhone battery is draining quickly.")
        assert state.status == ConversationStatus.ACTIVE

        # Customer reports severe worsening / hardware hazard
        state, turn2 = mgr.process_message(
            state.conversation_id,
            "Now it's completely dead, won't turn on at all, and it's getting extremely hot!"
        )
        assert state.status == ConversationStatus.ESCALATED
        assert state.stage == ResolutionStage.ESCALATED
        assert state.escalation_package is not None
        assert state.escalation_package.reason == "CUSTOMER_REPORTED_PROBLEM_WORSENING"
        assert state.escalation_package.recommended_specialist_tier == "TIER_2_TECHNICAL_URGENT"


class TestScenarioG_TopicDriftMultipleIssues:
    """Scenario G: Topic switch cleanly recognized."""

    def test_topic_drift_handled_cleanly(self, tmp_path):
        mgr = ConversationManager(base_dir=tmp_path)
        state, turn1 = mgr.create_conversation(initial_message="My iPhone Wi-Fi keeps disconnecting.")
        assert state.problem_family == "CONNECTIVITY_WIFI"

        # Customer reports unrelated problem
        state, turn2 = mgr.process_message(
            state.conversation_id,
            "Actually, never mind that, my screen digitizer has completely cracked and touch is not responding."
        )
        assert state.problem_family == "DISPLAY"


class TestScenarioH_ContextRetentionAcrossLongDialogue:
    """Scenario H: Fact retention across multiple turns."""

    def test_context_and_confirmed_facts_retained(self, tmp_path):
        mgr = ConversationManager(base_dir=tmp_path)
        state, turn1 = mgr.create_conversation(
            initial_message="I have an iPhone 13 Pro running iOS 17.2 on Verizon and cellular keeps dropping."
        )
        assert state.confirmed_facts.get("device_model") == "Iphone 13 Pro"
        assert state.confirmed_facts.get("os_version") == "IOS 17.2"
        assert state.confirmed_facts.get("carrier") == "Verizon"

        # Advance 4 turns
        mgr.process_message(state.conversation_id, "I tried that and still dropped calls.")
        mgr.process_message(state.conversation_id, "Same issue after doing that too.")
        mgr.process_message(state.conversation_id, "No luck at all.")

        # Ensure facts are strictly retained
        final_state = mgr.get_conversation(state.conversation_id)
        assert final_state.confirmed_facts.get("device_model") == "Iphone 13 Pro"
        assert final_state.confirmed_facts.get("os_version") == "IOS 17.2"
        assert final_state.confirmed_facts.get("carrier") == "Verizon"


class TestScenarioI_SemanticEquivalenceRepeatPrevention:
    """Scenario I: Action semantic aliases recognized ('power cycle' == 'reboot' == 'restart')."""

    def test_power_cycle_prevents_restart_recommendation(self, tmp_path):
        assert ActionCatalog.is_equivalent("power cycle", "restart_device")
        assert ActionCatalog.is_equivalent("reboot", "restart")
        assert ActionCatalog.is_equivalent("turned phone off and on", "restart_device")

        mgr = ConversationManager(base_dir=tmp_path)
        state, turn = mgr.create_conversation(
            initial_message="My Wi-Fi is broken and I already power cycled the device."
        )
        assert state.is_action_attempted("restart_device")
        assert state.current_action.canonical_name != "restart_device"


class TestScenarioJ_SafeHumanEscalationFullContextPackage:
    """Scenario J: Exhausted troubleshooting triggers escalation with full context package."""

    def test_exhausted_troubleshooting_produces_full_escalation_package(self, tmp_path):
        mgr = ConversationManager(base_dir=tmp_path)
        state, turn = mgr.create_conversation(
            initial_message="I have an iPhone 14 Pro running iOS 16.5 and Wi-Fi refuses to connect."
        )

        # Exhaust all progressive actions for Wi-Fi (check_airplane_mode, restart_device, forget_and_reconnect_wifi, reset_network_settings)
        sequence = ActionCatalog.get_progressive_sequence("CONNECTIVITY_WIFI")
        for _ in range(len(sequence) + 1):
            if state.status == ConversationStatus.ESCALATED:
                break
            state, turn = mgr.process_message(state.conversation_id, "I did that and it still does not connect.")

        assert state.status == ConversationStatus.ESCALATED
        assert state.stage == ResolutionStage.ESCALATED
        assert state.escalation_package is not None
        assert state.escalation_package.reason == "ALL_TROUBLESHOOTING_ACTIONS_EXHAUSTED"
        assert len(state.escalation_package.attempted_actions) >= len(sequence)
        assert "device_model" in state.escalation_package.confirmed_facts
