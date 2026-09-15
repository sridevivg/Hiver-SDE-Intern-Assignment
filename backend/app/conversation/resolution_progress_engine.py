"""
SupportGraph AI — Resolution Progress Engine & Multi-Turn State Machine.

Orchestrates multi-turn conversation progression:
- Classifies customer messages and tracks confirmed facts.
- Evaluates customer action attempt results.
- Determines when targeted clarification is strictly necessary.
- Progressively recommends untried non-destructive troubleshooting actions.
- Automatically triggers safe human escalation when actions are exhausted,
  problem worsens, or messages remain unclear.
- Recognizes successful resolution and cleanly closes conversations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

try:
    from app.conversation.action_catalog import ActionCatalog
    from app.conversation.action_tracker import ResolutionActionTracker
    from app.conversation.clarification_engine import ClarificationEngine
    from app.conversation.conversation_state import (
        ActionStatus,
        ConversationState,
        ConversationStatus,
        ConversationTurn,
        MessageRoleType,
        ResolutionStage,
        TroubleshootingAction,
    )
    from app.core.logging import get_logger
    from app.understanding.problem_extractor import ProblemExtractor
except ModuleNotFoundError:
    from backend.app.conversation.action_catalog import ActionCatalog  # type: ignore[no-redef]
    from backend.app.conversation.action_tracker import (  # type: ignore[no-redef]
        ResolutionActionTracker,
    )
    from backend.app.conversation.clarification_engine import (  # type: ignore[no-redef]
        ClarificationEngine,
    )
    from backend.app.conversation.conversation_state import (  # type: ignore[no-redef]
        ActionStatus,
        ConversationState,
        ConversationStatus,
        ConversationTurn,
        MessageRoleType,
        ResolutionStage,
        TroubleshootingAction,
    )
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]
    from backend.app.understanding.problem_extractor import (  # type: ignore[no-redef]
        ProblemExtractor,
    )

logger = get_logger(__name__)

# Lazy historical index accessor to avoid slow startup in tests
_corpus_index = None


def get_corpus_index():
    global _corpus_index
    if _corpus_index is None:
        try:
            try:
                from app.retrieval.historical_corpus_index import HistoricalCorpusIndex
            except ModuleNotFoundError:
                from backend.app.retrieval.historical_corpus_index import HistoricalCorpusIndex  # type: ignore[no-redef]
            _corpus_index = HistoricalCorpusIndex()
        except Exception as exc:
            logger.warning("Could not initialize HistoricalCorpusIndex in progress engine: %s", exc)
            _corpus_index = False
    return _corpus_index if _corpus_index is not False else None


class ResolutionProgressEngine:
    """
    State machine controlling conversation progression and resolution lifecycle.
    """

    @classmethod
    def process_turn(
        cls,
        state: ConversationState,
        customer_message: str,
    ) -> ConversationTurn:
        """
        Process incoming customer turn, update state, and generate agent response turn.
        """
        try:
            from app.conversation.turn_classifier import TurnClassifier
        except ModuleNotFoundError:
            from backend.app.conversation.turn_classifier import TurnClassifier  # type: ignore[no-redef]

        # 1. Classify customer turn
        role_type, extracted_facts, referenced_action = TurnClassifier.classify_turn(
            customer_message, state
        )

        # 2. Record customer turn in state
        customer_turn = state.add_turn(
            speaker="customer",
            content=customer_message,
            role_type=role_type,
            identified_facts=extracted_facts,
            action_referenced=referenced_action,
        )

        # 3. Update confirmed facts from customer turn
        for k, v in extracted_facts.items():
            state.update_fact(k, v, confirmed=True)

        # 4. Handle UNCLEAR_MESSAGE
        if role_type == MessageRoleType.UNCLEAR_MESSAGE:
            state.consecutive_unclear_turns += 1
            if state.consecutive_unclear_turns >= 2:
                # Escalate to human
                esc_package = state.escalate(
                    reason="UNRESOLVED_AMBIGUITY_OR_UNCLEAR_MESSAGES",
                    summary="Customer message could not be clarified after repeated attempts.",
                    specialist_tier="TIER_1_GENERAL",
                )
                agent_reply = (
                    "I want to make sure you get the exact support you need. Since I'm having "
                    "trouble understanding the issue from our conversation, I am connecting you "
                    "with a human support specialist who can assist you directly."
                )
                return state.add_turn(
                    speaker="agent",
                    content=agent_reply,
                    role_type=MessageRoleType.CONTINUATION,
                )
            else:
                # Ask politely for clarification
                state.stage = ResolutionStage.CLARIFYING
                agent_reply = (
                    "I didn't quite catch that. Could you describe in a little more detail "
                    "what issue you are experiencing with your device?"
                )
                return state.add_turn(
                    speaker="agent",
                    content=agent_reply,
                    role_type=MessageRoleType.CONTINUATION,
                )

        # Non-unclear turn resets consecutive unclear counter
        state.consecutive_unclear_turns = 0

        # 5. Handle PROBLEM_WORSENING
        if role_type == MessageRoleType.PROBLEM_WORSENING:
            state.escalate(
                reason="CUSTOMER_REPORTED_PROBLEM_WORSENING",
                summary="Customer reported that the problem deteriorated or device worsened.",
                specialist_tier="TIER_2_TECHNICAL_URGENT",
            )
            agent_reply = (
                "I am very sorry to hear that the issue has gotten worse. To prevent any further "
                "damage to your device, I have stopped automated troubleshooting and escalated "
                "your case to our Senior Technical Support team. A specialist will assist you shortly."
            )
            return state.add_turn(
                speaker="agent",
                content=agent_reply,
                role_type=MessageRoleType.CONTINUATION,
            )

        # 6. Handle RESOLUTION_CONFIRMATION
        if role_type == MessageRoleType.RESOLUTION_CONFIRMATION:
            if state.current_action:
                state.current_action.status = ActionStatus.SUCCESS
                state.record_attempted_action(
                    canonical_name=state.current_action.canonical_name,
                    description=state.current_action.description,
                    status=ActionStatus.SUCCESS,
                    result_notes=customer_message,
                    turn_index=customer_turn.turn_index,
                )
            state.mark_resolved(
                summary="Customer confirmed the issue is successfully resolved.",
                resolved_by_action=state.current_action.canonical_name if state.current_action else None,
            )
            agent_reply = (
                "Wonderful! I'm glad to hear that resolved the problem for you. Thank you for "
                "contacting Apple Support. If you experience any other issues, don't hesitate to reach back out."
            )
            return state.add_turn(
                speaker="agent",
                content=agent_reply,
                role_type=MessageRoleType.RESOLUTION_CONFIRMATION,
            )

        # 7. Handle customer action attempt results
        # If current_action was waiting for result and user replied (negative or attempt result)
        if state.current_action and state.stage == ResolutionStage.AWAITING_RESULT:
            state.record_attempted_action(
                canonical_name=state.current_action.canonical_name,
                description=state.current_action.description,
                status=ActionStatus.FAILED,
                result_notes=customer_message,
                turn_index=customer_turn.turn_index,
            )
            state.current_action = None

        # Check for any action mentioned by customer as already attempted
        ResolutionActionTracker.process_customer_turn(state, customer_message, customer_turn.turn_index)

        # 8. Identify or update problem family
        if not state.problem_family:
            try:
                profile = ProblemExtractor().extract(customer_message)
                if profile and profile.primary_problem_family:
                    state.problem_family = profile.primary_problem_family.value
                    state.operational_goal = profile.primary_symptom
            except Exception as exc:
                logger.warning("ProblemExtractor error: %s", exc)
                state.problem_family = "GENERAL_DEVICE_FUNCTIONALITY"

        # 9. Handle NEW_PROBLEM_IN_EXISTING_CONVERSATION
        if role_type == MessageRoleType.NEW_PROBLEM_IN_EXISTING_CONVERSATION:
            try:
                profile = ProblemExtractor().extract(customer_message)
                if profile and profile.primary_problem_family:
                    old_fam = state.problem_family
                    state.problem_family = profile.primary_problem_family.value
                    state.operational_goal = profile.primary_symptom
                    # Transition stage
                    state.stage = ResolutionStage.UNDERSTANDING
            except Exception as exc:
                logger.warning("ProblemExtractor switch error: %s", exc)

        # 10. Check if targeted clarification is strictly required
        should_clarify, missing_field, question_text = ClarificationEngine.should_ask_clarification(
            state, customer_message
        )
        if should_clarify and question_text:
            state.stage = ResolutionStage.CLARIFYING
            state.clarification_questions_asked.append(missing_field)
            return state.add_turn(
                speaker="agent",
                content=question_text,
                role_type=MessageRoleType.CLARIFICATION_ANSWER,
            )

        # 11. Select next untried progressive troubleshooting action
        next_action, is_exhausted = ResolutionActionTracker.select_next_action(state)

        if is_exhausted or next_action is None:
            # All actions exhausted -> Safe Human Escalation
            state.escalate(
                reason="ALL_TROUBLESHOOTING_ACTIONS_EXHAUSTED",
                summary=f"All progressive troubleshooting actions exhausted for {state.problem_family}.",
                specialist_tier="TIER_2_TECHNICAL",
            )
            attempted_names = [a.canonical_name.replace("_", " ") for a in state.attempted_actions]
            attempts_str = ", ".join(attempted_names) if attempted_names else "standard steps"
            agent_reply = (
                f"We have tried all standard troubleshooting steps for this issue (including {attempts_str}) "
                "without success. To provide you with deeper diagnostics and hardware support, "
                "I am escalating your case directly to an Apple Technical Specialist."
            )
            return state.add_turn(
                speaker="agent",
                content=agent_reply,
                role_type=MessageRoleType.CONTINUATION,
            )

        # 12. Format next progressive action recommendation
        canonical_name = next_action["canonical_name"]
        description = next_action["description"]
        expected_outcome = next_action.get("expected_outcome", "")

        # Record action in state
        action_obj = TroubleshootingAction(
            canonical_name=canonical_name,
            description=description,
            status=ActionStatus.IN_PROGRESS,
            expected_outcome=expected_outcome,
            is_destructive=next_action.get("is_destructive", False),
        )
        state.current_action = action_obj
        state.recommended_action_history.append(canonical_name)
        state.stage = ResolutionStage.AWAITING_RESULT

        # Optional: Evidence retrieval for grounding
        evidence_used = []
        idx = get_corpus_index()
        if idx:
            try:
                candidates = idx.search_candidates(customer_message, top_n=2)
                for cand_rec, sim, boost in candidates:
                    evidence_used.append({
                        "case_id": cand_rec.case_id,
                        "similarity": round(float(sim), 3),
                        "boosted_similarity": round(float(boost), 3),
                        "problem_family": cand_rec.primary_problem_family.value if cand_rec.primary_problem_family else None,
                    })
            except Exception as exc:
                logger.debug("Evidence retrieval skipped in turn: %s", exc)

        action_display_name = canonical_name.replace("_", " ").title()
        reply_lines = [
            f"Let's try the following troubleshooting step: **{action_display_name}**.",
            f"\n{description}",
        ]
        if expected_outcome:
            reply_lines.append(f"\n*Expected result*: {expected_outcome}")
        reply_lines.append("\nPlease give that a try and let me know if it resolves the issue or if the problem persists.")

        agent_reply = "\n".join(reply_lines)

        return state.add_turn(
            speaker="agent",
            content=agent_reply,
            role_type=MessageRoleType.ACTION_ATTEMPT_RESULT,
            action_referenced=canonical_name,
            evidence_used=evidence_used,
            confidence_score=0.92,
        )
