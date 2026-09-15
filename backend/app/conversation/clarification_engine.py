"""
SupportGraph AI — Clarification Question Engine.

Determines whether clarification is strictly necessary before troubleshooting,
prevents redundant or multi-question dumps, and generates crisp, single-question inquiries.
"""

from __future__ import annotations

try:
    from app.conversation.conversation_state import ConversationState
except ModuleNotFoundError:
    from backend.app.conversation.conversation_state import (  # type: ignore[no-redef]
        ConversationState,
    )


class ClarificationEngine:
    """
    Evaluates when clarifying questions are strictly necessary.
    Rules:
    1. Do not ask clarification if a safe non-destructive action can proceed immediately.
    2. Never ask multiple questions in a single turn (max 1 question).
    3. Do not re-ask information already present in confirmed_facts or previously asked.
    4. Only ask if missing information materially affects the troubleshooting path.
    """

    @classmethod
    def should_ask_clarification(
        cls,
        state: ConversationState,
        text: str,
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Evaluate if a clarification question should be asked.
        Returns: (should_ask, missing_field_name, question_text)
        """
        # If we already asked a question in the previous turn and customer didn't answer it,
        # or if we have already asked 2 clarification questions in this conversation,
        # do not trap the user in a clarification loop; proceed to troubleshooting.
        if len(state.clarification_questions_asked) >= 2:
            return False, None, None

        # Do not interrupt active progressive troubleshooting
        if state.stage in ("TROUBLESHOOTING", "AWAITING_RESULT") and state.attempted_actions:
            return False, None, None

        # Check for genuine ambiguity in problem family or severe underspecification
        text_lower = text.lower().strip()

        # Very vague messages (e.g. "it doesn't work", "broken", "help me with my phone")
        if len(text.split()) <= 4 and not state.problem_family:
            field = "problem_symptom"
            if field not in state.clarification_questions_asked:
                q = "Could you tell me a bit more about what specifically is happening with your device?"
                return True, field, q

        # Network issues where home router vs general Wi-Fi distinction changes resolution
        if state.problem_family == "CONNECTIVITY_WIFI":
            if (
                "network_scope" not in state.confirmed_facts
                and "network_scope" not in state.clarification_questions_asked
                and len(state.attempted_actions) >= 2
            ):
                field = "network_scope"
                q = "Does this disconnect occur on all Wi-Fi networks, or only on your home network?"
                return True, field, q

        # Audio issues where hardware vs Bluetooth audio output distinction is critical
        if state.problem_family == "AUDIO":
            if (
                "audio_output_type" not in state.confirmed_facts
                and "audio_output_type" not in state.clarification_questions_asked
                and "speaker" not in text_lower
                and "airpods" not in text_lower
                and "headphones" not in text_lower
                and len(state.attempted_actions) >= 1
            ):
                field = "audio_output_type"
                q = "Are you experiencing no sound through the built-in device speakers, or through headphones/Bluetooth?"
                return True, field, q

        # Storage / Update issues requiring device model or storage info
        if state.problem_family in ("SYSTEM_UPDATE", "STORAGE") and len(state.attempted_actions) >= 1:
            if (
                "storage_available" not in state.confirmed_facts
                and "storage_available" not in state.clarification_questions_asked
            ):
                field = "storage_available"
                q = "Could you check approximately how much free space is showing in Settings > General > iPhone Storage?"
                return True, field, q

        return False, None, None
