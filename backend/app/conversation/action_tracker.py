"""
SupportGraph AI — Resolution Action Tracker & Repeat Prevention.

Tracks attempted troubleshooting actions, recognizes customer-reported prior attempts,
maintains attempt history, and guarantees zero repeated or semantically equivalent
troubleshooting recommendations.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from app.conversation.action_catalog import ActionCatalog
    from app.conversation.conversation_state import (
        ActionStatus,
        ConversationState,
        TroubleshootingAction,
    )
except ModuleNotFoundError:
    from backend.app.conversation.action_catalog import ActionCatalog  # type: ignore[no-redef]
    from backend.app.conversation.conversation_state import (  # type: ignore[no-redef]
        ActionStatus,
        ConversationState,
        TroubleshootingAction,
    )


class ResolutionActionTracker:
    """
    Manages tracking of attempted troubleshooting actions and guarantees
    that no attempted or semantically equivalent action is re-recommended.
    """

    @classmethod
    def process_customer_turn(
        cls,
        state: ConversationState,
        customer_text: str,
        turn_index: int,
    ) -> Optional[TroubleshootingAction]:
        """
        Inspect customer message for any reported action attempt.
        If an action was attempted, records it in state.attempted_actions.
        Returns the recorded TroubleshootingAction if found, else None.
        """
        text_lower = customer_text.lower()

        # Check if the customer references an action
        action_name = ActionCatalog.identify_action(customer_text)
        if not action_name:
            return None

        # Check for attempt indicators
        attempt_indicators = [
            r"\b(already|tried|did|done|attempted|performed)\b",
            r"\b(i\s*(?:restarted|rebooted|toggled|unpaired|reset|cleaned|updated|checked|cleared))\b",
            r"\b(after\s*(?:restarting|rebooting|resetting|updating))\b",
            r"\b(didn'?t\s*work|still\s*not\s*working|no\s*luck|same\s*issue)\b",
        ]

        has_attempt_indicator = any(re.search(ind, text_lower) for ind in attempt_indicators)

        # Or if the current state was awaiting results of this action
        is_current_action = (
            state.current_action is not None
            and state.current_action.canonical_name == action_name
        )

        if has_attempt_indicator or is_current_action:
            meta = ActionCatalog.get_action_metadata(action_name)
            description = meta.get("description", action_name) if meta else action_name
            is_destructive = meta.get("is_destructive", False) if meta else False

            action = state.record_attempted_action(
                canonical_name=action_name,
                description=description,
                status=ActionStatus.FAILED,
                result_notes=customer_text,
                turn_index=turn_index,
                is_destructive=is_destructive,
            )
            return action

        return None

    @classmethod
    def get_attempted_canonical_names(cls, state: ConversationState) -> Set[str]:
        """
        Return the set of canonical names for all actions attempted or already recommended.
        """
        attempted: Set[str] = set()
        for act in state.attempted_actions:
            attempted.add(act.canonical_name)

        # Also add anything in recommended_action_history to prevent repeating
        for rec in state.recommended_action_history:
            canonical = ActionCatalog.get_canonical_name(rec)
            if canonical:
                attempted.add(canonical)
            else:
                attempted.add(rec)

        return attempted

    @classmethod
    def select_next_action(
        cls,
        state: ConversationState,
    ) -> Tuple[Optional[Dict[str, Any]], bool]:
        """
        Select the next unattempted progressive action for the current problem family.
        Returns: (next_action_dict, is_exhausted)
        - If an unattempted safe action is available: (action_dict, False)
        - If all progressive actions have been attempted: (None, True)
        """
        attempted_set = cls.get_attempted_canonical_names(state)
        next_action = ActionCatalog.get_next_untried_action(
            problem_family=state.problem_family,
            attempted_canonical_names=attempted_set,
        )

        if next_action is None:
            return None, True

        return next_action, False

    @classmethod
    def is_action_allowed(
        cls,
        state: ConversationState,
        candidate_action_name: str,
    ) -> bool:
        """
        Check if a candidate action is allowed to be recommended.
        Returns False if the action or any semantic alias was already attempted or recommended.
        """
        canonical = ActionCatalog.get_canonical_name(candidate_action_name)
        if not canonical:
            canonical = candidate_action_name.strip().lower()

        attempted_set = cls.get_attempted_canonical_names(state)
        return canonical not in attempted_set
