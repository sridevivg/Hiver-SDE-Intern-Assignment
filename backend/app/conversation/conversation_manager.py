"""
SupportGraph AI — Conversation Manager & Isolated Multi-Turn Persistence.

Provides thread-safe in-memory caching, disk persistence in data/conversations/{id}.json,
and append-only audit trail logging in data/conversations/audit/{id}.jsonl.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

try:
    from app.conversation.conversation_state import (
        ConversationState,
        ConversationStatus,
        ConversationTurn,
    )
    from app.conversation.resolution_progress_engine import ResolutionProgressEngine
    from app.core.logging import get_logger
except ModuleNotFoundError:
    from backend.app.conversation.conversation_state import (  # type: ignore[no-redef]
        ConversationState,
        ConversationStatus,
        ConversationTurn,
    )
    from backend.app.conversation.resolution_progress_engine import (  # type: ignore[no-redef]
        ResolutionProgressEngine,
    )
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]

logger = get_logger(__name__)

CONVERSATIONS_DIR = Path("data/conversations")
AUDIT_DIR = CONVERSATIONS_DIR / "audit"


class ConversationManager:
    """
    Manages active conversations, state persistence, turn processing, and audit logs.
    """

    def __init__(
        self,
        base_dir: Optional[Path] = None,
    ) -> None:
        self.base_dir = Path(base_dir) if base_dir else CONVERSATIONS_DIR
        self.audit_dir = self.base_dir / "audit"
        self._cache: Dict[str, ConversationState] = {}
        self._lock = Lock()

        # Ensure directories exist
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.audit_dir.mkdir(parents=True, exist_ok=True)

    def create_conversation(
        self,
        customer_id: Optional[str] = None,
        initial_message: Optional[str] = None,
    ) -> Tuple[ConversationState, Optional[ConversationTurn]]:
        """
        Initialize a new conversation state. If initial_message is provided,
        processes the first turn immediately.
        """
        with self._lock:
            state = ConversationState(customer_id=customer_id)
            self._cache[state.conversation_id] = state

        # Audit initial creation
        self._append_audit(state.conversation_id, {
            "event": "CONVERSATION_CREATED",
            "conversation_id": state.conversation_id,
            "customer_id": customer_id,
            "timestamp": datetime.utcnow().isoformat(),
        })

        first_turn = None
        if initial_message:
            state, first_turn = self.process_message(state.conversation_id, initial_message)
        else:
            self.save_state(state)

        return state, first_turn

    def get_conversation(self, conversation_id: str) -> Optional[ConversationState]:
        """Retrieve conversation state by ID from cache or disk."""
        with self._lock:
            if conversation_id in self._cache:
                return self._cache[conversation_id]

        file_path = self.base_dir / f"{conversation_id}.json"
        if not file_path.exists():
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            state = ConversationState(**data)
            with self._lock:
                self._cache[conversation_id] = state
            return state
        except Exception as exc:
            logger.error("Failed to load conversation %s: %s", conversation_id, exc)
            return None

    def process_message(
        self,
        conversation_id: str,
        customer_message: str,
    ) -> Tuple[ConversationState, ConversationTurn]:
        """
        Submit a customer message into an active conversation, process it through
        the ResolutionProgressEngine, update state, and persist audit record.
        """
        state = self.get_conversation(conversation_id)
        if not state:
            raise ValueError(f"Conversation {conversation_id} not found.")

        if state.status in (ConversationStatus.RESOLVED, ConversationStatus.ESCALATED):
            # Reopen or note continuation if customer reaches back out
            if state.status == ConversationStatus.RESOLVED:
                state.status = ConversationStatus.ACTIVE

        agent_turn = ResolutionProgressEngine.process_turn(state, customer_message)

        # Persist updated state
        self.save_state(state)

        # Append to audit trail
        self._append_audit(conversation_id, {
            "event": "TURN_PROCESSED",
            "conversation_id": conversation_id,
            "timestamp": datetime.utcnow().isoformat(),
            "customer_message": customer_message,
            "agent_response": agent_turn.content,
            "role_type": agent_turn.role_type,
            "status": state.status,
            "stage": state.stage,
            "confirmed_facts": state.confirmed_facts,
            "attempted_actions": [a.canonical_name for a in state.attempted_actions],
        })

        return state, agent_turn

    def resolve_conversation(
        self,
        conversation_id: str,
        summary: Optional[str] = None,
    ) -> ConversationState:
        """Explicitly resolve a conversation."""
        state = self.get_conversation(conversation_id)
        if not state:
            raise ValueError(f"Conversation {conversation_id} not found.")

        state.mark_resolved(summary=summary)
        self.save_state(state)

        self._append_audit(conversation_id, {
            "event": "CONVERSATION_RESOLVED",
            "conversation_id": conversation_id,
            "timestamp": datetime.utcnow().isoformat(),
            "summary": summary,
        })

        return state

    def save_state(self, state: ConversationState) -> None:
        """Persist state to disk in isolated JSON format."""
        file_path = self.base_dir / f"{state.conversation_id}.json"
        temp_path = self.base_dir / f"{state.conversation_id}.json.tmp"

        with self._lock:
            self._cache[state.conversation_id] = state

        try:
            # Atomic write via temporary file
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(state.model_dump_json(indent=2))
            os.replace(temp_path, file_path)
        except Exception as exc:
            logger.error("Failed to persist conversation %s: %s", state.conversation_id, exc)
            if temp_path.exists():
                temp_path.unlink()

    def get_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Return full turn history for a conversation."""
        state = self.get_conversation(conversation_id)
        if not state:
            return []
        return [turn.model_dump() for turn in state.turns]

    def get_audit_trail(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Read append-only audit trail for a conversation."""
        audit_file = self.audit_dir / f"{conversation_id}.jsonl"
        if not audit_file.exists():
            return []

        entries = []
        with open(audit_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return entries

    def _append_audit(self, conversation_id: str, payload: Dict[str, Any]) -> None:
        """Append entry to audit log file."""
        audit_file = self.audit_dir / f"{conversation_id}.jsonl"
        try:
            with open(audit_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, default=str) + "\n")
        except Exception as exc:
            logger.warning("Failed to write audit log for %s: %s", conversation_id, exc)


# Singleton conversation manager instance
_global_conversation_manager: Optional[ConversationManager] = None


def get_conversation_manager() -> ConversationManager:
    """Get or create singleton ConversationManager."""
    global _global_conversation_manager
    if _global_conversation_manager is None:
        _global_conversation_manager = ConversationManager()
    return _global_conversation_manager
