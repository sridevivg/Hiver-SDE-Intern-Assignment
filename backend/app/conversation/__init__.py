"""
SupportGraph AI — Conversation Package (Phase 11).
"""

try:
    from app.conversation.action_catalog import ActionCatalog
    from app.conversation.action_tracker import ResolutionActionTracker
    from app.conversation.clarification_engine import ClarificationEngine
    from app.conversation.conversation_manager import (
        ConversationManager,
        get_conversation_manager,
    )
    from app.conversation.conversation_state import (
        ActionStatus,
        ConversationState,
        ConversationStatus,
        ConversationTurn,
        EscalationPackage,
        MessageRoleType,
        ResolutionStage,
        TroubleshootingAction,
    )
    from app.conversation.resolution_progress_engine import ResolutionProgressEngine
    from app.conversation.turn_classifier import TurnClassifier
except ModuleNotFoundError:
    from backend.app.conversation.action_catalog import ActionCatalog  # type: ignore[no-redef]
    from backend.app.conversation.action_tracker import (  # type: ignore[no-redef]
        ResolutionActionTracker,
    )
    from backend.app.conversation.clarification_engine import (  # type: ignore[no-redef]
        ClarificationEngine,
    )
    from backend.app.conversation.conversation_manager import (  # type: ignore[no-redef]
        ConversationManager,
        get_conversation_manager,
    )
    from backend.app.conversation.conversation_state import (  # type: ignore[no-redef]
        ActionStatus,
        ConversationState,
        ConversationStatus,
        ConversationTurn,
        EscalationPackage,
        MessageRoleType,
        ResolutionStage,
        TroubleshootingAction,
    )
    from backend.app.conversation.resolution_progress_engine import (  # type: ignore[no-redef]
        ResolutionProgressEngine,
    )
    from backend.app.conversation.turn_classifier import (  # type: ignore[no-redef]
        TurnClassifier,
    )

__all__ = [
    "ActionCatalog",
    "ActionStatus",
    "ClarificationEngine",
    "ConversationManager",
    "ConversationState",
    "ConversationStatus",
    "ConversationTurn",
    "EscalationPackage",
    "MessageRoleType",
    "ResolutionActionTracker",
    "ResolutionProgressEngine",
    "ResolutionStage",
    "TroubleshootingAction",
    "TurnClassifier",
    "get_conversation_manager",
]
