"""
SupportGraph AI — NLP LLM Assistant Module (Phase 5.7)

Re-exports the AnnotationAssistant service with hierarchical intent decision logic.
"""
from __future__ import annotations

try:
    from app.evaluation.annotation_assistant import (
        AISuggestion,
        AnnotationAssistant,
        GroqAnnotationAssistant,
        GroqAuthenticationError,
        GroqConfigurationError,
        GroqConnectionError,
        GroqRateLimitError,
    )
except ModuleNotFoundError:
    from backend.app.evaluation.annotation_assistant import (  # type: ignore[no-redef]
        AISuggestion,
        AnnotationAssistant,
        GroqAnnotationAssistant,
        GroqAuthenticationError,
        GroqConfigurationError,
        GroqConnectionError,
        GroqRateLimitError,
    )

__all__ = [
    "AnnotationAssistant",
    "AISuggestion",
    "GroqAnnotationAssistant",
    "GroqConfigurationError",
    "GroqAuthenticationError",
    "GroqRateLimitError",
    "GroqConnectionError",
]
