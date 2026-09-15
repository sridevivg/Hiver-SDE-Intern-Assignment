"""
SupportGraph AI — Intent Classification & Uncertainty Routing Package (Phase 6)

Exposes:
- TopKIntentClassifier: Multi-candidate intent prediction engine
- IntentRouter: Deterministic uncertainty-aware routing engine
- RuntimeEscalationManager: Append-only HITL escalation review logger
- compute_normalized_entropy: Shannon entropy uncertainty metric
"""
from __future__ import annotations

try:
    from app.intent.classifier import TopKIntentClassifier, compute_normalized_entropy
    from app.intent.escalation import RuntimeEscalationManager
    from app.intent.router import IntentRouter
except ModuleNotFoundError:
    from backend.app.intent.classifier import (  # type: ignore[no-redef]
        TopKIntentClassifier,
        compute_normalized_entropy,
    )
    from backend.app.intent.escalation import RuntimeEscalationManager  # type: ignore[no-redef]
    from backend.app.intent.router import IntentRouter  # type: ignore[no-redef]

__all__ = [
    "TopKIntentClassifier",
    "IntentRouter",
    "RuntimeEscalationManager",
    "compute_normalized_entropy",
]
