"""
SupportGraph AI — API Routes Package (Phase 6)
"""
from __future__ import annotations

try:
    from app.api.routes.intent import router as intent_router
except ModuleNotFoundError:
    from backend.app.api.routes.intent import router as intent_router  # type: ignore[no-redef]

__all__ = ["intent_router"]
