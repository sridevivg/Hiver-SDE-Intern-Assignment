"""
SupportGraph AI — Understanding Package (Phase 7)

Exposes:
- ProblemExtractor: Structured customer problem representation extractor
- CustomerProblemProfile: Pydantic model for device, symptom, and causality features
"""
from __future__ import annotations

try:
    from app.understanding.problem_extractor import CustomerProblemProfile, ProblemExtractor
except ModuleNotFoundError:
    from backend.app.understanding.problem_extractor import (  # type: ignore[no-redef]
        CustomerProblemProfile,
        ProblemExtractor,
    )

__all__ = ["CustomerProblemProfile", "ProblemExtractor"]
