"""
SupportGraph AI — Problem Family Registry Re-export (Phase 10.2)

Re-exports problem family registry components for the retrieval package.
"""
from __future__ import annotations

try:
    from app.understanding.problem_family_registry import (
        PROBLEM_FAMILY_DEFINITIONS,
        OperationalProblemFamily,
        ProblemFamilyDefinition,
        ProblemFamilyDetector,
    )
except ModuleNotFoundError:
    from backend.app.understanding.problem_family_registry import (  # type: ignore[no-redef]
        PROBLEM_FAMILY_DEFINITIONS,
        OperationalProblemFamily,
        ProblemFamilyDefinition,
        ProblemFamilyDetector,
    )

__all__ = [
    "OperationalProblemFamily",
    "ProblemFamilyDefinition",
    "ProblemFamilyDetector",
    "PROBLEM_FAMILY_DEFINITIONS",
]
