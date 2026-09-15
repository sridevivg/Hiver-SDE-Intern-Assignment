"""
SupportGraph AI — Operational Problem Profile (Phase 10.2)

Defines the enhanced structured representation of customer support issues,
grounded in operational problem families rather than narrow symptom categories.
"""
from __future__ import annotations

from typing import Any, List, Literal, Optional

from pydantic import ConfigDict, Field

try:
    from app.retrieval.problem_family_registry import OperationalProblemFamily
    from app.understanding.problem_extractor import CustomerProblemProfile
except ModuleNotFoundError:
    from backend.app.retrieval.problem_family_registry import (  # type: ignore[no-redef]
        OperationalProblemFamily,
    )
    from backend.app.understanding.problem_extractor import (  # type: ignore[no-redef]
        CustomerProblemProfile,
    )


class OperationalProblemProfile(CustomerProblemProfile):
    """
    Extensible operational customer problem representation.

    Extends CustomerProblemProfile to preserve 100% backward compatibility while
    introducing operational problem families and operational entity tracking.
    """
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    primary_problem_family: OperationalProblemFamily = Field(
        default=OperationalProblemFamily.GENERAL_DEVICE_FUNCTIONALITY,
        description="Core operational category representing the functional failure domain",
    )
    secondary_problem_families: list[OperationalProblemFamily] = Field(
        default_factory=list,
        description="Secondary or co-occurring operational problem categories",
    )
    operational_entities: list[str] = Field(
        default_factory=list,
        description="Key entities extracted (e.g. apps, ports, buttons, services, versions)",
    )
    context_trigger: Optional[str] = Field(
        default=None,
        description="Specific contextual event triggering the problem (e.g. update, charging, pairing)",
    )
    family_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence score for primary problem family classification",
    )

    def to_dict(self) -> dict[str, Any]:
        data = self.model_dump()
        data["primary_problem_family"] = self.primary_problem_family.value
        data["secondary_problem_families"] = [f.value for f in self.secondary_problem_families]
        return data
