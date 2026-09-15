"""
SupportGraph AI — Candidate Evidence Store (Phase 13).

Stores candidate human resolutions in data/evidence_candidates/.
Strictly isolated from search/retrieval index: candidate resolutions CANNOT
be searched or used by the SupportResolutionEngine until promoted.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from app.core.logging import get_logger
    from app.feedback.schemas import HumanResolution, ResolutionLifecycleState
except ModuleNotFoundError:
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]
    from backend.app.feedback.schemas import (  # type: ignore[no-redef]
        HumanResolution,
        ResolutionLifecycleState,
    )

logger = get_logger(__name__)

DEFAULT_CANDIDATE_DIR = Path("data/evidence_candidates")


class CandidateEvidenceStore:
    """
    File-backed store for candidate human specialist resolutions.
    Ensures persistent, isolated storage prior to validation and approval.
    """

    def __init__(self, base_dir: Optional[Path | str] = None) -> None:
        self.base_dir = Path(base_dir or DEFAULT_CANDIDATE_DIR)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _file_path(self, resolution_id: str) -> Path:
        return self.base_dir / f"{resolution_id}.json"

    def save_candidate(self, resolution: HumanResolution) -> str:
        """Persist candidate resolution to disk."""
        path = self._file_path(resolution.resolution_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(resolution.model_dump(mode="json"), f, indent=2, default=str)
        logger.info("Saved candidate resolution %s (%s)", resolution.resolution_id, resolution.promotion_status.value)
        return resolution.resolution_id

    def get_candidate(self, resolution_id: str) -> Optional[HumanResolution]:
        """Retrieve candidate resolution by ID."""
        path = self._file_path(resolution_id)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return HumanResolution.model_validate(data)
        except Exception as exc:
            logger.error("Failed to load candidate resolution %s: %s", resolution_id, exc)
            return None

    def list_candidates(
        self,
        status: Optional[ResolutionLifecycleState] = None,
        problem_family: Optional[str] = None,
    ) -> List[HumanResolution]:
        """List all candidate resolutions matching optional filters."""
        candidates: List[HumanResolution] = []
        for file in sorted(self.base_dir.glob("*.json")):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                cand = HumanResolution.model_validate(data)
                if status and cand.promotion_status != status:
                    continue
                if problem_family and cand.problem_family != problem_family:
                    continue
                candidates.append(cand)
            except Exception as exc:
                logger.warning("Error reading candidate file %s: %s", file, exc)
        return candidates

    def update_candidate(self, resolution: HumanResolution) -> bool:
        """Update existing candidate record."""
        path = self._file_path(resolution.resolution_id)
        if not path.exists():
            return False
        self.save_candidate(resolution)
        return True
