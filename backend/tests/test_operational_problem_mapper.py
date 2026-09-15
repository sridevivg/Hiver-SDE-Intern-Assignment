"""
SupportGraph AI — Tests: Operational Problem Mapper (Phase 10.2)

Tests:
- ProblemExtractor outputs valid CustomerProblemProfile / OperationalProblemProfile
- All extracted fields (primary_problem_family, entities, triggers) are populated
- Backward compatibility with existing pipeline consumers
"""
from __future__ import annotations

import pytest

from backend.app.retrieval.problem_family_registry import OperationalProblemFamily
from backend.app.understanding.operational_problem_profile import (
    OperationalProblemProfile,
)
from backend.app.understanding.problem_extractor import (
    CustomerProblemProfile,
    ProblemExtractor,
)


def test_extractor_populates_operational_family_wifi():
    """Verify ProblemExtractor populates primary_problem_family for WiFi."""
    extractor = ProblemExtractor()
    profile = extractor.extract("why does WiFi turn on by itself when I turn it off manually @AppleSupport")
    assert profile.primary_problem_family == OperationalProblemFamily.CONNECTIVITY_WIFI
    assert "Wi-Fi" in profile.evidence_summary or "CONNECTIVITY_WIFI" in profile.evidence_summary


def test_extractor_populates_operational_family_mac_freeze():
    """Verify ProblemExtractor populates primary_problem_family for Mac crash/freeze."""
    extractor = ProblemExtractor()
    profile = extractor.extract("@AppleSupport my MacBook Air has the dreaded spinning beach ball. Help please.")
    assert profile.primary_problem_family in (
        OperationalProblemFamily.CRASH_FREEZE,
        OperationalProblemFamily.PERFORMANCE,
    )
    assert profile.device == "MacBook Air"


def test_extractor_populates_app_store_issue():
    """Verify ProblemExtractor populates primary_problem_family for App Store."""
    extractor = ProblemExtractor()
    profile = extractor.extract("@AppleSupport App Store not working for me either, can't download apps")
    assert profile.primary_problem_family == OperationalProblemFamily.SOFTWARE_APP
    assert any("App Store" in entity or "Store" in entity for entity in profile.operational_entities)


def test_extractor_backward_compatibility():
    """Verify CustomerProblemProfile retains all Phase 7-10 attributes."""
    extractor = ProblemExtractor()
    profile = extractor.extract("My battery is draining fast on iPhone 8 after updating to iOS 11")
    assert profile.device == "iPhone 8"
    assert "battery" in profile.primary_symptom
    assert profile.update_related is True
    assert profile.primary_problem_family == OperationalProblemFamily.POWER_BATTERY
    assert hasattr(profile, "information_sufficiency")
    assert hasattr(profile, "hardware_related")
    assert hasattr(profile, "software_related")
    assert hasattr(profile, "financial_related")


def test_operational_problem_profile_model():
    """Verify OperationalProblemProfile can be constructed and serialized."""
    prof = OperationalProblemProfile(
        device="iPhone X",
        primary_symptom="Face ID not working",
        primary_problem_family=OperationalProblemFamily.CAMERA_MEDIA,
        operational_entities=["iPhone X", "Face ID"],
        context_trigger="after screen replacement",
        family_confidence=0.88,
    )
    data = prof.to_dict()
    assert data["primary_problem_family"] == "CAMERA_MEDIA"
    assert "Face ID" in data["operational_entities"]
