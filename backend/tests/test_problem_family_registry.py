"""
SupportGraph AI — Tests: Problem Family Registry (Phase 10.2)

Tests:
- All 20 operational problem families are defined with valid metadata
- Pattern matching works accurately across standard problem types
- Conflict detection between incompatible families
- Compatibility detection between related families
"""
from __future__ import annotations

import pytest

from backend.app.retrieval.problem_family_registry import (
    PROBLEM_FAMILY_DEFINITIONS,
    OperationalProblemFamily,
    ProblemFamilyDetector,
)


def test_all_20_families_defined():
    """Verify all 20 required operational problem families exist and have definitions."""
    assert len(OperationalProblemFamily) == 20
    for family in OperationalProblemFamily:
        assert family in PROBLEM_FAMILY_DEFINITIONS
        defn = PROBLEM_FAMILY_DEFINITIONS[family]
        assert defn.display_name
        assert defn.description
        assert isinstance(defn.keywords, list)


def test_detect_wifi_family():
    """Verify WiFi inquiries map to CONNECTIVITY_WIFI."""
    detector = ProblemFamilyDetector()
    family, secondaries, conf = detector.detect_family(
        message="why does WiFi turn on by itself when I turn it off manually",
        primary_symptom="wireless Wi-Fi or network connectivity failure",
    )
    assert family == OperationalProblemFamily.CONNECTIVITY_WIFI
    assert conf >= 0.60


def test_detect_bluetooth_family():
    """Verify Bluetooth issues map to CONNECTIVITY_BLUETOOTH."""
    detector = ProblemFamilyDetector()
    family, secondaries, conf = detector.detect_family(
        message="My car audio keeps dropping Bluetooth connection to my iPhone",
    )
    assert family == OperationalProblemFamily.CONNECTIVITY_BLUETOOTH


def test_detect_software_app_family():
    """Verify App Store / app crash issues map to SOFTWARE_APP."""
    detector = ProblemFamilyDetector()
    family, secondaries, conf = detector.detect_family(
        message="The App Store is not downloading any updates or apps",
    )
    assert family == OperationalProblemFamily.SOFTWARE_APP


def test_detect_crash_freeze_family():
    """Verify beach ball and frozen Mac map to CRASH_FREEZE."""
    detector = ProblemFamilyDetector()
    family, secondaries, conf = detector.detect_family(
        message="My MacBook Air has the dreaded spinning beach ball and is frozen",
    )
    assert family == OperationalProblemFamily.CRASH_FREEZE


def test_detect_sync_backup_family():
    """Verify photo sync and iCloud issues map to SYNC_BACKUP."""
    detector = ProblemFamilyDetector()
    family, secondaries, conf = detector.detect_family(
        message="photos not syncing properly between OSX high Sierra and iOS iCloud",
    )
    assert family == OperationalProblemFamily.SYNC_BACKUP


def test_conflict_detection():
    """Verify mutually exclusive families are flagged as conflicting."""
    detector = ProblemFamilyDetector()
    assert detector.are_conflicting(
        OperationalProblemFamily.ACCOUNT_ACCESS,
        OperationalProblemFamily.AUDIO,
    )
    assert detector.are_conflicting(
        OperationalProblemFamily.BILLING_PAYMENT,
        OperationalProblemFamily.DISPLAY,
    )
    # Compatible / related families should NOT conflict
    assert not detector.are_conflicting(
        OperationalProblemFamily.CONNECTIVITY_WIFI,
        OperationalProblemFamily.NETWORK_CELLULAR,
    )
    assert not detector.are_conflicting(
        OperationalProblemFamily.POWER_BATTERY,
        OperationalProblemFamily.CHARGING,
    )


def test_compatibility_detection():
    """Verify related problem families are identified as compatible."""
    detector = ProblemFamilyDetector()
    assert detector.are_compatible(
        OperationalProblemFamily.CONNECTIVITY_WIFI,
        OperationalProblemFamily.NETWORK_CELLULAR,
    )
    assert detector.are_compatible(
        OperationalProblemFamily.CRASH_FREEZE,
        OperationalProblemFamily.PERFORMANCE,
    )
    assert detector.are_compatible(
        OperationalProblemFamily.POWER_BATTERY,
        OperationalProblemFamily.POWER_BATTERY,
    )
