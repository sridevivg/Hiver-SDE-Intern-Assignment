"""
Tests for ProblemExtractor (Phase 7 Problem Understanding Layer)
"""
import pytest

from backend.app.understanding.problem_extractor import (
    CustomerProblemProfile,
    ProblemExtractor,
)


@pytest.fixture
def extractor() -> ProblemExtractor:
    return ProblemExtractor()


def test_extract_battery_issue_with_update(extractor: ProblemExtractor) -> None:
    message = "My battery is draining completely within 2 hours since updating to iOS 11 on my iPhone 7."
    profile: CustomerProblemProfile = extractor.extract(message)

    assert "iPhone 7" in profile.device or profile.device == "iPhone"
    assert profile.product_or_service is not None and "iOS" in profile.product_or_service
    assert "battery" in profile.primary_symptom.lower() or "drain" in profile.primary_symptom.lower()
    assert profile.update_related is True
    assert profile.possible_cause is not None
    assert "software update" in profile.possible_cause.lower()
    assert profile.information_sufficiency == "sufficient"


def test_extract_audio_issue(extractor: ProblemExtractor) -> None:
    message = "I cannot hear sound from my iPad speakers when watching videos."
    profile = extractor.extract(message)

    assert profile.device == "iPad"
    assert "audio" in profile.primary_symptom.lower() or "speaker" in profile.primary_symptom.lower() or "sound" in profile.primary_symptom.lower()
    assert profile.update_related is False
    assert profile.information_sufficiency in ("sufficient", "partial")


def test_extract_multi_symptom(extractor: ProblemExtractor) -> None:
    message = "Updated my phone and now screen is black and audio doesn't work."
    profile = extractor.extract(message)

    assert profile.update_related is True
    assert len(profile.secondary_symptoms) > 0 or "and" in profile.primary_symptom


def test_extract_vague_message(extractor: ProblemExtractor) -> None:
    message = "help broken"
    profile = extractor.extract(message)

    assert profile.information_sufficiency in ("insufficient", "partial")
    assert profile.device is None


def test_extract_wifi_general_support(extractor: ProblemExtractor) -> None:
    message = "My iPhone will not connect to home Wi-Fi network."
    profile = extractor.extract(message)

    assert "iPhone" in profile.device
    assert "wi-fi" in profile.primary_symptom.lower() or "wifi" in profile.primary_symptom.lower()
