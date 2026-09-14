"""
SupportGraph AI — Unit Tests for Text Preprocessing (Phase 4)
"""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from app.nlp.text_preprocessing import is_usable_text, normalize_text
except ModuleNotFoundError:
    from backend.app.nlp.text_preprocessing import is_usable_text, normalize_text


def test_normalize_empty_and_none():
    assert normalize_text(None) == ""
    assert normalize_text("") == ""
    assert normalize_text("   \n\t  ") == ""


def test_normalize_url():
    text = "Check this page: https://support.apple.com/kb/HT201263 for help"
    normalized = normalize_text(text)
    assert "<URL>" in normalized
    assert "https://support.apple.com" not in normalized


def test_normalize_user_mentions():
    text = "@AppleSupport my iPhone 8 won't start after iOS 11 update @tim_cook"
    normalized = normalize_text(text)
    assert "<USER>" in normalized
    assert "@AppleSupport" not in normalized
    assert "@tim_cook" not in normalized
    assert "iPhone 8" in normalized
    assert "iOS 11" in normalized


def test_normalize_whitespace():
    text = "Why  is   my   battery\n\ndraining\tso fast?"
    normalized = normalize_text(text)
    assert normalized == "Why is my battery draining so fast?"


def test_preserves_product_names_and_punctuation():
    text = "My iPhone X display froze on error #4013! What can I do?"
    normalized = normalize_text(text)
    assert "iPhone X" in normalized
    assert "error #4013" in normalized
    assert "!" in normalized
    assert "?" in normalized


def test_is_usable_text():
    assert not is_usable_text(None)
    assert not is_usable_text("")
    assert not is_usable_text("  ")
    assert not is_usable_text("ab", min_length=3)
    assert is_usable_text("abc", min_length=3)
    assert is_usable_text("My iPhone won't turn on")
