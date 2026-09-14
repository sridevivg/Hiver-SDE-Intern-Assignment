"""
SupportGraph AI — Text Preprocessing for Intent Discovery

Conservative, semantic-preserving text normalization for customer-support messages:
- Normalizes URLs and user mentions to standard tokens (<URL>, <USER>)
- Normalizes irregular whitespace
- Preserves casing, product names, version numbers, error codes, and meaningful punctuation
- Does NOT apply aggressive stemming, lemmatization, or blind stopword stripping
"""
from __future__ import annotations

import re
from typing import Optional


# Regular expressions for URL and Twitter user mention patterns
URL_REGEX = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
MENTION_REGEX = re.compile(r"@[A-Za-z0-9_]+", flags=re.IGNORECASE)
WHITESPACE_REGEX = re.compile(r"\s+")


def normalize_text(text: Optional[str]) -> str:
    """
    Apply conservative normalization to a customer message.

    Args:
        text: Raw customer message text.

    Returns:
        Normalized text string. Empty string if input is None or whitespace.
    """
    if text is None:
        return ""

    s = str(text)

    # 1. Replace URLs with <URL> token
    s = URL_REGEX.sub("<URL>", s)

    # 2. Replace user mentions with <USER> token
    s = MENTION_REGEX.sub("<USER>", s)

    # 3. Collapse multiple spaces, newlines, and tabs into a single space
    s = WHITESPACE_REGEX.sub(" ", s).strip()

    return s


def is_usable_text(text: Optional[str], min_length: int = 3) -> bool:
    """
    Check if a message contains usable text for intent discovery.

    Args:
        text: Raw or normalized text.
        min_length: Minimum character length required (default: 3).

    Returns:
        True if text has at least `min_length` non-whitespace characters.
    """
    if not text:
        return False
    cleaned = WHITESPACE_REGEX.sub(" ", str(text)).strip()
    return len(cleaned) >= min_length
