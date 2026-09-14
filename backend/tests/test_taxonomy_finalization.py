"""
SupportGraph AI — Unit Tests for Candidate Taxonomy Finalization (Phase 5)

Tests:
1. Candidate taxonomy construction from empirical audit evidence
2. Validation of operational intent count (7–10 bounds)
3. Completeness of definitions, inclusion/exclusion rules, and examples
4. Detection of duplicate IDs or names and snake_case enforcement
5. JSON serialization and deserialization roundtrip
6. Rule-based candidate intent assignment heuristic
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from app.nlp.taxonomy_finalization import (
        IntentDefinition,
        TaxonomyCandidate,
        assign_candidate_intent,
        build_candidate_taxonomy,
        load_candidate_taxonomy,
        save_candidate_taxonomy,
        validate_taxonomy_consistency,
    )
except ModuleNotFoundError:
    from backend.app.nlp.taxonomy_finalization import (  # type: ignore[no-redef]
        IntentDefinition,
        TaxonomyCandidate,
        assign_candidate_intent,
        build_candidate_taxonomy,
        load_candidate_taxonomy,
        save_candidate_taxonomy,
        validate_taxonomy_consistency,
    )


def test_build_candidate_taxonomy() -> None:
    """Verify default taxonomy builds successfully with 9 operational intents."""
    taxonomy = build_candidate_taxonomy()
    assert taxonomy.status == "candidate_human_review_required"
    assert taxonomy.selected_brand == "AppleSupport"
    assert len(taxonomy.intents) == 9
    assert taxonomy.intent_count == 9
    assert 7 <= taxonomy.intent_count <= 10


def test_intent_definition_completeness() -> None:
    """Verify each intent definition adheres to the strict operational schema."""
    taxonomy = build_candidate_taxonomy()
    intent_names = taxonomy.intent_names

    # Check key separated categories
    assert "account_access_issue" in intent_names
    assert "billing_purchase_issue" in intent_names
    assert "software_update_problem" in intent_names
    assert "battery_power_issue" in intent_names
    assert "display_touch_issue" in intent_names
    assert "hardware_audio_connection_issue" in intent_names
    assert "keyboard_typing_issue" in intent_names
    assert "mac_software_issue" in intent_names
    assert "general_device_support" in intent_names

    for intent in taxonomy.intents:
        assert intent.intent_id.startswith("intent_")
        assert len(intent.definition.strip()) > 20
        assert len(intent.include_when) >= 2
        assert len(intent.exclude_when) >= 2
        assert len(intent.example_messages) >= 2
        assert len(intent.source_clusters) >= 1
        assert intent.review_status == "candidate_human_review_required"


def test_taxonomy_consistency_validation_success() -> None:
    """Verify that a valid taxonomy candidate passes consistency checks."""
    taxonomy = build_candidate_taxonomy()
    errors = validate_taxonomy_consistency(taxonomy)
    assert errors == []


def test_taxonomy_consistency_validation_catches_errors() -> None:
    """Verify that malformed taxonomies trigger validation errors."""
    invalid_taxonomy = TaxonomyCandidate(
        status="candidate_human_review_required",
        taxonomy_version="test_v1",
        selected_brand="AppleSupport",
        intent_count=2,  # < 7 intents
        intents=[
            IntentDefinition(
                intent_id="id_1",
                intent_name="Invalid CamelCase",
                definition="",
                include_when=[],
                exclude_when=[],
                example_messages=[],
                source_clusters=[],
            ),
            IntentDefinition(
                intent_id="id_1",  # duplicate id
                intent_name="valid_name_here",
                definition="some def",
                include_when=["inc"],
                exclude_when=["exc"],
                example_messages=["ex1", "ex2"],
                source_clusters=[1],
            ),
        ],
    )

    errors = validate_taxonomy_consistency(invalid_taxonomy)
    assert len(errors) >= 3
    error_text = " ".join(errors)
    assert "outside target range 7–10" in error_text
    assert "Duplicate intent_id" in error_text
    assert "snake_case" in error_text


def test_taxonomy_serialization_roundtrip(tmp_path: Path) -> None:
    """Verify that candidate taxonomy can be serialized and reloaded without loss."""
    taxonomy = build_candidate_taxonomy()
    json_path = tmp_path / "test_taxonomy.json"

    saved_path = save_candidate_taxonomy(taxonomy, json_path)
    assert saved_path.exists()

    loaded = load_candidate_taxonomy(json_path)
    assert loaded.status == taxonomy.status
    assert loaded.intent_count == taxonomy.intent_count
    assert loaded.intent_names == taxonomy.intent_names
    assert len(loaded.merge_decisions) == len(taxonomy.merge_decisions)
    assert len(loaded.split_decisions) == len(taxonomy.split_decisions)


def test_assign_candidate_intent() -> None:
    """Verify regex-assisted candidate intent mapping."""
    # Battery
    assert assign_candidate_intent("my battery drains in 2 hours", 1) == "battery_power_issue"
    # Keyboard
    assert assign_candidate_intent("letter i typing an exclamation mark bug", 0) == "keyboard_typing_issue"
    # Software update
    assert assign_candidate_intent("phone stuck on apple logo since downloading update", 3) == "software_update_problem"
    # Account
    assert assign_candidate_intent("cannot sign in to my apple id account", 4) == "account_access_issue"
    # Billing
    assert assign_candidate_intent("refund for unauthorized subscription charge on itunes", 4) == "billing_purchase_issue"
    # Mac software
    assert assign_candidate_intent("macbook high sierra safari crashing constantly", 5) == "mac_software_issue"
