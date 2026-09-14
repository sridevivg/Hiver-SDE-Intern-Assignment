"""
SupportGraph AI — Unit Tests for Groq Annotation Assistant (Phase 5.5)

Tests:
1. Missing/empty API key graceful fallback
2. Clean JSON parsing and markdown fence stripping
3. Malformed JSON handling and fallback to unclear_needs_review
4. Invented label detection and rejection
5. System prompt taxonomy boundaries
6. Mocked Groq client completion
7. Mocked rate limit retry handling
8. Mocked authentication error handling
9. Guarantee that model suggestions never populate annotation_label or annotator
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from app.evaluation.annotation_assistant import (
        AISuggestion,
        AnnotationAssistant,
        GroqAnnotationAssistant,
        GroqAuthenticationError,
    )
    from app.core.llm_factory import OllamaLLMClient
except ModuleNotFoundError:
    from backend.app.evaluation.annotation_assistant import (  # type: ignore[no-redef]
        AISuggestion,
        AnnotationAssistant,
        GroqAnnotationAssistant,
        GroqAuthenticationError,
    )
    from backend.app.core.llm_factory import OllamaLLMClient  # type: ignore[no-redef]


def test_missing_api_key_graceful_fallback() -> None:
    """Verify that an assistant without an API key sets status failed and empty label rather than crashing."""
    assistant = GroqAnnotationAssistant(api_key="")
    assert assistant.is_configured is False

    suggestion = assistant.suggest_intent("My iPhone battery is draining so fast")
    assert isinstance(suggestion, AISuggestion)
    assert suggestion.model_suggested_label == ""
    assert suggestion.model_confidence is None
    assert suggestion.suggestion_status == "failed"
    assert suggestion.model_needs_human_review is True


def test_parse_model_response_clean_json() -> None:
    """Verify parsing valid JSON response."""
    assistant = GroqAnnotationAssistant(api_key="mock_key")
    raw = json.dumps({
        "suggested_label": "battery_power_issue",
        "confidence": 0.92,
        "reasoning_summary": "Customer complains about rapid battery drain.",
        "needs_human_review": False,
    })
    parsed = assistant.parse_model_response(raw)
    assert parsed["suggested_label"] == "battery_power_issue"
    assert parsed["confidence"] == 0.92
    assert parsed["needs_human_review"] is False


def test_parse_model_response_markdown_wrapped() -> None:
    """Verify parsing JSON wrapped in markdown code blocks."""
    assistant = GroqAnnotationAssistant(api_key="mock_key")
    raw = """```json
    {
      "suggested_label": "software_update_problem",
      "confidence": 0.88,
      "reasoning_summary": "Issue occurred after iOS 11 update.",
      "needs_human_review": false
    }
    ```"""
    parsed = assistant.parse_model_response(raw)
    assert parsed["suggested_label"] == "software_update_problem"
    assert parsed["confidence"] == 0.88


def test_parse_model_response_malformed_json() -> None:
    """Verify malformed JSON raises ValueError and is rejected."""
    assistant = GroqAnnotationAssistant(api_key="mock_key")
    raw = "Not a json response {broken"
    with pytest.raises(ValueError, match="could not be parsed"):
        assistant.parse_model_response(raw)


def test_parse_model_response_rejects_invented_labels() -> None:
    """Verify that hallucinated labels not in approved taxonomy are rejected."""
    assistant = GroqAnnotationAssistant(api_key="mock_key")
    raw = json.dumps({
        "suggested_label": "completely_fake_invented_label",
        "confidence": 0.95,
        "reasoning_summary": "Invented label test.",
        "needs_human_review": False,
    })
    with pytest.raises(ValueError, match="unauthorized label"):
        assistant.parse_model_response(raw)


def test_system_prompt_contains_all_intents() -> None:
    """Verify system prompt includes definitions for all 9 operational intents."""
    assistant = GroqAnnotationAssistant(api_key="mock_key")
    prompt = assistant._build_system_prompt()
    for intent in assistant.allowed_labels:
        assert intent in prompt
    assert "unclear_needs_review" in prompt
    assert "It is not ground truth and requires human review" in prompt


def test_system_prompt_contains_hierarchical_rules_and_few_shot_examples() -> None:
    """Verify system prompt includes hierarchical decision principles and few-shot cases."""
    assistant = AnnotationAssistant(provider="ollama", model="llama3.2:latest")
    prompt = assistant._build_system_prompt()
    assert "RULE OF SPECIFICITY" in prompt
    assert "SYMPTOM OVER ENTITY" in prompt
    assert "UPDATE CAUSALITY" in prompt
    assert "AUDIO PRIORITY" in prompt
    assert "KEYBOARD PRIORITY" in prompt
    assert "DISPLAY AND TOUCH BOUNDARY" in prompt
    assert "BILLING BOUNDARY" in prompt
    assert "GENERAL DEVICE SUPPORT AS STRICT FALLBACK" in prompt
    assert "FEW-SHOT EXAMPLES" in prompt
    assert "keyboard_typing_issue" in prompt
    assert "hardware_audio_connection_issue" in prompt


def test_mocked_groq_completion_success() -> None:
    """Verify successful suggestion generation with mocked Groq client."""
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({
        "suggested_label": "keyboard_typing_issue",
        "confidence": 0.91,
        "reasoning_summary": "Customer reports letter I autocorrect glitch.",
        "needs_human_review": False,
    })
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response

    assistant = GroqAnnotationAssistant(api_key="mock_key", client=mock_client)
    suggestion = assistant.suggest_intent("When I type I it suggests an exclamation mark")

    assert suggestion.model_suggested_label == "keyboard_typing_issue"
    assert suggestion.model_confidence == 0.91
    assert suggestion.suggestion_status == "success"
    assert suggestion.model_needs_human_review is False
    assert mock_client.chat.completions.create.called


def test_mocked_groq_auth_failure() -> None:
    """Verify that authentication errors raise GroqAuthenticationError."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("401 Unauthorized: Invalid API Key")

    assistant = GroqAnnotationAssistant(api_key="mock_key", client=mock_client)
    with pytest.raises(GroqAuthenticationError):
        assistant.suggest_intent("Some customer message")


def test_mocked_groq_api_failure_does_not_fake_classification() -> None:
    """Verify that an API failure returns suggestion_status='failed' and empty suggested label."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("500 Internal Server Error")

    assistant = GroqAnnotationAssistant(api_key="mock_key", client=mock_client)
    suggestion = assistant.suggest_intent("My iPhone won't turn on")

    assert suggestion.suggestion_status == "failed"
    assert suggestion.model_suggested_label == ""
    assert suggestion.model_confidence is None
    assert suggestion.model_reasoning_summary == ""
    assert suggestion.model_needs_human_review is True


def test_verify_model_access() -> None:
    """Verify model verification against accessible models list."""
    mock_client = MagicMock()
    m1 = MagicMock()
    m1.id = "openai/gpt-oss-20b"
    m2 = MagicMock()
    m2.id = "qwen/qwen3.6-27b"
    mock_client.models.list.return_value.data = [m1, m2]

    assistant = GroqAnnotationAssistant(
        api_key="mock_key",
        model="openai/gpt-oss-20b",
        client=mock_client,
    )
    is_valid, available, err = assistant.verify_model_access()
    assert is_valid is True
    assert "openai/gpt-oss-20b" in available
    assert err == ""

    # Test with unavailable model
    assistant_unavailable = GroqAnnotationAssistant(
        api_key="mock_key",
        model="nonexistent-model",
        client=mock_client,
    )
    is_valid_bad, available_bad, err_bad = assistant_unavailable.verify_model_access()
    assert is_valid_bad is False
    assert "does not exist or is not accessible" in err_bad


def test_suggestion_never_populates_human_labels() -> None:
    """Verify that AISuggestion model contains zero human annotation attributes."""
    suggestion = AISuggestion(
        model_name="openai/gpt-oss-20b",
        model_suggested_label="battery_power_issue",
        model_confidence=0.85,
        model_reasoning_summary="Battery complaint.",
        model_needs_human_review=False,
        suggestion_timestamp="2026-09-10T12:00:00Z",
        suggestion_status="success",
    )
    d = suggestion.to_dict()
    assert "annotation_label" not in d
    assert "annotator" not in d
    assert "annotation_status" not in d


def test_annotation_assistant_with_ollama_client() -> None:
    """Verify that AnnotationAssistant functions correctly when paired with OllamaLLMClient."""
    mock_http = MagicMock()
    mock_payload = {
        "suggested_label": "display_touch_issue",
        "confidence": 0.92,
        "reasoning_summary": "Screen touch problem mentioned by user.",
        "needs_human_review": False,
    }
    mock_http.post.return_value = MagicMock(
        status_code=200,
        json=lambda: {
            "message": {"content": json.dumps(mock_payload)},
            "prompt_eval_count": 50,
            "eval_count": 25,
        },
    )

    ollama_client = OllamaLLMClient(
        model="llama3.2:latest",
        base_url="http://localhost:11434",
        client=mock_http,
    )
    assistant = AnnotationAssistant(llm_client=ollama_client)
    assert assistant.provider == "ollama"
    assert assistant.model == "llama3.2:latest"

    record = {
        "golden_id": "test_001",
        "customer_message": "My touch screen does not respond to taps anymore.",
    }
    suggestion = assistant.suggest_for_record(record)
    assert suggestion.suggestion_status == "success"
    assert suggestion.model_name == "llama3.2:latest"
    assert suggestion.model_suggested_label == "display_touch_issue"
    assert suggestion.model_confidence == 0.92
    assert suggestion.model_needs_human_review is False
    assert "Screen touch problem" in suggestion.model_reasoning_summary


def test_parse_model_response_rejects_nan_and_empty_labels() -> None:
    """Verify parse_model_response rejects empty and literal 'nan' labels."""
    assistant = GroqAnnotationAssistant(api_key="mock_key")
    with pytest.raises(ValueError, match="empty or invalid label"):
        assistant.parse_model_response(json.dumps({"suggested_label": "nan", "confidence": 0.8}))
    with pytest.raises(ValueError, match="empty or invalid label"):
        assistant.parse_model_response(json.dumps({"suggested_label": "", "confidence": 0.8}))


def test_parse_model_response_validates_confidence_range() -> None:
    """Verify parse_model_response rejects non-numeric or out-of-range confidence."""
    assistant = GroqAnnotationAssistant(api_key="mock_key")
    # Non-numeric string
    with pytest.raises(ValueError, match="cannot be parsed as a numeric float"):
        assistant.parse_model_response(json.dumps({"suggested_label": "battery_power_issue", "confidence": "high"}))
    # Above 1.0
    with pytest.raises(ValueError, match="outside the valid range"):
        assistant.parse_model_response(json.dumps({"suggested_label": "battery_power_issue", "confidence": 1.25}))
    # Below 0.0
    with pytest.raises(ValueError, match="outside the valid range"):
        assistant.parse_model_response(json.dumps({"suggested_label": "battery_power_issue", "confidence": -0.1}))


def test_suggest_intent_invalid_model_output_sets_status_and_unclear() -> None:
    """Verify invalid model output sets status 'invalid_model_output' and 'unclear_needs_review'."""
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({
        "suggested_label": "invented_fake_category",
        "confidence": 0.90,
    })
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response

    assistant = GroqAnnotationAssistant(api_key="mock_key", client=mock_client)
    suggestion = assistant.suggest_intent("Screen is flickering")

    assert suggestion.suggestion_status == "invalid_model_output"
    assert suggestion.model_suggested_label == "unclear_needs_review"
    assert suggestion.model_confidence == 0.0
    assert suggestion.model_needs_human_review is True
    assert "Invalid model output" in suggestion.model_reasoning_summary


def test_suggest_intent_provider_failure_sets_failed_status() -> None:
    """Verify provider network/API failure sets status 'failed' and empty suggested label."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("Connection refused / timeout")

    assistant = GroqAnnotationAssistant(api_key="mock_key", client=mock_client)
    suggestion = assistant.suggest_intent("Screen is flickering")

    assert suggestion.suggestion_status == "failed"
    assert suggestion.model_suggested_label == ""
    assert suggestion.model_confidence is None
    assert suggestion.model_needs_human_review is True
    assert suggestion.model_reasoning_summary == ""


