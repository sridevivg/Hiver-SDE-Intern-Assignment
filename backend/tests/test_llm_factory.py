import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import httpx
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from app.core.config import settings
    from app.core.llm_factory import (
        BaseLLMClient,
        GroqLLMClient,
        OllamaLLMClient,
        LLMCompletionResponse,
        LLMConfigurationError,
        LLMAuthenticationError,
        LLMConnectionError,
        LLMResponseError,
        LLMError,
        get_llm_client,
    )
except ModuleNotFoundError:
    from backend.app.core.config import settings  # type: ignore[no-redef]
    from backend.app.core.llm_factory import (  # type: ignore[no-redef]
        BaseLLMClient,
        GroqLLMClient,
        OllamaLLMClient,
        LLMCompletionResponse,
        LLMConfigurationError,
        LLMAuthenticationError,
        LLMConnectionError,
        LLMResponseError,
        LLMError,
        get_llm_client,
    )


# ---------------------------------------------------------------------------
# Factory Resolution Tests
# ---------------------------------------------------------------------------
def test_factory_resolves_groq() -> None:
    client = get_llm_client(provider="groq", api_key="mock_key")
    assert isinstance(client, GroqLLMClient)
    assert client.provider == "groq"
    assert client.model == settings.groq_model


def test_factory_resolves_ollama() -> None:
    client = get_llm_client(provider="ollama", base_url="http://localhost:11434")
    assert isinstance(client, OllamaLLMClient)
    assert client.provider == "ollama"
    assert client.model == settings.ollama_model


def test_factory_resolves_default_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "llm_provider", "ollama")
    client = get_llm_client()
    assert isinstance(client, OllamaLLMClient)
    assert client.provider == "ollama"


def test_factory_rejects_unsupported_provider() -> None:
    with pytest.raises(LLMConfigurationError) as excinfo:
        get_llm_client(provider="anthropic_unsupported")
    assert "Unsupported LLM provider" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Groq Client Tests
# ---------------------------------------------------------------------------
def test_groq_client_sanitizes_api_key() -> None:
    client = GroqLLMClient(api_key="gsk_supersecret12345")
    sanitized = client._sanitize("Error with gsk_supersecret12345 occurred")
    assert "gsk_supersecret12345" not in sanitized
    assert "[REDACTED]" in sanitized


def test_groq_verify_access_missing_key() -> None:
    client = GroqLLMClient(api_key="", client=None)
    valid, models, err = client.verify_access()
    assert valid is False
    assert "missing or empty" in err


def test_groq_verify_access_success() -> None:
    mock_model = MagicMock()
    mock_model.id = "openai/gpt-oss-20b"
    mock_groq = MagicMock()
    mock_groq.models.list.return_value = MagicMock(data=[mock_model])

    client = GroqLLMClient(
        model="openai/gpt-oss-20b",
        api_key="mock_key",
        client=mock_groq,
    )
    valid, models, err = client.verify_access()
    assert valid is True
    assert "openai/gpt-oss-20b" in models
    assert err == ""


def test_groq_verify_access_model_not_found() -> None:
    mock_model = MagicMock()
    mock_model.id = "other-model"
    mock_groq = MagicMock()
    mock_groq.models.list.return_value = MagicMock(data=[mock_model])

    client = GroqLLMClient(
        model="nonexistent-model",
        api_key="mock_key",
        client=mock_groq,
    )
    valid, models, err = client.verify_access()
    assert valid is False
    assert "does not exist or is not accessible" in err


def test_groq_completion_success() -> None:
    mock_groq = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '{"label": "battery_power_issue"}'
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage.prompt_tokens = 45
    mock_response.usage.completion_tokens = 12
    mock_groq.chat.completions.create.return_value = mock_response

    client = GroqLLMClient(
        model="openai/gpt-oss-20b",
        api_key="mock_key",
        client=mock_groq,
    )
    resp = client.chat_completion(
        messages=[{"role": "user", "content": "hello"}],
    )
    assert resp.content == '{"label": "battery_power_issue"}'
    assert resp.model == "openai/gpt-oss-20b"
    assert resp.model_name == "openai/gpt-oss-20b"
    assert resp.provider == "groq"
    assert resp.prompt_tokens == 45
    assert resp.completion_tokens == 12


def test_groq_completion_auth_failure() -> None:
    mock_groq = MagicMock()
    mock_groq.chat.completions.create.side_effect = Exception("401 Unauthorized: Invalid API Key")

    client = GroqLLMClient(
        model="openai/gpt-oss-20b",
        api_key="mock_key",
        client=mock_groq,
    )
    with pytest.raises(LLMAuthenticationError):
        client.chat_completion(messages=[{"role": "user", "content": "hello"}])


# ---------------------------------------------------------------------------
# Ollama Client Tests
# ---------------------------------------------------------------------------
def test_ollama_list_models_and_verify() -> None:
    mock_http = MagicMock()
    mock_http.get.return_value = MagicMock(
        status_code=200,
        json=lambda: {"models": [{"name": "llama3.2:latest"}]},
    )

    client = OllamaLLMClient(
        model="llama3.2:latest",
        base_url="http://localhost:11434",
        client=mock_http,
    )
    models = client.list_models()
    assert models == ["llama3.2:latest"]

    valid, available, err = client.verify_access()
    assert valid is True
    assert "llama3.2:latest" in available
    assert err == ""


def test_ollama_verify_access_tag_matching() -> None:
    # Specifying 'llama3.2' should match 'llama3.2:latest'
    mock_http = MagicMock()
    mock_http.get.return_value = MagicMock(
        status_code=200,
        json=lambda: {"models": [{"name": "llama3.2:latest"}]},
    )
    client = OllamaLLMClient(
        model="llama3.2",
        base_url="http://localhost:11434",
        client=mock_http,
    )
    valid, available, err = client.verify_access()
    assert valid is True
    assert err == ""


def test_ollama_verify_access_missing_model() -> None:
    mock_http = MagicMock()
    mock_http.get.return_value = MagicMock(
        status_code=200,
        json=lambda: {"models": [{"name": "mistral:latest"}]},
    )
    client = OllamaLLMClient(
        model="llama3.2:latest",
        base_url="http://localhost:11434",
        client=mock_http,
    )
    valid, available, err = client.verify_access()
    assert valid is False
    assert "is not installed locally" in err


def test_ollama_verify_access_connection_failure() -> None:
    mock_http = MagicMock()
    mock_http.get.side_effect = httpx.ConnectError("Connection refused")

    client = OllamaLLMClient(
        model="llama3.2:latest",
        base_url="http://localhost:11434",
        client=mock_http,
    )
    valid, available, err = client.verify_access()
    assert valid is False
    assert "Could not connect to Ollama" in err


def test_ollama_completion_success() -> None:
    mock_http = MagicMock()
    mock_http.post.return_value = MagicMock(
        status_code=200,
        json=lambda: {
            "message": {"content": '{"label": "display_touch_issue"}'},
            "prompt_eval_count": 30,
            "eval_count": 15,
        },
    )

    client = OllamaLLMClient(
        model="llama3.2:latest",
        base_url="http://localhost:11434",
        client=mock_http,
    )
    resp = client.chat_completion(
        messages=[{"role": "user", "content": "Screen unresponsive"}],
        response_format={"type": "json_object"},
    )
    assert resp.content == '{"label": "display_touch_issue"}'
    assert resp.model == "llama3.2:latest"
    assert resp.provider == "ollama"
    assert resp.prompt_tokens == 30
    assert resp.completion_tokens == 15


def test_ollama_completion_connect_error() -> None:
    mock_http = MagicMock()
    mock_http.post.side_effect = httpx.ConnectError("Connection refused")

    client = OllamaLLMClient(
        model="llama3.2:latest",
        base_url="http://localhost:11434",
        client=mock_http,
    )
    with pytest.raises(LLMConnectionError) as excinfo:
        client.chat_completion(messages=[{"role": "user", "content": "hi"}])
    assert "Failed to connect to Ollama daemon" in str(excinfo.value)


def test_ollama_completion_http_error() -> None:
    mock_http = MagicMock()
    mock_resp = MagicMock(status_code=500, text="Internal server error")
    mock_http.post.return_value = mock_resp

    client = OllamaLLMClient(
        model="llama3.2:latest",
        base_url="http://localhost:11434",
        client=mock_http,
    )
    with pytest.raises(LLMError) as excinfo:
        client.chat_completion(messages=[{"role": "user", "content": "hi"}])
    assert "Ollama returned HTTP 500" in str(excinfo.value)
