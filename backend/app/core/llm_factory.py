"""
SupportGraph AI — Unified LLM Factory (Groq & Ollama)

Provides a unified interface and factory pattern for LLM generation:
- Cloud Groq API client (fast remote inference via Groq SDK)
- Local Ollama client (offline privacy-first inference via local Ollama daemon)
- Switchable seamlessly via LLM_PROVIDER in .env ("groq" or "ollama")
- Comprehensive pre-flight health checks and model access verification
"""
from __future__ import annotations

import logging
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx

try:
    from app.core.config import settings
    from app.core.logging import get_logger
except ModuleNotFoundError:
    from backend.app.core.config import settings  # type: ignore[no-redef]
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class LLMError(Exception):
    """Base exception for all LLM errors."""


class LLMConfigurationError(LLMError):
    """Raised when configuration for the selected LLM provider is invalid or missing."""


class LLMAuthenticationError(LLMError):
    """Raised when authentication fails (e.g. invalid API key)."""


class LLMRateLimitError(LLMError):
    """Raised when provider rate limits are encountered and exhausted."""


class LLMConnectionError(LLMError):
    """Raised when connection to LLM host or server fails."""


class LLMResponseError(LLMError):
    """Raised when LLM response is malformed or cannot be parsed."""


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------
@dataclass
class LLMCompletionResponse:
    """Standardized response container across all LLM providers."""
    content: str
    model: str
    provider: str
    raw_response: Optional[dict[str, Any]] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None

    @property
    def model_name(self) -> str:
        """Alias for model identifier."""
        return self.model



# ---------------------------------------------------------------------------
# Abstract Base Client
# ---------------------------------------------------------------------------
class BaseLLMClient(ABC):
    """Abstract base class for all LLM provider client implementations."""

    @property
    @abstractmethod
    def provider(self) -> str:
        """Name of the LLM provider (e.g. 'groq' or 'ollama')."""

    @property
    @abstractmethod
    def model(self) -> str:
        """Active model identifier."""

    @abstractmethod
    def list_models(self) -> list[str]:
        """Return list of model IDs available from this provider."""

    @abstractmethod
    def verify_access(self) -> tuple[bool, list[str], str]:
        """
        Verify connectivity and check whether configured model is accessible.

        Returns:
            (is_valid, available_models, error_message)
        """

    @abstractmethod
    def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        response_format: Optional[dict[str, str]] = None,
        max_tokens: int = 1024,
    ) -> LLMCompletionResponse:
        """
        Execute a chat completion request.

        Args:
            messages: List of message dictionaries with 'role' and 'content'.
            temperature: Sampling temperature (default 0.0 for deterministic).
            response_format: Optional format dict, e.g. {"type": "json_object"}.
            max_tokens: Maximum completion tokens to generate.

        Returns:
            LLMCompletionResponse object with string content and metadata.
        """


# ---------------------------------------------------------------------------
# Groq Provider Client
# ---------------------------------------------------------------------------
class GroqLLMClient(BaseLLMClient):
    """LLM client wrapping the official Groq SDK."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        client: Optional[Any] = None,
    ) -> None:
        self._provider = "groq"
        self._timeout = timeout

        if api_key is not None:
            self._api_key = api_key.strip()
        else:
            self._api_key = (
                os.getenv("GROQ_API_KEY", "")
                or getattr(settings, "groq_api_key", "")
            ).strip()

        if model is not None and model.strip():
            self._model = model.strip()
        else:
            self._model = (
                os.getenv("GROQ_MODEL", "")
                or getattr(settings, "groq_model", "")
                or "openai/gpt-oss-20b"
            ).strip()

        self._client = client
        if self._client is None and self._api_key:
            try:
                from groq import Groq
                self._client = Groq(api_key=self._api_key, timeout=self._timeout)
            except ImportError:
                logger.warning("The 'groq' python package is not installed.")
                self._client = None
            except Exception as exc:
                logger.warning("Failed to initialize Groq client: %s", exc)
                self._client = None

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model(self) -> str:
        return self._model

    @property
    def api_key(self) -> str:
        return self._api_key

    def _sanitize(self, msg: str) -> str:
        """Redact API key from log/error messages."""
        if self._api_key and self._api_key in msg:
            return msg.replace(self._api_key, "[REDACTED]")
        return msg

    def list_models(self) -> list[str]:
        if not self._api_key or self._client is None:
            return []
        try:
            resp = self._client.models.list()
            data = getattr(resp, "data", [])
            model_ids = [m.id for m in data if hasattr(m, "id")]
            return sorted(model_ids)
        except Exception as exc:
            logger.warning("Failed to list Groq models: %s", self._sanitize(str(exc)))
            return []

    def verify_access(self) -> tuple[bool, list[str], str]:
        if not self._api_key:
            return False, [], "GROQ_API_KEY is missing or empty in environment/settings."
        if self._client is None:
            return False, [], "Groq client is not initialized (verify package 'groq' is installed)."

        accessible = self.list_models()
        if not accessible:
            return False, [], "Failed to retrieve accessible models from Groq API or account has no available models."

        if self._model not in accessible:
            return (
                False,
                accessible,
                f"Configured Groq model '{self._model}' does not exist or is not accessible to this API key. Available: {accessible}",
            )
        return True, accessible, ""

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        response_format: Optional[dict[str, str]] = None,
        max_tokens: int = 1024,
    ) -> LLMCompletionResponse:
        if not self._api_key or self._client is None:
            raise LLMConfigurationError("GROQ_API_KEY is not configured or Groq client is uninitialized.")

        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                kwargs: dict[str, Any] = {
                    "model": self._model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if response_format:
                    kwargs["response_format"] = response_format

                try:
                    response = self._client.chat.completions.create(**kwargs)
                except Exception as create_exc:
                    # Retry without response_format if provider returns json_validate_failed
                    if "json_validate_failed" in str(create_exc).lower() and response_format:
                        kwargs.pop("response_format", None)
                        response = self._client.chat.completions.create(**kwargs)
                    else:
                        raise

                content = response.choices[0].message.content or ""
                usage = getattr(response, "usage", None)
                p_tokens = getattr(usage, "prompt_tokens", None) if usage else None
                c_tokens = getattr(usage, "completion_tokens", None) if usage else None
                return LLMCompletionResponse(
                    content=content,
                    model=self._model,
                    provider=self._provider,
                    raw_response={"choices": [{"message": {"content": content}}]},
                    prompt_tokens=p_tokens,
                    completion_tokens=c_tokens,
                )

            except Exception as exc:
                exc_str = self._sanitize(str(exc))
                is_auth = "auth" in exc_str.lower() or "401" in exc_str or "unauthorized" in exc_str.lower()
                if is_auth:
                    raise LLMAuthenticationError(f"Groq API authentication failed: {exc_str}")

                is_rate_limit = "rate_limit" in exc_str.lower() or "429" in exc_str
                is_tpd = "tokens per day" in exc_str.lower() or "tpd" in exc_str.lower()
                if is_rate_limit:
                    if is_tpd:
                        raise LLMRateLimitError(f"Groq daily token quota exceeded: {exc_str}")
                    if attempt < max_retries:
                        wait_time = 2.0 * (attempt + 1)
                        logger.warning("Groq rate limit encountered. Backing off for %.1fs...", wait_time)
                        time.sleep(wait_time)
                        continue

                if attempt == max_retries:
                    raise LLMError(f"Groq completion failed after retries: {exc_str}")

        raise LLMError("Groq completion loop exited unexpectedly.")


# ---------------------------------------------------------------------------
# Ollama Provider Client
# ---------------------------------------------------------------------------
class OllamaLLMClient(BaseLLMClient):
    """LLM client communicating directly with the local Ollama daemon."""

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        http_client: Optional[httpx.Client] = None,
        client: Optional[Any] = None,
    ) -> None:
        self._provider = "ollama"
        self._timeout = timeout

        resolved_url = (
            base_url
            or os.getenv("OLLAMA_BASE_URL", "")
            or getattr(settings, "ollama_base_url", "")
            or "http://localhost:11434"
        ).strip().rstrip("/")
        self._base_url = resolved_url

        if model is not None and model.strip():
            self._model = model.strip()
        else:
            self._model = (
                os.getenv("OLLAMA_MODEL", "")
                or getattr(settings, "ollama_model", "")
                or "llama3.2:latest"
            ).strip()

        self._client = client or http_client or httpx.Client(base_url=self._base_url, timeout=self._timeout)

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model(self) -> str:
        return self._model

    @property
    def base_url(self) -> str:
        return self._base_url

    def list_models(self) -> list[str]:
        try:
            resp = self._client.get("/api/tags")
            if resp.status_code != 200:
                logger.warning("Ollama /api/tags returned HTTP %d: %s", resp.status_code, resp.text)
                return []
            data = resp.json()
            models_list = data.get("models", [])
            model_names = [m.get("name") or m.get("model", "") for m in models_list if m.get("name") or m.get("model")]
            return sorted(set(model_names))
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            logger.warning("Failed to connect to Ollama at %s: %s", self._base_url, exc)
            return []
        except Exception as exc:
            logger.warning("Error fetching Ollama models: %s", exc)
            return []

    def verify_access(self) -> tuple[bool, list[str], str]:
        try:
            available = self.list_models()
        except Exception as exc:
            return False, [], f"Could not connect to Ollama server at {self._base_url}: {exc}"

        if not available:
            # Check if server itself responded with 0 models or if connection was refused
            try:
                ping = self._client.get("/")
                if ping.status_code == 200:
                    return False, [], f"Ollama is running at {self._base_url}, but has no models installed. Run 'ollama pull {self._model}'."
            except Exception:
                pass
            return (
                False,
                [],
                f"Could not connect to Ollama at {self._base_url}. Ensure the Ollama daemon is running ('ollama serve').",
            )

        # Allow matching with or without tag suffix (e.g. 'llama3.2' matches 'llama3.2:latest')
        target = self._model.lower()
        matched = target in [m.lower() for m in available] or any(
            m.lower().split(":")[0] == target.split(":")[0] for m in available
        )

        if not matched:
            return (
                False,
                available,
                f"Configured Ollama model '{self._model}' is not installed locally. Available: {available}. Run 'ollama pull {self._model}'.",
            )

        return True, available, ""

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        response_format: Optional[dict[str, str]] = None,
        max_tokens: int = 1024,
    ) -> LLMCompletionResponse:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        if response_format and response_format.get("type") == "json_object":
            payload["format"] = "json"

        try:
            resp = self._client.post("/api/chat", json=payload)
        except (httpx.ConnectError, httpx.ConnectTimeout) as conn_exc:
            raise LLMConnectionError(
                f"Failed to connect to Ollama daemon at {self._base_url}. Is Ollama running? ({conn_exc})"
            )
        except Exception as req_exc:
            raise LLMError(f"Ollama request error: {req_exc}")

        if resp.status_code != 200:
            raise LLMError(f"Ollama returned HTTP {resp.status_code}: {resp.text}")

        try:
            data = resp.json()
            content = data.get("message", {}).get("content", "")
            p_tokens = data.get("prompt_eval_count")
            c_tokens = data.get("eval_count")
            return LLMCompletionResponse(
                content=content,
                model=self._model,
                provider=self._provider,
                raw_response=data,
                prompt_tokens=p_tokens,
                completion_tokens=c_tokens,
            )
        except Exception as parse_exc:
            raise LLMResponseError(f"Failed to decode Ollama response JSON: {parse_exc}")


# ---------------------------------------------------------------------------
# LLM Factory
# ---------------------------------------------------------------------------
def get_llm_client(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: Optional[float] = None,
    **kwargs: Any,
) -> BaseLLMClient:
    """
    Factory function to instantiate and return an LLM client.

    Resolution order for provider:
    1. Explicit `provider` argument ("groq" or "ollama")
    2. `LLM_PROVIDER` environment variable
    3. `settings.llm_provider` from configuration
    4. Default: "groq"

    Args:
        provider: "groq" or "ollama"
        model: Optional model identifier override
        api_key: Optional API key override (for Groq)
        base_url: Optional base URL override (for Ollama)
        timeout: Optional timeout in seconds
        **kwargs: Additional provider-specific kwargs (e.g. client injection for tests)

    Returns:
        BaseLLMClient instance (GroqLLMClient or OllamaLLMClient).

    Raises:
        LLMConfigurationError: If an unsupported provider name is given.
    """
    resolved_provider = (
        provider
        or os.getenv("LLM_PROVIDER", "")
        or getattr(settings, "llm_provider", "")
        or "groq"
    ).strip().lower()

    if resolved_provider == "ollama":
        return OllamaLLMClient(
            model=model,
            base_url=base_url,
            timeout=timeout or 60.0,
            http_client=kwargs.get("http_client"),
        )
    elif resolved_provider == "groq":
        return GroqLLMClient(
            model=model,
            api_key=api_key,
            timeout=timeout or 30.0,
            client=kwargs.get("client"),
        )
    else:
        raise LLMConfigurationError(
            f"Unsupported LLM provider '{resolved_provider}'. Supported options are: 'groq', 'ollama'."
        )
