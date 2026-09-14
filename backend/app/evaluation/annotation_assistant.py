"""
SupportGraph AI — Groq Annotation Assistant Service (Phase 5.5)

Provides AI-assisted intent label suggestions for the Golden Evaluation Set.

SCIENTIFIC INTEGRITY RULES:
1. The AI suggestion is an advisory hint, NOT ground truth.
2. The AI suggestion MUST NEVER automatically become the final human annotation.
3. The fields `annotation_label` and `annotator` are reserved exclusively for humans.
4. Suggestions are strictly constrained to the 9 operational intents + `unclear_needs_review`.
5. Product names (iPhone, Mac) are entities, not intents.
6. Temporal events (iOS 11, late 2017 autocorrect bug) must not distort intent classification.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

try:
    from app.core.config import settings
    from app.core.llm_factory import (
        BaseLLMClient,
        GroqLLMClient,
        LLMAuthenticationError,
        LLMConfigurationError,
        LLMConnectionError,
        LLMError,
        LLMRateLimitError,
        OllamaLLMClient,
        get_llm_client,
    )
    from app.core.logging import get_logger
    from app.nlp.taxonomy_finalization import (
        DEFAULT_CANDIDATE_TAXONOMY_PATH,
        TaxonomyCandidate,
        create_initial_candidate_taxonomy,
        load_candidate_taxonomy,
    )
except ModuleNotFoundError:
    from backend.app.core.config import settings  # type: ignore[no-redef]
    from backend.app.core.llm_factory import (  # type: ignore[no-redef]
        BaseLLMClient,
        GroqLLMClient,
        LLMAuthenticationError,
        LLMConfigurationError,
        LLMConnectionError,
        LLMError,
        LLMRateLimitError,
        OllamaLLMClient,
        get_llm_client,
    )
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]
    from backend.app.nlp.taxonomy_finalization import (  # type: ignore[no-redef]
        DEFAULT_CANDIDATE_TAXONOMY_PATH,
        TaxonomyCandidate,
        create_initial_candidate_taxonomy,
        load_candidate_taxonomy,
    )

logger = get_logger(__name__)

# Default model identifiers
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
FALLBACK_GROQ_MODEL = "openai/gpt-oss-20b"
DEFAULT_OLLAMA_MODEL = "llama3.2:latest"


# ---------------------------------------------------------------------------
# Exceptions (mapped to LLM Factory exceptions for backwards compatibility)
# ---------------------------------------------------------------------------
class GroqConfigurationError(LLMConfigurationError):
    """Raised when Groq API key or required configuration is missing or invalid."""


class GroqAuthenticationError(LLMAuthenticationError):
    """Raised when Groq API rejects authentication credentials."""


class GroqRateLimitError(LLMRateLimitError):
    """Raised when Groq API rate limits are encountered and exhausted."""


class GroqConnectionError(LLMConnectionError):
    """Raised when connection to Groq API servers fails."""


# ---------------------------------------------------------------------------
# Structured Output Model
# ---------------------------------------------------------------------------
@dataclass
class AISuggestion:
    """Structured AI intent suggestion. Advisory only; never ground truth."""
    model_name: str
    model_suggested_label: str = ""
    model_confidence: Optional[float] = None
    model_reasoning_summary: str = ""
    model_needs_human_review: bool = True
    suggestion_timestamp: str = ""
    suggestion_status: str = "success"  # "success", "failed"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Unified Annotation Assistant Service
# ---------------------------------------------------------------------------
class AnnotationAssistant:
    """
    Assistant service generating structured intent suggestions via LLM Factory (Groq or Ollama).
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        taxonomy_path: Optional[Path | str] = None,
        llm_client: Optional[BaseLLMClient] = None,
        client: Optional[Any] = None,
    ) -> None:
        """
        Initialize the Annotation Assistant.

        Args:
            provider: LLM provider ("groq" or "ollama"). Defaults to LLM_PROVIDER or settings.
            api_key: Optional Groq API key. Defaults to GROQ_API_KEY env or settings.
            model: Model identifier override.
            base_url: Optional Ollama base URL override.
            taxonomy_path: Path to candidate taxonomy JSON.
            llm_client: Optional pre-configured BaseLLMClient instance.
            client: Optional pre-configured client (useful for unit test mocking).
        """
        # Resolve LLM client
        if llm_client is not None:
            self.llm_client = llm_client
        elif client is not None:
            self.llm_client = GroqLLMClient(
                model=model,
                api_key=api_key or "mock_key",
                client=client,
            )
        else:
            self.llm_client = get_llm_client(
                provider=provider,
                model=model,
                api_key=api_key,
                base_url=base_url,
            )

        self.provider: str = self.llm_client.provider
        self.model: str = self.llm_client.model
        self.api_key: str = getattr(self.llm_client, "api_key", "")
        self._client: Optional[Any] = getattr(self.llm_client, "_client", None)

        # Load taxonomy definitions
        tax_path = Path(taxonomy_path) if taxonomy_path else Path(DEFAULT_CANDIDATE_TAXONOMY_PATH)
        if tax_path.exists():
            self.taxonomy = load_candidate_taxonomy(tax_path)
        else:
            self.taxonomy = create_initial_candidate_taxonomy()

        self.allowed_labels: list[str] = sorted(self.taxonomy.intent_names)
        self.all_valid_labels: set[str] = set(self.allowed_labels) | {"unclear_needs_review"}

        self._system_prompt = self._build_system_prompt()

    @property
    def is_configured(self) -> bool:
        """Return True if LLM client is properly configured and initialized."""
        if self.provider == "groq":
            return bool(self.api_key and self._client is not None)
        return bool(self.llm_client is not None)

    def _build_system_prompt(self) -> str:
        """Construct deterministic system instructions incorporating taxonomy boundaries and hierarchical rules."""
        intents_bullet_list = []
        for intent in self.taxonomy.intents:
            inc = "; ".join(intent.include_when[:2])
            exc = "; ".join(intent.exclude_when[:2])
            intents_bullet_list.append(
                f"- `{intent.intent_name}`:\n"
                f"    Definition: {intent.definition}\n"
                f"    Include when: {inc}\n"
                f"    Exclude when: {exc}"
            )

        intents_text = "\n".join(intents_bullet_list)

        prompt = f"""You are a specialized Customer Support Annotation Assistant for AppleSupport.
Your job is to SUGGEST the single primary operational intent for an incoming customer tweet.

IMPORTANT DISCLAIMER:
This is an annotation suggestion. It is not ground truth and requires human review.

ALLOWED LABELS (Select EXACTLY ONE from this list):
{intents_text}
- `unclear_needs_review`: Use ONLY when the customer message is completely ambiguous, incomprehensible, fragmented, non-English, or missing necessary details to determine what the problem is.

CORE HIERARCHICAL CLASSIFICATION RULES:
1. RULE OF SPECIFICITY: A clearly supported specific operational intent (Rules 2-7 below) ALWAYS takes precedence over `general_device_support`. You MUST NOT choose `general_device_support` if any specific category fits.
2. SYMPTOM OVER ENTITY: Do NOT classify an issue as `mac_software_issue` merely because a Mac/MacBook is mentioned. Classify by the actual functional symptom:
   - MacBook speaker has no sound / volume defect -> `hardware_audio_connection_issue`
   - MacBook physical keys / keyboard stopped working -> `keyboard_typing_issue`
   - MacBook shutdown / battery / charging failure -> `battery_power_issue`
   - MacBook screen flickering / display defect -> `display_touch_issue`
   Use `mac_software_issue` ONLY when the primary problem is specifically Mac/macOS software behavior (Time Machine, Finder freeze, macOS syncing, spinning beach ball, kernel panic, macOS installer error).
3. UPDATE CAUSALITY: Strongly prioritize `software_update_problem` when the customer explicitly states that the issue:
   - Started after an update, happened ever since updating, or occurred during downloading/installing a software update.
   - Is caused by a new iOS/macOS version (e.g. "iOS 11 is glitched out", "bugs in iOS 11", "update is ruining my life", "download stuck").
   - Complains generally about multiple lag/bug symptoms following an update.
   (Exception: If the customer exclusively describes one isolated component issue like keyboard autocorrect, map to that specific component).
4. AUDIO PRIORITY: Explicit evidence of no sound, speakers crackling/low volume, audio playback failure (e.g. "music app produces no sound"), headphones, AirPods, or Bluetooth audio MUST be classified as `hardware_audio_connection_issue`. NEVER classify an audio problem as `display_touch_issue` or `mac_software_issue`.
5. KEYBOARD PRIORITY: Explicit evidence of keyboard, typing, physical/virtual keys, autocorrect suggestions, predictive text, or character glitching (such as the letter 'I' converting to unicode symbol blocks or autocorrecting to unwanted words) MUST be classified as `keyboard_typing_issue`. Do NOT classify text typing or autocorrect glitches as `display_touch_issue`.
6. DISPLAY AND TOUCH BOUNDARY: Use `display_touch_issue` ONLY for genuine screen hardware defects, broken/shattered glass, touch digitizer unresponsiveness, screen flickering, or black screens. Do NOT classify a message as display-related merely because an app notification appears incorrectly or because text/music displays on a screen.
7. BILLING BOUNDARY: Use `billing_purchase_issue` ONLY when there is actual evidence of payments, charges, subscriptions, refunds, credit cards, or purchase transactions. Technical App Store download/installation failures with no payment mention belong under `general_device_support`.
8. GENERAL DEVICE SUPPORT AS STRICT FALLBACK: Use `general_device_support` ONLY when the customer clearly needs technical help but NONE of the more specific taxonomy intents apply (e.g. general WiFi connectivity across devices, location tracking settings, warranty/exchange questions, general device troubleshooting).
9. UNCLEAR / NON-SUPPORT BOUNDARY: Use `unclear_needs_review` when the message is pure non-support social chatter, praise/compliments without an issue, or when key context is completely missing (e.g. "it works on my phone but not MacBook" without stating what "it" is).

FEW-SHOT EXAMPLES:
Example 1 (Autocorrect / Letter Glitch):
Customer Message: "@AppleSupport please fix this letter 'eye' problem!! I paid all this money for this phone & I'm still seeing blocks!"
Response:
{{"suggested_label": "keyboard_typing_issue", "confidence": 0.95, "reasoning_summary": "Customer reports the letter 'I' typing glitch where text input displays as block symbols, which is a keyboard typing/autocorrect issue.", "needs_human_review": false}}

Example 2 (Audio / Music App No Sound):
Customer Message: "@AppleSupport the music app is not producing sound. Other apps produce sound. Can play music but no sound comes out. Help."
Response:
{{"suggested_label": "hardware_audio_connection_issue", "confidence": 0.95, "reasoning_summary": "Customer describes a failure of audio sound output when playing music, which is an audio functionality issue.", "needs_human_review": false}}

Example 3 (Mac System Behavior / Spinning Beach Ball):
Customer Message: "@AppleSupport my MacBook Air has the dreaded spinning beach ball and Time Machine notifications won't stop. Help please."
Response:
{{"suggested_label": "mac_software_issue", "confidence": 0.95, "reasoning_summary": "Customer reports macOS UI freezing (spinning beach ball) and Time Machine system behavior on a Mac, which is a Mac software issue.", "needs_human_review": false}}

Example 4 (Technical App Download Failure vs. Billing):
Customer Message: "@AppleSupport why aren't my apps downloading??? I've tried with and without WiFi..."
Response:
{{"suggested_label": "general_device_support", "confidence": 0.90, "reasoning_summary": "Customer reports technical issue downloading apps over WiFi with no billing or payment failure mentioned, fitting general device support.", "needs_human_review": false}}

Example 5 (Update Causality & General Bug Complaints):
Customer Message: "@AppleSupport ever since the new update my phone has been EXTREMELY laggy, keyboard lags and opening apps lags. Please fix all these iOS 11 bugs."
Response:
{{"suggested_label": "software_update_problem", "confidence": 0.95, "reasoning_summary": "Customer explicitly attributes system-wide lag and bugs to a recent iOS 11 software update, indicating a software update problem.", "needs_human_review": false}}

OUTPUT FORMAT:
Return a JSON object ONLY, with no extra text or markdown formatting:
{{
  "suggested_label": "<one of the allowed labels>",
  "confidence": <float between 0.0 and 1.0>,
  "reasoning_summary": "<concise 1-2 sentence explanation of why this label applies>",
  "needs_human_review": <true if borderline or ambiguous, else false>
}}
"""
        return prompt

    def list_accessible_models(self) -> list[str]:
        """
        Query LLM provider for list of accessible model identifiers.

        Returns:
            Sorted list of accessible model IDs.
        """
        return self.llm_client.list_models()

    def verify_model_access(self) -> tuple[bool, list[str], str]:
        """
        Verify connectivity and check whether configured model is accessible.

        Returns:
            (is_valid, available_models, error_message)
        """
        return self.llm_client.verify_access()

    def parse_model_response(self, raw_content: str) -> dict[str, Any]:
        """
        Clean, parse, and validate JSON returned by the model.

        Raises:
            ValueError: If the response is not valid JSON or specifies an invalid label.
        """
        text = raw_content.strip()
        # Strip markdown code blocks if wrapped
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\n?", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\n?```$", "", text)
            text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to decode JSON response from LLM: %s | Raw text: %s", exc, raw_content[:120])
            raise ValueError(f"Model response could not be parsed as valid JSON: {exc}")

        # Validate suggested_label
        raw_label = str(data.get("suggested_label", "")).strip()
        if not raw_label or raw_label.lower() in ("nan", "none", "null", "undefined"):
            raise ValueError(f"Model suggested empty or invalid label '{raw_label}'.")
        if raw_label not in self.all_valid_labels:
            logger.warning("Model suggested unauthorized label '%s'. Rejecting.", raw_label)
            raise ValueError(f"Model suggested unauthorized label '{raw_label}'. Rejected.")

        # Validate confidence
        raw_conf = data.get("confidence")
        if raw_conf is None or raw_conf == "":
            raise ValueError("Model response missing required numeric 'confidence' field.")
        try:
            confidence = float(raw_conf)
        except (ValueError, TypeError):
            raise ValueError(f"Model confidence '{raw_conf}' cannot be parsed as a numeric float.")

        import math
        if math.isnan(confidence) or math.isinf(confidence) or confidence < 0.0 or confidence > 1.0:
            raise ValueError(f"Model confidence {confidence} is outside the valid range [0.0, 1.0].")

        confidence = round(confidence, 4)

        # Validate reasoning
        reasoning = str(data.get("reasoning_summary", "")).strip()
        if not reasoning:
            reasoning = f"Suggested label: {raw_label}."

        needs_review = bool(data.get("needs_human_review", confidence < 0.70))

        return {
            "suggested_label": raw_label,
            "confidence": confidence,
            "reasoning_summary": reasoning,
            "needs_human_review": needs_review,
        }

    def suggest_intent(
        self,
        customer_message: str,
        normalized_message: str = "",
        conversation_context: str = "",
    ) -> AISuggestion:
        """
        Generate an intent label suggestion for a single customer message.

        Args:
            customer_message: Raw customer opening message.
            normalized_message: Cleaned message text.
            conversation_context: Brand first reply if available.

        Returns:
            AISuggestion dataclass instance.
        """
        now_utc = datetime.now(timezone.utc).isoformat()

        if not self.is_configured:
            return AISuggestion(
                model_name=self.model,
                model_suggested_label="",
                model_confidence=None,
                model_reasoning_summary="",
                model_needs_human_review=True,
                suggestion_timestamp=now_utc,
                suggestion_status="failed",
            )

        user_content = f"Customer Opening Message:\n\"{customer_message}\"\n"
        if normalized_message and normalized_message != customer_message:
            user_content += f"\nNormalized Text:\n\"{normalized_message}\"\n"
        if conversation_context and conversation_context != "No brand response observed":
            user_content += f"\nConversation Context (First Brand Turn):\n\"{conversation_context}\"\n"

        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
        ]

        try:
            response = self.llm_client.chat_completion(
                messages=messages,
                temperature=0.0,
                response_format={"type": "json_object"},
                max_tokens=1024,
            )
        except LLMAuthenticationError as auth_err:
            logger.error("Authentication failed: %s", auth_err)
            raise GroqAuthenticationError(str(auth_err))
        except Exception as exc:
            logger.warning("LLM provider call failed for %s (%s): %s", self.model, self.provider, exc)
            return AISuggestion(
                model_name=self.model,
                model_suggested_label="",
                model_confidence=None,
                model_reasoning_summary="",
                model_needs_human_review=True,
                suggestion_timestamp=now_utc,
                suggestion_status="failed",
            )

        # Parse and validate response content
        try:
            parsed = self.parse_model_response(response.content)
            return AISuggestion(
                model_name=self.model,
                model_suggested_label=parsed["suggested_label"],
                model_confidence=parsed["confidence"],
                model_reasoning_summary=parsed["reasoning_summary"],
                model_needs_human_review=parsed["needs_human_review"],
                suggestion_timestamp=now_utc,
                suggestion_status="success",
            )
        except Exception as parse_exc:
            logger.warning("Invalid model output from %s (%s): %s", self.model, self.provider, parse_exc)
            return AISuggestion(
                model_name=self.model,
                model_suggested_label="unclear_needs_review",
                model_confidence=0.0,
                model_reasoning_summary=f"Invalid model output: {parse_exc}. Flagged for human review.",
                model_needs_human_review=True,
                suggestion_timestamp=now_utc,
                suggestion_status="invalid_model_output",
            )

    def suggest_for_record(self, record: dict[str, Any] | pd.Series) -> AISuggestion:
        """Generate suggestion from a record dict or Series."""
        return self.suggest_intent(
            customer_message=str(record.get("customer_message", "")),
            normalized_message=str(record.get("normalized_message", "")),
            conversation_context=str(record.get("conversation_context", "")),
        )


class GroqAnnotationAssistant(AnnotationAssistant):
    """
    Backwards-compatible subclass of AnnotationAssistant.
    Defaults provider to 'groq' if not explicitly configured otherwise.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if "provider" not in kwargs:
            kwargs["provider"] = os.getenv("LLM_PROVIDER") or getattr(settings, "llm_provider", "groq")
        super().__init__(*args, **kwargs)
