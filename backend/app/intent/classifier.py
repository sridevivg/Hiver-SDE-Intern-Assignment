"""
SupportGraph AI — Top-K Intent Classifier (Phase 6)

Predicts ranked candidate intents with calibrated confidences and reasoning
for incoming customer support messages.

Core Principles:
1. Predicts Top-K candidate intents (default K=3) rather than a single overconfident label.
2. Validates all labels against the approved operational intent taxonomy + unclear_needs_review.
3. Computes mathematically grounded probabilities, confidence margins, and normalized entropy.
4. Provides deterministic fallback heuristic classification when LLM client is offline or unconfigured.
5. Advisory output designed for uncertainty-aware routing and human-in-the-loop escalation.
"""
from __future__ import annotations

import json
import logging
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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
        get_llm_client,
    )
    from app.core.logging import get_logger
    from app.nlp.taxonomy_finalization import (
        DEFAULT_CANDIDATE_TAXONOMY_PATH,
        INTENT_KEYWORD_PATTERNS,
        TaxonomyCandidate,
        assign_candidate_intent,
        create_initial_candidate_taxonomy,
        load_candidate_taxonomy,
    )
    from app.schemas.intent_routing import IntentAnalysis, IntentPrediction
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
        get_llm_client,
    )
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]
    from backend.app.nlp.taxonomy_finalization import (  # type: ignore[no-redef]
        DEFAULT_CANDIDATE_TAXONOMY_PATH,
        INTENT_KEYWORD_PATTERNS,
        TaxonomyCandidate,
        assign_candidate_intent,
        create_initial_candidate_taxonomy,
        load_candidate_taxonomy,
    )
    from backend.app.schemas.intent_routing import IntentAnalysis, IntentPrediction  # type: ignore[no-redef]

logger = get_logger(__name__)


def compute_normalized_entropy(probabilities: list[float]) -> float:
    """
    Compute normalized Shannon entropy for a probability distribution.
    H_norm = - sum(p * log2(p)) / log2(K)

    Returns:
        float in [0.0, 1.0], where 0.0 is complete certainty and 1.0 is maximum confusion.
    """
    valid_probs = [p for p in probabilities if p > 0.0]
    k = len(valid_probs)
    if k <= 1:
        return 0.0

    entropy = -sum(p * math.log2(p) for p in valid_probs)
    max_entropy = math.log2(len(probabilities)) if len(probabilities) > 1 else 1.0
    if max_entropy <= 0.0:
        return 0.0

    norm_h = entropy / max_entropy
    return round(min(max(norm_h, 0.0), 1.0), 4)


class TopKIntentClassifier:
    """
    Uncertainty-aware Top-K Intent Classifier.
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
        top_k: int = 3,
    ) -> None:
        """
        Initialize the Top-K Intent Classifier.
        """
        if llm_client is not None:
            self.llm_client = llm_client
        elif client is not None:
            self.llm_client = GroqLLMClient(
                model=model or settings.groq_model,
                api_key=api_key or "mock_key",
                client=client,
            )
        else:
            self.llm_client = get_llm_client(
                provider=provider or settings.llm_provider,
                model=model,
                api_key=api_key or settings.groq_api_key,
                base_url=base_url or settings.ollama_base_url,
            )

        self.provider: str = self.llm_client.provider
        self.model: str = self.llm_client.model
        self.top_k: int = top_k or settings.top_k_predictions_count

        # Load taxonomy
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
        """Check if classifier LLM backend is properly configured."""
        if self.provider == "groq":
            api_key = getattr(self.llm_client, "api_key", "")
            return bool(api_key and getattr(self.llm_client, "_client", None) is not None)
        return bool(self.llm_client is not None)

    def _build_system_prompt(self) -> str:
        """Construct prompt instructions for Top-K multi-candidate prediction."""
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

        prompt = f"""You are the Runtime Intent Classification Engine for SupportGraph AI (AppleSupport).
Given an incoming customer support message, predict the Top-3 candidate operational intents ranked by probability.

ALLOWED TAXONOMY LABELS:
{intents_text}
- `unclear_needs_review`: Use ONLY when the customer message is completely ambiguous, incomprehensible, fragmented, non-English, or missing necessary details.

CLASSIFICATION RULES:
1. RULE OF SPECIFICITY: A specific operational intent always takes precedence over `general_device_support`.
2. SYMPTOM OVER ENTITY: Classify by the actual symptom (audio, battery, keyboard, display), not merely hardware mention (Mac vs iPhone).
3. UPDATE CAUSALITY: Prioritize `software_update_problem` when the customer attributes errors/bugs to an iOS/macOS update.
4. CALIBRATED PROBABILITIES: Provide realistic, calibrated confidence scores for the top candidate intents reflecting true ambiguity.
   - If clear and unambiguous: Top-1 confidence should be high (0.90-0.98), Top-2 low (0.02-0.08).
   - If ambiguous or touching multiple areas: Top-1 (0.45-0.60) and Top-2 (0.35-0.45) should reflect competing interpretations.
   - The confidences across candidate predictions should sum to approximately 1.0.

OUTPUT FORMAT:
Return a JSON object ONLY, with no markdown code blocks or surrounding text:
{{
  "predictions": [
    {{
      "intent": "<top_1_intent>",
      "confidence": <float between 0.0 and 1.0>,
      "reasoning": "<concise reason for top candidate>"
    }},
    {{
      "intent": "<top_2_intent>",
      "confidence": <float between 0.0 and 1.0>,
      "reasoning": "<concise reason for 2nd candidate>"
    }},
    {{
      "intent": "<top_3_intent>",
      "confidence": <float between 0.0 and 1.0>,
      "reasoning": "<concise reason for 3rd candidate>"
    }}
  ]
}}
"""
        return prompt

    def parse_predictions_json(self, raw_content: str, requested_k: int = 3) -> list[IntentPrediction]:
        """
        Parse and validate Top-K predictions JSON from LLM response.
        """
        text = raw_content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\n?", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\n?```$", "", text)
            text = text.strip()

        data = json.loads(text)

        # Support both {"predictions": [...]} and single object {"suggested_label": ...}
        raw_list = []
        if isinstance(data, dict) and "predictions" in data and isinstance(data["predictions"], list):
            raw_list = data["predictions"]
        elif isinstance(data, dict) and "suggested_label" in data:
            raw_list = [
                {
                    "intent": data.get("suggested_label"),
                    "confidence": data.get("confidence", 0.90),
                    "reasoning": data.get("reasoning_summary", ""),
                }
            ]
        elif isinstance(data, list):
            raw_list = data
        else:
            raise ValueError(f"Unrecognized prediction format in LLM response: {type(data)}")

        if not raw_list:
            raise ValueError("Empty predictions list in model response.")

        validated: list[IntentPrediction] = []
        seen_intents: set[str] = set()

        for item in raw_list:
            if not isinstance(item, dict):
                continue
            intent = str(item.get("intent", "")).strip()
            if not intent or intent not in self.all_valid_labels:
                logger.warning("Ignoring unauthorized candidate intent: '%s'", intent)
                continue
            if intent in seen_intents:
                continue
            seen_intents.add(intent)

            try:
                conf = float(item.get("confidence", 0.0))
            except (ValueError, TypeError):
                conf = 0.0
            conf = max(0.0, min(1.0, conf))

            reasoning = str(item.get("reasoning", "")).strip()
            validated.append(IntentPrediction(intent=intent, confidence=conf, reasoning=reasoning))

        if not validated:
            raise ValueError("No valid approved taxonomy predictions found in model output.")

        # Ensure sorted descending by confidence
        validated.sort(key=lambda p: p.confidence, reverse=True)

        # Normalize probabilities across candidate predictions
        total_conf = sum(p.confidence for p in validated)
        if total_conf > 0.0:
            for p in validated:
                p.confidence = round(p.confidence / total_conf, 4)
        else:
            uniform = round(1.0 / len(validated), 4)
            for p in validated:
                p.confidence = uniform

        return validated[:requested_k]

    def classify_heuristic_fallback(
        self,
        customer_message: str,
        requested_k: int = 3,
    ) -> list[IntentPrediction]:
        """
        Deterministic heuristic classifier used when LLM client is offline or unconfigured.
        Produces ranked candidate intents with calibrated score distributions.
        """
        t_low = customer_message.lower()
        scores: dict[str, float] = {}

        # 1. Evaluate keyword matches
        for intent_name, pattern in INTENT_KEYWORD_PATTERNS.items():
            matches = len(pattern.findall(t_low))
            if matches > 0:
                scores[intent_name] = scores.get(intent_name, 0.0) + (matches * 1.5)

        # 2. Check for unclear / social noise
        if len(t_low.split()) < 3 and not scores:
            scores["unclear_needs_review"] = 2.0

        # 3. Default fallback
        if not scores:
            scores["general_device_support"] = 1.0
        elif "general_device_support" not in scores:
            scores["general_device_support"] = 0.2  # baseline prior

        # Sort and take top_k
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        total = sum(v for _, v in ranked[:requested_k])

        predictions = []
        for intent, sc in ranked[:requested_k]:
            norm_conf = round(sc / total, 4) if total > 0 else round(1.0 / len(ranked[:requested_k]), 4)
            predictions.append(
                IntentPrediction(
                    intent=intent,
                    confidence=norm_conf,
                    reasoning=f"Heuristic pattern match score {sc:.2f}",
                )
            )

        # If fewer than requested_k, add remaining allowed labels with small probability
        if len(predictions) < requested_k:
            remaining_labels = [lbl for lbl in self.allowed_labels if lbl not in [p.intent for p in predictions]]
            while len(predictions) < requested_k and remaining_labels:
                predictions.append(
                    IntentPrediction(
                        intent=remaining_labels.pop(0),
                        confidence=0.01,
                        reasoning="Taxonomy background candidate",
                    )
                )
            # Re-normalize
            total = sum(p.confidence for p in predictions)
            for p in predictions:
                p.confidence = round(p.confidence / total, 4)

        predictions.sort(key=lambda p: p.confidence, reverse=True)
        return predictions

    def classify(
        self,
        customer_message: str,
        normalized_message: str = "",
        conversation_context: str = "",
        top_k: Optional[int] = None,
    ) -> IntentAnalysis:
        """
        Classify customer message returning Top-K candidate predictions and uncertainty metrics.
        """
        k = top_k or self.top_k
        model_name = self.model

        if not self.is_configured:
            logger.info("Classifier LLM unconfigured. Using deterministic heuristic fallback.")
            predictions = self.classify_heuristic_fallback(customer_message, requested_k=k)
            model_name = "heuristic_fallback"
        else:
            user_content = f"Customer Opening Message:\n\"{customer_message}\"\n"
            if normalized_message and normalized_message != customer_message:
                user_content += f"\nNormalized Text:\n\"{normalized_message}\"\n"
            if conversation_context and conversation_context != "No brand response observed":
                user_content += f"\nConversation Context:\n\"{conversation_context}\"\n"

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
                predictions = self.parse_predictions_json(response.content, requested_k=k)
            except Exception as exc:
                logger.warning("LLM prediction failed (%s). Falling back to heuristic: %s", exc, exc)
                predictions = self.classify_heuristic_fallback(customer_message, requested_k=k)
                model_name = f"{self.model}_fallback"

        # Calculate uncertainty metrics
        top_conf = predictions[0].confidence if predictions else 0.0
        sec_conf = predictions[1].confidence if len(predictions) > 1 else 0.0
        margin = round(max(0.0, top_conf - sec_conf), 4)

        entropy = compute_normalized_entropy([p.confidence for p in predictions])

        return IntentAnalysis(
            top_predictions=predictions,
            top_confidence=top_conf,
            confidence_margin=margin,
            normalized_entropy=entropy,
            model_name=model_name,
        )
