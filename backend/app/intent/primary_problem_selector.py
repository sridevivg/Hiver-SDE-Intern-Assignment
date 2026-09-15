"""
SupportGraph AI — Primary Problem Selector (Phase 7)

Identifies the core operational problem the customer is asking support to resolve:
- Decouples primary symptom (what is failing) from contextual cause (e.g. software update)
- Prevents misclassification of WiFi/network inquiries into audio/hardware categories
- Maps general performance complaints after OS updates to software_update_problem
"""
from __future__ import annotations

import logging
import re
from typing import Optional, Tuple

try:
    from app.core.logging import get_logger
    from app.schemas.intent_routing import IntentAnalysis, IntentPrediction
    from app.understanding.problem_extractor import CustomerProblemProfile
except ModuleNotFoundError:
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]
    from backend.app.schemas.intent_routing import (  # type: ignore[no-redef]
        IntentAnalysis,
        IntentPrediction,
    )
    from backend.app.understanding.problem_extractor import CustomerProblemProfile  # type: ignore[no-redef]

logger = get_logger(__name__)


class PrimaryProblemSelector:
    """
    Selects and explains the primary operational customer problem.
    """

    def select_primary_problem(
        self,
        message: str,
        profile: CustomerProblemProfile,
        analysis: IntentAnalysis,
    ) -> tuple[str, Optional[str], str]:
        """
        Determine primary operational intent, contextual cause, and decision rationale.

        Returns:
            (primary_intent, contextual_cause_intent, explanation)
        """
        t_low = message.lower()
        top_1 = analysis.top_1_intent
        top_2 = analysis.top_2_intent

        # Rule 1: Specific Component Symptom with Update Cause (e.g. Battery Drain after Update)
        if profile.update_related:
            if "battery" in profile.primary_symptom or "power" in profile.primary_symptom:
                return (
                    "battery_power_issue",
                    "software_update_problem",
                    "Primary customer complaint is rapid battery drain; software update is identified as contextual cause.",
                )
            if "audio" in profile.primary_symptom or "speaker" in profile.primary_symptom:
                return (
                    "hardware_audio_connection_issue",
                    "software_update_problem",
                    "Primary customer complaint is speaker/audio failure; software update is identified as contextual cause.",
                )
            if "keyboard" in profile.primary_symptom or "autocorrect" in profile.primary_symptom:
                return (
                    "keyboard_typing_issue",
                    "software_update_problem",
                    "Primary customer complaint is keyboard/autocorrect anomaly; software update is contextual trigger.",
                )
            if "screen" in profile.primary_symptom or "display" in profile.primary_symptom:
                return (
                    "display_touch_issue",
                    "software_update_problem",
                    "Primary customer complaint is display malfunction; software update is contextual trigger.",
                )

            # If general lag/bug without isolated component symptom -> software_update_problem
            if re.search(r"\b(lag|slow|bug|glitch|freeze|crash|bricked|won't\s+work)\b", t_low):
                return (
                    "software_update_problem",
                    None,
                    "Customer reports broad system sluggishness and instability attributed directly to the software update.",
                )

        # Rule 2: WiFi / Connectivity Guardrail
        # If customer asks about WiFi / internet / Bluetooth without speaker/audio context, map to general_device_support
        if re.search(r"\b(wifi|wi-fi|internet|connection|cellular|no\s+service)\b", t_low):
            if not re.search(r"\b(speaker|sound|audio|mic|headphone|airpod|volume)\b", t_low):
                return (
                    "general_device_support",
                    None,
                    "WiFi and network connectivity troubleshooting falls under general device support triage.",
                )

        # Rule 3: Financial & Billing Priority
        if profile.financial_related or re.search(r"\b(charge|charged|bill|billing|refund|subscription|payment)\b", t_low):
            return (
                "billing_purchase_issue",
                None,
                "Customer inquiry concerns billing, payment charges, or subscription refunds.",
            )

        # Rule 4: Credential & Account Access
        if re.search(r"\b(password|passcode|apple\s*id|locked\s+account|2fa|verification\s+code)\b", t_low):
            return (
                "account_access_issue",
                None,
                "Customer inquiry concerns account credential access, password recovery, or account lockout.",
            )

        # Default to Classifier Top-1
        return (
            top_1,
            None,
            f"Selected primary intent '{top_1}' supported by message problem profile.",
        )
