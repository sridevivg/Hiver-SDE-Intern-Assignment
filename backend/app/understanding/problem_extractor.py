"""
SupportGraph AI — Problem Understanding Layer (Phase 7)

Extracts structured customer problem representations before intent classification:
- Identifies device entities, products/services, primary and secondary symptoms
- Explicitly separates primary operational symptom from contextual causes (e.g. software updates)
- Evaluates information sufficiency (sufficient, partial, insufficient)
- Operates deterministically with optional LLM enhancement
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

try:
    from app.core.config import settings
    from app.core.llm_factory import BaseLLMClient, get_llm_client
    from app.core.logging import get_logger
    from app.understanding.problem_family_registry import (
        OperationalProblemFamily,
        ProblemFamilyDetector,
    )
except ModuleNotFoundError:
    from backend.app.core.config import settings  # type: ignore[no-redef]
    from backend.app.core.llm_factory import BaseLLMClient, get_llm_client  # type: ignore[no-redef]
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]
    from backend.app.understanding.problem_family_registry import (  # type: ignore[no-redef]
        OperationalProblemFamily,
        ProblemFamilyDetector,
    )

logger = get_logger(__name__)

DEVICE_PATTERNS = {
    "iPhone": re.compile(r"\b(iphone\s*\d+\s*(?:plus|pro|max)?|iphone\s*x[s|r]?|iphone\s*se|iphone)\b", re.IGNORECASE),
    "MacBook": re.compile(r"\b(macbook\s*pro|macbook\s*air|macbook|mac\s*pro|mac\s*mini|imac)\b", re.IGNORECASE),
    "iPad": re.compile(r"\b(ipad\s*pro|ipad\s*air|ipad\s*mini|ipad)\b", re.IGNORECASE),
    "Apple Watch": re.compile(r"\b(apple\s*watch|iwatch|watchos)\b", re.IGNORECASE),
    "AirPods": re.compile(r"\b(airpods\s*pro|airpods|earpods|airpod)\b", re.IGNORECASE),
    "Apple TV": re.compile(r"\b(apple\s*tv|tvos)\b", re.IGNORECASE),
}

SERVICE_PATTERNS = {
    "iOS": re.compile(r"\b(ios\s*\d+(?:\.\d+)*|ios11|ios)\b", re.IGNORECASE),
    "macOS": re.compile(r"\b(macos|high\s*sierra|sierra|os\s*x)\b", re.IGNORECASE),
    "App Store": re.compile(r"\b(app\s*store|itunes|itunes\s*store)\b", re.IGNORECASE),
    "Apple ID / iCloud": re.compile(r"\b(apple\s*id|appleid|icloud|2fa|two-factor)\b", re.IGNORECASE),
    "Apple Music": re.compile(r"\b(apple\s*music|music\s*subscription)\b", re.IGNORECASE),
}

SYMPTOM_PATTERNS = {
    "battery_drain": (
        re.compile(r"\b(battery|drain|draining|dies|die|percentage|overheating|battery\s+health)\b", re.IGNORECASE),
        "rapid battery drain / power depletion",
    ),
    "charging_failure": (
        re.compile(r"\b(charge|charging|charger|won't\s+charge|not\s+charging|power\s+cord)\b", re.IGNORECASE),
        "device failure to charge or power up",
    ),
    "audio_issue": (
        re.compile(r"\b(speaker|sound|audio|crackling|volume|no\s+sound|loudspeaker|microphone|mic)\b", re.IGNORECASE),
        "speaker or microphone audio failure",
    ),
    "display_issue": (
        re.compile(r"\b(screen|display|cracked|black\s+screen|flicker|flickering|lines\s+on\s+screen|touch|digitizer)\b", re.IGNORECASE),
        "display malfunction or physical screen damage",
    ),
    "keyboard_issue": (
        re.compile(r"\b(keyboard|typing|autocorrect|letter\s+i|predictive|keys|type|letter\s+eye)\b", re.IGNORECASE),
        "keyboard text input or autocorrect glitch",
    ),
    "system_lag_freeze": (
        re.compile(r"\b(lag|lagging|freeze|freezing|slow|frozen|crashing|crash|beach\s+ball)\b", re.IGNORECASE),
        "system sluggishness, freezing, or app crashes",
    ),
    "account_lock": (
        re.compile(r"\b(password|passcode|locked|disabled|login|log\s+in|sign\s+in|verification\s+code)\b", re.IGNORECASE),
        "account credential lockout or login authentication failure",
    ),
    "billing_dispute": (
        re.compile(r"\b(charged|charge|billed|billing|refund|subscription|unauthorized|credit\s+card|receipt)\b", re.IGNORECASE),
        "unauthorized charge, billing discrepancy, or refund request",
    ),
    "wifi_connectivity": (
        re.compile(r"\b(wifi|wi-fi|wireless\s*network)\b", re.IGNORECASE),
        "wireless Wi-Fi or network connectivity failure",
    ),
    "bluetooth_connectivity": (
        re.compile(r"\b(bluetooth|pair|pairing|connect\s+device|carplay)\b", re.IGNORECASE),
        "Bluetooth pairing or peripheral connection failure",
    ),
    "cellular_connectivity": (
        re.compile(r"\b(no\s*service|cellular|lte|4g|5g|sim|dropped\s*calls?|carrier)\b", re.IGNORECASE),
        "cellular reception or carrier connection failure",
    ),
    "software_app_issue": (
        re.compile(r"\b(app\s*store|apps?\s*(?:crashing|won'?t\s*open|freezes|quits)|safari|apple\s*music|spotify)\b", re.IGNORECASE),
        "application malfunction or App Store failure",
    ),
    "sync_backup_issue": (
        re.compile(r"\b(icloud|sync|syncing|synced|photos\s*not\s*syncing|time\s*machine|backup\s*failed)\b", re.IGNORECASE),
        "cloud storage or device synchronization failure",
    ),
    "camera_media_issue": (
        re.compile(r"\b(camera|photos?\b|video|face\s*id|screen\s*recording)\b", re.IGNORECASE),
        "camera or media capture failure",
    ),
    "storage_space_issue": (
        re.compile(r"\b(storage\s*almost\s*full|storage\s*full|not\s*enough\s*storage|out\s*of\s*space)\b", re.IGNORECASE),
        "device or system storage limitation",
    ),
    "notification_issue": (
        re.compile(r"\b(notifications?|alerts?|do\s*not\s*disturb)\b", re.IGNORECASE),
        "notifications or alert delivery failure",
    ),
}

CAUSE_PATTERNS = {
    "software_update": re.compile(
        r"\b(updated|update|updating|after\s+update|since\s+update|after\s+updating|since\s+updating|ios\s*11|ios11|new\s+update|latest\s+update|upgraded)\b",
        re.IGNORECASE,
    ),
    "physical_drop": re.compile(r"\b(dropped|fell|cracked|smashed|water|liquid)\b", re.IGNORECASE),
    "purchase_event": re.compile(r"\b(bought|purchased|subscription|renewal)\b", re.IGNORECASE),
}


class CustomerProblemProfile(BaseModel):
    """Structured understanding of a customer's reported support issue."""
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    device: Optional[str] = Field(default=None, description="Device mentioned (e.g. iPhone, MacBook)")
    product_or_service: Optional[str] = Field(default=None, description="Software service or OS mentioned")
    primary_symptom: str = Field(default="unspecified_device_issue", description="Core operational problem")
    secondary_symptoms: list[str] = Field(default_factory=list, description="Secondary or co-occurring symptoms")
    user_action_or_failure: str = Field(default="", description="Specific failure description")
    possible_cause: Optional[str] = Field(default=None, description="Contextual cause (e.g. iOS update)")
    update_related: bool = Field(default=False, description="Whether an update was cited as contextual cause")
    hardware_related: bool = Field(default=False, description="Whether physical hardware components are affected")
    software_related: bool = Field(default=False, description="Whether software or OS behavior is affected")
    financial_related: bool = Field(default=False, description="Whether financial/billing transactions are involved")
    information_sufficiency: Literal["sufficient", "partial", "insufficient"] = Field(
        default="sufficient",
        description="Whether message contains enough evidence to determine operational intent",
    )
    evidence_summary: str = Field(default="", description="Synthesized factual evidence summary")
    primary_problem_family: OperationalProblemFamily = Field(
        default=OperationalProblemFamily.GENERAL_DEVICE_FUNCTIONALITY,
        description="Core operational category representing the functional failure domain",
    )
    secondary_problem_families: list[OperationalProblemFamily] = Field(
        default_factory=list,
        description="Secondary operational problem categories",
    )
    operational_entities: list[str] = Field(
        default_factory=list,
        description="Key entities extracted (e.g. apps, ports, buttons, services)",
    )
    context_trigger: Optional[str] = Field(
        default=None,
        description="Specific contextual event triggering the problem",
    )
    family_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence score for primary problem family classification",
    )

    def to_dict(self) -> dict[str, Any]:
        data = self.model_dump()
        data["primary_problem_family"] = self.primary_problem_family.value
        data["secondary_problem_families"] = [f.value for f in self.secondary_problem_families]
        return data


class ProblemExtractor:
    """
    Extracts structured problem profiles from customer support inquiries.
    """

    def __init__(
        self,
        llm_client: Optional[BaseLLMClient] = None,
        family_detector: Optional[ProblemFamilyDetector] = None,
    ) -> None:
        self.llm_client = llm_client
        self.family_detector = family_detector or ProblemFamilyDetector()

    def extract_heuristic(self, message: str) -> CustomerProblemProfile:
        """
        Deterministic, rule-guided extraction of problem components.
        """
        text_clean = message.strip()
        text_lower = text_clean.lower()

        # 1. Device Extraction
        detected_device = None
        for dev_name, pat in DEVICE_PATTERNS.items():
            m = pat.search(text_clean)
            if m:
                detected_device = m.group(0).strip()
                # Normalize capitalization
                if detected_device.lower().startswith("iphone"):
                    detected_device = "iPhone" + detected_device[6:]
                elif detected_device.lower().startswith("ipad"):
                    detected_device = "iPad" + detected_device[4:]
                elif detected_device.lower().startswith("macbook"):
                    detected_device = "MacBook" + detected_device[7:]
                else:
                    detected_device = detected_device.title()
                break
        if not detected_device and re.search(r"\b(phone|device)\b", text_lower):
            detected_device = "iPhone"

        # 2. Service / OS Extraction
        detected_service = None
        for srv_name, pat in SERVICE_PATTERNS.items():
            m = pat.search(text_clean)
            if m:
                detected_service = m.group(0).strip()
                break

        # 3. Contextual Cause vs Symptom
        detected_cause = None
        is_update_related = False
        for cause_name, pat in CAUSE_PATTERNS.items():
            if pat.search(text_lower):
                if cause_name == "software_update":
                    detected_cause = "recent software update"
                    is_update_related = True
                elif cause_name == "physical_drop":
                    detected_cause = "physical drop or impact"
                elif cause_name == "purchase_event":
                    detected_cause = "subscription or purchase transaction"
                break

        # 4. Symptoms Extraction
        found_symptoms = []
        for sym_key, (pat, desc) in SYMPTOM_PATTERNS.items():
            if pat.search(text_lower):
                found_symptoms.append((sym_key, desc))

        # Separate primary vs secondary symptom
        primary_symptom = "unspecified_device_issue"
        secondary_symptoms: list[str] = []

        if found_symptoms:
            primary_symptom = found_symptoms[0][1]
            if len(found_symptoms) > 1:
                secondary_symptoms = [s[1] for s in found_symptoms[1:]]
        elif is_update_related:
            primary_symptom = "general post-update instability or sluggishness"

        # 5. Categorical Flags
        is_financial = bool(
            re.search(r"\b(charge|charged|bill|billing|refund|subscription|itunes\s+store|app\s+store\s+purchase)\b", text_lower)
        )
        is_hardware = bool(
            re.search(r"\b(screen|battery|charging|speaker|mic|microphone|audio|hardware|camera|cracked)\b", text_lower)
        )
        is_software = bool(
            is_update_related
            or re.search(r"\b(app|ios|macos|safari|lag|freeze|crash|bug|keyboard|autocorrect|sync)\b", text_lower)
        )

        # 6. Information Sufficiency Evaluation
        words = text_clean.split()
        if len(words) <= 4 and not found_symptoms:
            sufficiency: Literal["sufficient", "partial", "insufficient"] = "insufficient"
        elif primary_symptom == "unspecified_device_issue" and not is_update_related:
            sufficiency = "partial"
        elif is_update_related and primary_symptom == "general post-update instability or sluggishness" and len(words) < 8:
            sufficiency = "partial"
        else:
            sufficiency = "sufficient"

        # 7. Operational Entities Extraction
        entities: list[str] = []
        if detected_device:
            entities.append(detected_device)
        if detected_service:
            entities.append(detected_service)
        # Check specific apps/features
        app_match = re.search(r"\b(safari|itunes|app\s*store|apple\s*music|podcast|mail|spotify|camera|time\s*machine)\b", text_lower)
        if app_match:
            entities.append(app_match.group(0).title())

        # 8. Operational Problem Family Detection
        primary_family, secondary_families, family_conf = self.family_detector.detect_family(
            message=text_clean,
            primary_symptom=primary_symptom,
        )

        # 9. Evidence Summary
        summary_parts = []
        if detected_device:
            summary_parts.append(f"Device: {detected_device}")
        summary_parts.append(f"Family: {primary_family.value}")
        summary_parts.append(f"Primary Symptom: {primary_symptom}")
        if secondary_symptoms:
            summary_parts.append(f"Secondary Symptoms: {', '.join(secondary_symptoms)}")
        if detected_cause:
            summary_parts.append(f"Reported Cause: {detected_cause}")

        evidence_summary = " | ".join(summary_parts)

        return CustomerProblemProfile(
            device=detected_device,
            product_or_service=detected_service,
            primary_symptom=primary_symptom,
            secondary_symptoms=secondary_symptoms,
            user_action_or_failure=text_clean[:120],
            possible_cause=detected_cause,
            update_related=is_update_related,
            hardware_related=is_hardware,
            software_related=is_software,
            financial_related=is_financial,
            information_sufficiency=sufficiency,
            evidence_summary=evidence_summary,
            primary_problem_family=primary_family,
            secondary_problem_families=secondary_families,
            operational_entities=entities,
            context_trigger=detected_cause,
            family_confidence=family_conf,
        )

    def extract(self, message: str) -> CustomerProblemProfile:
        """
        Extract structured problem profile.
        """
        return self.extract_heuristic(message)
