"""
SupportGraph AI — Operational Problem Family Registry (Phase 10.2)

Provides an extensible, corpus-informed taxonomy of operational customer problems.
Decouples evidence relevance from rigid 5-symptom keyword checks and enables
deep multi-dimensional evidence retrieval across all AppleSupport problem families.
"""
from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, ConfigDict, Field

try:
    from app.core.logging import get_logger
except ModuleNotFoundError:
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]

logger = get_logger(__name__)


class OperationalProblemFamily(str, Enum):
    """Corpus-informed operational problem families for Apple customer support."""

    POWER_BATTERY = "POWER_BATTERY"
    CHARGING = "CHARGING"
    AUDIO = "AUDIO"
    DISPLAY = "DISPLAY"
    INPUT_KEYBOARD = "INPUT_KEYBOARD"
    CONNECTIVITY_WIFI = "CONNECTIVITY_WIFI"
    CONNECTIVITY_BLUETOOTH = "CONNECTIVITY_BLUETOOTH"
    NETWORK_CELLULAR = "NETWORK_CELLULAR"
    SOFTWARE_APP = "SOFTWARE_APP"
    SYSTEM_UPDATE = "SYSTEM_UPDATE"
    CRASH_FREEZE = "CRASH_FREEZE"
    PERFORMANCE = "PERFORMANCE"
    ACCOUNT_ACCESS = "ACCOUNT_ACCESS"
    BILLING_PAYMENT = "BILLING_PAYMENT"
    SYNC_BACKUP = "SYNC_BACKUP"
    STORAGE = "STORAGE"
    CAMERA_MEDIA = "CAMERA_MEDIA"
    ACCESSORY_PERIPHERAL = "ACCESSORY_PERIPHERAL"
    NOTIFICATION_ALERTS = "NOTIFICATION_ALERTS"
    GENERAL_DEVICE_FUNCTIONALITY = "GENERAL_DEVICE_FUNCTIONALITY"


class ProblemFamilyDefinition(BaseModel):
    """Operational definition, keywords, related families, and conflict pairs for a problem family."""
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    family: OperationalProblemFamily
    display_name: str
    description: str
    patterns: list[str] = Field(default_factory=list, description="Regex patterns for detection")
    keywords: list[str] = Field(default_factory=list, description="Key operational tokens")
    related_families: list[OperationalProblemFamily] = Field(
        default_factory=list,
        description="Families with overlapping or compatible troubleshooting context",
    )
    conflicting_families: list[OperationalProblemFamily] = Field(
        default_factory=list,
        description="Mutually exclusive families indicating contradictory symptoms",
    )


PROBLEM_FAMILY_DEFINITIONS: dict[OperationalProblemFamily, ProblemFamilyDefinition] = {
    OperationalProblemFamily.POWER_BATTERY: ProblemFamilyDefinition(
        family=OperationalProblemFamily.POWER_BATTERY,
        display_name="Power & Battery",
        description="Rapid battery drain, battery health degradation, overheating, power depletion",
        patterns=[
            r"\b(battery\s*(?:drain|health|life|percentage|dies|dead|draining)|drains?\s*fast|losing\s*charge|overheating)\b",
            r"\b(battery\s*status|maximum\s*capacity|battery\s*service)\b",
        ],
        keywords=["battery", "drain", "draining", "power", "overheat", "overheating", "health"],
        related_families=[
            OperationalProblemFamily.CHARGING,
            OperationalProblemFamily.PERFORMANCE,
            OperationalProblemFamily.SYSTEM_UPDATE,
        ],
        conflicting_families=[
            OperationalProblemFamily.ACCOUNT_ACCESS,
            OperationalProblemFamily.BILLING_PAYMENT,
            OperationalProblemFamily.AUDIO,
            OperationalProblemFamily.DISPLAY,
        ],
    ),
    OperationalProblemFamily.CHARGING: ProblemFamilyDefinition(
        family=OperationalProblemFamily.CHARGING,
        display_name="Charging & Power Input",
        description="Device failure to charge, slow charging, defective charger, port issues",
        patterns=[
            r"\b(won'?t\s*charge|not\s*charging|charges?\s*slowly|unsupported\s*accessory|lightning\s*cable|charger|charging\s*port)\b",
            r"\b(magsafe|wireless\s*charging|plugged\s*in\s*not\s*charging)\b",
        ],
        keywords=["charge", "charging", "charger", "cable", "lightning", "magsafe", "port"],
        related_families=[
            OperationalProblemFamily.POWER_BATTERY,
            OperationalProblemFamily.ACCESSORY_PERIPHERAL,
        ],
        conflicting_families=[
            OperationalProblemFamily.ACCOUNT_ACCESS,
            OperationalProblemFamily.BILLING_PAYMENT,
            OperationalProblemFamily.AUDIO,
        ],
    ),
    OperationalProblemFamily.AUDIO: ProblemFamilyDefinition(
        family=OperationalProblemFamily.AUDIO,
        display_name="Audio & Sound",
        description="Speaker, microphone, volume, distorted sound, crackling, AirPods audio",
        patterns=[
            r"\b(speaker|microphone|mic\b|volume|sound|crackling|distortion|no\s*sound|can'?t\s*hear|audio\s*(?:issue|problem|glitch|quality|cutting)|earpiece|headphones?|airpods?\s*(?:sound|mic|audio))\b",
            r"\b(call\s*volume|speakerphone|ringer|buzzing\s*sound)\b",
        ],
        keywords=["speaker", "sound", "volume", "mic", "microphone", "audio", "earpiece", "airpods"],
        related_families=[
            OperationalProblemFamily.CONNECTIVITY_BLUETOOTH,
            OperationalProblemFamily.ACCESSORY_PERIPHERAL,
        ],
        conflicting_families=[
            OperationalProblemFamily.DISPLAY,
            OperationalProblemFamily.INPUT_KEYBOARD,
            OperationalProblemFamily.BILLING_PAYMENT,
            OperationalProblemFamily.ACCOUNT_ACCESS,
        ],
    ),
    OperationalProblemFamily.DISPLAY: ProblemFamilyDefinition(
        family=OperationalProblemFamily.DISPLAY,
        display_name="Display & Screen",
        description="Cracked screen, black screen, touch responsiveness, flickering, lines on display",
        patterns=[
            r"\b(screen|display|touch\s*screen|digitizer|cracked\s*screen|black\s*screen|blank\s*screen|flicker|flickering|lines\s*on\s*screen|unresponsive\s*touch)\b",
            r"\b(screen\s*(?:glitch|dim|burn|tint|frozen|freeze))\b",
        ],
        keywords=["screen", "display", "touch", "cracked", "black screen", "flicker", "digitizer"],
        related_families=[
            OperationalProblemFamily.CRASH_FREEZE,
            OperationalProblemFamily.SYSTEM_UPDATE,
        ],
        conflicting_families=[
            OperationalProblemFamily.AUDIO,
            OperationalProblemFamily.BILLING_PAYMENT,
            OperationalProblemFamily.ACCOUNT_ACCESS,
            OperationalProblemFamily.CONNECTIVITY_WIFI,
        ],
    ),
    OperationalProblemFamily.INPUT_KEYBOARD: ProblemFamilyDefinition(
        family=OperationalProblemFamily.INPUT_KEYBOARD,
        display_name="Keyboard & Typing Input",
        description="Keyboard typing anomalies, autocorrect glitch, letter 'i' bug, predictive text",
        patterns=[
            r"\b(keyboard|typing|autocorrect|autocorrection|predictive\s*text|letter\s*i\b|capital\s*i\b|typing\s*lag|keys?\s*stuck|virtual\s*keyboard)\b",
            r"\b(question\s*mark\s*(?:box|symbol)|weird\s*symbol\s*instead\s*of\s*i)\b",
        ],
        keywords=["keyboard", "typing", "type", "autocorrect", "letter i", "predictive", "keys"],
        related_families=[
            OperationalProblemFamily.PERFORMANCE,
            OperationalProblemFamily.SYSTEM_UPDATE,
        ],
        conflicting_families=[
            OperationalProblemFamily.AUDIO,
            OperationalProblemFamily.POWER_BATTERY,
            OperationalProblemFamily.BILLING_PAYMENT,
        ],
    ),
    OperationalProblemFamily.CONNECTIVITY_WIFI: ProblemFamilyDefinition(
        family=OperationalProblemFamily.CONNECTIVITY_WIFI,
        display_name="Wi-Fi Connectivity",
        description="Wi-Fi disconnecting, cannot connect to wireless network, Wi-Fi turning on automatically, grayed out toggle",
        patterns=[
            r"\b(wi-?fi|wireless\s*network|wifi\s*drops?|wifi\s*keeps?\s*disconnecting|can'?t\s*connect\s*to\s*wi-?fi|wi-?fi\s*turns?\s*on|wi-?fi\s*grayed?\s*out|slow\s*wi-?fi)\b",
            r"\b(control\s*center\s*wi-?fi|wifi\s*toggle|no\s*internet\s*connection)\b",
        ],
        keywords=["wifi", "wi-fi", "wireless", "network", "connect", "disconnect", "hotspot"],
        related_families=[
            OperationalProblemFamily.NETWORK_CELLULAR,
            OperationalProblemFamily.CONNECTIVITY_BLUETOOTH,
            OperationalProblemFamily.SYSTEM_UPDATE,
        ],
        conflicting_families=[
            OperationalProblemFamily.AUDIO,
            OperationalProblemFamily.BILLING_PAYMENT,
            OperationalProblemFamily.DISPLAY,
        ],
    ),
    OperationalProblemFamily.CONNECTIVITY_BLUETOOTH: ProblemFamilyDefinition(
        family=OperationalProblemFamily.CONNECTIVITY_BLUETOOTH,
        display_name="Bluetooth Connectivity",
        description="Bluetooth pairing failures, car audio disconnecting, peripheral connection loss",
        patterns=[
            r"\b(bluetooth|bt\s*connection|bluetooth\s*pairing|pair\s*(?:with|device)|bluetooth\s*drops?|carplay|won'?t\s*pair)\b",
            r"\b(bluetooth\s*(?:connection|audio|not\s*connecting|issues?|turned\s*on|toggle|device|disconnects?))\b",
            r"\b(pairing\s*unsuccessful|car\s*audio\s*bluetooth|bluetooth\s*in\s*car)\b",
        ],
        keywords=["bluetooth", "pair", "pairing", "connect", "carplay", "disconnect"],
        related_families=[
            OperationalProblemFamily.CONNECTIVITY_WIFI,
            OperationalProblemFamily.ACCESSORY_PERIPHERAL,
            OperationalProblemFamily.AUDIO,
        ],
        conflicting_families=[
            OperationalProblemFamily.BILLING_PAYMENT,
            OperationalProblemFamily.ACCOUNT_ACCESS,
        ],
    ),
    OperationalProblemFamily.NETWORK_CELLULAR: ProblemFamilyDefinition(
        family=OperationalProblemFamily.NETWORK_CELLULAR,
        display_name="Cellular & Carrier Network",
        description="No service, searching for signal, dropped phone calls, cellular data failure, SMS/group text issues",
        patterns=[
            r"\b(no\s*service|searching\s*for\s*signal|cellular\s*data|mobile\s*data|dropped\s*calls?|lte\b|3g\b|4g\b|5g\b|sim\s*failure|invalid\s*sim)\b",
            r"\b(group\s*texts?|sms\s*not\s*sending|cannot\s*make\s*calls?|carrier\s*settings)\b",
        ],
        keywords=["cellular", "no service", "lte", "signal", "carrier", "sim", "calls", "data"],
        related_families=[
            OperationalProblemFamily.CONNECTIVITY_WIFI,
            OperationalProblemFamily.SYSTEM_UPDATE,
        ],
        conflicting_families=[
            OperationalProblemFamily.AUDIO,
            OperationalProblemFamily.BILLING_PAYMENT,
        ],
    ),
    OperationalProblemFamily.SOFTWARE_APP: ProblemFamilyDefinition(
        family=OperationalProblemFamily.SOFTWARE_APP,
        display_name="Software & App Issues",
        description="App Store downloads/updates failing, third-party apps crashing, Safari errors, Apple Music playback issues",
        patterns=[
            r"\b(app\s*store|apps?\s*(?:crashing|won'?t\s*open|freezes|quits|bugged)|can'?t\s*download\s*apps?|update\s*apps?)\b",
            r"\b(safari|apple\s*music|podcast\s*app|spotify\s*on\s*iphone|mail\s*app|blank\s*screen\s*on\s*safari|app\s*closing)\b",
        ],
        keywords=["app", "apps", "app store", "safari", "music", "crash", "download", "install"],
        related_families=[
            OperationalProblemFamily.CRASH_FREEZE,
            OperationalProblemFamily.SYSTEM_UPDATE,
            OperationalProblemFamily.PERFORMANCE,
        ],
        conflicting_families=[
            OperationalProblemFamily.CHARGING,
            OperationalProblemFamily.BILLING_PAYMENT,
        ],
    ),
    OperationalProblemFamily.SYSTEM_UPDATE: ProblemFamilyDefinition(
        family=OperationalProblemFamily.SYSTEM_UPDATE,
        display_name="System & OS Update",
        description="Update installation failures, post-update system instability, iOS/macOS version issues",
        patterns=[
            r"\b(after\s*(?:the\s*)?update|since\s*(?:the\s*)?update|new\s*update|ios\s*11|ios11|high\s*sierra|macos\s*update|failed\s*update)\b",
            r"\b(update\s*(?:stuck|bricked|ruined|buggy|glitchy)|can'?t\s*install\s*update|verifying\s*update)\b",
        ],
        keywords=["update", "updated", "updating", "ios", "high sierra", "macos", "upgrade", "version"],
        related_families=[
            OperationalProblemFamily.CRASH_FREEZE,
            OperationalProblemFamily.PERFORMANCE,
            OperationalProblemFamily.POWER_BATTERY,
            OperationalProblemFamily.CONNECTIVITY_WIFI,
        ],
        conflicting_families=[
            OperationalProblemFamily.BILLING_PAYMENT,
            OperationalProblemFamily.ACCOUNT_ACCESS,
        ],
    ),
    OperationalProblemFamily.CRASH_FREEZE: ProblemFamilyDefinition(
        family=OperationalProblemFamily.CRASH_FREEZE,
        display_name="Crashes, Freezing & Reboot",
        description="Device frozen, spinning beach ball, boot loop, random restarts, Apple logo stuck",
        patterns=[
            r"\b(freeze|freezing|frozen|spinning\s*beach\s*ball|beach\s*ball|boot\s*loop|restarts?\s*randomly|stuck\s*on\s*apple\s*logo)\b",
            r"\b(device\s*bricked|force\s*restart|system\s*crash|keeps?\s*crashing|black\s*screen\s*spinning\s*wheel)\b",
        ],
        keywords=["freeze", "frozen", "crash", "beach ball", "restart", "reboot", "stuck", "logo"],
        related_families=[
            OperationalProblemFamily.PERFORMANCE,
            OperationalProblemFamily.SYSTEM_UPDATE,
            OperationalProblemFamily.DISPLAY,
        ],
        conflicting_families=[
            OperationalProblemFamily.BILLING_PAYMENT,
            OperationalProblemFamily.ACCOUNT_ACCESS,
        ],
    ),
    OperationalProblemFamily.PERFORMANCE: ProblemFamilyDefinition(
        family=OperationalProblemFamily.PERFORMANCE,
        display_name="Performance & Responsiveness",
        description="Device slow, lagging, sluggish animation, delay when typing or opening apps",
        patterns=[
            r"\b(phone\s*slow|device\s*slow|lagging|extremely\s*laggy|so\s*slow|sluggish|system\s*delay|delayed\s*response)\b",
            r"\b(takes\s*forever\s*to\s*load|performance\s*dropped|laggy\s*interface)\b",
        ],
        keywords=["slow", "lag", "laggy", "sluggish", "delay", "performance", "speed"],
        related_families=[
            OperationalProblemFamily.SYSTEM_UPDATE,
            OperationalProblemFamily.CRASH_FREEZE,
            OperationalProblemFamily.POWER_BATTERY,
        ],
        conflicting_families=[
            OperationalProblemFamily.BILLING_PAYMENT,
            OperationalProblemFamily.ACCOUNT_ACCESS,
        ],
    ),
    OperationalProblemFamily.ACCOUNT_ACCESS: ProblemFamilyDefinition(
        family=OperationalProblemFamily.ACCOUNT_ACCESS,
        display_name="Account, Apple ID & Authentication",
        description="Locked Apple ID, forgotten password, two-factor authentication, login failure",
        patterns=[
            r"\b(apple\s*id|password|passcode|locked\s*account|disabled\s*account|sign\s*in|login|log\s*in|two-?factor|2fa|verification\s*code)\b",
            r"\b(iforgot|reset\s*password|security\s*questions|account\s*recovery)\b",
        ],
        keywords=["apple id", "password", "locked", "login", "sign in", "2fa", "verification", "account"],
        related_families=[
            OperationalProblemFamily.BILLING_PAYMENT,
            OperationalProblemFamily.SYNC_BACKUP,
        ],
        conflicting_families=[
            OperationalProblemFamily.AUDIO,
            OperationalProblemFamily.DISPLAY,
            OperationalProblemFamily.CHARGING,
            OperationalProblemFamily.CONNECTIVITY_WIFI,
        ],
    ),
    OperationalProblemFamily.BILLING_PAYMENT: ProblemFamilyDefinition(
        family=OperationalProblemFamily.BILLING_PAYMENT,
        display_name="Billing, Subscriptions & Purchases",
        description="Unauthorized charges, subscription renewal/cancellation, refund requests, iTunes receipts",
        patterns=[
            r"\b(charged|charge|billed|billing|refund|subscription|cancel\s*subscription|itunes\s*store\s*bill|reportaproblem|credit\s*card)\b",
            r"\b(unauthorized\s*purchase|receipt|accidental\s*purchase|payment\s*method)\b",
        ],
        keywords=["charge", "charged", "billing", "bill", "refund", "subscription", "purchase", "itunes"],
        related_families=[
            OperationalProblemFamily.ACCOUNT_ACCESS,
            OperationalProblemFamily.SOFTWARE_APP,
        ],
        conflicting_families=[
            OperationalProblemFamily.AUDIO,
            OperationalProblemFamily.DISPLAY,
            OperationalProblemFamily.CHARGING,
            OperationalProblemFamily.CONNECTIVITY_WIFI,
            OperationalProblemFamily.POWER_BATTERY,
        ],
    ),
    OperationalProblemFamily.SYNC_BACKUP: ProblemFamilyDefinition(
        family=OperationalProblemFamily.SYNC_BACKUP,
        display_name="iCloud, Sync & Backup",
        description="Photos not syncing, iCloud backup failing, Time Machine notifications, cross-device sync",
        patterns=[
            r"\b(icloud|sync|syncing|synced|photos\s*not\s*syncing|time\s*machine|backup\s*failed|restore\s*from\s*backup)\b",
            r"\b(cloud\s*backup|icloud\s*storage|sync\s*between\s*(?:mac|iphone)|not\s*backing\s*up)\b",
        ],
        keywords=["icloud", "sync", "backup", "photos", "time machine", "restore", "cloud"],
        related_families=[
            OperationalProblemFamily.STORAGE,
            OperationalProblemFamily.ACCOUNT_ACCESS,
        ],
        conflicting_families=[
            OperationalProblemFamily.AUDIO,
            OperationalProblemFamily.CHARGING,
            OperationalProblemFamily.BILLING_PAYMENT,
        ],
    ),
    OperationalProblemFamily.STORAGE: ProblemFamilyDefinition(
        family=OperationalProblemFamily.STORAGE,
        display_name="Device & Cloud Storage",
        description="Storage full, system/other storage taking up space, cannot install due to insufficient memory",
        patterns=[
            r"\b(storage\s*almost\s*full|storage\s*full|not\s*enough\s*storage|system\s*storage|other\s*storage|free\s*up\s*space|disk\s*space)\b",
            r"\b(insufficient\s*storage|manage\s*storage)\b",
        ],
        keywords=["storage", "space", "full", "disk", "gigabytes", "memory", "clean"],
        related_families=[
            OperationalProblemFamily.SYNC_BACKUP,
            OperationalProblemFamily.SYSTEM_UPDATE,
            OperationalProblemFamily.PERFORMANCE,
        ],
        conflicting_families=[
            OperationalProblemFamily.AUDIO,
            OperationalProblemFamily.CHARGING,
        ],
    ),
    OperationalProblemFamily.CAMERA_MEDIA: ProblemFamilyDefinition(
        family=OperationalProblemFamily.CAMERA_MEDIA,
        display_name="Camera, Photos & Recording",
        description="Camera black screen, blurry camera, flash not working, screen recording failure, Face ID",
        patterns=[
            r"\b(camera|photos?\s*blurry|black\s*camera|camera\s*not\s*working|flash\s*not\s*working|face\s*id|face\s*recognition|screen\s*recording)\b",
            r"\b(front\s*camera|rear\s*camera|shutter|take\s*photos?)\b",
        ],
        keywords=["camera", "photos", "photo", "screen recording", "face id", "video", "flash"],
        related_families=[
            OperationalProblemFamily.DISPLAY,
            OperationalProblemFamily.SOFTWARE_APP,
        ],
        conflicting_families=[
            OperationalProblemFamily.BILLING_PAYMENT,
            OperationalProblemFamily.ACCOUNT_ACCESS,
            OperationalProblemFamily.INPUT_KEYBOARD,
        ],
    ),
    OperationalProblemFamily.ACCESSORY_PERIPHERAL: ProblemFamilyDefinition(
        family=OperationalProblemFamily.ACCESSORY_PERIPHERAL,
        display_name="Accessories & Peripherals",
        description="Apple Watch pairing/sync, Apple Pencil, dongles, adapters, external display connection",
        patterns=[
            r"\b(apple\s*watch|iwatch|apple\s*pencil|airtags?|dongle|adapter|external\s*monitor|headphone\s*jack\s*adapter)\b",
            r"\b(watch\s*sync|watchos|accessory\s*not\s*supported)\b",
        ],
        keywords=["apple watch", "pencil", "accessory", "adapter", "dongle", "watch", "airpods"],
        related_families=[
            OperationalProblemFamily.CONNECTIVITY_BLUETOOTH,
            OperationalProblemFamily.CHARGING,
            OperationalProblemFamily.AUDIO,
        ],
        conflicting_families=[
            OperationalProblemFamily.BILLING_PAYMENT,
            OperationalProblemFamily.ACCOUNT_ACCESS,
        ],
    ),
    OperationalProblemFamily.NOTIFICATION_ALERTS: ProblemFamilyDefinition(
        family=OperationalProblemFamily.NOTIFICATION_ALERTS,
        display_name="Notifications & Alerts",
        description="Notifications not showing, no sound on alerts, Do Not Disturb issues, badge count errors",
        patterns=[
            r"\b(notifications?|alerts?|no\s*notification\s*sound|badge\s*count|do\s*not\s*disturb|banner\s*notifications?)\b",
            r"\b(missed\s*notifications?|notification\s*center)\b",
        ],
        keywords=["notification", "notifications", "alert", "alerts", "banner", "do not disturb", "badge"],
        related_families=[
            OperationalProblemFamily.AUDIO,
            OperationalProblemFamily.SOFTWARE_APP,
            OperationalProblemFamily.SYSTEM_UPDATE,
        ],
        conflicting_families=[
            OperationalProblemFamily.BILLING_PAYMENT,
            OperationalProblemFamily.CHARGING,
        ],
    ),
    OperationalProblemFamily.GENERAL_DEVICE_FUNCTIONALITY: ProblemFamilyDefinition(
        family=OperationalProblemFamily.GENERAL_DEVICE_FUNCTIONALITY,
        display_name="General Device Functionality",
        description="Broad device support, device settings, basic questions, general operational assistance",
        patterns=[
            r"\b(help\s*with\s*(?:my\s*)?device|phone\s*issue|general\s*question|how\s*do\s*i|settings)\b",
        ],
        keywords=["help", "phone", "device", "question", "support"],
        related_families=list(OperationalProblemFamily),
        conflicting_families=[],
    ),
}


class ProblemFamilyDetector:
    """Detects operational problem families from customer inquiries and problem profiles."""

    def __init__(self) -> None:
        self._compiled_patterns: dict[OperationalProblemFamily, list[re.Pattern]] = {}
        for family, defn in PROBLEM_FAMILY_DEFINITIONS.items():
            self._compiled_patterns[family] = [
                re.compile(p, re.IGNORECASE) for p in defn.patterns
            ]

    def detect_family(
        self,
        message: str,
        primary_symptom: str = "",
        candidate_intent: str = "",
    ) -> tuple[OperationalProblemFamily, list[OperationalProblemFamily], float]:
        """
        Detect the primary and secondary operational problem families for a query.

        Returns:
            (primary_family, secondary_families, confidence)
        """
        combined = f"{message} {primary_symptom}".lower()
        scored_families: list[tuple[OperationalProblemFamily, float]] = []

        # Intent heuristic baseline
        intent_prior_map: dict[str, OperationalProblemFamily] = {
            "battery_power_issue": OperationalProblemFamily.POWER_BATTERY,
            "hardware_audio_connection_issue": OperationalProblemFamily.AUDIO,
            "display_touch_issue": OperationalProblemFamily.DISPLAY,
            "keyboard_typing_issue": OperationalProblemFamily.INPUT_KEYBOARD,
            "account_access_issue": OperationalProblemFamily.ACCOUNT_ACCESS,
            "billing_purchase_issue": OperationalProblemFamily.BILLING_PAYMENT,
            "software_update_problem": OperationalProblemFamily.SYSTEM_UPDATE,
            "mac_software_issue": OperationalProblemFamily.SOFTWARE_APP,
        }

        # Step 1: Score patterns
        for family, patterns in self._compiled_patterns.items():
            if family == OperationalProblemFamily.GENERAL_DEVICE_FUNCTIONALITY:
                continue

            score = 0.0
            defn = PROBLEM_FAMILY_DEFINITIONS[family]

            # Regex pattern matches (strongest signal)
            for pat in patterns:
                matches = pat.findall(combined)
                if matches:
                    score += 2.0 * len(matches)

            # Keyword matches
            for kw in defn.keywords:
                if re.search(r"\b" + re.escape(kw) + r"\b", combined):
                    score += 0.5

            # Prior boost from existing intent if aligned
            if candidate_intent and intent_prior_map.get(candidate_intent) == family:
                score += 1.0

            if score > 0.0:
                scored_families.append((family, score))

        # Sort by score descending
        scored_families.sort(key=lambda x: x[1], reverse=True)

        if not scored_families:
            # Fallback to intent prior if available
            if candidate_intent in intent_prior_map:
                primary = intent_prior_map[candidate_intent]
                return (primary, [], 0.50)
            return (OperationalProblemFamily.GENERAL_DEVICE_FUNCTIONALITY, [], 0.30)

        top_family, top_score = scored_families[0]
        secondaries = [f for f, s in scored_families[1:4] if s >= 1.0]

        # Calculate calibrated confidence
        confidence = min(0.98, max(0.50, round(top_score / (top_score + 2.0), 2)))
        return (top_family, secondaries, confidence)

    def are_conflicting(
        self,
        family_a: OperationalProblemFamily,
        family_b: OperationalProblemFamily,
    ) -> bool:
        """Check whether two problem families represent mutually exclusive operational problems."""
        if (
            family_a == OperationalProblemFamily.GENERAL_DEVICE_FUNCTIONALITY
            or family_b == OperationalProblemFamily.GENERAL_DEVICE_FUNCTIONALITY
        ):
            return False
        defn_a = PROBLEM_FAMILY_DEFINITIONS.get(family_a)
        if defn_a and family_b in defn_a.conflicting_families:
            return True
        defn_b = PROBLEM_FAMILY_DEFINITIONS.get(family_b)
        if defn_b and family_a in defn_b.conflicting_families:
            return True
        return False

    def are_compatible(
        self,
        family_a: OperationalProblemFamily,
        family_b: OperationalProblemFamily,
    ) -> bool:
        """Check whether two problem families are identical or troubleshooting-compatible."""
        if family_a == family_b:
            return True
        defn_a = PROBLEM_FAMILY_DEFINITIONS.get(family_a)
        if defn_a and family_b in defn_a.related_families:
            return True
        defn_b = PROBLEM_FAMILY_DEFINITIONS.get(family_b)
        if defn_b and family_a in defn_b.related_families:
            return True
        return False
