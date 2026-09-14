"""
SupportGraph AI — Taxonomy Calibration & Recommendation Layer (Phase 5.9)

Learns transparent, explainable operational signals from human ground-truth decisions:
- Extracts contextual taxonomy evidence without training black-box ML models
- Generates separate calibrated recommendations for pending records
- Identifies AI-human agreement, disagreement, and ambiguity
- Enhances risk-based HITL queue prioritization
- Strictly preserves ground-truth immutability and prediction separation.
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

try:
    from app.core.logging import get_logger
    from app.evaluation.human_review import (
        get_allowed_taxonomy_labels,
        is_valid_ai_suggestion,
    )
    from app.nlp.taxonomy_finalization import (
        DEFAULT_CANDIDATE_TAXONOMY_PATH,
        load_candidate_taxonomy,
    )
except ModuleNotFoundError:
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]
    from backend.app.evaluation.human_review import (  # type: ignore[no-redef]
        get_allowed_taxonomy_labels,
        is_valid_ai_suggestion,
    )
    from backend.app.nlp.taxonomy_finalization import (  # type: ignore[no-redef]
        DEFAULT_CANDIDATE_TAXONOMY_PATH,
        load_candidate_taxonomy,
    )

logger = get_logger(__name__)


class CalibrationStatus(str, Enum):
    """Status of taxonomy calibration compared to AI suggestion."""
    AGREES_WITH_AI = "AGREES_WITH_AI"
    DISAGREES_WITH_AI = "DISAGREES_WITH_AI"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    AMBIGUOUS = "AMBIGUOUS"


@dataclass
class CalibrationSignal:
    """A transparent, contextual evidence signal extracted from customer message/dialogue."""
    name: str
    target_intent: str
    evidence_text: str
    weight: float
    is_negative: bool = False
    description: str = ""

    @property
    def category(self) -> str:
        return self.target_intent

    @property
    def matched_phrases(self) -> list[str]:
        return [self.evidence_text]


@dataclass
class CalibrationResult:
    """Structured calibrated recommendation and evidence assessment for a record."""
    candidate_label: Optional[str]
    calibration_confidence: float
    supporting_signals: list[str] = field(default_factory=list)
    conflicting_signals: list[str] = field(default_factory=list)
    calibration_status: CalibrationStatus = CalibrationStatus.INSUFFICIENT_EVIDENCE
    reasoning: str = ""
    abstained: bool = False

    @property
    def status(self) -> CalibrationStatus:
        return self.calibration_status

    @status.setter
    def status(self, val: CalibrationStatus) -> None:
        self.calibration_status = val

    @property
    def confidence(self) -> float:
        return self.calibration_confidence

    @confidence.setter
    def confidence(self, val: float) -> None:
        self.calibration_confidence = val

    def to_dict(self) -> dict[str, Any]:
        return {
            "calibration_candidate_label": self.candidate_label or "",
            "calibration_confidence": round(self.calibration_confidence, 2),
            "calibration_supporting_signals": "; ".join(self.supporting_signals) if self.supporting_signals else "none",
            "calibration_conflicting_signals": "; ".join(self.conflicting_signals) if self.conflicting_signals else "none",
            "calibration_status": self.calibration_status.value,
            "calibration_reasoning": self.reasoning,
            "calibration_abstained": str(self.abstained),
        }


# ---------------------------------------------------------------------------
# Transparent Contextual Signal Patterns
# ---------------------------------------------------------------------------

SIGNAL_RULES: list[dict[str, Any]] = [
    # -----------------------------------------------------------------------
    # 1. SOFTWARE UPDATE PROBLEM
    # -----------------------------------------------------------------------
    {
        "intent": "software_update_problem",
        "signal_name": "update_causality",
        "pattern": r"\b(since|after|ever since|installed|updated to|updating to)\s+(the\s+)?(new\s+|latest\s+)?(update|updating|ios(\s*11(\.\d+)?)?|high sierra|macos|software update)\b",
        "weight": 0.88,
        "description": "Explicit temporal causality linking problem onset to a software update",
    },
    {
        "intent": "software_update_problem",
        "signal_name": "update_installation_failure",
        "pattern": r"\b(update won't download|unable to verify update|update stuck|install failed|cannot update|error downloading update|update error|high sierra download)\b",
        "weight": 0.90,
        "description": "Technical failure during software update download or verification",
    },
    {
        "intent": "software_update_problem",
        "signal_name": "general_update_bug_complaint",
        "pattern": r"\b(ios\s*11\s+(is|has)\s+(full of|so many|a lot of)?\s*bugs|glitched out mess|update ruined|buggy update|fix (the|this) update|since update|issue after the update)\b",
        "weight": 0.85,
        "description": "General user complaint about system-wide instability introduced by a named update",
    },

    # -----------------------------------------------------------------------
    # 2. HARDWARE AUDIO & CONNECTION ISSUE
    # -----------------------------------------------------------------------
    {
        "intent": "hardware_audio_connection_issue",
        "signal_name": "audio_sound_malfunction",
        "pattern": r"\b(no sound|speakers? (is |are )?(not working|crackling|distorted|broken)|sound cuts out|no audio|microphone (is |are )?(not working|broken)|speaker volume|music (playing|plays) but no sound|audio glitch)\b",
        "weight": 0.88,
        "description": "Physical speaker, microphone, or sound output malfunction",
    },
    {
        "intent": "hardware_audio_connection_issue",
        "signal_name": "peripheral_hardware_connection",
        "pattern": r"\b(airpods|earphones?|earbuds?|headphones?|wired printer|usb(-|\s*)c hub|usb female|c-type to usb|lightning (adapter|dongle)|headphone jack|aux cable|connect.*printer|external display adapter)\b",
        "weight": 0.88,
        "description": "Physical peripheral cable, audio accessory, dongle, hub, or hardware port inquiry",
    },
    {
        "intent": "hardware_audio_connection_issue",
        "signal_name": "bluetooth_headphone_disconnect",
        "pattern": r"\b(airpods (won't connect|disconnecting|no sound|one side)|bluetooth (disconnect|won't pair|audio))\b",
        "weight": 0.85,
        "description": "Wireless audio / Bluetooth peripheral pairing or sound failure",
    },

    # -----------------------------------------------------------------------
    # 3. KEYBOARD & TYPING ISSUE
    # -----------------------------------------------------------------------
    {
        "intent": "keyboard_typing_issue",
        "signal_name": "letter_i_bug",
        "pattern": r"\b(letter\s+['\"]?i['\"]?|letter\s+eye|autocorrect\s+(changing|glitch|bug|changing is to)|auto correct|unicode (box|symbol)|question mark in (a )?box|fix this i\b|crap when (i )?type|capital i|word problem,?\s+auto\s*correct)\b",
        "weight": 0.92,
        "description": "Specific iOS autocorrect rendering glitch affecting text input (e.g. letter 'I' symbol)",
    },
    {
        "intent": "keyboard_typing_issue",
        "signal_name": "text_input_issue",
        "pattern": r"\b(autocorrect|can't type|cannot type|keyboard (won't appear|freez|lag|slow|not working)|predictive text (glitch|wrong)|keys not responding|type to group texts|typing)\b",
        "weight": 0.88,
        "description": "Keyboard interface unresponsiveness or text prediction malfunction",
    },

    # -----------------------------------------------------------------------
    # 4. BATTERY & POWER ISSUE
    # -----------------------------------------------------------------------
    {
        "intent": "battery_power_issue",
        "signal_name": "rapid_battery_drain",
        "pattern": r"\b(battery(\s+\w+)?\s+(is\s+|are\s+)?(draining|drain|dies|dead|drains|low|dying|depleting|percentage|problem)|(draining|drain|drains|depleting)\s+(\w+\s+)?(fast|quickly)|battery life|battery health|running out of battery|percent(age)?\s+in\s+\d+\s+min)\b",
        "weight": 0.92,
        "description": "Rapid battery depletion or power consumption complaint",
    },
    {
        "intent": "battery_power_issue",
        "signal_name": "power_charging_failure",
        "pattern": r"\b(won't\s+turn\s+on|won't\s+power\s+on|won't\s+charge|not\s+charging|take\s+.*to\s+charge|refuses\s+to\s+charge|charger\s+(broke|sucks|not recognized|hot)|laptop\s+charger|shutting\s+down|overheating)\b",
        "weight": 0.92,
        "description": "Device failure to boot, charge, or maintain power",
    },

    # -----------------------------------------------------------------------
    # 5. MAC SOFTWARE ISSUE
    # -----------------------------------------------------------------------
    {
        "intent": "mac_software_issue",
        "signal_name": "macos_system_subsystem_behavior",
        "pattern": r"\b(time machine (notifications|backup|error)|spinning beach ball|macbook (freeze|frozen|slow|hang|boot)|macos recovery|kernel panic|finder (freeze|not responding|crashing)|safari crashes|os x (freeze|crash)|high sierra|reinstalled macos|formated and reinstalled)\b",
        "weight": 0.88,
        "description": "Mac-specific operating system, Time Machine, or macOS desktop application failure",
    },

    # -----------------------------------------------------------------------
    # 6. DISPLAY & TOUCH ISSUE
    # -----------------------------------------------------------------------
    {
        "intent": "display_touch_issue",
        "signal_name": "screen_touch_hardware_failure",
        "pattern": r"\b(touch\s*screen|touchscreen (is |not |unresponsive)|screen (is )?(cracked|broken|black|flicker|flickering|shattered|lines)|touch (not responding|unresponsive|stopped working)|ghost touch|digitizer)\b",
        "weight": 0.90,
        "description": "Physical display panel, backlight, or touch digitizer hardware failure",
    },

    # -----------------------------------------------------------------------
    # 7. ACCOUNT ACCESS ISSUE
    # -----------------------------------------------------------------------
    {
        "intent": "account_access_issue",
        "signal_name": "apple_id_auth_lock",
        "pattern": r"\b(apple\s*id (locked|disabled|password)|forgot (my )?password|two(-|\s*)factor (authentication|code)|verification code (not receiving|failed)|icloud (login|locked)|security questions|locked out of (my )?apple\s*id)\b",
        "weight": 0.92,
        "description": "Apple ID account credentials, two-factor authentication, or security lockout",
    },

    # -----------------------------------------------------------------------
    # 8. BILLING & PURCHASE ISSUE
    # -----------------------------------------------------------------------
    {
        "intent": "billing_purchase_issue",
        "signal_name": "payment_charges_dispute",
        "pattern": r"\b(unauthorized charge|charged twice|refund|itunes (receipt|bill|charge)|subscription charge|credit card declined|apple pay charge|billed for app)\b",
        "weight": 0.92,
        "description": "Financial transaction, unauthorized debit, subscription billing, or refund request",
    },
]


def extract_taxonomy_signals(text: str) -> list[CalibrationSignal]:
    """
    Extract all matching transparent calibration signals from customer message.
    """
    if not text or not text.strip():
        return []

    cleaned = text.strip()
    # Normalize extra whitespace
    normalized = " ".join(cleaned.split())

    extracted: list[CalibrationSignal] = []

    for rule in SIGNAL_RULES:
        pat = rule["pattern"]
        match = re.search(pat, normalized, flags=re.IGNORECASE)
        if match:
            sig = CalibrationSignal(
                name=rule["signal_name"],
                target_intent=rule["intent"],
                evidence_text=match.group(0),
                weight=rule["weight"],
                is_negative=False,
                description=rule.get("description", ""),
            )
            extracted.append(sig)

    return extracted


def evaluate_taxonomy_calibration(
    record: dict[str, Any] | pd.Series,
    model_suggested_label: Optional[str] = None,
    model_confidence: Optional[float] = None,
    allowed_labels: Optional[set[str]] = None,
) -> CalibrationResult:
    """
    Generate an explainable, calibrated taxonomy recommendation for a golden record.

    Evidence Fusion:
    1. Extracts transparent contextual signals from customer_message.
    2. Sums signal weights per candidate intent.
    3. Evaluates dominant intent vs competing intents.
    4. Compares calibrated candidate against AI suggestion (AGREES / DISAGREES / INSUFFICIENT / AMBIGUOUS).
    5. Never forces a prediction if evidence is weak or conflicting.
    """
    if allowed_labels is None:
        allowed_labels = get_allowed_taxonomy_labels()

    cust_msg = str(record.get("customer_message", "")).strip()
    raw_norm = str(record.get("normalized_message", "")).strip()
    norm_msg = raw_norm if (raw_norm and (not cust_msg or cust_msg in raw_norm)) else cust_msg

    raw_ai_label = model_suggested_label if model_suggested_label is not None else record.get("model_suggested_label")
    ai_label = str(raw_ai_label).strip() if raw_ai_label is not None and not pd.isna(raw_ai_label) else ""
    if ai_label.lower() in ("nan", "none", "null", "undefined"):
        ai_label = ""

    raw_ai_conf = model_confidence if model_confidence is not None else record.get("model_confidence")
    ai_conf: float = 0.0
    if raw_ai_conf is not None and not pd.isna(raw_ai_conf) and str(raw_ai_conf).strip():
        try:
            ai_conf = float(raw_ai_conf)
        except (ValueError, TypeError):
            ai_conf = 0.0

    # Extract signals
    signals = extract_taxonomy_signals(norm_msg)

    # If no specific signals matched
    if not signals:
        # Check if message is too sparse
        words = norm_msg.split()
        if len(norm_msg) < 15 or len(words) <= 2:
            return CalibrationResult(
                candidate_label="unclear_needs_review",
                calibration_confidence=0.70,
                supporting_signals=["sparse_content"],
                conflicting_signals=[],
                calibration_status=(
                    CalibrationStatus.AGREES_WITH_AI
                    if ai_label == "unclear_needs_review"
                    else CalibrationStatus.DISAGREES_WITH_AI if ai_label
                    else CalibrationStatus.INSUFFICIENT_EVIDENCE
                ),
                reasoning="Message is extremely short without sufficient diagnostic context.",
                abstained=False,
            )

        # Insufficient evidence to suggest a specific operational intent
        return CalibrationResult(
            candidate_label=None,
            calibration_confidence=0.0,
            supporting_signals=[],
            conflicting_signals=[],
            calibration_status=CalibrationStatus.INSUFFICIENT_EVIDENCE,
            reasoning="No decisive operational signals detected in customer message.",
            abstained=True,
        )

    # Accumulate weights per intent
    intent_scores: dict[str, float] = {}
    intent_signals: dict[str, list[str]] = {}

    for sig in signals:
        intent_scores[sig.target_intent] = intent_scores.get(sig.target_intent, 0.0) + sig.weight
        intent_signals.setdefault(sig.target_intent, []).append(f"{sig.name} ('{sig.evidence_text}')")

    sorted_intents = sorted(intent_scores.items(), key=lambda x: -x[1])
    top_intent, top_score = sorted_intents[0]

    # Check if AI label is strongly supported by extracted signals
    if ai_label and ai_label in intent_scores:
        ai_score = intent_scores[ai_label]
        # If AI label is top intent OR close to top intent with high confidence
        if top_intent == ai_label or (ai_score >= 0.85 and (top_score - ai_score) <= 0.10):
            calib_conf = min(0.95, max(0.70, ai_score))
            supp_sigs = intent_signals[ai_label]
            return CalibrationResult(
                candidate_label=ai_label,
                calibration_confidence=calib_conf,
                supporting_signals=supp_sigs,
                conflicting_signals=[],
                calibration_status=CalibrationStatus.AGREES_WITH_AI,
                reasoning=f"Detected supporting evidence for '{ai_label}': {'; '.join(supp_sigs)}.",
                abstained=False,
            )

    # Check for ambiguity (competing top candidates with close scores when AI label does not match)
    if len(sorted_intents) > 1:
        second_intent, second_score = sorted_intents[1]
        if top_score - second_score < 0.20 and second_score >= 0.80:
            return CalibrationResult(
                candidate_label=top_intent,
                calibration_confidence=min(0.75, top_score),
                supporting_signals=intent_signals[top_intent],
                conflicting_signals=intent_signals[second_intent],
                calibration_status=CalibrationStatus.AMBIGUOUS,
                reasoning=f"Ambiguous competing signals between '{top_intent}' ({top_score:.2f}) and '{second_intent}' ({second_score:.2f}).",
                abstained=False,
            )

    # Dominant intent determined
    calib_conf = min(0.95, max(0.70, top_score))
    supp_sigs = intent_signals[top_intent]
    conf_sigs: list[str] = []
    for other_intent, other_sig_list in intent_signals.items():
        if other_intent != top_intent:
            conf_sigs.extend(other_sig_list)

    # Determine calibration status vs AI suggestion
    if not ai_label:
        status = CalibrationStatus.INSUFFICIENT_EVIDENCE
    elif top_intent == ai_label:
        status = CalibrationStatus.AGREES_WITH_AI
    else:
        status = CalibrationStatus.DISAGREES_WITH_AI

    reasoning = f"Detected supporting evidence for '{top_intent}': {'; '.join(supp_sigs)}."

    return CalibrationResult(
        candidate_label=top_intent,
        calibration_confidence=calib_conf,
        supporting_signals=supp_sigs,
        conflicting_signals=conf_sigs,
        calibration_status=status,
        reasoning=reasoning,
        abstained=False,
    )


# ---------------------------------------------------------------------------
# Empirical Disagreement & Error Analysis (N = 57 Human Ground Truth)
# ---------------------------------------------------------------------------

@dataclass
class DisagreementAnalysisReport:
    """Statistical audit report comparing human decisions against AI baseline predictions."""
    total_reviewed: int
    agreement_count: int
    agreement_rate: float
    disagreement_count: int
    disagreement_rate: float
    confusion_matrix: dict[str, dict[str, int]]
    top_correction_pairs: list[tuple[str, str, int]]
    calibration_agreement_with_human: int
    calibration_human_agreement_rate: float

    @property
    def agreements(self) -> int:
        return self.agreement_count

    @property
    def disagreements(self) -> int:
        return self.disagreement_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_reviewed": self.total_reviewed,
            "agreement_count": self.agreement_count,
            "agreement_rate": round(self.agreement_rate, 4),
            "disagreement_count": self.disagreement_count,
            "disagreement_rate": round(self.disagreement_rate, 4),
            "confusion_matrix": self.confusion_matrix,
            "top_correction_pairs": [
                {"from_ai": p[0], "to_human": p[1], "count": p[2]}
                for p in self.top_correction_pairs
            ],
            "calibration_agreement_with_human": self.calibration_agreement_with_human,
            "calibration_human_agreement_rate": round(self.calibration_human_agreement_rate, 4),
        }


def analyze_human_ai_disagreements(
    df: pd.DataFrame,
    allowed_labels: Optional[set[str]] = None,
) -> DisagreementAnalysisReport:
    """
    Analyze disagreement patterns across all completed human reviews in the dataset.
    """
    if allowed_labels is None:
        allowed_labels = get_allowed_taxonomy_labels()

    reviewed_records = []
    for _, row in df.iterrows():
        lbl = str(row.get("annotation_label", "")).strip()
        stat = str(row.get("annotation_status", "")).strip().lower()
        if lbl and lbl.lower() not in ("nan", "none", "null") and stat not in ("", "pending", "pending_human_review"):
            reviewed_records.append(row.to_dict() if hasattr(row, "to_dict") else dict(row))

    total_reviewed = len(reviewed_records)
    if total_reviewed == 0:
        return DisagreementAnalysisReport(
            total_reviewed=0,
            agreement_count=0,
            agreement_rate=0.0,
            disagreement_count=0,
            disagreement_rate=0.0,
            confusion_matrix={},
            top_correction_pairs=[],
            calibration_agreement_with_human=0,
            calibration_human_agreement_rate=0.0,
        )

    agreement_count = 0
    disagreement_count = 0
    correction_pairs: list[tuple[str, str]] = []
    matrix: dict[str, dict[str, int]] = {}

    calib_human_agrees = 0

    for rec in reviewed_records:
        human_lbl = str(rec.get("annotation_label", "")).strip()
        ai_sug = str(rec.get("model_suggested_label", "")).strip()

        # Build confusion matrix: True Intent (Human) -> Predicted Intent (AI)
        matrix.setdefault(human_lbl, {})
        matrix[human_lbl][ai_sug] = matrix[human_lbl].get(ai_sug, 0) + 1

        if human_lbl == ai_sug and human_lbl != "unclear_needs_review":
            agreement_count += 1
        else:
            disagreement_count += 1
            if ai_sug and human_lbl:
                correction_pairs.append((ai_sug, human_lbl))

        # Evaluate calibration on this record
        calib = evaluate_taxonomy_calibration(rec, allowed_labels=allowed_labels)
        if calib.candidate_label == human_lbl:
            calib_human_agrees += 1

    agreement_rate = agreement_count / total_reviewed if total_reviewed > 0 else 0.0
    disagreement_rate = disagreement_count / total_reviewed if total_reviewed > 0 else 0.0
    calib_human_agreement_rate = calib_human_agrees / total_reviewed if total_reviewed > 0 else 0.0

    top_pairs = Counter(correction_pairs).most_common(10)
    top_pairs_formatted = [(p[0][0], p[0][1], p[1]) for p in top_pairs]

    return DisagreementAnalysisReport(
        total_reviewed=total_reviewed,
        agreement_count=agreement_count,
        agreement_rate=agreement_rate,
        disagreement_count=disagreement_count,
        disagreement_rate=disagreement_rate,
        confusion_matrix=matrix,
        top_correction_pairs=top_pairs_formatted,
        calibration_agreement_with_human=calib_human_agrees,
        calibration_human_agreement_rate=calib_human_agreement_rate,
    )
