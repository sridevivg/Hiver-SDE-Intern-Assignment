"""
SupportGraph AI — Review Prioritization Engine (Phase 5.6)

Implements deterministic risk scoring and queue generation for human review:
- Evaluates AI suggestions and metadata against transparent, explainable risk rules
- Assigns priority levels: CRITICAL, HIGH, MEDIUM, LOW
- Computes reproducible priority scores (0–100) and human-readable reasons
- Performs deterministic Quality Control (QC) sampling on low-priority records
- Orders review queues by risk: CRITICAL -> HIGH -> MEDIUM -> QC LOW -> REMAINING LOW
- Strictly preserves scientific integrity: never assigns ground truth labels.
"""
from __future__ import annotations

import logging
import math
import random
import re
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
    from app.evaluation.taxonomy_calibration import (
        CalibrationResult,
        CalibrationStatus,
        evaluate_taxonomy_calibration,
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
    from backend.app.evaluation.taxonomy_calibration import (  # type: ignore[no-redef]
        CalibrationResult,
        CalibrationStatus,
        evaluate_taxonomy_calibration,
    )
    from backend.app.nlp.taxonomy_finalization import (  # type: ignore[no-redef]
        DEFAULT_CANDIDATE_TAXONOMY_PATH,
        load_candidate_taxonomy,
    )

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Threshold Constants
# ---------------------------------------------------------------------------
LOW_CONFIDENCE_THRESHOLD: float = 0.70
MEDIUM_CONFIDENCE_THRESHOLD: float = 0.85
HIGH_CONFIDENCE_THRESHOLD: float = 0.90

DEFAULT_QC_SAMPLE_PERCENTAGE: float = 0.15
DEFAULT_QC_MIN_SAMPLE_SIZE: int = 5
QC_RANDOM_SEED: int = 42

# Domain keyword markers for detecting multi-issue ambiguity (competing operational symptoms)
DOMAIN_KEYWORD_PATTERNS: dict[str, list[str]] = {
    "battery_power": ["battery", "drain", "charging", "charger", "overheating", "battery die"],
    "keyboard_typing": ["keyboard", "autocorrect", "typing", "letter i", "predictive text"],
    "software_update": ["update failed", "install error", "update stuck", "failed update", "cannot update"],
    "display_touch": ["screen flicker", "touchscreen", "unresponsive touch", "black screen", "display glitch"],
    "account_access": ["apple id", "appleid", "password reset", "locked account", "security questions"],
    "billing_purchase": ["refund", "unauthorized charge", "credit card charge", "itunes billing", "subscription charge"],
    "audio_sound": ["airpods", "no sound", "microphone", "speaker volume", "audio glitch"],
    "connectivity": ["no wifi", "bluetooth disconnect", "cellular signal", "no service", "cannot connect"],
    "mac_software": ["macbook boot", "macos recovery", "kernel panic", "finder freeze", "mac os"],
}

# Common non-English grammatical patterns and distinctive markers
FOREIGN_GRAMMATICAL_PHRASES: list[tuple[str, str]] = [
    # Spanish
    (r"\bpor favor\b", "Spanish 'por favor'"),
    (r"\bno funciona\b", "Spanish 'no funciona'"),
    (r"\bactualización\b", "Spanish 'actualización'"),
    (r"\bactualicé\b", "Spanish 'actualicé'"),
    (r"\btengo un\b", "Spanish 'tengo un'"),
    (r"\bmi teléfono\b", "Spanish 'mi teléfono'"),
    (r"\bespero que\b", "Spanish 'espero que'"),
    (r"\bcon la\b", "Spanish 'con la'"),
    (r"\bde errores\b", "Spanish 'de errores'"),
    (r"\bayuda por favor\b", "Spanish 'ayuda por favor'"),
    (r"\bno puedo\b", "Spanish 'no puedo'"),
    (r"\bva fatal\b", "Spanish 'va fatal'"),
    # Portuguese
    (r"\bnão funciona\b", "Portuguese 'não funciona'"),
    (r"\bmeu celular\b", "Portuguese 'meu celular'"),
    (r"\bminha bateria\b", "Portuguese 'minha bateria'"),
    (r"\bobrigado pela\b", "Portuguese 'obrigado pela'"),
    (r"\batualização\b", "Portuguese 'atualização'"),
    # French
    (r"\bs'il vous plaît\b", "French 's'il vous plaît'"),
    (r"\bmise à jour\b", "French 'mise à jour'"),
    (r"\bne fonctionne pas\b", "French 'ne fonctionne pas'"),
    (r"\bmon écran\b", "French 'mon écran'"),
    (r"\bj'ai un\b", "French 'j'ai un'"),
    (r"\bproblème avec\b", "French 'problème avec'"),
    # German
    (r"\bund ich\b", "German 'und ich'"),
    (r"\bich dachte\b", "German 'ich dachte'"),
    (r"\bfunktioniert nicht\b", "German 'funktioniert nicht'"),
    (r"\bmein macbook\b", "German 'mein macbook'"),
    (r"\bkeine verbindung\b", "German 'keine verbindung'"),
    (r"\bbitte hilfe\b", "German 'bitte hilfe'"),
    (r"\bvon irgendwelchen\b", "German 'von irgendwelchen'"),
    (r"\bnach dem update\b", "German 'nach dem update'"),
    # Italian
    (r"\bnon funziona\b", "Italian 'non funziona'"),
    (r"\bper favore\b", "Italian 'per favore'"),
    (r"\baggiornamento\b", "Italian 'aggiornamento'"),
    (r"\bil mio\b", "Italian 'il mio'"),
    (r"\bnon riesco\b", "Italian 'non riesco'"),
]

# Distinctive foreign vocabulary words that are unambiguous
# (Strictly excludes ordinary English words like 'problem', 'issue', 'update', 'phone', 'battery', 'screen')
FOREIGN_DISTINCTIVE_WORDS: set[str] = {
    # Spanish
    "hola", "ayuda", "gracias", "pantalla", "batería", "arreglen", "reiniciar",
    "teléfono", "mejore", "solución", "problemas", "velocidad",
    # Portuguese
    "olá", "ajuda", "obrigado", "obrigada", "ecrã", "tela", "reiniciar",
    # French
    "bonjour", "merci", "problème", "problèmes", "batterie", "écran", "redémarrer", "téléphone",
    # German
    "hallo", "hilfe", "danke", "akku", "bildschirm", "neustart", "irgendwelchen",
    "dachte", "bleibt", "liebes", "problemen", "probleme",
    # Italian
    "ciao", "grazie", "aiuto", "schermo", "riavviare", "batteria",
}

# Common English stopwords used for evaluating language proportion
COMMON_ENGLISH_WORDS: set[str] = {
    "the", "i", "my", "is", "a", "to", "and", "in", "it", "you", "of", "for", "on",
    "that", "this", "with", "have", "not", "be", "are", "from", "at", "your", "all",
    "can", "has", "so", "me", "if", "they", "we", "do", "get", "just", "what", "no",
    "how", "but", "when", "or", "like", "up", "an", "out", "by", "about", "did",
    "please", "help", "thanks", "fix", "issue", "problem", "phone", "update", "working",
    "work", "apple", "iphone", "ios", "support", "device", "screen", "battery", "app",
    "after", "time", "back", "still", "now", "why", "again", "new", "even", "then",
    "would", "could", "should", "some", "any", "which", "there", "their", "will",
    "been", "only", "letter", "eye", "paid", "money", "crash", "crashes", "download",
    "photos", "syncing", "between", "high", "sierra", "reinstalled", "member", "shop",
    "safari", "favour", "sisters", "turn", "word", "changing", "cannot", "cant",
    "won't", "wont", "dont", "don't", "freeze", "freezing", "restart", "reboot"
}


class ReviewPriority(str, Enum):
    """Enumeration of human review urgency levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ReviewRecommendation(str, Enum):
    """Actionable human review mode recommendations."""
    INDIVIDUAL = "individual_review_required"
    QC_REVIEW = "qc_review_required"
    GROUP_REVIEW = "safe_for_group_review"


@dataclass
class LanguageEvidence:
    """
    Detailed evidence assessment for non-English content detection.

    Strengths:
    - none: Clean English message
    - weak: Isolated foreign word / loanword with English context (does NOT escalate priority)
    - moderate: Multilingual phrasing or multiple foreign tokens
    - strong: High non-Latin character density, or strong foreign grammar with low/no English words
    """
    detected: bool = False
    strength: str = "none"  # "none", "weak", "moderate", "strong"
    reason: Optional[str] = None
    matched_indicators: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "detected": self.detected,
            "strength": self.strength,
            "reason": self.reason,
            "matched_indicators": self.matched_indicators,
        }


@dataclass
class RecordPriority:
    """Structured assessment of review priority for a single golden record."""
    golden_id: str
    priority: ReviewPriority
    priority_score: int
    priority_reason: str
    qc_sample: bool = False
    review_recommendation: str = ReviewRecommendation.INDIVIDUAL.value
    risk_flags: list[str] = field(default_factory=list)
    risk_factors: list[str] = field(default_factory=list)
    language_evidence: Optional[LanguageEvidence] = None
    calibration_result: Optional[CalibrationResult] = None

    def __post_init__(self) -> None:
        if not self.risk_factors and self.risk_flags:
            self.risk_factors = list(self.risk_flags)
        elif not self.risk_flags and self.risk_factors:
            self.risk_flags = list(self.risk_factors)

    def to_dict(self) -> dict[str, Any]:
        res: dict[str, Any] = {
            "golden_id": self.golden_id,
            "priority": self.priority.value,
            "priority_score": self.priority_score,
            "priority_reason": self.priority_reason,
            "qc_sample": self.qc_sample,
            "review_recommendation": self.review_recommendation,
            "risk_flags": "; ".join(self.risk_flags),
            "risk_factors": "; ".join(self.risk_flags),
        }
        if self.calibration_result:
            res["calibration_status"] = self.calibration_result.status.value
            res["calibration_candidate_label"] = self.calibration_result.candidate_label
            res["calibration_confidence"] = self.calibration_result.confidence
        return res


def detect_non_english_evidence(text: str) -> LanguageEvidence:
    """
    Deterministic, layered heuristic for detecting non-English customer messages.
    Categorizes evidence into:
    - strong: High non-Latin script character density, or distinctive foreign grammar without English context
    - moderate: Multilingual phrasing with non-English functional phrases
    - weak: Single isolated foreign word / loanword with English context (does NOT escalate priority)
    - none: Standard English message
    """
    if not text or not text.strip():
        return LanguageEvidence(detected=False, strength="none", reason=None)

    cleaned = text.strip()
    # Strip user handles, URLs, and hashtags for language calculation
    content_only = re.sub(r"@\w+", "", cleaned)
    content_only = re.sub(r"https?://\S+", "", content_only)
    content_only = re.sub(r"#\w+", "", content_only).strip()

    if not content_only:
        return LanguageEvidence(detected=False, strength="none", reason=None)

    # 1. Check Non-Latin alphabetic characters (Arabic, Cyrillic, CJK, Devanagari, Thai, Hebrew, Greek, etc.)
    alpha_chars = [c for c in content_only if c.isalpha()]
    total_alpha = len(alpha_chars)
    if total_alpha >= 5:
        # Non-Latin characters outside Basic Latin, Latin-1 Supplement, and Latin Extended-A/B
        non_latin_chars = [c for c in alpha_chars if ord(c) > 0x024F]
        non_latin_ratio = len(non_latin_chars) / total_alpha
        if non_latin_ratio > 0.15:
            return LanguageEvidence(
                detected=True,
                strength="strong",
                reason=f"High non-Latin script character density ({len(non_latin_chars)}/{total_alpha} characters)",
                matched_indicators=["non_latin_script"],
            )

    # Normalize lower case for phrase and token matching
    lower_text = content_only.lower()

    # 2. Check foreign grammatical phrases
    matched_phrases: list[str] = []
    for pat, desc in FOREIGN_GRAMMATICAL_PHRASES:
        if re.search(pat, lower_text, flags=re.IGNORECASE):
            matched_phrases.append(desc)

    # 3. Tokenize words (removing punctuation)
    words = re.findall(r"\b[a-zà-ÿ0-9'-]+\b", lower_text)
    english_word_count = sum(1 for w in words if w in COMMON_ENGLISH_WORDS)
    foreign_words_matched: list[str] = [w for w in words if w in FOREIGN_DISTINCTIVE_WORDS]

    all_matches = list(dict.fromkeys(matched_phrases + foreign_words_matched))

    if not all_matches:
        return LanguageEvidence(
            detected=False,
            strength="none",
            reason=None,
            matched_indicators=[],
        )

    num_matches = len(all_matches)

    # Case A: Strong evidence
    # - Distinct multi-word foreign phrase or >= 2 foreign indicators with low/no English stopwords
    if (len(matched_phrases) >= 1 and english_word_count <= 2) or (num_matches >= 2 and english_word_count <= 2):
        return LanguageEvidence(
            detected=True,
            strength="strong",
            reason=f"Strong non-English grammatical indicators ({', '.join(all_matches[:3])})",
            matched_indicators=all_matches,
        )

    # Case B: Moderate evidence
    # - Multiple foreign indicators in multilingual message (e.g. bilingual text)
    if num_matches >= 2:
        return LanguageEvidence(
            detected=True,
            strength="moderate",
            reason=f"Multilingual or foreign phrasing detected ({', '.join(all_matches[:3])})",
            matched_indicators=all_matches,
        )

    # Case C: Weak evidence
    # - Single isolated foreign word or 1 phrase in predominantly English message
    # Weak evidence is recorded but does NOT escalate priority
    return LanguageEvidence(
        detected=True,
        strength="weak",
        reason=f"Isolated non-English token ('{all_matches[0]}') in predominantly English message",
        matched_indicators=all_matches,
    )


def detect_non_english_heuristic(text: str) -> tuple[bool, str]:
    """
    Deterministic heuristic for flagging potentially non-English messages.
    Returns (is_non_english, matched_detail).
    Only returns True if evidence strength is 'moderate' or 'strong'.
    """
    evidence = detect_non_english_evidence(text)
    if evidence.detected and evidence.strength in ("moderate", "strong"):
        return True, evidence.reason or "Non-English text detected"
    return False, ""


def detect_multi_domain_overlap(text: str) -> list[str]:
    """
    Identify if a message mentions keywords from multiple distinct operational domains.
    """
    cleaned = text.lower()
    matched_domains: list[str] = []
    for domain, keywords in DOMAIN_KEYWORD_PATTERNS.items():
        if any(re.search(r"\b" + re.escape(kw) + r"\b", cleaned) for kw in keywords):
            matched_domains.append(domain)
    return matched_domains


def evaluate_record_priority(
    record: dict[str, Any] | pd.Series,
    allowed_labels: Optional[set[str]] = None,
    low_threshold: float = LOW_CONFIDENCE_THRESHOLD,
    medium_threshold: float = MEDIUM_CONFIDENCE_THRESHOLD,
    high_threshold: float = HIGH_CONFIDENCE_THRESHOLD,
) -> RecordPriority:
    """
    Calculate deterministic review priority, score, explainable reason, and risk flags.

    Priority hierarchy:
    1. CRITICAL (85-100): Invalid suggestion, taxonomy mismatch, unclear_needs_review label,
       explicit model_needs_human_review flag, confidence < low_threshold, or severely truncated text.
    2. HIGH (60-84): Borderline confidence (0.70-0.85), true language uncertainty (moderate/strong),
       multi-intent domain overlap, candidate intent divergence, or sparse message without context.
    3. MEDIUM (35-59): Moderate confidence (0.85-0.90) with valid taxonomy label.
    4. LOW (0-34): High confidence (>= 0.90) with clear, unambiguous operational context.
    """
    if allowed_labels is None:
        allowed_labels = get_allowed_taxonomy_labels()

    gid = str(record.get("golden_id", "")).strip() or "unknown_id"
    cust_msg = str(record.get("customer_message", "")).strip()
    raw_norm = str(record.get("normalized_message", "")).strip()
    norm_msg = raw_norm if (raw_norm and (not cust_msg or cust_msg in raw_norm)) else cust_msg
    context = str(record.get("conversation_context", "")).strip()
    cand_intent = str(record.get("candidate_intent", "")).strip()

    raw_sug_label = record.get("model_suggested_label")
    sug_label = str(raw_sug_label).strip() if raw_sug_label is not None and not pd.isna(raw_sug_label) else ""
    if sug_label.lower() in ("nan", "none", "null", "undefined"):
        sug_label = ""

    raw_status = record.get("suggestion_status")
    status = str(raw_status).strip().lower() if raw_status is not None and not pd.isna(raw_status) else ""

    raw_needs_review = record.get("model_needs_human_review")
    needs_review = False
    if raw_needs_review is not None and not pd.isna(raw_needs_review):
        needs_review = str(raw_needs_review).strip().lower() in ("true", "1", "yes")

    raw_conf = record.get("model_confidence")
    confidence: Optional[float] = None
    if raw_conf is not None and not pd.isna(raw_conf) and str(raw_conf).strip() != "":
        try:
            val = float(raw_conf)
            if not math.isnan(val) and not math.isinf(val):
                confidence = max(0.0, min(1.0, val))
        except (ValueError, TypeError):
            confidence = None

    # Language evidence assessment
    lang_evidence = detect_non_english_evidence(cust_msg)

    # Taxonomy calibration assessment (Phase 5.9)
    calib_res = evaluate_taxonomy_calibration(record, allowed_labels=allowed_labels)

    # =========================================================================
    # 1. CRITICAL CHECKS
    # =========================================================================

    # Check invalid or failed status
    if status in ("failed", "invalid_model_output", "error"):
        return RecordPriority(
            golden_id=gid,
            priority=ReviewPriority.CRITICAL,
            priority_score=100,
            priority_reason=f"AI suggestion run {status}: human annotation required.",
            risk_flags=["invalid_suggestion"],
            language_evidence=lang_evidence,
            calibration_result=calib_res,
        )

    # Check missing or unauthorized label
    if not is_valid_ai_suggestion(sug_label, allowed_labels=allowed_labels, suggestion_status=status):
        return RecordPriority(
            golden_id=gid,
            priority=ReviewPriority.CRITICAL,
            priority_score=100,
            priority_reason=f"Missing or unauthorized AI label '{sug_label}'.",
            risk_flags=["invalid_suggestion", "taxonomy_mismatch"],
            language_evidence=lang_evidence,
            calibration_result=calib_res,
        )

    # Check model suggestion is 'unclear_needs_review'
    if sug_label == "unclear_needs_review":
        return RecordPriority(
            golden_id=gid,
            priority=ReviewPriority.CRITICAL,
            priority_score=92,
            priority_reason="AI classified message as 'unclear_needs_review'.",
            risk_flags=["unclear_needs_review_suggested"],
            language_evidence=lang_evidence,
            calibration_result=calib_res,
        )

    # Check model_needs_human_review flag
    if needs_review:
        conf_str = f"{confidence:.2f}" if confidence is not None else "N/A"
        return RecordPriority(
            golden_id=gid,
            priority=ReviewPriority.CRITICAL,
            priority_score=90,
            priority_reason=f"AI flagged record for mandatory review (confidence: {conf_str}).",
            risk_flags=["model_requested_review"],
            language_evidence=lang_evidence,
            calibration_result=calib_res,
        )

    # Check missing confidence
    if confidence is None:
        return RecordPriority(
            golden_id=gid,
            priority=ReviewPriority.CRITICAL,
            priority_score=90,
            priority_reason="Missing numeric AI confidence score.",
            risk_flags=["low_confidence"],
            language_evidence=lang_evidence,
            calibration_result=calib_res,
        )

    # Check low confidence (< 0.70)
    if confidence < low_threshold:
        score = int(85 + ((low_threshold - confidence) / low_threshold) * 15)
        return RecordPriority(
            golden_id=gid,
            priority=ReviewPriority.CRITICAL,
            priority_score=min(100, score),
            priority_reason=f"Low AI confidence: {confidence:.2f} (< {low_threshold:.2f}).",
            risk_flags=["low_confidence"],
            language_evidence=lang_evidence,
            calibration_result=calib_res,
        )

    # Check severe truncation / very short message without context
    words = [w for w in norm_msg.split() if w]
    is_severely_short = len(norm_msg) < 15 or len(words) <= 2
    if is_severely_short and (not context or context == "No brand response observed"):
        return RecordPriority(
            golden_id=gid,
            priority=ReviewPriority.CRITICAL,
            priority_score=88,
            priority_reason="Severely truncated customer message without conversation context.",
            risk_flags=["truncated_message"],
            language_evidence=lang_evidence,
            calibration_result=calib_res,
        )

    # =========================================================================
    # 2. HIGH PRIORITY CHECKS
    # =========================================================================

    high_risk_reasons: list[str] = []
    high_risk_flags: list[str] = []
    high_score_base = 60

    # A. Borderline confidence (0.70 <= confidence < 0.85)
    if low_threshold <= confidence < medium_threshold:
        fraction = (medium_threshold - confidence) / (medium_threshold - low_threshold)
        high_score_base = max(high_score_base, int(65 + fraction * 15))
        high_risk_reasons.append(f"Borderline confidence: {confidence:.2f}")
        high_risk_flags.append("borderline_confidence")

    # B. True Language Uncertainty (ONLY moderate or strong evidence)
    if lang_evidence.detected and lang_evidence.strength in ("moderate", "strong"):
        high_score_base = max(high_score_base, 75)
        high_risk_reasons.append(f"Language uncertainty: {lang_evidence.reason}")
        high_risk_flags.append("true_language_uncertainty")

    # C. Intent divergence: candidate_intent differs from AI suggestion (when confidence < 0.90)
    if cand_intent and cand_intent not in ("", "unclear_needs_review") and cand_intent != sug_label:
        if confidence < high_threshold:
            high_score_base = max(high_score_base, 72)
            high_risk_reasons.append(
                f"Intent divergence: cluster candidate '{cand_intent}' vs AI '{sug_label}'"
            )
            high_risk_flags.append("intent_divergence")

    # D. Multi-domain keyword overlap (when confidence < 0.90)
    domains = detect_multi_domain_overlap(norm_msg)
    if len(domains) >= 2 and confidence < high_threshold:
        high_score_base = max(high_score_base, 68)
        high_risk_reasons.append(f"Multi-intent overlap: {', '.join(domains[:3])}")
        high_risk_flags.append("multi_intent_overlap")

    # E. Short message with sparse context (< 25 chars or <= 4 words without context)
    if (len(norm_msg) < 25 or len(words) <= 4) and (not context or context == "No brand response observed") and confidence < high_threshold:
        high_score_base = max(high_score_base, 65)
        high_risk_reasons.append("Sparse message with minimal context")
        high_risk_flags.append("sparse_message")

    # F. Taxonomy calibration disagreement (Phase 5.9)
    if calib_res.status == CalibrationStatus.DISAGREES_WITH_AI and calib_res.candidate_label:
        high_score_base = max(high_score_base, 78)
        sig_names = [s.name if hasattr(s, "name") else str(s) for s in calib_res.supporting_signals]
        sig_str = f" ({', '.join(sig_names[:2])})" if sig_names else ""
        high_risk_reasons.append(
            f"Taxonomy calibration divergence: AI '{sug_label}' vs Calibrated '{calib_res.candidate_label}'{sig_str}"
        )
        high_risk_flags.append("calibration_disagreement")
    elif calib_res.status == CalibrationStatus.AMBIGUOUS and calib_res.conflicting_signals:
        conf_names = [s.name if hasattr(s, "name") else str(s) for s in calib_res.conflicting_signals]
        high_score_base = max(high_score_base, 66)
        high_risk_reasons.append(f"Taxonomy signal conflict: {', '.join(conf_names[:2])}")
        high_risk_flags.append("calibration_conflict")

    if high_risk_flags:
        return RecordPriority(
            golden_id=gid,
            priority=ReviewPriority.HIGH,
            priority_score=min(84, high_score_base),
            priority_reason="; ".join(high_risk_reasons),
            risk_flags=high_risk_flags,
            language_evidence=lang_evidence,
            calibration_result=calib_res,
        )

    # =========================================================================
    # 3. MEDIUM PRIORITY CHECKS (0.85 <= confidence < 0.90)
    # =========================================================================
    if medium_threshold <= confidence < high_threshold:
        fraction = (high_threshold - confidence) / (high_threshold - medium_threshold)
        med_score = int(35 + fraction * 24)
        return RecordPriority(
            golden_id=gid,
            priority=ReviewPriority.MEDIUM,
            priority_score=min(59, max(35, med_score)),
            priority_reason=f"Moderate confidence: {confidence:.2f} with valid taxonomy match.",
            qc_sample=False,
            review_recommendation=ReviewRecommendation.INDIVIDUAL.value,
            risk_flags=["moderate_confidence"],
            language_evidence=lang_evidence,
            calibration_result=calib_res,
        )

    # =========================================================================
    # 4. LOW PRIORITY (Clean, high-confidence records >= 0.90)
    # =========================================================================
    low_score = max(5, int(30 - (confidence - high_threshold) * 200))
    low_flags = ["high_confidence_clean"]
    if calib_res.status == CalibrationStatus.AGREES_WITH_AI:
        low_flags.append("calibration_agreement")

    return RecordPriority(
        golden_id=gid,
        priority=ReviewPriority.LOW,
        priority_score=min(34, max(0, low_score)),
        priority_reason=f"High confidence ({confidence:.2f}) with unambiguous operational context.",
        qc_sample=False,
        review_recommendation=ReviewRecommendation.GROUP_REVIEW.value,
        risk_flags=low_flags,
        language_evidence=lang_evidence,
        calibration_result=calib_res,
    )


def apply_qc_sampling(
    records: list[RecordPriority],
    sample_size: Optional[int] = None,
    sample_percentage: float = DEFAULT_QC_SAMPLE_PERCENTAGE,
    seed: int = QC_RANDOM_SEED,
) -> list[RecordPriority]:
    """
    Select a deterministic, reproducible quality-control sample from LOW priority records.

    Args:
        records: List of evaluated RecordPriority objects.
        sample_size: Explicit count override for QC sample size.
        sample_percentage: Proportion of low-priority records to sample (e.g. 0.15 = 15%).
        seed: Fixed random seed ensuring determinism.

    Returns:
        Updated list of RecordPriority instances with qc_sample marked.
    """
    # Identify indices of low-priority records
    low_indices = [i for i, r in enumerate(records) if r.priority == ReviewPriority.LOW]
    if not low_indices:
        return records

    # Calculate quota
    if sample_size is not None:
        target_count = max(0, min(sample_size, len(low_indices)))
    else:
        target_count = max(
            DEFAULT_QC_MIN_SAMPLE_SIZE,
            int(math.ceil(len(low_indices) * sample_percentage)),
        )
        target_count = min(target_count, len(low_indices))

    # Sort low indices deterministically by golden_id to prevent any system ordering variance
    sorted_low_indices = sorted(low_indices, key=lambda idx: records[idx].golden_id)

    # Deterministic pseudo-random selection
    rng = random.Random(seed)
    chosen_indices = set(rng.sample(sorted_low_indices, target_count))

    updated_records: list[RecordPriority] = []
    for idx, rec in enumerate(records):
        if idx in chosen_indices:
            updated = RecordPriority(
                golden_id=rec.golden_id,
                priority=rec.priority,
                priority_score=rec.priority_score,
                priority_reason=f"{rec.priority_reason} [QC SAMPLE: Random quality-control validation]",
                qc_sample=True,
                review_recommendation=ReviewRecommendation.QC_REVIEW.value,
                risk_flags=rec.risk_flags + ["qc_sample"],
                language_evidence=rec.language_evidence,
                calibration_result=rec.calibration_result,
            )
            updated_records.append(updated)
        else:
            rec_rec = (
                ReviewRecommendation.GROUP_REVIEW.value
                if rec.priority == ReviewPriority.LOW
                else ReviewRecommendation.INDIVIDUAL.value
            )
            updated = RecordPriority(
                golden_id=rec.golden_id,
                priority=rec.priority,
                priority_score=rec.priority_score,
                priority_reason=rec.priority_reason,
                qc_sample=rec.qc_sample,
                review_recommendation=rec_rec,
                risk_flags=rec.risk_flags,
                language_evidence=rec.language_evidence,
                calibration_result=rec.calibration_result,
            )
            updated_records.append(updated)

    return updated_records


def build_prioritized_review_queue(
    df: pd.DataFrame,
    taxonomy_path: Optional[Path | str] = None,
    sample_size: Optional[int] = None,
    sample_percentage: float = DEFAULT_QC_SAMPLE_PERCENTAGE,
    seed: int = QC_RANDOM_SEED,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Construct a prioritized review queue from a golden suggestions DataFrame.

    Order of records in output queue:
    1. CRITICAL priority (score desc, golden_id asc)
    2. HIGH priority (score desc, golden_id asc)
    3. MEDIUM priority (score desc, golden_id asc)
    4. LOW priority with QC sample = True (score desc, golden_id asc)
    5. LOW priority remaining (score desc, golden_id asc)

    Returns:
        (sorted_queue_df, queue_summary_stats)
    """
    allowed_labels = get_allowed_taxonomy_labels(taxonomy_path)

    # Ensure all columns are strings and nan safe
    df_clean = df.copy()
    for col in df_clean.columns:
        df_clean[col] = df_clean[col].fillna("").astype(object)
        df_clean[col] = df_clean[col].apply(
            lambda x: "" if str(x).strip().lower() in ("nan", "none", "null", "undefined") else str(x).strip()
        )

    # Assess priority for each row
    assessments: list[RecordPriority] = []
    for _, row in df_clean.iterrows():
        assessment = evaluate_record_priority(row, allowed_labels=allowed_labels)
        assessments.append(assessment)

    # Apply deterministic QC sampling to low priority records
    sampled_assessments = apply_qc_sampling(
        assessments,
        sample_size=sample_size,
        sample_percentage=sample_percentage,
        seed=seed,
    )

    # Attach priority metadata to dataframe
    df_clean["priority"] = [a.priority.value for a in sampled_assessments]
    df_clean["priority_score"] = [a.priority_score for a in sampled_assessments]
    df_clean["priority_reason"] = [a.priority_reason for a in sampled_assessments]
    df_clean["qc_sample"] = ["True" if a.qc_sample else "False" for a in sampled_assessments]
    df_clean["review_recommendation"] = [a.review_recommendation for a in sampled_assessments]
    df_clean["risk_flags"] = ["; ".join(a.risk_flags) for a in sampled_assessments]
    df_clean["risk_factors"] = ["; ".join(a.risk_flags) for a in sampled_assessments]

    # Attach calibration metadata to dataframe (Phase 5.9)
    df_clean["calibration_candidate_label"] = [
        a.calibration_result.candidate_label if a.calibration_result else "" for a in sampled_assessments
    ]
    df_clean["calibration_confidence"] = [
        f"{a.calibration_result.confidence:.2f}" if a.calibration_result and a.calibration_result.confidence > 0 else ""
        for a in sampled_assessments
    ]
    df_clean["calibration_supporting_signals"] = [
        "; ".join(s.name if hasattr(s, "name") else str(s) for s in a.calibration_result.supporting_signals)
        if a.calibration_result else ""
        for a in sampled_assessments
    ]
    df_clean["calibration_conflicting_signals"] = [
        "; ".join(s.name if hasattr(s, "name") else str(s) for s in a.calibration_result.conflicting_signals)
        if a.calibration_result else ""
        for a in sampled_assessments
    ]
    df_clean["calibration_status"] = [
        a.calibration_result.status.value if a.calibration_result else "" for a in sampled_assessments
    ]

    # Partition pending vs already human reviewed
    pending_indices: list[int] = []
    reviewed_indices: list[int] = []

    for idx, row in df_clean.iterrows():
        lbl = str(row.get("annotation_label", "")).strip()
        stat = str(row.get("annotation_status", "")).strip().lower()
        if lbl and stat not in ("", "pending", "pending_human_review"):
            reviewed_indices.append(idx)
        else:
            pending_indices.append(idx)

    # Priority ranking tier key for sorting:
    # 0 = Critical
    # 1 = High
    # 2 = Medium
    # 3 = Low QC Sample
    # 4 = Low Non-QC
    def get_sort_key(idx: int) -> tuple[int, int, str]:
        rec = sampled_assessments[idx]
        if rec.priority == ReviewPriority.CRITICAL:
            tier = 0
        elif rec.priority == ReviewPriority.HIGH:
            tier = 1
        elif rec.priority == ReviewPriority.MEDIUM:
            tier = 2
        elif rec.qc_sample:
            tier = 3
        else:
            tier = 4
        # Descending score: -rec.priority_score, then tie-break by golden_id
        return (tier, -rec.priority_score, rec.golden_id)

    sorted_pending_indices = sorted(pending_indices, key=get_sort_key)
    sorted_reviewed_indices = sorted(reviewed_indices, key=lambda idx: df_clean.at[idx, "golden_id"])

    # Final combined order: prioritized pending records first, followed by already reviewed records
    final_order = sorted_pending_indices + sorted_reviewed_indices
    sorted_df = df_clean.iloc[final_order].reset_index(drop=True)

    # Calculate summary metrics
    total = len(df_clean)
    num_reviewed = len(reviewed_indices)
    num_pending = len(pending_indices)

    crit_count = sum(1 for a in sampled_assessments if a.priority == ReviewPriority.CRITICAL)
    high_count = sum(1 for a in sampled_assessments if a.priority == ReviewPriority.HIGH)
    med_count = sum(1 for a in sampled_assessments if a.priority == ReviewPriority.MEDIUM)
    low_count = sum(1 for a in sampled_assessments if a.priority == ReviewPriority.LOW)
    qc_count = sum(1 for a in sampled_assessments if a.qc_sample)

    calib_agree_count = sum(
        1 for a in sampled_assessments
        if a.calibration_result and a.calibration_result.status == CalibrationStatus.AGREES_WITH_AI
    )
    calib_disagree_count = sum(
        1 for a in sampled_assessments
        if a.calibration_result and a.calibration_result.status == CalibrationStatus.DISAGREES_WITH_AI
    )
    calib_insufficient_count = sum(
        1 for a in sampled_assessments
        if a.calibration_result and a.calibration_result.status == CalibrationStatus.INSUFFICIENT_EVIDENCE
    )

    summary = {
        "total_records": total,
        "already_human_reviewed": num_reviewed,
        "pending_human_review": num_pending,
        "critical_priority": crit_count,
        "high_priority": high_count,
        "medium_priority": med_count,
        "low_priority": low_count,
        "qc_sampled_low_priority": qc_count,
        "individual_review_required": sum(1 for a in sampled_assessments if a.review_recommendation == ReviewRecommendation.INDIVIDUAL.value),
        "qc_review_required": sum(1 for a in sampled_assessments if a.review_recommendation == ReviewRecommendation.QC_REVIEW.value),
        "safe_for_group_review": sum(1 for a in sampled_assessments if a.review_recommendation == ReviewRecommendation.GROUP_REVIEW.value),
        "calibration_agreements": calib_agree_count,
        "calibration_disagreements": calib_disagree_count,
        "calibration_insufficient_evidence": calib_insufficient_count,
        "total_in_review_queue": total,
        "qc_random_seed": seed,
    }

    return sorted_df, summary


# ---------------------------------------------------------------------------
# Safe Group Review Structures & Helpers (Phase 5.8 & 5.9)
# ---------------------------------------------------------------------------

ESCALATION_RISK_FLAGS: set[str] = {
    "invalid_suggestion",
    "taxonomy_mismatch",
    "unclear_needs_review_suggested",
    "model_requested_review",
    "low_confidence",
    "borderline_confidence",
    "moderate_confidence",
    "true_language_uncertainty",
    "intent_divergence",
    "multi_intent_overlap",
    "sparse_message",
    "truncated_message",
    "qc_sample",
    "calibration_disagreement",
    "calibration_conflict",
}


@dataclass
class SafeReviewGroup:
    """Represents a deterministic bundle of low-risk records eligible for grouped review."""
    group_id: str
    suggested_label: str
    record_indices: list[int]
    records: list[dict[str, Any]]
    min_confidence: float
    max_confidence: float
    risk_flags: list[str]
    size: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "suggested_label": self.suggested_label,
            "record_indices": self.record_indices,
            "min_confidence": self.min_confidence,
            "max_confidence": self.max_confidence,
            "risk_flags": self.risk_flags,
            "size": self.size,
        }


def is_eligible_for_group_review(
    record: dict[str, Any] | pd.Series,
    allowed_labels: Optional[set[str]] = None,
    min_confidence: float = HIGH_CONFIDENCE_THRESHOLD,
) -> tuple[bool, str]:
    """
    Strictly verify if a record satisfies all criteria for Safe Group Review:
    1. priority == LOW
    2. qc_sample == False
    3. risk_flags contain no escalation flags (including calibration disagreements/conflicts)
    4. model_needs_human_review == False
    5. model_confidence >= min_confidence (0.90)
    6. Not already human reviewed (annotation_label is empty)
    7. annotation_status in ("pending_human_review", "pending", "")
    8. No critical / high / medium priority
    9. No language uncertainty requiring review
    10. No intent divergence requiring review
    11. Calibration status does not disagree with AI suggestion

    Returns:
        (is_eligible: bool, reason: str)
    """
    if allowed_labels is None:
        allowed_labels = get_allowed_taxonomy_labels()

    # Check 6 & 7: Check if already human reviewed
    raw_lbl = record.get("annotation_label")
    lbl = str(raw_lbl).strip() if raw_lbl is not None and not pd.isna(raw_lbl) else ""
    if lbl.lower() in ("nan", "none", "null"):
        lbl = ""

    raw_astat = record.get("annotation_status")
    astat = str(raw_astat).strip().lower() if raw_astat is not None and not pd.isna(raw_astat) else ""

    if lbl and astat not in ("", "pending", "pending_human_review"):
        return False, f"Record is already human reviewed (label: '{lbl}', status: '{astat}')."

    # Check AI suggestion validity
    raw_sug = record.get("model_suggested_label")
    sug_label = str(raw_sug).strip() if raw_sug is not None and not pd.isna(raw_sug) else ""
    raw_status = record.get("suggestion_status")
    status = str(raw_status).strip().lower() if raw_status is not None and not pd.isna(raw_status) else ""

    if not is_valid_ai_suggestion(sug_label, allowed_labels, status):
        return False, f"AI suggestion '{sug_label}' is missing or not in approved taxonomy."

    if sug_label == "unclear_needs_review":
        return False, "AI suggestion is 'unclear_needs_review' (requires individual adjudication)."

    # Check 4: model_needs_human_review
    raw_needs = record.get("model_needs_human_review")
    if raw_needs is not None and not pd.isna(raw_needs):
        if str(raw_needs).strip().lower() in ("true", "1", "yes"):
            return False, "Model explicitly requested human review (model_needs_human_review=True)."

    # Check 5: Confidence
    raw_conf = record.get("model_confidence")
    if raw_conf is None or pd.isna(raw_conf) or str(raw_conf).strip() == "":
        return False, "Missing numeric AI confidence score."
    try:
        conf_val = float(raw_conf)
        if conf_val < min_confidence:
            return False, f"Confidence {conf_val:.2f} is below safe group threshold ({min_confidence:.2f})."
    except (ValueError, TypeError):
        return False, f"Invalid confidence value '{raw_conf}'."

    # Check 2: QC sample flag
    raw_qc = record.get("qc_sample")
    if raw_qc is not None and not pd.isna(raw_qc):
        if str(raw_qc).strip().lower() in ("true", "1", "yes"):
            return False, "Record is marked as QC sample (requires mandatory individual audit)."

    # Check evaluated priority and risk flags
    # If priority is present on record, check it; otherwise evaluate on the fly
    raw_prio = record.get("priority")
    prio_str = str(raw_prio).strip().lower() if raw_prio is not None and not pd.isna(raw_prio) else ""

    if prio_str:
        if prio_str in ("critical", "high", "medium"):
            return False, f"Record priority is '{prio_str.upper()}' (requires individual review)."
        if prio_str != "low":
            return False, f"Record priority '{prio_str}' is not LOW."
    else:
        # Evaluate dynamically
        assessment = evaluate_record_priority(record, allowed_labels=allowed_labels)
        if assessment.priority != ReviewPriority.LOW:
            return False, f"Evaluated priority is '{assessment.priority.value.upper()}' ({assessment.priority_reason})."
        if assessment.qc_sample:
            return False, "Evaluated as QC sample."

    # Check 3: Escalation risk flags
    raw_flags = record.get("risk_flags") or record.get("risk_factors")
    flags_list: list[str] = []
    if raw_flags is not None and not pd.isna(raw_flags):
        if isinstance(raw_flags, list):
            flags_list = [str(f).strip() for f in raw_flags if str(f).strip()]
        else:
            flags_list = [f.strip() for f in str(raw_flags).replace(";", ",").split(",") if f.strip()]

    found_escalations = [f for f in flags_list if f in ESCALATION_RISK_FLAGS]
    if found_escalations:
        return False, f"Record contains escalation risk flags: {', '.join(found_escalations)}."

    # Check calibration status and signals
    raw_calib_status = record.get("calibration_status")
    calib_status_str = str(raw_calib_status).strip() if raw_calib_status is not None and not pd.isna(raw_calib_status) else ""
    if calib_status_str == CalibrationStatus.DISAGREES_WITH_AI.value:
        return False, "Calibration recommendation disagrees with AI suggestion (mandatory individual review)."

    raw_calib_conflict = record.get("calibration_conflicting_signals")
    if raw_calib_conflict is not None and not pd.isna(raw_calib_conflict) and str(raw_calib_conflict).strip() and str(raw_calib_conflict).strip().lower() != "none":
        return False, f"Record has conflicting calibration signals: {str(raw_calib_conflict).strip()}."

    # If calibration status was not provided in record, evaluate dynamically
    if not calib_status_str:
        dynamic_calib = evaluate_taxonomy_calibration(record, allowed_labels=allowed_labels)
        if dynamic_calib.status == CalibrationStatus.DISAGREES_WITH_AI:
            return False, f"Calibration analysis suggests '{dynamic_calib.candidate_label}' instead of '{sug_label}'."
        if dynamic_calib.status == CalibrationStatus.AMBIGUOUS:
            conf_str = ', '.join(s.name if hasattr(s, "name") else str(s) for s in dynamic_calib.conflicting_signals)
            return False, f"Calibration detected conflicting signals: {conf_str}."

    # Check recommendation
    raw_rec = record.get("review_recommendation")
    rec_str = str(raw_rec).strip().lower() if raw_rec is not None and not pd.isna(raw_rec) else ""
    if rec_str and rec_str not in ("safe_for_group_review", "group_review"):
        return False, f"Review recommendation is '{rec_str}'."

    return True, "Eligible for safe group review."


def build_safe_review_groups(
    df: pd.DataFrame,
    max_group_size: int = 10,
    allowed_labels: Optional[set[str]] = None,
) -> list[SafeReviewGroup]:
    """
    Construct deterministic Safe Review Groups for pending low-risk records.

    Grouping and Ordering Rules:
    1. Filter pending records meeting all is_eligible_for_group_review criteria.
    2. Group records by model_suggested_label.
    3. Within each intent group, sort deterministically:
       - model_confidence descending
       - golden_id ascending
    4. Partition large groups into chunks of at most max_group_size (default: 10).
    5. Sort overall groups deterministically: suggested_label alphabetically, group_id ascending.

    Returns:
        List of SafeReviewGroup instances.
    """
    if allowed_labels is None:
        allowed_labels = get_allowed_taxonomy_labels()

    # Find eligible rows
    eligible_by_intent: dict[str, list[tuple[int, dict[str, Any]]]] = {}

    for idx, row in df.iterrows():
        rec_dict = row.to_dict() if hasattr(row, "to_dict") else dict(row)
        eligible, _ = is_eligible_for_group_review(rec_dict, allowed_labels=allowed_labels)
        if eligible:
            intent = str(rec_dict.get("model_suggested_label", "")).strip()
            eligible_by_intent.setdefault(intent, []).append((idx, rec_dict))

    groups: list[SafeReviewGroup] = []

    # Sort intent names alphabetically
    for intent in sorted(eligible_by_intent.keys()):
        items = eligible_by_intent[intent]

        # Sort items within intent: confidence desc, golden_id asc
        def sort_key(item: tuple[int, dict[str, Any]]) -> tuple[float, str]:
            _, r = item
            try:
                conf = float(r.get("model_confidence", 0.0))
            except (ValueError, TypeError):
                conf = 0.0
            gid = str(r.get("golden_id", ""))
            return (-conf, gid)

        sorted_items = sorted(items, key=sort_key)

        # Chunk into max_group_size batches
        num_chunks = math.ceil(len(sorted_items) / max_group_size) if max_group_size > 0 else 1
        for chunk_idx in range(num_chunks):
            chunk = sorted_items[chunk_idx * max_group_size : (chunk_idx + 1) * max_group_size]
            if not chunk:
                continue

            chunk_indices = [idx for idx, _ in chunk]
            chunk_records = [r for _, r in chunk]

            confidences: list[float] = []
            for r in chunk_records:
                try:
                    confidences.append(float(r.get("model_confidence", 0.0)))
                except (ValueError, TypeError):
                    confidences.append(0.0)

            min_conf = min(confidences) if confidences else 0.0
            max_conf = max(confidences) if confidences else 0.0

            # Aggregate risk flags
            all_flags: set[str] = set()
            for r in chunk_records:
                rf = r.get("risk_flags")
                if rf:
                    if isinstance(rf, list):
                        all_flags.update(str(f).strip() for f in rf)
                    else:
                        all_flags.update(f.strip() for f in str(rf).replace(";", ",").split(",") if f.strip())

            flags_display = sorted(list(all_flags)) if all_flags else ["none"]
            safe_intent_slug = re.sub(r"[^a-zA-Z0-9_]", "_", intent)
            grp_id = f"grp_{safe_intent_slug}_{chunk_idx + 1:02d}"

            group = SafeReviewGroup(
                group_id=grp_id,
                suggested_label=intent,
                record_indices=chunk_indices,
                records=chunk_records,
                min_confidence=min_conf,
                max_confidence=max_conf,
                risk_flags=flags_display,
                size=len(chunk),
            )
            groups.append(group)

    return groups

