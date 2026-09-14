"""
SupportGraph AI — Unit Tests for Review Prioritization & Golden Set Validation (Phase 5.6)

Tests:
1. Priority classification rules (CRITICAL, HIGH, MEDIUM, LOW)
2. Quality Control (QC) sampling determinism with fixed random seed
3. Queue ordering (CRITICAL -> HIGH -> MEDIUM -> QC LOW -> REMAINING LOW)
4. Row preservation (200 records in -> 200 records out)
5. Scientific integrity guarantees (no automatic AI-to-ground-truth copying)
6. Dataset audit engine and error detection
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from app.evaluation.human_review import (
        InvalidReviewActionError,
        ReviewAction,
        apply_review_decision,
    )
    from app.evaluation.review_prioritization import (
        HIGH_CONFIDENCE_THRESHOLD,
        LOW_CONFIDENCE_THRESHOLD,
        MEDIUM_CONFIDENCE_THRESHOLD,
        QC_RANDOM_SEED,
        LanguageEvidence,
        RecordPriority,
        ReviewPriority,
        apply_qc_sampling,
        build_prioritized_review_queue,
        detect_multi_domain_overlap,
        detect_non_english_evidence,
        detect_non_english_heuristic,
        evaluate_record_priority,
    )
    from backend.scripts.validate_golden_dataset import audit_golden_dataset
except ModuleNotFoundError:
    from backend.app.evaluation.human_review import (  # type: ignore[no-redef]
        InvalidReviewActionError,
        ReviewAction,
        apply_review_decision,
    )
    from backend.app.evaluation.review_prioritization import (  # type: ignore[no-redef]
        HIGH_CONFIDENCE_THRESHOLD,
        LOW_CONFIDENCE_THRESHOLD,
        MEDIUM_CONFIDENCE_THRESHOLD,
        QC_RANDOM_SEED,
        LanguageEvidence,
        RecordPriority,
        ReviewPriority,
        apply_qc_sampling,
        build_prioritized_review_queue,
        detect_multi_domain_overlap,
        detect_non_english_evidence,
        detect_non_english_heuristic,
        evaluate_record_priority,
    )
    from backend.scripts.validate_golden_dataset import audit_golden_dataset  # type: ignore[no-redef]
    from backend.app.evaluation.taxonomy_calibration import evaluate_taxonomy_calibration  # type: ignore[no-redef]


@pytest.fixture
def base_clean_record() -> dict[str, str]:
    return {
        "golden_id": "gold_001",
        "tweet_id": "1141068",
        "conversation_id": "conv_1141068",
        "customer_message": "My iPhone battery is draining extremely fast after updating to the latest iOS.",
        "normalized_message": "<USER> My iPhone battery is draining extremely fast after updating to the latest iOS.",
        "candidate_intent": "battery_power_issue",
        "source_cluster": "3",
        "conversation_context": "@user We can help with battery issues. Send us a DM.",
        "annotation_label": "",
        "annotation_status": "pending_human_review",
        "annotator": "",
        "notes": "",
        "model_name": "llama3.2:latest",
        "model_suggested_label": "battery_power_issue",
        "model_confidence": "0.95",
        "model_reasoning_summary": "Customer complains of severe battery drain.",
        "model_needs_human_review": "False",
        "suggestion_timestamp": "2026-09-14T12:00:00Z",
        "suggestion_status": "success",
    }


# =============================================================================
# 1. PRIORITY CLASSIFICATION TESTS
# =============================================================================

def test_clean_high_confidence_is_low_priority(base_clean_record: dict[str, str]) -> None:
    """Verify that a high-confidence, clean record receives LOW priority."""
    prio = evaluate_record_priority(base_clean_record)
    assert prio.priority == ReviewPriority.LOW
    assert prio.priority_score <= 34
    assert "High confidence" in prio.priority_reason


def test_needs_human_review_flag_triggers_critical(base_clean_record: dict[str, str]) -> None:
    """Verify that model_needs_human_review=True elevates priority to CRITICAL."""
    rec = dict(base_clean_record)
    rec["model_needs_human_review"] = "True"
    rec["model_confidence"] = "0.95"

    prio = evaluate_record_priority(rec)
    assert prio.priority == ReviewPriority.CRITICAL
    assert prio.priority_score >= 85
    assert "mandatory review" in prio.priority_reason


def test_low_confidence_triggers_critical(base_clean_record: dict[str, str]) -> None:
    """Verify that confidence below LOW_CONFIDENCE_THRESHOLD triggers CRITICAL."""
    rec = dict(base_clean_record)
    rec["model_confidence"] = "0.45"

    prio = evaluate_record_priority(rec)
    assert prio.priority == ReviewPriority.CRITICAL
    assert prio.priority_score >= 85
    assert "Low AI confidence: 0.45" in prio.priority_reason


def test_failed_suggestion_status_triggers_critical(base_clean_record: dict[str, str]) -> None:
    """Verify that failed/invalid suggestion status triggers CRITICAL."""
    rec = dict(base_clean_record)
    rec["suggestion_status"] = "failed"
    rec["model_suggested_label"] = ""

    prio = evaluate_record_priority(rec)
    assert prio.priority == ReviewPriority.CRITICAL
    assert prio.priority_score == 100


def test_unclear_needs_review_suggestion_triggers_critical(base_clean_record: dict[str, str]) -> None:
    """Verify that AI suggestion 'unclear_needs_review' triggers CRITICAL."""
    rec = dict(base_clean_record)
    rec["model_suggested_label"] = "unclear_needs_review"
    rec["model_confidence"] = "0.95"

    prio = evaluate_record_priority(rec)
    assert prio.priority == ReviewPriority.CRITICAL
    assert prio.priority_score >= 90
    assert "unclear_needs_review" in prio.priority_reason


def test_missing_or_nan_label_triggers_critical(base_clean_record: dict[str, str]) -> None:
    """Verify that empty, 'nan', or unauthorized labels trigger CRITICAL."""
    rec = dict(base_clean_record)
    rec["model_suggested_label"] = "nan"

    prio = evaluate_record_priority(rec)
    assert prio.priority == ReviewPriority.CRITICAL
    assert prio.priority_score == 100


def test_borderline_confidence_triggers_high_priority(base_clean_record: dict[str, str]) -> None:
    """Verify that confidence in range [0.70, 0.85) triggers HIGH priority."""
    rec = dict(base_clean_record)
    rec["model_confidence"] = "0.78"

    prio = evaluate_record_priority(rec)
    assert prio.priority == ReviewPriority.HIGH
    assert 60 <= prio.priority_score <= 84
    assert "Borderline confidence" in prio.priority_reason


def test_non_english_heuristic_triggers_high_priority(base_clean_record: dict[str, str]) -> None:
    """Verify that non-English messages trigger HIGH priority."""
    rec = dict(base_clean_record)
    rec["customer_message"] = "Hola @AppleSupport mi pantalla está rota y no funciona el táctil por favor ayuda."
    rec["normalized_message"] = "<USER> mi pantalla está rota y no funciona el táctil por favor ayuda."
    rec["model_confidence"] = "0.95"

    prio = evaluate_record_priority(rec)
    assert prio.priority == ReviewPriority.HIGH
    assert "true_language_uncertainty" in prio.risk_flags
    assert "Language uncertainty" in prio.priority_reason


def test_intent_divergence_triggers_high_priority(base_clean_record: dict[str, str]) -> None:
    """Verify that divergence between cluster candidate intent and AI suggestion triggers HIGH."""
    rec = dict(base_clean_record)
    rec["candidate_intent"] = "keyboard_typing_issue"
    rec["model_suggested_label"] = "general_device_support"
    rec["model_confidence"] = "0.88"

    prio = evaluate_record_priority(rec)
    assert prio.priority == ReviewPriority.HIGH
    assert "intent_divergence" in prio.risk_flags
    assert "Intent divergence" in prio.priority_reason


def test_multi_domain_overlap_triggers_high_priority(base_clean_record: dict[str, str]) -> None:
    """Verify that multiple conflicting operational symptoms elevate priority to HIGH when confidence is not overwhelming."""
    rec = dict(base_clean_record)
    rec["customer_message"] = "My battery drain is awful and my touchscreen has unresponsive touch issues."
    rec["normalized_message"] = "<USER> My battery drain is awful and my touchscreen has unresponsive touch issues."
    rec["model_confidence"] = "0.85"

    prio = evaluate_record_priority(rec)
    assert prio.priority == ReviewPriority.HIGH
    assert "multi_intent_overlap" in prio.risk_flags
    assert "Multi-intent overlap" in prio.priority_reason


def test_moderate_confidence_triggers_medium_priority(base_clean_record: dict[str, str]) -> None:
    """Verify that confidence in range [0.85, 0.90) without other risks triggers MEDIUM."""
    rec = dict(base_clean_record)
    rec["model_confidence"] = "0.87"
    rec["candidate_intent"] = "battery_power_issue"

    prio = evaluate_record_priority(rec)
    assert prio.priority == ReviewPriority.MEDIUM
    assert 35 <= prio.priority_score <= 59
    assert "moderate_confidence" in prio.risk_flags
    assert "Moderate confidence" in prio.priority_reason


# =============================================================================
# 1B. REGRESSION TESTS: FALSE POSITIVE PREVENTION & LANGUAGE EVIDENCE
# =============================================================================

@pytest.mark.parametrize(
    "english_text",
    [
        "my phone has a problem",
        "this update caused a problem",
        "my iPhone has an issue after the update",
        "please fix this letter eye problem",
        "photos not syncing properly between OSX high Sierra and iOS. Same problem",
        "there’s been a problem with my sisters iPhone 7 like it won’t turn on or charge",
        "Formated and reinstalled MacOS High Sierra by Genius Bar member at Apple Shop but still having problems",
        "so my iPhone crashes touch seems to stop working altho I can access the control center without any problem",
        "Looks like its a common problem with the High Sierra download as there are many reports online",
        "still having the word problem, auto correct changing is to I.S ?! #iOS11",
    ],
)
def test_english_messages_with_problem_issue_update_do_not_trigger_language_risk(
    base_clean_record: dict[str, str], english_text: str
) -> None:
    """
    REGRESSION TEST: Verify that ordinary English messages containing words like 'problem',
    'issue', 'update', 'phone' do NOT trigger language uncertainty or escalate priority.
    """
    evidence = detect_non_english_evidence(english_text)
    assert evidence.detected is False
    assert evidence.strength == "none"

    is_foreign, reason = detect_non_english_heuristic(english_text)
    assert is_foreign is False
    assert reason == ""

    rec = dict(base_clean_record)
    rec["customer_message"] = english_text
    rec["normalized_message"] = f"<USER> {english_text}"
    rec["model_confidence"] = "0.95"
    calib = evaluate_taxonomy_calibration(rec)
    if calib.candidate_label:
        rec["model_suggested_label"] = calib.candidate_label
        rec["candidate_intent"] = calib.candidate_label

    prio = evaluate_record_priority(rec)
    assert prio.priority == ReviewPriority.LOW
    assert "true_language_uncertainty" not in prio.risk_flags
    assert "Language uncertainty" not in prio.priority_reason
    assert "Non-English" not in prio.priority_reason


@pytest.mark.parametrize(
    "foreign_text, expected_lang",
    [
        ("Espero que con la actualización mi iPhone mejore", "Spanish"),
        ("Und ich dachte mein MacBook bleibt frei von irgendwelchen Problemen. Liebes @AppleSupport", "German"),
        ("Bonjour @AppleSupport ma batterie se décharge très vite après la mise à jour s'il vous plaît", "French"),
        ("Olá @AppleSupport minha bateria descarrega muito rápido depois da atualização por favor ajudem", "Portuguese"),
        ("Ciao @AppleSupport il mio schermo non funziona dopo l'aggiornamento per favore aiuto", "Italian"),
    ],
)
def test_genuine_non_english_messages_detected_appropriately(
    base_clean_record: dict[str, str], foreign_text: str, expected_lang: str
) -> None:
    """
    Verify that genuine non-English texts are accurately detected with moderate/strong evidence.
    """
    evidence = detect_non_english_evidence(foreign_text)
    assert evidence.detected is True
    assert evidence.strength in ("moderate", "strong")

    is_foreign, reason = detect_non_english_heuristic(foreign_text)
    assert is_foreign is True
    assert len(reason) > 0

    rec = dict(base_clean_record)
    rec["customer_message"] = foreign_text
    rec["normalized_message"] = f"<USER> {foreign_text}"
    rec["model_confidence"] = "0.95"

    prio = evaluate_record_priority(rec)
    assert prio.priority == ReviewPriority.HIGH
    assert "true_language_uncertainty" in prio.risk_flags
    assert "Language uncertainty" in prio.priority_reason


def test_non_latin_script_characters_trigger_strong_evidence(base_clean_record: dict[str, str]) -> None:
    """Verify that non-Latin script characters (e.g. Cyrillic, Arabic, CJK) trigger strong language evidence."""
    cyrillic_msg = "У меня перестал работать экран после обновления iOS на iPhone"
    evidence = detect_non_english_evidence(cyrillic_msg)
    assert evidence.detected is True
    assert evidence.strength == "strong"
    assert "non-Latin" in (evidence.reason or "")

    rec = dict(base_clean_record)
    rec["customer_message"] = cyrillic_msg
    rec["model_confidence"] = "0.95"
    prio = evaluate_record_priority(rec)
    assert prio.priority == ReviewPriority.HIGH
    assert "true_language_uncertainty" in prio.risk_flags


def test_weak_language_evidence_does_not_escalate_priority(base_clean_record: dict[str, str]) -> None:
    """
    Verify that a single isolated foreign token in a clear English context is marked 'weak'
    and does NOT escalate a high-confidence record to HIGH priority.
    """
    mixed_msg = "I said merci to the technician at the store who fixed my screen."
    evidence = detect_non_english_evidence(mixed_msg)
    assert evidence.detected is True
    assert evidence.strength == "weak"

    is_foreign, _ = detect_non_english_heuristic(mixed_msg)
    assert is_foreign is False

    rec = dict(base_clean_record)
    rec["customer_message"] = mixed_msg
    rec["model_confidence"] = "0.95"

    prio = evaluate_record_priority(rec)
    assert prio.priority == ReviewPriority.LOW
    assert "true_language_uncertainty" not in prio.risk_flags


def test_structured_risk_flags_population(base_clean_record: dict[str, str]) -> None:
    """Verify that explicit deterministic risk flags are correctly populated across categories."""
    # 1. Low confidence
    rec_low = dict(base_clean_record, model_confidence="0.65")
    prio_low = evaluate_record_priority(rec_low)
    assert "low_confidence" in prio_low.risk_flags
    assert prio_low.priority == ReviewPriority.CRITICAL

    # 2. Borderline confidence
    rec_border = dict(base_clean_record, model_confidence="0.75")
    prio_border = evaluate_record_priority(rec_border)
    assert "borderline_confidence" in prio_border.risk_flags
    assert prio_border.priority == ReviewPriority.HIGH

    # 3. Model requested review
    rec_req = dict(base_clean_record, model_needs_human_review="True")
    prio_req = evaluate_record_priority(rec_req)
    assert "model_requested_review" in prio_req.risk_flags
    assert prio_req.priority == ReviewPriority.CRITICAL

    # 4. Clean record
    rec_clean = dict(base_clean_record, model_confidence="0.95")
    prio_clean = evaluate_record_priority(rec_clean)
    assert "high_confidence_clean" in prio_clean.risk_flags
    assert prio_clean.priority == ReviewPriority.LOW


# =============================================================================
# 2. DETERMINISM & QC SAMPLING TESTS
# =============================================================================

def test_priority_evaluation_is_deterministic(base_clean_record: dict[str, str]) -> None:
    """Verify that calling evaluate_record_priority multiple times produces identical results."""
    p1 = evaluate_record_priority(base_clean_record)
    p2 = evaluate_record_priority(base_clean_record)
    assert p1.priority == p2.priority
    assert p1.priority_score == p2.priority_score
    assert p1.priority_reason == p2.priority_reason


def test_qc_sampling_is_deterministic_and_reproducible() -> None:
    """Verify that QC sampling with a fixed seed produces exactly the same sample."""
    records = [
        RecordPriority(golden_id=f"gold_{i:03d}", priority=ReviewPriority.LOW, priority_score=20, priority_reason="Clean")
        for i in range(1, 51)
    ]
    sampled_1 = apply_qc_sampling(records, sample_percentage=0.15, seed=42)
    sampled_2 = apply_qc_sampling(records, sample_percentage=0.15, seed=42)

    qc_ids_1 = [r.golden_id for r in sampled_1 if r.qc_sample]
    qc_ids_2 = [r.golden_id for r in sampled_2 if r.qc_sample]

    assert len(qc_ids_1) == 8  # 15% of 50 = ceil(7.5) = 8
    assert qc_ids_1 == qc_ids_2


def test_qc_sampling_only_samples_low_priority_records() -> None:
    """Verify that only LOW priority records are eligible for QC sampling."""
    records = [
        RecordPriority(golden_id="gold_crit", priority=ReviewPriority.CRITICAL, priority_score=95, priority_reason="Crit"),
        RecordPriority(golden_id="gold_high", priority=ReviewPriority.HIGH, priority_score=75, priority_reason="High"),
        RecordPriority(golden_id="gold_med", priority=ReviewPriority.MEDIUM, priority_score=45, priority_reason="Med"),
        RecordPriority(golden_id="gold_low1", priority=ReviewPriority.LOW, priority_score=20, priority_reason="Low 1"),
        RecordPriority(golden_id="gold_low2", priority=ReviewPriority.LOW, priority_score=15, priority_reason="Low 2"),
    ]
    sampled = apply_qc_sampling(records, sample_size=2, seed=42)
    for r in sampled:
        if r.priority != ReviewPriority.LOW:
            assert r.qc_sample is False


# =============================================================================
# 3. REVIEW QUEUE GENERATOR TESTS
# =============================================================================

def test_build_prioritized_review_queue_preserves_all_records() -> None:
    """Verify queue builder retains all 200 records without dropping or duplicating."""
    rows = []
    for i in range(1, 201):
        rows.append({
            "golden_id": f"gold_{i:03d}",
            "tweet_id": str(1000000 + i),
            "conversation_id": f"conv_{1000000 + i}",
            "customer_message": f"Sample customer message {i}",
            "normalized_message": f"<USER> Sample customer message {i}",
            "candidate_intent": "battery_power_issue",
            "source_cluster": "1",
            "conversation_context": "No brand response observed",
            "annotation_label": "",
            "annotation_status": "pending_human_review",
            "annotator": "",
            "notes": "",
            "model_name": "llama3.2:latest",
            "model_suggested_label": "battery_power_issue" if i % 2 == 0 else "display_touch_issue",
            "model_confidence": "0.95" if i > 20 else ("0.50" if i <= 10 else "0.75"),
            "model_reasoning_summary": "Test summary",
            "model_needs_human_review": "True" if i == 5 else "False",
            "suggestion_timestamp": "2026-09-14T12:00:00Z",
            "suggestion_status": "success",
        })

    df = pd.DataFrame(rows)
    queue_df, summary = build_prioritized_review_queue(df, seed=42)

    assert len(queue_df) == 200
    assert summary["total_records"] == 200
    assert queue_df["golden_id"].nunique() == 200
    assert "priority" in queue_df.columns
    assert "priority_score" in queue_df.columns
    assert "priority_reason" in queue_df.columns
    assert "qc_sample" in queue_df.columns

    # Verify critical items appear first
    first_priority = queue_df.iloc[0]["priority"]
    assert first_priority == "critical"


def test_queue_sorting_hierarchy() -> None:
    """Verify the strict sorting hierarchy: CRITICAL -> HIGH -> MEDIUM -> QC LOW -> REMAINING LOW."""
    rows = [
        # Record 1: Low priority
        {
            "golden_id": "g_low",
            "customer_message": "My iPhone battery life has been fantastic until this morning when it suddenly shut down unexpectedly.",
            "model_suggested_label": "battery_power_issue",
            "model_confidence": "0.95",
            "model_needs_human_review": "False",
            "suggestion_status": "success",
            "candidate_intent": "battery_power_issue",
            "conversation_context": "@user Please send us a DM.",
        },
        # Record 2: Critical priority
        {
            "golden_id": "g_crit",
            "customer_message": "I am not sure what is happening with this device.",
            "model_suggested_label": "battery_power_issue",
            "model_confidence": "0.95",
            "model_needs_human_review": "True",
            "suggestion_status": "success",
            "candidate_intent": "battery_power_issue",
            "conversation_context": "@user Please send us a DM.",
        },
        # Record 3: High priority
        {
            "golden_id": "g_high",
            "customer_message": "My iPhone has strange issues with charging sometimes.",
            "model_suggested_label": "battery_power_issue",
            "model_confidence": "0.75",
            "model_needs_human_review": "False",
            "suggestion_status": "success",
            "candidate_intent": "battery_power_issue",
            "conversation_context": "@user Please send us a DM.",
        },
        # Record 4: Medium priority
        {
            "golden_id": "g_med",
            "customer_message": "My phone is experiencing moderate battery drain throughout the day during normal usage.",
            "model_suggested_label": "battery_power_issue",
            "model_confidence": "0.88",
            "model_needs_human_review": "False",
            "suggestion_status": "success",
            "candidate_intent": "battery_power_issue",
            "conversation_context": "@user Please send us a DM.",
        },
    ]
    df = pd.DataFrame(rows)
    queue_df, _ = build_prioritized_review_queue(df, seed=42)

    priorities = queue_df["priority"].tolist()
    assert priorities[0] == "critical"
    assert priorities[1] == "high"
    assert priorities[2] == "medium"
    assert priorities[3] == "low"


# =============================================================================
# 4. SCIENTIFIC INTEGRITY & AUDIT TESTS
# =============================================================================

def test_queue_generation_never_populates_annotation_label(base_clean_record: dict[str, str]) -> None:
    """Verify that prioritization and queue building never automatically populate annotation_label."""
    df = pd.DataFrame([base_clean_record])
    queue_df, _ = build_prioritized_review_queue(df)

    assert queue_df.at[0, "annotation_label"] == ""
    assert queue_df.at[0, "annotator"] == ""
    assert queue_df.at[0, "annotation_status"] == "pending_human_review"


def test_audit_golden_dataset_detects_auto_copy_violation(base_clean_record: dict[str, str]) -> None:
    """Verify that audit_golden_dataset catches unreviewed records with non-empty annotation_label."""
    allowed = {"battery_power_issue", "unclear_needs_review"}

    # Violating record: has annotation_label filled, but status is pending_human_review
    bad_rec = dict(base_clean_record)
    bad_rec["annotation_label"] = "battery_power_issue"
    bad_rec["annotation_status"] = "pending_human_review"
    bad_rec["annotator"] = ""

    df = pd.DataFrame([bad_rec])
    report = audit_golden_dataset(df, allowed_labels=allowed, expected_count=1)

    assert report.is_valid is False
    assert any("SCIENTIFIC INTEGRITY VIOLATION" in err for err in report.errors)


def test_audit_golden_dataset_catches_literal_nan_strings(base_clean_record: dict[str, str]) -> None:
    """Verify that audit_golden_dataset catches illegal literal 'nan' strings."""
    allowed = {"battery_power_issue", "unclear_needs_review"}

    bad_rec = dict(base_clean_record)
    bad_rec["annotation_label"] = "nan"
    bad_rec["annotation_status"] = "reviewed"
    bad_rec["annotator"] = "Alice"

    df = pd.DataFrame([bad_rec])
    report = audit_golden_dataset(df, allowed_labels=allowed, expected_count=1)

    assert report.is_valid is False
    assert any("illegal literal string" in err for err in report.errors)


def test_review_recommendation_classification(base_clean_record: dict[str, str]) -> None:
    """Verify review_recommendation values across critical, high, medium, QC-low, and safe-low records."""
    # Critical record
    crit_rec = dict(base_clean_record)
    crit_rec["model_needs_human_review"] = "True"
    prio_crit = evaluate_record_priority(crit_rec)
    assert prio_crit.review_recommendation == "individual_review_required"

    # High record
    high_rec = dict(base_clean_record)
    high_rec["model_confidence"] = "0.75"
    prio_high = evaluate_record_priority(high_rec)
    assert prio_high.review_recommendation == "individual_review_required"

    # Medium record
    med_rec = dict(base_clean_record)
    med_rec["model_confidence"] = "0.88"
    prio_med = evaluate_record_priority(med_rec)
    assert prio_med.review_recommendation == "individual_review_required"

    # Clean Low record (before QC)
    low_rec = dict(base_clean_record)
    low_rec["model_confidence"] = "0.95"
    prio_low = evaluate_record_priority(low_rec)
    assert prio_low.review_recommendation == "safe_for_group_review"

    # QC Sample record
    sampled = apply_qc_sampling([prio_low], sample_size=1, seed=42)
    assert sampled[0].qc_sample is True
    assert sampled[0].review_recommendation == "qc_review_required"


def test_grouped_approval_preserves_annotator_and_label() -> None:
    """Verify that explicit group approval assigns the exact suggested label, annotator, and reviewed status."""
    group_rows = [
        {"golden_id": "gold_101", "model_suggested_label": "battery_power_issue", "suggestion_status": "success", "annotation_label": "", "annotation_status": "pending_human_review", "annotator": ""},
        {"golden_id": "gold_102", "model_suggested_label": "battery_power_issue", "suggestion_status": "success", "annotation_label": "", "annotation_status": "pending_human_review", "annotator": ""},
    ]

    for row in group_rows:
        updated = apply_review_decision(row, action=ReviewAction.ACCEPT, annotator="sridevi")
        assert updated["annotation_label"] == "battery_power_issue"
        assert updated["annotator"] == "sridevi"
        assert updated["annotation_status"] == "reviewed"

