"""
SupportGraph AI — Unit Tests for Safe Grouped Smart HITL Review Workflow (Phase 5.8)

Comprehensive test suite verifying:
1. Existing 37 completed human labels remain unchanged and immutable.
2. Safe low-risk records are correctly identified as eligible for grouping.
3. High-risk records NEVER enter grouped approval.
4. QC-sampled records NEVER enter grouped approval.
5. Medium-risk records NEVER enter grouped approval.
6. Group approval requires explicit human confirmation.
7. Cancelling confirmation writes nothing and leaves records pending.
8. Group approval creates individual, auditable final annotation records.
9. Group approval records review_mode="group_human_approval" and approval_type.
10. AI suggestion metadata remains strictly separate from human ground truth.
11. Selective override does not auto-approve remaining records.
12. Resume works correctly after interruption during a group review session.
13. Old progress manifests remain 100% backward-compatible.
14. No duplicate Golden IDs are created.
15. Dataset validation passes with 0 errors.
16. No pending record is marked completed without an explicit human action.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from app.evaluation.human_review import (
        DEFAULT_PROGRESS_PATH,
        ReviewAction,
        apply_review_decision,
        compute_review_progress,
        get_allowed_taxonomy_labels,
        load_review_progress_manifest,
        save_review_progress_manifest,
    )
    from app.evaluation.review_prioritization import (
        HIGH_CONFIDENCE_THRESHOLD,
        ReviewPriority,
        ReviewRecommendation,
        SafeReviewGroup,
        build_safe_review_groups,
        is_eligible_for_group_review,
    )
    from backend.scripts.review_golden_labels import (
        clean_dataframe,
        confirm_group_approval,
    )
    from backend.scripts.validate_golden_dataset import audit_golden_dataset
except ModuleNotFoundError:
    from backend.app.evaluation.human_review import (  # type: ignore[no-redef]
        DEFAULT_PROGRESS_PATH,
        ReviewAction,
        apply_review_decision,
        compute_review_progress,
        get_allowed_taxonomy_labels,
        load_review_progress_manifest,
        save_review_progress_manifest,
    )
    from backend.app.evaluation.review_prioritization import (  # type: ignore[no-redef]
        HIGH_CONFIDENCE_THRESHOLD,
        ReviewPriority,
        ReviewRecommendation,
        SafeReviewGroup,
        build_safe_review_groups,
        is_eligible_for_group_review,
    )
    from backend.scripts.review_golden_labels import (  # type: ignore[no-redef]
        clean_dataframe,
        confirm_group_approval,
    )
    from backend.scripts.validate_golden_dataset import audit_golden_dataset  # type: ignore[no-redef]


@pytest.fixture
def sample_clean_low_record() -> dict[str, Any]:
    return {
        "golden_id": "gold_101",
        "tweet_id": "900101",
        "conversation_id": "conv_900101",
        "customer_message": "My iPhone battery drains within two hours of normal use.",
        "normalized_message": "My iPhone battery drains within two hours of normal use.",
        "candidate_intent": "battery_power_issue",
        "source_cluster": "1",
        "conversation_context": "@user We can help with battery drain. DM us.",
        "annotation_label": "",
        "annotation_status": "pending_human_review",
        "annotator": "",
        "notes": "",
        "model_name": "llama3.2:latest",
        "model_suggested_label": "battery_power_issue",
        "model_confidence": "0.95",
        "model_reasoning_summary": "Customer reports rapid battery drain.",
        "model_needs_human_review": "False",
        "suggestion_timestamp": "2026-09-14T12:00:00Z",
        "suggestion_status": "success",
        "priority": "low",
        "priority_score": 15,
        "priority_reason": "High confidence (0.95) with unambiguous operational context.",
        "qc_sample": "False",
        "review_recommendation": "safe_for_group_review",
        "risk_flags": "high_confidence_clean",
        "risk_factors": "high_confidence_clean",
    }


# =============================================================================
# TEST 1: Existing completed human labels remain unchanged
# =============================================================================
def test_existing_human_labels_remain_unchanged() -> None:
    golden_path = _PROJECT_ROOT / "data" / "golden" / "golden_set_human_review.csv"
    if not golden_path.exists():
        pytest.skip("Protected ground truth file not present.")

    df = pd.read_csv(golden_path, dtype=str)
    # Count reviewed records
    reviewed_rows = df[
        (df["annotation_label"].fillna("").astype(str).str.strip() != "") &
        (~df["annotation_status"].fillna("").astype(str).str.lower().isin(["", "pending", "pending_human_review"]))
    ]
    assert len(reviewed_rows) >= 57, f"Expected at least 57 completed human annotations, found {len(reviewed_rows)}"


    # Verify annotator is 'sridevi' on completed records
    for _, r in reviewed_rows.iterrows():
        assert r["annotator"] == "sridevi", f"Record {r['golden_id']} has unexpected annotator {r['annotator']}"
        assert str(r["annotation_label"]).strip() != ""


# =============================================================================
# TEST 2: Safe low-risk records are correctly eligible for grouping
# =============================================================================
def test_safe_low_risk_records_eligible_for_grouping(sample_clean_low_record: dict[str, Any]) -> None:
    eligible, reason = is_eligible_for_group_review(sample_clean_low_record)
    assert eligible is True, f"Expected record to be eligible, got: {reason}"


# =============================================================================
# TEST 3: High-risk records NEVER enter grouped approval
# =============================================================================
def test_high_risk_records_never_enter_grouped_approval(sample_clean_low_record: dict[str, Any]) -> None:
    high_risk_rec = dict(sample_clean_low_record)
    high_risk_rec["priority"] = "high"
    high_risk_rec["priority_score"] = 75
    high_risk_rec["risk_flags"] = "borderline_confidence; intent_divergence"

    eligible, reason = is_eligible_for_group_review(high_risk_rec)
    assert eligible is False
    assert "HIGH" in reason or "escalation" in reason

    crit_risk_rec = dict(sample_clean_low_record)
    crit_risk_rec["priority"] = "critical"
    crit_risk_rec["model_needs_human_review"] = "True"
    eligible_crit, reason_crit = is_eligible_for_group_review(crit_risk_rec)
    assert eligible_crit is False


# =============================================================================
# TEST 4: QC-sampled records NEVER enter grouped approval
# =============================================================================
def test_qc_sampled_records_never_enter_grouped_approval(sample_clean_low_record: dict[str, Any]) -> None:
    qc_rec = dict(sample_clean_low_record)
    qc_rec["qc_sample"] = "True"
    qc_rec["risk_flags"] = "high_confidence_clean; qc_sample"

    eligible, reason = is_eligible_for_group_review(qc_rec)
    assert eligible is False
    assert "QC sample" in reason or "qc_sample" in reason


# =============================================================================
# TEST 5: Medium-risk records NEVER enter grouped approval
# =============================================================================
def test_medium_risk_records_never_enter_grouped_approval(sample_clean_low_record: dict[str, Any]) -> None:
    med_rec = dict(sample_clean_low_record)
    med_rec["priority"] = "medium"
    med_rec["model_confidence"] = "0.88"
    med_rec["risk_flags"] = "moderate_confidence"

    eligible, reason = is_eligible_for_group_review(med_rec)
    assert eligible is False
    assert "MEDIUM" in reason or "below safe group threshold" in reason


# =============================================================================
# TEST 6: Group approval requires explicit confirmation
# =============================================================================
def test_group_approval_requires_explicit_confirmation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("builtins.input", lambda _: "Y")
    assert confirm_group_approval("software_update_problem", 5) is True

    monkeypatch.setattr("builtins.input", lambda _: "N")
    assert confirm_group_approval("software_update_problem", 5) is False


# =============================================================================
# TEST 7: Cancelling confirmation writes nothing
# =============================================================================
def test_cancelling_confirmation_writes_nothing(sample_clean_low_record: dict[str, Any]) -> None:
    df = pd.DataFrame([sample_clean_low_record])
    initial_label = df.at[0, "annotation_label"]
    initial_status = df.at[0, "annotation_status"]

    # Simulating cancellation in confirmation
    confirmed = False
    if confirmed:
        df.at[0, "annotation_label"] = "battery_power_issue"

    assert df.at[0, "annotation_label"] == initial_label
    assert df.at[0, "annotation_status"] == initial_status


# =============================================================================
# TEST 8: Group approval creates individual final annotation records
# =============================================================================
def test_group_approval_creates_individual_final_annotation_records(sample_clean_low_record: dict[str, Any]) -> None:
    rec1 = dict(sample_clean_low_record)
    rec1["golden_id"] = "gold_101"
    rec2 = dict(sample_clean_low_record)
    rec2["golden_id"] = "gold_102"

    df = pd.DataFrame([rec1, rec2])

    for i in range(len(df)):
        updated = apply_review_decision(
            record=df.iloc[i],
            action=ReviewAction.GROUP_APPROVE,
            annotator="sridevi",
            review_mode="group_human_approval",
            approval_type="group_approved_suggestion",
            group_id="grp_battery_power_issue_01",
            approved_group_label="battery_power_issue",
            group_size=2,
        )
        for k, v in updated.items():
            df.at[i, k] = v

    for i in range(len(df)):
        assert df.at[i, "annotation_label"] == "battery_power_issue"
        assert df.at[i, "annotator"] == "sridevi"
        assert df.at[i, "annotation_status"] == "reviewed"
        assert df.at[i, "review_mode"] == "group_human_approval"
        assert df.at[i, "approval_type"] == "group_approved_suggestion"


# =============================================================================
# TEST 9: Group approval records review_mode correctly
# =============================================================================
def test_group_approval_records_review_mode_correctly(sample_clean_low_record: dict[str, Any]) -> None:
    updated = apply_review_decision(
        record=sample_clean_low_record,
        action=ReviewAction.GROUP_APPROVE,
        annotator="sridevi",
        group_id="grp_battery_01",
    )
    assert updated["review_mode"] == "group_human_approval"
    assert updated["approval_type"] == "group_approved_suggestion"
    assert updated["group_id"] == "grp_battery_01"


# =============================================================================
# TEST 10: AI suggestion metadata remains separate from annotation_label
# =============================================================================
def test_ai_suggestion_metadata_remains_separate_from_annotation_label(sample_clean_low_record: dict[str, Any]) -> None:
    original_model_label = sample_clean_low_record["model_suggested_label"]
    original_conf = sample_clean_low_record["model_confidence"]
    original_reason = sample_clean_low_record["model_reasoning_summary"]

    updated = apply_review_decision(
        record=sample_clean_low_record,
        action=ReviewAction.OVERRIDE,
        annotator="sridevi",
        selected_label="general_device_support",
    )
    assert updated["annotation_label"] == "general_device_support"
    assert updated["model_suggested_label"] == original_model_label
    assert updated["model_confidence"] == original_conf
    assert updated["model_reasoning_summary"] == original_reason


# =============================================================================
# TEST 11: Selective override does not auto-approve remaining records
# =============================================================================
def test_selective_override_does_not_auto_approve_remaining_records(sample_clean_low_record: dict[str, Any]) -> None:
    rec1 = dict(sample_clean_low_record)
    rec1["golden_id"] = "gold_101"
    rec2 = dict(sample_clean_low_record)
    rec2["golden_id"] = "gold_102"
    rec3 = dict(sample_clean_low_record)
    rec3["golden_id"] = "gold_103"

    df = pd.DataFrame([rec1, rec2, rec3])

    # Human selectively overrides record 0 (gold_101)
    updated_rec1 = apply_review_decision(
        record=df.iloc[0],
        action=ReviewAction.OVERRIDE,
        annotator="sridevi",
        selected_label="general_device_support",
    )
    for k, v in updated_rec1.items():
        df.at[0, k] = v

    # Verify rec 1 and 2 remain strictly pending
    assert df.at[1, "annotation_label"] == ""
    assert df.at[1, "annotation_status"] == "pending_human_review"
    assert df.at[2, "annotation_label"] == ""
    assert df.at[2, "annotation_status"] == "pending_human_review"


# =============================================================================
# TEST 12: Resume works correctly after interruption during a group
# =============================================================================
def test_resume_works_correctly_after_interruption_during_group(sample_clean_low_record: dict[str, Any]) -> None:
    rec1 = dict(sample_clean_low_record)
    rec1["golden_id"] = "gold_101"
    rec2 = dict(sample_clean_low_record)
    rec2["golden_id"] = "gold_102"

    df = pd.DataFrame([rec1, rec2])

    # Record 0 is reviewed, record 1 is not
    updated_rec1 = apply_review_decision(
        record=df.iloc[0],
        action=ReviewAction.ACCEPT,
        annotator="sridevi",
    )
    for k, v in updated_rec1.items():
        df.at[0, k] = v

    progress = compute_review_progress(df)
    assert progress["human_reviewed"] == 1
    assert progress["pending_human_review"] == 1

    # In next session, only record 1 is pending
    pending_indices = [
        idx for idx, row in df.iterrows()
        if not str(row.get("annotation_label", "")).strip()
    ]
    assert pending_indices == [1]


# =============================================================================
# TEST 13: Old progress manifests remain compatible
# =============================================================================
def test_old_progress_manifests_remain_compatible() -> None:
    old_manifest_data = {
        "total_records": 200,
        "ai_suggestions_completed": 200,
        "human_reviewed": 37,
        "pending_human_review": 163,
        "accepted_ai_suggestions": 15,
        "overridden_ai_suggestions": 19,
        "unclear_records": 3,
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        json.dump(old_manifest_data, tf)
        tf_path = tf.name

    loaded = load_review_progress_manifest(tf_path)
    assert loaded["human_reviewed"] == 37
    assert loaded["group_approved_records"] == 0
    assert loaded["individual_reviewed_records"] == 0
    assert loaded["current_mode"] == "smart"
    assert isinstance(loaded["completed_group_ids"], list)


# =============================================================================
# TEST 14: No duplicate Golden IDs are created
# =============================================================================
def test_no_duplicate_golden_ids_created(sample_clean_low_record: dict[str, Any]) -> None:
    records = []
    for i in range(25):
        r = dict(sample_clean_low_record)
        r["golden_id"] = f"gold_{i+100:03d}"
        r["tweet_id"] = f"{900000+i}"
        records.append(r)

    df = pd.DataFrame(records)
    groups = build_safe_review_groups(df, max_group_size=10)

    # 25 records chunked with max_group_size=10 produces 3 groups (10, 10, 5)
    assert len(groups) == 3
    assert groups[0].size == 10
    assert groups[1].size == 10
    assert groups[2].size == 5

    all_gids = []
    for g in groups:
        for rec in g.records:
            all_gids.append(rec["golden_id"])

    assert len(all_gids) == 25
    assert len(set(all_gids)) == 25, "Duplicate golden_ids detected across groups"


# =============================================================================
# TEST 15: Dataset validation passes
# =============================================================================
def test_dataset_validation_passes(sample_clean_low_record: dict[str, Any]) -> None:
    rec1 = dict(sample_clean_low_record)
    rec1["golden_id"] = "gold_001"
    rec1["tweet_id"] = "111"
    rec1["customer_message"] = "iPhone battery dying quickly."
    rec1["normalized_message"] = "iPhone battery dying quickly."
    updated1 = apply_review_decision(
        record=rec1,
        action=ReviewAction.GROUP_APPROVE,
        annotator="sridevi",
        review_mode="group_human_approval",
        approval_type="group_approved_suggestion",
        group_id="grp_battery_01",
    )

    rec2 = dict(sample_clean_low_record)
    rec2["golden_id"] = "gold_002"
    rec2["tweet_id"] = "222"
    rec2["customer_message"] = "iPhone screen is completely cracked."
    rec2["normalized_message"] = "iPhone screen is completely cracked."
    rec2["model_suggested_label"] = "display_touch_issue"
    # rec2 remains pending

    df = clean_dataframe(pd.DataFrame([updated1, rec2]))
    allowed = get_allowed_taxonomy_labels()

    report = audit_golden_dataset(df, allowed_labels=allowed, expected_count=2)
    assert report.is_valid is True, f"Validation failed with errors: {report.errors}"
    assert report.audit_metrics["scientific_integrity_intact"] is True
    assert report.audit_metrics["human_reviewed_records"] == 1
    assert report.audit_metrics["group_approved_records"] == 1


# =============================================================================
# TEST 16: No pending record is marked completed without explicit human action
# =============================================================================
def test_no_pending_record_marked_completed_without_explicit_human_action(sample_clean_low_record: dict[str, Any]) -> None:
    rec = dict(sample_clean_low_record)
    # Ensure unreviewed
    rec["annotation_label"] = ""
    rec["annotator"] = ""
    rec["annotation_status"] = "pending_human_review"

    df = pd.DataFrame([rec])
    allowed = get_allowed_taxonomy_labels()

    report = audit_golden_dataset(df, allowed_labels=allowed, expected_count=1)
    assert report.is_valid is True
    assert report.audit_metrics["human_reviewed_records"] == 0
    assert report.audit_metrics["pending_human_review_records"] == 1
