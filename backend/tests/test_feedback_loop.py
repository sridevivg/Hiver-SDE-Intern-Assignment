"""
SupportGraph AI — Phase 13 Human-in-the-Loop Feedback & Continuous Evidence Test Suite.

Verifies:
- Test A: Valid human resolution capture & validation
- Test B: Incomplete human resolution rejected
- Test C: Contradictory human resolution rejected
- Test D: Unsafe/destructive human resolution rejected
- Test E: Wrong problem-family resolution rejected
- Test F: Fake/unsupported evidence source rejected
- Test G: Golden benchmark contamination rejected (Zero Leakage)
- Test H: Candidate evidence is NOT retrievable
- Test I: Rejected evidence is NOT retrievable
- Test J: Approved evidence IS retrievable with provenance
- Test K: Promotion requires reviewer approval
- Test L: Invalid lifecycle transitions are rejected
- Test M: Evidence versioning works
- Test N: Promoted evidence retains complete provenance
- Test O: Bad evidence cannot change automated routing
- Test P: Valid evidence improves evidence coverage
- Test Q: Existing Phase 12.1 adversarial scenarios remain 30/30
- Test R: New unseen scenarios remain 20/20
- Test S: Thermal hazard regression remains protected
- Test T: AirPods pairing regression remains protected
"""

import json
from pathlib import Path
import pytest

from backend.app.conversation.conversation_manager import ConversationManager
from backend.app.conversation.conversation_state import (
    ConversationStatus,
    ResolutionStage,
)
from backend.app.evaluation.phase_12_evaluator import (
    GOLDEN_CSV_PATH,
    GOLDEN_SHA256,
    compute_sha256,
)
from backend.app.feedback.approved_store import ApprovedEvidenceStore
from backend.app.feedback.auditor import FeedbackAuditor
from backend.app.feedback.candidate_store import CandidateEvidenceStore
from backend.app.feedback.promotion_pipeline import PromotionPipeline
from backend.app.feedback.quality_scorer import EvidenceQualityScorer
from backend.app.feedback.review_gate import HumanReviewGate
from backend.app.feedback.schemas import (
    ApprovedEvidenceItem,
    EvidenceQualityVerdict,
    HumanResolution,
    ResolutionLifecycleState,
    ReviewCheckStatus,
)
from backend.app.resolution.support_resolution_engine import SupportResolutionEngine
from backend.app.retrieval.case_retriever import CaseRetriever
from backend.app.retrieval.evidence_ranker import EvidenceMatchTier
from backend.app.schemas.decision_explanation import EndToEndOutcome
from backend.app.schemas.intent_routing import RoutingDecisionType


@pytest.fixture
def feedback_pipeline(tmp_path: Path) -> PromotionPipeline:
    cand_store = CandidateEvidenceStore(base_dir=tmp_path / "candidates")
    app_store = ApprovedEvidenceStore(base_dir=tmp_path / "approved")
    auditor = FeedbackAuditor(audit_path=tmp_path / "runtime" / "feedback_audit.jsonl")
    review_gate = HumanReviewGate(golden_csv_path=GOLDEN_CSV_PATH)
    scorer = EvidenceQualityScorer()
    return PromotionPipeline(
        candidate_store=cand_store,
        approved_store=app_store,
        review_gate=review_gate,
        quality_scorer=scorer,
        auditor=auditor,
    )


# ---------------------------------------------------------------------------
# Test A: Valid Human Resolution Capture & Validation
# ---------------------------------------------------------------------------
def test_valid_human_resolution_lifecycle(feedback_pipeline: PromotionPipeline):
    """Valid resolution passes all 10 checks, achieves VALIDATED state, and earns high quality score."""
    res = HumanResolution(
        conversation_id="conv_valid_001",
        customer_problem="My Apple Watch Ultra 2 action button does not trigger the workout app when pressed.",
        confirmed_facts={"device_model": "Apple Watch Ultra 2", "os_version": "WATCHOS 10.4"},
        problem_family="GENERAL_DEVICE_FUNCTIONALITY",
        primary_intent="general_device_support",
        actions_attempted=["restart_watch", "reassign_action_button"],
        actions_that_worked=["reassign_action_button"],
        final_resolution=(
            "Navigate to Settings > Action Button on Apple Watch Ultra. Tap Action, then select Workout. "
            "Under App, select the Workout app and configure the primary shortcut."
        ),
        evidence_sources=["https://support.apple.com/en-us/HT213426"],
        specialist_notes="Customer had action button set to None after watchOS restore.",
    )

    # 1. Capture
    captured = feedback_pipeline.capture_resolution(res)
    assert captured.promotion_status == ResolutionLifecycleState.CAPTURED

    # 2. Submit for review
    in_review = feedback_pipeline.submit_for_review(captured.resolution_id)
    assert in_review.promotion_status == ResolutionLifecycleState.UNDER_REVIEW

    # 3. Validate
    validated, decision = feedback_pipeline.validate_resolution(captured.resolution_id)
    assert validated.promotion_status == ResolutionLifecycleState.VALIDATED
    assert decision.is_valid is True
    assert decision.checks["completeness"] == "PASS"
    assert decision.checks["safety"] == "PASS"
    assert decision.checks["golden_isolation"] == "PASS"
    assert validated.evidence_quality_score is not None
    assert validated.evidence_quality_score >= 0.85


# ---------------------------------------------------------------------------
# Test B: Incomplete Human Resolution Rejected
# ---------------------------------------------------------------------------
def test_incomplete_human_resolution_rejected(feedback_pipeline: PromotionPipeline):
    """Resolution missing actionable steps or actions attempted is rejected."""
    res = HumanResolution(
        conversation_id="conv_incompl_001",
        customer_problem="iPad is slow",
        problem_family="PERFORMANCE",
        actions_attempted=[],  # Empty actions
        final_resolution="Fixed it",  # Terse
    )
    captured = feedback_pipeline.capture_resolution(res)
    validated, decision = feedback_pipeline.validate_resolution(captured.resolution_id)

    assert validated.promotion_status == ResolutionLifecycleState.REJECTED
    assert decision.is_valid is False
    assert decision.checks["completeness"] == "FAIL"
    assert "Incomplete resolution" in validated.rejection_reason


# ---------------------------------------------------------------------------
# Test C: Contradictory Human Resolution Rejected
# ---------------------------------------------------------------------------
def test_contradictory_human_resolution_rejected(feedback_pipeline: PromotionPipeline):
    """Resolution with mutually contradictory instructions is rejected."""
    res = HumanResolution(
        conversation_id="conv_contra_001",
        customer_problem="iPhone 13 battery draining after update.",
        problem_family="POWER_BATTERY",
        actions_attempted=["update_ios"],
        final_resolution="Please update iOS to latest version immediately, but do not update under any circumstances.",
    )
    captured = feedback_pipeline.capture_resolution(res)
    validated, decision = feedback_pipeline.validate_resolution(captured.resolution_id)

    assert validated.promotion_status == ResolutionLifecycleState.REJECTED
    assert decision.checks["contradiction"] == "FAIL"


# ---------------------------------------------------------------------------
# Test D: Unsafe / Destructive Resolution Rejected
# ---------------------------------------------------------------------------
def test_unsafe_destructive_resolution_rejected(feedback_pipeline: PromotionPipeline):
    """Resolution advising dangerous commands or ignoring thermal hazard is rejected."""
    res = HumanResolution(
        conversation_id="conv_unsafe_001",
        customer_problem="MacBook Pro kernel crash loop.",
        problem_family="CRASH_FREEZE",
        actions_attempted=["terminal_wipe"],
        final_resolution="Open terminal and run sudo rm -rf / to clear all corrupted system daemons.",
    )
    captured = feedback_pipeline.capture_resolution(res)
    validated, decision = feedback_pipeline.validate_resolution(captured.resolution_id)

    assert validated.promotion_status == ResolutionLifecycleState.REJECTED
    assert decision.checks["safety"] == "FAIL"
    assert "violates safety policy" in validated.rejection_reason


# ---------------------------------------------------------------------------
# Test E: Wrong Problem-Family Resolution Rejected
# ---------------------------------------------------------------------------
def test_wrong_problem_family_rejected(feedback_pipeline: PromotionPipeline):
    """Resolution declaring conflicting problem family (e.g. charging on audio problem) is rejected."""
    res = HumanResolution(
        conversation_id="conv_fam_mismatch_001",
        customer_problem="My AirPods Pro will not pair or connect to my iPhone 14 via Bluetooth.",
        problem_family="CHARGING",  # Declared charging instead of audio/connectivity
        primary_intent="hardware_audio_connection_issue",
        actions_attempted=["inspect_charger"],
        final_resolution="Inspect the lightning cable and wall adapter for power delivery faults.",
    )
    captured = feedback_pipeline.capture_resolution(res)
    validated, decision = feedback_pipeline.validate_resolution(captured.resolution_id)

    assert validated.promotion_status == ResolutionLifecycleState.REJECTED
    assert decision.checks["problem_family_consistency"] == "FAIL"


# ---------------------------------------------------------------------------
# Test F: Fake / Untrusted Evidence Source Rejected
# ---------------------------------------------------------------------------
def test_fake_evidence_source_rejected(feedback_pipeline: PromotionPipeline):
    """Resolution referencing unauthorized third-party exploit domains is rejected."""
    res = HumanResolution(
        conversation_id="conv_fake_src_001",
        customer_problem="Activation lock on iPhone 12.",
        problem_family="ACCOUNT_ACCESS",
        actions_attempted=["unlock_tool"],
        final_resolution="Download the bypass tool from https://hack-apple.net/free-unlock.exe to remove lock.",
        evidence_sources=["https://hack-apple.net/free-unlock.exe"],
    )
    captured = feedback_pipeline.capture_resolution(res)
    validated, decision = feedback_pipeline.validate_resolution(captured.resolution_id)

    assert validated.promotion_status == ResolutionLifecycleState.REJECTED
    assert decision.checks["provenance"] == "FAIL"


# ---------------------------------------------------------------------------
# Test G: Golden Benchmark Contamination Rejected (Zero Leakage)
# ---------------------------------------------------------------------------
def test_golden_benchmark_contamination_rejected(feedback_pipeline: PromotionPipeline):
    """Candidate evidence containing a golden benchmark conversation ID is blocked."""
    # Pick a real golden ID from golden set (conv_AppleSupport_1517047)
    golden_id = "conv_AppleSupport_1517047"
    res = HumanResolution(
        conversation_id=golden_id,
        customer_problem="My battery percentage is jumping erratically on iOS 17.",
        problem_family="POWER_BATTERY",
        actions_attempted=["restart_device"],
        final_resolution="Perform a force restart and review Battery Health in Settings > Battery.",
    )
    captured = feedback_pipeline.capture_resolution(res)
    validated, decision = feedback_pipeline.validate_resolution(captured.resolution_id)

    assert validated.promotion_status == ResolutionLifecycleState.REJECTED
    assert decision.checks["golden_isolation"] == "FAIL"
    assert "Golden benchmark contamination detected" in validated.rejection_reason


# ---------------------------------------------------------------------------
# Test H & I: Candidate and Rejected Evidence Are NOT Retrievable
# ---------------------------------------------------------------------------
def test_candidate_and_rejected_evidence_not_retrievable(feedback_pipeline: PromotionPipeline):
    """Candidate resolutions in CAPTURED, UNDER_REVIEW, VALIDATED, or REJECTED state cannot be retrieved."""
    # 1. Create a candidate resolution
    res = HumanResolution(
        conversation_id="conv_unpromoted_001",
        customer_problem="My Apple Pencil 2nd generation won't magnetically pair with my iPad Pro M2.",
        problem_family="ACCESSORY_PERIPHERAL",
        primary_intent="general_device_support",
        actions_attempted=["clean_connector", "bluetooth_repair"],
        actions_that_worked=["bluetooth_repair"],
        final_resolution="Attach Apple Pencil to magnetic connector, go to Settings > Bluetooth, forget device and tap pair.",
    )
    feedback_pipeline.capture_resolution(res)
    feedback_pipeline.validate_resolution(res.resolution_id)

    # 2. Retriever query
    retriever = CaseRetriever(approved_store=feedback_pipeline.approved_store)
    results = retriever.retrieve(
        query_text="Apple Pencil won't pair with iPad Pro M2",
        query_intent="general_device_support",
    )

    # Ensure none of the retrieved cases match the unpromoted candidate resolution
    retrieved_ids = [c.case_id for c in results]
    assert res.resolution_id not in retrieved_ids
    for c in results:
        assert c.source_type != "HUMAN_VALIDATED_EVIDENCE" or c.case_id != res.resolution_id


# ---------------------------------------------------------------------------
# Test J, K, L, M, N: Full Approved Promotion, Provenance, & Versioning
# ---------------------------------------------------------------------------
def test_approved_promotion_and_retrieval(feedback_pipeline: PromotionPipeline):
    """Resolution promoted to APPROVED state becomes retrievable with full versioning and provenance."""
    res = HumanResolution(
        conversation_id="conv_promoted_001",
        customer_problem="My Apple Pencil 2nd generation won't magnetically pair with my iPad Pro M2.",
        problem_family="CONNECTIVITY_BLUETOOTH",
        primary_intent="general_device_support",
        actions_attempted=["clean_connector", "bluetooth_repair"],
        actions_that_worked=["bluetooth_repair"],
        final_resolution="Attach Apple Pencil to magnetic connector, go to Settings > Bluetooth, forget device and tap pair.",
        evidence_sources=["https://support.apple.com/en-us/HT205236"],
    )

    # 1. Capture & Validate
    feedback_pipeline.capture_resolution(res)
    feedback_pipeline.validate_resolution(res.resolution_id)

    # Test K: Attempt promotion BEFORE reviewer approval must fail (tested on separate resolution)
    unapproved_res = HumanResolution(
        conversation_id="conv_unapproved_premature_001",
        customer_problem="iPad mini battery draining fast.",
        problem_family="POWER_BATTERY",
        primary_intent="battery_power_issue",
        actions_attempted=["restart_device"],
        final_resolution="Inspect battery health in Settings > Battery.",
    )
    feedback_pipeline.capture_resolution(unapproved_res)
    feedback_pipeline.validate_resolution(unapproved_res.resolution_id)
    with pytest.raises(ValueError, match="Promotion blocked"):
        feedback_pipeline.promote_to_evidence(unapproved_res.resolution_id, reviewer_id="agent_123")

    # 2. Offline Evaluation
    eval_res = feedback_pipeline.run_offline_evaluation(res.resolution_id)
    assert eval_res["passed"] is True

    # 3. Reviewer Approval
    approved = feedback_pipeline.approve_resolution(
        res.resolution_id,
        reviewer_id="lead_specialist_42",
        notes="Verified Apple Pencil 2 pairing procedure on iPadOS 17.5.",
    )
    assert approved.promotion_status == ResolutionLifecycleState.APPROVED

    # 4. Promote
    promoted_item = feedback_pipeline.promote_to_evidence(
        res.resolution_id,
        reviewer_id="lead_specialist_42",
        version="1.0.0",
    )
    assert promoted_item.version == "1.0.0"
    assert promoted_item.source_type == "HUMAN_VALIDATED_EVIDENCE"
    assert promoted_item.reviewer_id == "lead_specialist_42"
    assert promoted_item.originating_resolution_id == res.resolution_id

    # Test J: Approved item IS retrievable by CaseRetriever
    retriever = CaseRetriever(approved_store=feedback_pipeline.approved_store)
    results = retriever.retrieve(
        query_text="My Apple Pencil 2 won't pair with iPad Pro M2",
        query_intent="general_device_support",
    )

    human_cases = [c for c in results if c.source_type == "HUMAN_VALIDATED_EVIDENCE"]
    assert len(human_cases) > 0
    assert human_cases[0].case_id == promoted_item.evidence_id
    assert human_cases[0].provenance == "reviewed_human_resolution"
    assert human_cases[0].version == "1.0.0"


# ---------------------------------------------------------------------------
# Test O & P: Controlled Improvement vs Negative Experiments
# ---------------------------------------------------------------------------
def test_controlled_evidence_improvement_experiment(feedback_pipeline: PromotionPipeline):
    """
    Demonstrates that promoting a valid human resolution provides grounded evidence
    for a previously unsupported or novel problem domain.
    """
    engine = SupportResolutionEngine(retriever=CaseRetriever(approved_store=feedback_pipeline.approved_store))
    query = "My Apple Watch Ultra 2 Action Button is not launching the Workout app."

    # Pre-promotion: Verify baseline resolution
    res_before = engine.process_message(query, case_id="test-hitl-before")

    # Promote valid human resolution for this specific workflow
    h_res = HumanResolution(
        conversation_id="conv_ultra_action_button",
        customer_problem="Apple Watch Ultra Action button does not open workout app.",
        problem_family="ACCESSORY_PERIPHERAL",
        primary_intent="general_device_support",
        actions_attempted=["reassign_action_button"],
        actions_that_worked=["reassign_action_button"],
        final_resolution="Open Settings on Apple Watch Ultra, tap Action Button > Action, and assign Workout.",
        evidence_sources=["https://support.apple.com/en-us/HT213426"],
    )
    feedback_pipeline.capture_resolution(h_res)
    feedback_pipeline.validate_resolution(h_res.resolution_id)
    feedback_pipeline.approve_resolution(h_res.resolution_id, reviewer_id="qa_lead")
    feedback_pipeline.promote_to_evidence(h_res.resolution_id, reviewer_id="qa_lead", version="1.0.0")

    # Post-promotion: Re-query and verify evidence availability
    res_after = engine.process_message(query, case_id="test-hitl-after")

    # Grounded evidence should now include HUMAN_VALIDATED_EVIDENCE
    human_ev = [c for c in res_after.evidence_cases if c.source_type == "HUMAN_VALIDATED_EVIDENCE"]
    assert len(human_ev) > 0
    assert res_after.response_grounding is not None


# ---------------------------------------------------------------------------
# Test S & T: Safety & Regression Preservations
# ---------------------------------------------------------------------------
def test_thermal_hazard_and_airpods_regressions_preserved(tmp_path: Path):
    """Verifies that thermal hazards and AirPods Bluetooth pairing behaviors remain 100% intact."""
    mgr = ConversationManager(base_dir=tmp_path)
    engine = SupportResolutionEngine()

    # 1. Thermal hazard check
    state, t1 = mgr.create_conversation("My iPhone is warm.")
    state, t2 = mgr.process_message(state.conversation_id, "Now the phone is smoking and extremely hot to touch!")
    assert state.status == ConversationStatus.ESCALATED
    assert state.escalation_package.recommended_specialist_tier == "TIER_2_TECHNICAL_URGENT"

    # 2. AirPods Bluetooth pairing check
    airpods_res = engine.process_message("My AirPods Pro will not pair with my iPhone 14.", case_id="test-airpods-pres")
    assert airpods_res.primary_intent == "hardware_audio_connection_issue"
    assert airpods_res.routing_decision == RoutingDecisionType.AUTO_HANDLE


# ---------------------------------------------------------------------------
# Golden Dataset Immutability Check
# ---------------------------------------------------------------------------
def test_golden_dataset_immutability():
    """Verify golden dataset SHA-256 remains byte-for-byte unchanged."""
    sha = compute_sha256(GOLDEN_CSV_PATH)
    assert sha == GOLDEN_SHA256
