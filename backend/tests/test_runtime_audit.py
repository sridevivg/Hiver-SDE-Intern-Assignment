"""
Tests for backend/app/resolution/resolution_auditor.py (Phase 8)
"""
import pytest

from backend.app.resolution.resolution_auditor import (
    ResolutionAuditRecord,
    ResolutionAuditor,
)


def test_runtime_auditor_isolation(tmp_path):
    audit_file = tmp_path / "test_audit.csv"
    auditor = ResolutionAuditor(audit_path=audit_file)

    rec = ResolutionAuditRecord(
        case_id="case_test_01",
        customer_message="@AppleSupport My battery drains fast.",
        device="iPhone 11",
        primary_symptom="battery_power",
        primary_intent="battery_power_issue",
        top_confidence=0.95,
        evidence_verdict="STRONG_EVIDENCE",
        direct_matches_count=2,
        routing_decision="AUTO_HANDLE",
        retrieved_case_ids="case_1,case_2",
        match_tiers="DIRECT_PROBLEM_MATCH,DIRECT_PROBLEM_MATCH",
        response_grounding_status="PASS",
        grounding_score=0.95,
    )

    auditor.log_decision(rec)

    stats = auditor.get_audit_stats()
    assert stats["total_logged_decisions"] == 1
    assert stats["auto_handle_count"] == 1
    assert stats["escalation_count"] == 0
    assert stats["grounding_pass_count"] == 1
    assert stats["evidence_verdicts"]["STRONG_EVIDENCE"] == 1
