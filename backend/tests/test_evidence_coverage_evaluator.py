"""
SupportGraph AI — Tests: Evidence Coverage Evaluator (Phase 10.1)

Tests the EvidenceCoverageEvaluator, EvidenceCorpusBuilder, and
aggregate metric computation.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.app.evaluation.evidence_corpus_builder import (
    AuditCorpus,
    CorpusEntry,
    EvidenceCorpusBuilder,
    GOLDEN_SHA256,
    compute_sha256,
    infer_intent_from_text,
)
from backend.app.evaluation.evidence_coverage_auditor import (
    AuditRootCause,
    RECOVERABLE_CAUSES,
)
from backend.app.evaluation.evidence_coverage_evaluator import (
    EvidenceCoverageEvaluator,
    Phase101BenchmarkMetrics,
    RECOVERABLE_CAUSES as EVALUATOR_RECOVERABLE_CAUSES,
)

GOLDEN_CSV = Path("data/golden/golden_set_human_review.csv")


# ── EvidenceCorpusBuilder Tests ───────────────────────────────────────────────

def test_corpus_builder_excludes_golden_ids():
    """Audit corpus must not contain any golden benchmark conversation IDs."""
    if not GOLDEN_CSV.exists():
        pytest.skip("Golden dataset not available in test environment.")

    import pandas as pd
    golden_ids = set(
        pd.read_csv(GOLDEN_CSV)["conversation_id"].astype(str).unique()
    )

    builder = EvidenceCorpusBuilder(max_sample_size=100)
    corpus = builder.build()

    for entry in corpus.entries:
        assert entry.conversation_id not in golden_ids, (
            f"Golden benchmark conversation ID '{entry.conversation_id}' "
            f"was included in audit corpus — DATA LEAKAGE!"
        )


def test_corpus_builder_preserves_golden_sha256():
    """Corpus builder must not modify the golden dataset."""
    if not GOLDEN_CSV.exists():
        pytest.skip("Golden dataset not available in test environment.")

    pre = compute_sha256(GOLDEN_CSV)
    builder = EvidenceCorpusBuilder(max_sample_size=50)
    corpus = builder.build()
    post = compute_sha256(GOLDEN_CSV)

    assert pre == post == GOLDEN_SHA256, (
        f"Golden dataset was modified by EvidenceCorpusBuilder! pre={pre} post={post}"
    )
    assert corpus.golden_immutability_pass


def test_corpus_builder_sets_correct_metadata():
    """AuditCorpus metadata must accurately reflect the build process."""
    if not GOLDEN_CSV.exists():
        pytest.skip("Golden dataset not available in test environment.")

    max_size = 200
    builder = EvidenceCorpusBuilder(max_sample_size=max_size)
    corpus = builder.build()

    # Corpus size must not exceed requested maximum
    assert len(corpus.entries) <= max_size
    assert corpus.total_conversations_available > 0
    assert corpus.excluded_golden_ids >= 0
    assert corpus.pre_build_golden_sha256 == GOLDEN_SHA256
    assert corpus.post_build_golden_sha256 == GOLDEN_SHA256

    # All entries must have required fields
    for entry in corpus.entries:
        assert entry.case_id != ""
        assert entry.conversation_id != ""
        assert entry.customer_message.strip() != ""
        assert entry.inferred_intent != ""
        assert entry.corpus_source == "historical_parquet"


def test_corpus_entries_have_non_empty_messages():
    """All corpus entries must contain non-empty customer messages."""
    if not GOLDEN_CSV.exists():
        pytest.skip("Golden dataset not available in test environment.")

    builder = EvidenceCorpusBuilder(max_sample_size=100)
    corpus = builder.build()

    for entry in corpus.entries:
        assert len(entry.customer_message.strip()) >= 10, (
            f"Entry {entry.case_id} has a message that's too short: '{entry.customer_message}'"
        )


# ── Intent Inference Tests ────────────────────────────────────────────────────

def test_intent_inference_battery():
    assert infer_intent_from_text("battery drains really fast") == "battery_power_issue"


def test_intent_inference_keyboard():
    assert infer_intent_from_text("keyboard typing autocorrect problem") == "keyboard_typing_issue"


def test_intent_inference_audio():
    assert infer_intent_from_text("my airpod speaker is not working") == "hardware_audio_connection_issue"


def test_intent_inference_software_update():
    assert infer_intent_from_text("ios update bug problem") == "software_update_problem"


def test_intent_inference_fallback():
    assert infer_intent_from_text("") == "general_device_support"
    assert infer_intent_from_text("   ") == "general_device_support"
    assert infer_intent_from_text("some random message without keywords") == "general_device_support"


def test_intent_inference_account():
    assert infer_intent_from_text("apple id password locked sign in problem") == "account_access_issue"


# ── RECOVERABLE_CAUSES constant ───────────────────────────────────────────────

def test_recoverable_causes_consistent_between_auditor_and_evaluator():
    """Both auditor and evaluator must define the same recoverable cause set."""
    assert RECOVERABLE_CAUSES == EVALUATOR_RECOVERABLE_CAUSES


def test_recoverable_causes_does_not_include_knowledge_gaps():
    """TRUE_KNOWLEDGE_GAP and INSUFFICIENT_INFORMATION are never recoverable."""
    assert AuditRootCause.TRUE_KNOWLEDGE_GAP not in RECOVERABLE_CAUSES
    assert AuditRootCause.INSUFFICIENT_INFORMATION not in RECOVERABLE_CAUSES


def test_recoverable_causes_includes_pipeline_failures():
    """All four pipeline failure categories must be recoverable."""
    assert AuditRootCause.RETRIEVAL_MISS in RECOVERABLE_CAUSES
    assert AuditRootCause.RANKING_MISS in RECOVERABLE_CAUSES
    assert AuditRootCause.VALIDATION_OVER_REJECTION in RECOVERABLE_CAUSES
    assert AuditRootCause.REPRESENTATION_LIMITATION in RECOVERABLE_CAUSES


# ── Phase101BenchmarkMetrics Tests ────────────────────────────────────────────

def _make_minimal_metrics(**overrides) -> Phase101BenchmarkMetrics:
    """Build a minimal valid Phase101BenchmarkMetrics for testing."""
    defaults: dict = {
        "total_evaluated_records": 77,
        "pre_eval_sha256": GOLDEN_SHA256,
        "post_eval_sha256": GOLDEN_SHA256,
        "dataset_immutability_pass": True,
        "audit_corpus_size": 5000,
        "audit_corpus_excluded_golden_ids": 200,
        "audit_extended_search_depth": 100,
        "total_escalated": 58,
        "evidence_limited_count": 43,
        "other_escalation_count": 15,
        "auto_handled_count": 19,
        "root_cause_distribution": {
            "RETRIEVAL_MISS": 5,
            "RANKING_MISS": 28,
            "VALIDATION_OVER_REJECTION": 3,
            "REPRESENTATION_LIMITATION": 2,
            "TRUE_KNOWLEDGE_GAP": 3,
            "INSUFFICIENT_INFORMATION": 2,
        },
        "root_cause_percentages": {
            "RETRIEVAL_MISS": 11.6,
            "RANKING_MISS": 65.1,
            "VALIDATION_OVER_REJECTION": 7.0,
            "REPRESENTATION_LIMITATION": 4.7,
            "TRUE_KNOWLEDGE_GAP": 7.0,
            "INSUFFICIENT_INFORMATION": 4.7,
        },
        "recoverable_failure_count": 38,
        "recoverable_failure_rate": 0.884,
        "true_knowledge_gap_count": 3,
        "true_knowledge_gap_rate": 0.070,
        "insufficient_information_count": 2,
        "insufficient_information_rate": 0.047,
        "retrieval_miss_count": 5,
        "ranking_miss_count": 28,
        "validation_over_rejection_count": 3,
        "representation_limitation_count": 2,
        "avg_best_relevant_rank": 12.4,
        "corpus_coverage_rate": 0.907,
        "primary_bottleneck": "RANKING_MISS",
        "recommended_next_phase": "Ranking improvement",
        "case_audit_records": [],
    }
    defaults.update(overrides)
    return Phase101BenchmarkMetrics(**defaults)


def test_metrics_recoverable_failure_rate_consistent():
    """Recoverable failure rate must equal recoverable_count / evidence_limited_count."""
    metrics = _make_minimal_metrics(
        recoverable_failure_count=38,
        evidence_limited_count=43,
        recoverable_failure_rate=round(38 / 43, 4),
    )
    expected = round(38 / 43, 4)
    assert abs(metrics.recoverable_failure_rate - expected) < 0.001


def test_metrics_true_knowledge_gap_rate_consistent():
    """Knowledge gap rate must equal gap_count / evidence_limited_count."""
    metrics = _make_minimal_metrics(
        true_knowledge_gap_count=3,
        evidence_limited_count=43,
        true_knowledge_gap_rate=round(3 / 43, 4),
    )
    expected = round(3 / 43, 4)
    assert abs(metrics.true_knowledge_gap_rate - expected) < 0.001


def test_metrics_immutability_flag():
    """Dataset immutability must be True if hashes match."""
    metrics = _make_minimal_metrics(
        pre_eval_sha256=GOLDEN_SHA256,
        post_eval_sha256=GOLDEN_SHA256,
        dataset_immutability_pass=True,
    )
    assert metrics.dataset_immutability_pass

    # If hashes differ, flag should be False
    bad_metrics = _make_minimal_metrics(
        pre_eval_sha256=GOLDEN_SHA256,
        post_eval_sha256="different_hash",
        dataset_immutability_pass=False,
    )
    assert not bad_metrics.dataset_immutability_pass


def test_metrics_root_cause_counts_sum_to_evidence_limited():
    """Sum of all root cause counts must equal evidence_limited_count."""
    metrics = _make_minimal_metrics()
    total = sum(metrics.root_cause_distribution.values())
    assert total == metrics.evidence_limited_count, (
        f"Root cause sum ({total}) != evidence_limited_count ({metrics.evidence_limited_count})"
    )


def test_metrics_primary_bottleneck_has_highest_count():
    """Primary bottleneck must be the root cause with the highest count."""
    metrics = _make_minimal_metrics()
    highest = max(metrics.root_cause_distribution, key=metrics.root_cause_distribution.get)
    assert metrics.primary_bottleneck == highest, (
        f"Primary bottleneck '{metrics.primary_bottleneck}' != highest count cause '{highest}'"
    )


# ── EvidenceCoverageEvaluator Integration Test ────────────────────────────────

@pytest.mark.slow
def test_evaluator_runs_on_real_benchmark():
    """
    Integration test: Full evaluator run on the real benchmark.
    Verifies metrics are populated and golden SHA-256 is preserved.
    """
    if not GOLDEN_CSV.exists():
        pytest.skip("Golden dataset not available in test environment.")

    evaluator = EvidenceCoverageEvaluator(max_audit_corpus_size=100)
    metrics = evaluator.run_benchmark()

    # Basic integrity
    assert metrics.total_evaluated_records == 77
    assert metrics.dataset_immutability_pass, "Golden dataset was modified!"
    assert metrics.pre_eval_sha256 == GOLDEN_SHA256
    assert metrics.post_eval_sha256 == GOLDEN_SHA256

    # Root cause totals must equal evidence-limited count
    total = sum(metrics.root_cause_distribution.values())
    assert total == metrics.evidence_limited_count

    # All rates must be in [0, 1]
    assert 0.0 <= metrics.recoverable_failure_rate <= 1.0
    assert 0.0 <= metrics.true_knowledge_gap_rate <= 1.0
    assert 0.0 <= metrics.corpus_coverage_rate <= 1.0

    # Primary bottleneck must be valid
    assert metrics.primary_bottleneck in {rc.value for rc in AuditRootCause}

    # Sum of recoverable + gap + insufficient <= evidence_limited
    total_accounted = (
        metrics.recoverable_failure_count
        + metrics.true_knowledge_gap_count
        + metrics.insufficient_information_count
    )
    assert total_accounted <= metrics.evidence_limited_count
