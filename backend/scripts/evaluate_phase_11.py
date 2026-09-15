"""
SupportGraph AI — Phase 11 Evaluation Driver.

Evaluates multi-turn evidence-grounded support resolution across 10 benchmark scenarios (A–J),
validates golden dataset SHA-256 before and after execution, and exports audit artifacts.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

try:
    from app.evaluation.conversation_resolution_evaluator import (
        ConversationResolutionEvaluator,
    )
except ModuleNotFoundError:
    from backend.app.evaluation.conversation_resolution_evaluator import (  # type: ignore[no-redef]
        ConversationResolutionEvaluator,
    )

GOLDEN_CSV_PATH = PROJECT_ROOT / "data" / "golden" / "golden_set_human_review.csv"
EXPECTED_GOLDEN_SHA256 = "1d3e9b3b8bdef3750437e59b17ab7c27167fd1548c2bb29de984151296c5b45a"
REPORT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "phase_11"
REPORT_OUTPUT_FILE = REPORT_OUTPUT_DIR / "multiturn_resolution_audit.json"


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> int:
    print("=" * 80)
    print("SUPPORTGRAPH AI — PHASE 11 BENCHMARK EVALUATION")
    print("Multi-Turn Evidence-Grounded Support Resolution & Conversation State Management")
    print("=" * 80)

    # 1. Pre-evaluation Golden Dataset Integrity Verification
    print("\n[Step 1/4] Verifying Pre-Evaluation Golden Dataset Integrity...")
    pre_hash = compute_sha256(GOLDEN_CSV_PATH)
    print(f"  Golden Set Path:   {GOLDEN_CSV_PATH}")
    print(f"  Golden SHA-256:    {pre_hash}")
    if pre_hash != EXPECTED_GOLDEN_SHA256:
        print(f"ERROR: Golden set SHA-256 mismatch! Expected {EXPECTED_GOLDEN_SHA256}")
        return 1
    print("  STATUS: PASS (Protected golden dataset is intact and untampered)")

    # 2. Run Benchmark Evaluation across Scenarios A through J
    print("\n[Step 2/4] Running Multi-Turn Evaluation across Scenarios A through J...")
    eval_scratch = PROJECT_ROOT / "data" / "conversations" / "eval_scratch"
    evaluator = ConversationResolutionEvaluator(working_dir=eval_scratch)
    report = evaluator.run_benchmark()

    print("\nScenario Evaluation Results:")
    print("-" * 80)
    print(f"{'Scenario ID':<12} | {'Status':<6} | {'Turns':<5} | {'Final State':<12} | {'Name'}")
    print("-" * 80)
    for res in report.results:
        status_str = "PASS" if res.passed else "FAIL"
        print(f"{res.scenario_id:<12} | {status_str:<6} | {res.total_turns:<5} | {res.final_status:<12} | {res.name}")
        if not res.passed and res.error_message:
            print(f"   --> Error: {res.error_message}")
    print("-" * 80)

    print("\nQuantitative Benchmark Summary Metrics:")
    print(f"  Total Scenarios Evaluated:         {report.total_scenarios}")
    print(f"  Passed Scenarios:                  {report.passed_scenarios} / {report.total_scenarios} ({report.scenario_pass_rate * 100:.1f}%)")
    print(f"  Repeat Prevention Rate:            {report.repeat_prevention_rate * 100:.1f}% (Target: 100%)")
    print(f"  Semantic Aliasing Accuracy:        {report.semantic_aliasing_accuracy * 100:.1f}% (Target: 100%)")
    print(f"  Clarification Precision:           {report.clarification_precision * 100:.1f}% (Target: 100%)")
    print(f"  Resolution Confirmation Accuracy:  {report.resolution_confirmation_accuracy * 100:.1f}% (Target: 100%)")
    print(f"  Safe Escalation Fidelity:          {report.safe_escalation_fidelity * 100:.1f}% (Target: 100%)")
    print(f"  Context Fact Retention Rate:       {report.context_fact_retention_rate * 100:.1f}% (Target: 100%)")
    print(f"  Unsafe Auto-Handle Rate:           {report.unsafe_handle_rate * 100:.1f}% (Target: 0.0%)")

    # 3. Post-evaluation Golden Dataset Integrity Verification
    print("\n[Step 3/4] Verifying Post-Evaluation Golden Dataset Integrity...")
    post_hash = compute_sha256(GOLDEN_CSV_PATH)
    print(f"  Post-Eval SHA-256: {post_hash}")
    if post_hash != EXPECTED_GOLDEN_SHA256 or post_hash != pre_hash:
        print("ERROR: Golden dataset was modified during evaluation! Leakage/corruption detected!")
        return 1
    print("  STATUS: PASS (Zero dataset leakage; golden set remains bit-for-bit identical)")

    # 4. Save JSON Audit Artifact
    print("\n[Step 4/4] Exporting Evaluation Report Artifact...")
    REPORT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_dict = report.to_dict()
    report_dict["golden_sha256_verified"] = post_hash
    with open(REPORT_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    print(f"  Report saved to: {REPORT_OUTPUT_FILE}")

    if report.passed_scenarios == report.total_scenarios and report.unsafe_handle_rate == 0.0:
        print("\n" + "=" * 80)
        print("SUCCESS: Phase 11 multi-turn benchmark evaluation PASSED with 100% compliance!")
        print("=" * 80)
        return 0
    else:
        print("\n" + "=" * 80)
        print("FAILURE: Some benchmark scenarios failed compliance checks.")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
