"""
SupportGraph AI — Interactive & Multi-Scenario Evidence Resolution Simulator (Phase 8)

Demonstrates the full evidence-grounded resolution pipeline:
1. Customer Message
2. Problem Understanding Profile
3. Top-3 Intent Candidates
4. Retrieved Historical Evidence with Operational Match Tiers
5. Operational Evidence Verdict (STRONG, MODERATE, WEAK, CONFLICTING)
6. Response Grounding Verification (Checklist & Safety)
7. Final Decision (AUTO_HANDLE vs ESCALATE_TO_HUMAN) with Grounded Reply or Enriched Human Package.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from backend.app.core.logging import get_logger
    from backend.app.resolution.evidence_validator import EvidenceVerdict
    from backend.app.resolution.response_verifier import VerificationStatus
    from backend.app.resolution.support_resolution_engine import (
        SupportResolutionEngine,
        SupportResolutionResult,
    )
    from backend.app.schemas.intent_routing import RoutingDecisionType
except ModuleNotFoundError:
    from app.core.logging import get_logger  # type: ignore[no-redef]
    from app.resolution.evidence_validator import (  # type: ignore[no-redef]
        EvidenceVerdict,
    )
    from app.resolution.response_verifier import (  # type: ignore[no-redef]
        VerificationStatus,
    )
    from app.resolution.support_resolution_engine import (  # type: ignore[no-redef]
        SupportResolutionEngine,
        SupportResolutionResult,
    )
    from app.schemas.intent_routing import (  # type: ignore[no-redef]
        RoutingDecisionType,
    )

logger = get_logger(__name__)

SCENARIOS = [
    {
        "id": "scenario_1",
        "title": "Scenario 1: Single Clear Hardware Symptom with Strong Historical Evidence",
        "message": "@AppleSupport My iPhone 12 battery is draining down to 10% in just 2 hours after moderate use.",
    },
    {
        "id": "scenario_2",
        "title": "Scenario 2: Update Trigger with Battery Symptom (Cause Decoupled)",
        "message": "@AppleSupport Ever since updating to iOS 11.1 on my iPhone 7, my battery drains super fast.",
    },
    {
        "id": "scenario_3",
        "title": "Scenario 3: Vague Inquiry / Missing Information (Insufficient Info Veto)",
        "message": "@AppleSupport My phone is completely broken and nothing works. Please fix it!",
    },
    {
        "id": "scenario_4",
        "title": "Scenario 4: Multi-Symptom Hardware Conflict (Display + Battery)",
        "message": "@AppleSupport My iPhone screen keeps flickering black and also the battery dies in 15 minutes.",
    },
    {
        "id": "scenario_5",
        "title": "Scenario 5: Peripheral Audio Issue with Conflicting Context",
        "message": "@AppleSupport My AirPods keep disconnecting every time I play music in Spotify.",
    },
]


def render_resolution_result(msg: str, res: SupportResolutionResult, title: str = "") -> None:
    print("\n" + "=" * 80)
    if title:
        print(f"DEMO: {title}")
        print("=" * 80)
    print("CUSTOMER MESSAGE:")
    print(f"  \"{msg}\"")

    # Problem Understanding
    print("\n[1] PROBLEM UNDERSTANDING")
    p_summary = res.problem_summary
    print(f"  • Summary:                {p_summary}")
    print(f"  • Ambiguity Type:         {res.ambiguity_analysis.ambiguity_type.value}")
    print(f"  • Ambiguity Rationale:    {res.ambiguity_analysis.ambiguity_reason}")

    # Top Intent Candidates
    print("\n[2] TOP INTENT CANDIDATES")
    if res.escalation_package and res.escalation_package.top_candidates:
        cand_list = [(p.intent, p.confidence) for p in res.escalation_package.top_candidates]
    else:
        cand_list = [(res.primary_intent, res.confidence)]
        for c in res.ambiguity_analysis.competing_candidates:
            if c != res.primary_intent:
                cand_list.append((c, max(0.05, res.confidence - res.ambiguity_analysis.confidence_margin)))

    for i, (c_name, c_conf) in enumerate(cand_list[:3], 1):
        print(f"  {i}. {c_name:<35s} (Confidence: {c_conf:.1%})")

    # Historical Evidence
    print("\n[3] HISTORICAL EVIDENCE MATCHING")
    if res.evidence_cases:
        for i, case in enumerate(res.evidence_cases[:3], 1):
            tier_sym = "★" if case.match_tier.value == "DIRECT_PROBLEM_MATCH" else "●"
            print(f"  CASE {i} [{case.case_id}] {tier_sym} Tier: {case.match_tier.value}")
            print(f"    • Historical Msg:   \"{case.historical_customer_message[:90]}...\"")
            print(f"    • Operational Sim:  {case.operational_similarity:.2f} (TF-IDF Cosine: {case.similarity_score:.2f})")
            print(f"    • Explanation:      {case.retrieval_explanation}")
    else:
        print("  (No historical evidence cases retrieved)")

    # Evidence Verdict
    print("\n[4] OPERATIONAL EVIDENCE VERDICT")
    if res.evidence_validation:
        ev = res.evidence_validation
        print(f"  • Verdict:                {ev.evidence_verdict.value}")
        print(f"  • Auto Resolution OK?:    {'YES' if ev.auto_resolution_allowed else 'NO'}")
        print(f"  • Direct Matches:         {ev.direct_problem_matches}")
        print(f"  • Symptom Agreement:      {ev.symptom_agreement:.1%}")
        print(f"  • Intent Agreement:       {ev.intent_agreement:.1%}")
        print(f"  • Support Strength:       {ev.resolution_support_strength:.2f}")
        for r in ev.reasons:
            print(f"    - {r}")

    # Response Grounding Verification
    print("\n[5] RESPONSE GROUNDING VERIFICATION")
    if res.response_grounding:
        rg = res.response_grounding
        p_mark = "✓" if rg.addresses_primary_symptom else "✗"
        e_mark = "✓" if rg.grounded else "✗"
        c_mark = "✓" if len(rg.unsupported_claims) == 0 else "✗"
        print(f"  {p_mark} Addresses Primary Symptom")
        print(f"  {e_mark} Grounded in Verified Evidence (Score: {rg.support_score:.2f})")
        print(f"  {c_mark} No Hazardous / Unsupported Troubleshooting Claims")
        print(f"  • Verification Status:    {rg.verification_status.value}")
        print(f"  • Rationale:              {rg.explanation}")

    # Final Decision
    print("\n[6] FINAL ROUTING & ACTION")
    if res.routing_decision == RoutingDecisionType.AUTO_HANDLE:
        print(f"  DECISION: >>> AUTO_HANDLE <<<")
        print(f"\n  GENERATED BRAND SUPPORT RESPONSE:")
        print(f"  ----------------------------------------------------------------------")
        print(f"  \"{res.grounded_response}\"")
        print(f"  ----------------------------------------------------------------------")
    else:
        print(f"  DECISION: >>> ESCALATE_TO_HUMAN <<<")
        if res.escalation_package:
            pkg = res.escalation_package
            print(f"\n  ENRICHED HUMAN ESCALATION PACKAGE:")
            print(f"  • Why Not Auto-Handled:   {'; '.join(pkg.why_not_auto_handled[:2])}")
            print(f"  • Recommended Action:     {pkg.recommended_human_action}")
            print(f"  • Candidate Approaches:")
            for intent_name, strategy in list(pkg.candidate_resolution_approaches.items())[:2]:
                print(f"    - [{intent_name}]: {strategy[:80]}...")
    print("=" * 80 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="SupportGraph AI Phase 8 Evidence Resolution Simulator")
    parser.add_argument("--message", type=str, default="", help="Custom message to test")
    parser.add_argument("--interactive", action="store_true", help="Interactive terminal prompt")
    args = parser.parse_args()

    engine = SupportResolutionEngine()

    if args.message:
        res = engine.process_message(args.message, case_id="custom_cli_test", log_audit=True)
        render_resolution_result(args.message, res, title="Custom Inquiry")
        return

    if args.interactive:
        print("\n=== SupportGraph AI Phase 8 Interactive Simulator ===")
        print("Type 'exit' or 'quit' to stop.\n")
        while True:
            try:
                user_msg = input("\nEnter customer support message: ").strip()
                if user_msg.lower() in ("exit", "quit", ""):
                    break
                res = engine.process_message(user_msg, case_id="interactive_cli", log_audit=True)
                render_resolution_result(user_msg, res, title="Interactive Adjudication")
            except (KeyboardInterrupt, EOFError):
                break
        return

    # Run standard demonstration scenarios
    print("\n" + "#" * 80)
    print("SUPPORTGRAPH AI — PHASE 8 OPERATIONAL DEMONSTRATION SUITE")
    print("Evidence-Grounded Response Generation & Resolution Verification")
    print("#" * 80)

    for sc in SCENARIOS:
        res = engine.process_message(sc["message"], case_id=sc["id"], log_audit=False)
        render_resolution_result(sc["message"], res, title=sc["title"])


if __name__ == "__main__":
    main()
