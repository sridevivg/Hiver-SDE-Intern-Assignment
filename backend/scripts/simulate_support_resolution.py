"""
SupportGraph AI — Evidence-Aware Support Resolution Simulator (Phase 7)

Demonstrates the complete Phase 7 resolution pipeline:
1. Problem Understanding: Structured device, symptom, causality extraction
2. Top-K Candidate Generation: Ranked intent predictions
3. Primary Problem Selection: Decouples causal trigger from core operational symptom
4. Ambiguity Analysis: Classifies CLEAR vs CAUSE_VS_SYMPTOM vs GENUINE_AMBIGUITY
5. Evidence Retrieval: Historical AppleSupport cases categorized into operational match tiers
6. Resolution Outcome: Brand-grounded support response (Auto-Handle) or Decision-Support Package (Escalate)

Usage:
  python -m backend.scripts.simulate_support_resolution
  python -m backend.scripts.simulate_support_resolution --message "battery draining fast after updating to iOS 11.1"
  python -m backend.scripts.simulate_support_resolution --interactive
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

try:
    from app.core.config import settings
    from app.resolution.support_resolution_engine import (
        SupportResolutionEngine,
        SupportResolutionResult,
    )
    from app.schemas.intent_routing import RoutingDecisionType
except ModuleNotFoundError:
    from backend.app.core.config import settings  # type: ignore[no-redef]
    from backend.app.resolution.support_resolution_engine import (  # type: ignore[no-redef]
        SupportResolutionEngine,
        SupportResolutionResult,
    )
    from backend.app.schemas.intent_routing import (  # type: ignore[no-redef]
        RoutingDecisionType,
    )


def format_resolution_output(message: str, result: SupportResolutionResult) -> str:
    """Format the evidence-aware resolution output for clean terminal display."""
    border = "=" * 80
    sub_border = "-" * 80

    lines = [
        border,
        " SUPPORTGRAPH AI — EVIDENCE-AWARE SUPPORT RESOLUTION PIPELINE",
        border,
        f"Customer Message:\n  \"{message}\"",
        sub_border,
        "1. Problem Understanding Profile:",
        f"  • Summary:                 {result.problem_summary}",
        f"  • Ambiguity Classification: {result.ambiguity_analysis.ambiguity_type.value}",
        f"  • Primary Intent Selected: {result.primary_intent} (Confidence: {result.confidence * 100:.1f}%)",
    ]

    amb = result.ambiguity_analysis
    if amb.contextual_cause:
        lines.append(f"  • Contextual Cause:        {amb.contextual_cause} (Disambiguated from core symptom)")

    lines.append(sub_border)
    lines.append("2. Historical Evidence Retrieved:")
    if not result.evidence_cases:
        lines.append("  (No relevant historical cases retrieved)")
    else:
        for idx, case in enumerate(result.evidence_cases, 1):
            tier_badge = f"[{case.match_tier.value}]"
            lines.append(
                f"  Case #{idx} {tier_badge:<25} Sim: {case.similarity_score:.3f} (OpSim: {case.operational_similarity:.3f}) | Intent: {case.historical_intent}"
            )
            lines.append(f"    Inquiry:    \"{case.historical_customer_message[:75]}...\"")
            lines.append(f"    Resolution: \"{case.historical_brand_response[:75]}...\"")
            lines.append(f"    Rationale:  {case.retrieval_explanation}")

    lines.append(sub_border)
    decision_badge = (
        "🟢 [AUTO_HANDLE — BRAND TROUBLESHOOTING RESPONSE]"
        if result.routing_decision == RoutingDecisionType.AUTO_HANDLE
        else "🔴 [ESCALATE_TO_HUMAN — DECISION SUPPORT PACKAGE]"
    )
    lines.append(f"3. Final Resolution Decision: {decision_badge}")
    lines.append(f"   Explanation: {result.explanation}")

    if result.grounded_response:
        lines.append(sub_border)
        lines.append("4. Grounded Brand Support Response:")
        lines.append(f"  \"{result.grounded_response}\"")

    if result.escalation_package:
        lines.append(sub_border)
        lines.append("4. Human Review Escalation Package:")
        lines.append(f"  • Ambiguity Reason: {result.escalation_package.ambiguity_reason}")
        lines.append("  • Candidate Intent Options:")
        for cand in result.escalation_package.top_candidates:
            lines.append(f"    - {cand.intent:<32} (Conf: {cand.confidence * 100:.1f}%)")
        lines.append("  • Candidate Resolution Workflows:")
        for int_name, workflow in result.escalation_package.candidate_resolution_approaches.items():
            lines.append(f"    [{int_name}]: {workflow[:70]}...")

    lines.append(border)
    return "\n".join(lines)


def run_simulation(
    message: str,
    engine: Optional[SupportResolutionEngine] = None,
) -> None:
    eng = engine or SupportResolutionEngine()
    result = eng.process_message(customer_message=message)
    output = format_resolution_output(message, result)
    print(output)


def interactive_mode() -> None:
    print("\nStarting SupportGraph AI Interactive Resolution Simulator (Ctrl+C or 'exit' to quit)\n")
    eng = SupportResolutionEngine()

    while True:
        try:
            msg = input("\nEnter customer inquiry: ").strip()
            if not msg:
                continue
            if msg.lower() in ("exit", "quit", "q"):
                print("Exiting simulator.")
                break
            run_simulation(msg, engine=eng)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting simulator.")
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="SupportGraph AI — Evidence-Aware Support Resolution Simulator")
    parser.add_argument("--message", type=str, help="Customer message text to process")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive simulation loop")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")

    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
    elif args.message:
        eng = SupportResolutionEngine()
        if args.json:
            res = eng.process_message(args.message)
            print(json.dumps(res.to_dict(), indent=2))
        else:
            run_simulation(args.message, engine=eng)
    else:
        # Run three representative Phase 7 demonstration scenarios
        scenarios = [
            (
                "SCENARIO 1: Clear Operational Problem (Battery Issue)",
                "My iPhone 7 battery is draining completely within 2 hours of use. Health is down to 74%."
            ),
            (
                "SCENARIO 2: Cause vs Symptom Disambiguation (Update Trigger + Hardware Symptom)",
                "Ever since I updated to iOS 11.0.3, my iPhone 8 microphone produces buzzing noise on calls."
            ),
            (
                "SCENARIO 3: Genuine Ambiguity & Multi-Symptom Escalation",
                "phone updated yesterday and now screen flickers, audio doesn't play, and it won't connect to wifi or charge"
            ),
            (
                "SCENARIO 4: Insufficient Information / Vague Query",
                "phone broken please help"
            ),
        ]

        print("\n" + "=" * 80)
        print(" SUPPORTGRAPH AI — PHASE 7 DEMONSTRATION SUITE")
        print("=" * 80 + "\n")

        eng = SupportResolutionEngine()
        for title, msg in scenarios:
            print(f">>> {title}")
            run_simulation(msg, engine=eng)
            print()


if __name__ == "__main__":
    main()
