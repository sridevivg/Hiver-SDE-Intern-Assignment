"""
SupportGraph AI — Safe Multi-Signal Routing Simulator (Phase 7.1)

Demonstrates the 5 core Phase 7.1 validation scenarios:
1. Scenario A — Clear Case: "My iPhone battery dies within two hours." -> AUTO_HANDLE
2. Scenario B — Cause vs Symptom: "My battery drains quickly after the latest iOS update." -> AUTO_HANDLE (Disambiguated)
3. Scenario C — Genuine Ambiguity: "My phone is messed up after the update." -> ESCALATE_TO_HUMAN
4. Scenario D — Insufficient Information: "Apple please fix this." -> ESCALATE_TO_HUMAN (Veto)
5. Scenario E — High Confidence but Contradictory Evidence: -> ESCALATE_TO_HUMAN (Safety Veto)

Usage:
  python -m backend.scripts.simulate_safe_routing
  python -m backend.scripts.simulate_safe_routing --interactive
  python -m backend.scripts.simulate_safe_routing --message "my screen is shattered"
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

try:
    from app.core.config import settings
    from app.intent.ambiguity_decision_gate import AmbiguityDecisionGate, DecisionGateResult
    from app.resolution.support_resolution_engine import (
        SupportResolutionEngine,
        SupportResolutionResult,
    )
    from app.schemas.intent_routing import RoutingDecisionType
except ModuleNotFoundError:
    from backend.app.core.config import settings  # type: ignore[no-redef]
    from backend.app.intent.ambiguity_decision_gate import (  # type: ignore[no-redef]
        AmbiguityDecisionGate,
        DecisionGateResult,
    )
    from backend.app.resolution.support_resolution_engine import (  # type: ignore[no-redef]
        SupportResolutionEngine,
        SupportResolutionResult,
    )
    from backend.app.schemas.intent_routing import (  # type: ignore[no-redef]
        RoutingDecisionType,
    )


def format_safe_simulation_output(message: str, result: SupportResolutionResult) -> str:
    """Format safe decision gate simulation output for terminal display."""
    border = "=" * 80
    sub_border = "-" * 80

    lines = [
        border,
        " SUPPORTGRAPH AI — SAFE DECISION GATE SIMULATION (PHASE 7.1)",
        border,
        f"Customer Message:\n  \"{message}\"",
        sub_border,
        "1. Problem Understanding & Primary Extraction:",
        f"  • Understood Summary:     {result.problem_summary}",
        f"  • Primary Intent Selected: {result.primary_intent} (AI Confidence: {result.confidence * 100:.1f}%)",
    ]

    gate = result.gate_result
    if gate:
        sig = gate.signals
        lines.extend([
            sub_border,
            "2. Multi-Signal Clarity Evaluation:",
            f"  • Signal A (Information Sufficiency):   {sig.sufficiency.value:<16} ({sig.sufficiency_reason})",
            f"  • Signal B (Problem Strength):          {sig.problem_strength.value:<16} ({sig.strength_reason})",
            f"  • Signal C (Candidate Conflict):        {sig.candidate_conflict.value:<16} ({sig.conflict_reason})",
            f"  • Signal D (Symptom-Intent Agreement):  {sig.evidence_agreement.value:<16} ({sig.agreement_reason})",
            f"  • Signal E (Retrieval Evidence):        {sig.retrieval_quality.value:<16} ({sig.retrieval_reason})",
            f"  • Signal F (Multi-Symptom Complexity):  {sig.multi_symptom.value:<16} ({sig.multi_symptom_reason})",
            sub_border,
            "3. Safety Checklist & Veto Status:",
        ])
        for check_name, passed in gate.checklist.items():
            mark = "✓ PASS" if passed else "✗ FAIL"
            lines.append(f"  [{mark}] {check_name}")

        if sig.is_vetoed:
            lines.append(f"  🚨 SAFETY VETO TRIGGERED: {'; '.join(sig.veto_reasons)}")

    lines.append(sub_border)
    decision_badge = (
        "🟢 [AUTO_HANDLE — SAFE BRAND TROUBLESHOOTING REPLY]"
        if result.routing_decision == RoutingDecisionType.AUTO_HANDLE
        else "🔴 [ESCALATE_TO_HUMAN — DECISION SUPPORT PACKAGE]"
    )
    lines.append(f"4. Final Gating Decision: {decision_badge}")
    lines.append(f"   Rationale: {result.explanation}")

    if result.grounded_response:
        lines.append(sub_border)
        lines.append("5. Grounded Brand Response:")
        lines.append(f"  \"{result.grounded_response}\"")

    if result.escalation_package:
        lines.append(sub_border)
        lines.append("5. Human Review Escalation Package:")
        lines.append("  • Why AI Did Not Auto-Handle:")
        for reason in result.escalation_package.why_not_auto_handled:
            lines.append(f"    - {reason}")
        lines.append("  • Candidate Intent Hypotheses:")
        for cand in result.escalation_package.top_candidates:
            lines.append(f"    - {cand.intent:<32} (Conf: {cand.confidence * 100:.1f}%)")
        lines.append("  • Related Historical Support Cases:")
        for c in result.escalation_package.related_evidence[:2]:
            lines.append(f"    - [{c.match_tier.value}] \"{c.historical_customer_message[:60]}...\"")

    lines.append(border)
    return "\n".join(lines)


def run_simulation(message: str, engine: Optional[SupportResolutionEngine] = None) -> None:
    eng = engine or SupportResolutionEngine()
    res = eng.process_message(customer_message=message)
    print(format_safe_simulation_output(message, res))


def interactive_mode() -> None:
    print("\nStarting SupportGraph AI Interactive Safe Routing Simulator (Ctrl+C or 'exit' to quit)\n")
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
    parser = argparse.ArgumentParser(description="SupportGraph AI — Safe Multi-Signal Routing Simulator")
    parser.add_argument("--message", type=str, help="Customer message text to simulate")
    parser.add_argument("--interactive", action="store_true", help="Interactive terminal mode")
    parser.add_argument(
        "--provider",
        type=str,
        default="heuristic",
        choices=["groq", "ollama", "heuristic"],
        help="Classifier provider override ('heuristic', 'groq', 'ollama')",
    )

    args = parser.parse_args()

    from backend.app.intent.classifier import TopKIntentClassifier
    from backend.app.schemas.intent_routing import IntentAnalysis

    if args.provider == "heuristic":
        clf = TopKIntentClassifier(api_key="")
        clf.classify = lambda customer_message, normalized_message="", conversation_context="", top_k=3: (
            clf.classify_heuristic_fallback(customer_message, requested_k=top_k or 3)
            and IntentAnalysis(
                top_predictions=clf.classify_heuristic_fallback(customer_message, requested_k=top_k or 3),
                top_confidence=clf.classify_heuristic_fallback(customer_message, requested_k=top_k or 3)[0].confidence,
                confidence_margin=round(
                    max(
                        0.0,
                        clf.classify_heuristic_fallback(customer_message, requested_k=top_k or 3)[0].confidence
                        - (clf.classify_heuristic_fallback(customer_message, requested_k=top_k or 3)[1].confidence if len(clf.classify_heuristic_fallback(customer_message, requested_k=top_k or 3)) > 1 else 0.0),
                    ),
                    4,
                ),
                normalized_entropy=0.10,
                model_name="heuristic_baseline",
            )
        )
    else:
        clf = TopKIntentClassifier(provider=args.provider)

    eng = SupportResolutionEngine(classifier=clf)

    if args.interactive:
        interactive_mode()
    elif args.message:
        run_simulation(args.message, engine=eng)
    else:
        # Run 5 key Phase 7.1 scenarios
        scenarios = [
            (
                "SCENARIO A: Clear Problem (Battery Drain)",
                "My iPhone battery dies within two hours.",
            ),
            (
                "SCENARIO B: Cause vs Symptom Disambiguation",
                "My battery drains quickly after the latest iOS update.",
            ),
            (
                "SCENARIO C: Genuine Ambiguity (Vague Post-Update Complaint)",
                "My phone is messed up after the update.",
            ),
            (
                "SCENARIO D: Insufficient Information (Veto Demonstration)",
                "Apple please fix this.",
            ),
            (
                "SCENARIO E: Multi-Symptom Complexity (Disparate Subsystems)",
                "My screen flickers, speakers crackle, and it won't connect to wifi or charge.",
            ),
        ]

        print("\n" + "=" * 80)
        print(" SUPPORTGRAPH AI — PHASE 7.1 SAFE DECISION GATE DEMONSTRATION SUITE")
        print("=" * 80 + "\n")

        for title, msg in scenarios:
            print(f">>> {title}")
            run_simulation(msg, engine=eng)
            print()


if __name__ == "__main__":
    main()
