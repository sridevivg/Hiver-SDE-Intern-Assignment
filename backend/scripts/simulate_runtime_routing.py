"""
SupportGraph AI — Runtime Intent Routing Simulator (Phase 6)

Simulates the runtime message intake pipeline:
1. Customer Message Intake
2. Top-K Intent Prediction with Calibrated Confidence
3. Uncertainty Analysis (Margin, Entropy)
4. Deterministic Explainable Routing (AUTO_HANDLE vs ESCALATE_TO_HUMAN)

Usage:
  python -m backend.scripts.simulate_runtime_routing --message "my battery is draining in 2 hours since updating"
  python -m backend.scripts.simulate_runtime_routing --interactive
"""
from __future__ import annotations

import argparse
import sys
from typing import Optional

try:
    from app.core.config import settings
    from app.intent.classifier import TopKIntentClassifier
    from app.intent.router import IntentRouter
    from app.schemas.intent_routing import IntentAnalysis, RoutingDecision, RoutingDecisionType
except ModuleNotFoundError:
    from backend.app.core.config import settings  # type: ignore[no-redef]
    from backend.app.intent.classifier import TopKIntentClassifier  # type: ignore[no-redef]
    from backend.app.intent.router import IntentRouter  # type: ignore[no-redef]
    from backend.app.schemas.intent_routing import (  # type: ignore[no-redef]
        IntentAnalysis,
        RoutingDecision,
        RoutingDecisionType,
    )


def format_simulation_output(message: str, analysis: IntentAnalysis, routing: RoutingDecision) -> str:
    """Format the routing simulation output into a clean terminal report."""
    border = "=" * 76
    sub_border = "-" * 76

    lines = [
        border,
        " SUPPORTGRAPH AI — RUNTIME INTENT ROUTING SIMULATION",
        border,
        f"Customer Message:\n  \"{message}\"",
        sub_border,
        "Top-K Candidate Intent Predictions:",
    ]

    medals = ["🥇 Top-1", "🥈 Top-2", "🥉 Top-3", "   Top-4", "   Top-5"]
    for idx, pred in enumerate(analysis.top_predictions):
        tag = medals[idx] if idx < len(medals) else f"   Top-{idx+1}"
        bar_len = int(pred.confidence * 25)
        bar = "█" * bar_len + "░" * (25 - bar_len)
        lines.append(
            f"  {tag}: {pred.intent:<32} {bar} {pred.confidence * 100:>5.1f}%"
        )
        if pred.reasoning:
            lines.append(f"         Reasoning: {pred.reasoning}")

    lines.append(sub_border)
    lines.append("Uncertainty Signals:")
    lines.append(f"  • Top-1 Confidence:     {analysis.top_confidence * 100:.1f}% (Threshold: {routing.confidence_threshold * 100:.1f}%)")
    lines.append(f"  • Confidence Margin:    {analysis.confidence_margin * 100:.1f}% (Threshold: {routing.margin_threshold * 100:.1f}%)")
    if analysis.normalized_entropy is not None:
        lines.append(f"  • Normalized Entropy:   {analysis.normalized_entropy:.4f}")

    lines.append(sub_border)
    decision_badge = (
        "🟢 [AUTO_HANDLE]"
        if routing.decision == RoutingDecisionType.AUTO_HANDLE
        else "🔴 [ESCALATE_TO_HUMAN]"
    )
    lines.append(f"Routing Decision:  {decision_badge}")
    lines.append(f"Routing Reason:\n  {routing.reason}")
    lines.append(border)

    return "\n".join(lines)


def run_simulation(
    message: str,
    classifier: Optional[TopKIntentClassifier] = None,
    router: Optional[IntentRouter] = None,
) -> None:
    clf = classifier or TopKIntentClassifier()
    rt = router or IntentRouter()

    analysis = clf.classify(customer_message=message)
    decision = rt.route(analysis)

    output = format_simulation_output(message, analysis, decision)
    print(output)


def interactive_mode() -> None:
    print("\nStarting SupportGraph AI Interactive Routing Simulator (Ctrl+C or 'exit' to quit)\n")
    clf = TopKIntentClassifier()
    rt = IntentRouter()

    while True:
        try:
            msg = input("\nEnter customer message: ").strip()
            if not msg:
                continue
            if msg.lower() in ("exit", "quit", "q"):
                print("Exiting simulator.")
                break
            run_simulation(msg, classifier=clf, router=rt)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting simulator.")
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="SupportGraph AI — Runtime Intent Routing Simulator")
    parser.add_argument("--message", type=str, help="Customer message text to simulate")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive message simulation loop")

    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
    elif args.message:
        run_simulation(args.message)
    else:
        # Run sample demonstration cases
        samples = [
            "My battery is draining completely within 2 hours after updating to iOS 11.0.3 on iPhone 7.",
            "I can't hear anything from the loudspeaker when receiving calls, but headphones work fine.",
            "Why did you charge my card $9.99 for an iTunes subscription I canceled last week?",
            "Screen is cracked and unresponsive.",
            "phone broken please help",
        ]
        print("\n=== Running Sample Customer Support Routing Simulations ===\n")
        clf = TopKIntentClassifier()
        rt = IntentRouter()
        for sample in samples:
            run_simulation(sample, classifier=clf, router=rt)
            print()


if __name__ == "__main__":
    main()
