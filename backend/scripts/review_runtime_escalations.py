"""
SupportGraph AI — Runtime HITL Escalation Review CLI (Phase 6)

Interactive CLI tool for human operators to review and adjudicate runtime escalated cases.

Key Features:
- Displays complete customer message and AI Top-K candidate predictions
- Displays transparent routing rationale (e.g. confidence margin, low confidence)
- Allows operator to:
  [1] Accept Top-1 candidate
  [2] Select Top-2 candidate
  [3] Override with any approved taxonomy intent
  [4] Mark as unclear / needs review
  [s] Skip
- Logs decisions append-only to data/runtime/runtime_escalation_reviews.csv
- Preserves complete data isolation from Phase 5 golden benchmark datasets.
"""
from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path
from typing import Optional

try:
    from app.core.config import settings
    from app.intent.classifier import TopKIntentClassifier
    from app.intent.escalation import RuntimeEscalationManager
    from app.intent.router import IntentRouter
    from app.nlp.taxonomy_finalization import (
        DEFAULT_CANDIDATE_TAXONOMY_PATH,
        load_candidate_taxonomy,
    )
    from app.schemas.intent_routing import (
        HumanReviewActionRequest,
        HumanReviewActionType,
        IntentAnalysis,
        RoutingDecision,
        RoutingDecisionType,
    )
except ModuleNotFoundError:
    from backend.app.core.config import settings  # type: ignore[no-redef]
    from backend.app.intent.classifier import TopKIntentClassifier  # type: ignore[no-redef]
    from backend.app.intent.escalation import RuntimeEscalationManager  # type: ignore[no-redef]
    from backend.app.intent.router import IntentRouter  # type: ignore[no-redef]
    from backend.app.nlp.taxonomy_finalization import (  # type: ignore[no-redef]
        DEFAULT_CANDIDATE_TAXONOMY_PATH,
        load_candidate_taxonomy,
    )
    from backend.app.schemas.intent_routing import (  # type: ignore[no-redef]
        HumanReviewActionRequest,
        HumanReviewActionType,
        IntentAnalysis,
        RoutingDecision,
        RoutingDecisionType,
    )


def review_single_case(
    case_id: str,
    message: str,
    analysis: IntentAnalysis,
    routing: RoutingDecision,
    escalation_mgr: RuntimeEscalationManager,
    allowed_intents: list[str],
    reviewer_id: str = "human_operator",
) -> Optional[str]:
    """Present a single escalated case to human reviewer and log their decision."""
    border = "=" * 74
    sub_border = "-" * 74

    print("\n" + border)
    print(f" ESCALATED SUPPORT CASE [ID: {case_id}]")
    print(border)
    print(f"Customer Message:\n  \"{message}\"")
    print(sub_border)
    print("AI Candidate Intent Predictions:")
    medals = ["🥇 [1] Top-1", "🥈 [2] Top-2", "🥉 [3] Top-3"]
    for idx, pred in enumerate(analysis.top_predictions[:3]):
        tag = medals[idx] if idx < len(medals) else f"    [{idx+1}]"
        bar = "█" * int(pred.confidence * 20) + "░" * (20 - int(pred.confidence * 20))
        print(f"  {tag}: {pred.intent:<32} {bar} {pred.confidence * 100:>5.1f}%")
        if pred.reasoning:
            print(f"         Reasoning: {pred.reasoning}")

    print(sub_border)
    print(f"Escalation Rationale:\n  {routing.reason}")
    print(sub_border)
    print("Available Review Actions:")
    print("  [1] Accept Top-1 prediction")
    if len(analysis.top_predictions) > 1:
        print(f"  [2] Select Top-2 prediction ('{analysis.top_predictions[1].intent}')")
    print("  [3] Override with another taxonomy intent")
    print("  [4] Mark as unclear_needs_review")
    print("  [s] Skip this case")
    print("  [q] Quit review session")

    while True:
        choice = input("\nSelect action [1/2/3/4/s/q]: ").strip().lower()

        if choice == "q":
            return "QUIT"
        elif choice == "s":
            print("Skipped case.")
            return "SKIPPED"
        elif choice == "1":
            req = HumanReviewActionRequest(
                case_id=case_id,
                customer_message=message,
                action=HumanReviewActionType.ACCEPT_TOP_1,
                selected_intent=analysis.top_1_intent,
                reviewer_id=reviewer_id,
                top_predictions=analysis.top_predictions,
                confidence_margin=analysis.confidence_margin,
                routing_reason=routing.reason,
            )
            escalation_mgr.record_human_decision(req, analysis=analysis, routing=routing)
            print(f" Recorded: Accepted Top-1 intent '{analysis.top_1_intent}'")
            return "RECORDED"
        elif choice == "2":
            if len(analysis.top_predictions) < 2:
                print("No Top-2 prediction available.")
                continue
            top_2_label = analysis.top_predictions[1].intent
            req = HumanReviewActionRequest(
                case_id=case_id,
                customer_message=message,
                action=HumanReviewActionType.SELECT_TOP_2,
                selected_intent=top_2_label,
                reviewer_id=reviewer_id,
                top_predictions=analysis.top_predictions,
                confidence_margin=analysis.confidence_margin,
                routing_reason=routing.reason,
            )
            escalation_mgr.record_human_decision(req, analysis=analysis, routing=routing)
            print(f" Recorded: Selected Top-2 intent '{top_2_label}'")
            return "RECORDED"
        elif choice == "3":
            print("\nSelect Taxonomy Intent:")
            for idx, intent_name in enumerate(allowed_intents, 1):
                print(f"  {idx}. {intent_name}")
            try:
                sub_choice = input("Enter number (or 0 to cancel): ").strip()
                sub_idx = int(sub_choice)
                if sub_idx == 0:
                    continue
                if 1 <= sub_idx <= len(allowed_intents):
                    chosen_intent = allowed_intents[sub_idx - 1]
                    notes = input("Optional reviewer notes: ").strip()
                    req = HumanReviewActionRequest(
                        case_id=case_id,
                        customer_message=message,
                        action=HumanReviewActionType.OVERRIDE_INTENT,
                        selected_intent=chosen_intent,
                        reviewer_notes=notes,
                        reviewer_id=reviewer_id,
                        top_predictions=analysis.top_predictions,
                        confidence_margin=analysis.confidence_margin,
                        routing_reason=routing.reason,
                    )
                    escalation_mgr.record_human_decision(req, analysis=analysis, routing=routing)
                    print(f" Recorded: Overridden with '{chosen_intent}'")
                    return "RECORDED"
                else:
                    print("Invalid selection.")
            except ValueError:
                print("Invalid input.")
        elif choice == "4":
            notes = input("Optional reviewer notes: ").strip()
            req = HumanReviewActionRequest(
                case_id=case_id,
                customer_message=message,
                action=HumanReviewActionType.MARK_UNCLEAR,
                selected_intent="unclear_needs_review",
                reviewer_notes=notes,
                reviewer_id=reviewer_id,
                top_predictions=analysis.top_predictions,
                confidence_margin=analysis.confidence_margin,
                routing_reason=routing.reason,
            )
            escalation_mgr.record_human_decision(req, analysis=analysis, routing=routing)
            print(" Recorded: Marked as unclear_needs_review")
            return "RECORDED"
        else:
            print("Invalid option. Please enter 1, 2, 3, 4, s, or q.")


def main() -> None:
    parser = argparse.ArgumentParser(description="SupportGraph AI — Runtime HITL Escalation Review CLI")
    parser.add_argument("--message", type=str, help="Specific customer message to review")
    parser.add_argument("--reviewer", type=str, default="human_operator", help="Reviewer identifier")
    parser.add_argument("--demo", action="store_true", help="Run interactive review over sample ambiguous cases")

    args = parser.parse_args()

    clf = TopKIntentClassifier()
    rt = IntentRouter()
    escalation_mgr = RuntimeEscalationManager()
    allowed_intents = sorted(clf.allowed_labels)

    if args.message:
        analysis = clf.classify(args.message)
        decision = rt.route(analysis)
        case_id = f"case_{uuid.uuid4().hex[:8]}"
        review_single_case(
            case_id=case_id,
            message=args.message,
            analysis=analysis,
            routing=decision,
            escalation_mgr=escalation_mgr,
            allowed_intents=allowed_intents,
            reviewer_id=args.reviewer,
        )
    elif args.demo:
        sample_cases = [
            "since the update my sound is crackling but screen also flickered once",
            "phone won't turn on and battery was low",
            "charged twice on my bill and app is also frozen",
            "why is my keyboard typing question marks after updating iOS?",
        ]
        print("\n=== Launching Demo Runtime Escalation Review Queue ===")
        for idx, sample in enumerate(sample_cases, 1):
            analysis = clf.classify(sample)
            decision = rt.route(analysis)
            case_id = f"demo_case_{idx:03d}"
            res = review_single_case(
                case_id=case_id,
                message=sample,
                analysis=analysis,
                routing=decision,
                escalation_mgr=escalation_mgr,
                allowed_intents=allowed_intents,
                reviewer_id=args.reviewer,
            )
            if res == "QUIT":
                break
    else:
        print("Please provide --message \"...\" or --demo to run escalation review.")
        parser.print_help()


if __name__ == "__main__":
    main()
