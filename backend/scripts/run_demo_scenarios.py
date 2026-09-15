"""
SupportGraph AI — Phase 14 Isolated Demo Scenarios.

Executes 7 deterministic demonstration scenarios through the production engine.
All demo records are:
  - Tagged with is_demo=True and case_id prefix DEMO_ONLY_
  - Isolated from the golden benchmark, historical corpus, and approved evidence
  - Clearly labeled DEMO / SYNTHETIC in all outputs

Do NOT promote demo resolutions to trusted evidence.
Do NOT confuse demo results with live production metrics.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.logging import get_logger
from app.resolution.support_resolution_engine import SupportResolutionEngine
from app.schemas.intent_routing import RoutingDecisionType

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────
# Scenario Definitions
# All inputs are synthetic / illustrative, NOT real customer data.
# ─────────────────────────────────────────────────────────────

DEMO_SCENARIOS = [
    {
        "demo_id": "DEMO_ONLY_01_clear_case",
        "label": "DEMO 1 — Clear Support Case (AirPods Pairing)",
        "description": "A clear, unambiguous AirPods Bluetooth pairing failure.",
        "customer_message": "My AirPods Pro will not connect to my iPhone 15. They appear in Bluetooth settings but never finish pairing.",
        "expected_decision": "AUTO_HANDLE",
        "expected_outcome_contains": "AUTO_HANDLED",
        "data_label": "DEMO / SYNTHETIC",
    },
    {
        "demo_id": "DEMO_ONLY_02_ambiguous",
        "label": "DEMO 2 — Ambiguous Case (Clarification Required)",
        "description": "Underspecified message that requires clarification before resolution.",
        "customer_message": "broken",
        "expected_decision": "ESCALATE_TO_HUMAN",
        "expected_outcome_contains": "CLARIFICATION",
        "data_label": "DEMO / SYNTHETIC",
    },
    {
        "demo_id": "DEMO_ONLY_03_evidence_limited",
        "label": "DEMO 3 — Evidence-Limited Case",
        "description": "Highly specialized issue with no historical precedent in the corpus.",
        "customer_message": "My custom USB-C DAC oscilloscope adapter won't communicate with the iPad kernel USB driver.",
        "expected_decision": "ESCALATE_TO_HUMAN",
        "expected_outcome_contains": "EVIDENCE_LIMITED",
        "data_label": "DEMO / SYNTHETIC",
    },
    {
        "demo_id": "DEMO_ONLY_04_conflict",
        "label": "DEMO 4 — Conflict Case",
        "description": "Conflicting or multi-symptom message requiring human review.",
        "customer_message": "My iPhone battery drains fast but also the device overheats only when charging and then stops working when cold.",
        "expected_decision": "ESCALATE_TO_HUMAN",
        "expected_outcome_contains": "",  # any escalation is valid
        "data_label": "DEMO / SYNTHETIC",
    },
    {
        "demo_id": "DEMO_ONLY_05_thermal_hazard",
        "label": "DEMO 5 — Thermal Hazard (Safety Veto)",
        "description": "Device smoking / extremely hot. System must immediately halt and escalate urgently.",
        "customer_message": "My iPhone battery is bulging and the device is extremely hot and smoking. Glass is cracking.",
        "expected_decision": "ESCALATE_TO_HUMAN",
        "expected_outcome_contains": "ESCALATED",
        "data_label": "DEMO / SYNTHETIC",
    },
    {
        "demo_id": "DEMO_ONLY_06_battery_drain",
        "label": "DEMO 6 — Battery Drain (Resolution)",
        "description": "Classic battery drain issue that resolves cleanly via historical evidence.",
        "customer_message": "My iPhone 14 battery drains very fast on iOS 17, even when idle with no apps open.",
        "expected_decision": "AUTO_HANDLE",
        "expected_outcome_contains": "AUTO_HANDLED",
        "data_label": "DEMO / SYNTHETIC",
    },
    {
        "demo_id": "DEMO_ONLY_07_wifi",
        "label": "DEMO 7 — Wi-Fi Connectivity Issue",
        "description": "Wi-Fi dropping intermittently — tests evidence retrieval and grounding verification.",
        "customer_message": "My MacBook Pro keeps losing Wi-Fi connection every few minutes since updating to macOS Sonoma.",
        "expected_decision": None,  # accept any safe decision
        "expected_outcome_contains": "",
        "data_label": "DEMO / SYNTHETIC",
    },
]


def run_demo_scenarios(engine: Optional[SupportResolutionEngine] = None) -> List[Dict[str, Any]]:
    """
    Execute all demo scenarios through the production engine.
    Returns list of result dicts.
    Demos are isolated from trusted evidence stores.
    """
    if engine is None:
        engine = SupportResolutionEngine()

    results: List[Dict[str, Any]] = []

    print("\n" + "=" * 70)
    print("SupportGraph AI — Phase 14 Demo Scenarios")
    print("ALL DATA IS DEMO / SYNTHETIC — NOT PRODUCTION DATA")
    print("=" * 70)

    passed = 0
    total = len(DEMO_SCENARIOS)

    for sc in DEMO_SCENARIOS:
        print(f"\n{'─' * 60}")
        print(f"[{sc['data_label']}] {sc['label']}")
        print(f"Input: \"{sc['customer_message'][:80]}...\"" if len(sc['customer_message']) > 80 else f"Input: \"{sc['customer_message']}\"")

        try:
            result = engine.process_message(
                customer_message=sc["customer_message"],
                case_id=sc["demo_id"],
                is_demo=True,
            )

            routing = result.routing_decision.value
            outcome = result.outcome.value if result.outcome else ""
            explanation = result.explanation[:200] if result.explanation else ""

            # Check expected outcome
            outcome_ok = True
            if sc.get("expected_decision"):
                if sc["expected_decision"] == "AUTO_HANDLE":
                    outcome_ok = routing == "AUTO_HANDLE"
                else:
                    outcome_ok = routing == "ESCALATE_TO_HUMAN"
            if sc.get("expected_outcome_contains"):
                if sc["expected_outcome_contains"] not in outcome:
                    outcome_ok = False

            status_label = "PASS" if outcome_ok else "FAIL"
            if outcome_ok:
                passed += 1

            print(f"Decision:  {routing}")
            print(f"Outcome:   {outcome}")
            print(f"Reasoning: {explanation[:150]}")
            print(f"Status:    {status_label}")

            results.append({
                "demo_id": sc["demo_id"],
                "label": sc["label"],
                "data_label": sc["data_label"],
                "routing_decision": routing,
                "outcome": outcome,
                "status": status_label,
                "explanation_preview": explanation,
                "is_demo": True,
            })

        except Exception as exc:
            print(f"ERROR: {exc}")
            results.append({
                "demo_id": sc["demo_id"],
                "label": sc["label"],
                "data_label": sc["data_label"],
                "routing_decision": "ERROR",
                "outcome": "ERROR",
                "status": "ERROR",
                "error": str(exc),
                "is_demo": True,
            })

    print(f"\n{'=' * 70}")
    print(f"Demo Results: {passed}/{total} passed")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = run_demo_scenarios()
    out_path = Path("reports/phase_14")
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "demo_scenarios_results.json", "w") as f:
        json.dump({"demo_runs": results, "data_label": "DEMO / SYNTHETIC"}, f, indent=2, default=str)
    print(f"\nDemo results saved to {out_path / 'demo_scenarios_results.json'}")
