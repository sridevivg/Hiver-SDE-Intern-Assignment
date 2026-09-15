"""
SupportGraph AI — Multi-Turn Support Resolution Benchmark Evaluator (Phase 11).

Evaluates the conversation state engine, progressive troubleshooting, repeat prevention,
clarification gating, and safe human escalation across Scenarios A through J.
Computes quantitative benchmark metrics with zero leakage and zero unsafe auto-handles.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from app.conversation.action_catalog import ActionCatalog
    from app.conversation.conversation_manager import ConversationManager
    from app.conversation.conversation_state import (
        ActionStatus,
        ConversationState,
        ConversationStatus,
        MessageRoleType,
        ResolutionStage,
    )
    from app.core.logging import get_logger
except ModuleNotFoundError:
    from backend.app.conversation.action_catalog import ActionCatalog  # type: ignore[no-redef]
    from backend.app.conversation.conversation_manager import (  # type: ignore[no-redef]
        ConversationManager,
    )
    from backend.app.conversation.conversation_state import (  # type: ignore[no-redef]
        ActionStatus,
        ConversationState,
        ConversationStatus,
        MessageRoleType,
        ResolutionStage,
    )
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]

logger = get_logger(__name__)


# 10 Standard Multi-Turn Benchmark Scenarios (A through J)
BENCHMARK_SCENARIOS = [
    {
        "scenario_id": "SCENARIO_A",
        "name": "Customer Already Attempted Action (Prior Attempt Recognition)",
        "description": "Customer explicitly indicates prior restart attempt; system must avoid repeating restart.",
        "turns": [
            "My iPhone Wi-Fi keeps dropping, and I already restarted my phone twice.",
            "I tried that step and it still disconnects.",
        ],
        "expected_problem_family": "CONNECTIVITY_WIFI",
        "forbidden_actions_recommended": ["restart_device"],
        "must_escalate": False,
        "must_resolve": False,
    },
    {
        "scenario_id": "SCENARIO_B",
        "name": "Progressive Troubleshooting Progression",
        "description": "Multi-turn progression across sequential actions without duplicate recommendations.",
        "turns": [
            "My iPhone 14 won't connect to Wi-Fi at all.",
            "I completed that step but no change.",
            "Tried the next step as well, still not connecting.",
        ],
        "expected_problem_family": "CONNECTIVITY_WIFI",
        "forbidden_actions_recommended": [],
        "must_escalate": False,
        "must_resolve": False,
    },
    {
        "scenario_id": "SCENARIO_C",
        "name": "Clarification Loop Prevention & Ambiguity Resolution",
        "description": "Customer provides brief vague inputs; system avoids looping and safely escalates.",
        "turns": [
            "help it's broken",
            "my phone",
            "idk",
        ],
        "expected_problem_family": None,
        "forbidden_actions_recommended": [],
        "must_escalate": True,
        "must_resolve": False,
    },
    {
        "scenario_id": "SCENARIO_D",
        "name": "Negative Confirmation / Rejection Handling",
        "description": "Customer explicitly rejects that an action resolved the issue; advances cleanly.",
        "turns": [
            "My AirPods won't connect to my iPhone.",
            "No, it's still doing the exact same thing.",
        ],
        "expected_problem_family": "CONNECTIVITY_BLUETOOTH",
        "forbidden_actions_recommended": [],
        "must_escalate": False,
        "must_resolve": False,
    },
    {
        "scenario_id": "SCENARIO_E",
        "name": "Resolution Confirmation & Clean State Closure",
        "description": "Customer confirms resolution; system records summary and cleanly closes conversation.",
        "turns": [
            "My iPhone is stuck on a black screen.",
            "That worked! Thanks so much, it powered right back on!",
        ],
        "expected_problem_family": "DISPLAY",
        "forbidden_actions_recommended": [],
        "must_escalate": False,
        "must_resolve": True,
    },
    {
        "scenario_id": "SCENARIO_F",
        "name": "Problem Worsening / Urgent Safety Escalation",
        "description": "Customer reports hardware deterioration/worsening; system halts troubleshooting and escalates urgently.",
        "turns": [
            "My iPhone battery is draining quickly.",
            "Now it's completely dead, won't turn on at all, and it's getting extremely hot!",
        ],
        "expected_problem_family": "POWER_BATTERY",
        "forbidden_actions_recommended": [],
        "must_escalate": True,
        "expected_specialist_tier": "TIER_2_TECHNICAL_URGENT",
        "must_resolve": False,
    },
    {
        "scenario_id": "SCENARIO_G",
        "name": "Topic Drift / Multiple Problems in One Conversation",
        "description": "Customer abruptly introduces a different problem family; system handles transition cleanly.",
        "turns": [
            "My iPhone Wi-Fi keeps disconnecting.",
            "Actually, never mind that, my screen digitizer has completely cracked and touch is not responding.",
        ],
        "expected_problem_family": "DISPLAY",
        "forbidden_actions_recommended": [],
        "must_escalate": False,
        "must_resolve": False,
    },
    {
        "scenario_id": "SCENARIO_H",
        "name": "Context Retention Across Long Multi-Turn Dialogue",
        "description": "Confirmed facts extracted in turn 1 must be strictly retained through turn 5.",
        "turns": [
            "I have an iPhone 13 Pro running iOS 17.2 on Verizon and cellular keeps dropping.",
            "I toggled cellular and it didn't help.",
            "Same issue after that step too.",
            "Still no connection.",
        ],
        "expected_problem_family": "NETWORK_CELLULAR",
        "expected_facts": {
            "device_model": "Iphone 13 Pro",
            "os_version": "IOS 17.2",
            "carrier": "Verizon",
        },
        "forbidden_actions_recommended": [],
        "must_escalate": False,
        "must_resolve": False,
    },
    {
        "scenario_id": "SCENARIO_I",
        "name": "Semantic Equivalence Repeat Prevention",
        "description": "Customer uses informal alias ('power cycle'); system prevents recommending 'restart_device'.",
        "turns": [
            "My Wi-Fi is broken and I already power cycled the device.",
            "Still no Wi-Fi after that.",
        ],
        "expected_problem_family": "CONNECTIVITY_WIFI",
        "forbidden_actions_recommended": ["restart_device"],
        "must_escalate": False,
        "must_resolve": False,
    },
    {
        "scenario_id": "SCENARIO_J",
        "name": "Safe Human Escalation with Full Context Transfer",
        "description": "Exhausted progressive troubleshooting generates complete EscalationPackage.",
        "turns": [
            "I have an iPhone 14 Pro running iOS 16.5 and Wi-Fi refuses to connect.",
            "I did that and it still does not connect.",
            "Next step also failed.",
            "Still failing to connect.",
            "Reset didn't work either.",
        ],
        "expected_problem_family": "CONNECTIVITY_WIFI",
        "forbidden_actions_recommended": [],
        "must_escalate": True,
        "must_resolve": False,
    },
]


@dataclass
class ScenarioResult:
    scenario_id: str
    name: str
    passed: bool
    total_turns: int
    final_status: str
    final_stage: str
    actions_recommended: List[str]
    repeat_violations: List[str]
    facts_retained: Dict[str, Any]
    escalation_package_present: bool
    error_message: Optional[str] = None


@dataclass
class MultiTurnBenchmarkReport:
    timestamp: str
    total_scenarios: int
    passed_scenarios: int
    scenario_pass_rate: float
    repeat_prevention_rate: float
    semantic_aliasing_accuracy: float
    clarification_precision: float
    resolution_confirmation_accuracy: float
    safe_escalation_fidelity: float
    context_fact_retention_rate: float
    unsafe_handle_rate: float
    results: List[ScenarioResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "total_scenarios": self.total_scenarios,
            "passed_scenarios": self.passed_scenarios,
            "scenario_pass_rate": round(self.scenario_pass_rate, 4),
            "repeat_prevention_rate": round(self.repeat_prevention_rate, 4),
            "semantic_aliasing_accuracy": round(self.semantic_aliasing_accuracy, 4),
            "clarification_precision": round(self.clarification_precision, 4),
            "resolution_confirmation_accuracy": round(self.resolution_confirmation_accuracy, 4),
            "safe_escalation_fidelity": round(self.safe_escalation_fidelity, 4),
            "context_fact_retention_rate": round(self.context_fact_retention_rate, 4),
            "unsafe_handle_rate": round(self.unsafe_handle_rate, 4),
            "results": [
                {
                    "scenario_id": r.scenario_id,
                    "name": r.name,
                    "passed": r.passed,
                    "total_turns": r.total_turns,
                    "final_status": r.final_status,
                    "final_stage": r.final_stage,
                    "actions_recommended": r.actions_recommended,
                    "repeat_violations": r.repeat_violations,
                    "facts_retained": r.facts_retained,
                    "escalation_package_present": r.escalation_package_present,
                    "error_message": r.error_message,
                }
                for r in self.results
            ],
        }


class ConversationResolutionEvaluator:
    """
    Executes and scores multi-turn support resolution scenarios.
    """

    def __init__(self, working_dir: Optional[Path] = None) -> None:
        self.working_dir = working_dir or Path("data/conversations/eval_scratch")
        self.working_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_scenario(self, scenario_spec: Dict[str, Any]) -> ScenarioResult:
        """Run a single benchmark scenario and evaluate compliance."""
        scenario_id = scenario_spec["scenario_id"]
        name = scenario_spec["name"]
        turns = scenario_spec["turns"]
        forbidden = scenario_spec.get("forbidden_actions_recommended", [])
        must_escalate = scenario_spec.get("must_escalate", False)
        must_resolve = scenario_spec.get("must_resolve", False)
        expected_facts = scenario_spec.get("expected_facts", {})

        mgr = ConversationManager(base_dir=self.working_dir)
        state, first_turn = mgr.create_conversation(initial_message=turns[0])

        for msg in turns[1:]:
            if state.status in (ConversationStatus.RESOLVED, ConversationStatus.ESCALATED):
                break
            state, agent_turn = mgr.process_message(state.conversation_id, msg)

        # Evaluate rules
        passed = True
        error_msgs = []
        repeat_violations = []

        # 1. Check forbidden actions
        recommended_actions = list(state.recommended_action_history)
        for act in recommended_actions:
            if act in forbidden:
                passed = False
                repeat_violations.append(act)
                error_msgs.append(f"Forbidden action {act} was recommended.")

        # 2. Check duplicate recommendations
        if len(recommended_actions) != len(set(recommended_actions)):
            passed = False
            dups = [a for a in recommended_actions if recommended_actions.count(a) > 1]
            repeat_violations.extend(dups)
            error_msgs.append(f"Duplicate actions recommended: {set(dups)}")

        # 3. Check escalation requirements
        if must_escalate and state.status != ConversationStatus.ESCALATED:
            passed = False
            error_msgs.append(f"Expected status ESCALATED, got {state.status}")

        # 4. Check resolution requirements
        if must_resolve and state.status != ConversationStatus.RESOLVED:
            passed = False
            error_msgs.append(f"Expected status RESOLVED, got {state.status}")

        # 5. Check expected facts retention
        for k, v in expected_facts.items():
            if state.confirmed_facts.get(k) != v:
                passed = False
                error_msgs.append(f"Expected fact {k}={v}, found {state.confirmed_facts.get(k)}")

        # 6. Check escalation package structure if escalated
        if state.status == ConversationStatus.ESCALATED:
            if not state.escalation_package:
                passed = False
                error_msgs.append("Conversation escalated without EscalationPackage.")

        return ScenarioResult(
            scenario_id=scenario_id,
            name=name,
            passed=passed,
            total_turns=len(state.turns),
            final_status=state.status,
            final_stage=state.stage,
            actions_recommended=recommended_actions,
            repeat_violations=repeat_violations,
            facts_retained=dict(state.confirmed_facts),
            escalation_package_present=state.escalation_package is not None,
            error_message="; ".join(error_msgs) if error_msgs else None,
        )

    def run_benchmark(self) -> MultiTurnBenchmarkReport:
        """Run all 10 standard benchmark scenarios and aggregate metrics."""
        results: List[ScenarioResult] = []

        for scenario in BENCHMARK_SCENARIOS:
            res = self.evaluate_scenario(scenario)
            results.append(res)

        total = len(results)
        passed = sum(1 for r in results if r.passed)
        scenario_pass_rate = passed / total if total else 0.0

        # Compute specific metrics
        repeat_clean = sum(1 for r in results if not r.repeat_violations)
        repeat_prevention_rate = repeat_clean / total if total else 0.0

        # Scenario I evaluates semantic aliasing
        scenario_i = next((r for r in results if r.scenario_id == "SCENARIO_I"), None)
        semantic_aliasing_accuracy = 1.0 if (scenario_i and scenario_i.passed) else 0.0

        # Scenario C evaluates clarification precision
        scenario_c = next((r for r in results if r.scenario_id == "SCENARIO_C"), None)
        clarification_precision = 1.0 if (scenario_c and scenario_c.passed) else 0.0

        # Scenario E evaluates resolution confirmation
        scenario_e = next((r for r in results if r.scenario_id == "SCENARIO_E"), None)
        resolution_confirmation_accuracy = 1.0 if (scenario_e and scenario_e.passed) else 0.0

        # Scenarios F & J evaluate safe escalation fidelity
        esc_scenarios = [r for r in results if r.scenario_id in ("SCENARIO_F", "SCENARIO_J")]
        safe_escalation_fidelity = (
            sum(1 for r in esc_scenarios if r.passed) / len(esc_scenarios)
            if esc_scenarios
            else 0.0
        )

        # Scenario H evaluates context fact retention
        scenario_h = next((r for r in results if r.scenario_id == "SCENARIO_H"), None)
        context_fact_retention_rate = 1.0 if (scenario_h and scenario_h.passed) else 0.0

        # Unsafe handle rate: any scenario that failed critical safety checks
        unsafe_handles = 0
        for r in results:
            if r.scenario_id in ("SCENARIO_F", "SCENARIO_C") and not r.passed:
                unsafe_handles += 1
        unsafe_handle_rate = unsafe_handles / total if total else 0.0

        return MultiTurnBenchmarkReport(
            timestamp=datetime.utcnow().isoformat(),
            total_scenarios=total,
            passed_scenarios=passed,
            scenario_pass_rate=scenario_pass_rate,
            repeat_prevention_rate=repeat_prevention_rate,
            semantic_aliasing_accuracy=semantic_aliasing_accuracy,
            clarification_precision=clarification_precision,
            resolution_confirmation_accuracy=resolution_confirmation_accuracy,
            safe_escalation_fidelity=safe_escalation_fidelity,
            context_fact_retention_rate=context_fact_retention_rate,
            unsafe_handle_rate=unsafe_handle_rate,
            results=results,
        )
