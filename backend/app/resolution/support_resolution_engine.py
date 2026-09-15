"""
SupportGraph AI — Support Resolution Engine & Brand Response Generator (Phase 8 + Phase 10)

End-to-end evidence-grounded resolution pipeline:
1.  Problem Understanding: Extracts structured device, symptom, and causality features
2.  Top-K Candidate Generation: Generates ranked intent candidates
3.  Primary Problem Selection: Decouples causal trigger from core operational symptom
4.  Ambiguity Analysis: Determines operational ambiguity type and reasons
5.  Evidence Retrieval: Retrieves and operationalizes historical AppleSupport cases into match tiers
6.  Multi-Signal Safe Decision Gate: Applies empirical 6-signal clarity gating and safety vetoes
7a. Operational Evidence Validation: Validates evidence strength, agreement, and consistency
7b. Multi-Case Evidence Synthesis: Synthesizes composite evidence across 5 independent dimensions  [Phase 10]
7c. Evidence Conflict Detection: Screens composite evidence for operational contradictions          [Phase 10]
7d. Composite Evidence Verdict: Selects best of 7a vs 7b; updates auto-resolution authorization   [Phase 10]
8.  Evidence-Grounded Response Generation: Synthesizes brand-compliant reply grounded in evidence
9.  Response Grounding Verification: Verifies safety, symptom focus, and corroboration
10. Final Routing & Decision Support Packaging: Safe auto-handling or rich human escalation.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

try:
    from app.core.config import settings
    from app.core.logging import get_logger
    from app.intent.ambiguity_analyzer import (
        AmbiguityAnalysisResult,
        AmbiguityAnalyzer,
        AmbiguityType,
    )
    from app.intent.ambiguity_decision_gate import (
        AmbiguityDecisionGate,
        DecisionGateResult,
    )
    from app.intent.classifier import TopKIntentClassifier
    from app.intent.primary_problem_selector import PrimaryProblemSelector
    from app.resolution.evidence_conflict_detector import EvidenceConflictDetector
    from app.resolution.evidence_synthesizer import (
        CompositeEvidencePackage,
        MultiCaseEvidenceSynthesizer,
    )
    from app.resolution.evidence_validator import (
        CompositeEvidenceVerdict,
        EvidenceValidationResult,
        EvidenceVerdict,
        ResolutionEvidenceValidator,
    )
    from app.resolution.resolution_auditor import (
        ResolutionAuditRecord,
        ResolutionAuditor,
    )
    from app.resolution.response_generator import (
        EvidenceGroundedResponseGenerator,
        GroundedResponseCandidate,
    )
    from app.resolution.response_verifier import (
        ResponseGroundingResult,
        ResponseGroundingVerifier,
        VerificationStatus,
    )
    from app.retrieval.case_retriever import CaseRetriever
    from app.retrieval.evidence_ranker import (
        EvidenceMatchTier,
        RetrievedEvidenceCase,
    )
    from app.schemas.decision_explanation import (
        DecisionExplanation,
        EndToEndOutcome,
    )
    from app.schemas.intent_routing import (
        IntentAnalysis,
        IntentPrediction,
        RoutingDecisionType,
    )
    from app.understanding.problem_extractor import (
        CustomerProblemProfile,
        ProblemExtractor,
    )
    from app.observability.auditor import (
        DecisionTrace,
        get_trace_store,
        get_metrics_auditor,
    )
except ModuleNotFoundError:
    from backend.app.core.config import settings  # type: ignore[no-redef]
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]
    from backend.app.intent.ambiguity_analyzer import (  # type: ignore[no-redef]
        AmbiguityAnalysisResult,
        AmbiguityAnalyzer,
        AmbiguityType,
    )
    from backend.app.intent.ambiguity_decision_gate import (  # type: ignore[no-redef]
        AmbiguityDecisionGate,
        DecisionGateResult,
    )
    from backend.app.intent.classifier import TopKIntentClassifier  # type: ignore[no-redef]
    from backend.app.intent.primary_problem_selector import (  # type: ignore[no-redef]
        PrimaryProblemSelector,
    )
    from backend.app.resolution.evidence_conflict_detector import (  # type: ignore[no-redef]
        EvidenceConflictDetector,
    )
    from backend.app.resolution.evidence_synthesizer import (  # type: ignore[no-redef]
        CompositeEvidencePackage,
        MultiCaseEvidenceSynthesizer,
    )
    from backend.app.resolution.evidence_validator import (  # type: ignore[no-redef]
        CompositeEvidenceVerdict,
        EvidenceValidationResult,
        EvidenceVerdict,
        ResolutionEvidenceValidator,
    )
    from backend.app.resolution.resolution_auditor import (  # type: ignore[no-redef]
        ResolutionAuditRecord,
        ResolutionAuditor,
    )
    from backend.app.resolution.response_generator import (  # type: ignore[no-redef]
        EvidenceGroundedResponseGenerator,
        GroundedResponseCandidate,
    )
    from backend.app.resolution.response_verifier import (  # type: ignore[no-redef]
        ResponseGroundingResult,
        ResponseGroundingVerifier,
        VerificationStatus,
    )
    from backend.app.retrieval.case_retriever import CaseRetriever  # type: ignore[no-redef]
    from backend.app.retrieval.evidence_ranker import (  # type: ignore[no-redef]
        EvidenceMatchTier,
        RetrievedEvidenceCase,
    )
    from backend.app.schemas.decision_explanation import (  # type: ignore[no-redef]
        DecisionExplanation,
        EndToEndOutcome,
    )
    from backend.app.schemas.intent_routing import (  # type: ignore[no-redef]
        IntentAnalysis,
        IntentPrediction,
        RoutingDecisionType,
    )
    from backend.app.understanding.problem_extractor import (  # type: ignore[no-redef]
        CustomerProblemProfile,
        ProblemExtractor,
    )
    from backend.app.observability.auditor import (  # type: ignore[no-redef]
        DecisionTrace,
        get_trace_store,
        get_metrics_auditor,
    )

logger = get_logger(__name__)

# Standard AppleSupport Brand Troubleshooting Strategies for Decision Support
BRAND_RESOLUTION_STRATEGIES = {
    "battery_power_issue": (
        "Check Battery Health in Settings > Battery. If Maximum Capacity is below 80%, recommend service. "
        "Review battery usage by app, disable Background App Refresh for high-drain apps, and verify charging accessories."
    ),
    "software_update_problem": (
        "Ensure device is connected to stable Wi-Fi and power. If device is unresponsive or experiencing system lag, "
        "perform a force restart. If update verification fails, use Finder or iTunes on a computer to complete restore."
    ),
    "hardware_audio_connection_issue": (
        "Inspect speaker and microphone grilles for dirt/debris. Adjust volume sliders under Settings > Sounds. "
        "For Bluetooth or AirPods issues, forget device in Settings > Bluetooth and pair again."
    ),
    "display_touch_issue": (
        "Clean screen surface and remove screen protectors or cases. Force restart the device. "
        "If touch remains unresponsive or physical glass is cracked, schedule a Genius Bar appointment."
    ),
    "keyboard_typing_issue": (
        "Navigate to Settings > General > Keyboard > Text Replacement to inspect autocorrect shortcuts. "
        "Reset Keyboard Dictionary in Settings > General > Reset > Reset Keyboard Dictionary."
    ),
    "account_access_issue": (
        "Direct customer to https://iforgot.apple.com to securely reset Apple ID credentials. "
        "Advise customer to check trusted devices for two-factor authentication verification codes."
    ),
    "billing_purchase_issue": (
        "Direct customer to https://reportaproblem.apple.com to inspect recent charges and request refunds. "
        "Guide customer to Settings > [Name] > Subscriptions to view or cancel active renewals."
    ),
    "mac_software_issue": (
        "Boot Mac into Safe Mode to clear kernel caches. Run First Aid in Disk Utility. "
        "Reset NVRAM/PRAM if display/audio hardware fails to respond on macOS."
    ),
    "general_device_support": (
        "Perform a standard device restart. Verify Wi-Fi and cellular network connectivity. "
        "Ask customer for specific device model and operating system version for targeted troubleshooting."
    ),
}


class HumanEscalationPackage(BaseModel):
    """Rich decision support bundle presented to a human reviewer for escalated cases."""
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    customer_message: str
    problem_profile: CustomerProblemProfile
    ambiguity_type: AmbiguityType
    ambiguity_reason: str
    top_candidates: list[IntentPrediction]
    related_evidence: list[RetrievedEvidenceCase]
    evidence_verdict: EvidenceVerdict = EvidenceVerdict.WEAK_EVIDENCE
    candidate_resolution_approaches: dict[str, str] = Field(default_factory=dict)
    decision_checklist: dict[str, bool] = Field(default_factory=dict, description="Multi-signal validation checks")
    why_not_auto_handled: list[str] = Field(default_factory=list, description="Explicit reasons AI did not auto-handle")
    grounding_result: Optional[ResponseGroundingResult] = Field(default=None, description="Grounding verification result if attempted")
    recommended_human_action: str = Field(default="", description="Suggested adjudication guideline for reviewer")
    # Phase 10: Composite evidence breakdown for human reviewers
    composite_evidence_verdict: Optional[str] = Field(default=None, description="Phase 10 composite evidence verdict")
    composite_evidence_summary: Optional[str] = Field(default=None, description="Plain-language synthesis summary for reviewer")
    covered_evidence_dimensions: list[str] = Field(default_factory=list, description="Evidence dimensions with coverage")
    missing_evidence_dimensions: list[str] = Field(default_factory=list, description="Evidence dimensions with no coverage")
    conflicting_evidence_pairs: list[str] = Field(default_factory=list, description="Conflicting intent pairs (if any)")


class SupportResolutionResult(BaseModel):
    """Complete output of the Phase 8/10 evidence-grounded support resolution engine."""
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    routing_decision: RoutingDecisionType = Field(..., description="AUTO_HANDLE or ESCALATE_TO_HUMAN")
    primary_intent: str = Field(..., description="Selected primary operational intent")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    problem_summary: str = Field(..., description="Structured summary of understood problem")
    ambiguity_analysis: AmbiguityAnalysisResult = Field(..., description="Ambiguity detection output")
    gate_result: Optional[DecisionGateResult] = Field(default=None, description="Safe decision gate evaluation")
    evidence_validation: Optional[EvidenceValidationResult] = Field(default=None, description="Operational evidence validation result")
    composite_evidence: Optional[CompositeEvidencePackage] = Field(default=None, description="Phase 10 multi-case composite evidence synthesis result")
    evidence_cases: list[RetrievedEvidenceCase] = Field(default_factory=list, description="Retrieved historical evidence")
    resolution_strategy: str = Field(default="", description="Recommended brand troubleshooting workflow")
    grounded_response: Optional[str] = Field(default=None, description="Generated brand support reply (if auto-handled)")
    response_grounding: Optional[ResponseGroundingResult] = Field(default=None, description="Verification of generated response")
    escalation_package: Optional[HumanEscalationPackage] = Field(default=None, description="Human review packet (if escalated)")
    explanation: str = Field(default="", description="Why this decision and resolution was produced")
    outcome: Optional[EndToEndOutcome] = Field(default=None, description="Phase 12 mutually exclusive terminal outcome")
    decision_explanation: Optional[DecisionExplanation] = Field(default=None, description="Structured 'Why Did AI Decide This?' explanation")

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class SupportResolutionEngine:
    """
    Unified evidence-grounded resolution engine with Operational Evidence Validation,
    Response Generation, and Response Grounding Verification.
    """

    def __init__(
        self,
        extractor: Optional[ProblemExtractor] = None,
        classifier: Optional[TopKIntentClassifier] = None,
        problem_selector: Optional[PrimaryProblemSelector] = None,
        ambiguity_analyzer: Optional[AmbiguityAnalyzer] = None,
        retriever: Optional[CaseRetriever] = None,
        decision_gate: Optional[AmbiguityDecisionGate] = None,
        evidence_validator: Optional[ResolutionEvidenceValidator] = None,
        evidence_synthesizer: Optional[MultiCaseEvidenceSynthesizer] = None,
        conflict_detector: Optional[EvidenceConflictDetector] = None,
        response_generator: Optional[EvidenceGroundedResponseGenerator] = None,
        response_verifier: Optional[ResponseGroundingVerifier] = None,
        auditor: Optional[ResolutionAuditor] = None,
    ) -> None:
        self.extractor = extractor or ProblemExtractor()
        self.classifier = classifier or TopKIntentClassifier()
        self.problem_selector = problem_selector or PrimaryProblemSelector()
        self.ambiguity_analyzer = ambiguity_analyzer or AmbiguityAnalyzer()
        self.retriever = retriever or CaseRetriever()
        self.decision_gate = decision_gate or AmbiguityDecisionGate()
        self.evidence_validator = evidence_validator or ResolutionEvidenceValidator()
        self.evidence_synthesizer = evidence_synthesizer or MultiCaseEvidenceSynthesizer()
        self.conflict_detector = conflict_detector or EvidenceConflictDetector()
        self.response_generator = response_generator or EvidenceGroundedResponseGenerator()
        self.response_verifier = response_verifier or ResponseGroundingVerifier()
        self.auditor = auditor or ResolutionAuditor()

    def process_message(
        self,
        customer_message: str,
        normalized_message: str = "",
        top_k_evidence: int = 3,
        case_id: Optional[str] = None,
        log_audit: bool = True,
        is_demo: bool = False,
    ) -> SupportResolutionResult:
        """
        Process incoming customer inquiry through the full Phase 8 evidence-grounded resolution pipeline.
        Phase 14: Instruments each step for structured decision tracing and latency observability.
        The trace layer is strictly read-only and does NOT alter routing decisions.
        """
        msg_clean = customer_message.strip()

        # Phase 14: Initialise structured decision trace
        _trace_case_id = case_id or f"req_{abs(hash(msg_clean)) % 1000000:06d}"
        _trace = DecisionTrace(case_id=_trace_case_id, is_demo=is_demo)
        _trace.record_customer_message(msg_clean)
        _metrics = get_metrics_auditor()

        # Step 1: Problem Understanding
        _t0 = time.perf_counter()
        profile: CustomerProblemProfile = self.extractor.extract(msg_clean)
        _step1_ms = (time.perf_counter() - _t0) * 1000
        _trace.record_problem_understanding(
            device=profile.device,
            primary_symptom=profile.primary_symptom,
            problem_family=str(profile.primary_problem_family.value if hasattr(profile.primary_problem_family, 'value') else profile.primary_problem_family),
            possible_cause=profile.possible_cause,
            latency_ms=_step1_ms,
        )

        # Step 2: Top-K Intent Candidate Generation
        _t0 = time.perf_counter()
        try:
            analysis: IntentAnalysis = self.classifier.classify(
                customer_message=msg_clean,
                normalized_message=normalized_message,
            )
            _step2_ms = (time.perf_counter() - _t0) * 1000
            _metrics.log_llm_request(success=True, latency_ms=_step2_ms)
        except Exception as _exc:
            _step2_ms = (time.perf_counter() - _t0) * 1000
            _metrics.log_llm_request(success=False, latency_ms=_step2_ms, reason=str(_exc))
            _metrics.log_fallback(reason=f"classifier: {_exc}", latency_ms=_step2_ms)
            raise
        _trace.record_intent_candidates(
            candidates=[{"intent": p.intent, "confidence": round(p.confidence, 3)} for p in analysis.top_predictions[:5]],
            latency_ms=_step2_ms,
        )

        # Step 3: Primary Problem Selection
        _t0 = time.perf_counter()
        primary_intent, cause_intent, sel_reason = self.problem_selector.select_primary_problem(
            message=msg_clean,
            profile=profile,
            analysis=analysis,
        )
        _trace.record_primary_problem(
            primary_intent=primary_intent,
            cause_intent=cause_intent,
            reason=sel_reason,
            latency_ms=(time.perf_counter() - _t0) * 1000,
        )

        # Step 4: Ambiguity Analysis
        _t0 = time.perf_counter()
        ambiguity_res: AmbiguityAnalysisResult = self.ambiguity_analyzer.analyze(
            message=msg_clean,
            profile=profile,
            analysis=analysis,
            primary_intent=primary_intent,
            contextual_cause_intent=cause_intent,
        )
        _trace.record_ambiguity(
            ambiguity_type=ambiguity_res.ambiguity_type.value,
            ambiguity_reason=ambiguity_res.ambiguity_reason,
            latency_ms=(time.perf_counter() - _t0) * 1000,
        )

        # Step 5: Evidence Retrieval & Operational Match
        _t0 = time.perf_counter()
        try:
            evidence_cases = self.retriever.retrieve(
                query_text=msg_clean,
                query_intent=primary_intent,
                query_profile=profile,
                top_k=top_k_evidence,
            )
        except Exception as exc:
            logger.warning("Evidence retrieval failed, safely degrading to empty evidence: %s", exc)
            evidence_cases = []
        _trace.record_evidence_retrieval(
            case_count=len(evidence_cases),
            tiers=[c.match_tier.value for c in evidence_cases],
            latency_ms=(time.perf_counter() - _t0) * 1000,
        )

        # Step 6: Multi-Signal Safe Decision Gate (Phase 7.1)
        _t0 = time.perf_counter()
        gate_result: DecisionGateResult = self.decision_gate.evaluate_gate(
            message=msg_clean,
            profile=profile,
            analysis=analysis,
            primary_intent=primary_intent,
            contextual_cause_intent=cause_intent,
            evidence_cases=evidence_cases,
        )
        _trace.record_safety_gate(
            gate_decision=gate_result.decision.value,
            safety_signals=gate_result.checklist,
            latency_ms=(time.perf_counter() - _t0) * 1000,
        )

        # Step 7a: Operational Evidence Validation (Phase 8 — unchanged)
        _t0 = time.perf_counter()
        evidence_val: EvidenceValidationResult = self.evidence_validator.evaluate_evidence(
            profile=profile,
            primary_intent=primary_intent,
            evidence_cases=evidence_cases,
        )
        _trace.record_evidence_validation(
            verdict=evidence_val.evidence_verdict.value,
            direct_matches=evidence_val.direct_problem_matches,
            symptom_agreement=evidence_val.symptom_agreement,
            auto_resolution_allowed=evidence_val.auto_resolution_allowed,
            latency_ms=(time.perf_counter() - _t0) * 1000,
        )

        # Step 7b–7d: Multi-Case Evidence Synthesis (Phase 10)
        # Only synthesize when single-case validation is not already STRONG.
        # STRONG_EVIDENCE is passed through directly as DIRECT_STRONG_EVIDENCE.
        composite_evidence: CompositeEvidencePackage = self.evidence_synthesizer.synthesize(
            primary_intent=primary_intent,
            profile=profile,
            evidence_cases=evidence_cases,
            single_case_validation=evidence_val,
            dimension_matches=evidence_val.dimension_matches,
        )

        # Step 7d: Use composite verdict to evaluate auto_resolution_allowed.
        # Safety contract: Any detected conflict (composite or single-case) immediately blocks auto-resolution.
        if (
            composite_evidence.composite_verdict == CompositeEvidenceVerdict.CONFLICTING_COMPOSITE_EVIDENCE
            or (composite_evidence.conflict_result and composite_evidence.conflict_result.has_conflict)
            or evidence_val.evidence_verdict == EvidenceVerdict.CONFLICTING_EVIDENCE
        ):
            composite_auto_allowed = False
        elif (
            not evidence_val.auto_resolution_allowed
            and composite_evidence.composite_verdict == CompositeEvidenceVerdict.COMPOSITE_STRONG_EVIDENCE
        ):
            composite_auto_allowed = True
        else:
            composite_auto_allowed = evidence_val.auto_resolution_allowed

        # Step 7c: Conflict check trace
        _conflict_result = composite_evidence.conflict_result
        _trace.record_conflict_check(
            has_conflict=bool(_conflict_result and _conflict_result.has_conflict),
            conflicting_pairs=(_conflict_result.conflicting_pairs if _conflict_result else []),
            latency_ms=0.0,
        )

        # Step 8: Resolution Strategy & Response Synthesis
        resolution_strategy = BRAND_RESOLUTION_STRATEGIES.get(
            primary_intent,
            BRAND_RESOLUTION_STRATEGIES["general_device_support"],
        )

        _t0 = time.perf_counter()
        _gen_fallback = False
        try:
            candidate_response = self.response_generator.generate_response(
                profile=profile,
                primary_intent=primary_intent,
                evidence_cases=evidence_cases,
                validation_result=evidence_val,
            )
        except Exception as exc:
            logger.warning("Response generation failed, safely degrading to fallback: %s", exc)
            _gen_fallback = True
            _metrics.log_fallback(reason=f"response_generator: {exc}", latency_ms=(time.perf_counter() - _t0) * 1000)
            candidate_response = GroundedResponseCandidate(
                response_text="Thank you for contacting Apple Support. Please send us a direct message at https://apple.co/dm with your device model so our technical support team can assist you further.",
                grounded_intent=primary_intent,
                contains_dm_link=True,
                contains_diagnostic_step=False,
            )
        _trace.record_response_generation(
            latency_ms=(time.perf_counter() - _t0) * 1000,
            fallback_used=_gen_fallback,
        )

        # Step 9: Response Grounding Verification (Phase 8)
        _t0 = time.perf_counter()
        try:
            grounding_result = self.response_verifier.verify_response(
                customer_message=msg_clean,
                profile=profile,
                primary_intent=primary_intent,
                candidate_response=candidate_response,
                evidence_cases=evidence_cases,
                validation_result=evidence_val,
            )
        except Exception as exc:
            logger.warning("Response verification failed, safely blocking auto-handling: %s", exc)
            grounding_result = ResponseGroundingResult(
                verification_status=VerificationStatus.FAIL,
                support_score=0.0,
                explanation=f"Verification exception: {exc}",
            )
        _trace.record_grounding_verification(
            status=grounding_result.verification_status.value,
            score=grounding_result.support_score,
            explanation=grounding_result.explanation,
            latency_ms=(time.perf_counter() - _t0) * 1000,
        )

        problem_summary = (
            f"Understood problem for {profile.device or 'device'}: {profile.primary_symptom}."
        )
        if profile.possible_cause:
            problem_summary += f" Contextual cause: {profile.possible_cause}."

        # Step 10: Multi-Stage Safety Decision Rule
        # AUTO_HANDLE requires: Gate PASS + Evidence Authorized + Grounding Verification PASS
        # composite_auto_allowed = True if either single-case STRONG or composite STRONG
        can_auto_handle = (
            gate_result.decision == RoutingDecisionType.AUTO_HANDLE
            and composite_auto_allowed
            and grounding_result.verification_status == VerificationStatus.PASS
        )

        escalation_reasons: list[str] = list(gate_result.escalation_reasons)
        if not composite_auto_allowed:
            escalation_reasons.extend(evidence_val.reasons)
            if (
                composite_evidence.composite_verdict
                not in (CompositeEvidenceVerdict.DIRECT_STRONG_EVIDENCE, CompositeEvidenceVerdict.COMPOSITE_STRONG_EVIDENCE)
            ):
                escalation_reasons.append(
                    f"Composite evidence verdict: {composite_evidence.composite_verdict.value}. "
                    f"{composite_evidence.synthesis_rationale}"
                )
        if grounding_result.verification_status == VerificationStatus.FAIL:
            escalation_reasons.append(f"Grounding verification failed: {grounding_result.explanation}")

        # Construct Structured Decision Explanation & EndToEndOutcome (Phase 12)
        if can_auto_handle:
            outcome = EndToEndOutcome.SAFE_AUTO_HANDLED
            decision_str = "AUTO_HANDLE"
            summary_str = "Auto-handling authorized: Problem clearly understood, supported by corroborated historical evidence, and response verified grounded."
            pos_factors = [
                f"✓ Problem identified: {profile.primary_symptom} ({profile.primary_problem_family.value if hasattr(profile.primary_problem_family, 'value') else profile.primary_problem_family})",
                f"✓ Decision gate passed ({gate_result.decision.value})",
                f"✓ Corroborated by {evidence_val.direct_problem_matches} direct historical case(s)",
                f"✓ Response grounding verification passed (Score: {grounding_result.support_score:.2f})",
                "✓ No hazardous or ungrounded actions detected",
            ]
            neg_factors = []
            rec_human_actions = []
        else:
            decision_str = "ESCALATE_TO_HUMAN"
            if grounding_result.verification_status == VerificationStatus.FAIL:
                outcome = EndToEndOutcome.ESCALATED_VERIFICATION_FAILURE
                summary_str = "Escalated to human: Synthesized response failed safety and grounding verification."
            elif evidence_val.evidence_verdict == EvidenceVerdict.CONFLICTING_EVIDENCE or (
                composite_evidence and composite_evidence.conflict_result and composite_evidence.conflict_result.has_conflict
            ):
                outcome = EndToEndOutcome.ESCALATED_CONFLICT
                summary_str = "Escalated to human: Conflicting operational symptoms or candidate solutions detected."
            elif ambiguity_res.ambiguity_type in (
                AmbiguityType.GENUINE_AMBIGUITY,
                AmbiguityType.UNCLEAR_INSUFFICIENT,
                AmbiguityType.MULTI_SYMPTOM,
            ):
                outcome = EndToEndOutcome.ESCALATED_AMBIGUOUS
                summary_str = "Escalated to human: Severe symptom underspecification or operational ambiguity detected."
            elif evidence_val.evidence_verdict == EvidenceVerdict.WEAK_EVIDENCE or not composite_auto_allowed:
                outcome = EndToEndOutcome.ESCALATED_EVIDENCE_LIMITED
                summary_str = "Escalated to human: Insufficient historical support evidence available in corpus to authorize autonomous handling."
            else:
                outcome = EndToEndOutcome.ESCALATED_HUMAN_REQUIRED
                summary_str = f"Escalated to human: Multi-signal safety gate required human adjudication: {gate_result.explanation}"

            pos_factors = []
            if gate_result.decision == RoutingDecisionType.AUTO_HANDLE:
                pos_factors.append("✓ Decision gate passed initial clarity checks")
            if profile.primary_symptom != "unspecified_device_issue":
                pos_factors.append(f"✓ Operational symptom extracted: {profile.primary_symptom}")
            if evidence_val.direct_problem_matches > 0:
                pos_factors.append(f"✓ {evidence_val.direct_problem_matches} direct historical case(s) found")

            neg_factors = list(escalation_reasons)
            rec_human_actions = [
                f"Review candidate hypotheses ({', '.join(pred.intent for pred in analysis.top_predictions[:2])}) and select appropriate troubleshooting path."
            ]

        checklist = {
            "problem_understood": profile.primary_symptom != "unspecified_device_issue",
            "gate_passed": (gate_result.decision == RoutingDecisionType.AUTO_HANDLE),
            "evidence_authorized": bool(composite_auto_allowed),
            "response_grounded": (grounding_result.verification_status == VerificationStatus.PASS),
            "no_unsafe_action": not bool(grounding_result.unsupported_claims),
        }

        decision_explanation = DecisionExplanation(
            decision=decision_str,
            outcome=outcome,
            summary=summary_str,
            positive_factors=pos_factors,
            negative_factors=neg_factors,
            recommended_human_actions=rec_human_actions,
            checklist=checklist,
        )

        # Construct Result
        if can_auto_handle:
            composite_label = composite_evidence.composite_verdict.value
            explanation = (
                f"Auto-handled: {gate_result.explanation} Evidence verdict: {evidence_val.evidence_verdict.value} "
                f"[Composite: {composite_label}] "
                f"({evidence_val.direct_problem_matches} direct matches, {evidence_val.symptom_agreement:.0%} symptom agreement). "
                f"Response verification passed (score: {grounding_result.support_score:.2f})."
            )

            result = SupportResolutionResult(
                routing_decision=RoutingDecisionType.AUTO_HANDLE,
                primary_intent=primary_intent,
                confidence=analysis.top_confidence,
                problem_summary=problem_summary,
                ambiguity_analysis=ambiguity_res,
                gate_result=gate_result,
                evidence_validation=evidence_val,
                composite_evidence=composite_evidence,
                evidence_cases=evidence_cases,
                resolution_strategy=resolution_strategy,
                grounded_response=candidate_response.response_text,
                response_grounding=grounding_result,
                escalation_package=None,
                explanation=explanation,
                outcome=outcome,
                decision_explanation=decision_explanation,
            )
        else:
            # Build Candidate Approaches
            approaches = {}
            for pred in analysis.top_predictions[:3]:
                approaches[pred.intent] = BRAND_RESOLUTION_STRATEGIES.get(
                    pred.intent,
                    BRAND_RESOLUTION_STRATEGIES["general_device_support"],
                )

            # Recommend human action guideline
            if evidence_val.evidence_verdict == EvidenceVerdict.CONFLICTING_EVIDENCE:
                rec_action = f"Clarify whether customer is experiencing '{primary_intent}' or competing historical intent before providing steps."
            elif ambiguity_res.ambiguity_type == AmbiguityType.CAUSE_VS_SYMPTOM:
                rec_action = f"Confirm if customer prioritizes '{primary_intent}' symptom over '{cause_intent}' update trigger."
            elif ambiguity_res.ambiguity_type == AmbiguityType.MULTI_SYMPTOM:
                rec_action = "Triage multiple reported symptoms individually starting with primary hardware issue."
            else:
                rec_action = f"Review candidate hypotheses ({', '.join(pred.intent for pred in analysis.top_predictions[:2])}) and select appropriate troubleshooting path."

            escalation_pkg = HumanEscalationPackage(
                customer_message=msg_clean,
                problem_profile=profile,
                ambiguity_type=ambiguity_res.ambiguity_type,
                ambiguity_reason=ambiguity_res.ambiguity_reason,
                top_candidates=analysis.top_predictions[:3],
                related_evidence=evidence_cases,
                evidence_verdict=evidence_val.evidence_verdict,
                candidate_resolution_approaches=approaches,
                decision_checklist={
                    **gate_result.checklist,
                    "evidence_validated": evidence_val.auto_resolution_allowed,
                    "composite_evidence_authorized": composite_auto_allowed,
                    "response_grounded": (grounding_result.verification_status == VerificationStatus.PASS),
                },
                why_not_auto_handled=escalation_reasons,
                grounding_result=grounding_result,
                recommended_human_action=rec_action,
                # Phase 10: composite evidence breakdown for reviewer
                composite_evidence_verdict=composite_evidence.composite_verdict.value,
                composite_evidence_summary=composite_evidence.synthesis_rationale,
                covered_evidence_dimensions=composite_evidence.covered_dimensions,
                missing_evidence_dimensions=composite_evidence.missing_dimensions,
                conflicting_evidence_pairs=(
                    composite_evidence.conflict_result.conflicting_pairs
                    if composite_evidence.conflict_result else []
                ),
            )

            explanation = (
                f"Escalated to human review: Evidence verdict: {evidence_val.evidence_verdict.value} "
                f"[Composite: {composite_evidence.composite_verdict.value}]. "
                f"Issues: {'; '.join(escalation_reasons[:2])}."
            )

            result = SupportResolutionResult(
                routing_decision=RoutingDecisionType.ESCALATE_TO_HUMAN,
                primary_intent=primary_intent,
                confidence=analysis.top_confidence,
                problem_summary=problem_summary,
                ambiguity_analysis=ambiguity_res,
                gate_result=gate_result,
                evidence_validation=evidence_val,
                composite_evidence=composite_evidence,
                evidence_cases=evidence_cases,
                resolution_strategy=resolution_strategy,
                grounded_response=None,
                response_grounding=grounding_result,
                escalation_package=escalation_pkg,
                explanation=explanation,
                outcome=outcome,
                decision_explanation=decision_explanation,
            )

        # Log decision to isolated runtime audit storage
        if log_audit:
            audit_id = case_id or f"req_{abs(hash(msg_clean)) % 1000000:06d}"
            cand_str = ", ".join(f"{p.intent} ({p.confidence:.2f})" for p in analysis.top_predictions[:3])
            case_ids_str = ", ".join(c.case_id for c in evidence_cases)
            tiers_str = ", ".join(c.match_tier.value for c in evidence_cases)

            audit_rec = ResolutionAuditRecord(
                case_id=audit_id,
                customer_message=msg_clean,
                device=profile.device,
                primary_symptom=profile.primary_symptom,
                possible_cause=profile.possible_cause,
                primary_intent=primary_intent,
                top_confidence=analysis.top_confidence,
                candidate_intents=cand_str,
                evidence_verdict=evidence_val.evidence_verdict.value,
                direct_matches_count=evidence_val.direct_problem_matches,
                routing_decision=result.routing_decision.value,
                retrieved_case_ids=case_ids_str,
                match_tiers=tiers_str,
                response_grounding_status=grounding_result.verification_status.value,
                grounding_score=grounding_result.support_score,
                human_escalation_reason="; ".join(escalation_reasons) if not can_auto_handle else "",
            )
            self.auditor.log_decision(audit_rec)

        # Phase 14: Save structured decision trace (read-only, non-blocking)
        try:
            _trace.record_final_decision(
                routing_decision=result.routing_decision.value,
                outcome=result.outcome.value if result.outcome else "",
                escalation_reason="; ".join(escalation_reasons) if not can_auto_handle else "",
            )
            get_trace_store().save(_trace)
        except Exception as _trace_exc:
            logger.debug("Non-critical: failed to save decision trace: %s", _trace_exc)

        return result
