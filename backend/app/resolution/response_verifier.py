"""
SupportGraph AI — Response Grounding Verifier (Phase 8)

Validates synthesized support responses prior to automated dispatch:
- Checks that the response addresses the customer's actual primary symptom
- Verifies that post-update symptoms are not erroneously addressed with generic OS restore loops
- Detects unauthorized, hazardous, or unsupported troubleshooting claims
- Confirms consistency with retrieved historical AppleSupport evidence
- Emits structured PASS / FAIL verification verdict; triggers human escalation on failure.
"""
from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

try:
    from app.core.logging import get_logger
    from app.resolution.evidence_validator import (
        EvidenceValidationResult,
        EvidenceVerdict,
    )
    from app.resolution.response_generator import GroundedResponseCandidate
    from app.retrieval.evidence_ranker import (
        EvidenceMatchTier,
        RetrievedEvidenceCase,
    )
    from app.understanding.problem_extractor import CustomerProblemProfile
except ModuleNotFoundError:
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]
    from backend.app.resolution.evidence_validator import (  # type: ignore[no-redef]
        EvidenceValidationResult,
        EvidenceVerdict,
    )
    from backend.app.resolution.response_generator import (  # type: ignore[no-redef]
        GroundedResponseCandidate,
    )
    from backend.app.retrieval.evidence_ranker import (  # type: ignore[no-redef]
        EvidenceMatchTier,
        RetrievedEvidenceCase,
    )
    from backend.app.understanding.problem_extractor import (  # type: ignore[no-redef]
        CustomerProblemProfile,
    )

logger = get_logger(__name__)


class VerificationStatus(str, Enum):
    """Verification verdict for customer response grounding."""
    PASS = "PASS"
    FAIL = "FAIL"


class ResponseGroundingResult(BaseModel):
    """Detailed output of response grounding and safety verification."""
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    verification_status: VerificationStatus = Field(..., description="PASS if response is safe and grounded, else FAIL")
    grounded: bool = Field(..., description="True if response is fully supported by evidence and matches symptom")
    support_score: float = Field(..., ge=0.0, le=1.0, description="Quantitative grounding score [0.0, 1.0]")
    addresses_primary_symptom: bool = Field(default=True, description="True if response directly addresses primary symptom")
    avoids_cause_overfocus: bool = Field(default=True, description="True if response avoids misleading focus on causal context")
    unsupported_claims: list[str] = Field(default_factory=list, description="List of detected unsupported or hazardous claims")
    contradictions_detected: list[str] = Field(default_factory=list, description="List of detected contradictions with evidence")
    evidence_references: list[str] = Field(default_factory=list, description="IDs of historical cases corroborating response")
    explanation: str = Field(default="", description="Human-readable verification rationale")

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


# Patterns indicating hazardous or unsupported troubleshooting recommendations
UNSUPPORTED_CLAIM_PATTERNS = [
    (r"\b(jailbreak|jailbreaking|cydia|sideload)\b", "Mentions unauthorized jailbreaking or sideloading"),
    (r"\b(take\s+apart|disassemble|unscrew|open\s+up\s+the\s+phone|pry\s+open)\b", "Recommends unauthorized physical hardware disassembly"),
    (r"\b(replace\s+(logic\s*board|motherboard|screen\s+yourself))\b", "Suggests uncertified DIY component replacement"),
    (r"\b(guaranteed\s+(refund|free\s+replacement|cash\s+back))\b", "Promises guaranteed refund or replacement without entitlement check"),
    (r"\b(downgrade\s+ios|flash\s+firmware|custom\s+rom)\b", "Recommends unsupported firmware downgrade"),
    (r"\b(wipe\s+(everything|entire\s+disk)\s+without\s+backup)\b", "Recommends destructive disk wipe without backup safety warning"),
    (r"\b(third-party\s+cleaner|cleanmymac|registry\s+cleaner)\b", "Recommends untrusted third-party maintenance utilities"),
]


class ResponseGroundingVerifier:
    """
    Verifies that generated support replies are strictly grounded in operational evidence.
    """

    def __init__(self, min_support_score: float = 0.60) -> None:
        self.min_support_score = min_support_score

    def verify_response(
        self,
        customer_message: str,
        profile: CustomerProblemProfile,
        primary_intent: str,
        candidate_response: GroundedResponseCandidate,
        evidence_cases: list[RetrievedEvidenceCase],
        validation_result: Optional[EvidenceValidationResult] = None,
    ) -> ResponseGroundingResult:
        """
        Perform rigorous multi-point verification on the candidate response.
        """
        resp_text = candidate_response.response_text.lower()
        unsupported_claims: list[str] = []
        contradictions: list[str] = []

        # Check 1: Detect unsupported/hazardous claims
        for pattern, desc in UNSUPPORTED_CLAIM_PATTERNS:
            if re.search(pattern, resp_text):
                unsupported_claims.append(desc)

        # Check 2: Verify primary symptom alignment
        addresses_primary = True
        q_sym = profile.primary_symptom.lower()

        if "battery" in q_sym:
            if not any(w in resp_text for w in ["battery", "power", "charge", "health"]):
                addresses_primary = False
        elif "audio" in q_sym:
            if not any(w in resp_text for w in ["sound", "audio", "speaker", "volume", "airpod", "headphone", "bluetooth"]):
                addresses_primary = False
        elif "display" in q_sym:
            if not any(w in resp_text for w in ["display", "screen", "touch", "cracked"]):
                addresses_primary = False
        elif "keyboard" in q_sym:
            if not any(w in resp_text for w in ["keyboard", "typing", "autocorrect", "text replacement"]):
                addresses_primary = False
        elif "account" in q_sym:
            if not any(w in resp_text for w in ["account", "apple id", "password", "iforgot", "sign in", "security"]):
                addresses_primary = False
        elif "billing" in q_sym:
            if not any(w in resp_text for w in ["billing", "charge", "refund", "purchase", "subscription", "reportaproblem"]):
                addresses_primary = False

        if not addresses_primary:
            contradictions.append(
                f"Response fails to address primary symptom '{profile.primary_symptom}' for intent '{primary_intent}'."
            )

        # Check 3: Check cause overfocus
        # If user mentions update but symptom is battery, response should NOT just give update restore steps
        avoids_cause_overfocus = True
        if profile.update_related and primary_intent != "software_update_problem":
            if "finder or itunes" in resp_text and "battery" in q_sym and "battery" not in resp_text:
                avoids_cause_overfocus = False
                contradictions.append("Response over-indexes on update trigger while ignoring core battery symptom.")

        # Check 4: Evidence Agreement & Corroboration
        has_corroborating_evidence = False
        evidence_refs: list[str] = []
        if evidence_cases:
            for case in evidence_cases:
                if case.match_tier in (EvidenceMatchTier.DIRECT_PROBLEM_MATCH, EvidenceMatchTier.RELATED_SYMPTOM):
                    evidence_refs.append(case.case_id)
                    has_corroborating_evidence = True

        if not has_corroborating_evidence and validation_result:
            if validation_result.evidence_verdict in (EvidenceVerdict.WEAK_EVIDENCE, EvidenceVerdict.CONFLICTING_EVIDENCE):
                contradictions.append("No historical support cases corroborate this troubleshooting response.")

        # Check 5: DM / Escalation link presence
        if not candidate_response.contains_dm_link or "https://apple.co/dm" not in resp_text:
            contradictions.append("Missing official AppleSupport DM continuation link.")

        # Compute support score
        base_score = 1.0
        if unsupported_claims:
            base_score -= 0.60 * len(unsupported_claims)
        if not addresses_primary:
            base_score -= 0.40
        if not avoids_cause_overfocus:
            base_score -= 0.30
        if not has_corroborating_evidence:
            base_score -= 0.25
        if not candidate_response.contains_diagnostic_step:
            base_score -= 0.20

        support_score = round(max(0.0, min(1.0, base_score)), 4)

        # Final Verification Decision
        is_pass = (
            len(unsupported_claims) == 0
            and addresses_primary
            and avoids_cause_overfocus
            and support_score >= self.min_support_score
        )

        status = VerificationStatus.PASS if is_pass else VerificationStatus.FAIL
        grounded = is_pass and len(contradictions) == 0

        if is_pass:
            explanation = (
                f"Response verification PASSED (score: {support_score:.2f}). "
                f"Addresses '{profile.primary_symptom}' with brand-compliant steps corroborated by {len(evidence_refs)} case(s)."
            )
        else:
            reasons_list = unsupported_claims + contradictions
            explanation = (
                f"Response verification FAILED (score: {support_score:.2f}). "
                f"Issues detected: {'; '.join(reasons_list)}."
            )

        return ResponseGroundingResult(
            verification_status=status,
            grounded=grounded,
            support_score=support_score,
            addresses_primary_symptom=addresses_primary,
            avoids_cause_overfocus=avoids_cause_overfocus,
            unsupported_claims=unsupported_claims,
            contradictions_detected=contradictions,
            evidence_references=evidence_refs,
            explanation=explanation,
        )
