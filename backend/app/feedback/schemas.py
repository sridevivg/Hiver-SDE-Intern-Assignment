"""
SupportGraph AI — Human-in-the-Loop Feedback & Evidence Data Models (Phase 13).

Defines:
- ResolutionLifecycleState: Strict state machine for human resolution processing
- HumanResolution: Structured capture of human specialist resolutions
- ReviewDecision & ReviewChecklist: Deterministic 10-check validation models
- EvidenceQualityEvaluation: 8-dimensional evidence quality scoring model
- ApprovedEvidenceItem: Versioned, promoted trusted operational evidence
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field
import uuid


class ResolutionLifecycleState(str, Enum):
    """
    Lifecycle state for human resolutions.
    Enforces the principle: Human resolutions enter as CANDIDATE_EVIDENCE,
    and must pass validation, offline evaluation, and explicit reviewer approval
    before PROMOTED state into trusted retrieval.
    """
    CAPTURED = "CAPTURED"
    UNDER_REVIEW = "UNDER_REVIEW"
    VALIDATED = "VALIDATED"
    REJECTED = "REJECTED"
    CANDIDATE_EVIDENCE = "CANDIDATE_EVIDENCE"
    OFFLINE_EVALUATION = "OFFLINE_EVALUATION"
    APPROVED = "APPROVED"
    PROMOTED = "PROMOTED"
    PROMOTION_BLOCKED = "PROMOTION_BLOCKED"


class ReviewCheckStatus(str, Enum):
    """Outcome of an individual deterministic validation check."""
    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"


class EvidenceQualityVerdict(str, Enum):
    """Categorical verdict for evidence quality."""
    HIGH_QUALITY = "HIGH_QUALITY"
    ACCEPTABLE_QUALITY = "ACCEPTABLE_QUALITY"
    REJECTED_QUALITY = "REJECTED_QUALITY"


class ReviewChecklist(BaseModel):
    """Results of the 10 deterministic review gate checks."""
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    completeness: ReviewCheckStatus = ReviewCheckStatus.PASS
    problem_family_consistency: ReviewCheckStatus = ReviewCheckStatus.PASS
    fact_consistency: ReviewCheckStatus = ReviewCheckStatus.PASS
    action_result_consistency: ReviewCheckStatus = ReviewCheckStatus.PASS
    provenance: ReviewCheckStatus = ReviewCheckStatus.PASS
    contradiction: ReviewCheckStatus = ReviewCheckStatus.PASS
    safety: ReviewCheckStatus = ReviewCheckStatus.PASS
    plausibility: ReviewCheckStatus = ReviewCheckStatus.PASS
    unsupported_claims: ReviewCheckStatus = ReviewCheckStatus.PASS
    golden_isolation: ReviewCheckStatus = ReviewCheckStatus.PASS

    def to_dict(self) -> Dict[str, str]:
        return {k: v.value for k, v in self.model_dump().items()}


class EvidenceQualityEvaluation(BaseModel):
    """Multi-dimensional evidence quality evaluation profile."""
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    overall_score: float = Field(..., ge=0.0, le=1.0, description="Weighted 0.0-1.0 evidence quality score")
    verdict: EvidenceQualityVerdict = Field(..., description="Categorical quality verdict")
    dimension_scores: Dict[str, float] = Field(default_factory=dict, description="Per-dimension scores (0.0-1.0)")
    positive_factors: List[str] = Field(default_factory=list, description="Verified high-quality signals")
    blocking_factors: List[str] = Field(default_factory=list, description="Quality defects or safety blockers")
    vetoed: bool = Field(default=False, description="Whether a hard safety veto was triggered")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class ReviewDecision(BaseModel):
    """Structured deterministic decision produced by the HumanReviewGate."""
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    decision: str = Field(..., description="APPROVE, REJECT, or NEEDS_REVISION")
    is_valid: bool = Field(..., description="Whether the resolution passed all deterministic checks")
    checks: Dict[str, str] = Field(default_factory=dict, description="Checklist results for all 10 criteria")
    rejection_reasons: List[str] = Field(default_factory=list, description="Specific failure reasons if any")
    quality_evaluation: Optional[EvidenceQualityEvaluation] = Field(default=None, description="Quality scoring results")
    reviewer_id: Optional[str] = Field(default=None, description="Identifier of human reviewer")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class HumanResolution(BaseModel):
    """
    Structured record of a customer resolution provided or reviewed by a human specialist.
    """
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    resolution_id: str = Field(default_factory=lambda: f"res_{uuid.uuid4().hex[:12]}")
    conversation_id: str = Field(..., description="Originating conversation identifier")
    case_id: Optional[str] = Field(default=None, description="Associated case tracking ID")
    customer_problem: str = Field(..., description="Summary or transcript of the customer issue")
    confirmed_facts: Dict[str, Any] = Field(default_factory=dict, description="Facts verified during interaction")
    problem_family: str = Field(..., description="Operational problem family name")
    primary_intent: str = Field(default="general_device_support", description="Operational intent category")
    actions_attempted: List[str] = Field(default_factory=list, description="Troubleshooting actions tried")
    actions_that_worked: List[str] = Field(default_factory=list, description="Actions that successfully fixed the issue")
    final_resolution: str = Field(..., description="Exact technical resolution or customer guidance")
    resolution_status: str = Field(default="RESOLVED", description="Terminal state: RESOLVED or ESCALATED")
    specialist_notes: str = Field(default="", description="Internal notes from the support engineer")
    evidence_sources: List[str] = Field(default_factory=list, description="Documentation or KB article URLs")
    specialist_confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Specialist confidence score")
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    reviewed_at: Optional[str] = Field(default=None)
    reviewer_id: Optional[str] = Field(default=None)
    review_status: str = Field(default="PENDING")
    rejection_reason: Optional[str] = Field(default=None)
    validation_result: Optional[Dict[str, Any]] = Field(default=None)
    promotion_status: ResolutionLifecycleState = Field(default=ResolutionLifecycleState.CAPTURED)
    evidence_quality_score: Optional[float] = Field(default=None)
    version: str = Field(default="1.0.0")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class ApprovedEvidenceItem(BaseModel):
    """
    Versioned, promoted operational evidence item ready for retrieval in production.
    """
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    evidence_id: str = Field(default_factory=lambda: f"ev_{uuid.uuid4().hex[:12]}")
    version: str = Field(default="1.0.0", description="Semantic evidence version (e.g. 1.0.0)")
    originating_resolution_id: str = Field(..., description="Originating human resolution ID")
    source_type: str = Field(default="HUMAN_VALIDATED_EVIDENCE", description="Provenance discriminator")
    provenance: str = Field(default="reviewed_human_resolution", description="Audit provenance source")
    problem_family: str = Field(..., description="Operational problem family")
    primary_intent: str = Field(default="general_device_support", description="Operational intent")
    customer_problem_summary: str = Field(..., description="Representative customer issue description")
    brand_guidance: str = Field(..., description="Verified brand support guidance")
    operational_actions: List[str] = Field(default_factory=list, description="Canonical troubleshooting actions")
    evidence_quality_score: float = Field(..., ge=0.0, le=1.0, description="Audited quality score")
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    approved_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    promoted_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    reviewer_id: str = Field(default="system_reviewer", description="Authorizing reviewer ID")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()
