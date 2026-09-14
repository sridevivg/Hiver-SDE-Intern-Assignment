"""
SupportGraph AI — Resolution Package (Phase 8 + Phase 9)

Exposes:
- SupportResolutionEngine: End-to-end evidence-grounded resolution pipeline
- SupportResolutionResult: Complete resolution response model
- HumanEscalationPackage: Ambiguous case review packet
- ResolutionEvidenceValidator: Multi-dimensional operational evidence validator
- EvidenceValidationResult, EvidenceVerdict, EvidenceDimensionMatch
- EvidenceGroundedResponseGenerator: Grounded AppleSupport response generator
- GroundedResponseCandidate
- ResponseGroundingVerifier: Response safety and grounding verifier
- ResponseGroundingResult, VerificationStatus
- ResolutionAuditor, ResolutionAuditRecord
- BRAND_RESOLUTION_STRATEGIES: Grounded AppleSupport troubleshooting patterns
- EscalationQualityAnalyzer, EscalationCategory, EscalationQualityResult  (Phase 9)
- EscalationRecoveryEngine, EscalationRecoveryResult, RecoveryOutcome      (Phase 9)
"""
from __future__ import annotations

try:
    from app.resolution.escalation_quality_analyzer import (
        EscalationCategory,
        EscalationQualityAnalyzer,
        EscalationQualityResult,
    )
    from app.resolution.escalation_recovery_engine import (
        EscalationRecoveryEngine,
        EscalationRecoveryResult,
        RecoveryOutcome,
    )
    from app.resolution.evidence_validator import (
        EvidenceDimensionMatch,
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
    from app.resolution.support_resolution_engine import (
        BRAND_RESOLUTION_STRATEGIES,
        HumanEscalationPackage,
        SupportResolutionEngine,
        SupportResolutionResult,
    )
except ModuleNotFoundError:
    from backend.app.resolution.escalation_quality_analyzer import (  # type: ignore[no-redef]
        EscalationCategory,
        EscalationQualityAnalyzer,
        EscalationQualityResult,
    )
    from backend.app.resolution.escalation_recovery_engine import (  # type: ignore[no-redef]
        EscalationRecoveryEngine,
        EscalationRecoveryResult,
        RecoveryOutcome,
    )
    from backend.app.resolution.evidence_validator import (  # type: ignore[no-redef]
        EvidenceDimensionMatch,
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
    from backend.app.resolution.support_resolution_engine import (  # type: ignore[no-redef]
        BRAND_RESOLUTION_STRATEGIES,
        HumanEscalationPackage,
        SupportResolutionEngine,
        SupportResolutionResult,
    )

__all__ = [
    "SupportResolutionEngine",
    "SupportResolutionResult",
    "HumanEscalationPackage",
    "BRAND_RESOLUTION_STRATEGIES",
    "ResolutionEvidenceValidator",
    "EvidenceValidationResult",
    "EvidenceVerdict",
    "EvidenceDimensionMatch",
    "EvidenceGroundedResponseGenerator",
    "GroundedResponseCandidate",
    "ResponseGroundingVerifier",
    "ResponseGroundingResult",
    "VerificationStatus",
    "ResolutionAuditor",
    "ResolutionAuditRecord",
    # Phase 9
    "EscalationQualityAnalyzer",
    "EscalationCategory",
    "EscalationQualityResult",
    "EscalationRecoveryEngine",
    "EscalationRecoveryResult",
    "RecoveryOutcome",
]
