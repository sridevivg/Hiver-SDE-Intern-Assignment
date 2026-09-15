"""
SupportGraph AI — Resolution Package (Phase 8 + Phase 9 + Phase 10)

Exposes:
- SupportResolutionEngine, SupportResolutionResult, HumanEscalationPackage
- ResolutionEvidenceValidator, EvidenceValidationResult, EvidenceVerdict, EvidenceDimensionMatch
- CompositeEvidenceVerdict                                                  (Phase 10)
- EvidenceGroundedResponseGenerator, GroundedResponseCandidate
- ResponseGroundingVerifier, ResponseGroundingResult, VerificationStatus
- ResolutionAuditor, ResolutionAuditRecord
- BRAND_RESOLUTION_STRATEGIES
- EscalationQualityAnalyzer, EscalationCategory, EscalationQualityResult   (Phase 9)
- EscalationRecoveryEngine, EscalationRecoveryResult, RecoveryOutcome       (Phase 9)
- MultiCaseEvidenceSynthesizer, CompositeEvidencePackage                    (Phase 10)
- CaseEvidenceContribution, ContributionStrength, DimensionCoverage         (Phase 10)
- EvidenceConflictDetector, ConflictDetectionResult, ConflictType           (Phase 10)
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
    from app.resolution.evidence_conflict_detector import (
        ConflictDetectionResult,
        ConflictType,
        EvidenceConflictDetector,
    )
    from app.resolution.evidence_synthesizer import (
        CaseEvidenceContribution,
        CompositeEvidencePackage,
        ContributionStrength,
        DimensionCoverage,
        MultiCaseEvidenceSynthesizer,
    )
    from app.resolution.evidence_validator import (
        CompositeEvidenceVerdict,
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
    from backend.app.resolution.evidence_conflict_detector import (  # type: ignore[no-redef]
        ConflictDetectionResult,
        ConflictType,
        EvidenceConflictDetector,
    )
    from backend.app.resolution.evidence_synthesizer import (  # type: ignore[no-redef]
        CaseEvidenceContribution,
        CompositeEvidencePackage,
        ContributionStrength,
        DimensionCoverage,
        MultiCaseEvidenceSynthesizer,
    )
    from backend.app.resolution.evidence_validator import (  # type: ignore[no-redef]
        CompositeEvidenceVerdict,
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
    # Phase 8 core
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
    # Phase 10
    "CompositeEvidenceVerdict",
    "MultiCaseEvidenceSynthesizer",
    "CompositeEvidencePackage",
    "CaseEvidenceContribution",
    "ContributionStrength",
    "DimensionCoverage",
    "EvidenceConflictDetector",
    "ConflictDetectionResult",
    "ConflictType",
]
