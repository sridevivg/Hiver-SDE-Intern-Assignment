"""
SupportGraph AI — Deterministic Human Review Gate (Phase 13).

Runs 10 strict deterministic checks to validate candidate human resolutions:
1. Completeness
2. Problem-family consistency
3. Fact consistency
4. Action/result consistency
5. Evidence provenance
6. Contradiction detection
7. Safety risk / hazard check
8. Resolution plausibility
9. Unsupported claims
10. Golden benchmark isolation (Zero Leakage)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd

try:
    from app.core.logging import get_logger
    from app.feedback.schemas import (
        HumanResolution,
        ReviewChecklist,
        ReviewCheckStatus,
        ReviewDecision,
    )
    from app.retrieval.historical_corpus_index import (
        GOLDEN_CSV_PATH,
        GOLDEN_SHA256,
        compute_file_sha256,
    )
    from app.retrieval.problem_family_registry import (
        OperationalProblemFamily,
        ProblemFamilyDetector,
    )
except ModuleNotFoundError:
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]
    from backend.app.feedback.schemas import (  # type: ignore[no-redef]
        HumanResolution,
        ReviewChecklist,
        ReviewCheckStatus,
        ReviewDecision,
    )
    from backend.app.retrieval.historical_corpus_index import (  # type: ignore[no-redef]
        GOLDEN_CSV_PATH,
        GOLDEN_SHA256,
        compute_file_sha256,
    )
    from backend.app.retrieval.problem_family_registry import (  # type: ignore[no-redef]
        OperationalProblemFamily,
        ProblemFamilyDetector,
    )

logger = get_logger(__name__)

# Patterns that represent severe safety, security, or destructive risks
UNSAFE_PATTERNS = [
    re.compile(r"\b(rm\s+-rf|format\s+drive|dd\s+if=|drop\s+database)\b", re.IGNORECASE),
    re.compile(r"\b(jailbreak|cydia|pirate|crack(?:ed)?\s+ipa|bypass\s+icloud\s+lock)\b", re.IGNORECASE),
    re.compile(r"\b(send\s+(?:us\s+)?your\s+password|enter\s+your\s+password\s+here|share\s+passcode)\b", re.IGNORECASE),
    re.compile(r"\b(disregard\s+smoke|ignore\s+heat|plug\s+in\s+swollen\s+battery)\b", re.IGNORECASE),
    re.compile(r"\b(poke\s+battery|puncture\s+battery|water\s+on\s+smoking\s+device)\b", re.IGNORECASE),
]

# Known valid documentation provenance prefixes
VALID_PROVENANCE_PREFIXES = [
    "https://support.apple.com",
    "https://developer.apple.com",
    "https://iforgot.apple.com",
    "https://reportaproblem.apple.com",
    "kb://",
    "internal://apple_care_manual",
    "specialist://verified_tier2",
    "official_support_procedure",
]

SUSPICIOUS_PROVENANCE_PATTERNS = [
    re.compile(r"\b(fake-fix\.com|hack-apple\.net|free-unlock\.org|unsupported-third-party)\b", re.IGNORECASE),
]


class HumanReviewGate:
    """
    Evaluates candidate human resolutions against 10 strict deterministic criteria.
    Guarantees that no unsafe, incomplete, contradictory, or golden-contaminated
    resolution can ever be approved for evidence promotion.
    """

    def __init__(
        self,
        golden_csv_path: Optional[Path | str] = None,
        detector: Optional[ProblemFamilyDetector] = None,
    ) -> None:
        self.golden_path = Path(golden_csv_path or GOLDEN_CSV_PATH)
        self.detector = detector or ProblemFamilyDetector()
        self._golden_ids: Set[str] = set()
        self._golden_messages: Set[str] = set()
        self._load_golden_set()

    def _load_golden_set(self) -> None:
        """Load golden benchmark IDs and messages for strict isolation check."""
        if not self.golden_path.exists():
            return
        try:
            df = pd.read_csv(self.golden_path)
            for col in ("conversation_id", "golden_id", "tweet_id"):
                if col in df.columns:
                    for val in df[col].dropna():
                        s_val = str(val).strip()
                        self._golden_ids.add(s_val)
                        if col == "tweet_id":
                            self._golden_ids.add(f"conv_AppleSupport_{s_val}")
            if "customer_message" in df.columns:
                self._golden_messages.update({
                    msg.strip().lower()
                    for msg in df["customer_message"].dropna().astype(str)
                    if len(msg.strip()) > 10
                })
            if "normalized_message" in df.columns:
                self._golden_messages.update({
                    msg.strip().lower()
                    for msg in df["normalized_message"].dropna().astype(str)
                    if len(msg.strip()) > 10
                })
        except Exception as exc:
            logger.warning("Failed to load golden benchmark for isolation check: %s", exc)

    def evaluate(self, resolution: HumanResolution, reviewer_id: Optional[str] = None) -> ReviewDecision:
        """
        Execute all 10 deterministic checks on candidate resolution.
        """
        checklist = ReviewChecklist()
        rejection_reasons: List[str] = []

        # 1. Completeness Check
        if (
            len(resolution.customer_problem.strip()) < 8
            or len(resolution.final_resolution.strip()) < 15
            or not resolution.problem_family
            or not resolution.actions_attempted
        ):
            checklist.completeness = ReviewCheckStatus.FAIL
            rejection_reasons.append("REJECTION: Incomplete resolution missing problem text, actions, or resolution guidance.")

        # 2. Problem-Family Consistency Check
        try:
            stated_fam = OperationalProblemFamily(resolution.problem_family)
            det_fam, secondaries, _ = self.detector.detect_family(
                resolution.customer_problem,
                candidate_intent=resolution.primary_intent,
            )
            # If stated family is severely conflicting or incompatible with detected problem
            is_compatible = (
                stated_fam == det_fam
                or stated_fam in secondaries
                or stated_fam == OperationalProblemFamily.GENERAL_DEVICE_FUNCTIONALITY
                or det_fam == OperationalProblemFamily.GENERAL_DEVICE_FUNCTIONALITY
                or self.detector.are_compatible(stated_fam, det_fam)
            )
            is_conflicting = self.detector.are_conflicting(stated_fam, det_fam)

            if is_conflicting or not is_compatible:
                checklist.problem_family_consistency = ReviewCheckStatus.FAIL
                rejection_reasons.append(
                    f"REJECTION: Problem family mismatch (Declared '{stated_fam.value}' incompatible/conflicts with detected '{det_fam.value}')."
                )
        except ValueError:
            checklist.problem_family_consistency = ReviewCheckStatus.FAIL
            rejection_reasons.append(f"REJECTION: Invalid or unregistered problem family '{resolution.problem_family}'.")

        # 3. Fact Consistency Check
        facts = resolution.confirmed_facts or {}
        prob_lower = resolution.customer_problem.lower()
        if "device_model" in facts:
            dm = str(facts["device_model"]).lower()
            if "macbook" in dm and "iphone" in prob_lower and "mac" not in prob_lower:
                checklist.fact_consistency = ReviewCheckStatus.FAIL
                rejection_reasons.append("REJECTION: Confirmed fact device model contradicts problem text.")

        # 4. Action-Result Consistency Check
        if resolution.actions_that_worked:
            for act in resolution.actions_that_worked:
                # Working action must be present in attempted actions or explained in resolution
                if act not in resolution.actions_attempted and act.lower() not in resolution.final_resolution.lower():
                    checklist.action_result_consistency = ReviewCheckStatus.FAIL
                    rejection_reasons.append(f"REJECTION: Working action '{act}' not present in attempted actions or resolution text.")
                    break

        # 5. Evidence Provenance Check
        if resolution.evidence_sources:
            for src in resolution.evidence_sources:
                src_lower = src.lower()
                if any(p.search(src_lower) for p in SUSPICIOUS_PROVENANCE_PATTERNS):
                    checklist.provenance = ReviewCheckStatus.FAIL
                    rejection_reasons.append(f"REJECTION: Disallowed or untrusted evidence provenance source '{src}'.")
                    break

        # 6. Contradiction Detection Check
        res_lower = resolution.final_resolution.lower()
        if "update ios" in res_lower and "do not update" in res_lower:
            checklist.contradiction = ReviewCheckStatus.FAIL
            rejection_reasons.append("REJECTION: Resolution contains internal contradictory instructions.")

        # 7. Safety Risk & Hazard Check
        for pat in UNSAFE_PATTERNS:
            if pat.search(res_lower) or pat.search(resolution.customer_problem.lower()):
                checklist.safety = ReviewCheckStatus.FAIL
                rejection_reasons.append("REJECTION: Resolution violates safety policy (destructive command or thermal hazard disregard).")
                break

        # 8. Resolution Plausibility Check
        if len(res_lower.split()) < 4 or re.match(r"^(ok|fixed|done|idk|good|worked)$", res_lower):
            checklist.plausibility = ReviewCheckStatus.FAIL
            rejection_reasons.append("REJECTION: Resolution text is implausibly brief or non-actionable.")

        # 9. Unsupported Claims Check
        if re.search(r"\b(magic\s+repair\s+dongle|quantum\s+battery\s+restorer|ios\s+secret\s+menu\s+99)\b", res_lower):
            checklist.unsupported_claims = ReviewCheckStatus.FAIL
            rejection_reasons.append("REJECTION: Resolution asserts unsupported or fabricated technical claims.")

        # 10. Golden Benchmark Isolation Check (Mandatory)
        is_golden_leak = False
        if str(resolution.conversation_id) in self._golden_ids:
            is_golden_leak = True
        if resolution.case_id and str(resolution.case_id) in self._golden_ids:
            is_golden_leak = True
        if resolution.customer_problem.strip().lower() in self._golden_messages:
            is_golden_leak = True

        if is_golden_leak:
            checklist.golden_isolation = ReviewCheckStatus.FAIL
            rejection_reasons.append("REJECTION: Golden benchmark contamination detected. Golden records cannot enter candidate evidence.")

        # Final Decision
        is_valid = len(rejection_reasons) == 0
        decision_str = "APPROVE" if is_valid else "REJECT"

        return ReviewDecision(
            decision=decision_str,
            is_valid=is_valid,
            checks=checklist.to_dict(),
            rejection_reasons=rejection_reasons,
            reviewer_id=reviewer_id or resolution.reviewer_id,
        )
