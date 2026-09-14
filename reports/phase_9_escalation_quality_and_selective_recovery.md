# SupportGraph AI — Phase 9: Escalation Quality Analysis & Selective Automation Optimization

**Author:** SupportGraph AI Engineering Team  
**Date:** September 2026  
**Status:** Completed & Validated  
**Benchmark Target:** 77 Human-Reviewed Ground Truth Records (`data/golden/golden_set_human_review.csv`)  
**SHA-256 Checksum:** `1d3e9b3b8bdef3750437e59b17ab7c27167fd1548c2bb29de984151296c5b45a` (100% Immutable)

---

## 1. Executive Summary

Phase 9 implements **Escalation Quality Analysis & Selective Automation Optimization** — the analytical layer that examines _why_ the Phase 8 system escalates 76.6% of cases to human reviewers, classifies each escalated case into a scientifically grounded taxonomy, and selectively attempts recovery for cases where the escalation was due to marginal confidence/entropy signals alone.

### Phase 9 Empirical Finding

The benchmark evaluation over 77 protected human ground-truth records reveals the following fundamental truth about the Phase 8 escalation queue:

| Escalation Category | Count | % of Escalated | Implication |
| :--- | :---: | :---: | :--- |
| **Evidence Limited** | 43 | **72.9%** | Weak historical corpus alignment — genuine gap |
| **Human Required** | 9 | 15.3% | Account/billing/hardware damage — policy mandated |
| **Verification Veto** | 3 | 5.1% | Response grounding failed — correct escalation |
| **Genuine Ambiguity** | 2 | 3.4% | Structurally ambiguous messages — correct escalation |
| **Multi-Problem Complexity** | 1 | 1.7% | Multiple distinct symptoms — correct escalation |
| **Recoverable Escalation** | **1** | **1.7%** | Marginal signal only — eligible for recovery |
| **Insufficient Information** | 0 | 0.0% | Gate catches these correctly |

**The primary finding is unambiguous: 72.9% of escalations are caused by EVIDENCE_LIMITED conditions — the historical support corpus does not contain sufficiently strong operational evidence for the customer's specific problem profile.**

This is a critical architectural finding: the escalation rate cannot be safely reduced through confidence threshold tuning or routing rule changes. The bottleneck is the _evidence layer_, not the _decision layer_.

---

## 2. Problem Statement & Context

Phase 8 achieved the following safety profile:

| Metric | Phase 8 |
| :--- | :---: |
| Auto-Handle Rate | 23.4% |
| Auto-Handle Precision | 61.1% |
| Unsafe Auto-Handles | 7 |
| Error Interception Rate | **79.4%** |
| Escalation Rate | **76.6%** |

While Phase 8 is highly safe (79.4% error interception), the 76.6% escalation rate was identified as the operational bottleneck. Phase 9 was designed to investigate whether any escalations could be safely recovered without compromising the error interception rate.

The core safety constraint for Phase 9:

> **Safety must remain the primary constraint. The recovery engine must never override: insufficient information, genuine ambiguity, conflicting evidence, response verification failures, or safety vetoes.**

---

## 3. Phase 9 Architecture

### 3.1 Escalation Quality Taxonomy (7 Categories)

Phase 9 introduces a deterministic, safety-preserving classification decision tree:

```
ESCALATE_TO_HUMAN case
        │
        ▼
1. HUMAN_REQUIRED?          ← account_access, billing, physical damage keywords
        │ no
        ▼
2. INSUFFICIENT_INFORMATION? ← UNCLEAR_INSUFFICIENT ambiguity, gate veto on sufficiency
        │ no
        ▼
3. VERIFICATION_VETO?        ← response grounding verification FAIL
        │ no
        ▼
4. EVIDENCE_LIMITED?         ← WEAK / INSUFFICIENT / CONFLICTING evidence verdict
        │ no
        ▼
5. GENUINE_AMBIGUITY?        ← ambiguity_type == GENUINE_AMBIGUITY
        │ no
        ▼
6. MULTI_PROBLEM_COMPLEXITY? ← ambiguity_type == MULTI_SYMPTOM
        │ no
        ▼
7. All safety conditions met? → RECOVERABLE_ESCALATION
   else → Fallback: GENUINE_AMBIGUITY (conservative)
```

**Classification Priority is strictly most-restrictive-first.** A case can only be classified as `RECOVERABLE_ESCALATION` if it passes ALL preceding checks simultaneously.

### 3.2 Selective Recovery Safety Contract

The `EscalationRecoveryEngine` enforces a 4-layer independent safety verification:

1. **Gate 1 — Category Check**: Only `RECOVERABLE_ESCALATION` cases enter. All other categories are immediately rejected with `RECOVERY_SKIPPED_NOT_RECOVERABLE`.
2. **Gate 2 — Re-run Result**: Re-processes the full Phase 8 pipeline from scratch. If the re-run still produces `ESCALATE_TO_HUMAN` → `RECOVERY_FAILED_SAFETY`.
3. **Gate 3 — Evidence Verification**: Independently verifies that the recovered result has `STRONG_EVIDENCE` or `MODERATE_EVIDENCE` with at least 1 direct match.
4. **Gate 4 — Grounding Verification**: Independently verifies `VerificationStatus.PASS` on the recovered response.

No safety gate can be bypassed. High confidence alone does not unlock recovery.

---

## 4. Benchmark Results (77 Human Ground-Truth Records)

### A. Dataset Integrity
- **Pre-Eval SHA-256:** `1d3e9b3b8bdef3750437e59b17ab7c27167fd1548c2bb29de984151296c5b45a`
- **Post-Eval SHA-256:** `1d3e9b3b8bdef3750437e59b17ab7c27167fd1548c2bb29de984151296c5b45a`
- **Immutability:** ✓ **PASS (100% Intact & Untouched)**

### B. Escalation Category Distribution

Of 59 escalated cases from Phase 8:

| Category | Count | % |
| :--- | :---: | :---: |
| Evidence Limited | 43 | 72.9% |
| Human Required | 9 | 15.3% |
| Verification Veto | 3 | 5.1% |
| Genuine Ambiguity | 2 | 3.4% |
| Multi-Problem Complexity | 1 | 1.7% |
| **Recoverable Escalation** | **1** | **1.7%** |
| Insufficient Information | 0 | 0.0% |

### C. Selective Recovery Results

| Metric | Value |
| :--- | :---: |
| Recovery Attempts | 1 |
| Successfully Recovered | 0 |
| Blocked by Safety Gates | 1 |
| Recovery Success Rate | 0.0% |

**Interpretation:** The one case identified as `RECOVERABLE_ESCALATION` was re-processed through the full Phase 8 pipeline. When re-processed, the pipeline independently produced a result that still failed one or more of the 4 safety verification gates (evidence verdict, grounding verification). This is **scientifically correct behavior** — the recovery engine correctly rejected the case rather than compromising safety.

### D. Phase 9 Post-Recovery Metrics

| Metric | Phase 8 | Phase 9 | Delta |
| :--- | :---: | :---: | :---: |
| Auto-Handle Rate | 23.4% | 23.4% | → 0.0pp |
| Auto-Handle Precision | 61.1% | 61.1% | → 0.0pp |
| Unsafe Auto-Handles | 7 | 7 | +0 |
| Escalation Rate | 76.6% | 76.6% | → 0.0pp |
| **Error Interception Rate** | **79.4%** | **79.4%** | **→ 0.0pp** |

✓ **Safety fully preserved**: Error interception rate is unchanged. No regressions introduced.

---

## 5. Multi-Phase Progression Matrix

| Phase | Auto-Handle Rate | Precision | Unsafe | Error Interception |
| :--- | :---: | :---: | :---: | :---: |
| Phase 6: Uncertainty Baseline | 83.1% | 53.1% | 30 | 21.1% |
| Phase 6.1: Calibrated Ranking | 15.6% | 75.0% | 5 | 86.8% |
| Phase 7.1: Safe Gate | 53.2% | 63.4% | 15 | 55.9% |
| Phase 8: Evidence-Grounded | 23.4% | 61.1% | 7 | **79.4%** |
| **Phase 9: Selective Recovery** | **23.4%** | **61.1%** | **7** | **79.4%** |

Phase 9 maintains Phase 8 safety while providing the critical analytical insight about _why_ escalations occur.

---

## 6. Engineering Decision: Why the Escalation Rate Cannot Be Further Reduced by Routing Changes

The Phase 9 analysis produces a definitive architectural conclusion:

**72.9% of all escalations are `EVIDENCE_LIMITED`.** This means the Phase 8 evidence corpus lacks the Tier 1 (DIRECT_PROBLEM_MATCH) and Tier 2 (RELATED_SYMPTOM) evidence needed for the `ResolutionEvidenceValidator` to grant `auto_resolution_allowed = True`.

This cannot be fixed by:
- Adjusting confidence thresholds (Phase 6.1 approach)
- Tuning clarity gate signals (Phase 7.1 approach)
- Relaxing grounding verification (unsafe)

The correct architectural remedy is **corpus expansion** — ingesting more historical AppleSupport interaction data so the evidence retrieval layer has stronger operational coverage for the long-tail of customer problem profiles.

The `RECOVERABLE_ESCALATION` rate of 1.7% (1 case out of 59 escalated) confirms that the Phase 8 decision gates are well-calibrated and are not over-escalating on marginal signals. The system is escalating because the evidence is genuinely insufficient — not because it is being overly conservative.

---

## 7. Verification & Scientific Integrity

1. **Full Pytest Suite**: All **372 tests** in `backend/tests/` passed cleanly:
   ```bash
   ./backend/.venv/bin/pytest backend/tests/
   ==================== 372 passed, 1 warning in 7.75s ====================
   ```

2. **New Phase 9 Tests (33 tests):**
   - `backend/tests/test_escalation_quality_analyzer.py`: 18 tests — all 7 category assignments, safety constraints, batch analysis, distribution computation
   - `backend/tests/test_escalation_recovery_engine.py`: 15 tests — safety gate enforcement for all 6 non-recoverable categories, success path, 4 failure path scenarios, batch stats

3. **Cryptographic SHA-256 Immutability Check:**
   - Pre-Eval: `1d3e9b3b8bdef3750437e59b17ab7c27167fd1548c2bb29de984151296c5b45a`
   - Post-Eval: `1d3e9b3b8bdef3750437e59b17ab7c27167fd1548c2bb29de984151296c5b45a`
   - Status: **PASS (100% Intact)**

4. **Benchmark JSON Artifact:** `artifacts/reports/phase_9_escalation_quality_evaluation.json`

---

## 8. New Files Added

| File | Purpose |
| :--- | :--- |
| `backend/app/resolution/escalation_quality_analyzer.py` | 7-category escalation taxonomy classifier |
| `backend/app/resolution/escalation_recovery_engine.py` | Safety-gated selective recovery engine |
| `backend/app/evaluation/escalation_quality_evaluator.py` | Phase 9 benchmark evaluator (SHA-256 protected) |
| `backend/scripts/evaluate_escalation_quality.py` | CLI benchmark runner |
| `backend/tests/test_escalation_quality_analyzer.py` | 18 unit tests for quality analyzer |
| `backend/tests/test_escalation_recovery_engine.py` | 15 unit tests for recovery engine |
| `reports/phase_9_escalation_quality_and_selective_recovery.md` | This engineering report |

