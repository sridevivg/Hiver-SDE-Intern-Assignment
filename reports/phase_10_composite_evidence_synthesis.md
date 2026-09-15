# SupportGraph AI — Phase 10: Evidence Coverage Expansion & Multi-Case Evidence Synthesis

**Author:** SupportGraph AI Engineering Team  
**Date:** September 2026  
**Status:** Completed & Validated  
**Benchmark Target:** 77 Human-Reviewed Ground Truth Records (`data/golden/golden_set_human_review.csv`)  
**SHA-256 Checksum:** `1d3e9b3b8bdef3750437e59b17ab7c27167fd1548c2bb29de984151296c5b45a` (100% Immutable)

---

## 1. Executive Summary

Phase 10 addresses the core empirical bottleneck discovered in Phase 9: **72.9% of all escalations are `EVIDENCE_LIMITED`**. 

Prior to Phase 10, the system required a *single* historical support case to strongly match the customer's entire problem profile. When no single case matched completely, the case was marked `WEAK_EVIDENCE` and escalated to human reviewers — even if the customer's problem components (e.g. primary symptom, software update contextual trigger, device family, and resolution pattern) were thoroughly documented across multiple historical cases.

Phase 10 introduces **Multi-Case Evidence Synthesis & Evidence Conflict Detection**:
- Evaluates composite evidence across **5 independent operational dimensions**: Symptom, Context, Device, Intent, and Resolution Pattern.
- Enforces strict safety invariants: **Symptom coverage is mandatory**, weak semantic matches cannot be upgraded, near-duplicate cases are deduplicated, and operational contradictions trigger immediate conflict vetoes.
- Authorizes automated resolution only when composite evidence achieves `COMPOSITE_STRONG_EVIDENCE` with zero operational conflicts and 100% response grounding verification.

### Key Benchmark Results (77 Ground-Truth Records)

| Metric | Phase 8 Baseline | Phase 9 Baseline | Phase 10 (Composite Evidence) | Delta vs Phase 9 |
| :--- | :---: | :---: | :---: | :---: |
| **Usable Evidence Coverage** | 24.7% (19 cases) | 24.7% (19 cases) | **27.3% (21 cases)** | **↑ +2.6pp** |
| **Auto-Handle Rate** | 23.4% (18 cases) | 23.4% (18 cases) | **24.7% (19 cases)** | **↑ +1.3pp** |
| **Auto-Handle Precision** | 61.1% | 61.1% | **63.2%** | **↑ +2.1pp** |
| **Unsafe Auto-Handles** | 7 | 7 | **7** | **0 (Zero Increase)** |
| **Error Interception Rate** | 79.4% | 79.4% | **79.4%** | **Preserved (≥79.4%)** |
| **Dataset Immutability (SHA-256)** | PASS | PASS | **PASS** | **100% Immutable** |

---

## 2. Core Scientific Principles & Non-Negotiable Safety Rules

Phase 10 is built on two fundamental axioms:

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. NO SINGLE EXACT MATCH DOES NOT NECESSARILY MEAN NO SUFFICIENT EVIDENCE │
│ 2. MULTIPLE WEAK SEMANTIC MATCHES MUST NEVER BECOME STRONG EVIDENCE     │
└─────────────────────────────────────────────────────────────────────────┘
```

### Non-Negotiable Safety Invariants:
1. **Symptom Coverage is Mandatory**: `COMPOSITE_STRONG_EVIDENCE` requires at least one retrieved case with verified direct/related symptom coverage. Multiple cases covering device, context, and resolution steps without the primary symptom cannot form strong evidence.
2. **Weak Semantic Tier Boundary**: `WEAK_SEMANTIC_MATCH` cases are restricted to maximum `ContributionStrength.LOW` and cannot contribute `HIGH` or `MEDIUM` to any operational dimension.
3. **Context Cannot Override Symptom**: `RELATED_CONTEXT` cases (e.g. general iOS update notices) cannot independently establish symptom coverage.
4. **Near-Duplicate Deduplication**: Cases with TF-IDF cosine similarity $\ge 0.92$ are deduplicated before synthesis to prevent artificial evidence inflation from repetitive corpus fragments.
5. **Conflict Pre-Screening**: `EvidenceConflictDetector` screens for mutually exclusive operational intents and symptom contradictions. Conflicting cases are excluded, and dominant contradictions force `CONFLICTING_COMPOSITE_EVIDENCE`.
6. **No Bypass of Response Grounding**: Composite evidence authorized cases must still pass `ResponseGroundingVerifier` (Step 9) before final automated handling.

---

## 3. Architecture & Implementation

```
Customer Message
       │
       ▼
[1] Problem Understanding (CustomerProblemProfile)
       │
       ▼
[2] Top-K Intent Candidate Generation
       │
       ▼
[3] Primary Problem Selection (Cause vs Core Symptom)
       │
       ▼
[4] Ambiguity Analysis & Multi-Signal Gate (Phase 7.1)
       │
       ▼
[5] Historical Case Retrieval (Top-K)
       │
       ├─── Step 7a: Single-Case Evidence Validation (ResolutionEvidenceValidator)
       │       │
       │       ├── If STRONG_EVIDENCE ──────────► [DIRECT_STRONG_EVIDENCE] (Passthrough)
       │       │
       │       └── If WEAK / MODERATE / INSUFFICIENT
       │               │
       │               ▼
       ├─── Step 7b: Evidence Deduplication (TF-IDF Cosine Similarity < 0.92)
       │               │
       │               ▼
       ├─── Step 7c: Evidence Conflict Detection (EvidenceConflictDetector)
       │               │ (Excludes contradicting intents / symptoms)
       │               ▼
       └─── Step 7d: Multi-Case Evidence Synthesis (MultiCaseEvidenceSynthesizer)
                       │ (Evaluates 5 Dimensions: Symptom, Context, Device, Intent, Resolution)
                       ▼
       CompositeEvidenceVerdict (DIRECT_STRONG | COMPOSITE_STRONG | MODERATE | WEAK | CONFLICTING)
                       │
       ┌───────────────┴───────────────┐
       ▼                               ▼
[8] Response Generation         [10b] Escalation Package Enriched
       │                               (Includes composite breakdown:
       ▼                                covered/missing dims, conflict notes)
[9] Response Grounding Verifier
       │
       ▼
[10a] Safe Auto-Handle
```

### Components Introduced in Phase 10:

1. **`backend/app/resolution/evidence_synthesizer.py`**:
   - `MultiCaseEvidenceSynthesizer`: Core synthesis engine.
   - `CompositeEvidencePackage`: Structured output containing verdict, dimension breakdown, case contributions, deduplication log, and explainability notes.
   - `CaseEvidenceContribution`: Per-case contribution record (`contributes_to`, `contribution_strength`, `is_duplicate`, `allowed_for_synthesis`).
   - `DimensionCoverage`: Per-dimension coverage record across the 5 operational dimensions.

2. **`backend/app/resolution/evidence_conflict_detector.py`**:
   - `EvidenceConflictDetector`: Detects operational contradictions using `CONTRADICTORY_INTENT_PAIRS` and symptom contradiction signals.
   - `ConflictDetectionResult`: Categorizes conflicts into `INTENT_CONFLICT`, `SYMPTOM_CONTRADICTION`, or `MIXED_CONFLICT`.

3. **`backend/app/resolution/evidence_validator.py` (Modified)**:
   - Added `CompositeEvidenceVerdict` enum with 6 distinct verdict states.
   - Preserved full backward compatibility with Phase 8 `EvidenceVerdict`.

4. **`backend/app/resolution/support_resolution_engine.py` (Modified)**:
   - Wired multi-case synthesis as Steps 7b–7d.
   - Enriched `HumanEscalationPackage` with composite evidence breakdown for reviewer decision support.

5. **`backend/app/evaluation/composite_evidence_evaluator.py` & `backend/scripts/evaluate_composite_evidence.py`**:
   - Automated benchmark evaluation runner with strict SHA-256 dataset immutability verification.

---

## 4. Empirical Evaluation Results

### 4.1 Composite Evidence Verdict Distribution

On the 77 protected human ground-truth records:

| Verdict | Count | % of Total | Operational Action |
| :--- | :---: | :---: | :--- |
| `DIRECT_STRONG_EVIDENCE` | 19 | 24.7% | Auto-handle authorized (single direct match) |
| `COMPOSITE_STRONG_EVIDENCE` | 2 | 2.6% | Auto-handle authorized (multi-case synthesized) |
| `MODERATE_COMPOSITE_EVIDENCE` | 2 | 2.6% | Escalated (partial coverage) |
| `WEAK_COMPOSITE_EVIDENCE` | 54 | 70.1% | Escalated (lexical/semantic only) |
| `CONFLICTING_COMPOSITE_EVIDENCE` | 0 | 0.0% | Escalated (contradiction veto) |
| `INSUFFICIENT_COMPOSITE_EVIDENCE` | 0 | 0.0% | Escalated (empty corpus) |

### 4.2 Evidence Dimension Coverage & Operational Quality

- **Resolution Pattern Coverage Rate:** 75.3% (58/77 cases have clear troubleshooting patterns in retrieved brand replies)
- **Symptom Coverage Rate:** 32.5% (25/77 cases have retrieved cases addressing the primary symptom)
- **Device Coverage Rate:** 22.1% (17/77 cases explicitly match customer device family)
- **Context Coverage Rate:** 14.3% (11/77 cases match update/purchase contextual triggers)
- **Average Intent Consistency:** 48.7%
- **Average Composite Score:** 0.3760

### 4.3 Conflict Detection Safety Metrics

- **Cases with Conflicts Detected:** 2 (2.6%)
- **Intent Conflicts Detected:** 2 (e.g., `battery_power_issue` vs `billing_purchase_issue`)
- **Symptom Contradictions:** 0
- **Total Cases Excluded by Conflict Screening:** 3 retrieved cases excluded from contributing to composite scores.

---

## 5. Multi-Phase Evolution Progression Matrix

| Phase | Milestone | Auto-Handle Rate | Precision | Unsafe Auto | Error Interception |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Phase 6** | Uncertainty-Aware Routing | 36.4% | 82.1% | 5 | 85.3% |
| **Phase 6.1** | Routing Calibration | 35.1% | 85.2% | 4 | 88.2% |
| **Phase 7.1** | Ambiguity Gating & Vetoes | 31.2% | 87.5% | 3 | 91.2% |
| **Phase 8** | Single-Case Evidence Grounding | 23.4% | 61.1% | 7 | 79.4% |
| **Phase 9** | Escalation Quality Analysis | 23.4% | 61.1% | 7 | 79.4% |
| **Phase 10** | **Multi-Case Evidence Synthesis** | **24.7%** | **63.2%** | **7** | **79.4%** |

### Key Observations:
1. **Precision Improvement**: Precision increased from 61.1% to 63.2% (+2.1pp) because synthesized cases have verified dimensional corroboration.
2. **Zero Unsafe Expansion**: The number of unsafe auto-handles remained unchanged at 7 (0 new errors introduced).
3. **Safety Invariant Preserved**: Error interception rate remained at 79.4%, meeting the non-negotiable benchmark constraint.

---

## 6. Test Suite & Verification Results

- **Total Test Modules:** 40
- **Total Tests Passing:** 383 passed, 0 failures, 1 warning (unrelated KMeans convergence warning).
- **New Unit Tests Added (11 tests):**
  - `backend/tests/test_evidence_synthesizer.py` (7 tests)
  - `backend/tests/test_evidence_conflict_detector.py` (4 tests)
- **Dataset Immutability Verification:**
  - `pre_eval_sha256`: `1d3e9b3b8bdef3750437e59b17ab7c27167fd1548c2bb29de984151296c5b45a`
  - `post_eval_sha256`: `1d3e9b3b8bdef3750437e59b17ab7c27167fd1548c2bb29de984151296c5b45a`
  - Result: **100% Immutable (PASS)**.

---

## 7. Conclusion & Next Steps

Phase 10 successfully unlocks multi-case evidence synthesis for SupportGraph AI. By replacing single-case exclusivity with a safety-governed 5-dimension contribution model, the system expanded usable evidence coverage to 27.3% and increased auto-handle precision to 63.2% without introducing a single new unsafe auto-handle.

### Recommended Next Steps (Phase 11):
1. **Dynamic Top-K Retrieval**: Expand retrieval depth from $k=3$ to $k=5$ for complex multi-symptom queries to further increase composite evidence candidate variety.
2. **Corpus Expansion**: Index additional historical resolution templates to address the remaining 54 `WEAK_COMPOSITE_EVIDENCE` cases.
3. **Continuous Review Feedback Loop**: Ingest human reviewer adjudications back into the evidence index to expand verified case coverage dynamically.
