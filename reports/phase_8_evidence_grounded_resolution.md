# SupportGraph AI — Phase 8: Evidence-Grounded Response Generation & Resolution Verification

**Author:** SupportGraph AI Engineering Team  
**Date:** September 2026  
**Status:** Completed & Validated  
**Benchmark Target:** 77 Human-Reviewed Ground Truth Records (`data/golden/golden_set_human_review.csv`)  
**SHA-256 Checksum:** `1d3e9b3b8bdef3750437e59b17ab7c27167fd1548c2bb29de984151296c5b45a` (100% Immutable)

---

## 1. Executive Summary & Problem Discovery

In previous phases, SupportGraph AI built intent classification, golden dataset curation, uncertainty-aware routing (Phase 6), threshold calibration (Phase 6.1), problem understanding (Phase 7), and multi-signal clarity gating (Phase 7.1). While Phase 7.1 successfully cut unsafe auto-handles by 46.4% and raised auto-handle precision to 63.4%, benchmark evaluation revealed that **confidence and clarity signals alone remain insufficient for safe, fully automated customer response delivery**.

A system cannot safely resolve a customer problem merely by asking:
> *"Am I confident about this intent?"*

Instead, the system must establish:
> *"Do I have strong enough operational evidence from real historical support cases to safely generate this resolution, and is the generated response verified to be grounded in that evidence without unsupported troubleshooting claims?"*

Phase 8 implements **Operational Evidence Validation** and **Response Grounding Verification**, transforming SupportGraph AI into an end-to-end evidence-grounded resolution system.

---

## 2. Core Architectural Principles

### Principle 1: Semantic Similarity $\ne$ Operational Equivalence
Dense TF-IDF and vector embeddings are used exclusively for **candidate historical case retrieval**. Candidate cases are then re-evaluated across 5 operational dimensions (Device Match, Product/Service Match, Symptom Match, Intent Match, and Cause Match) before being admitted as supporting evidence.

### Principle 2: CLEAR $\ne$ HIGH CONFIDENCE $\ne$ RESOLUTION-READY
Auto-handling requires three distinct gates to pass simultaneously:
1. **Clarity Gate (`AmbiguityDecisionGate`)**: Zero safety vetoes (sufficient information, no unresolved candidate conflict, strong primary problem).
2. **Evidence Validation (`ResolutionEvidenceValidator`)**: Strong operational evidence (`STRONG_EVIDENCE` or calibrated `MODERATE_EVIDENCE` with direct problem matches and symptom agreement).
3. **Response Grounding Verification (`ResponseGroundingVerifier`)**: Verified `PASS` ensuring the generated brand response directly addresses the customer's primary symptom, avoids cause overfocus, and introduces zero hazardous/unsupported claims.

---

## 3. Architecture & New Components

```
                    CUSTOMER SUPPORT INQUIRY
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. Problem Understanding Layer (ProblemExtractor)           │
│    ├── Device: iPhone 12 | OS: iOS 11.1                     │
│    ├── Primary Symptom: rapid battery drain                 │
│    └── Possible Cause: software_update_problem              │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Top-K Candidate Generation (TopKIntentClassifier)        │
│    └── Ranked hypotheses with confidences & margin          │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Primary Problem Selection (PrimaryProblemSelector)       │
│    └── Decouples causal trigger from core operational issue │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Ambiguity Analysis (AmbiguityAnalyzer)                   │
│    └── CLEAR_PRIMARY | CAUSE_VS_SYMPTOM | MULTI_SYMPTOM     │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Historical Case Retrieval (CaseRetriever)                │
│    └── Retrieves candidate cases from historical corpus     │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. Operational Evidence Validation (ResolutionEvidenceVal)  │
│    ├── 5 Operational Match Dimensions                       │
│    ├── Match Tiers: DIRECT | RELATED | WEAK                 │
│    └── Verdict: STRONG | MODERATE | WEAK | CONFLICTING      │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. Evidence-Grounded Response Synthesis (ResponseGenerator) │
│    └── Empathetic, brand-compliant 4-part troubleshooting   │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 8. Response Grounding Verifier (ResponseGroundingVerifier)  │
│    ├── ✓ Addresses Primary Symptom                          │
│    ├── ✓ Grounded in Verified Evidence                      │
│    └── ✓ No Hazardous / Unsupported Claims                 │
└──────────────────────────────┬──────────────────────────────┘
                               │
             ┌─────────────────┴─────────────────┐
             ▼                                   ▼
    [AUTO_HANDLE PATH]                 [ESCALATE_TO_HUMAN PATH]
    Brand Troubleshooting Response     Enriched Decision Support Package
    + Direct Historical Evidence       + Multi-Signal Checklist
                                       + Candidate Hypotheses
                                       + Evidence Verdict & Case Context
                                       + Recommended Human Action
```

---

## 4. Operational Evidence Match Tiers & Verdicts

Every retrieved historical case is classified into one of four operational tiers:

| Tier | Classification | Operational Criteria |
| :--- | :--- | :--- |
| **Tier 1** | `DIRECT_PROBLEM_MATCH` | Shares primary operational intent and core functional symptom (`battery_power`, `sound`, `display`). |
| **Tier 2** | `RELATED_SYMPTOM` | Addresses the same functional symptom under differing device or contextual circumstances. |
| **Tier 3** | `RELATED_CONTEXT` | Shares contextual trigger (e.g. software update) but involves completely different symptoms. |
| **Tier 4** | `WEAK_SEMANTIC_MATCH` | Shares lexical vocabulary/keywords without true operational problem equivalence. |

### Evidence Verdict Hierarchy
1. **`STRONG_EVIDENCE`**: At least 1 `DIRECT_PROBLEM_MATCH` with $\ge 50\%$ symptom agreement and zero intent contradictions. Auto-handling permitted.
2. **`MODERATE_EVIDENCE`**: Related symptom support meeting composite support strength threshold ($\ge 0.55$) without conflicting cases.
3. **`WEAK_EVIDENCE`**: Only Tier 4 weak semantic matches or low support strength. Immediately redirects to human escalation.
4. **`CONFLICTING_EVIDENCE`**: Retrieved cases contradict the primary intent (e.g. billing charge vs battery issue). Immediately escalates.
5. **`INSUFFICIENT_EVIDENCE`**: Corpus empty or no cases retrieved.

---

## 5. Response Grounding & Safety Verification

The `ResponseGroundingVerifier` conducts a 5-point audit on candidate responses before customer dispatch:
1. **Primary Symptom Grounding:** Verifies that the reply addresses the extracted symptom (e.g. checking battery health when battery drain is reported).
2. **Cause vs Symptom Alignment:** Confirms post-update inquiries receive symptom-specific troubleshooting rather than generic OS restore instructions.
3. **Unsupported Claims Detection:** Scans for hazardous/unsupported patterns:
   - Unauthorized physical hardware disassembly (`"take apart phone"`, `"unscrew"`)
   - Uncertified DIY component replacements (`"replace logic board yourself"`)
   - Unauthorized firmware modifications (`"jailbreak"`, `"downgrade iOS"`)
   - Destructive operations without backup warnings (`"wipe entire disk without backup"`)
   - Unverified financial/warranty promises (`"guaranteed refund"`, `"guaranteed free replacement"`)
4. **Evidence Corroboration:** Requires that troubleshooting steps align with verified historical AppleSupport support cases.
5. **Brand Communication Standards:** Ensures presence of official DM escalation link (`https://apple.co/DM`).

---

## 6. Empirical Benchmark Evaluation (77 Human Ground-Truth Records)

Evaluation against the 77 completed human-reviewed ground truth records produced the following results:

### A. Intent Classification Performance
- **Primary Intent Accuracy (Top-1):** 55.8%
- **Top-2 Candidate Coverage:** 79.2%
- **Top-3 Candidate Coverage:** 81.8%
- **Macro F1 Score:** 0.4936

### B. Operational Evidence Validation Metrics
- **Direct Evidence Availability:** 49.4% of benchmark inquiries matched $\ge 1$ historical Tier 1 case
- **Average Intent Agreement:** 48.5%
- **Average Symptom Agreement:** 27.7%
- **Evidence Consistency:** 73.6%
- **Verdict Distribution:**
  - `STRONG_EVIDENCE`: 19 (24.7%)
  - `WEAK_EVIDENCE`: 50 (64.9%)
  - `MODERATE_EVIDENCE`: 6 (7.8%)
  - `CONFLICTING_EVIDENCE`: 2 (2.6%)
- **Retrieved Match Tiers (231 Total Retrieved Cases):**
  - `DIRECT_PROBLEM_MATCH`: 60 (26.0%)
  - `RELATED_SYMPTOM`: 22 (9.5%)
  - `RELATED_CONTEXT`: 22 (9.5%)
  - `WEAK_SEMANTIC_MATCH`: 127 (55.0%)

### C. Routing & Safety Performance
- **Auto-Handle Count / Rate:** 18 / 23.4%
- **Auto-Handle Precision:** 61.1%
- **Unsafe Auto-Handles (Errors):** **7** (Down from 15 in Phase 7.1 and 28 in Phase 7 — **-75.0% error reduction**)
- **Escalation Count / Rate:** 59 / 76.6%
- **Error Interception Rate:** **79.4%** (27 out of 34 potential errors intercepted before customer exposure)
- **Human Assistance (Top-2):** **72.9%** (True ground-truth intent is in Top-2 candidates for human reviewers)

### D. Response Grounding & Verification
- **Response Grounding Pass Rate:** 96.1%
- **Average Grounding Score:** 0.87 / 1.00
- **Hazardous / Unsupported Claims Detected:** 0 in production templates

---

## 7. Multi-Phase Progression Comparison Matrix

| Phase / Architecture | Auto-Handle Rate | Auto-Handle Precision | Unsafe Auto-Handles (Errors) | Error Interception Rate | Key Architectural Advance |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Phase 6: Uncertainty Baseline** | 83.1% | 53.1% | 30 | 21.1% | Initial Top-K & entropy routing |
| **Phase 6.1: Calibrated Ranking** | 15.6% | 75.0% | 5 | 86.8% | Grid search threshold optimization |
| **Phase 7: Evidence-Aware** | 84.4% | 56.9% | 28 | 17.6% | Problem extraction & cause decoupling |
| **Phase 7.1: Multi-Signal Safe Gate** | 53.2% | 63.4% | 15 | 55.9% | 6-signal clarity gating & hard safety vetoes |
| **Phase 8: Evidence-Grounded Resolution** | **23.4%** | **61.1%** | **7** | **79.4%** | **Multi-dimensional evidence validation & response grounding verification** |

---

## 8. Verification & Scientific Integrity

1. **Full Pytest Suite**: All **339 tests** in `backend/tests/` passed cleanly:
   ```bash
   ./backend/.venv/bin/pytest backend/tests/
   ======================= 339 passed, 1 warning in 37.81s =======================
   ```
2. **Cryptographic SHA-256 Immutability Check**:
   - `data/golden/golden_set_human_review.csv` Pre-Hash: `1d3e9b3b8bdef3750437e59b17ab7c27167fd1548c2bb29de984151296c5b45a`
   - Post-Hash: `1d3e9b3b8bdef3750437e59b17ab7c27167fd1548c2bb29de984151296c5b45a`
   - Immutability Status: **PASS (100% Intact & Untouched)**
3. **Isolated Runtime Audit Log**:
   - Runtime resolution events are persisted to `data/runtime/evidence_resolution_audit.csv` without modifying benchmark data.
