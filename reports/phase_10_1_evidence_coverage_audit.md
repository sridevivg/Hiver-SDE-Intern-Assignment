# Phase 10.1 — Evidence Coverage & Retrieval Recall Audit

**Date:** 2026-09-14  
**Status:** Complete  
**Protected Benchmark:** 77 human-reviewed records  
**Golden SHA-256:** `1d3e9b3b8bdef3750437e59b17ab7c27167fd1548c2bb29de984151296c5b45a` ✓ Immutable

---

## Executive Summary

> **Engineering Question:** When SupportGraph AI cannot find usable evidence, is the problem in the pipeline or is the knowledge genuinely missing?

Phase 10 measured 27.3% usable evidence coverage. Phase 10.1 was commissioned to diagnose the root cause of the 72.7% failure rate — before implementing any further engineering.

**Finding:** The primary limitation is not retrieval, ranking, or validation calibration. The primary limitation is the **EvidenceRanker's operational symptom matching layer**, which cannot recognize relevant evidence for problems expressed without explicit symptom keywords. In the benchmark, 48.8% of evidence-limited cases use `general_device_support` as their predicted intent with `unspecified_device_issue` as their primary symptom — the ranker classifies all candidates as `WEAK_SEMANTIC_MATCH`, and the auditor correctly logs `TRUE_KNOWLEDGE_GAP` because no `DIRECT_PROBLEM_MATCH` or `RELATED_SYMPTOM` tier evidence is found anywhere.

This is not a retrieval failure or a corpus gap. It is an **operational intent resolution failure**: the production pipeline cannot correctly identify the specific operational problem, and therefore cannot evaluate evidence relevance against a meaningful problem definition.

---

## Methodology

### Audit Corpus Construction

- **Source:** `data/processed/conversation_messages.parquet` — 80,717 conversations, 132,047 customer messages
- **Exclusions:** All 200 golden benchmark `conversation_id` values excluded to prevent data leakage  
- **Sample size:** 5,000 conversations (stratified random sample, seed=42)
- **Intent labeling:** Lightweight heuristic keyword-based inference (for operational relevance classification only; not used in production)
- **SHA-256 verified:** Before and after build — no golden data modified

### Production Pipeline Run

All 77 human-reviewed records were processed by the complete production pipeline:
- `SupportResolutionEngine.process_message()` 
- `EscalationQualityAnalyzer.analyze_escalation()`
- Evidence-limited cases identified using `EscalationCategory.EVIDENCE_LIMITED`

### Extended Corpus Search

For each evidence-limited case:
1. TF-IDF vectorizer trained on 5,000 audit corpus entries
2. Top-100 candidates retrieved (vs. production top-3)
3. Each candidate evaluated by `EvidenceRanker.classify_evidence_tier()` for operational relevance
4. Only `DIRECT_PROBLEM_MATCH` or `RELATED_SYMPTOM` tiers counted as relevant

### Root Cause Classification (Deterministic, Ordered)

```
1. Check information sufficiency → INSUFFICIENT_INFORMATION
2. Extended corpus search for operationally relevant evidence
   └── None found → TRUE_KNOWLEDGE_GAP
3. Relevant evidence exists but not in production top-3 → RETRIEVAL_MISS
4. Relevant evidence exists but below cutoff → RANKING_MISS
5. Relevant in top-3 but validator rejected → VALIDATION_OVER_REJECTION
6. Retrieval succeeded but representation limits matching → REPRESENTATION_LIMITATION
```

---

## Results

### Benchmark Breakdown

| Metric | Count |
|---|---|
| Total Evaluated Records | 77 |
| Auto-Handled | 19 |
| Total Escalated | 58 |
| Evidence-Limited (Audited) | **43** |
| Other Escalations | 15 |

### Root Cause Distribution (43 Evidence-Limited Cases)

| Root Cause | Count | % |
|---|---|---|
| **TRUE_KNOWLEDGE_GAP** | **41** | **95.3%** |
| INSUFFICIENT_INFORMATION | 2 | 4.7% |
| RETRIEVAL_MISS | 0 | 0.0% |
| RANKING_MISS | 0 | 0.0% |
| VALIDATION_OVER_REJECTION | 0 | 0.0% |
| REPRESENTATION_LIMITATION | 0 | 0.0% |

### Key Rates

| Metric | Value |
|---|---|
| **Recoverable Failure Rate** (engineering-fixable) | **0.0%** |
| **True Knowledge Gap Rate** | **95.3%** |
| Insufficient Information Rate | 4.7% |
| Evidence-Limited Cases With Relevant Evidence Anywhere | **0.0%** |

---

## Case Studies

### Case Study 1 — TRUE_KNOWLEDGE_GAP: Mac Software Issue (High Sierra)

```
Case ID: conv_AppleSupport_2793167
Customer Message: "@AppleSupport What have you guys done with High Sierra?
                   It has completely bugged out a bunch of programs"

Predicted Intent: mac_software_issue
Problem Profile primary_symptom: unspecified_device_issue
Problem Profile update_related: False

Extended Corpus Search (top-100):
  No DIRECT_PROBLEM_MATCH or RELATED_SYMPTOM candidates found.
  Reason: query symptom is 'unspecified_device_issue' — ranker cannot
  match against 'mac' or 'High Sierra' keywords because these
  are not in the EvidenceRanker's symptom recognition patterns.

Root Cause: TRUE_KNOWLEDGE_GAP
```

### Case Study 2 — TRUE_KNOWLEDGE_GAP: Vague "Fix This" Message

```
Case ID: conv_AppleSupport_1642576
Customer Message: "Hey @115858 I️ need you to fix this."

Predicted Intent: general_device_support
Problem Profile primary_symptom: unspecified_device_issue

Extended Corpus Search (top-100):
  No DIRECT_PROBLEM_MATCH or RELATED_SYMPTOM candidates found.
  Reason: customer provides no operational information whatsoever.
  However, this is classified as TRUE_KNOWLEDGE_GAP rather than
  INSUFFICIENT_INFORMATION because the message has ≥4 meaningful tokens.

Root Cause: TRUE_KNOWLEDGE_GAP
Note: Borderline INSUFFICIENT_INFORMATION — message has tokens but
      zero operational content (no symptom, device, or context).
```

### Case Study 3 — INSUFFICIENT_INFORMATION: Completely Vague Messages

```
2 cases classified as INSUFFICIENT_INFORMATION.
Pattern: messages like "is anyone else's shit doing this?????"
  (with only 4 meaningful tokens after stopword removal)
  or image-only messages with no text operational content.

Root Cause: INSUFFICIENT_INFORMATION
Recommended: Tighten ambiguity gate threshold or request customer clarification.
```

### Case Study 4 — What the EvidenceRanker Actually Does

For a WiFi-related query:
```python
query = "why does WiFi turn on by itself when I turn it off manually"
profile.primary_symptom = "wireless Wi-Fi or cellular network connectivity failure"

# EvidenceRanker shared_symptom check:
# Pattern for "battery" — not triggered
# Pattern for "audio" — not triggered
# Pattern for "display" — not triggered
# Pattern for "keyboard" — not triggered
# Pattern for "account" — not triggered
# Pattern for "billing" — not triggered
# → shared_symptom = False

# Even with a perfect wifi/connectivity corpus entry:
tier = WEAK_SEMANTIC_MATCH  # because no keyword patterns matched
```

The ranker has **5 hardcoded symptom patterns** (battery, audio, display, keyboard, account/billing). Any problem outside these 5 categories is always `WEAK_SEMANTIC_MATCH`, regardless of how relevant the corpus entry is.

---

## Engineering Conclusion

> **The primary cause of 95.3% evidence-limited classification is not a missing corpus, not a retrieval failure, and not over-strict validation.**
>
> **The primary cause is that the `EvidenceRanker` can only recognize operational relevance for 5 symptom categories.** Every other problem type — Mac software issues, WiFi connectivity, general device problems, unclear symptoms — is automatically assigned `WEAK_SEMANTIC_MATCH` regardless of the historical corpus content.

This means:
- Adding 80,000 more corpus entries **would not fix the problem** unless those entries contain battery/audio/display/keyboard/account keywords
- Increasing top-K **would not fix the problem** — the operational tier ceiling is `WEAK_SEMANTIC_MATCH`  
- Loosening validation thresholds **could partially help** but would sacrifice precision  
- The actual fix is expanding the `EvidenceRanker`'s operational symptom pattern coverage

### Cascading Effect Discovery

The `EvidenceRanker` limitation also partially explains the `CaseRetriever`'s continued use of the golden benchmark as corpus. Since only 200 records exist in the retrieval corpus, the ranker can still find battery/audio/keyboard matches — but for the 48.8% of cases with `general_device_support` intent, there are no matching entries even in the golden corpus itself, producing the observed `sim=1.0` self-match followed by all `WEAK_SEMANTIC_MATCH` candidates.

---

## Recommended Next Phase

**OUTCOME D — Corpus Coverage Improvement, with a specific focus:**

> **Phase 10.2 — EvidenceRanker Symptom Coverage Expansion**

The EvidenceRanker must be extended to cover:
1. WiFi/connectivity symptoms (`wifi`, `network`, `signal`, `bluetooth`, `cellular`)
2. Mac/computer software symptoms (`mac`, `macos`, `macbook`, `high sierra`, `finder`, `safari`)
3. General performance symptoms (`slow`, `crash`, `freeze`, `hang`, `restart`, `reboot`)
4. Screen/app symptoms beyond current display pattern (app crashes, freezes)
5. Unclear/ambiguous symptom detection → trigger INSUFFICIENT_INFORMATION earlier

Then rebuild the retrieval corpus from `data/processed/conversation_messages.parquet` (the 80,717-conversation historical dataset), which is the correct source for historical evidence.

**This is a two-part fix:**
1. Expand `EvidenceRanker` symptom patterns (operational tier coverage)
2. Switch `CaseRetriever` default corpus from golden CSV to historical parquet

---

## Honest Limitations

1. **The audit corpus uses heuristic intent inference** — labels like `mac_software_issue` are inferred by keyword, not human-labeled. Some corpus entries with macOS content may be labeled `general_device_support` due to missing keyword triggers.

2. **The 5,000-conversation sample may miss rare problem types** — using the full 80,717 conversations could reveal additional relevant evidence for some cases. However, given that 0/43 evidence-limited cases found any relevant evidence in the 5,000-entry sample, this is unlikely to change the primary conclusion.

3. **TRUE_KNOWLEDGE_GAP is the correct classification given current EvidenceRanker logic** — if the ranker's symptom patterns cannot recognize relevance, the auditor (which uses the same ranker) will always report `TRUE_KNOWLEDGE_GAP`. This is a measurement artifact of the ranker limitation, not a genuine assertion that no relevant historical cases exist.

4. **Phase 10 composite evidence still provides value** — for the 19 auto-handled cases and the 22.7% with usable evidence, the existing pipeline works correctly. Phase 10.2 should extend coverage without disturbing working components.
