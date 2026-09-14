# SupportGraph AI — Final Intent Taxonomy Design Guidelines

> **Purpose:** Comprehensive operational guidelines for human reviewers to transform Phase 4 unsupervised clusters and Phase 4.5 cluster audit findings into a robust, deployable, and scientifically sound final customer support intent taxonomy.

---

## 1. Core Intent Definition

An **intent** in SupportGraph AI represents the **underlying customer goal, task, or operational failure mode** that prompted the customer to initiate contact with customer support.

### The Intent Principle
- An intent answers: **"What obstacle is the customer facing, or what action do they want completed?"**
- An intent is **NOT** a physical hardware device or marketing brand name (e.g. *iPhone*, *MacBook*, *iPad*).
- An intent is **NOT** a general customer emotional state or sentiment (e.g. *frustrated*, *angry*, *happy*).
- An intent is **NOT** the brand's response strategy (e.g. *ask_for_details*, *request_private_message*).

---

## 2. Taxonomy Design Principles

The final human-reviewed intent taxonomy must satisfy six foundational criteria:

### Principle 1: Target Cardinality (5–10 Practical Intents)
- The operational taxonomy must contain **approximately 5 to 10 discrete intents**.
- **Why:** Taxonomies with dozens or hundreds of fine-grained classes suffer from severe class ambiguity, low inter-annotator agreement, extreme sample sparsity, and degraded classification precision. A 5–10 class taxonomy ensures every category has thousands of historical training examples and distinct decision boundaries.

### Principle 2: Customer Problem Over Product Entity
- Support cases must be categorized by the nature of the malfunction or request, not the device hardware.
- A battery draining issue on an iPhone 7, iPhone X, or iPad shares the same diagnostic workflow (battery health inspection, background app refresh check, low power mode guidance). They belong to the same intent (`battery_power_issue`).

### Principle 3: Generalizability Across Time (Avoid Event-Specific Anchors)
- The raw dataset is a historical snapshot from late 2017, heavily influenced by transient release incidents (e.g. the iOS 11 launch and the notorious "A [?]" autocorrect glitch).
- The final production taxonomy must represent **enduring support categories** that remain valid across software updates and product cycles.
- Temporary release bugs should either be consolidated into broad general categories (e.g., merging autocorrect complaints into `keyboard_typing_glitch` or `software_bug_report`) or flagged as event-specific anomalies rather than permanent operational intents.

### Principle 4: Mutual Distinguishability
- Categories should have clear, non-overlapping semantic boundaries so that human annotators and machine learning classifiers can assign labels with high inter-rater agreement (Cohen's Kappa $\ge 0.80$).

### Principle 5: Grounded Retrieval Utility
- The intent label must directly assist the retrieval component in Phase 8 by fetching historically similar resolutions from past cases with the same underlying problem.

### Principle 6: Auditability & Reproducibility
- Every intent must be justified by cluster evidence: centroid-nearest exemplars, diverse samples, and recurring TF-IDF keyword profiles.

---

## 3. Taxonomy Granularity Spectrum: Concrete Examples

To assist reviewers in calibrating the proper level of abstraction, use this spectrum:

| Level | Intent Name | Evaluation | Rationale |
|---|---|---|---|
| **Too Specific** | `iphone_8_ios_11_0_3_battery_drain` | ❌ Unusable | Over-fits to device model and patch version. Fragmentary; cannot generalize to other devices or updates. |
| **Too Broad** | `technical_issue` or `hardware_problem` | ❌ Unusable | Bucket category that lumps screens, batteries, keyboards, and Wi-Fi together. Fails to guide response retrieval. |
| **Event-Specific** | `autocorrect_letter_i_box_symbol_glitch` | ⚠️ Risky | Describes a 2-week transient software bug from November 2017. Irrelevant once iOS 11.1.1 was released. |
| **Well-Calibrated** | `battery_power_issue` | ✅ Ideal | Coherent problem domain; actionable troubleshooting path; timeless across devices and OS versions. |
| **Well-Calibrated** | `software_update_problem` | ✅ Ideal | Captures installation failures, post-update app crashes, and update freezes across all versions. |
| **Well-Calibrated** | `account_access_issue` | ✅ Ideal | Covers Apple ID lockouts, password resets, verification code failures, and iCloud authentication. |
| **Well-Calibrated** | `display_hardware_damage` | ✅ Ideal | Encompasses cracked screens, unresponsive touch digitizers, and black screens requiring service. |
| **Well-Calibrated** | `app_store_billing_issue` | ✅ Ideal | Addresses subscription inquiries, unrecognized charges, refunds, and app download errors. |

---

## 4. Discovered Clusters and Review Recommendations

Based on the quantitative audit of the 8 clusters discovered in Phase 4:

### Cluster Mapping Overview

| Cluster ID | Phase 4 Heuristic Label | Coherence & Separation Signals | Generalization Status | Recommended Human Action | Proposed Operational Intent |
|:---:|:---|:---|:---:|:---:|:---|
| **0** | `keyboard_typing_glitch` | High cohesion; heavily dominated by "A [?]" autocorrect bug | `EVENT_SPECIFIC` | **REVIEW_FOR_MERGE** with Cluster 7 into a generalized `keyboard_typing_glitch` | `keyboard_typing_glitch` |
| **1** | `battery_power_issue` | High cohesion; clear semantic focus on battery life & drain | `GENERALIZABLE` | **KEEP_AS_SEPARATE** | `battery_power_issue` |
| **2** | `possible_display_hardware_issue` | High variance; mixes broken screens, touch failure, and generic complaints | `MIXED` | **REVIEW_FOR_SPLIT** or consolidate into `hardware_display_audio_issue` | `hardware_display_issue` |
| **3** | `software_update_issue` | High cohesion; post-update performance and app failures | `MIXED` (iOS 11 heavy) | **REVIEW_FOR_MERGE** with Cluster 6 into unified `software_update_problem` | `software_update_problem` |
| **4** | `app_store_billing_issue` | Strong cohesion; iTunes, Store, iCloud cases, and account questions | `GENERALIZABLE` | **KEEP_AS_SEPARATE** or refine into `account_billing_store_issue` | `account_store_billing_issue` |
| **5** | `possible_software_update_issue` | Broad variance; covers macOS Sierra, MacBook memory, and Apple Music | `MIXED` | **REVIEW_FOR_SPLIT**; evaluate Mac/Services separation | `mac_os_software_issue` |
| **6** | `software_update_issue` | Overlaps heavily with Cluster 3 (centroid sim $\ge 0.85$); generic update bugs | `MIXED` (iOS 11 heavy) | **REVIEW_FOR_MERGE** with Cluster 3 | `software_update_problem` |
| **7** | `possible_keyboard_typing_glitch` | Overlaps heavily with Cluster 0; vulgar/frustrated tweets about typing glitch | `EVENT_SPECIFIC` | **REVIEW_FOR_MERGE** with Cluster 0 | `keyboard_typing_glitch` |

---

## 5. Candidate Unified Taxonomy Proposal (7 Intents)

Reviewers are encouraged to evaluate the following consolidated 7-intent target taxonomy:

1. **`software_update_problem`** *(Merged Clusters 3 & 6)*  
   *Scope:* Issues arising during or immediately after operating system updates (installation loops, device sluggishness post-update, app incompatibilities).
2. **`battery_power_issue`** *(Cluster 1)*  
   *Scope:* Rapid battery drain, unexpected device shutdowns, failure to charge, percentage jumping.
3. **`hardware_display_issue`** *(Cluster 2 refined)*  
   *Scope:* Physical damage, cracked screens, unresponsive touch digitizers, display blackouts, speaker/mic defects.
4. **`account_store_billing_issue`** *(Cluster 4)*  
   *Scope:* Apple ID authentication, password resets, iCloud sync errors, App Store purchase disputes, unexpected subscription billing.
5. **`keyboard_typing_glitch`** *(Merged Clusters 0 & 7)*  
   *Scope:* Keyboard input glitches, predictive text errors, autocorrect bugs, character rendering anomalies.
6. **`mac_os_software_issue`** *(Cluster 5 refined)*  
   *Scope:* Mac/MacBook desktop operating system issues, Safari browser crashes, Sierra installation, iTunes desktop library errors.
7. **`general_device_inquiry`** *(Unresolved / Miscellaneous)*  
   *Scope:* High-level customer inquiries, store reservation requests, warranty questions, or mixed opening complaints that cannot be isolated to a single component.

---

## 6. Review Process Protocol

When performing human review in `data/interim/intent_taxonomy_human_review.csv`:
1. **Never edit `cluster_id` or original cluster metrics.**
2. **Review all three sample tiers:** Check the 20 centroid-nearest examples, the 20 random examples, and the 10 diverse examples.
3. **Inspect the resolution patterns:** Notice whether Apple typically sends troubleshooting links, requests a DM, or asks for device model clarification.
4. **Assign `recommended_human_action`:** Confirm whether to merge, split, or maintain as separate.
5. **Set `final_intent_label`:** Enter the standardized snake_case label once agreement is reached.
6. **Update `review_status`:** Transition from `pending_human_review` to `accepted`, `merged`, `split`, or `rejected`.
