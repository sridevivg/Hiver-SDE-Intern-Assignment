# SupportGraph AI — Phase 5.9 Annotation Pattern Analysis & Taxonomy Calibration Report

**Dataset Coverage:** 57 / 200 Records (28.5%)  
**Observed Annotation Agreement:** 35.1% (95% CI: [24.0%, 48.1%])  
**Human Overrides:** 37 (64.9%)  

> [!IMPORTANT]
> **Scientific Integrity Disclaimer:** OBSERVED ANNOTATION AGREEMENT on N=57 completed records (28.5% coverage). This reflects annotator agreement with baseline LLM suggestions during curation and must NOT be interpreted as generalized production model accuracy.

---

## 1. Executive Summary & Agreement Statistics

| Metric | Value | Description |
| :--- | :--- | :--- |
| **Total Golden Records** | `200` | Target evaluation benchmark size |
| **Completed Human Reviews** | `57` | Authoritative ground-truth annotations |
| **Pending Human Reviews** | `143` | Unreviewed records remaining in queue |
| **Agreed AI Suggestions** | `20` | Human confirmed baseline AI suggestion |
| **Overridden AI Suggestions** | `37` | Human assigned a different taxonomy label |
| **Marked 'unclear_needs_review'** | `3` | Records requiring special investigation |
| **Observed Agreement Rate** | `35.1%` | Preliminary agreement on reviewed sample |

---

## 2. Confusion Matrix (AI Baseline vs Human Ground Truth)

Rows represent **Baseline AI Suggestions**; columns represent **Human True Annotations**.

| Baseline AI Suggestion \ Human GT | `account_access_issue` | `battery_power_issue` | `billing_purchase_issue` | `display_touch_issue` | `general_device_support` | `hardware_audio_connection_issue` | `keyboard_typing_issue` | `mac_software_issue` | `software_update_problem` | `unclear_needs_review` | **Total** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `account_access_issue` | - | - | - | - | 2 | - | - | - | - | - | **2** |
| `battery_power_issue` | - | 1 | - | - | - | - | - | - | - | - | **1** |
| `billing_purchase_issue` | - | - | - | - | 1 | - | - | - | - | - | **1** |
| `display_touch_issue` | - | - | - | 1 | 2 | 2 | 2 | - | 1 | - | **8** |
| `general_device_support` | - | 3 | - | 2 | 13 | 3 | 2 | 4 | 6 | 1 | **34** |
| `hardware_audio_connection_issue` | - | - | - | - | - | - | - | - | - | - | **0** |
| `keyboard_typing_issue` | - | - | - | - | - | - | 1 | - | 1 | - | **2** |
| `mac_software_issue` | - | - | - | - | - | 1 | 2 | - | - | 1 | **4** |
| `software_update_problem` | - | - | - | - | - | - | - | - | 3 | - | **3** |
| `unclear_needs_review` | - | - | - | - | - | - | 1 | - | - | 1 | **2** |
| **Total Ground Truth** | **0** | **4** | **0** | **3** | **18** | **6** | **8** | **4** | **11** | **3** | **57** |

### Per-Intent Performance Breakdown

| Intent Label | Ground Truth Support | AI Predicted Support | Agreed | Observed Precision | Observed Recall | Observed F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `general_device_support` | 18 | 34 | 13 | 38.2% | 72.2% | 0.500 |
| `software_update_problem` | 11 | 3 | 3 | 100.0% | 27.3% | 0.429 |
| `keyboard_typing_issue` | 8 | 2 | 1 | 50.0% | 12.5% | 0.200 |
| `hardware_audio_connection_issue` | 6 | 0 | 0 | 0.0% | 0.0% | 0.000 |
| `battery_power_issue` | 4 | 1 | 1 | 100.0% | 25.0% | 0.400 |
| `mac_software_issue` | 4 | 4 | 0 | 0.0% | 0.0% | 0.000 |
| `display_touch_issue` | 3 | 8 | 1 | 12.5% | 33.3% | 0.182 |
| `unclear_needs_review` | 3 | 2 | 1 | 50.0% | 33.3% | 0.400 |
| `account_access_issue` | 0 | 2 | 0 | 0.0% | 0.0% | 0.000 |
| `billing_purchase_issue` | 0 | 1 | 0 | 0.0% | 0.0% | 0.000 |

---

## 3. Top AI → Human Correction Patterns

The following table itemizes the most frequent systematic correction pairs observed in the human review data:

| Baseline AI Suggestion | Human Ground Truth | Count | % Overrides | Recurring Semantic Signals | Actionable Recommendation |
| :--- | :--- | :---: | :---: | :--- | :--- |
| `general_device_support` | `software_update_problem` | **6** | 16.2% | 'ios' (5), '115858' (4), 'new' (3) | Enforce strict priority rules routing messages with 'software_update_problem' operational symptoms away from general fallback 'general_device_support'. |
| `general_device_support` | `mac_software_issue` | **4** | 10.8% | 'high' (4), 'sierra' (4), 'high sierra' (4) | Enforce strict priority rules routing messages with 'mac_software_issue' operational symptoms away from general fallback 'general_device_support'. |
| `general_device_support` | `battery_power_issue` | **3** | 8.1% | 'nothing' (2), 'tried' (2), '115858' (1) | Enforce strict priority rules routing messages with 'battery_power_issue' operational symptoms away from general fallback 'general_device_support'. |
| `general_device_support` | `hardware_audio_connection_issue` | **3** | 8.1% | 'usb' (3), 'type' (2), 'iphone' (1) | Enforce strict priority rules routing messages with 'hardware_audio_connection_issue' operational symptoms away from general fallback 'general_device_support'. |
| `account_access_issue` | `general_device_support` | **2** | 5.4% | '115858' (2), 'siri' (1), 'hate' (1) | Retain 'general_device_support' only when no specific subsystem, hardware component, or update causality is mentioned. |
| `display_touch_issue` | `general_device_support` | **2** | 5.4% | 'facetimed' (1), 'mate' (1), 'everyday' (1) | Retain 'general_device_support' only when no specific subsystem, hardware component, or update causality is mentioned. |
| `display_touch_issue` | `hardware_audio_connection_issue` | **2** | 5.4% | 'sound' (3), 'can' (2), 'music' (2) | Clarify decision boundary between 'display_touch_issue' and 'hardware_audio_connection_issue' in annotation guidelines. |
| `display_touch_issue` | `keyboard_typing_issue` | **2** | 5.4% | 'seeing' (2), 'fix' (2), 'fix this' (2) | Clarify decision boundary between 'display_touch_issue' and 'keyboard_typing_issue' in annotation guidelines. |
| `general_device_support` | `display_touch_issue` | **2** | 5.4% | 'screen' (2), '115858' (1), 'love' (1) | Enforce strict priority rules routing messages with 'display_touch_issue' operational symptoms away from general fallback 'general_device_support'. |
| `general_device_support` | `keyboard_typing_issue` | **2** | 5.4% | '115858' (2), 'letter' (1), 'looking' (1) | Enforce strict priority rules routing messages with 'keyboard_typing_issue' operational symptoms away from general fallback 'general_device_support'. |
| `mac_software_issue` | `keyboard_typing_issue` | **2** | 5.4% | 'apple' (2), 'issue' (2), '115858' (1) | Clarify decision boundary between 'mac_software_issue' and 'keyboard_typing_issue' in annotation guidelines. |
| `billing_purchase_issue` | `general_device_support` | **1** | 2.7% | 'aren' (1), 'apps' (1), 'downloading' (1) | Retain 'general_device_support' only when no specific subsystem, hardware component, or update causality is mentioned. |
| `display_touch_issue` | `software_update_problem` | **1** | 2.7% | 'can' (2), 'music' (2), 'app' (2) | Clarify decision boundary between 'display_touch_issue' and 'software_update_problem' in annotation guidelines. |
| `general_device_support` | `unclear_needs_review` | **1** | 2.7% | 'thank' (1), 'great' (1), 'chat' (1) | Enforce strict priority rules routing messages with 'unclear_needs_review' operational symptoms away from general fallback 'general_device_support'. |
| `keyboard_typing_issue` | `software_update_problem` | **1** | 2.7% | 'horrible' (2), 'please' (1), 'heaven' (1) | Clarify decision boundary between 'keyboard_typing_issue' and 'software_update_problem' in annotation guidelines. |
| `mac_software_issue` | `hardware_audio_connection_issue` | **1** | 2.7% | 'und' (1), 'ich' (1), 'dachte' (1) | Clarify decision boundary between 'mac_software_issue' and 'hardware_audio_connection_issue' in annotation guidelines. |
| `mac_software_issue` | `unclear_needs_review` | **1** | 2.7% | 'works' (1), 'phone' (1), 'macbook' (1) | Clarify decision boundary between 'mac_software_issue' and 'unclear_needs_review' in annotation guidelines. |
| `unclear_needs_review` | `keyboard_typing_issue` | **1** | 2.7% | '115858' (1), 'letter' (1), 'pls' (1) | Clarify decision boundary between 'unclear_needs_review' and 'keyboard_typing_issue' in annotation guidelines. |

---

## 4. `general_device_support` Diagnostic Audit

**Audit Verdict:** OVER-USED AS CATCH-ALL: AI over-predicts general_device_support (61.8% override rate). Human annotators consistently reclassify into concrete operational intents.

- **AI Predictions:** 34 (59.6% of reviewed sample)
- **Human Confirmed:** 13 (38.2%)
- **Human Overridden:** 21 (61.8%)

### Primary Reclassification Destinations for `general_device_support`

| Target Intent | Reclassification Count | Share of Overrides |
| :--- | :---: | :---: |
| `software_update_problem` | 6 | 28.6% |
| `mac_software_issue` | 4 | 19.0% |
| `hardware_audio_connection_issue` | 3 | 14.3% |
| `battery_power_issue` | 3 | 14.3% |
| `keyboard_typing_issue` | 2 | 9.5% |
| `display_touch_issue` | 2 | 9.5% |
| `unclear_needs_review` | 1 | 4.8% |

### Distinguishing Semantic Characteristics
- **Retained in `general_device_support`:** `115858`, `screen`, `macbook`, `turn`, `new`
- **Overridden into Specific Intents:** `115858`, `ios`, `iphone`, `please`, `fix`, `high`, `sierra`, `high sierra`

### Actionable Specificity Rules
1. DO NOT use general_device_support if a specific subsystem (audio, screen, battery, keyboard, macOS) is named.
1. DO NOT use general_device_support if problem onset is attributed to a software update ('since update', 'after updating').
1. USE general_device_support strictly for generic assistance, store inquiries, broad operational how-to questions, or when multiple disparate subsystems are mentioned without a dominant root cause.

---

## 5. Taxonomy Overlap & Boundary Disambiguation

### Boundary: `general_device_support` vs `software_update_problem`
- **Observed Disagreements:** 6
- **Ambiguity Source:** Customer reports vague instability or general issue after a software update.
- **Proposed Guideline:** If the message explicitly links problem onset to an update ('after update', 'since ios 11'), assign software_update_problem. If it is a generic query without update causality, assign general_device_support.

### Boundary: `general_device_support` vs `hardware_audio_connection_issue`
- **Observed Disagreements:** 3
- **Ambiguity Source:** Customer asks about cables, adapters, printers, or AirPods.
- **Proposed Guideline:** Any mention of physical ports (USB, Lightning, 3.5mm jack), peripheral cables, or audio accessories routes to hardware_audio_connection_issue.

### Boundary: `display_touch_issue` vs `keyboard_typing_issue`
- **Observed Disagreements:** 2
- **Ambiguity Source:** Visual rendering glitches affecting autocorrect / letter I glitch.
- **Proposed Guideline:** If the glitch specifically affects text input or the letter 'I' box symbol, assign keyboard_typing_issue. If the screen panel itself is flickering, cracked, or unresponsive, assign display_touch_issue.

### Boundary: `mac_software_issue` vs `general_device_support`
- **Observed Disagreements:** 4
- **Ambiguity Source:** Desktop Mac / MacBook troubleshooting vs generic device inquiries.
- **Proposed Guideline:** Assign mac_software_issue when macOS-specific features (High Sierra, Finder, Safari, Time Machine, macOS install) are mentioned.

### Boundary: `battery_power_issue` vs `hardware_audio_connection_issue`
- **Observed Disagreements:** 0
- **Ambiguity Source:** Charging cable / lead failure vs battery depletion.
- **Proposed Guideline:** If the charger or cable is physically defective/broken, assign hardware_audio_connection_issue. If the battery is draining fast or device won't charge/hold power, assign battery_power_issue.

---

## 6. Scientific Sample Size & Statistical Limitations

- PRELIMINARY SAMPLE SIZE: Analysis is based on N=57 of 200 golden records (28.5% coverage).
- NOT GENERALIZED MODEL ACCURACY: Observed agreement rates describe human review decisions against baseline predictions and may shift as annotation progresses.
- SAMPLING VARIANCE: Per-intent precision/recall on low-support categories (< 5 examples) have wide statistical confidence intervals.
- RE-EVALUATION REQUIREMENT: This analysis layer should be re-run at N=100 and N=200 completed human reviews.
