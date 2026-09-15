# SupportGraph AI — Phase 5.9 Annotation Pattern Analysis & Taxonomy Calibration Report

**Dataset Coverage:** 77 / 200 Records (38.5%)  
**Observed Annotation Agreement:** 31.2% (95% CI: [21.9%, 42.2%])  
**Human Overrides:** 53 (68.8%)  

> [!IMPORTANT]
> **Scientific Integrity Disclaimer:** OBSERVED ANNOTATION AGREEMENT on N=77 completed records (38.5% coverage). This reflects annotator agreement with baseline LLM suggestions during curation and must NOT be interpreted as generalized production model accuracy.

---

## 1. Executive Summary & Agreement Statistics

| Metric | Value | Description |
| :--- | :--- | :--- |
| **Total Golden Records** | `200` | Target evaluation benchmark size |
| **Completed Human Reviews** | `77` | Authoritative ground-truth annotations |
| **Pending Human Reviews** | `123` | Unreviewed records remaining in queue |
| **Agreed AI Suggestions** | `24` | Human confirmed baseline AI suggestion |
| **Overridden AI Suggestions** | `53` | Human assigned a different taxonomy label |
| **Marked 'unclear_needs_review'** | `5` | Records requiring special investigation |
| **Observed Agreement Rate** | `31.2%` | Preliminary agreement on reviewed sample |

---

## 2. Confusion Matrix (AI Baseline vs Human Ground Truth)

Rows represent **Baseline AI Suggestions**; columns represent **Human True Annotations**.

| Baseline AI Suggestion \ Human GT | `account_access_issue` | `battery_power_issue` | `billing_purchase_issue` | `display_touch_issue` | `general_device_support` | `hardware_audio_connection_issue` | `keyboard_typing_issue` | `mac_software_issue` | `software_update_problem` | `unclear_needs_review` | **Total** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `account_access_issue` | - | - | 1 | - | 2 | 1 | - | - | 1 | - | **5** |
| `battery_power_issue` | - | 1 | - | - | - | - | - | - | - | - | **1** |
| `billing_purchase_issue` | - | - | - | - | 3 | - | - | - | - | - | **3** |
| `display_touch_issue` | - | - | - | 1 | 5 | 2 | 2 | - | 1 | - | **11** |
| `general_device_support` | - | 4 | - | 2 | 15 | 3 | 3 | 5 | 6 | 3 | **41** |
| `hardware_audio_connection_issue` | - | 2 | - | - | 1 | - | - | - | - | - | **3** |
| `keyboard_typing_issue` | - | - | - | - | - | - | 2 | - | 1 | - | **3** |
| `mac_software_issue` | - | - | - | - | - | 1 | 2 | - | - | 1 | **4** |
| `software_update_problem` | - | - | - | - | - | - | - | - | 4 | - | **4** |
| `unclear_needs_review` | - | - | - | - | - | - | 1 | - | - | 1 | **2** |
| **Total Ground Truth** | **0** | **7** | **1** | **3** | **26** | **7** | **10** | **5** | **13** | **5** | **77** |

### Per-Intent Performance Breakdown

| Intent Label | Ground Truth Support | AI Predicted Support | Agreed | Observed Precision | Observed Recall | Observed F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `general_device_support` | 26 | 41 | 15 | 36.6% | 57.7% | 0.448 |
| `software_update_problem` | 13 | 4 | 4 | 100.0% | 30.8% | 0.471 |
| `keyboard_typing_issue` | 10 | 3 | 2 | 66.7% | 20.0% | 0.308 |
| `battery_power_issue` | 7 | 1 | 1 | 100.0% | 14.3% | 0.250 |
| `hardware_audio_connection_issue` | 7 | 3 | 0 | 0.0% | 0.0% | 0.000 |
| `mac_software_issue` | 5 | 4 | 0 | 0.0% | 0.0% | 0.000 |
| `unclear_needs_review` | 5 | 2 | 1 | 50.0% | 20.0% | 0.286 |
| `display_touch_issue` | 3 | 11 | 1 | 9.1% | 33.3% | 0.143 |
| `billing_purchase_issue` | 1 | 3 | 0 | 0.0% | 0.0% | 0.000 |
| `account_access_issue` | 0 | 5 | 0 | 0.0% | 0.0% | 0.000 |

---

## 3. Top AI → Human Correction Patterns

The following table itemizes the most frequent systematic correction pairs observed in the human review data:

| Baseline AI Suggestion | Human Ground Truth | Count | % Overrides | Recurring Semantic Signals | Actionable Recommendation |
| :--- | :--- | :---: | :---: | :--- | :--- |
| `general_device_support` | `software_update_problem` | **6** | 11.3% | 'ios' (5), '115858' (4), 'new' (3) | Enforce strict priority rules routing messages with 'software_update_problem' operational symptoms away from general fallback 'general_device_support'. |
| `display_touch_issue` | `general_device_support` | **5** | 9.4% | 'screen' (2), 'going' (2), 'recording' (1) | Retain 'general_device_support' only when no specific subsystem, hardware component, or update causality is mentioned. |
| `general_device_support` | `mac_software_issue` | **5** | 9.4% | 'high' (5), 'sierra' (5), 'high sierra' (5) | Enforce strict priority rules routing messages with 'mac_software_issue' operational symptoms away from general fallback 'general_device_support'. |
| `general_device_support` | `battery_power_issue` | **4** | 7.5% | 'now' (2), 'can' (2), 'working' (2) | Enforce strict priority rules routing messages with 'battery_power_issue' operational symptoms away from general fallback 'general_device_support'. |
| `billing_purchase_issue` | `general_device_support` | **3** | 5.7% | 'app' (2), 'store' (2), 'working' (2) | Retain 'general_device_support' only when no specific subsystem, hardware component, or update causality is mentioned. |
| `general_device_support` | `hardware_audio_connection_issue` | **3** | 5.7% | 'usb' (3), 'type' (2), 'don' (1) | Enforce strict priority rules routing messages with 'hardware_audio_connection_issue' operational symptoms away from general fallback 'general_device_support'. |
| `general_device_support` | `keyboard_typing_issue` | **3** | 5.7% | 'fix' (2), '115858' (2), 'fix this' (2) | Enforce strict priority rules routing messages with 'keyboard_typing_issue' operational symptoms away from general fallback 'general_device_support'. |
| `general_device_support` | `unclear_needs_review` | **3** | 5.7% | 'anyone' (1), 'else' (1), 'shit' (1) | Enforce strict priority rules routing messages with 'unclear_needs_review' operational symptoms away from general fallback 'general_device_support'. |
| `account_access_issue` | `general_device_support` | **2** | 3.8% | '115858' (2), 'siri' (1), 'hate' (1) | Retain 'general_device_support' only when no specific subsystem, hardware component, or update causality is mentioned. |
| `display_touch_issue` | `hardware_audio_connection_issue` | **2** | 3.8% | 'sound' (3), 'music' (2), 'can' (2) | Clarify decision boundary between 'display_touch_issue' and 'hardware_audio_connection_issue' in annotation guidelines. |
| `display_touch_issue` | `keyboard_typing_issue` | **2** | 3.8% | 'fix' (2), 'seeing' (2), 'fix this' (2) | Clarify decision boundary between 'display_touch_issue' and 'keyboard_typing_issue' in annotation guidelines. |
| `general_device_support` | `display_touch_issue` | **2** | 3.8% | 'screen' (2), '115858' (1), 'love' (1) | Enforce strict priority rules routing messages with 'display_touch_issue' operational symptoms away from general fallback 'general_device_support'. |
| `hardware_audio_connection_issue` | `battery_power_issue` | **2** | 3.8% | 'charger' (2), '115858' (1), 'broke' (1) | Clarify decision boundary between 'hardware_audio_connection_issue' and 'battery_power_issue' in annotation guidelines. |
| `mac_software_issue` | `keyboard_typing_issue` | **2** | 3.8% | 'apple' (2), 'issue' (2), 'formated' (1) | Clarify decision boundary between 'mac_software_issue' and 'keyboard_typing_issue' in annotation guidelines. |
| `account_access_issue` | `billing_purchase_issue` | **1** | 1.9% | 'want' (1), 'trade' (1), 'iphone' (1) | Clarify decision boundary between 'account_access_issue' and 'billing_purchase_issue' in annotation guidelines. |
| `account_access_issue` | `hardware_audio_connection_issue` | **1** | 1.9% | 'can' (1), 'find' (1), 'airpods' (1) | Clarify decision boundary between 'account_access_issue' and 'hardware_audio_connection_issue' in annotation guidelines. |
| `account_access_issue` | `software_update_problem` | **1** | 1.9% | 'latest' (1), 'ios' (1), 'update' (1) | Clarify decision boundary between 'account_access_issue' and 'software_update_problem' in annotation guidelines. |
| `display_touch_issue` | `software_update_problem` | **1** | 1.9% | 'can' (2), 'music' (2), 'app' (2) | Clarify decision boundary between 'display_touch_issue' and 'software_update_problem' in annotation guidelines. |
| `hardware_audio_connection_issue` | `general_device_support` | **1** | 1.9% | 'turn' (2), 'wifi' (1), 'manually' (1) | Retain 'general_device_support' only when no specific subsystem, hardware component, or update causality is mentioned. |
| `keyboard_typing_issue` | `software_update_problem` | **1** | 1.9% | 'horrible' (2), 'please' (1), 'heaven' (1) | Clarify decision boundary between 'keyboard_typing_issue' and 'software_update_problem' in annotation guidelines. |
| `mac_software_issue` | `hardware_audio_connection_issue` | **1** | 1.9% | 'und' (1), 'ich' (1), 'dachte' (1) | Clarify decision boundary between 'mac_software_issue' and 'hardware_audio_connection_issue' in annotation guidelines. |
| `mac_software_issue` | `unclear_needs_review` | **1** | 1.9% | 'works' (1), 'phone' (1), 'macbook' (1) | Clarify decision boundary between 'mac_software_issue' and 'unclear_needs_review' in annotation guidelines. |
| `unclear_needs_review` | `keyboard_typing_issue` | **1** | 1.9% | '115858' (1), 'letter' (1), 'pls' (1) | Clarify decision boundary between 'unclear_needs_review' and 'keyboard_typing_issue' in annotation guidelines. |

---

## 4. `general_device_support` Diagnostic Audit

**Audit Verdict:** OVER-USED AS CATCH-ALL: AI over-predicts general_device_support (63.4% override rate). Human annotators consistently reclassify into concrete operational intents.

- **AI Predictions:** 41 (53.2% of reviewed sample)
- **Human Confirmed:** 15 (36.6%)
- **Human Overridden:** 26 (63.4%)

### Primary Reclassification Destinations for `general_device_support`

| Target Intent | Reclassification Count | Share of Overrides |
| :--- | :---: | :---: |
| `software_update_problem` | 6 | 23.1% |
| `mac_software_issue` | 5 | 19.2% |
| `battery_power_issue` | 4 | 15.4% |
| `unclear_needs_review` | 3 | 11.5% |
| `keyboard_typing_issue` | 3 | 11.5% |
| `hardware_audio_connection_issue` | 3 | 11.5% |
| `display_touch_issue` | 2 | 7.7% |

### Distinguishing Semantic Characteristics
- **Retained in `general_device_support`:** `115858`, `phone`, `screen`, `macbook`, `can`
- **Overridden into Specific Intents:** `115858`, `fix`, `iphone`, `ios`, `high`, `sierra`, `please`, `high sierra`

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
- **Observed Disagreements:** 4
- **Ambiguity Source:** Customer asks about cables, adapters, printers, or AirPods.
- **Proposed Guideline:** Any mention of physical ports (USB, Lightning, 3.5mm jack), peripheral cables, or audio accessories routes to hardware_audio_connection_issue.

### Boundary: `display_touch_issue` vs `keyboard_typing_issue`
- **Observed Disagreements:** 2
- **Ambiguity Source:** Visual rendering glitches affecting autocorrect / letter I glitch.
- **Proposed Guideline:** If the glitch specifically affects text input or the letter 'I' box symbol, assign keyboard_typing_issue. If the screen panel itself is flickering, cracked, or unresponsive, assign display_touch_issue.

### Boundary: `mac_software_issue` vs `general_device_support`
- **Observed Disagreements:** 5
- **Ambiguity Source:** Desktop Mac / MacBook troubleshooting vs generic device inquiries.
- **Proposed Guideline:** Assign mac_software_issue when macOS-specific features (High Sierra, Finder, Safari, Time Machine, macOS install) are mentioned.

### Boundary: `battery_power_issue` vs `hardware_audio_connection_issue`
- **Observed Disagreements:** 2
- **Ambiguity Source:** Charging cable / lead failure vs battery depletion.
- **Proposed Guideline:** If the charger or cable is physically defective/broken, assign hardware_audio_connection_issue. If the battery is draining fast or device won't charge/hold power, assign battery_power_issue.

---

## 6. Scientific Sample Size & Statistical Limitations

- PRELIMINARY SAMPLE SIZE: Analysis is based on N=77 of 200 golden records (38.5% coverage).
- NOT GENERALIZED MODEL ACCURACY: Observed agreement rates describe human review decisions against baseline predictions and may shift as annotation progresses.
- SAMPLING VARIANCE: Per-intent precision/recall on low-support categories (< 5 examples) have wide statistical confidence intervals.
- RE-EVALUATION REQUIREMENT: This analysis layer should be re-run at N=100 and N=200 completed human reviews.
