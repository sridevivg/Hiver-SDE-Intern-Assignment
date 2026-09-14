# SupportGraph AI — Phase 5.9 Taxonomy Calibration & Recommendation Report

**Generated:** 2026-09-14 11:58:11 UTC  
**Dataset:** 200 Golden Records (57 Human Reviewed, 143 Pending)  
**Baseline AI Agreement:** 33.3% (19/57 records)  

---

## 1. Executive Summary

The Phase 5.9 Taxonomy Calibration Layer extracts explainable contextual signals from the completed human ground truth ($N=57$) without training a machine learning model on the golden dataset. It provides transparent advisory guidance for pending records, escalates high-risk discrepancies to mandatory individual review, and identifies safe low-risk candidates for group review.

| Metric | Reviewed Ground Truth (N=57) | Pending Queue (N=143) | Total Dataset (N=200) |
| :--- | :--- | :--- | :--- |
| **Total Records** | 57 | 143 | 200 |
| **Agrees with AI Suggestion** | 19 (33.3%) | 24 (16.8%) | 43 (21.5%) |
| **Disagrees with AI Suggestion** | 38 (66.7%) | 8 (5.6%) | 46 (23.0%) |
| **Insufficient Evidence / Neutral** | 0 (Evaluated) | 111 (77.6%) | 111 (55.5%) |
| **Ambiguous Signals** | 0 (Evaluated) | 0 (0.0%) | 0 (0.0%) |

---

## 2. Empirical Ground Truth Distribution (N=57)

Completed human ground truth records are immutable and protected:

| Taxonomy Intent Label | Human Ground Truth Count | Percentage |
| :--- | :--- | :--- |
| `general_device_support` | 18 | 31.6% |
| `software_update_problem` | 11 | 19.3% |
| `keyboard_typing_issue` | 8 | 14.0% |
| `hardware_audio_connection_issue` | 6 | 10.5% |
| `mac_software_issue` | 4 | 7.0% |
| `battery_power_issue` | 4 | 7.0% |
| `unclear_needs_review` | 3 | 5.3% |
| `display_touch_issue` | 3 | 5.3% |

---

## 3. Disagreement Pattern Analysis (AI vs Human Ground Truth)

Out of 57 reviewed records, the human annotator overrode the baseline AI suggestion in **38 cases (66.7%)**.

| Baseline AI Suggestion | Human True Ground Truth | Occurrences | Root Cause & Contextual Signal |
| :--- | :--- | :--- | :--- |
| `general_device_support` | `software_update_problem` | 6 | Update causality / regression phrasing (*'since update'*, *'after updating'*). |
| `general_device_support` | `mac_software_issue` | 4 | macOS desktop / laptop software (*'High Sierra'*, *'MacBook boot'*). |
| `general_device_support` | `hardware_audio_connection_issue` | 3 | Peripherals / cables / audio accessories (*'wired printer USB'*, *'AirPods no sound'*). |
| `general_device_support` | `battery_power_issue` | 3 | Taxonomy category specificity / operational context. |
| `display_touch_issue` | `hardware_audio_connection_issue` | 2 | Taxonomy category specificity / operational context. |
| `account_access_issue` | `general_device_support` | 2 | Taxonomy category specificity / operational context. |
| `general_device_support` | `keyboard_typing_issue` | 2 | Text input / autocorrect / letter I glitch (*'autocorrect bug'*, *'letter eye'*). |
| `display_touch_issue` | `keyboard_typing_issue` | 2 | Taxonomy category specificity / operational context. |
| `display_touch_issue` | `general_device_support` | 2 | Taxonomy category specificity / operational context. |
| `mac_software_issue` | `keyboard_typing_issue` | 2 | Taxonomy category specificity / operational context. |
| `general_device_support` | `display_touch_issue` | 2 | Screen rendering / touch unresponsiveness (*'screen flickering'*, *'black screen'*). |
| `keyboard_typing_issue` | `software_update_problem` | 1 | Taxonomy category specificity / operational context. |
| `mac_software_issue` | `unclear_needs_review` | 1 | Taxonomy category specificity / operational context. |
| `unclear_needs_review` | `keyboard_typing_issue` | 1 | Taxonomy category specificity / operational context. |
| `general_device_support` | `unclear_needs_review` | 1 | Taxonomy category specificity / operational context. |
| `mac_software_issue` | `hardware_audio_connection_issue` | 1 | Taxonomy category specificity / operational context. |
| `display_touch_issue` | `software_update_problem` | 1 | Taxonomy category specificity / operational context. |
| `billing_purchase_issue` | `general_device_support` | 1 | Taxonomy category specificity / operational context. |

---

## 4. Transparent Contextual Calibration Rules

The calibration engine uses 10 transparent, deterministic rule-sets corresponding to the approved taxonomy categories:

1. **`software_update_problem`**: Matches update regression phrasing (`since update`, `after updating ios`), installation stalls (`update stuck`), and release version regressions (`ios 11.1`).
2. **`hardware_audio_connection_issue`**: Matches audio peripherals (`airpods`, `earphones`, `microphone`, `headphone jack`) and physical connectivity (`usb cable`, `lightning port`, `wired printer`).
3. **`keyboard_typing_issue`**: Matches predictive text (`autocorrect`, `predictive text`), letter glitches (`letter i`, `letter eye`, `unicode glitch`), and keyboard unresponsiveness.
4. **`battery_power_issue`**: Matches battery depletion (`battery draining fast`, `battery percentage drop`), charging failures (`not charging`, `overheating`), and unexpected shutdowns.
5. **`mac_software_issue`**: Matches macOS operating system (`high sierra`, `macos`, `kernel panic`), desktop applications (`finder freeze`, `safari crash`), and recovery utilities.
6. **`display_touch_issue`**: Matches screen unresponsiveness (`touchscreen unresponsive`, `ghost touch`), visual glitches (`screen flicker`, `black display`), and display damage.
7. **`account_access_issue`**: Matches Apple ID credentials (`apple id locked`, `forgot password`, `two factor authentication`, `security questions`).
8. **`billing_purchase_issue`**: Matches subscriptions (`unauthorized charge`, `itunes subscription`), refund requests (`refund`), and payment method errors (`card declined`).
9. **`general_device_support`**: Matches general inquiry when no specific operational subsystem or update causality is identified.
10. **`unclear_needs_review`**: Flags ambiguous multi-domain conflicts, severe message truncations, or foreign language texts.

---

## 5. Pending Queue Calibration Breakdown (N=143)

Evaluating the 143 pending records with the calibration layer yields the following distribution:

| Calibration Status | Count | Percentage | Workflow Action |
| :--- | :--- | :--- | :--- |
| **`AGREES_WITH_AI`** | 24 | 16.8% | Eligible for Safe Group Review (if low-risk & non-QC) |
| **`DISAGREES_WITH_AI`** | 8 | 5.6% | Escalated to Mandatory Individual Review with `[C]` recommendation |
| **`INSUFFICIENT_EVIDENCE`** | 111 | 77.6% | Standard priority scoring based on baseline AI confidence |
| **`AMBIGUOUS`** | 0 | 0.0% | Escalated to Individual Review due to multi-intent overlap |

### Calibrated Candidate Breakdown for Pending Records

| Calibrated Candidate Intent | Pending Record Count |
| :--- | :--- |
| `battery_power_issue` | 15 |
| `keyboard_typing_issue` | 7 |
| `account_access_issue` | 3 |
| `mac_software_issue` | 3 |
| `hardware_audio_connection_issue` | 2 |
| `software_update_problem` | 1 |
| `billing_purchase_issue` | 1 |

---

## 6. Scientific Integrity Guarantees

1. **Zero Ground-Truth Contamination**: Calibration candidates are stored in distinct metadata columns (`calibration_candidate_label`, `calibration_confidence`, `calibration_status`) and are NEVER automatically copied into `annotation_label`.
2. **No Model Training on Test Data**: No fine-tuning, embedding fitting, or machine learning training was performed on the golden dataset.
3. **Explicit Human Action**: Calibrated suggestions can only become ground-truth annotations through explicit human review actions (`[C]` action in interactive CLI).
4. **Full Immutability**: All 57 completed human annotations remain 100% byte-for-byte identical.
