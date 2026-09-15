# Phase 7.1 — Ambiguity Detection Calibration & Safe Decision Gating Report

## 1. Three-Strategy Comparative Evaluation
| Metric | Strategy A (Confidence-Only) | Strategy B (Phase 7 Ambiguity) | Strategy C (Phase 7.1 Multi-Signal Gate) | Delta (C vs B) |
| :--- | :--- | :--- | :--- | :--- |
| **Evaluated Benchmark Records** | `77` | `77` | `77` | — |
| **Auto-Handle Count** | `64` | `65` | `41` | `-24` |
| **Auto-Handle Rate** | `83.1%` | `84.4%` | `53.2%` | `-31.2%` |
| **Auto-Handle Accuracy (Precision)** | `57.8%` | `56.9%` | `63.4%` | `+6.5%` |
| **Unsafe Auto-Handles (Errors)** | `27` | `28` | `15` | `-13` |
| **Escalation Count** | `13` | `12` | `36` | `+24` |
| **Escalation Rate** | `16.9%` | `15.6%` | `46.8%` | `+31.2%` |
| **Error Interception Rate** | `20.6%` | `17.6%` | `55.9%` | `+38.2%` |
| **Correct Cases Escalated** | `6` | `6` | `17` | `+11` |
| **Human Assistance (Top-2)** | `69.2%` | `66.7%` | `66.7%` | `+0.0%` |
| **Human Assistance (Top-3)** | `76.9%` | `75.0%` | `69.4%` | `-5.6%` |

## 2. Signal Interception Effectiveness Analysis
| Clarity Signal | Records Triggered | Errors Intercepted | Correct Cases Escalated | Interception Precision |
| :--- | :--- | :--- | :--- | :--- |
| `SIGNAL_A_INSUFFICIENT_INFO` | `31` | `15` | `16` | `48.4%` |
| `SIGNAL_B_WEAK_PROBLEM` | `29` | `15` | `14` | `51.7%` |
| `SIGNAL_C_GENUINE_CONFLICT` | `4` | `3` | `1` | `75.0%` |
| `SIGNAL_D_EVIDENCE_DISAGREEMENT` | `14` | `7` | `7` | `50.0%` |
| `SIGNAL_F_MULTI_SYMPTOM_COMPLEXITY` | `1` | `1` | `0` | `100.0%` |
| `SAFETY_VETO_COMBINED` | `32` | `17` | `15` | `53.1%` |

## 3. Remaining Unsafe Auto-Handle Error Categorization
Total remaining unsafe auto-handles: `15`

| Golden ID | Predicted vs True Intent | Error Category | Explanation |
| :--- | :--- | :--- | :--- |
| `gold_068` | `display_touch_issue` vs `general_device_support` | `FALSE_CLARITY_DECISION` | Multi-signal gate passed with high confidence but model misclassified 'general_device_support' as 'display_touch_issue'. |
| `gold_137` | `display_touch_issue` vs `general_device_support` | `FALSE_CLARITY_DECISION` | Multi-signal gate passed with high confidence but model misclassified 'general_device_support' as 'display_touch_issue'. |
| `gold_144` | `display_touch_issue` vs `general_device_support` | `MODEL_CLASSIFICATION_ERROR` | Device/service was Mac/Safari, but classifier predicted general iOS/device support. |
| `gold_154` | `general_device_support` vs `battery_power_issue` | `RETRIEVAL_MISMATCH` | Historical evidence cases did not contain an operational match to support the prediction. |
| `gold_005` | `display_touch_issue` vs `general_device_support` | `FALSE_CLARITY_DECISION` | Multi-signal gate passed with high confidence but model misclassified 'general_device_support' as 'display_touch_issue'. |
| `gold_014` | `keyboard_typing_issue` vs `general_device_support` | `FALSE_CLARITY_DECISION` | Multi-signal gate passed with high confidence but model misclassified 'general_device_support' as 'keyboard_typing_issue'. |
| `gold_082` | `keyboard_typing_issue` vs `software_update_problem` | `MULTIPLE_VALID_INTERPRETATIONS` | Inquiry mentions software update trigger while primary ground truth intent is 'software_update_problem'. |
| `gold_098` | `display_touch_issue` vs `general_device_support` | `FALSE_CLARITY_DECISION` | Multi-signal gate passed with high confidence but model misclassified 'general_device_support' as 'display_touch_issue'. |
| `gold_103` | `general_device_support` vs `display_touch_issue` | `FALSE_CLARITY_DECISION` | Multi-signal gate passed with high confidence but model misclassified 'display_touch_issue' as 'general_device_support'. |
| `gold_106` | `general_device_support` vs `hardware_audio_connection_issue` | `RETRIEVAL_MISMATCH` | Historical evidence cases did not contain an operational match to support the prediction. |
| `gold_109` | `general_device_support` vs `software_update_problem` | `RETRIEVAL_MISMATCH` | Historical evidence cases did not contain an operational match to support the prediction. |
| `gold_125` | `display_touch_issue` vs `general_device_support` | `FALSE_CLARITY_DECISION` | Multi-signal gate passed with high confidence but model misclassified 'general_device_support' as 'display_touch_issue'. |
| `gold_177` | `general_device_support` vs `hardware_audio_connection_issue` | `RETRIEVAL_MISMATCH` | Historical evidence cases did not contain an operational match to support the prediction. |
| `gold_190` | `billing_purchase_issue` vs `battery_power_issue` | `RETRIEVAL_MISMATCH` | Historical evidence cases did not contain an operational match to support the prediction. |
| `gold_198` | `keyboard_typing_issue` vs `hardware_audio_connection_issue` | `MODEL_CLASSIFICATION_ERROR` | Device/service was Mac/Safari, but classifier predicted general iOS/device support. |
