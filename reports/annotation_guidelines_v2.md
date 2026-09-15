# SupportGraph AI — Operational Taxonomy & Annotation Guidelines (v2.0)

> **Human-in-the-Loop Annotation Standard Operating Procedure (SOP)**  
> **Version:** 2.0 (Phase 5.11 Evidence-Based Calibration)  
> **Dataset Baseline:** AppleSupport Customer Conversations ($N=200$ Golden Benchmark)  
> **Scientific Mandate:** Grounded in observed empirical review patterns ($N=77$ human reviews; 31.2% observed AI agreement; 137 evaluated consistency boundaries).

---

## 1. Operational Intent Taxonomy Overview

SupportGraph AI uses a 10-category operational intent taxonomy designed to support deterministic routing, retrieval-augmented resolution retrieval, and escalation gating.

| Intent Category | Scope & Core Definition | Prototypical Customer Needs |
|---|---|---|
| `account_access_issue` | Authentication, credentials, Apple ID recovery, two-factor lockouts, iCloud sign-in barriers. | *"Locked out of Apple ID"*, *"Verification code not sent"*, *"Forgot iCloud password"*. |
| `battery_power_issue` | Power depletion, charging failure, battery degradation, overheating during charging, refusal to power on. | *"Battery drains in 2 hours"*, *"Phone won't charge past 80%"*, *"Shuts off at 20%"*. |
| `billing_purchase_issue` | Financial transactions, App Store charges, unauthorized purchases, subscription cancellations, refund disputes. | *"Charged twice for Apple Music"*, *"Refund for in-app purchase"*, *"Unexpected iTunes receipt"*. |
| `display_touch_issue` | Physical screen defects, touchscreen unresponsiveness, digitizer ghost touches, OLED/LCD flickering, backlight failure. | *"Screen unresponsive to touch"*, *"Display flickering green"*, *"Cracked screen touch deadzone"*. |
| `general_device_support` | **Strict Fallback / Triage:** Diffuse inquiries, multi-issue confusion, broad UI settings triage, generic feature advice without specific subsystem failure. | *"What is best antivirus for Mac?"*, *"How do I set the clock?"*, *"Device feels sluggish overall"*. |
| `hardware_audio_connection_issue` | Audio input/output (speakers, mics, AirPods, EarPods), Bluetooth pairing, Wi-Fi hardware adapter toggles, physical ports/cables. | *"No sound from speaker"*, *"AirPods won't connect"*, *"Wi-Fi switch greyed out in Settings"*, *"USB-C adapter compatibility"*. |
| `keyboard_typing_issue` | Text input, software keyboard rendering, predictive text, autocorrect glitches, hardware keyboard key sticking. | *"Letter 'I' turns into [?] symbol"*, *"Autocorrect changing normal words"*, *"Spacebar not working on laptop"*. |
| `mac_software_issue` | Desktop macOS operating system, macOS installer errors, Time Machine, desktop Safari, Finder, macOS kernel panics. | *"MacBook spinning beach ball"*, *"High Sierra installer stuck"*, *"Kernel panic on boot"*. |
| `software_update_problem` | iOS/macOS active upgrade failures, update installation loops, OTA download errors, direct OS version regressions. | *"iOS 11 update bricked my phone"*, *"Update stuck on Apple logo"*, *"Cannot download 11.1 patch"*. |
| `unclear_needs_review` | Truncated tweets, foreign language without context, unparseable slang, non-actionable expressions lacking support inquiry. | *"A [?]"*, *"asdfghjk"*, *"Non-English without translation"*. |

---

## 2. Core Annotation Principles

### 2.1 The Primary Operational Friction Principle
When a customer message describes multiple symptoms or mentions several technologies, annotators must label the record based on **the primary operational problem requiring support remediation**, rather than matching superficial keywords.

- **Example:** *"Ever since I updated to iOS 11, my phone speaker has no sound."*
  - Mentions: `update`, `iOS 11`, `speaker`, `sound`.
  - Primary Friction: Audio speaker failure.
  - Correct Label: `hardware_audio_connection_issue` (audio remediation path).

### 2.2 Strict Ground-Truth Immutability & AI Advisory Stance
- AI suggestions (`model_suggested_label`) and calibrated hints (`calibration_candidate_label`) are **strictly advisory cognitive aids**.
- Human annotators must exercise critical domain judgment. Overriding an AI suggestion is expected whenever the model falls back to a generic label or misses the focal operational defect.

---

## 3. Detailed Category Boundary Guidelines

### 3.1 `general_device_support` — Strict Fallback Boundary
Empirical audit in Phase 5.9 demonstrated that baseline LLMs over-predict `general_device_support` (63.4% human override rate), using it as a catch-all category.

- **WHEN TO USE:**
  - The customer's inquiry represents diffuse, non-specific device behavior (e.g., *"My device is acting weird today"*).
  - General settings navigation or configuration advice without hardware or subsystem malfunction.
  - Multi-issue general triage where no single operational intent dominates.
- **WHEN NOT TO USE:**
  - An explicit audio, keyboard, battery, display, update, billing, or account symptom is present.
  - The customer mentions an OS update as the causal trigger of an identifiable defect.
  - A macOS platform-specific troubleshooting workflow is required.

---

### 3.2 `software_update_problem` vs Symptom-Specific Categories
A common boundary ambiguity occurs when an OS update causes a concrete symptom.

| Customer Message Example | Dominant Focus | Recommended Intent | Rationale |
|---|---|---|---|
| *"iOS 11.1 update download is stuck at 99% and won't finish"* | Upgrade Process | `software_update_problem` | Customer cannot complete the OS installation. |
| *"Since updating to iOS 11 my battery drains from 100% to 0% in 1 hour"* | Battery Degradation | `battery_power_issue` | Diagnostic remediation involves battery health & power optimization. |
| *"After updating my phone the keyboard autocorrect turns letter I into symbols"* | Keystroke / Text Input | `keyboard_typing_issue` | Remediation involves keyboard dictionary reset / text replacement. |
| *"iOS 11 is completely broken and full of bugs, roll me back"* | General OS Release | `software_update_problem` | Broad dissatisfaction with the OS version release itself. |

---

### 3.3 `display_touch_issue` vs `general_device_support` & UI Software Features
The presence of the word *"screen"* does not automatically imply a hardware display defect.

- **Display/Touch Issue Evidence:**
  - Physical digitizer deadzones, screen unresponsive to taps/swipes.
  - Visual display hardware artifacts (green lines, backlight failure, screen flickering).
- **Non-Display Issues Involving "Screen":**
  - *"Podcast app home screen layout is confusing"* $\rightarrow$ `general_device_support`.
  - *"Screen recording feature not saving video"* $\rightarrow$ `general_device_support` or `mac_software_issue`.
  - *"App crashes when opened on home screen"* $\rightarrow$ `general_device_support`.

---

### 3.4 `billing_purchase_issue` vs General App Store Support
Mentions of the App Store do not automatically indicate a financial billing dispute.

- **Billing/Purchase Evidence (Required):**
  - Charges, receipts, double billing, unexpected payment deductions.
  - In-app purchase failures, subscription cancellation requests, refund claims.
- **General App Store Triage (Non-Billing):**
  - *"App Store search is not loading"* $\rightarrow$ `general_device_support`.
  - *"Why aren't my iOS app updates automatic in App Store?"* $\rightarrow$ `software_update_problem` or `general_device_support`.
  - *"Cannot sign in to App Store"* $\rightarrow$ `account_access_issue`.

---

### 3.5 `hardware_audio_connection_issue` vs `battery_power_issue` & General Accessories
- **Audio / Wireless Connectivity:**
  - Bluetooth pairing failures, car audio sync, AirPods dropouts.
  - Wi-Fi hardware toggle greyed out in Settings.
  - USB-C / Lightning adapter, headphone jack, or printer hardware connection.
- **Battery / Power:**
  - Device refuses to turn on even when charged.
  - Charger cable failure or battery indicator stuck.

---

## 4. Summary Decision Flowchart for Annotators

```text
[Incoming Customer Message]
  │
  ├─► Is the text unparseable, truncated, or non-English without context? ──► [unclear_needs_review]
  │
  ├─► Does the issue involve passwords, Apple ID lockouts, 2FA, or iCloud auth? ──► [account_access_issue]
  │
  ├─► Does the issue involve payments, charges, subscriptions, or refunds? ──► [billing_purchase_issue]
  │
  ├─► Is the primary defect keyboard typing, autocorrect, or character glyphs? ──► [keyboard_typing_issue]
  │
  ├─► Is the primary defect physical screen digitizer, touch, or display panel? ──► [display_touch_issue]
  │
  ├─► Is the primary defect audio sound, Bluetooth, Wi-Fi hardware, or cables? ──► [hardware_audio_connection_issue]
  │
  ├─► Is the primary defect battery drain, charging failure, or power cut? ──► [battery_power_issue]
  │
  ├─► Is the issue desktop macOS-specific (MacBook, High Sierra installer, Mac Finder)? ──► [mac_software_issue]
  │
  ├─► Is the issue an active OS update download/install failure or broad OS release complaint? ──► [software_update_problem]
  │
  └─► None of the above / diffuse non-specific general triage? ──► [general_device_support]
```
