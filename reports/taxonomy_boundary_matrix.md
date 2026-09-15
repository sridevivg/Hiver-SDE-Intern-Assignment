# SupportGraph AI — Taxonomy Boundary Decision Matrix

> **Reference Matrix for Intent Category Disambiguation**  
> Grounded in empirical boundary analysis across $N=77$ human-reviewed golden records ($2,926$ evaluated pairs).

---

## 1. Primary Boundary Decision Matrix

| Category A | Category B | Primary Decision Rule | Distinguishing Positive Signals for Category A | Distinguishing Positive Signals for Category B | Boundary Case Example |
|---|---|---|---|---|---|
| `general_device_support` | `software_update_problem` | Is an active OS installation/upgrade process or version release the primary complaint? | Settings questions, generic antivirus/sluggishness, feature advice without update causality. | Active upgrade failure, installation loop, "after iOS 11 update", "iOS 11 bugs". | **[gold_128] vs [gold_040]:** *"hey can u fix IOS"* (`general_device_support`) vs *"Please fix all your fucking issues with iOS 11"* (`software_update_problem`). |
| `general_device_support` | `hardware_audio_connection_issue` | Is physical audio, Bluetooth, or wireless hardware adapter the focal malfunction? | UI control queries, auto-brightness, generic settings navigation. | Speaker no sound, Bluetooth pairing failure, Wi-Fi greyed out, USB-C adapter cable. | **[gold_089] vs [gold_106]:** *"why is my wifi on AGAIN"* (`general_device_support`) vs *"i can’t turn on my WiFi on my iPhone 7"* (`hardware_audio_connection_issue`). |
| `general_device_support` | `battery_power_issue` | Is power drain, charging, or unexpected shutdown the central operational symptom? | Broad device dissatisfaction, multi-issue triage. | Rapid battery depletion (e.g. 2 hrs), won't charge, shut down at 30%, charging cable. | **[gold_093] vs [gold_190]:** *"my 2016 MacBook won’t turn on"* (`general_device_support`) vs *"iPhone 7 won’t turn on even though charged"* (`battery_power_issue`). |
| `general_device_support` | `mac_software_issue` | Does remediation require macOS desktop software troubleshooting (Finder, desktop Safari, Time Machine)? | Cross-device questions, general Apple service referrals. | MacBook spinning beach ball, High Sierra download loop, desktop macOS kernel panic. | **[gold_052] vs [gold_086]:** *"best antivirus for MacBook Pro"* (`general_device_support`) vs *"MacBook Air has the dreaded spinning beach ball"* (`mac_software_issue`). |
| `display_touch_issue` | `keyboard_typing_issue` | Is the root defect the physical touchscreen digitizer or software text/keystroke processing? | Unresponsive screen, touch deadzone, display flickering, cracked screen touch failure. | Autocorrect corruption, letter "I" glyph glitch, keyboard lag, sticking spacebar. | **[gold_103] vs [gold_084]:** *"phone crashes touch stops working"* (`display_touch_issue`) vs *"MacOS reinstalled but random key stopped working"* (`keyboard_typing_issue`). |
| `software_update_problem` | `battery_power_issue` | Is the primary support friction the OS installation failure or post-update battery drain? | Upgrade download stuck, verification loop, roll-back request. | Battery depletion, battery health percentage drop post-upgrade. | **[gold_025] vs [gold_180]:** *"actualizacion de errores iOS 11"* (`software_update_problem`) vs *"battery and velocity of iPhone 6 sucks on iOS 11"* (`battery_power_issue`). |
| `software_update_problem` | `hardware_audio_connection_issue` | Is the issue a general update failure or audio/connectivity loss following an update? | Upgrade verification failure, broad OS version dissatisfaction. | Specific loss of sound, mic failure, or Bluetooth drop after OS update. | **[gold_102] vs [gold_046]:** *"fix the music app ever since update"* (`software_update_problem`) vs *"music app is not producing sound"* (`hardware_audio_connection_issue`). |
| `billing_purchase_issue` | `general_device_support` | Is there explicit evidence of financial transactions, subscriptions, or unauthorized charges? | Subscription cancellation, double charge, refund dispute, receipt discrepancy. | App Store search not loading, app layout feedback, non-billing app triage. | **[gold_125]:** *"podcast app layout feedback"* (`general_device_support`) vs billing charge dispute (`billing_purchase_issue`). |

---

## 2. Multi-Symptom Disambiguation Rules

```text
Rule 1 (Specificity Precedence):
  Specific Subsystem Intent > general_device_support
  If a concrete operational subsystem (audio, battery, keyboard, display, macOS) is identifiable,
  do NOT use general_device_support.

Rule 2 (Symptom Over Entity):
  Physical Fault > Device Mention
  A MacBook with audio speaker failure is hardware_audio_connection_issue, NOT mac_software_issue.

Rule 3 (Update Causality):
  If an update triggers a concrete hardware/subsystem fault (e.g. battery drain),
  classify by the presenting operational defect (battery_power_issue).
  Reserve software_update_problem for active upgrade installation barriers or broad OS version release defects.

Rule 4 (Financial Evidence Requirement):
  billing_purchase_issue requires explicit payment, fee, subscription, charge, or refund context.
```
