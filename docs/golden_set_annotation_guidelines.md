# SupportGraph AI — Golden Evaluation Set Human Annotation Guidelines

> **Purpose:** Standard Operating Procedure (SOP) and decision criteria for human annotators labeling historical `AppleSupport` customer messages to create an immutable, portfolio-grade golden evaluation benchmark.

---

## 1. Annotation Objective

The objective of the SupportGraph AI Golden Set is to establish a rigorous, ground-truth benchmark of real historical customer support requests. 

This benchmark will evaluate:
- Downstream supervised intent classifiers.
- Retrieval-augmented generation (RAG) relevance.
- Automated routing and escalation policies.

To ensure scientific defensibility, labels must reflect **WHAT THE CUSTOMER NEEDS HELP WITH (the operational customer problem)**, rather than superficial entities, emotional sentiment, temporary bugs, or brand response strategies.

---

## 2. Unit of Annotation

- **Primary Unit of Annotation:** The **customer opening message** (`depth == 0`, `role == 'customer'`).
- **Context Availability:** Annotators have access to the first `AppleSupport` reply (`conversation_context`) to resolve genuine ambiguities regarding what Apple support understood the request to be.
- **Rule on Context:** Base the label primarily on the customer's stated problem in the opening tweet. Use conversation context strictly as a disambiguation aid, **not** to substitute for what the customer actually wrote.

---

## 3. The 9 Operational Intent Definitions

Annotators must select exactly **one primary label** from this approved candidate taxonomy (or `unclear_needs_review`):

### 1. `software_update_problem`
- **Definition:** Customer experiences device freeze, installation failure, operating system crash, or app malfunction during or immediately following an operating system update.
- **Inclusion:**
  - Phone stuck on Apple logo or reboot loop during/after updating.
  - Apps (WhatsApp, Instagram, etc.) crashing or failing to open specifically after an update.
  - Download, verification, or installation error codes for OS updates.
- **Exclusion:**
  - Battery drain issues post-update with no other system crash (label as `battery_power_issue`).
  - Desktop macOS update failures on MacBook/iMac (label as `mac_software_issue`).
- **Positive Example:** *"since I upgraded to iOS 11.0.3 my phone just giving issues, it just on life support"*

### 2. `battery_power_issue`
- **Definition:** Customer reports rapid battery drainage, sudden device power-offs, failure to charge, or overheating battery components.
- **Inclusion:**
  - Battery drops rapidly under idle or moderate usage (e.g. from 50% to 1% in minutes).
  - Phone unexpectedly dies despite showing remaining battery percentage.
  - Device will not charge or charger cable produces "accessory not supported" error.
- **Exclusion:**
  - Physical screen damage with device turned off (label as `display_touch_issue`).
- **Positive Example:** *"the battery drain is a bit much. iOS 11.0.2 on a A1533 iPhone 5S."*

### 3. `display_touch_issue`
- **Definition:** Customer reports physical damage, shattered glass, touch digitizer unresponsiveness, vertical lines, or display blackouts.
- **Inclusion:**
  - Touchscreen does not register finger taps, swipes, or typing.
  - Cracked, shattered, or chipped front glass after dropping.
  - Black screen, colored vertical lines, or ghost touch where screen clicks by itself.
- **Exclusion:**
  - Keyboard app failing to appear while rest of touch works (label as `keyboard_typing_issue`).
- **Positive Example:** *"my iPhone display froze completely and I cannot swipe to unlock"*

### 4. `account_access_issue`
- **Definition:** Customer cannot authenticate or access their Apple ID, iCloud account, or device due to forgotten credentials, locked security status, or verification failures.
- **Inclusion:**
  - Forgotten password, passcode, or Apple ID recovery key.
  - Account disabled or locked for security reasons.
  - Two-factor authentication (2FA) SMS or verification prompt not arriving.
  - iCloud Activation Lock screen preventing device setup.
- **Exclusion:**
  - Credit card billing inquiries or unauthorized subscription charges (label as `billing_purchase_issue`).
- **Positive Example:** *"forgot my apple id password and cannot login to my icloud account"*

### 5. `billing_purchase_issue`
- **Definition:** Customer reports unexpected financial charges, disputes recurring subscription fees, requests purchase refunds, or encounters payment method declines in the App Store/iTunes.
- **Inclusion:**
  - Unrecognized charges or double billing from iTunes/App Store on credit card statement.
  - Refund requests for accidental in-app purchases or subscription renewals.
  - Payment method declined or verification failure when attempting to purchase.
- **Exclusion:**
  - Cannot log in to account to view purchase history (label as `account_access_issue`).
- **Positive Example:** *"i was charged twice for my Apple Music subscription this month please refund"*

### 6. `keyboard_typing_issue`
- **Definition:** Customer reports text input glitches, predictive text errors, autocorrect replacement anomalies, or keyboard rendering bugs.
- **Inclusion:**
  - Autocorrect loop replacing letters with symbols (e.g., letter 'I' turning into 'A [?]').
  - Keyboard lags, stutters, or keys fail to register during typing.
  - Predictive text banner displaying corrupted suggestions.
- **Exclusion:**
  - Entire screen unresponsive to touch (label as `display_touch_issue`).
  - Mac physical laptop keyboard broken (label as `mac_software_issue`).
- **Positive Example:** *"y’all better fix these question marks that keep popping up whenever I try to use a damn I"*

### 7. `mac_software_issue`
- **Definition:** Customer reports software, operating system, or application problems specific to macOS and Mac computer hardware (MacBook, iMac, Mac mini).
- **Inclusion:**
  - macOS installation/upgrade freezes (Sierra, High Sierra).
  - Desktop Safari browser crashes, pop-up loops, or unresponsive tabs on a Mac.
  - MacBook trackpad gesture failures or desktop iTunes library sync errors.
- **Exclusion:**
  - Mobile iPhone/iPad iOS update issues (label as `software_update_problem`).
- **Positive Example:** *"lol wtf I swear there is something wrong with the memory of my macbook"*

### 8. `hardware_audio_connection_issue`
- **Definition:** Customer reports hardware defects or wireless connectivity malfunctions involving speakers, microphones, Bluetooth accessories, Wi-Fi toggles, or cellular signal hardware.
- **Inclusion:**
  - Distorted, muffled, or absent audio from earpiece or loud speaker.
  - Microphone not picking up audio during calls or voice memos.
  - Bluetooth repeatedly dropping or failing to pair with AirPods/car audio.
  - Wi-Fi toggle grayed out or cellular hardware showing persistent "No Service".
  - Physical volume/mute buttons jammed or unresponsive.
- **Exclusion:**
  - Display screen cracks or touch digitizer defects (label as `display_touch_issue`).
  - Charging port power issues (label as `battery_power_issue`).
- **Positive Example:** *"since upgrading loud speaker no longer works when receiving a call? Please fix!"*

### 9. `general_device_support`
- **Definition:** Customer submits general setup questions, store reservation/appointment inquiries, warranty inquiries, or broad multi-problem complaints requiring general triage.
- **Inclusion:**
  - Genius Bar appointment scheduling, store repair reservations, pickup inquiries.
  - General configuration questions (data transfer, setting up a new phone, ringer volume options).
  - Warranty coverage checks or AppleCare trade-in questions.
- **Exclusion:**
  - Specific technical malfunctions covered by classes 1 through 8 above.
- **Positive Example:** *"do I need an appointment to replace my battery at the Apple Store today?"*

---

## 4. Multi-Intent Messages: The Primary Intent Rule

Customer messages often mention more than one problem (e.g. *"I updated to iOS 11 and now my battery drains in 2 hours and my screen froze"*).

### Multi-Intent Decision Hierarchy:
1. **Identify the Core Root Cause:** If one issue triggered the other (e.g. update triggered battery drain), assign the **active root failure**:
   - If customer's primary complaint is that the battery dies in 2 hours: assign `battery_power_issue`.
   - If customer's phone is bricked or crashing across multiple functions after updating: assign `software_update_problem`.
2. **First / Emphasized Problem:** If two equal problems are stated with no causal link (e.g. *"my screen is cracked and my speaker is broken"*), assign the **first stated primary problem** (`display_touch_issue`).
3. **Never Multi-Label:** The current golden set is single-label multi-class. Assign exactly one primary intent.
4. **Document in Notes:** Note secondary intents in the `notes` column for future multi-label extensions.

---

## 5. Borderline and Edge-Case Guidance

| Case | Dilemma | Resolution |
|---|---|---|
| *"My phone is heating up while playing games"* | Battery vs. Hardware | Assign `battery_power_issue` (overheating in mobile devices is almost universally thermal power throttling). |
| *"Can't download apps from App Store"* | Account vs. Billing | Check message context: if asking for password/verification, assign `account_access_issue`; if payment method declined or receipt error, assign `billing_purchase_issue`. |
| *"Screen won't turn on"* | Display vs. Battery | If phone doesn't respond to charger at all, assign `battery_power_issue`; if phone vibrates/rings but display remains black, assign `display_touch_issue`. |
| *"Bluetooth says disconnected on my Mac"* | Mac vs. Hardware/Audio | If specific to MacBook/macOS environment, assign `mac_software_issue`. |

---

## 6. Ambiguity Handling: `unclear_needs_review`

If a customer message is so fragmented, ambiguous, or conversational that assigning an intent would require arbitrary guessing:
- Enter: `unclear_needs_review` in the `annotation_label` field.
- Document the reason in the `notes` field (e.g. *"Only says 'Hello please help' with no problem description"*).
- **Rule:** `unclear_needs_review` is tracked in the sampling audit and **does not silently contaminate the clean evaluation set**.

---

## 7. Historical Event Normalization

The dataset was collected in late 2017:
- **Do NOT invent temporary event labels** like `ios_11_bug` or `letter_i_autocorrect_issue`.
- Map the iOS 11 rollout complaints to `software_update_problem` or `battery_power_issue`.
- Map the 'A [?]' letter 'I' glitch to `keyboard_typing_issue`.
- This ensures the golden set benchmarks timeless, reusable machine learning classifiers.

---

## 8. Products Are Not Intents

Mentions of *iPhone 8*, *iPhone X*, *MacBook Pro*, or *iPad* are device entities.
- An iPhone 8 battery complaint $\rightarrow$ `battery_power_issue`.
- An iPad screen crack $\rightarrow$ `display_touch_issue`.
- Product names must **never** influence the choice of operational problem category.

---

## 9. Annotation Quality Checklist

Before submitting an annotated batch:
- [ ] Every record has a valid label from the 9 intents or `unclear_needs_review`.
- [ ] No `annotation_label` field is left blank.
- [ ] All labels use exact lowercase snake_case spelling.
- [ ] Multi-intent messages were resolved using the Primary Intent rule.
- [ ] Borderline cases were cross-checked against inclusion/exclusion criteria.
- [ ] Annotator name and completion timestamp are recorded in the manifest.
