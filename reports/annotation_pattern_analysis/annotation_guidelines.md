# SupportGraph AI — Evidence-Based Intent Annotation Guidelines (Phase 5.9)

Derived from empirical human ground truth and observed confusion patterns.

---

## `account_access_issue`
**Definition:** Authentication, Apple ID credentials, two-factor authentication, or security lockouts.  
**Status:** `OBSERVED FROM DATA`

### Include When:
- Apple ID locked, disabled, or password reset failure.
- Two-factor verification codes not received.
- iCloud account login credential failure.

### Do NOT Use When:
- Unauthorized financial charge on account (use billing_purchase_issue).

### Boundary Cases:
- Subscription billing password prompt routes to billing_purchase_issue if payment is disputed.

---

## `battery_power_issue`
**Definition:** Abnormal battery depletion, device overheating, charging failure, or sudden power loss.  
**Status:** `OBSERVED FROM DATA`

### Include When:
- Battery percentage drops rapidly (e.g. 100% to 20% in minutes).
- Device fails to charge, won't turn on, or overheats while charging.
- Battery health degradation complaints.

### Do NOT Use When:
- Charger cord or physical adapter is physically broken/torn (use hardware_audio_connection_issue).

### Boundary Cases:
- Battery drain following an update routes to battery_power_issue.

### Observed Examples from Ground Truth:
> *"@115858 there’s been a problem with my sisters iPhone 7 like it won’t turn on or nothing even though it’s  good charged, has not been dropped"*
> *"Hi @AppleSupport my c3yo iPad Air died. Tried restore as per website, error msg now MacBook can’t detect at all. Any ideas? Take to Store?"*
> *"@AppleSupport laptop keeps shutting down during startup, tried a bunch of resets but nothing's working PLS HELP."*

---

## `billing_purchase_issue`
**Definition:** Financial transactions, subscriptions, refunds, App Store charges, or Apple Pay errors.  
**Status:** `OBSERVED FROM DATA`

### Include When:
- Unauthorized charges, double billing, or unexpected subscription renewals.
- Refund requests for apps, in-app purchases, or media.
- Credit card declined on Apple ID.

### Do NOT Use When:
- Account credential lockout (use account_access_issue).

### Boundary Cases:
- Inquiry on hardware warranty coverage without financial dispute routes to general_device_support.

---

## `display_touch_issue`
**Definition:** Physical screen panel damage, backlight flickering, touch digitizer unresponsiveness, or display artifacts.  
**Status:** `OBSERVED FROM DATA`

### Include When:
- Screen flickering, black display, lines across screen, or shattered glass.
- Touchscreen digitizer completely unresponsive or ghost touch.

### Do NOT Use When:
- Only keyboard text input is glitching (use keyboard_typing_issue).

### Boundary Cases:
- Display unresponsiveness without keyboard context routes to display_touch_issue.

### Observed Examples from Ground Truth:
> *"@115858 for the love of #280characters would you please fix this I️ crap!  It’s annoying AF! Also apps are loading wrong on 6splus, because of X screen size."*
> *"@AppleSupport how much is it for an iPhone 7 glass screen? Struggling to find anything and I’m not wanting to book in the repair. Had this modern phone two weeks and it’s shattered off one little fall.  😔 #disappointed"*
> *"@AppleSupport so my iPhone crashes touch seems to stop working altho I can access wifi menu etc Ask apple the issue they tell me my phone screen needs replaced - get new phone and same issue happens is this a problem with iOS or the 8?"*

---

## `general_device_support`
**Definition:** Broad device inquiries, general troubleshooting assistance, store visit queries, or multi-topic questions without a single dominant subsystem.  
**Status:** `OBSERVED FROM DATA`

### Include When:
- Customer asks general how-to or store visit questions.
- Customer expresses general dissatisfaction without citing a specific failure mode.

### Do NOT Use When:
- Any specific operational subsystem (battery, screen, audio, update, keyboard, macOS) is cited.

### Boundary Cases:
- Use strictly as a fallback when no specific category applies.

### Observed Examples from Ground Truth:
> *"If I could turn auto brightness off and also not have my Bluetooth automatically turn on everytime I start my car that would be nice @115858"*
> *"#Siri. Hate new voice seems to works less well. Sounds like a teenager. @115858 @AppleSupport   Options?  Multiple selection."*
> *"Dear @115858 Please fix the podcast app. I miss knowing which archived episodes I haven't listened to and when the home screen display was accurate and functioning."*

---

## `hardware_audio_connection_issue`
**Definition:** Malfunction of physical audio components, headphones, speakers, microphones, or peripheral connectivity ports/cables.  
**Status:** `OBSERVED FROM DATA`

### Include When:
- AirPods, wired headphones, speaker crackling, microphone failure.
- Physical connection issues: USB-C hubs, lightning adapters, dongles, wired printer connections.
- Bluetooth audio disconnects or audio accessory failure.

### Do NOT Use When:
- Network/Wi-Fi router connectivity without physical hardware failure.

### Boundary Cases:
- MacBook speaker failure routes to hardware_audio_connection_issue, not mac_software_issue.

### Observed Examples from Ground Truth:
> *"@AppleSupport i can’t turn on my WiFi on my iPhone 7"*
> *"My IPhone glitch is that it won’t stay connected to any WiFi. @AppleSupport ?"*
> *"@AppleSupport Can I use my wired  printer  having USB on my MacBook pro which has  has c type?..Pls suggest if USB hub and c-type to USB female will work for it?"*

---

## `keyboard_typing_issue`
**Definition:** Malfunctions in text input, keyboard unresponsiveness, predictive text bugs, or autocorrect character glitches.  
**Status:** `OBSERVED FROM DATA`

### Include When:
- Autocorrect letter 'I' rendering bug (e.g. A [?] box glitch).
- Keyboard lag, freeze, or failure to appear on screen.
- Text prediction or auto-replacement errors.

### Do NOT Use When:
- Whole touchscreen digitizer is physically unresponsive (use display_touch_issue).

### Boundary Cases:
- Autocorrect bug accompanied by display glitch routes to keyboard_typing_issue.

### Observed Examples from Ground Truth:
> *"What is up @115858 @AppleSupport with the letter I looking like this 👉🏼 I️  on my IPhone"*
> *"@AppleSupport 🗣I am tired of seeing boxes ,with a question mark in the middle , where words are supposed to be. Fix this 💩 NOW !!"*
> *"@115858 fix this fucking question mark exclamation shit before I️ get a tracfone"*

---

## `mac_software_issue`
**Definition:** Issues specific to the macOS operating system, Mac desktop applications, or system utilities.  
**Status:** `OBSERVED FROM DATA`

### Include When:
- macOS High Sierra OS freezes, kernel panics, or spinning beach ball.
- Finder, Safari, or Time Machine errors and crashes.
- MacBook boot loops and macOS Recovery inquiries.

### Do NOT Use When:
- Physical MacBook hardware/speaker failures (use hardware_audio_connection_issue).

### Boundary Cases:
- iCloud photo sync between Mac and iOS routes to mac_software_issue if desktop sync fails.

### Observed Examples from Ground Truth:
> *"Hey @115858 @AppleSupport we need a lil favour. Safari 11.0.1 &amp; it’s relationship with High Sierra needs to be fixed. Many people are having multiple problems since the High Sierra update, specific sites are not loading. Multiple community posts have not fixed problem. Please help"*
> *"@AppleSupport since updating to high sierra I'm seeing lots of notifications from Time Machine, which I don't use. How do I turn off?"*
> *"@AppleSupport photos not syncing properly between OSX high Sierra and iOS.  Same problem in several versions  d was hoping would get fixed. Say an album with 172 pics but only 33 transfer.  Have rebuilt library. Wiped iPhone. Closed and reopened apps. Reset devices. Etc."*

---

## `software_update_problem`
**Definition:** Customer experiences system instability, installation failures, or regressions caused by an OS or app update.  
**Status:** `OBSERVED FROM DATA`

### Include When:
- Issue explicitly started after installing an update (iOS 11, High Sierra, app updates).
- Update download, verification, or installation fails or gets stuck.
- General complaint about bugs introduced by a named software release.

### Do NOT Use When:
- Problem has an isolated physical root cause (e.g. cracked display, physical cable broken).

### Boundary Cases:
- If battery drains rapidly after update, battery_power_issue takes precedence unless the user only complains about update bugs.

### Observed Examples from Ground Truth:
> *"@AppleSupport hey guys just curious - why aren’t iOS app updates automatic? Seems pretty clear here, no? Yet I manually have to check and click Update All daily... https://t.co/jFgs9KRl9U"*
> *"Can’t turn off WiFi from control center? New @115858 iOS is garbage. Who knows how to roll back to the good one?"*
> *"@AppleSupport Please for heaven sake, since newest update the voice to text is horrible and texting is horrible"*

---

## `unclear_needs_review`
**Definition:** Messages with insufficient information, non-English foreign text, severe truncation, or contradictory multi-domain claims.  
**Status:** `OBSERVED FROM DATA`

### Include When:
- Message consists of only 1-3 words without context.
- Message is entirely in a foreign language without clear diagnostic intent.
- Customer message is severely corrupted or unintelligible.

### Do NOT Use When:
- A plausible operational intent can be determined from the customer message.

### Boundary Cases:
- Use when the reviewer cannot confidently assign any of the 9 operational intents.

### Observed Examples from Ground Truth:
> *"@AppleSupport it works on my phone but not my MacBook"*
> *"pentatonix being made fun of on a popular animated show is on my bucket list"*
> *"@AppleSupport thank u for a great chat and purchase experience"*

---
