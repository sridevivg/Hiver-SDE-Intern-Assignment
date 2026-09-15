# Phase 5.10: Annotation Consistency & Taxonomy Boundary Validation

> **SupportGraph AI — Quality Assurance & Taxonomy Calibration**  
> **Generated:** 2026-09-14 12:41:53 UTC  
> **Source Dataset SHA-256:** `1d3e9b3b8bdef3750437e59b17ab7c27167fd1548c2bb29de984151296c5b45a`  
> **Evaluated Human Reviews:** 77 / 200 (38.5%)  

---

## Core Scientific Principles & Data Protection

> [!IMPORTANT]
> **READ-ONLY AUDIT LAYER:** This layer performs read-only semantic consistency validation across completed human annotations.
> 1. **Zero Ground Truth Modification:** `annotation_label` and `annotation_status` remain 100% immutable.
> 2. **No Hard Classification Rules:** Observed patterns are NOT converted into rigid production classifier rules.
> 3. **Non-Prescriptive Flagging:** Flagged candidates represent *Possible boundary inconsistencies*, never *Incorrect annotations*.
> 4. **Sample Size Caveat:** Findings are based on $N=57$ completed reviews ($28.5\%$) and must be re-evaluated as annotation proceeds.

---

## SECTION 1 — OBSERVED DATA

### 1.1 Consistency Analysis Metrics

| Metric | Observed Value | Description |
|---|---|---|
| **Total Golden Records** | `200` | Size of the stratified golden benchmark |
| **Completed Human Reviews** | `77` | Authoritative ground-truth annotations ($N$) |
| **Pending Reviews** | `123` | Unreviewed records ($71.5\%$) |
| **Total Pairs Evaluated** | `2,926` | Total unique 2-combinations $\binom{N}{2}$ |
| **Same-Label Pairs** | `513` | Pairs where both records share the exact same human intent |
| **Cross-Label Pairs** | `2,413` | Pairs assigned to different intent categories |
| **Total Consistency Candidates** | `137` | Cross-label pairs meeting similarity thresholds |
| **HIGH Priority Candidates** | `11` | Cross-label similarity $\ge 0.14$ |
| **MEDIUM Priority Candidates** | `20` | Cross-label similarity $0.09 - 0.14$ |
| **LOW Priority Candidates** | `106` | Cross-label similarity $0.05 - 0.09$ |

### 1.2 Within-Label Semantic Coherence

| Intent Label | Reviewed ($N$) | Mean Pairwise Similarity | Min Similarity | Max Similarity | Diversity Score | Data Sufficiency |
|---|---|---|---|---|---|---|
| `battery_power_issue` | `7` | `0.0146` | `0.0000` | `0.1246` | `0.9854` | ✅ Sufficient ($N \ge 5$) |
| `billing_purchase_issue` | `1` | `1.0000` | `1.0000` | `1.0000` | `0.0000` | ⚠️ Small ($N < 5$) |
| `display_touch_issue` | `3` | `0.0211` | `0.0126` | `0.0362` | `0.9789` | ⚠️ Small ($N < 5$) |
| `general_device_support` | `26` | `0.0118` | `0.0000` | `0.7944` | `0.9882` | ✅ Sufficient ($N \ge 5$) |
| `hardware_audio_connection_issue` | `7` | `0.0138` | `0.0000` | `0.1420` | `0.9862` | ✅ Sufficient ($N \ge 5$) |
| `keyboard_typing_issue` | `10` | `0.0242` | `0.0000` | `0.1797` | `0.9758` | ✅ Sufficient ($N \ge 5$) |
| `mac_software_issue` | `5` | `0.0465` | `0.0000` | `0.1070` | `0.9535` | ✅ Sufficient ($N \ge 5$) |
| `software_update_problem` | `13` | `0.0212` | `0.0000` | `0.1305` | `0.9788` | ✅ Sufficient ($N \ge 5$) |
| `unclear_needs_review` | `5` | `0.0000` | `0.0000` | `0.0000` | `1.0000` | ✅ Sufficient ($N \ge 5$) |

### 1.3 Top Flagged Consistency Candidate Pairs

| Rank | Priority | Sim | Golden ID A | Human Label A | Golden ID B | Human Label B | Shared Signals |
|---|---|---|---|---|---|---|---|
| #1 | **`HIGH`** | `0.2551` | `gold_085` | `general_device_support` | `gold_106` | `hardware_audio_connection_issue` | `turn on, turn, wifi` |
| #2 | **`HIGH`** | `0.2337` | `gold_025` | `software_update_problem` | `gold_180` | `battery_power_issue` | `mi iphone, el ios, ios 11` |
| #3 | **`HIGH`** | `0.2173` | `gold_106` | `hardware_audio_connection_issue` | `gold_109` | `software_update_problem` | `can turn, turn, wifi` |
| #4 | **`HIGH`** | `0.2073` | `gold_004` | `software_update_problem` | `gold_124` | `keyboard_typing_issue` | `iphone, like` |
| #5 | **`HIGH`** | `0.1914` | `gold_046` | `hardware_audio_connection_issue` | `gold_102` | `software_update_problem` | `the music, music app, music` |
| #6 | **`HIGH`** | `0.1871` | `gold_067` | `unclear_needs_review` | `gold_128` | `general_device_support` | `fix` |
| #7 | **`HIGH`** | `0.1811` | `gold_014` | `general_device_support` | `gold_064` | `keyboard_typing_issue` | `amp still, phone amp, still` |
| #8 | **`HIGH`** | `0.1614` | `gold_067` | `unclear_needs_review` | `gold_181` | `keyboard_typing_issue` | `need` |
| #9 | **`HIGH`** | `0.1545` | `gold_040` | `software_update_problem` | `gold_128` | `general_device_support` | `ios, fix` |
| #10 | **`HIGH`** | `0.1471` | `gold_040` | `software_update_problem` | `gold_169` | `keyboard_typing_issue` | `fucking, fix` |
| #11 | **`HIGH`** | `0.1407` | `gold_052` | `general_device_support` | `gold_198` | `hardware_audio_connection_issue` | `macbook pro, macbook, pro` |
| #12 | **`MEDIUM`** | `0.1399` | `gold_089` | `general_device_support` | `gold_106` | `hardware_audio_connection_issue` | `wifi on, my wifi, wifi` |
| #13 | **`MEDIUM`** | `0.1222` | `gold_093` | `general_device_support` | `gold_190` | `battery_power_issue` | `won turn, turn on, turn` |
| #14 | **`MEDIUM`** | `0.1202` | `gold_082` | `software_update_problem` | `gold_135` | `keyboard_typing_issue` | `question mark, fix this, question` |
| #15 | **`MEDIUM`** | `0.1136` | `gold_032` | `keyboard_typing_issue` | `gold_074` | `battery_power_issue` | `pls help, pls` |

---

## SECTION 2 — PRELIMINARY INTERPRETATION

### 2.1 Cross-Label Boundary Overlaps

#### 🔍 `general_device_support vs software_update_problem`
- **Reviewed Sample Support:** `general_device_support` ($N=26$) vs `software_update_problem` ($N=13$)
- **Mean Cross Similarity:** `0.0091` (Max: `0.1545` across 338 pairs)
- **Observed Boundary Pattern:** Customers frequently mention broad iOS version complaints alongside general dissatisfaction or settings confusion. When an update is mentioned as the triggering event, the boundary between general triage and update-specific troubleshooting can become blurred.

**Representative Cross-Boundary Examples:**

> **[gold_040] (software_update_problem):** "Dear @115858 
Please fix all your fucking issues with iOS 11. I’m sick and tired of all these bugs. 🙃"  
> **[gold_128] (general_device_support):** "hey @115858 can u fix IOS"  
> *Similarity:* `0.1545` | *Shared terms:* `ios, fix`
>
> **[gold_075] (general_device_support):** "@AppleSupport my app store’s not working :("  
> **[gold_102] (software_update_problem):** "Can @115948 or @115858 please fix the music app ...ever since the update .. I can’t pause music without going to app🙄 #sos"  
> *Similarity:* `0.1089` | *Shared terms:* `app`
>

#### 🔍 `general_device_support vs hardware_audio_connection_issue`
- **Reviewed Sample Support:** `general_device_support` ($N=26$) vs `hardware_audio_connection_issue` ($N=7$)
- **Mean Cross Similarity:** `0.0106` (Max: `0.2551` across 182 pairs)
- **Observed Boundary Pattern:** Bluetooth, Wi-Fi toggles, and audio accessories are often described alongside general device settings. Phrases such as 'can't turn off Wi-Fi in Control Center' bridge general UI triage and wireless connectivity.

**Representative Cross-Boundary Examples:**

> **[gold_085] (general_device_support):** "why does WiFi turn on by itself when I turn it off manually it’s annoying @AppleSupport"  
> **[gold_106] (hardware_audio_connection_issue):** "@AppleSupport i can’t turn on my WiFi on my iPhone 7"  
> *Similarity:* `0.2551` | *Shared terms:* `turn on, turn, wifi`
>
> **[gold_052] (general_device_support):** "@AppleSupport 👋, What is the best antivirus for MacBook Pro ?"  
> **[gold_198] (hardware_audio_connection_issue):** "@AppleSupport Can I use my wired  printer  having USB on my MacBook pro which has  has c type?..Pls suggest if USB hub and c-type to USB female will work for it?"  
> *Similarity:* `0.1407` | *Shared terms:* `macbook pro, macbook, pro`
>

#### 🔍 `general_device_support vs battery_power_issue`
- **Reviewed Sample Support:** `general_device_support` ($N=26$) vs `battery_power_issue` ($N=7$)
- **Mean Cross Similarity:** `0.0073` (Max: `0.1222` across 182 pairs)
- **Observed Boundary Pattern:** Power drain is occasionally reported alongside general device overheating or sluggish performance. Ambiguity arises when battery drain is mentioned as one of multiple diffuse symptoms.

**Representative Cross-Boundary Examples:**

> **[gold_093] (general_device_support):** "Out of the blue my 2016 MacBook won’t turn on and this is really an unexpected roadblock @115858"  
> **[gold_190] (battery_power_issue):** "@115858 there’s been a problem with my sisters iPhone 7 like it won’t turn on or nothing even though it’s  good charged, has not been dropped"  
> *Similarity:* `0.1222` | *Shared terms:* `won turn, turn on, turn, won`
>
> **[gold_075] (general_device_support):** "@AppleSupport my app store’s not working :("  
> **[gold_191] (battery_power_issue):** "Hi @AppleSupport my c3yo iPad Air died. Tried restore as per website, error msg now MacBook can’t detect at all. Any ideas? Take to Store?"  
> *Similarity:* `0.0730` | *Shared terms:* `store`
>

#### 🔍 `general_device_support vs mac_software_issue`
- **Reviewed Sample Support:** `general_device_support` ($N=26$) vs `mac_software_issue` ($N=5$)
- **Mean Cross Similarity:** `0.0054` (Max: `0.0698` across 130 pairs)
- **Observed Boundary Pattern:** Mac mentions (MacBook, iMac, macOS, Sierra) can overlap with general software questions. Reviewers must distinguish platform-specific desktop software troubleshooting from general cross-device questions.

**Representative Cross-Boundary Examples:**

> **[gold_085] (general_device_support):** "why does WiFi turn on by itself when I turn it off manually it’s annoying @AppleSupport"  
> **[gold_027] (mac_software_issue):** "@AppleSupport since updating to high sierra I'm seeing lots of notifications from Time Machine, which I don't use. How do I turn off?"  
> *Similarity:* `0.0698` | *Shared terms:* `turn`
>
> **[gold_052] (general_device_support):** "@AppleSupport 👋, What is the best antivirus for MacBook Pro ?"  
> **[gold_086] (mac_software_issue):** "@AppleSupport my MacBook Air has the dreaded spinning beach ball. Help please."  
> *Similarity:* `0.0552` | *Shared terms:* `macbook`
>

#### 🔍 `display_touch_issue vs keyboard_typing_issue`
- **Reviewed Sample Support:** `display_touch_issue` ($N=3$) vs `keyboard_typing_issue` ($N=10$)
- **Mean Cross Similarity:** `0.0109` (Max: `0.0919` across 30 pairs)
- **Observed Boundary Pattern:** Touchscreen unresponsiveness directly interferes with on-screen keyboard typing. Customers experiencing keyboard lag or phantom keystrokes may describe both screen and keyboard symptoms.

**Representative Cross-Boundary Examples:**

> **[gold_084] (keyboard_typing_issue):** "Formated and reinstalled MacOS High Sierra by Genius Bar member at Apple Shop but still problem of random key stopped working remains. It's time for Apple to acknowledge issue @AppleSupport even Google search shows many r facing same issue. No solution yet!"  
> **[gold_103] (display_touch_issue):** "@AppleSupport so my iPhone crashes touch seems to stop working altho I can access wifi menu etc Ask apple the issue they tell me my phone screen needs replaced - get new phone and same issue happens is this a problem with iOS or the 8?"  
> *Similarity:* `0.0919` | *Shared terms:* `same issue, working, problem, apple, issue`
>

### 2.2 Within-Label Outlier & Diversity Interpretation

- **`battery_power_issue` Outliers:**
  - `[gold_180]`: ".@115858 battery and velocity of my iPhone 6 really sucks / batería y velocidad de mi iPhone 6 con el iOS 11 realmente apestan #iOS11 #fail" (Mean intra-class similarity: `0.0043` vs class mean `0.0146`)
  - `[gold_154]`: "I’m so happy the #iPhoneX came out. Now my Fossil of a phone #iPhone7 can start working at an incredibly slow pace and shut off randomly. @115858 at its finest." (Mean intra-class similarity: `0.0070` vs class mean `0.0146`)
- **`display_touch_issue` Outliers:**
  - `[gold_026]`: "@115858 for the love of #280characters would you please fix this I️ crap!  It’s annoying AF! Also apps are loading wrong on 6splus, because of X screen size." (Mean intra-class similarity: `0.0135` vs class mean `0.0211`)
- **`general_device_support` Outliers:**
  - `[gold_128]`: "hey @115858 can u fix IOS" (Mean intra-class similarity: `0.0021` vs class mean `0.0118`)
  - `[gold_119]`: "#Siri. Hate new voice seems to works less well. Sounds like a teenager. @115858 @AppleSupport   Options?  Multiple selection." (Mean intra-class similarity: `0.0039` vs class mean `0.0118`)
- **`hardware_audio_connection_issue` Outliers:**
  - `[gold_157]`: "@AppleSupport where can I find out if my AirPods are still under the warranty?" (Mean intra-class similarity: `0.0000` vs class mean `0.0138`)
  - `[gold_198]`: "@AppleSupport Can I use my wired  printer  having USB on my MacBook pro which has  has c type?..Pls suggest if USB hub and c-type to USB female will work for it?" (Mean intra-class similarity: `0.0019` vs class mean `0.0138`)
- **`keyboard_typing_issue` Outliers:**
  - `[gold_181]`: ". @115858 my laptop not letting me use my keyboard or mouse.... y’all need to help me out" (Mean intra-class similarity: `0.0000` vs class mean `0.0242`)
  - `[gold_001]`: "@AppleSupport Just updated to latest #ios11. Autocorrect always suggests "minuet" when I want "minute". Trying to push baroque dances on us?" (Mean intra-class similarity: `0.0035` vs class mean `0.0242`)
- **`mac_software_issue` Outliers:**
  - `[gold_086]`: "@AppleSupport my MacBook Air has the dreaded spinning beach ball. Help please." (Mean intra-class similarity: `0.0000` vs class mean `0.0465`)
- **`software_update_problem` Outliers:**
  - `[gold_140]`: "Looks like its a common problem with the @115858 High Sierra download as there are a number of pics posted on the web. 
This is the same error message which I got today.
It doesn't allow you to quit download unless you know your way around computers. Needs fixing asap @116333 https://t.co/PIbCxNlyoc" (Mean intra-class similarity: `0.0018` vs class mean `0.0212`)
  - `[gold_109]`: "Can’t turn off WiFi from control center? New @115858 iOS is garbage. Who knows how to roll back to the good one?" (Mean intra-class similarity: `0.0135` vs class mean `0.0212`)

---

## SECTION 3 — PROPOSED SOFT ANNOTATION GUIDELINES

> [!TIP]
> **NON-COERCIVE GUIDELINES:** The following guidelines are advisory cognitive aids for human reviewers and prompt designers. They emphasize dominant operational friction over superficial keyword matches.

### Guideline for `general_device_support vs software_update_problem`
When the customer's primary friction point involves an active OS upgrade attempt, installation failure, or explicit post-update regression, consider evidence favoring 'software_update_problem'. Conversely, when the inquiry represents broad feature questions, general settings configuration, or unspecified device sluggishness without a focal update symptom, 'general_device_support' is usually appropriate.

### Guideline for `general_device_support vs hardware_audio_connection_issue`
When the dominant issue involves physical audio output (speakers, microphones, headphones), Bluetooth pairing, or wireless adapter connectivity, annotators should consider evidence favoring 'hardware_audio_connection_issue'. When the customer seeks general advice or expresses broad frustration with OS UI controls without an underlying hardware/audio fault, 'general_device_support' may be considered.

### Guideline for `general_device_support vs battery_power_issue`
When battery depletion, charging failure, percentage drop, or power preservation is the central complaint, evidence usually favors 'battery_power_issue'. If power is only mentioned tangentially amidst broad system complaints, consider the primary operational action requested.

### Guideline for `general_device_support vs mac_software_issue`
When the troubleshooting context specifically requires macOS desktop software remediation (e.g. Safari desktop, macOS installer, Time Machine, Finder), consider 'mac_software_issue'. If the inquiry involves generic multi-device queries or general support referral, 'general_device_support' may be appropriate.

### Guideline for `display_touch_issue vs keyboard_typing_issue`
When the core issue is character input, text prediction, autocorrect glitches, or keyboard layout behavior, evidence usually favors 'keyboard_typing_issue'. When the screen physical digitizer, touch unresponsiveness, display flickering, or backlight failure is the primary defect, annotators should consider 'display_touch_issue'.

---

## SECTION 4 — HUMAN CONSISTENCY REVIEW QUEUE

A top-priority consistency review queue containing the highest-ambiguity pairs has been prepared for controlled human validation:
- Reviewers can inspect pairs in the CLI using `python -m backend.scripts.analyze_annotation_consistency --interactive`.
- Available non-destructive actions:
  - `[A]` Labels are both appropriate (confirms distinct operational nuance).
  - `[B]` Record A should be reconsidered.
  - `[C]` Record B should be reconsidered.
  - `[D]` Both records should be reconsidered.
  - `[S]` Skip.
- Decisions are logged to `data/golden/consistency_review_decisions.json` without modifying ground-truth labels.
