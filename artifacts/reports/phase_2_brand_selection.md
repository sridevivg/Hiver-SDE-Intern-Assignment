# Phase 2 — Brand Selection Report

> **Generated:** 2026-09-09 17:08 UTC  
> **Script:** `python -m backend.scripts.select_brand`  
> **Status:** All statistics from actual dataset computation — nothing fabricated.

---

## Objective

Scientifically select ONE brand from the Customer Support on Twitter dataset
that is most suitable for building:
- Data-driven intent classification
- Historical resolution retrieval (RAG)
- Grounded reply generation
- Escalation decision experiments
- A 150–250 example manually labelled golden evaluation set

---

## Candidate Selection

**Method:** Top-10 brands by Phase 1 usable interaction volume  
**Source:** `data/interim/brand_candidates.csv` (Phase 1 output)  

**FACTS FROM THE DATA:**

| Brand | Phase 1 Usable Interactions |
|---|---|
| `AppleSupport` | 106,719 |
| `British_Airways` | 29,315 |
| `Tesco` | 38,501 |
| `AmazonHelp` | 169,287 |
| `Uber_Support` | 56,261 |
| `Delta` | 42,197 |
| `AmericanAir` | 36,598 |
| `SpotifyCares` | 43,243 |
| `comcastcares` | 33,007 |
| `TMobileHelp` | 34,287 |

---

## Usable Interaction Definition

Before scoring brands, a precise definition of a usable customer-support interaction was established:

- **Dataset fields used:** `tweet_id`, `author_id`, `inbound`, `created_at`, `text`, `in_response_to_tweet_id`
- **Direction of reply relationships:** Customer message (`inbound=True`) → Brand response (`inbound=False`, `in_response_to_tweet_id` = customer `tweet_id`)
- **Customer message identification:** Identified by `inbound == True`.
- **Brand message identification:** Identified by `inbound == False` with an authorized brand handle.
- **Customer-to-brand interaction count:** Direct reply pairs where a brand response links to an existing customer parent tweet.

---

## Evaluation Metrics

Five metrics are computed for each candidate. Each is documented below.

### A. Usable Interaction Volume (weight 30%)

* **Definition:** Count of brand outbound tweets where `in_response_to_tweet_id` is not null.
  These are direct replies from the brand to a customer tweet.
* **Formula:** `raw_volume = count(brand_rows where in_response_to_tweet_id IS NOT NULL)`
* **Why it matters:** Volume determines whether there is enough historical data for intent clustering,
  RAG retrieval, and evaluation set construction. A brand with <5,000 interactions
  would be insufficient for downstream tasks.

* **Source:** Reused from Phase 1 `brand_candidates.csv` — no recomputation needed.

---

### B. Response Coverage (weight 20%)

* **Definition:** Fraction of brand reply interactions whose parent (customer) tweet
  actually exists in the dataset.
* **Formula:** `coverage = replies_with_parent_in_dataset / total_brand_replies`
* **Why it matters:** If a brand's reply references a customer tweet that is not in the dataset,
  we cannot reconstruct the interaction for RAG. High coverage means we can
  actually use those interactions as historical grounding data.

---

### C. Conversation Reconstructability (weight 20%)

* **Definition:** Of brand replies whose parent tweet exists in the dataset,
  what fraction have that parent as a valid **inbound (customer)** tweet?
* **Formula:** `reconstructability = valid_customer_parents / replies_with_parent_in_dataset`
* **Why it matters:** A brand reply is only useful if we can identify the corresponding customer message.
  If the parent is another brand tweet (e.g., continuation), the pair is less useful
  as a standalone customer→brand interaction.

---

### D. Data Completeness (weight 15%)

* **Definition:** Composite of text completeness and reply chain integrity (weighted equally at 0.5 each):
  1. Text completeness: `1 - (null_or_empty_text_count / total_tweets)`
  2. Reply chain integrity: `1 - (broken_chain_count / total_reply_tweets)`
* **Formula:** `completeness = 0.5 × text_completeness + 0.5 × chain_integrity`
* **Why it matters:** Missing text or broken reply chains directly impair downstream processing.
  A brand with high volume but many broken chains is less useful.

---

### E. Issue / Message Diversity (weight 15%)

* **Definition:** Estimated diversity of customer issues directed at the brand,
  using TF-IDF + KMeans clustering entropy as a proxy.
* **Formula:** `diversity = H(cluster_distribution) / log(k)` (Shannon entropy normalized by log(k))
* **Why it matters:** High diversity means the brand handles many distinct issue types, enabling
  meaningful intent clustering. A brand that only handles one type of complaint
  limits intent taxonomy richness.

**Algorithm (seed=42, sample_size=5000):**
1. Sample up to 5000 customer messages (parents of brand replies)
2. Normalize: lowercase, remove @mentions and URLs
3. TF-IDF vectorization (max_features=500, min_df=2)
4. KMeans clustering (k=10, fixed seed)
5. Shannon entropy of cluster size distribution, normalized by `log(k)`

**Important caveat:** TF-IDF + KMeans is a proxy, not ground-truth topic modeling.
Results are meaningful for relative comparison only.

---

## Normalization

Min-Max Normalization is applied to each raw metric across the evaluated top-N candidate brands:

> `norm_val = (val - min_val) / (max_val - min_val)`

If all candidates share the identical metric value, the normalized score falls back to `0.0` safely.
All normalized scores are relative strictly to the evaluated candidate brands.

---

## Weighting Methodology

**Default scenario weights:**

| Metric | Weight | Rationale |
|---|---|---|
| Interaction Volume | 30% | Necessary condition for all downstream tasks |
| Response Coverage | 20% | Directly determines RAG usability |
| Reconstructability | 20% | Determines quality of historical pairs |
| Data Completeness | 15% | Data quality floor |
| Issue Diversity | 15% | Intent taxonomy richness |

Weights were chosen based on project priorities:
- Volume and coverage are threshold requirements (highest weights).
- Diversity and completeness are secondary quality indicators.
- All weights sum to 1.0.

---

## Quantitative Results

**FACTS FROM THE DATA:**

### Raw Metric Values

| Brand | Volume | Coverage | Reconstruct | Completeness | Diversity | Sample |
|---|---|---|---|---|---|---|
| `AppleSupport` | 106,719 | 0.9993 | 1.0000 | 0.9997 | 0.9575 | 5,000 |
| `British_Airways` | 29,315 | 0.9992 | 1.0000 | 0.9996 | 0.9304 | 5,000 |
| `Tesco` | 38,501 | 0.9992 | 0.9999 | 0.9996 | 0.9157 | 5,000 |
| `AmazonHelp` | 169,287 | 0.9973 | 0.9999 | 0.9986 | 0.9102 | 5,000 |
| `Uber_Support` | 56,261 | 0.9988 | 0.9994 | 0.9994 | 0.9205 | 5,000 |
| `Delta` | 42,197 | 0.9989 | 0.9992 | 0.9994 | 0.9116 | 5,000 |
| `AmericanAir` | 36,598 | 0.9982 | 1.0000 | 0.9991 | 0.9386 | 5,000 |
| `SpotifyCares` | 43,243 | 0.9991 | 0.9974 | 0.9996 | 0.9176 | 5,000 |
| `comcastcares` | 33,007 | 0.9990 | 0.9984 | 0.9995 | 0.8850 | 5,000 |
| `TMobileHelp` | 34,287 | 0.9983 | 0.9996 | 0.9992 | 0.8803 | 5,000 |

---

## Brand Ranking

### Normalized Scores and Final Ranking

> Note: Normalization is relative to the evaluated top-N set, not all 108 brands.

| Brand | Norm Volume | Norm Coverage | Norm Reconstruct | Norm Complete | Norm Diversity | Final Score | Rank |
|---|---|---|---|---|---|---|---|
| `AppleSupport` | 0.5530 | 1.0000 | 0.9929 | 1.0000 | 1.0000 | **0.8645** | **#1** |
| `British_Airways` | 0.0000 | 0.9261 | 0.9871 | 0.9261 | 0.6492 | **0.6189** | **#2** |
| `Tesco` | 0.0656 | 0.9326 | 0.9803 | 0.9326 | 0.4576 | **0.6108** | **#3** |
| `AmazonHelp` | 1.0000 | 0.0000 | 0.9798 | 0.0000 | 0.3869 | **0.5540** | **#4** |
| `Uber_Support` | 0.1925 | 0.7382 | 0.7774 | 0.7382 | 0.5198 | **0.5496** | **#5** |
| `Delta` | 0.0920 | 0.7725 | 0.6853 | 0.7725 | 0.4057 | **0.4959** | **#6** |
| `AmericanAir` | 0.0520 | 0.4385 | 1.0000 | 0.4385 | 0.7550 | **0.4824** | **#7** |
| `SpotifyCares` | 0.0995 | 0.9083 | 0.0000 | 0.9083 | 0.4822 | **0.4201** | **#8** |
| `comcastcares` | 0.0264 | 0.8534 | 0.3793 | 0.8534 | 0.0600 | **0.3915** | **#9** |
| `TMobileHelp` | 0.0355 | 0.5055 | 0.8450 | 0.5055 | 0.0000 | **0.3566** | **#10** |

---

## Sensitivity Analysis

Three weight scenarios were tested to assess whether the ranking is stable.

### Scenario Weights

| Scenario | Volume | Coverage | Reconstruct | Complete | Diversity |
|---|---|---|---|---|---|
| default | 30% | 20% | 20% | 15% | 15% |
| volume_focused | 50% | 15% | 15% | 10% | 10% |
| quality_focused | 15% | 25% | 30% | 15% | 15% |

### Scenario Rankings

| Brand | Default Score | Volume-Focused Score | Quality-Focused Score |
|---|---|---|---|
| `AppleSupport` | 0.8645 (#1) ★ | 0.7754 (#1) ★ | 0.9308 (#1) ★ |
| `British_Airways` | 0.6189 (#2) | 0.4445 (#5) | 0.7639 (#2) |
| `Tesco` | 0.6108 (#3) | 0.4588 (#3) | 0.7456 (#3) |
| `AmazonHelp` | 0.5540 (#4) | 0.6857 (#2) | 0.5020 (#7) |
| `Uber_Support` | 0.5496 (#5) | 0.4494 (#4) | 0.6354 (#4) |
| `Delta` | 0.4959 (#6) | 0.3825 (#6) | 0.5892 (#6) |
| `AmericanAir` | 0.4824 (#7) | 0.3612 (#7) | 0.5965 (#5) |
| `SpotifyCares` | 0.4201 (#8) | 0.3251 (#8) | 0.4506 (#10) |
| `comcastcares` | 0.3915 (#9) | 0.2895 (#9) | 0.4681 (#8) |
| `TMobileHelp` | 0.3566 (#10) | 0.2709 (#10) | 0.4610 (#9) |

### Sensitivity Conclusion

**The same brand (`AppleSupport`) wins across ALL weight scenarios.** This indicates a robust selection.
- **[default]** winner: `AppleSupport`
- **[volume_focused]** winner: `AppleSupport`
- **[quality_focused]** winner: `AppleSupport`

---

## Qualitative Sanity Check

> This section is a SANITY CHECK only.
> It verifies that the data contains real customer support interactions.
> It CANNOT override the quantitative ranking.

**Top 3 brands sampled:** `AppleSupport`, `British_Airways`, `Tesco`

Sample interactions (customer text truncated to 200 chars for privacy):


### AppleSupport

> **Customer:** @AppleSupport My iPhone 6 now crashes all the time 🙃.
Thank you for the wonderful update . It gets hanged and takes a million years to restart.
>  
> **AppleSupport:** @265764 Let's work together to see how we can get your iPhone running smoothly again. Which exact iOS 11 version are you using under Settings &gt; General &gt; About? Are there certain apps or tasks b

> **Customer:** @115858 @AppleSupport please make an option for us to be able to organize apps in alphabetical order!
>  
> **AppleSupport:** @771055 We always appreciate feedback. You can submit it here: https://t.co/eTPVYVFyd8

> **Customer:** @AppleSupport what’s up with this ? https://t.co/t9rJLxtrYU
>  
> **AppleSupport:** @515123 Here’s what you can do to work around the issue until it’s fixed in a future software update: https://t.co/xXaXeeSRt9

> **Customer:** It’s 9am and my iPhone X is at 45%. @115858 PLEASE fix this! I took it off charger at 100% at 6am. https://t.co/EIZPFKgE93
>  
> **AppleSupport:** @529644 We can help. DM us the iOS version you're using to get started. https://t.co/GDrqU22YpT

> **Customer:** At least once a year without fail, my MacBook dies right before I have a ton of assignment due... what the fuck @115858 .
>  
> **AppleSupport:** @557628 We'd love to help with this. What happens when you try to use it? DM us more details and we'll get started. https://t.co/GDrqU22YpT


### British_Airways

> **Customer:** @British_Airways Hang on, so I've figured out that you *can't* change the currency of the price quoted. Leaving from an European airport = euros. Am I right?
>  
> **British_Airways:** @177378 from.  I'm afraid there is no way to change this. (2/2) ^N

> **Customer:** Hey hey @British_Airways we are flying long haul to Singapore with you, are @23433 travel pillows allowed? @318940
>  
> **British_Airways:** @339477 Hi Alice. This will be fine to take on board with you. It'll need to go in your hand luggage too. Hope this helps. ^Kev

> **Customer:** @British_Airways That’s not MY problem!
>  
> **British_Airways:** @173892 Hi Melville. Unfortunately, if there’re air traffic restrictions we cannot go against that as they’re imposed by the airports. 1/2

> **Customer:** Decided to take a lift off a bigggg featherless birdie! @British_Airways @1090 @151094 @151095 @1091 👍🏼 #RestTheWings 🛫 https://t.co/byQeDIbTjb
>  
> **British_Airways:** @151093 Hi there, I hope it was a safe flight ! ^Tom

> **Customer:** @British_Airways My mom does not have a twitter, I'll b sure to provide this info to her #nohope
>  
> **British_Airways:** @767317 Sorry about that. The US phone number is 1-312-843-5794. The office is open from 09:00-13:00 EST. ^Chris


### Tesco

> **Customer:** @Tesco Thanks for replying, I went online (not on the app) and was able to check out straight away.
>  
> **Tesco:** @342473 Hi Kirst, I'm glad that you have been able to get your order through. It may be worth an uninstall/reinstall of the app in that case as your device may be running an old version. Thanks - Crai

> **Customer:** I don't like / can't afford waste but I've decided to abandon a @Tesco cauliflower as I've found a slug &amp; another grub on it. I wouldn't mind but it wasn't even an organic one but basic range #noc
>  
> **Tesco:** @680892 Hi Sally, I'm so sorry about this. I'd like to get this logged, fed back to our suppliers &amp; refund you. 1/4

> **Customer:** @Tesco How can I report  one of your drivers who I nearly ran into today due to going straight through a red light whilst on his mobile phone!
>  
> **Tesco:** @447688 Hi Jolene, i'm sorry my colleague wasn't driving safely and I hope you are OK. If it's OK with you I'd like to look in to this. 1/2

> **Customer:** @Tesco Yes it's the Superstore on Canal Road. Honestly the most disgusting toilets ever, not really any need for it.
>  
> **Tesco:** @234687 So I can make a log of your complaint today, can you please send a DM with your title, full name, address and email? Thanks - Struan 2/2 https://t.co/py5Z991Bme

> **Customer:** @Tesco Yes!! Never to be seen again?
>  
> **Tesco:** @790150 Hi Louisa, we do still sell this. The packaging has been updated though. Can you please DM me your local store so I can check it is still being stocked there? Thanks - Clare

---

## Final Brand Selection

### Selected Brand: `AppleSupport`

**Final score (default scenario):** 0.8645

**Selection rationale:** 'AppleSupport' ranked #1 in the default (balanced) scenario. In sensitivity analysis, 'AppleSupport' won 3/3 scenarios.

---

## Why This Brand Is Suitable

Based on actual computed metrics, `AppleSupport` is selected because:

1. **Volume:** 106,719 usable interactions — sufficient for intent clustering, RAG retrieval, and a 150–250 golden evaluation set.
2. **Coverage:** 99.9% of brand replies have their parent tweet in the dataset — high usability for historical grounding.
3. **Reconstructability:** 100.0% of linked pairs have valid inbound customer parents — conversation reconstruction is feasible.
4. **Completeness:** 0.9997 — high data integrity with minimal missing text or broken reply chains.
5. **Diversity:** 0.9575 — meaningful variety in customer issues enabling non-trivial intent taxonomy construction.

---

## Limitations

1. **Relative ranking:** All normalized scores are relative to the top-N candidate set. A brand excluded from the candidate list might have scored higher.

2. **Volume-filtered candidates:** Brands with <34,287 usable interactions were not evaluated. Some niche brands with excellent conversation quality may have been excluded.

3. **Diversity proxy:** TF-IDF + KMeans entropy is an approximation. Two brands with identical scores could have very different actual topic structures.

4. **Coverage ≠ quality:** Response coverage measures presence of parent tweets, not the semantic quality or helpfulness of brand responses.

5. **Historical Twitter data:** This dataset may not reflect current support operations, tone, or issue types for any of these brands.

6. **Sensitivity to weight choices:** Three scenarios were tested, but infinite weight combinations exist. Findings are robust within the tested space.

---

## What Is Misleading About The Headline Result?

1. **Weight subjectivity:** The 5 weights were chosen based on project priorities. Different weights (e.g., prioritizing diversity heavily) could change the ranking.

2. **Top-N filtering:** By evaluating only the top 10 by volume, we may miss a smaller brand with exceptional conversation quality and diversity.

3. **Diversity proxy imperfection:** A brand that handles only one type of issue (e.g., flight delays) might show high TF-IDF entropy if messages are lexically varied, while actually covering a narrow intent space.

4. **Temporal bias:** If the dataset is dominated by a specific time period (e.g., a major product launch), the interaction patterns may not generalize.

5. **Volume ≠ resolution quality:** A high volume of interactions does not guarantee that those interactions contain useful resolution patterns for grounding.

---

_Report auto-generated by `backend/scripts/select_brand.py`. Re-run to refresh with updated data or parameters._