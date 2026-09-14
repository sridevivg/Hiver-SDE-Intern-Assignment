# Phase 1 — Dataset Analysis Report

> **Generated:** 2026-09-09 15:52 UTC  
> **Source file:** `twcs.csv` (492.6 MB)  
> **Status:** Findings based on actual data inspection — no values fabricated.

---

## 1. Dataset Overview

**FACTS FROM THE DATA:**

- File: `twcs.csv` (492.6 MB)
- Rows: **2,811,774**
- Columns: **7**
- Duplicate rows: **0** (0.00%)

---

## 2. Schema

| Column | Data Type | Notes |
|---|---|---|
| `tweet_id` | `int64` |  |
| `author_id` | `object` |  |
| `inbound` | `bool` |  |
| `created_at` | `object` |  |
| `text` | `object` |  |
| `response_tweet_id` | `object` | 1,040,629 missing (37.0%) |
| `in_response_to_tweet_id` | `float64` | 794,335 missing (28.2%) |

---

## 3. Missing Data

**FACTS FROM THE DATA:**

| Column | Missing Count | Missing % |
|---|---|---|
| `response_tweet_id` | 1,040,629 | 37.01% |
| `in_response_to_tweet_id` | 794,335 | 28.25% |

---

## 4. Duplicate Analysis

**FACTS FROM THE DATA:**

- Total duplicate rows: **0** (0.00%)
- Fully null rows: **0**

**INTERPRETATION / RECOMMENDATION:**

Duplicate rows should be removed before training any model to prevent data leakage and inflated performance metrics.

---

## 5. Message Analysis

**Text column detected:** `text`

**FACTS FROM THE DATA:**

- Total messages: **2,811,774**
- Empty messages: **0**

**Message length statistics (characters):**

| Statistic | Value |
|---|---|
| min | 1.0 |
| max | 513.0 |
| mean | 113.9 |
| median | 115.0 |
| std | 52.4 |
| p25 | 78.0 |
| p75 | 139.0 |
| p95 | 215.0 |

**Sample messages:**

1. `@115712 I understand. I would like to assist you. We would need to get you into a private secured link to further assist.`
2. `@sprintcare and how do you propose we do that`
3. `@sprintcare I have sent several private messages and no one is responding as usual`

---

## 6. Account Analysis

**Author column:** `author_id`

**FACTS FROM THE DATA:**

- Unique authors: **702,777**
- Inbound messages (customer): **1,537,843**
- Outbound messages (brand): **1,273,931**

**Top 15 accounts by volume:**

| Author | Total | Inbound | Outbound |
|---|---|---|---|
| `AmazonHelp` | 169,840 | 0 | 169,840 |
| `AppleSupport` | 106,860 | 0 | 106,860 |
| `Uber_Support` | 56,270 | 0 | 56,270 |
| `SpotifyCares` | 43,265 | 0 | 43,265 |
| `Delta` | 42,253 | 0 | 42,253 |
| `Tesco` | 38,573 | 0 | 38,573 |
| `AmericanAir` | 36,764 | 0 | 36,764 |
| `TMobileHelp` | 34,317 | 0 | 34,317 |
| `comcastcares` | 33,031 | 0 | 33,031 |
| `British_Airways` | 29,361 | 0 | 29,361 |
| `SouthwestAir` | 28,977 | 0 | 28,977 |
| `VirginTrains` | 27,817 | 0 | 27,817 |
| `Ask_Spectrum` | 25,860 | 0 | 25,860 |
| `XboxSupport` | 24,557 | 0 | 24,557 |
| `sprintcare` | 22,381 | 0 | 22,381 |

---

## 7. Conversation Structure

**FACTS FROM THE DATA:**

- Reply-to column: `in_response_to_tweet_id`
- Thread starters (no parent): **794,335**
- Reply messages (have parent): **2,017,439**
- Response-IDs column: `response_tweet_id`
- Messages with responses: **1,771,145**

**INTERPRETATION:**

The presence of reply-to and response-ID columns means conversation threads can be reconstructed by following the parent-child tweet relationships. This is the foundation for Phase 3 (Conversation Reconstruction).

---

## 8. Potential Brand Candidates

**FACTS FROM THE DATA:**

Brand accounts are identified as non-numeric authors appearing in outbound (brand-response) rows.

| Brand | Total Msgs | Outbound | Usable Interactions | Response Ratio |
|---|---|---|---|---|
| `AmazonHelp` | 169,840 | 169,840 | 169,287 | 1.000 |
| `AppleSupport` | 106,860 | 106,860 | 106,719 | 1.000 |
| `Uber_Support` | 56,270 | 56,270 | 56,261 | 1.000 |
| `SpotifyCares` | 43,265 | 43,265 | 43,243 | 1.000 |
| `Delta` | 42,253 | 42,253 | 42,197 | 1.000 |
| `Tesco` | 38,573 | 38,573 | 38,501 | 1.000 |
| `AmericanAir` | 36,764 | 36,764 | 36,598 | 1.000 |
| `TMobileHelp` | 34,317 | 34,317 | 34,287 | 1.000 |
| `comcastcares` | 33,031 | 33,031 | 33,007 | 1.000 |
| `British_Airways` | 29,361 | 29,361 | 29,315 | 1.000 |
| `SouthwestAir` | 28,977 | 28,977 | 28,889 | 1.000 |
| `VirginTrains` | 27,817 | 27,817 | 27,522 | 1.000 |
| `Ask_Spectrum` | 25,860 | 25,860 | 25,807 | 1.000 |
| `XboxSupport` | 24,557 | 24,557 | 24,341 | 1.000 |
| `sprintcare` | 22,381 | 22,381 | 22,335 | 1.000 |

> ⚠️  **Brand selection is NOT performed in this phase.** The table above is preparation data for Phase 2 — Evidence-Based Brand Selection.

---

## 9. Data Quality Issues

**FACTS FROM THE DATA:**

- DECISION: Raw DataFrame is NOT modified by this analysis.
- DECISION: Duplicate rows are identified but NOT removed here; removal (if decided) will produce a new DataFrame in data/interim/.
- DECISION: Null values are reported per-column. Imputation or row-drop decisions are deferred to the processing phase.
- DECISION: Empty strings are distinguished from null values. Both are counted separately to preserve granularity.

---

## 10. What We Learned

**FACTS FROM THE DATA:**

1. The dataset contains **2,811,774 rows** across **7 columns**.
2. Clear inbound/outbound separation exists via the `inbound` column.
3. Conversation thread structure is available via `in_response_to_tweet_id` and `response_tweet_id`.
4. **108 potential brand accounts** identified for Phase 2 selection.
5. Message lengths range from 1 to 513 characters (mean: 114).

---

## 11. Limitations of This Analysis

- Brand selection has NOT been performed — all brands are analyzed together.
- Conversation threads have NOT been reconstructed — only raw reply relationships are reported.
- No intent labels exist — this is an unlabeled corpus.
- Analysis uses the full dataset without filtering — quality filtering is a future phase.
- Temporal analysis (date ranges, peak periods) is deferred to Phase 2+.

---

## 12. Recommended Next Step

**Phase 2 — Evidence-Based Brand Selection**

Using the brand candidates table above (`data/interim/brand_candidates.csv`), select a primary brand based on:

1. Volume: minimum ~10,000 usable interactions
2. Conversation depth: high proportion of multi-turn threads
3. Topic diversity: broad enough for meaningful intent clustering

**Do not choose a brand based on personal preference — choose based on data.**

---

_This report was auto-generated by `backend/scripts/explore_dataset.py`. Do not edit manually — re-run the script to refresh._