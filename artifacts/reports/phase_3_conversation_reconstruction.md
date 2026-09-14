# Phase 3 — Conversation Reconstruction

> **Generated:** 2026-09-09 17:37 UTC  
> **Selected Brand:** `AppleSupport`  
> **Method:** Deterministic reply graph component assembly  
> **Principle:** Strictly observed message relationships — zero fabricated context  

---

## Objective

Transform tweet-level interaction records for `AppleSupport` into structured,
reproducible conversation threads. This dataset serves as the factual foundation
for downstream intent discovery, historical RAG retrieval, and grounded reply generation.

---

## Input Data

- **Dataset Source:** Kaggle `thoughtvector/customer-support-on-twitter` (`twcs.csv`)
- **Total Dataset Rows:** 2,811,774 tweets
- **Total `AppleSupport` Tweets:** 106,860
- **Relevant Schema Fields:** `tweet_id`, `author_id`, `inbound`, `created_at`, `text`, `in_response_to_tweet_id`, `response_tweet_id`

---

## Selected Brand

**`AppleSupport`** was selected scientifically in Phase 2 using a 5-metric multi-criteria evaluation:
- 106,719 usable interaction volume (Phase 1)
- 99.93% response coverage (parent in dataset)
- 100.00% conversation reconstructability (linked parent is valid customer tweet)
- 0.9997 data completeness floor
- 0.9575 issue diversity score
- Winner in 3 out of 3 sensitivity analysis scenarios

---

## Reply Graph Model

The conversation pipeline models tweets as **nodes** and explicit replies as **directed edges**:

$$\text{Parent Tweet } A \longrightarrow \text{Child Reply } B$$

Where Child $B$ explicitly records `in_response_to_tweet_id = A`.
A conversation is formally defined as a **weakly connected component** of the reply graph
containing at least one customer tweet (`inbound=True`) and at least one `AppleSupport` tweet.

---

## Relationship Parsing

The Twitter dataset provides two fields containing reply relationships:

1. **`in_response_to_tweet_id` (Backward Pointer):**
   - Indicates the parent tweet that this tweet replies to.
   - Single scalar ID (or null for thread starters).
   - Used as the primary ground-truth backward edge.

2. **`response_tweet_id` (Forward Pointer):**
   - Indicates one or more child tweets replying to this tweet.
   - Stored as comma-separated integers (e.g. `'5,7'`, `'9,6,10'`).
   - Parsed by splitting on comma, trimming whitespace, and casting to integer IDs.

**Edge Agreement Classification:**
- `both`: Both child has `in_response_to_tweet_id = parent` AND parent lists child in `response_tweet_id`.
- `in_response_to`: Child points to parent, but parent does not index child in `response_tweet_id`.
- `response_tweet`: Parent points to child, but child is outside the dataset or missing `in_response_to`.

---

## Conversation Reconstruction Algorithm

The 16-step reconstruction follows strict scientific principles:
1. Load selected brand from Phase 2 artifact (`selected_brand.json`).
2. Validate dataset schema against required 7 columns.
3. Filter all tweets authored by the selected brand.
4. Traverse the full reply graph to find all weakly connected components containing the brand.
5. Subset the dataset to connected tweets to maximize memory efficiency.
6. Build bidirectional adjacency and forward/backward lookup tables.
7. Validate edge consistency across both pointer fields.
8. Detect anomalies (missing parents, missing children, timestamp reversals, cycles, self-references).
9. Extract connected conversation components.
10. Discard components lacking either role (must contain both customer and brand).
11. Identify conversation roots (nodes with in-degree 0 or parent outside dataset).
12. Assign deterministic conversation IDs: `conv_{brand}_{root_tweet_id}`.
13. Generate canonical breadth-first graph traversal ordered by depth, timestamp, and tweet ID.
14. Classify thread type based strictly on observed structure.
15. Assign deterministic data quality status (`high`, `medium`, `low`).
16. Export all processed artifacts to `data/processed/`.

---

## Conversation Root Definition

A tweet is defined as a conversation root if:
- It has no parent pointer (`in_response_to_tweet_id IS NULL`), OR
- Its parent pointer references a tweet ID outside the dataset (`broken_parent_root`).

When a connected component contains multiple candidate roots (e.g. branched threads or joined discussions),
the primary root is deterministically selected as the root with the **earliest timestamp**
(with lowest numeric `tweet_id` as tiebreaker).

---

## Message Ordering

Messages are ordered using a 3-tier hierarchy:
1. **Primary: Graph Depth (Parent-Child Topology)** — Parent messages strictly precede children.
2. **Secondary: Timestamp** — Sibling branches are ordered chronologically.
3. **Tertiary: Numeric Tweet ID** — Deterministic tiebreaker for identical timestamps.

If a child message has an earlier timestamp than its parent, **graph relationship takes precedence**,
and a `timestamp_anomaly` is formally recorded.

---

## Branching Strategy

Branching occurs when a single tweet receives multiple direct replies (e.g. multi-tweet agent answers or multiple customer follow-ups).
- Branching is **never flattened** into an artificial linear chain.
- Graph edges record the exact branching topology.
- Conversations with branching are flagged `has_branching = True` and typed as `branched`.

---

## Anomaly Detection

Every structural imperfection is recorded without halting execution:

| Anomaly Type | Count | Description |
|---|---|---|
| `broken_parent_relationships` | 326 | Child references parent not present in the dataset |
| `timestamp_anomalies` | 0 | Child timestamp is earlier than parent timestamp |
| `cycles_detected` | 0 | Cyclic reply relationships detected in graph |
| `orphan_messages` | 0 | Disconnected messages within component |
| **Conversations with Anomalies** | **6,758** | Any anomaly detected within conversation |

---

## Thread Type Distribution

| Thread Type | Count | % of Valid Conversations | Description |
|---|---|---|---|
| `customer_brand` | 52,404 | 64.9% |
| `multi_turn` | 9,662 | 12.0% |
| `customer_brand_customer_brand` | 8,585 | 10.6% |
| `branched` | 4,912 | 6.1% |
| `customer_brand_customer` | 4,828 | 6.0% |
| `incomplete` | 326 | 0.4% |

---

## Conversation Quality Distribution

Deterministic quality rules:
- **HIGH:** No anomalies, complete graph, >=2 messages, contains customer and brand.
- **MEDIUM:** Contains customer and brand, minor anomalies (timestamp or missing child pointers), graph still usable.
- **LOW:** Broken parent pointers, cycles, or incomplete structure.

| Quality Status | Count | % of Valid Conversations |
|---|---|---|
| `HIGH` | 73,959 | 91.6% |
| `MEDIUM` | 6,432 | 8.0% |
| `LOW` | 326 | 0.4% |

---

## Reconstruction Statistics

| Metric | Value |
|---|---|
| Total `AppleSupport` Tweets | 106,860 |
| Total Candidate Components | 80,717 |
| Total Valid Conversations (Customer + Brand) | 80,717 |
| Total Messages in Conversations | 238,907 |
| Mean Messages per Conversation | 2.96 |
| Median Messages per Conversation | 2.0 |
| Maximum Messages in Single Conversation | 282 |
| Branched Conversations | 4,926 |

---

## Sample Conversations

> Author IDs and personal handles have been anonymized for privacy. Text excerpts truncated.

### Sample 1: `conv_AppleSupport_700` (multi_turn, 5 messages)
- [depth 0] **Customer**: @AppleSupport why are my I️’s changing not showing up correctly on any of my social media platforms? https://t.co/GyRvpyVnkE
- [depth 1] **Customer**: @AppleSupport  https://t.co/NV0yucs0lB
- [depth 2] **AppleSupport**: @115854 We're here for you. Which version of the iOS are you running? Check from Settings &gt; General &gt; About.
- [depth 3] **Customer**: @AppleSupport The newest update. I️ made sure to download it yesterday.
- [depth 4] **AppleSupport**: @115854 Lets take a closer look into this issue. Select the following link to join us in a DM and we'll go from there. https://t.co/GDrqU22YpT

### Sample 2: `conv_AppleSupport_711` (branched, 11 messages)
- [depth 0] **Customer**: What is up with this “ I️ “
- [depth 1] **Customer**: Why are my I’s question marks ?
- [depth 2] **Customer**: @AppleSupport I️ need answers because it’s annoying 🙃
- [depth 3] **AppleSupport**: @115855 We'd like to look into this with you. Which model do you have and is iOS 11.1 installed? Any steps tried so far?
- [depth 4] **Customer**: @AppleSupport I️ have an iPhone 7 Plus and yes I️ do
- [depth 5] **AppleSupport**: @115855 That's great it has iOS 11.1 as we can rule out being outdated. Any steps tried since this started? Do you recall when it started?
- [depth 6] **Customer**: @AppleSupport Last night
- [depth 6] **Customer**: @AppleSupport This is what it looks like https://t.co/XCQU2l4xUB
- [depth 7] **AppleSupport**: @115855 Any steps tried since it started last night?
- [depth 8] **Customer**: @AppleSupport Tried resetting my settings .. restarting my phone .. all that
- [depth 9] **AppleSupport**: @115855 Let's go to DM for the next steps. DM us here: https://t.co/GDrqU22YpT

### Sample 3: `conv_AppleSupport_719` (customer_brand_customer_brand, 4 messages)
- [depth 0] **Customer**: Tf is wrong with my keyboard @115858
- [depth 1] **AppleSupport**: @115857 Fill us in on what is happening, then we can help out from there.
- [depth 2] **Customer**: @AppleSupport This is what is happening... https://t.co/X3SZSJXfAT
- [depth 3] **AppleSupport**: @115857 We'd like to investigate further with you. Send us a DM and we can troubleshoot more from there. https://t.co/GDrqU22YpT

---

## Limitations

1. **Missing Tweets:** Tweets deleted or private at scrape time create broken parent pointers.
2. **Public Context Only:** Twitter exchanges frequently deflect to Private DMs ('Please DM us your serial number'). The resolution occurs off-graph.
3. **Branched Discussions:** Multiple agents or users replying to a single tweet creates trees, not linear chat logs.
4. **Timestamp Anomalies:** Client-side clock drift occasionally produces child tweets with timestamps earlier than their parents.
5. **Inability to Prove Issue Resolution:** A closed conversation does not prove the customer's problem was resolved.
6. **Observed vs Hidden Structure:** Reconstruction captures strictly what was publicly observed on Twitter.

---

## What Is Misleading About The Headline Number?

A headline stating **'80,717 reconstructed conversations'** can easily be misinterpreted:

1. **Conversation Count ≠ Resolved Customer Issues:** Many conversations end with deflection to Apple Support web forms, phone numbers, or DMs without resolution evidence.
2. **A Complete Graph Does Not Imply High Satisfaction:** A customer may have abandoned the exchange out of frustration after 2 turns.
3. **Short Conversations Overrepresented:** The majority of conversations are exactly 2 turns (`Customer -> Brand`), which represents initial triage rather than complete troubleshooting.
4. **Private Information Gap:** The actual technical solution often occurred in private DMs or Apple retail appointments not present in the dataset.

---

_Report auto-generated by `backend/scripts/build_conversations.py`._
