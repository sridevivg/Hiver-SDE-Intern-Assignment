# REPORT — SupportGraph AI

> **Project Progress Report**
> Last Updated: Phase 4.5 — Human Taxonomy Review and Cluster Audit

---

## 1. Project Framing

### What Are We Building?

SupportGraph AI is an agentic customer support intelligence system. Given a real customer message, the system will:

1. **Classify the intent** — What is the customer asking for? What type of problem are they reporting?
2. **Retrieve historically similar resolutions** — What did human agents do when similar issues were raised in the past?
3. **Generate a grounded reply** — Produce a support response that is anchored in real, historical resolution patterns — not purely in LLM parametric knowledge.
4. **Decide on escalation** — Should this case be handled automatically, or does it require a human agent?

### What Does "Good" Look Like?

A successful system will demonstrate:

- **Intent classification that generalizes** — Labels emerged from the data, not from a hand-crafted taxonomy.
- **Grounded generation** — Generated replies can be traced back to retrieved historical support interactions.
- **Calibrated escalation** — The escalation classifier knows what it doesn't know and hands off appropriately.
- **Rigorous evaluation** — Performance is measured on a human-curated golden test set, not just on training data.
- **Failure documentation** — We know what the system gets wrong and why.
- **Reproducibility** — Every result can be regenerated from scratch with a single command.

### What "Good" Does NOT Mean

- High accuracy on a trivial, cleaned benchmark
- A demo that works only on cherry-picked examples
- A system that works but cannot be explained or audited

---

## 2. What Is Currently Implemented

### Phase 0 — Project Foundation ✅

- Clean modular project structure (frontend/backend separation)
- FastAPI backend with `/health` endpoint
- Environment-based configuration (Pydantic Settings)
- Structured logging
- React + TypeScript + Vite frontend scaffold
- Full `.gitignore`, `.env.example`, `requirements.txt`

### Phase 1 — Dataset Exploration ✅

- Dataset auto-detection and loading (`backend/app/data/loader.py`)
- Non-destructive cleaning analysis (`backend/app/data/cleaner.py`)
- Full exploration pipeline (`backend/app/data/explorer.py`)
- CLI exploration script (`backend/scripts/explore_dataset.py`)
- Unit tests for the data loader (`backend/tests/test_data_loader.py`)
- Generated artifacts (see `artifacts/` and `data/interim/`)

---

## 3. Phase 1 — Dataset Findings

> All statistics below come from running `python -m backend.scripts.explore_dataset` against the real dataset.
> Nothing is fabricated.

### Dataset Overview

| Property | Value |
|---|---|
| File | `twcs.csv` |
| Size | 492.6 MB |
| Rows | **2,811,774** |
| Columns | **7** |
| Duplicate rows | **0** (0.00%) |
| Fully null rows | **0** |

### Schema

| Column | Type | Missing |
|---|---|---|
| `tweet_id` | `int64` | 0 |
| `author_id` | `object` | 0 |
| `inbound` | `bool` | 0 |
| `created_at` | `object` | 0 |
| `text` | `object` | 0 |
| `response_tweet_id` | `object` | **1,040,629 (37.0%)** |
| `in_response_to_tweet_id` | `float64` | **794,335 (28.2%)** |

### Account Distribution

| Metric | Value |
|---|---|
| Unique authors | **702,777** |
| Inbound (customer) messages | **1,537,843** |
| Outbound (brand) messages | **1,273,931** |
| Brand account candidates | **108** |

### Message Statistics

| Statistic | Value |
|---|---|
| Total messages | 2,811,774 |
| Empty messages | 0 |
| Min length | 1 chars |
| Max length | 513 chars |
| Mean length | 113.9 chars |
| Median length | 115.0 chars |
| Std deviation | 52.4 chars |
| 95th percentile | 215.0 chars |

### Conversation Structure

| Metric | Value |
|---|---|
| Thread starters (no parent) | **794,335** |
| Reply messages (have parent) | **2,017,439** |
| Messages with responses | **1,771,145** |

### Top Brand Candidates (ranked by usable interactions)

| Brand | Usable Interactions | Response Ratio |
|---|---|---|
| AmazonHelp | 169,287 | 1.000 |
| AppleSupport | 106,719 | 1.000 |
| Uber_Support | 56,261 | 1.000 |
| SpotifyCares | 43,243 | 1.000 |
| Delta | 42,197 | 1.000 |
| Tesco | 38,501 | 1.000 |
| AmericanAir | 36,598 | 1.000 |
| TMobileHelp | 34,287 | 1.000 |
| comcastcares | 33,007 | 1.000 |
| British_Airways | 29,315 | 1.000 |

### Key Findings

**FACTS FROM THE DATA:**

1. The dataset contains **2,811,774 tweets** — significantly larger than typical benchmark datasets.
2. **Zero duplicate rows** — the dataset is already deduplicated at the raw level.
3. **Two columns have significant missing data**: `response_tweet_id` (37.0%) and `in_response_to_tweet_id` (28.2%). This is expected — thread starters have no parent, and final replies have no children.
4. **Clear inbound/outbound separation** exists via the `inbound` boolean column.
5. **108 brand accounts** identified — all are non-numeric handles with response_ratio ≈ 1.0 (pure outbound).
6. **Conversation threads are reconstructible** via `in_response_to_tweet_id` ↔ `response_tweet_id` linkages.
7. Tweet text length is tightly distributed (mean 113.9, std 52.4) — consistent with Twitter's character limits.

### Data Quality Issues Found

**FACTS FROM THE DATA:**

- `response_tweet_id` has 37.0% missing — expected for tweets that did not receive a response.
- `in_response_to_tweet_id` has 28.2% missing — expected for thread-starting tweets.
- No empty messages detected in the `text` column.
- No duplicate rows found — raw data quality is high.



---

## 4. Future Phases

> All future phases are **PLANNED — NOT IMPLEMENTED YET**.

| Phase | Goal |
|---|---|
| Phase 2 | Evidence-based brand selection |
| Phase 3 | Conversation thread reconstruction |
| Phase 4 | Data-driven intent discovery (unsupervised) |
| Phase 5 | Human-curated golden evaluation set |
| Phase 6 | Baseline models (TF-IDF retrieval, template replies) |
| Phase 7 | LangGraph agent system |
| Phase 8 | Historical RAG with vector retrieval |
| Phase 9 | Escalation policy learning |
| Phase 10 | Automated evaluation harness |
| Phase 11 | LLM-as-a-Judge quality scoring |
| Phase 12 | Failure mode analysis |
| Phase 13 | Frontend dashboard |
| Phase 14 | Final report and packaging |

---

## 5. Known Limitations (Phase 1 — Superseded by Phase 2)

- Conversation thread reconstruction has not been attempted yet.
- No ML models have been trained or evaluated.
- All Phase 1 findings are descriptive (exploratory) — no predictive claims are made.

---

## 6. Phase 2 — Brand Selection

> All statistics below come from running:
> `python -m backend.scripts.select_brand --top-n 10 --sample-size 5000 --random-seed 42`
> **Nothing is fabricated.**

### Methodology

Five evidence-based metrics, scored and weighted:

| Metric | Weight | Formula |
|---|---|---|
| Interaction Volume | 30% | Phase 1 `usable_interaction_count` |
| Response Coverage | 20% | replies with parent in dataset / total replies |
| Reconstructability | 20% | valid inbound parents / replies with parent found |
| Data Completeness | 15% | 0.5 × text_ok + 0.5 × chain_integrity |
| Issue Diversity | 15% | TF-IDF + KMeans entropy on 5,000 sampled messages |

All metrics normalized via min-max across the 10 candidate set.

### Raw Metric Values (FACTS FROM THE DATA)

| Brand | Volume | Coverage | Reconstruct | Completeness | Diversity |
|---|---|---|---|---|---|
| AppleSupport | 106,719 | 0.9993 | 1.0000 | 0.9997 | 0.9575 |
| British_Airways | 29,315 | 0.9992 | 1.0000 | 0.9996 | 0.9304 |
| Tesco | 38,501 | 0.9992 | 0.9999 | 0.9996 | 0.9157 |
| AmazonHelp | 169,287 | 0.9973 | 0.9999 | 0.9986 | 0.9102 |
| Uber_Support | 56,261 | 0.9988 | 0.9994 | 0.9994 | 0.9205 |
| Delta | 42,197 | 0.9989 | 0.9992 | 0.9994 | 0.9116 |
| AmericanAir | 36,598 | 0.9982 | 1.0000 | 0.9991 | 0.9386 |
| SpotifyCares | 43,243 | 0.9991 | 0.9974 | 0.9996 | 0.9176 |
| comcastcares | 33,007 | 0.9990 | 0.9984 | 0.9995 | 0.8850 |
| TMobileHelp | 34,287 | 0.9983 | 0.9996 | 0.9992 | 0.8803 |

### Final Ranking (Default Weights: 30/20/20/15/15)

| Rank | Brand | Final Score |
|---|---|---|
| #1 | **AppleSupport** | **0.8645** |
| #2 | British_Airways | 0.6189 |
| #3 | Tesco | 0.6108 |
| #4 | AmazonHelp | 0.5540 |
| #5 | Uber_Support | 0.5496 |
| #6 | Delta | 0.4959 |
| #7 | AmericanAir | 0.4824 |
| #8 | SpotifyCares | 0.4201 |
| #9 | comcastcares | 0.3915 |
| #10 | TMobileHelp | 0.3566 |

### Sensitivity Analysis (FACTS FROM COMPUTATION)

| Brand | Default | Volume-Focused | Quality-Focused |
|---|---|---|---|
| **AppleSupport** | **#1 ★** | **#1 ★** | **#1 ★** |
| British_Airways | #2 | #5 | #2 |
| Tesco | #3 | #3 | #3 |
| AmazonHelp | #4 | #2 | #7 |

**AppleSupport wins all 3 scenarios. The selection is robust.**

### Selected Brand

**`AppleSupport`** — score 0.8645

Key reasons (all data-derived):
- Perfect reconstructability (1.0000) — every linked parent is a valid customer tweet
- Highest diversity (0.9575) — widest range of issue types in sampled messages
- Highest completeness (0.9997) — minimal broken chains or missing text
- 106,719 usable interactions — sufficient for all downstream tasks

### Selection Limitations

- Ranking is relative to top-10 by volume; smaller brands not evaluated.
- Diversity metric is a TF-IDF proxy, not ground-truth topic modeling.
- Historical Twitter data may not reflect current operations.
- Volume-focused scenario would rank AmazonHelp #2 (vs #4 in default).

---

## 7. Phase 3 — Conversation Reconstruction

> All statistics below come from running:
> `python -m backend.scripts.build_conversations`
> **All values are calculated from the actual dataset — zero fabricated metrics.**

### Methodology

The reconstruction pipeline converts isolated tweets into ordered conversation trees:
1. **Reply Graph Extraction:** Backward (`in_response_to_tweet_id`) and forward (`response_tweet_id`) pointers build a directed reply graph.
2. **Connected Component Isolation:** Extracted all weakly connected subgraphs containing at least one customer tweet and at least one `AppleSupport` tweet.
3. **Deterministic Ordering:** Breadth-first traversal starting from the primary root, prioritized by topological depth, timestamp, and tweet ID.
4. **Structural Typing:** Deterministically classified threads into `customer_brand`, `customer_brand_customer`, `customer_brand_customer_brand`, `multi_turn`, `branched`, and `incomplete`.
5. **Quality Assignment:** Evaluated structural integrity into `HIGH`, `MEDIUM`, and `LOW` tiers.

### Reconstruction Statistics (AppleSupport)

| Metric | Value |
|---|---|
| Total `AppleSupport` Tweets | 106,860 |
| Total Candidate Components | 80,717 |
| **Total Valid Conversations** | **80,717** |
| **Total Messages in Conversations** | **238,907** |
| Mean Messages per Conversation | 2.96 |
| Median Messages per Conversation | 2.0 |
| Maximum Messages in Single Conversation | 282 |
| Branched Conversations | 4,926 (6.1%) |
| Conversations with Anomalies | 6,758 (8.4%) |

### Thread Type Distribution

| Thread Type | Count | Percentage | Description |
|---|---|---|---|
| `customer_brand` | 52,404 | 64.9% | Single customer question followed by single agent reply |
| `multi_turn` | 9,662 | 12.0% | Extended linear exchange (>4 messages or alternating) |
| `customer_brand_customer_brand` | 8,585 | 10.6% | Two-turn complete dialogue loop |
| `branched` | 4,912 | 6.1% | Non-linear tree (agent split into 1/2 + 2/2 or multiple inquiries) |
| `customer_brand_customer` | 4,828 | 6.0% | Customer question, agent reply, customer closing/acknowledgment |
| `incomplete` | 326 | 0.4% | Broken parent pointers to deleted or private tweets |

### Data Quality Distribution

| Quality Status | Count | Percentage | Criteria |
|---|---|---|---|
| **`HIGH`** | **73,959** | **91.6%** | Zero anomalies, unbroken graph, >= 2 messages, customer + brand present |
| **`MEDIUM`** | **6,432** | **8.0%** | Minor anomalies (forward response discrepancies or timestamps), fully usable |
| **`LOW`** | **326** | **0.4%** | Broken parent references to deleted tweets outside dataset |

### Anomaly Findings

- **Broken Parent Relationships:** 326 (parent tweet deleted or private at scrape time)
- **Timestamp Inversions:** 0
- **Cyclic Relationships:** 0
- **Orphan Messages:** 0

### Phase 3 Limitations & Headline Caveats

1. **Conversation Closure ≠ Customer Resolution:** A 2-turn conversation (`customer_brand`) does not prove the customer's hardware or software issue was resolved; it often indicates deflection to phone support, web portals, or private DMs.
2. **Short Dialogues Dominate:** 64.9% of all conversations are exactly 2 turns. Multi-turn deep technical troubleshooting accounts for approximately 28.6% of the dataset.
3. **Private Context Gap:** Detailed diagnostic information (serial numbers, logs) occurred in private messages outside the Twitter public dataset.

---

## 8. Phase 4 — Data-Driven Intent Discovery Findings

Phase 4 discovered a provisional candidate intent taxonomy directly from opening customer messages in high-quality `AppleSupport` conversations reconstructed in Phase 3.

### Discovery Corpus & Pipeline
- **Candidate Pool:** Customer opening messages (`role == 'customer'`, `depth == 0`) from `HIGH` quality conversations.
- **Total Eligible Messages:** 73,878 (21 messages excluded due to empty or whitespace-only text).
- **Discovery Sample:** 20,000 messages (deterministic random sample with seed `42`).
- **Embedding Model:** Local `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors, $L_2$ normalized). Zero external API calls.
- **Dimensionality Reduction:** 50-component PCA accounting for **64.6%** cumulative explained variance.

### Clustering Experiments across $K \in [4, 15]$

| $K$ | Silhouette Score | Davies-Bouldin Index | Calinski-Harabasz Score | Smallest Cluster | Largest Cluster | Tiny Count (<1%) | Notes |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 4 | 0.0644 | 3.8728 | 1407.2 | 3,829 | 6,316 | 0 | Under-segmented |
| 5 | 0.0571 | 3.4963 | 1258.5 | 3,135 | 5,252 | 0 | Lower silhouette |
| 6 | 0.0677 | 3.0627 | 1203.3 | 1,967 | 4,743 | 0 | Better separation |
| 7 | 0.0677 | 3.2979 | 1050.0 | 2,196 | 3,554 | 0 | Balanced sizes |
| **8** | **0.0815** | **3.1127** | **977.1** | **1,227** | **4,311** | **0** | **SELECTED: Peak Silhouette score; optimal balance** |
| 9 | 0.0702 | 3.1619 | 900.0 | 1,232 | 3,054 | 0 | Fragmentation begins |
| 10 | 0.0625 | 3.1635 | 847.2 | 705 | 2,893 | 0 | Moderate cohesion |
| 11 | 0.0560 | 3.1829 | 785.2 | 1,118 | 3,099 | 0 | Mid-range dip |
| 12 | 0.0691 | 2.8919 | 784.5 | 1,114 | 2,776 | 0 | Low Davies-Bouldin |
| 13 | 0.0644 | 3.0916 | 737.0 | 671 | 2,356 | 0 | Over-segmentation |
| 14 | 0.0703 | 2.9570 | 731.6 | 510 | 2,130 | 0 | Excess categories |
| 15 | 0.0758 | 2.9529 | 686.3 | 521 | 2,320 | 0 | Fragmented tail |

### Discovered Candidate Clusters ($K = 8$)

| Cluster ID | Size | % Corpus | Top TF-IDF Terms | Heuristic Proposed Label | Review Status |
|:---:|:---:|:---:|:---|:---|:---:|
| **0** | 3,108 | 15.54% | fix, user fix, letter, type, question | `keyboard_typing_glitch` | pending |
| **1** | 1,621 | 8.10% | battery, ios, iphone, battery life, life | `battery_power_issue` | pending |
| **2** | 3,220 | 16.10% | phone, iphone, user iphone, new, user user | `possible_display_hardware_issue` | pending |
| **3** | 2,690 | 13.45% | ios, ios11, iphone, update, user ios | `software_update_issue` | pending |
| **4** | 2,246 | 11.23% | store, iphone, icloud, itunes, user user | `app_store_billing_issue` | pending |
| **5** | 2,903 | 14.52% | music, macbook, user user, app, sierra | `possible_software_update_issue` | pending |
| **6** | 2,210 | 11.05% | update, phone, new, updated, new update | `software_update_issue` | pending |
| **7** | 2,002 | 10.01% | user url, user user, fix, shit, user fix | `possible_keyboard_typing_glitch` | pending |

### Provisional Status & Human Review
- `data/interim/provisional_intent_taxonomy.json` created with status `PROVISIONAL_HUMAN_REVIEW_REQUIRED`.
- All `final_intent_label` values remain `null` and `review_status` remains `pending` pending human review via `docs/intent_review_guide.md`.

---

## 9. Phase 4.5 — Taxonomy Audit Findings

Phase 4.5 performed a detailed quantitative and qualitative audit of the 8 candidate clusters discovered in Phase 4 across 20,000 sampled customer opening messages.

### Critical Interpretation of the 0.0815 Silhouette Score
- The Phase 4 silhouette score of $0.0815$ reflects substantial semantic overlap between customer problems on social media.
- **$K=8$ does NOT prove eight natural customer intents exist.** It represents the best practical mathematical partition among tested configurations.
- Human review is mandatory to convert this exploratory partition into an actionable operational taxonomy.

### Three-Tier Example Inspection & Coherence Summary

| Cluster ID | Size | % Corpus | Boundary Overlap Proxy | Generalization Status | Recommended Action | Top Focus |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **0** | 3,108 | 15.5% | 35.5% | `MIXED` | `REVIEW_FOR_MERGE` | Letter 'I' / autocorrect glitch |
| **1** | 1,621 | 8.1% | 43.1% | `MIXED` | `REVIEW_FOR_MERGE` | Battery drain & battery life |
| **2** | 3,220 | 16.1% | 75.5% | `GENERALIZABLE` | `REVIEW_FOR_SPLIT` | Display, touch, audio, phone freeze |
| **3** | 2,690 | 13.4% | 45.7% | `MIXED` | `REVIEW_FOR_MERGE` | iOS 11 update failures & bugs |
| **4** | 2,246 | 11.2% | 72.3% | `GENERALIZABLE` | `REVIEW_FOR_SPLIT` | App Store, iTunes, Apple ID |
| **5** | 2,903 | 14.5% | 82.6% | `GENERALIZABLE` | `REVIEW_FOR_MERGE` | Mac/Sierra, Safari, Apple Music |
| **6** | 2,210 | 11.1% | 56.2% | `MIXED` | `REVIEW_FOR_MERGE` | Post-update glitches & app issues |
| **7** | 2,002 | 10.0% | 48.9% | `GENERALIZABLE` | `REVIEW_FOR_MERGE` | Informal keyboard typing frustration |

### Resolution Pattern Audit (Across 20,000 Brand Responses)
- **Private Message Escalation (DM):** 51.4% (10,285 conversations)
- **Support Link Referrals:** 77.6% (15,519 conversations)
- **Clarification Questions:** 37.1% (7,417 conversations)
- **Troubleshooting Steps:** 23.3% (4,655 conversations)
- **Service / Genius Bar Referrals:** 0.6% (111 conversations)
- **Known Issue Confirmation:** 0.1% (28 conversations)
- *Core Architectural Principle:* Intent taxonomy (*what customer experiences*) and resolution taxonomy (*how support acts*) are strictly orthogonal.

### Candidate Merge & Split Analysis
- **Merge Candidates:**
  - Cluster 0 & Cluster 7 (centroid similarity 0.5144; shared keyboard/autocorrect problem)
  - Cluster 3 & Cluster 6 (both proposed `software_update_issue`; shared update failure symptoms)
  - Cluster 1 & Cluster 3 (centroid similarity 0.4455; customers reporting battery drain post-update)
- **Split Candidates:**
  - Cluster 2 (mixes screen damage, touch unresponsiveness, audio defects, and generic device slowness)
  - Cluster 5 (mixes desktop macOS Sierra/MacBook issues with cloud streaming Apple Music)

### Generated Review Artifacts
- `data/interim/cluster_merge_candidates.csv` (pairwise merge recommendations)
- `data/interim/cluster_split_candidates.csv` (internal heterogeneity signals)
- `data/interim/intent_taxonomy_human_review.csv` (review dossier with 3 example tiers, `final_intent_label = ""` and `review_status = "pending_human_review"`)
- `docs/final_intent_taxonomy_guidelines.md` (human taxonomy review protocol)

---

## 10. Phase 5 — Final Intent Taxonomy & Golden Set Foundation ✅

### Executive Summary
Phase 5 established the operational taxonomy candidate and golden evaluation infrastructure for SupportGraph AI:
- Defined a **9-intent candidate operational taxonomy** grounded in empirical cluster evidence from Phases 4 & 4.5.
- Implemented an evidence-based merge strategy (combining redundant update and keyboard clusters) and split strategy (decoupling account lockouts from billing disputes; isolating screen repairs from audio defects).
- Formulated a comprehensive Standard Operating Procedure in `docs/golden_set_annotation_guidelines.md`.
- Generated a stratified, deduplicated candidate evaluation set of exactly **200 real customer opening messages** (`seed=42`) enriched with brand dialogue context.
- Implemented strict label validation rejecting empty labels, duplicates, or out-of-taxonomy classes.
- Implemented an inter-annotator agreement engine measuring raw agreement and Cohen's Kappa, with an explicit scientific honesty rule reporting `INTER_ANNOTATOR_AGREEMENT_NOT_AVAILABLE` when only one annotator dataset exists.
- Built an immutable version-freezing pipeline with SHA256 checksum verification (`freeze_golden_set.py`).

### The 9 Operational Candidate Intents

| Intent ID | Intent Name | Source Clusters | Class Count in 20k Corpus | Golden Sample Quota |
|:---:|:---|:---:|:---:|:---:|
| `intent_01` | `software_update_problem` | 3, 6 | 4,900 | 23 |
| `intent_02` | `battery_power_issue` | 1 | 1,621 | 22 |
| `intent_03` | `display_touch_issue` | 2 | 1,142 | 22 |
| `intent_04` | `account_access_issue` | 4 | 348 | 22 |
| `intent_05` | `billing_purchase_issue` | 4 | 828 | 22 |
| `intent_06` | `keyboard_typing_issue` | 0, 7 | 5,110 | 22 |
| `intent_07` | `mac_software_issue` | 5 | 1,424 | 22 |
| `intent_08` | `hardware_audio_connection_issue` | 2 | 876 | 22 |
| `intent_09` | `general_device_support` | 2, 4, 5 | 3,671 | 23 |
| **Total** | **9 Operational Intents** | **All 8 Clusters** | **19,920 unique** | **200** |

### Golden Sampling Deduplication & Context
- Total candidate corpus: 20,000 messages.
- Duplicate `tweet_id`s removed: 0.
- Duplicate `normalized_text`s removed: 80.
- Brand response context enriched: 80,717 conversations indexed from `data/processed/conversation_messages.parquet`.
- Candidate evaluation dataset: `data/golden/golden_set_candidates.csv` (200 records).
- Evaluation template: `data/golden/golden_set_annotation_template.csv` (all `annotation_label` values strictly empty).
- Metadata manifest: `data/golden/golden_sampling_manifest.json`.

---

## 11. Phase 5.5 — AI-Assisted Human Annotation Workflow ✅

### Methodology & Scientific Framing
Phase 5.5 deployed a human-in-the-loop AI-assisted annotation workflow to accelerate the curation of the 200-example Golden Evaluation Set without compromising empirical rigor:
- **Model Engine:** Groq API leveraging `openai/gpt-oss-20b`.
- **Strict Role Demarcation:** The model drafts structured candidate recommendations (`model_suggested_label`, `model_confidence`, `model_reasoning_summary`, `model_needs_human_review`).
- **Isolation of Ground Truth:** The model has strictly **zero access** to `annotation_label`, `annotator`, or `annotation_status`. These fields remain `""` and `"pending_human_review"` until actively reviewed by a human annotator.
- **Interactive Review CLI:** [`backend/scripts/review_golden_labels.py`](backend/scripts/review_golden_labels.py) allows reviewers to inspect the customer message, conversation context, and AI suggestion, and execute one-key decisions: Accept (`A`), Override with taxonomy intent (`1–9`), mark Unclear (`U`), or Skip (`S`).
- **Model-Human Audit Metric:** Model-human alignment is strictly measured as `MODEL_HUMAN_AGREEMENT`, never conflated with human inter-rater reliability or Cohen's Kappa.

### Generated Artifacts
- `data/golden/golden_set_ai_suggestions.csv`: 200 records with advisory suggestions (ground-truth fields unpopulated).
- `data/golden/golden_annotation_progress.json`: Real-time progress manifest tracking human review completion.
- `artifacts/reports/phase_5_5_assisted_annotation.md`: Detailed scientific report on AI assistance vs. ground truth.

---

## 12. Known Limitations (Phase 5.5 Complete)

1. **Human Annotation Pending:** While AI suggestions have been enabled, final ground-truth labels require interactive human review before freezing.
2. **Cognitive Anchoring Bias Risk:** Presenting AI suggestions creates potential cognitive bias toward accepting model defaults; reviewers must exercise critical judgment on borderline cases.
3. **Twitter Constraint (280 Characters):** Customer opening messages on Twitter are concise and colloquial; annotators must occasionally rely on `conversation_context` (the brand's reply) to clarify technical ambiguity.
4. **Single-Brand Specificity:** The 9-intent taxonomy is tailored specifically to Apple consumer support and does not generalize directly to unrelated domains (e.g. airlines) without re-discovery.
5. **No Downstream ML Implemented:** Supervised classification, RAG retrieval, LangGraph orchestration, and automated evaluation runs are strictly deferred to future phases.




