# Phase 5 — Final Intent Taxonomy and Golden Set Foundation

> **SupportGraph AI — Scientific Methodology & Golden Evaluation Design Report**  
> **Brand Under Analysis:** AppleSupport  
> **Dataset Source:** Kaggle Customer Support on Twitter (thoughtvector/customer-support-on-twitter)  
> **Date:** September 2026  
> **Status:** Phase 5 Infrastructure Complete — Human Annotation Pending  

---

## 1. Objective

Phase 5 bridges the gap between unsupervised machine learning discovery and rigorous, trustworthy evaluation. The goal is to construct a **production-grade, operational customer intent taxonomy** and establish an **immutable, stratified golden evaluation dataset** of 200 real customer opening messages for `AppleSupport`.

Crucially, Phase 5 establishes the scientific infrastructure required to guarantee evaluation integrity:
- Defining 9 operational candidate intents grounded in empirical cluster evidence.
- Providing standardized human annotation guidelines.
- Sampling exactly 200 representative customer opening messages stratified across intents with zero duplicates.
- Providing strict automated validation to enforce data quality and block silent label corruption.
- Supporting measurable inter-annotator agreement (Cohen's Kappa) while maintaining complete scientific honesty when only a single annotator is available.
- Establishing an immutable version-freezing workflow with SHA256 cryptographic verification to prevent test set contamination.

---

## 2. Why Unsupervised Clusters Are NOT Ground-Truth Labels

A common failure mode in applied machine learning is treating unsupervised cluster IDs (such as $K=8$ from $K$-Means or Gaussian Mixture Models) as true classification labels. In SupportGraph AI, we explicitly reject this practice based on core scientific principles:

1. **Geometry is Not Semantics:** MiniBatchKMeans partitions points based purely on Euclidean distance in a 384-dimensional PCA-reduced dense vector space. It clusters on word co-occurrence, sentence length, and syntactic structure rather than operational business logic.
2. **Low Coherence and Continuous Manifold:** In Phase 4, the best mathematical clustering achieved a Silhouette score of $S = 0.0815$. This confirms that customer support tweets exist on a dense, continuous semantic continuum with fuzzy boundaries rather than well-separated discrete clusters.
3. **Temporal Anomaly Sensitivity:** Unsupervised clusters are heavily distorted by high-frequency temporal events. For example, Clusters 0 and 7 were dominated by tweets complaining about the November 2017 iOS autocorrect bug (where typing the letter 'I' produced an exclamation mark `[?]`). An automated system would treat this temporary glitch as a permanent intent (`ios_11_autocorrect_bug`), whereas operational customer support requires an enduring category: `keyboard_typing_issue`.
4. **Resolution Inconsistency:** Unsupervised clusters frequently lump together issues that require fundamentally different resolution workflows. For instance, Cluster 4 combined Apple ID account password lockouts (which require identity verification) with iTunes billing refund requests (which require purchase order lookups). Treating Cluster 4 as a single label would make downstream response routing impossible.

Therefore, unsupervised clustering serves exclusively as an **exploratory discovery aid**, not ground truth. The operational taxonomy must be constructed from reviewed human evidence.

---

## 3. Final Taxonomy Candidate Methodology

The final operational taxonomy was formulated by synthesizing the empirical findings from Phase 4 and Phase 4.5. The taxonomy satisfies four fundamental criteria:
- **Operational Focus:** Labels represent **WHAT THE CUSTOMER NEEDS HELP WITH** (the underlying technical or transactional problem), not emotional tone, product names alone, or brand resolution tactics.
- **Enduring Relevance:** Categories are invariant to specific patch versions (e.g. iOS 11.0.1 vs 11.0.3) or temporary historical events.
- **Actionable Routing:** Every intent maps to a distinct operational support capability or troubleshooting path.
- **Parsimony:** The taxonomy is constrained to 9 mutually distinct, manageable operational intents (falling squarely within the target 7–10 range).

### The 9 Candidate Operational Intents

| Intent ID | Intent Name | Source Clusters | Operational Rationale |
|:---|:---|:---|:---|
| `intent_01` | `software_update_problem` | 3, 6 | Merged redundant Clusters 3 and 6. Covers installation failures, bricked devices, reboot loops, and post-update app crashes. |
| `intent_02` | `battery_power_issue` | 1 | Preserved high-cohesion Cluster 1. Covers abnormal battery drain, sudden power shutoffs, overheating, and charging defects. |
| `intent_03` | `display_touch_issue` | 2 | Split from Cluster 2. Focuses exclusively on touchscreen unresponsiveness, digitizer malfunctions, screen flickering, and cracked glass. |
| `intent_04` | `account_access_issue` | 4 | Split from Cluster 4. Covers Apple ID login failures, two-factor authentication lockout, password resets, and disabled accounts. |
| `intent_05` | `billing_purchase_issue` | 4 | Split from Cluster 4. Covers App Store subscription charges, accidental in-app purchases, refund disputes, and credit card declines. |
| `intent_06` | `keyboard_typing_issue` | 0, 7 | Merged Clusters 0 and 7. Generalizes the transient 2017 autocorrect bug into an enduring operational category covering keyboard lag, dictation, and character rendering glitches. |
| `intent_07` | `mac_software_issue` | 5 | Split from Cluster 5. Covers desktop macOS crashes, Safari freezes, kernel panics, and Mac OS installation distinct from mobile iOS. |
| `intent_08` | `hardware_audio_connection_issue` | 2 | Split from Cluster 2. Covers physical audio defects (microphones, speakers, AirPods), Bluetooth pairing failures, and Wi-Fi hardware connection drops. |
| `intent_09` | `general_device_support` | 2, 4, 5 | Serves as a triage/routing class for general setup, store appointments, warranty questions, or compound multi-issue opening messages. |

---

## 4. Sampling Methodology

The golden evaluation dataset must serve as an unbiased, reliable benchmark for evaluating future intent classifiers, RAG retrieval modules, and agentic workflows.

### Sampling Pipeline
1. **Source Corpus:** Drawn from the 20,000 clean customer opening messages compiled in `data/interim/intent_discovery_corpus.csv`.
2. **Deduplication:**
   - Strict `tweet_id` uniqueness (eliminates retweets or multi-threaded message collisions).
   - Strict `normalized_text` uniqueness (eliminates bot spam, repeated boilerplate complaints, and duplicate inquiries).
   - In the 20,000 corpus, 80 exact text duplicates were removed, leaving 19,920 completely unique candidate customer messages.
3. **Stratified Allocation:**
   - To avoid minority intent starvation, sampling is stratified across all 9 candidate intents.
   - For a target of $N = 200$, quotas are computed dynamically: 7 intents receive exactly 22 samples, and 2 intents receive 23 samples ($\sum = 200$).
4. **Deterministic Execution:**
   - All random subsampling is controlled via `random_seed = 42` using a fixed NumPy random state generator.
5. **Contextual Enrichment:**
   - For each customer tweet, the sampler joins conversation history to retrieve the brand's first turn response from `data/processed/conversation_messages.parquet` (covering 80,717 conversations). This provides human annotators with clarifying context when the opening tweet is ambiguous.
6. **Zero Automated Labeling:**
   - The field `annotation_label` is strictly initialized to empty string `""`, `annotation_status` to `"pending"`, and `annotator` to `""`. No synthetic or machine-generated labels are permitted in the golden candidate set.

---

## 5. Golden Set Design & Schema

The golden set is stored in standardized CSV format with full quoting (`QUOTE_NONNUMERIC`) to prevent multiline text truncation. Every record adheres to a strict 12-column schema:

| Column Name | Type | Description |
|:---|:---|:---|
| `golden_id` | String | Unique evaluation identifier (`gold_001` through `gold_200`). |
| `tweet_id` | String | Original Twitter status ID. |
| `conversation_id` | String | SupportGraph conversation identifier linking to dialogue thread. |
| `customer_message` | String | Full raw customer opening message. |
| `normalized_message` | String | Preprocessed text (handles and URLs normalized). |
| `candidate_intent` | String | Heuristic discovery intent used solely for stratification balance. |
| `source_cluster` | Integer | Original Phase 4 unsupervised cluster ID (0–7). |
| `conversation_context` | String | First brand reply from historical dialogue to aid disambiguation. |
| `annotation_label` | String | Ground-truth human label (empty in template; populated by annotators). |
| `annotation_status` | String | Progress indicator (`pending` vs `completed`). |
| `annotator` | String | Name or ID of human reviewer. |
| `notes` | String | Annotator rationale, borderline notes, or ambiguity comments. |

---

## 6. Human Annotation Workflow

Human annotation is governed by the Standard Operating Procedure detailed in [`docs/golden_set_annotation_guidelines.md`](file:///Users/sridevi/Desktop/SupportGraph-AI/docs/golden_set_annotation_guidelines.md).

### Core Annotation Rules:
1. **Unit of Annotation:** The customer opening message (`customer_message`). The annotator may consult `conversation_context` only when the opening tweet lacks technical context.
2. **Primary Intent Rule:** If a customer mentions multiple issues (e.g. *"Updated to iOS 11 and now my battery is draining fast"*), annotators assign the **root triggering problem** (`software_update_problem`) or the most urgent operational blocker. Single-label assignment is enforced.
3. **Product Names are Entities, Not Intents:** Mentions of *iPhone 8*, *MacBook Pro*, or *Apple Watch* are contextual hardware entities. Annotators must classify based on the operational failure mode.
4. **Ambiguity Tracking:** Truly unintelligible, truncated, or non-English messages must be labeled as `unclear_needs_review`. They must not be forced into an arbitrary category.
5. **No Synthetic Shortcuts:** Annotations must be performed by human judgment, not LLM prompts or pattern-matching scripts.

---

## 7. Label Validation Workflow

Automated validation is implemented in `backend/app/evaluation/label_validation.py` and invoked via CLI:
```bash
python -m backend.scripts.validate_golden_set --input data/golden/golden_set_annotation_template.csv
```

The validation suite enforces 6 non-negotiable checks:
1. **Schema Integrity:** Verifies presence of all 12 mandatory columns.
2. **Tweet ID Uniqueness:** Fails if any duplicate `tweet_id` exists.
3. **Text Uniqueness:** Fails if any duplicate `normalized_message` exists.
4. **Taxonomy Conformance:** Checks that every assigned label belongs strictly to the approved 9 operational intents or `unclear_needs_review`. Fails on arbitrary or misspelled labels.
5. **Empty Label Detection:** When evaluating completed datasets (`--require-complete`), fails loudly if even a single row has an empty or whitespace label.
6. **Class Imbalance Monitoring:** Emits warnings if any annotated class falls below 3% of the total dataset ($< 6$ records).

---

## 8. Inter-Annotator Agreement Methodology

To evaluate annotation reliability, `backend/app/evaluation/annotation_agreement.py` provides mathematical comparison between two independent annotators:
- **Raw Agreement Percentage:** $P_o = \frac{\sum \text{Agreed}}{\text{Total}}$
- **Cohen's Kappa Coefficient:**
  $$\kappa = \frac{P_o - P_e}{1 - P_e}$$
  where $P_e$ is the hypothetical probability of chance agreement based on each rater's marginal label frequencies.
- **Landis & Koch Benchmark Scale:** Interprets $\kappa$ into standardized tiers (e.g. $\kappa > 0.80$ Almost Perfect, $0.61 - 0.80$ Substantial, $0.41 - 0.60$ Moderate).
- **Disagreement Matrix & Itemization:** Outputs contingency tables and extracts concrete disagreement examples for adjudicating edge cases.

### Scientific Honesty Principle
If only a single annotator dataset is supplied, the module strictly refuses to simulate or fabricate a second rater. It explicitly outputs:
`INTER_ANNOTATOR_AGREEMENT_NOT_AVAILABLE`
This prevents false claims of inter-rater reliability in academic or engineering reviews.

---

## 9. Golden Set Freezing & Leakage Prevention

A fundamental threat to machine learning validity is test-set contamination (data leakage). Phase 5 enforces three strict operational barriers:

1. **Physical Directory Separation:**
   - Candidate discovery corpus: `data/interim/intent_discovery_corpus.csv`
   - Training/development data: `data/processed/`
   - Golden evaluation data: `data/golden/`
   The golden dataset is stored in a dedicated, isolated directory and will NEVER be fed into classifier training or vector retrieval indices.
2. **Cryptographic Immutability:**
   When freezing completed annotations via `backend/scripts/freeze_golden_set.py`:
   - Enforces 100% completion (zero empty labels).
   - Writes `data/golden/golden_set_v1.csv`.
   - Computes a SHA256 cryptographic digest of the CSV file.
   - Writes `data/golden/golden_set_v1_manifest.json` capturing the exact checksum, row count, and label distribution.
3. **Overwrite Blocking:**
   Once `golden_set_v1.csv` is created, subsequent freeze attempts will fail with `FileExistsError` unless `--force-new-version` is explicitly passed or a new version tag (`v2`) is specified.

---

## 10. What Would Make This Golden Set Misleading?

To maintain rigorous self-scrutiny, we identify the exact vulnerabilities that could compromise the validity of this golden evaluation set:

1. **Single-Annotator Subjectivity:**
   If the entire dataset is labeled by only one annotator, individual biases regarding ambiguous cases (e.g. whether a freeze after update is `software_update_problem` or `general_device_support`) will directly skew model evaluation. Inter-annotator adjudication is essential for maximum rigor.
2. **Taxonomy Leakage via Keyword Pre-filtering:**
   Because stratified sampling utilized heuristic keyword matching to ensure balanced representation across minority intents, there is a risk that sampled tweets have slightly higher keyword density than completely uncurated raw traffic. Evaluators must recognize this trade-off between balanced coverage and purely passive distribution.
3. **Retrieval Index Contamination:**
   If the 200 golden evaluation conversation IDs are inadvertently left in the RAG retrieval vector database, the retrieval agent will retrieve the exact customer query's own resolution, inflating evaluation scores to near 100%. Strict ID exclusion filters must be applied in Phase 7.
4. **Temporal Concentration (Late 2017 Bias):**
   The source Kaggle dataset is heavily concentrated around October–December 2017 (the iOS 11 rollout). If the golden set over-indexes on iOS 11 software update complaints, the measured classifier performance will not reflect modern iOS 17/18 conversational distributions.
5. **Evaluating on Training Paraphrases:**
   If future training sets draw from the same conversation threads or author IDs without deduplication against `data/golden/golden_set_candidates.csv`, the model will memorize semantic paraphrases, yielding falsely optimistic generalization metrics.
6. **Ignoring Borderline & Ambiguous Queries:**
   If human annotators discard or re-label all difficult, colloquial, or frustratingly vague customer inquiries as clean categories, the evaluation set will misrepresent the real-world operational difficulty of frontline customer support triage.

---

## 11. Known Limitations

1. **Twitter Constraint (280 Characters):** Customer opening messages on Twitter are concise, informal, and frequently lack technical specifications, leading to inherent ambiguity that cannot always be resolved without the subsequent brand reply.
2. **Single-Brand Specificity:** The taxonomy and golden set are custom-tailored to `AppleSupport` hardware, software, and services. They do not generalize directly to telecom or airline customer support without re-adaptation.
3. **Pending Human Annotation:** The infrastructure, schema, and stratified candidate pool are fully implemented, verified, and validated; however, human domain experts must manually input the final `annotation_label` values before running supervised evaluation benchmarks.

---

## 12. Current Status & Next Steps

- **Completed:** Taxonomy finalization engine, 9-intent operational schema, 200-sample candidate evaluation set, validation engine, inter-annotator reliability module, and immutable freeze workflow.
- **Current Milestone:** **Phase 5 Infrastructure Complete — Human Annotation Pending**.
- **Exact Next Step:** Complete human annotation of `data/golden/golden_set_annotation_template.csv`, validate the completed set with `--require-complete`, and execute `freeze_golden_set` to publish immutable `golden_set_v1.csv`.
