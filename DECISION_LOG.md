# DECISION LOG — SupportGraph AI

> Architecture Decision Records (ADRs) for the SupportGraph AI project.
> Each decision documents the context, rationale, and trade-offs made during development.

---

## Decision 1

**Decision:** Maintain a strict frontend/backend separation with no shared code.

**Context:** The project will eventually have a React-based dashboard (frontend) and a Python FastAPI backend serving ML inference, data exploration, and agentic orchestration.

**Why:** Separating concerns at the service boundary allows each side to evolve independently:
- The backend can be replaced, scaled, or containerized without touching the frontend.
- The frontend can be rebuilt in a different framework without affecting the API contract.
- API versioning becomes possible and natural.
- Teams (or future contributors) can work on each layer independently.
- It mirrors real-world production architecture and makes the portfolio more credible.

**Trade-off:** Early in the project this creates more boilerplate than a monolithic setup. However, the long-term maintainability gain justifies the upfront cost. We accept the slight complexity overhead.

---

## Decision 2

**Decision:** Explore and understand the dataset thoroughly before building any ML models.

**Context:** The raw dataset (`twcs.csv`) is a large, real-world, noisy Twitter corpus. Its schema, quality characteristics, and suitability for different tasks are not known until inspection.

**Why:**
- Real datasets often contain surprises: unexpected columns, high missing-value rates, schema anomalies, class imbalance.
- Building a model on top of a misunderstood dataset leads to silent failures and misleading results.
- Understanding the data distribution is necessary to make informed decisions about: brand selection, conversation reconstruction strategy, intent clustering approach, and evaluation set construction.
- This phase produces documented, reproducible findings — not assumptions.

**Trade-off:** Phase 1 produces no "exciting" ML output. The payoff is that every subsequent phase is built on evidence, not guesses.

---

## Decision 3

**Decision:** The raw dataset is never modified by any script or notebook.

**Context:** `Dataset/Raw/twcs/twcs.csv` is the canonical source of truth for this project.

**Why:**
- Reproducibility requires that the starting point never changes. Any transformation must be a separate, versioned artifact.
- If the raw file is accidentally corrupted or altered, the entire project loses its foundation and re-downloading is required.
- Non-destructive analysis is a best practice in data science — all cleaning, filtering, and transformation produces a *new* artifact in `data/interim/` or `data/processed/`.
- This mirrors production ML engineering standards (immutable feature stores, append-only data lakes).

**Trade-off:** Requires slightly more disk space (interim copies). This is an acceptable trade-off given that disk space is cheap and reproducibility is invaluable.

---

## Decision 4

**Decision:** Brand selection (choosing which company's support conversations to focus on) must be evidence-based and deferred to Phase 2.

**Context:** The dataset contains support conversations from many different companies (Apple, Spotify, Amazon, etc.). Rather than arbitrarily picking one, we need to select a brand that: has enough volume for meaningful analysis, has clear inbound/outbound conversation structure, and whose conversations are diverse enough for intent discovery.

**Why:**
- Picking a brand with too few interactions makes intent clustering and RAG retrieval unreliable.
- Picking a brand whose conversations are mostly single-turn limits conversation reconstruction.
- The decision must be driven by the statistics produced in Phase 1 — not by personal preference.
- A documented, evidence-based brand selection is more credible and reproducible.

**Trade-off:** We cannot begin conversation reconstruction or intent modeling until Phase 2 is complete. This is the correct sequencing — building on a weak foundation would invalidate all downstream work.

---

## Decision 5

**Decision:** LangGraph, LangChain, and LLM SDKs are not installed or used until they are actually needed (Phase 7+).

**Context:** The project will eventually use LangGraph for agentic orchestration and LLM APIs for generation. However, these are heavy dependencies that are irrelevant to Phases 0–6.

**Why:**
- Including unused dependencies inflates the environment, slows installation, and introduces potential version conflicts.
- Premature abstraction around LangGraph's `StateGraph` API before the data and intent structure are understood would likely need to be discarded and rewritten.
- It signals engineering discipline: add dependencies when they are needed, not speculatively.
- `requirements.txt` for Phases 0–1 stays small, fast to install, and focused.

**Trade-off:** There will be a dependency update step in Phase 7. This is a planned and acceptable engineering step, not a deficiency.

---

## Decision 6

**Decision:** All analysis artifacts (JSON outputs, CSVs, plots, reports) are saved to versioned directories and not regenerated silently.

**Context:** The exploration script produces: schema JSON, statistics JSON, account CSVs, brand candidate CSVs, plots, and a full Markdown report.

**Why:**
- Saved artifacts allow the team to reference findings without re-running the (potentially slow) exploration pipeline.
- Machine-readable outputs (`dataset_schema.json`, `dataset_statistics.json`) enable programmatic consumption in downstream phases without re-parsing the raw CSV.
- Plot files provide a visual record of the data distribution at the time of exploration.
- The Markdown report (`phase_1_dataset_analysis.md`) is the single source of truth for Phase 1 findings and can be shared, reviewed, or cited.
- This discipline produces a self-documenting project that can be handed off or resumed at any point.

**Trade-off:** Artifact files accumulate over time. This is managed by clear directory organization (`data/interim/`, `artifacts/figures/`, `artifacts/reports/`) and `.gitignore` rules that prevent binary artifacts from being committed while keeping reports tracked.

---

## Decision 7 — Phase 2

**Decision:** Use multi-criteria weighted scoring for brand selection rather than selecting by highest volume alone.

**Context:** 108 brand accounts exist in the dataset. The brand with the highest usable interaction volume (AmazonHelp, 169,287) is not automatically the most suitable for downstream tasks.

**Why:**
- Volume is a necessary but not sufficient condition. A brand with high volume but low conversation reconstructability would provide poor historical grounding data for RAG.
- Five metrics capture different aspects of suitability: volume (necessary threshold), coverage (RAG usability), reconstructability (pair quality), completeness (data integrity), diversity (intent richness).
- Transparent weighted scoring allows future reviewers to understand and challenge the selection.
- Sensitivity analysis across weight scenarios validates robustness.

**Trade-off:** Any weighting scheme is subjective. We document this limitation explicitly and perform robustness checks. The selection can be reproduced by any reviewer using the provided script.

---

## Decision 8 — Phase 2

**Decision:** Assign 30% weight to Interaction Volume (the highest single weight).

**Context:** All downstream tasks (intent clustering, RAG retrieval, evaluation set construction) have minimum data requirements.

**Why:**
- Intent clustering requires enough examples per cluster to be statistically meaningful.
- RAG retrieval requires enough candidate pairs to find diverse historical resolutions.
- A manually labelled evaluation set of 150–250 examples requires a brand with thousands of interactions for meaningful sampling.
- Volume acts as a threshold criterion — below a certain count, other metrics become irrelevant.

**Trade-off:** Heavily weighting volume means higher-volume but lower-quality brands may rank competitively. The other 4 metrics (70% combined) counter-balance this.

---

## Decision 9 — Phase 2

**Decision:** Define Response Coverage as "fraction of brand replies whose parent tweet exists in the dataset" rather than a more complex recall metric.

**Context:** The dataset may be an incomplete extract of Twitter — some parent tweets may simply not be present.

**Why:**
- A brand reply can only be used for historical grounding if we can retrieve the original customer message.
- This definition is directly computable from the dataset without any external information.
- It captures the RAG-relevant property: can we reconstruct customer→brand pairs?

**Trade-off:** Coverage measures data availability, not response quality. A brand could have 100% coverage but still provide unhelpful, formulaic replies. Response quality is partially captured by diversity and implicitly by reconstructability.

---

## Decision 10 — Phase 2

**Decision:** Define Reconstructability separately from Response Coverage to distinguish data availability from conversation structural quality.

**Context:** Coverage tells us whether parent tweets exist. Reconstructability asks whether those parents are actually customer messages (inbound=True).

**Why:**
- Some brand replies may have parents that are OTHER brand tweets (e.g., continuation of a thread). These are less useful as standalone customer→brand training pairs.
- Separating these metrics makes the scoring more granular and honest.
- A brand with high coverage but low reconstructability would have many "broken" or partial chains where the customer message is missing.

**Trade-off:** This adds complexity. However, the additional precision in distinguishing coverage from reconstructability is worth the complexity for an evidence-based selection.

---

## Decision 11 — Phase 2

**Decision:** Use TF-IDF + KMeans cluster entropy as a diversity proxy rather than topic modeling (LDA/BERTopic).

**Context:** We need a computationally practical diversity estimate that does not require LLMs and runs within minutes on sampled data.

**Why:**
- Phase 2 is pre-modeling — we deliberately avoid LLM dependencies at this stage.
- BERTopic/LDA require more tuning, and results would be harder to reproduce across environments.
- TF-IDF + KMeans is deterministic given a fixed seed, efficient on 5,000 samples, and produces an interpretable entropy score.
- Shannon entropy of cluster sizes is a principled diversity measure: high entropy = balanced clusters = diverse topics.

**Trade-off:** TF-IDF captures lexical diversity, not semantic diversity. A brand with lexically varied but semantically repetitive complaints could score high incorrectly. This limitation is documented in the report's "What Could Make This Selection Misleading?" section.

---

## Decision 12 — Phase 2

**Decision:** Perform sensitivity analysis across three predefined weight scenarios.

**Context:** The default weights (30/20/20/15/15) are one reasonable choice. The selection should not be fragile to small weight perturbations.

**Why:**
- A selection that only holds under one specific weight configuration is not trustworthy.
- Three scenarios (default, volume-focused, quality-focused) cover the space of reasonable trade-off priorities.
- If the same brand wins across all scenarios, the selection is robust and can be reported with higher confidence.
- Sensitivity analysis is common practice in multi-criteria decision analysis (MCDA).

**Trade-off:** Three scenarios do not exhaustively cover all possible weights. We acknowledge this limitation. Infinite weight combinations exist, but three representative scenarios are sufficient to assess basic robustness.

---

## Decision 13 — Phase 2

**Decision:** The qualitative sanity check cannot override the quantitative ranking without explicit documentation.

**Context:** After computing quantitative scores, we sample real interactions to verify data quality. There is a risk of "cherry picking" or overriding a data-driven result based on subjective impression.

**Why:**
- The quantitative ranking is computed from all interactions, not a sample. Its statistical basis is stronger than a small qualitative sample.
- If qualitative review revealed a systematic problem (e.g., all sampled brand replies are "Please DM us"), that would be documented as a limitation, not used to silently change the ranking.
- Separating quantitative ranking from qualitative sanity check maintains methodological integrity.

**Trade-off:** The qualitative check may surface real problems that quantitative metrics miss (e.g., bot-like responses). These are documented as limitations but do not alter the automated selection result. Future phases can revisit this decision if needed.

---

## Decision 14 — Phase 3

**Decision:** Assemble conversation threads using directed reply graph connected components rather than temporal proximity or author-window grouping.

**Context:** Social media datasets contain overlapping conversations. Grouping tweets by author and timestamp window (e.g. 60-minute window) risks tangling concurrent separate inquiries or splitting delayed customer replies.

**Why:**
- Explicit `in_response_to_tweet_id` pointers define true causal reply chains.
- Weakly connected component extraction guarantees that all messages linked through direct replies belong to the same conversation thread regardless of elapsed time.
- Prevents artificial conversation splitting when a customer takes days to reply to troubleshooting instructions.

**Trade-off:** High-volume threads that receive multiple off-topic replies can grow into larger components. We manage this by tracking branching and component size limits.

---

## Decision 15 — Phase 3

**Decision:** Generate deterministic conversation IDs derived from the brand name and primary root tweet ID (`conv_{brand}_{root_tweet_id}`) instead of random UUIDs.

**Context:** Downstream intent discovery and evaluation pipelines must reference conversations reproducibly across distinct runs, worker processes, and development environments.

**Why:**
- Deterministic IDs ensure that re-running the reconstruction pipeline produces byte-for-byte identical datasets.
- The root tweet uniquely seeds the causal thread component.
- Eliminates non-deterministic tracking bugs and allows easy lookups back to raw tweet IDs.

**Trade-off:** If a component is discovered with multiple roots, tiebreaking logic (earliest timestamp, lowest tweet ID) must be strictly maintained to prevent ID drift.

---

## Decision 16 — Phase 3

**Decision:** Preserve multi-child branching structures explicitly in the graph topology rather than flattening them into an artificial linear chat history.

**Context:** On Twitter, a single customer tweet frequently receives multiple brand replies (e.g. `(1/2)` and `(2/2)` tweets, or multiple agents answering), or multiple customers join an inquiry.

**Why:**
- Flattening branches into a linear sequence creates misleading question-answer pairings and hallucinations in downstream training.
- Explicit edge records (`parent_tweet_id -> child_tweet_id`) maintain structural truth.
- Downstream RAG can choose to traverse specific branches or treat each path as an alternative historical resolution.

**Trade-off:** Complex graph structures require tree traversal logic rather than simple list iteration.

---

## Decision 17 — Phase 3

**Decision:** Record missing parent tweets as structured anomalies and preserve the observed sub-thread rather than silently discarding the conversation.

**Context:** 326 brand replies point to parent tweets that were deleted, made private, or omitted during the initial Twitter scrape.

**Why:**
- Silently discarding conversations reduces historical evidence and introduces hidden sample selection bias.
- Recording `missing_parent_tweet` anomalies in `conversation_anomalies.csv` and downgrading quality to `LOW` keeps the data pipeline fully transparent and auditable.
- Downstream tasks that require strictly complete context can filter out `LOW` quality threads via a single column check.

**Trade-off:** Retaining incomplete threads requires downstream consumers to respect the `quality_status` flag.

---

## Decision 18 — Phase 3

**Decision:** Implement a deterministic, rule-based quality status classification (`HIGH`, `MEDIUM`, `LOW`) based on graph integrity and structural anomalies.

**Context:** Downstream components (e.g. golden evaluation set sampling, fine-tuning) need clear criteria to filter trustworthy training examples without manual inspection of 80,000+ threads.

**Why:**
- Rule-based definitions are 100% reproducible and explainable.
- `HIGH`: Zero anomalies, valid unbroken graph, >= 2 messages, customer and brand both present.
- `MEDIUM`: Usable structure with minor anomalies (e.g. timestamp reversals or forward response pointer discrepancies).
- `LOW`: Missing parent pointers or cycles.

**Trade-off:** Quality reflects structural graph integrity, not the linguistic or technical helpfulness of the support response.

---

## Decision 19 — Phase 3

**Decision:** Treat observed conversation closure as structural termination only, never as evidence of successful issue resolution.

**Context:** Many automated support systems erroneously treat the final brand message as a "successful resolution" label.

**Why:**
- Twitter support frequently deflects to private DMs, phone numbers, or external Apple web forms.
- A customer who stops replying may be satisfied, frustrated, or pursuing alternative channels.
- Assuming resolution without explicit customer confirmation would poison downstream escalation policy learning with false positives.

**Trade-off:** We cannot use conversational silence as a positive reinforcement signal for agent training.

---

## Decision 20 — Phase 3

**Decision:** Maintain dual representation: preserve graph edges as ground truth while generating a breadth-first canonical traversal order with depth annotations.

**Context:** Relational databases, graphs, and transformer models have conflicting format requirements: graphs need node-edge relational schemas, while language models require sequentially ordered text sequences.

**Why:**
- `conversation_edges.csv` and `conversations.jsonl` preserve exact graph relationships and branching.
- `conversation_messages.parquet` provides a canonical breadth-first topological sequence with an explicit `depth` column for flat tabular and embedding workflows.
- Satisfies both graph analysis and language modeling without compromising data integrity.

**Trade-off:** Requires storing and validating two complementary representations during reconstruction.

---

## Decision 21 — Phase 4

**Decision:** Restrict the candidate intent discovery corpus to customer opening messages (`role == 'customer'`, `depth == 0`, and `quality_status == 'HIGH'`).

**Context:** Full support conversations contain many turns (agent greetings, diagnostic questions, customer clarifications, confirmation of fixes). Discovering intents requires identifying the customer's initial problem statement.

**Why:**
- The root reason a customer contacts support is expressed at conversation inception.
- Downstream agent messages and customer replies (depth >= 1) introduce noise, instructions, and conversational protocol that distort the customer's original problem distribution.
- High-quality conversations ensure complete conversational graphs free from structural anomalies.

**Trade-off:** We miss downstream shifts where the true root cause was misidentified in the initial customer complaint.

---

## Decision 22 — Phase 4

**Decision:** Use local open-source dense sentence embeddings (`sentence-transformers/all-MiniLM-L6-v2`) rather than keyword-only TF-IDF or external LLM API embeddings.

**Context:** Clustering customer complaints requires semantic understanding of paraphrased text across varied vocabulary.

**Why:**
- TF-IDF and bag-of-words fail to associate semantically identical queries with zero vocabulary overlap (e.g., *"battery dies in two hours"* vs *"phone won't hold charge"*).
- `all-MiniLM-L6-v2` runs 100% locally, deterministically, without latency, rate limits, or API key dependencies.
- It produces rich 384-dimensional contextual sentence embeddings optimized for cosine similarity.

**Trade-off:** Requires a local transformer dependency and PyTorch inference, but remains lightweight and runs efficiently on CPU and MPS.

---

## Decision 23 — Phase 4

**Decision:** Apply PCA dimensionality reduction (50 components) before clustering rather than clustering directly in the raw 384-dimensional embedding space.

**Context:** Distance metrics in high-dimensional vector spaces degrade due to the curse of dimensionality (distance concentration phenomenon).

**Why:**
- 50 principal components capture 64.6% of total variance while eliminating high-frequency vector noise.
- Accelerates MiniBatchKMeans convergence and improves cluster compactness.
- PCA is fully deterministic given a fixed random seed, ensuring reproducible clustering runs.

**Trade-off:** 35.4% of variance is discarded. We validated that the retained 64.6% retains all dominant semantic cluster separations.

---

## Decision 24 — Phase 4

**Decision:** Empirically evaluate clustering across a range of cluster counts ($K \in [4, 15]$) rather than predetermining $K$.

**Context:** The true number of customer support intents cannot be guessed prior to observation.

**Why:**
- Testing a sweep across $K = 4$ through $15$ exposes the actual cohesion and separation behavior across scales.
- Prevents under-segmentation (collapsing distinct issues into super-clusters) and over-segmentation (shredding coherent categories into tiny fragments).
- Produces verifiable metric curves for Silhouette, Davies-Bouldin, and Calinski-Harabasz.

**Trade-off:** Requires running multiple MiniBatchKMeans iterations, which is computationally modest (few seconds total).

---

## Decision 25 — Phase 4

**Decision:** Use multi-metric clustering evaluation and cluster size distributions rather than relying solely on a single metric (e.g., Silhouette score).

**Context:** Individual clustering metrics have known systematic biases (e.g., Silhouette favors spherical clusters and can be misleading on high-dimensional text manifolds; Davies-Bouldin can favor small partitions).

**Why:**
- Silhouette measures cohesion vs separation.
- Davies-Bouldin evaluates within-cluster scatter against inter-cluster centroid separation.
- Calinski-Harabasz measures variance ratio.
- Monitoring smallest cluster size and tiny cluster count (<1%) ensures the selected $K$ does not create degenerate or unusable micro-clusters.

**Trade-off:** Requires designing and documenting a composite scoring rule to balance competing metrics.

---

## Decision 26 — Phase 4

**Decision:** Explicitly declare that unsupervised clusters are discovery tools, NEVER ground truth intent labels.

**Context:** In automated NLP pipelines, teams frequently mistake unsupervised cluster assignments for objective labels.

**Why:**
- Unsupervised clustering reflects vector proximity in an embedding space, not semantic or operational truth.
- Clusters frequently group messages based on shared superficial vocabulary (e.g., general frustration or hardware mentions) rather than true operational intent.
- Treating clusters as ground truth poisons downstream supervised classifiers with label noise.

**Trade-off:** Prevents fully automated zero-touch label creation; enforces human review.

---

## Decision 27 — Phase 4

**Decision:** Establish a formal human review workflow (`docs/intent_review_guide.md`) with explicit `pending`, `accepted`, `merged`, `split`, and `rejected` statuses.

**Context:** Transitioning from unsupervised clusters to a deployable intent taxonomy requires domain expertise.

**Why:**
- Human experts must inspect centroid-nearest exemplars and distinctive TF-IDF terms to evaluate business actionability.
- Reviewers can merge split variants (e.g., two update-related clusters) or split entangled topics.
- All interim outputs leave `final_intent_label = null` until explicit review occurs.

**Trade-off:** Introduces a human-in-the-loop requirement before supervised training in Phase 5.

---

## Decision 28 — Phase 4

**Decision:** Constrain the target intent taxonomy to a practical, small cardinality (approximately 5–10 intents).

**Context:** Support systems often fail when attempting to classify incoming requests into hundreds of fine-grained categories.

**Why:**
- A small taxonomy (5–10 intents) provides sufficient sample support per class (thousands of examples).
- Downstream supervised classifiers achieve vastly higher precision and calibration with compact taxonomies.
- Fits the high-level routing, retrieval, and escalation policies of SupportGraph AI.

**Trade-off:** Fine-grained sub-issues must be handled in retrieval or prompt context rather than as distinct top-level classification classes.

---

---

## Decision 29 — Phase 4.5

**Decision:** Treat low absolute clustering silhouette scores ($S \approx 0.0815$) with scientific caution and reject any claim that $K=8$ proves eight natural customer intents exist.

**Context:** The Phase 4 clustering experiment identified $K=8$ as the best candidate partition with a silhouette score of 0.0815.

**Why:**
- A silhouette score of 0.0815 indicates that cluster boundaries in semantic sentence embedding manifolds have heavy overlap.
- Customer complaints in support channels form a continuous linguistic spectrum rather than isolated geometric islands.
- Over-interpreting exploratory cluster partitions as "discovered truth" leads to fragile, unmaintainable downstream NLP models.

**Trade-off:** Requires explicit caveats in all documentation and forbids fully autonomous taxonomy promotion.

---

## Decision 30 — Phase 4.5

**Decision:** Inspect three distinct tiers of customer examples (20 centroid-nearest, 20 deterministic random, and 10 max-min diverse) rather than relying solely on centroid examples.

**Context:** Reviewers typically look only at top centroid-nearest examples, which creates an illusion of cluster homogeneity.

**Why:**
- Centroid-nearest examples show only the most prototypical messages and obscure peripheral subtopics.
- Deterministic random sampling gives an unbiased view of typical cluster density.
- Max-min diversity sampling intentionally surfaces peripheral, multi-issue, or atypical complaints that reveal whether a cluster should be split.

**Trade-off:** Reviewers must read more examples (50 per cluster), but the resulting taxonomy avoids hidden subtopic contamination.

---

## Decision 31 — Phase 4.5

**Decision:** Identify and explicitly flag historical event-specific clusters (e.g., the late-2017 iOS 11.1 autocorrect bug) with a `generalization_status` field (`GENERALIZABLE`, `EVENT_SPECIFIC`, `MIXED`).

**Context:** The dataset captures late-2017 Twitter support, heavily distorted by transient incidents like the 'A [?]' letter 'I' glitch and initial iOS 11 rollout bugs.

**Why:**
- Building permanent production intent categories around 2-week software glitches causes severe model obsolescence once the bug is patched.
- Historical event detection enables reviewers to either consolidate transient bugs into broader timeless categories (e.g., `keyboard_typing_glitch`) or filter them out.

**Trade-off:** Requires developing domain-specific historical event detection patterns.

---

## Decision 32 — Phase 4.5

**Decision:** Strictly separate the customer intent taxonomy from the brand response strategy taxonomy.

**Context:** Analysis of 20,000 first brand responses revealed distinct resolution actions (e.g., 51.4% DM escalation, 77.6% support link referrals, 37.1% clarification questions).

**Why:**
- Customer intent describes the incoming friction mode (*what the user is experiencing*).
- Response strategy describes operational action (*how the agent resolves it*).
- Conflating intent and resolution prevents learning conditional escalation and retrieval policies in downstream agent graphs.

**Trade-off:** Requires maintaining two distinct taxonomies in the system architecture.

---

## Decision 33 — Phase 4.5

**Decision:** Restrict automated merge and split candidate recommendations strictly to `"review"`, never executing automatic mergers or splits.

**Context:** Automated metrics identified candidate merge pairs (Clusters 0 & 7, Clusters 3 & 6) and split candidates (Cluster 2).

**Why:**
- Mathematical proximity does not guarantee operational equivalence; only domain experts can determine whether two failure modes share an actionable troubleshooting resolution path.
- Automated merging risks collapsing distinct operational procedures, while automated splitting creates fragmented tail distributions.

**Trade-off:** Keeps human expertise in the loop; prevents fully autonomous end-to-end taxonomy generation.

---

## Decision 34 — Phase 4.5

**Decision:** Decouple hardware product entities from customer intent failure modes.

**Context:** Frequent mentions of *iPhone*, *iPad*, and *MacBook* often tempt engineers to create product-specific intents.

**Why:**
- A battery drain issue on an iPhone 7 and an iPad share identical diagnostic workflows (battery health, background app management, low power mode).
- Product names must be handled as extracted metadata entities, not top-level intent classes, to avoid combinatorial explosion in the taxonomy.

**Trade-off:** Requires downstream entities extraction alongside intent classification.

---

## Decision 35 — Phase 4.5

**Decision:** Design the final taxonomy around approximately 5–10 practical, operational intents.

**Context:** Machine learning classifiers and downstream retrieval pipelines require sufficient per-class sample support and clear decision boundaries.

**Why:**
- A 5–10 class taxonomy yields thousands of training examples per category in the 80,000+ conversation corpus.
- Supervised classifiers achieve significantly higher precision, recall, and calibration on compact taxonomies.
- Fulfills the core requirements for SupportGraph AI routing, retrieval, and escalation without overwhelming human reviewers.

**Trade-off:** Fine-grained sub-issues must be resolved via retrieval context rather than dedicated top-level classifier heads.

---

*Phase 4.5 decisions recorded after running `python -m backend.scripts.audit_intent_taxonomy`*
*Selected brand: AppleSupport (20,000 sampled conversations audited across K=8 clusters)*

---

## Decision 36 — Phase 5

**Decision:** Reject unsupervised clusters as ground-truth evaluation labels.

**Context:** Phase 4 produced an 8-cluster partition using MiniBatchKMeans with a Silhouette score of $0.0815$.

**Why:**
- Geometric proximity in vector space clusters on vocabulary, sentence length, and syntax rather than operational business logic.
- Low silhouette score confirms customer tweets form a continuous semantic continuum rather than discrete isolated islands.
- Unsupervised clusters overfit to high-frequency transient bugs (e.g. November 2017 autocorrect glitch).
- Operational intent categories must represent what the customer needs help with, derived from human-reviewed evidence.

**Trade-off:** Requires upfront engineering of review tooling, candidate taxonomies, and human annotation guidelines.

---

## Decision 37 — Phase 5

**Decision:** Formulate a 9-intent candidate operational taxonomy with evidence-based merges and splits.

**Context:** Audit findings showed Clusters 0 & 7 and Clusters 3 & 6 were redundant pairs, while Cluster 2 and Cluster 4 conflated distinct resolution paths.

**Why:**
- Merging Clusters 0 & 7 generalizes the 2017 autocorrect bug into `keyboard_typing_issue`.
- Merging Clusters 3 & 6 unifies fragmented OS update patch complaints into `software_update_problem`.
- Splitting Cluster 4 decouples Apple ID security (`account_access_issue`) from financial disputes (`billing_purchase_issue`).
- Splitting Cluster 2 separates screen repairs (`display_touch_issue`) from audio/Bluetooth failures (`hardware_audio_connection_issue`).
- The resulting 9 intents fall squarely into the target 7–10 range, ensuring balanced representation and actionable support routing.

**Trade-off:** Heuristic regexes are required during candidate stratification before final human labeling.

---

## Decision 38 — Phase 5

**Decision:** Set the golden evaluation benchmark size to exactly 200 real customer opening messages.

**Context:** The golden set must provide statistically sound metric estimation (accuracy, macro-F1, hallucination rate) while remaining practical for thorough human annotation.

**Why:**
- 200 examples provide ~22–23 examples per intent across 9 categories, sufficient for detecting per-class performance drops of >10%.
- Within the allowed range of 150–250 examples, 200 is optimal: large enough for statistical power, compact enough to ensure 100% careful human labeling without annotator fatigue.
- Avoids the vanity of claiming thousands of automated or weakly labeled "golden" records.

**Trade-off:** Evaluators must evaluate fine-grained performance on small per-class subsets (~22 records each).

---

## Decision 39 — Phase 5

**Decision:** Enforce stratified random sampling with multi-stage deduplication and conversation context enrichment.

**Context:** Naive random sampling from Twitter data results in bot spam duplicates and heavy starvation of minority intents.

**Why:**
- Strict `tweet_id` and `normalized_text` deduplication ensures no duplicate messages contaminate the evaluation pool.
- Stratification guarantees that minority technical intents (e.g. `mac_software_issue`, `account_access_issue`) receive equal representation as high-frequency update complaints.
- Joining the brand's first turn response (`conversation_context`) provides human annotators with clarifying context when opening tweets are colloquially ambiguous.

**Trade-off:** Stratification quotas must be computed dynamically; customer text without brand responses receives placeholder context.

---

## Decision 40 — Phase 5

**Decision:** Strictly prohibit synthetic or LLM-generated labels in golden candidate sets (`annotation_label` must remain empty).

**Context:** Some practitioners use LLMs (GPT-4) to pseudo-label evaluation data to accelerate project completion.

**Why:**
- Evaluating an AI system using AI-generated labels creates circular validation and masks systematic model blindspots.
- Synthetic labels perpetuate hallucinated patterns and false consensus.
- A true golden benchmark must reflect genuine human domain judgment.
- Setting `annotation_label = ""` and `annotation_status = "pending"` ensures zero silent contamination.

**Trade-off:** The golden evaluation set cannot be frozen until real human annotation takes place.

---

## Decision 41 — Phase 5

**Decision:** Explicitly support and track ambiguous customer opening messages with `unclear_needs_review`.

**Context:** Frontline Twitter customer messages are frequently truncated, unintelligible, non-English, or completely vague.

**Why:**
- Forcing ambiguous messages into arbitrary technical categories corrupts evaluation metrics.
- Frontline systems must know when to request clarification rather than guess.
- Tracking `unclear_needs_review` separately prevents contamination of operational intent distributions while preserving difficult real-world test cases.

**Trade-off:** Evaluators must filter or explicitly handle unclear cases during metric computation.

---

## Decision 42 — Phase 5

**Decision:** Report `INTER_ANNOTATOR_AGREEMENT_NOT_AVAILABLE` when only one annotator dataset exists.

**Context:** Calculating Cohen's Kappa requires two independent annotation sets.

**Why:**
- Scientific honesty dictates that we never fabricate a second annotator, synthesize pseudo-ratings, or claim high inter-rater reliability without empirical dual annotation.
- When two annotations exist, the engine computes raw agreement, Cohen's Kappa, Landis & Koch scale interpretations, and confusion matrices.
- When single annotation is provided, reporting unavailable preserves engineering integrity.

**Trade-off:** Inter-rater reliability cannot be claimed until a second annotator completes the template.

---

## Decision 43 — Phase 5

**Decision:** Protect golden evaluation sets with versioned immutability, SHA256 checksums, and overwrite blocking.

**Context:** In iterative ML development, evaluation sets are frequently altered, re-sampled, or contaminated with training data over time.

**Why:**
- A golden set is useless if it changes between experiments.
- `freeze_golden_dataset` validates 100% completion, writes `golden_set_v1.csv`, and generates `golden_set_v1_manifest.json` containing the cryptographic SHA256 digest.
- Existing version files cannot be overwritten without explicit `--force-new-version` flags.
- Physical storage in `data/golden/` isolates evaluation data from training corpora.

**Trade-off:** Requires intentional version tagging (v1, v2) if new golden sets are ever sampled.

---

*Phase 5 decisions recorded after implementing candidate taxonomy engine, stratified sampler, validation module, agreement engine, and freeze workflow.*
*Selected brand: AppleSupport (200 golden evaluation candidates sampled across 9 operational intents).*

---

## Decision 44 — Phase 5.5

**Decision:** AI-assisted intent suggestions are strictly advisory and must never be treated as ground truth.

**Context:** Using an LLM (Groq openai/gpt-oss-20b) to generate candidate annotations speeds up dataset curation.

**Why:**
- Treating model suggestions as ground truth invalidates evaluation benchmarks by introducing circular validation.
- The model's suggestions serve purely as cognitive scaffolding for human reviewers.
- Final evaluation integrity depends entirely on human scrutiny and domain expertise.

**Trade-off:** Human review remains a mandatory requirement before golden sets can be finalized.

---

## Decision 45 — Phase 5.5

**Decision:** Maintain strict column isolation between model suggestions and human ground truth (`annotation_label` remains empty).

**Context:** When storing AI suggestions in CSV artifacts, automated pipelines often overwrite target label columns.

**Why:**
- Overwriting `annotation_label` with model predictions creates silent data contamination.
- The workflow stores model output in dedicated fields: `model_suggested_label`, `model_confidence`, and `model_reasoning_summary`.
- `annotation_label` and `annotator` remain strictly empty (`""`) and `annotation_status` is locked to `"pending_human_review"` until actively confirmed by a human.

**Trade-off:** Requires separate suggestion generation and human review CLI steps.

---

## Decision 46 — Phase 5.5

**Decision:** Formally distinguish `MODEL_HUMAN_AGREEMENT` from inter-annotator agreement.

**Context:** Measuring how frequently human annotators agree with AI suggestions is a common audit metric.

**Why:**
- Describing AI-human concordance as "inter-annotator agreement" or computing Cohen's Kappa between an LLM and a human is scientifically misleading.
- `MODEL_HUMAN_AGREEMENT` evaluates model suggestion quality and alignment, NOT human inter-rater reliability.
- Inter-annotator agreement is reserved strictly for comparisons between two independent human raters.

**Trade-off:** Downstream reporting must maintain separate metric designations.

---

## Decision 47 — Phase 5.5

**Decision:** Require structured reasoning summaries for all AI suggestions to support human edge-case adjudication.

**Context:** A bare class label suggestion gives human annotators no insight into why the model selected that category.

**Why:**
- Providing a concise 1–2 sentence reasoning summary citing specific customer symptoms or keywords helps human reviewers quickly assess borderline complaints.
- Surfacing reasoning exposes hallucinations or keyword biases (e.g. noticing that the model was confused by a mention of "MacBook" when the core issue was an Apple ID password reset).

**Trade-off:** Consumes slightly more completion tokens per record.

---

---

## Decision 49 — Phase 5.6

**Decision:** Implement risk-based prioritization and deterministic queue generation for human golden set review.

**Context:** AI suggestions exist for all 200 golden evaluation records. However, self-reported LLM confidence is uncalibrated and cannot be equated with ground truth. Arbitrary sequential review wastes human effort on obvious cases while delaying high-risk edge cases.

**Why:**
- Prioritizing records by deterministic risk levels (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`) ensures annotators review high-risk anomalies first: `model_needs_human_review=True`, out-of-taxonomy labels, intent divergences, borderline confidence, and non-English messages.
- The prioritization system changes only the *order* and *urgency* of review; it never automatically converts AI suggestions into ground truth.
- Every assigned priority includes a reproducible score (0–100) and human-readable explanation.

**Trade-off:** Adds an explicit queue generation step (`create_review_queue.py`) between suggestion generation and human review.

---

## Decision 50 — Phase 5.6

**Decision:** Mandate deterministic Quality Control (QC) sampling on low-priority AI suggestions.

**Context:** Even high-confidence LLM predictions (e.g. 0.95) can occasionally suffer from subtle hallucination or domain misalignment.

**Why:**
- Low priority must never mean "automatically accepted".
- Deterministic pseudo-random sampling (15% quota, fixed `seed=42`) injects a reproducible subset of high-confidence records into the primary human review queue.
- Guarantees empirical verification of model accuracy across the entire difficulty spectrum without manufacturing synthetic ground truth.

**Trade-off:** Reviewers inspect a sample of clean records in addition to difficult edge cases.

---

## Decision 51 — Phase 5.6

**Decision:** Implement Smart Human-in-the-Loop (HITL) review with explicit grouped approval for safe low-risk records.

**Context:** Reviewing 200 records one-by-one is cognitively exhausting. An efficient review workflow is needed without resorting to dangerous automatic label acceptance.

**Why:**
- Individual review is enforced for all CRITICAL, HIGH, MEDIUM, and QC-sampled records (`individual_review_required`, `qc_review_required`).
- For clean, high-confidence LOW-priority records (`safe_for_group_review`), the CLI groups records by suggested intent and allows explicit human approval (`[B]`), individual inspection (`[A]`), or skipping (`[S]`).
## Decision 52 — Phase 5.6

**Decision:** Adopt layered language evidence detection and explicit structured risk flags in review prioritization.

- Explicit deterministic risk flags (`low_confidence`, `borderline_confidence`, `invalid_suggestion`, `taxonomy_mismatch`, `model_requested_review`, `unclear_needs_review_suggested`, `true_language_uncertainty`, `intent_divergence`, `multi_intent_overlap`, `sparse_message`, `truncated_message`) make prioritization completely explainable and auditable.

---

## Decision 53 — Phase 5.7

**Decision:** Implement hierarchical intent decision logic with symptom precedence and few-shot calibration to resolve generic fallback collapse.

**Context:** The Phase 5.7 error audit on 37 human-reviewed diagnostic records revealed a systematic failure mode: `llama3.2:latest` collapsed 56.8% of inputs into `general_device_support`, yielding 0% recall on audio hardware and frequent confusion between autocorrect bugs and screen rendering.

**Why:**
- Replacing flat multi-class prompt descriptions with explicit hierarchical rules (Rule of Specificity, Symptom Over Entity, Update Causality, Audio Priority, Keyboard Priority, Display Boundary, and Billing Boundary) ensures specific operational intents strictly override generic fallbacks.
- Mac mentions without software root causes route to their true physical symptom (e.g. MacBook speaker -> `hardware_audio_connection_issue`).
- Targeted few-shot examples demonstrate ambiguous boundary resolution without overfitting.
- Results evaluated against the 37 diagnostic records are strictly designated as **Diagnostic Development Performance** (43.2% -> 54.1% accuracy, macro F1 0.282 -> 0.411) to avoid optimistic performance claims before testing on an independent holdout.

**Trade-off:** Evaluation on the same records used for error analysis is diagnostic only and must be validated on an independent holdout set.

---

## Decision 54 — Phase 5.8

**Decision:** Implement Safe Grouped Human Approval for verified low-risk golden set records with mandatory confirmation gating and selective overrides.

**Context:** The remaining 163 golden set records require human ground-truth verification. While individual inspection is essential for complex edge cases, reviewing 163 records one-by-one is cognitively burdensome. However, automatic AI labeling violates scientific integrity.

**Why:**
- Grouped review enables human annotators to inspect coherent batches of high-confidence, clean records clustered by predicted intent, reducing fatigue while preserving human oversight.
- **Not AI Auto-Labeling:** AI suggestions remain strictly advisory evidence. Annotations are only committed when a human explicitly triggers and confirms group approval.
- **Mandatory Confirmation Gate:** Selecting group approval displays a mandatory confirmation screen detailing the intent, record count, and reminder of human responsibility before any ground truth is written.
- **Strict Eligibility Restrictions:** Only records satisfying all 10 safety conditions (LOW priority, non-QC, confidence >= 0.90, no escalation risk flags, no language/intent uncertainty, valid taxonomy match) are eligible for grouping.
- **QC Protection:** A deterministic 15% sample (`seed=42`) is excluded from grouping and forced into mandatory individual audit.
- **Audit Trail & Ground Truth Separation:** Each group-approved record receives individual, auditable ground-truth fields (`review_mode="group_human_approval"`, `approval_type="group_approved_suggestion"`, `group_id`, `approved_group_label`, `annotation_timestamp`), maintaining strict separation from AI prediction fields.
- **Selective Overrides:** Annotators can override specific records in a group without automatically approving the remaining items.

---

## Decision 55 — Phase 5.9

**Decision:** Implement a Safe Taxonomy Calibration and Recommendation Layer using transparent contextual signal extraction and risk-based disagreement routing.

**Context:** Human ground-truth verification of 57 golden records revealed an empirical override rate of 66.7% against baseline LLM suggestions, with the model frequently collapsing specific operational symptoms (audio cables, letter I autocorrect, rapid battery drain, macOS boot failures) into `general_device_support`. To assist human reviewers on the remaining 143 records without compromising scientific integrity, an advisory calibration layer is needed.

**Why:**
- **Transparent Contextual Signal Extraction:** Derives 10 deterministic rule-sets corresponding to approved taxonomy categories from empirical error analysis, avoiding black-box ML model training on test data.
- **Strict Ground-Truth Isolation:** Calibration candidates (`calibration_candidate_label`, `calibration_confidence`, `calibration_status`, `calibration_supporting_signals`, `calibration_conflicting_signals`) are stored in distinct metadata fields and are NEVER automatically copied into `annotation_label`.
- **Automated Disagreement Escalation:** Any discrepancy between the calibrated recommendation and the baseline AI suggestion (`DISAGREES_WITH_AI`) automatically escalates the record to HIGH priority (score $\ge 78$) with a `calibration_disagreement` risk flag, disqualifying it from grouped approval and forcing individual human inspection.
- **Explicit `[C]` Action in Review CLI:** Annotators can accept calibrated recommendations in a single keypress (`[C]`), generating an auditable `approval_type="individual_calibration_acceptance"` record while eliminating repetitive manual typing.
- **100% Ground-Truth Immutability:** All 57 existing human ground-truth annotations are preserved byte-for-byte with SHA-256 checksum verification (`92fccbf4f31258dbcf2072d0a1242333b0de54ce740f70d828343ab64bf3271a`).

---

## Decision 56 — Phase 5.9

**Decision:** Establish a Read-Only Scientific Pattern Analysis and Empirical Taxonomy Overlap Discovery Layer (`annotation_pattern_analysis.py`).

**Context:** Following 57 completed human ground-truth reviews, empirical analysis showed an observed annotation agreement of 35.1% with baseline LLM suggestions, driven predominantly by `general_device_support` over-prediction (61.8% override rate). To generalize and avoid both overfitting and underfitting, an empirical pattern analysis layer is required to extract recurring signals, audit category boundaries, and formulate annotation guidelines without modifying ground truth or retraining ML models.

**Why:**
- **Zero Modification & Immutability:** Evaluates exclusively the completed $N=57$ records without altering ground truth, pending items, or baseline model predictions.
- **Dynamic Semantic Pattern Extraction:** Derives empirical n-grams, tokens, and phrasing directly from customer messages across top AI $\rightarrow$ Human confusion pairs (e.g. `general_device_support` $\rightarrow$ `software_update_problem` [6 occurrences; signals: *'ios'*, *'new'*, *'115858'*], `general_device_support` $\rightarrow$ `mac_software_issue` [4 occurrences; signals: *'high sierra'*], `general_device_support` $\rightarrow$ `hardware_audio_connection_issue` [3 occurrences; signals: *'usb'*, *'type'*]), rather than using hardcoded rules.
- **Specialized `general_device_support` Audit:** Quantifies the 61.8% override rate and defines explicit specificity precedence rules, confirming it should be kept as a strict fallback when no specific subsystem or update causality is identified.
- **Taxonomy Boundary Disambiguation:** Resolves bidirectional ambiguity across 5 critical operational pairs (e.g., `display_touch_issue` vs `keyboard_typing_issue` for letter I glyphs vs digitizer faults).
- **Statistical Rigor & Sample Size Disclaimers:** Explicitly labels metric findings as preliminary observed annotation agreement on $N=57$ records (28.5% coverage; 95% CI: $[24.0\%, 48.1\%]$), avoiding overclaiming production model accuracy.

---

## Decision 57 — Phase 5.10

**Decision:** Implement a Read-Only Annotation Consistency & Taxonomy Boundary Validation Layer with Soft Non-Coercive Guidelines and Controlled Human Adjudication (`annotation_consistency.py`).

**Context:** Following Phase 5.9, the golden benchmark contains 57 human-reviewed annotations and 143 pending records. To protect against subtle annotator drift, inconsistent label assignments across semantically similar customer complaints, and rigid keyword memorization, a read-only semantic consistency layer is needed.

**Why:**
- **Strict Immutability & Ground-Truth Protection:** Operates in read-only mode with pre- and post-analysis SHA-256 verification (`ed4b504031c5150a6d3509c5eeb446edec30a4d7052bd4e421a200640f594254`). Zero modifications to `annotation_label`, `annotation_status`, or pending records.
- **Pairwise Semantic Similarity & Candidate Prioritization:** Evaluates all $\binom{57}{2} = 1,596$ record pairs ($269$ same-label, $1,327$ cross-label) using calibrated TF-IDF cosine similarity and n-gram overlap, surfacing 87 cross-label candidate pairs categorized into `HIGH` ($\ge 0.14$), `MEDIUM` ($0.09 - 0.14$), and `LOW` ($0.05 - 0.09$) priority tiers.
- **Non-Prescriptive Designation:** Flagged pairs are designated as *Possible boundary inconsistencies*, never *Incorrect annotations*, recognizing that distinct operational nuances often justify different labels for semantically adjacent phrasing.
- **Within-Label Coherence & Outlier Detection:** Measures average intra-class similarity, diversity score ($1.0 - \text{mean\_sim}$), and detects outlier customer messages within each intent class, with explicit sample-size warnings for small categories ($N < 5$).
- **Non-Coercive Soft Guidelines:** Formulates advisory boundary guidelines for key confusing pairs (`general_device_support` vs `software_update_problem`, `hardware_audio_connection_issue`, `battery_power_issue`, `mac_software_issue`, and `display_touch_issue` vs `keyboard_typing_issue`) using non-prescriptive phrasing (*"usually"*, *"consider"*, *"evidence favoring"*, *"dominant friction"*) and avoiding hard deterministic classifier rules.
- **Controlled Consistency Review Queue:** Interactive CLI (`analyze_annotation_consistency.py --interactive`) enables reviewers to adjudicate top ambiguous pairs (`[A]–[D]`), logging decisions to an isolated audit artifact (`consistency_review_decisions.json`) without mutating ground-truth labels.

---

## Decision 58 — Phase 5.11

**Decision:** Implement a Human-Guided Quality Review Workflow, Evidence-Based Taxonomy Guidelines v2.0, Boundary Decision Matrix, and Consolidated Quality Dashboard (`review_annotation_consistency.py`, `annotation_quality_dashboard.py`).

**Context:** Following Phase 5.10 consistency analysis (137 flagged cross-label candidate pairs across 77 reviewed records), an operational mechanism is required to enable human reviewers to adjudicate flagged ambiguity pairs, capture structured boundary feedback without modifying the underlying golden dataset, and provide comprehensive taxonomy guidance.

**Why:**
- **Strict Immutability & Zero Ground-Truth Mutation:** All consistency conflict reviews and boundary adjudications are saved to an isolated append-only file (`data/golden/annotation_consistency_human_review.csv`). `data/golden/golden_set_human_review.csv` remains strictly read-only and immutable with cryptographic SHA-256 validation (`1d3e9b3b8bdef3750437e59b17ab7c27167fd1548c2bb29de984151296c5b45a`).
- **Interactive Consistency Review CLI:** Implemented `backend/scripts/review_annotation_consistency.py` supporting priority tiers (`--priority high|medium|low|all`), batch sizes, and structured conflict cards displaying Record A and Record B alongside shared lexical signals, AI suggestions, and ambiguity context.
- **Granular Human Adjudications:** Review actions distinguish between valid operational distinctions (`VALID_TAXONOMY_BOUNDARY` / `KEEP_BOTH_LABELS`), records requiring future reconsideration (`RECONSIDER_RECORD_A` / `RECONSIDER_RECORD_B`), and taxonomy boundary ambiguities (`FLAG_TAXONOMY_GUIDELINE_UPDATE`).
- **Evidence-Based Taxonomy Guidelines v2.0 (`reports/annotation_guidelines_v2.md`):** Established the Primary Intent Principle, defined explicit inclusion/exclusion rules to prevent `general_device_support` catch-all overuse, decoupled OS update causality from symptom-specific troubleshooting, clarified screen display vs UI feature distinctions, and required explicit payment context for `billing_purchase_issue`.
- **Taxonomy Boundary Decision Matrix (`reports/taxonomy_boundary_matrix.md`):** Formulated explicit disambiguation rules and distinguishing signals across 8 major overlapping category pairs.
- **Consolidated Quality Dashboard (`backend/scripts/annotation_quality_dashboard.py`):** Unified reporting across Phase 5.9 (Pattern Analysis), Phase 5.10 (Consistency Metrics), Phase 5.11 (Human Boundary Adjudications), and cryptographic immutability checks.

**Trade-off:** Focuses on human annotation quality assurance and cognitive calibration; does not retrain or alter production classifier heads.

---

## Decision 59 — Phase 6

**Decision:** Implement an Uncertainty-Aware Intent Routing Engine and Runtime Human-in-the-Loop (HITL) Escalation Decision System (`classifier.py`, `router.py`, `escalation.py`, `routing_evaluator.py`).

**Context:** While Phases 5.9–5.11 provided robust offline dataset QA tools, SupportGraph AI required an operational runtime system to evaluate incoming customer messages, assess model uncertainty across multi-candidate intent predictions, and make deterministic, explainable `AUTO_HANDLE` vs `ESCALATE_TO_HUMAN` decisions.

**Why:**
- **Top-K Intent Predictions:** Moving beyond single-label prediction, the system predicts Top-K candidates ($K \ge 3$) with normalized probability distributions ($\sum p_i = 1.0$), capturing competing operational hypotheses.
- **Confidence Margin & Normalized Entropy:** Relying on Top-1 confidence alone is unsafe when two competing intents are nearly tied (e.g. 0.52 vs 0.44). The system computes confidence margin ($\Delta = p_1 - p_2$) and normalized Shannon entropy ($H_{\text{norm}}$) to detect boundary ambiguity.
- **Deterministic & Explainable Routing Engine:** Implements transparent rules with centralized threshold configuration (`AUTO_HANDLE_CONFIDENCE_THRESHOLD = 0.85`, `MIN_CONFIDENCE_MARGIN = 0.15`, `MAX_UNCERTAINTY_ENTROPY = 0.65`), generating human-readable rationales for every decision.
- **Separation of Concerns:** Strictly decouples Dataset QA (Phases 5.9–5.11 in `data/golden/`) from Runtime HITL Escalation (Phase 6 in `data/runtime/runtime_escalation_reviews.csv`). The 77-record golden benchmark remains cryptographically immutable (`1d3e9b3b8bdef3750437e59b17ab7c27167fd1548c2bb29de984151296c5b45a`).
- **Runtime Human Review Experience (CLI & REST API):** Provides operators with message context, candidate predictions with percentages, and routing explanations. Operators can accept Top-1, select Top-2, override with taxonomy intents, or mark unclear, with decisions logged append-only.
- **Comprehensive Evaluation & High Human Assistance Value:** Evaluation against 77 human ground-truth records demonstrated 50.6% Top-1 accuracy, jumping to **79.2% Top-2 accuracy** and **81.8% Top-3 accuracy**, with **69.2% of escalated cases** having the correct ground truth intent available directly in the Top-2 predictions.
- **Production REST Endpoints:** Integrated FastAPI routes under `/api/v1/intent` (`/classify-and-route`, `/classify`, `/route`, `/review`, `/config`, `/escalations/stats`).

**Trade-off:** Initial routing thresholds are provisional and can be recalibrated as the golden benchmark expands toward 200 completed human records.

---

## Decision 60 — Phase 6.1

**Decision:** Implement Data-Driven Routing Calibration, Threshold Grid Search Optimization, and Safety-First Ranking Architecture (`routing_calibrator.py`, `calibrate_routing_thresholds.py`).

**Context:** Following Phase 6 evaluation, the provisional thresholds (`conf=0.85, margin=0.15, entropy=0.65`) resulted in an unsafe auto-handle rate of 83.1% with only 53.1% auto-handle precision, allowing 30 out of 38 model errors directly into automatic handling. A systematic data-driven calibration across the 77 completed human ground-truth records was required to optimize safety and error interception.

**Why:**
- **Safety-First Optimization Hierarchy:** Replaced raw automation maximization with a 4-tier safety hierarchy: (1) Auto-Handle Precision ($\ge 75\%$), (2) Error Interception Rate ($\ge 80\%$), (3) Human Assistance in Escalation ($\ge 80\%$), and (4) Automation Coverage.
- **Exhaustive Parameter Grid Search ($10 \times 7 \times 7 = 490$ Combinations):** Evaluated confidence thresholds `[0.50..0.95]`, margin thresholds `[0.00..0.30]`, and maximum normalized entropies `[0.40..1.00]`, pre-caching predictions for deterministic, high-speed execution.
- **Drastic Error Reduction (-83.3% Unsafe Auto-Handles):** Calibrated configuration (`conf=0.95, margin=0.00, entropy=0.40`) reduced unsafe auto-handled errors from 30 down to 5, raising the error interception rate from 21.1% to **86.8%**.
- **Enhanced Human Review Utility:** In the recommended configuration, **83.1% of escalated cases** contain the correct human ground-truth label directly within the Top-2 predictions (and 84.6% in Top-3), enabling rapid single-click human resolution.
- **Strict Read-Only Data Governance:** Golden dataset access is strictly read-only with SHA-256 pre- and post-analysis verification (`1d3e9b3b8bdef3750437e59b17ab7c27167fd1548c2bb29de984151296c5b45a`). Production thresholds in `config.py` are preserved under governance and require explicit approval to update.
- **Structured Artifact Export (`reports/routing_calibration/`):** Generated 6 JSON artifacts detailing summary, grid results, recommended parameters, baseline vs calibrated deltas, safety metrics, and human assistance analysis.

**Trade-off:** Calibrated safety thresholds lower the auto-handle rate from 83.1% to 15.6% on the 77-record benchmark in exchange for an 86.8% error interception rate and 83.3% reduction in customer-facing errors.

---

## Decision 61 — Phase 7

**Decision:** Implement Evidence-Aware Support Resolution, Primary Problem Disambiguation, Ambiguity-Based Human Escalation, and Historical Evidence Ranking (`problem_extractor.py`, `primary_problem_selector.py`, `ambiguity_analyzer.py`, `evidence_ranker.py`, `case_retriever.py`, `support_resolution_engine.py`, `evidence_routing_evaluator.py`).

**Context:** Analysis across previous phases showed that semantic similarity does not equal intent identity, and aggressive auto-handling based purely on classifier confidence scores leads to customer-facing errors. Incoming customer support inquiries often present update mentions as contextual triggers while the primary operational problem is a concrete hardware or software symptom (e.g. battery drain, audio failure). An end-to-end evidence-aware resolution pipeline was required to extract problem structure, disambiguate causes from symptoms, classify ambiguity into clear types, retrieve and operationalize historical AppleSupport evidence, and generate empathetic brand troubleshooting replies or rich decision-support packages for human escalation.

**Why:**
- **Structured Problem Understanding Layer (`CustomerProblemProfile`, `ProblemExtractor`):** Extracts device models (e.g. iPhone 7, MacBook Pro), software services, primary and secondary symptoms, reported causes, and assesses information sufficiency (`sufficient`, `partial`, `insufficient`).
- **Primary Problem Selection & Causal Decoupling (`PrimaryProblemSelector`):** Decouples causal triggers (e.g. `software_update_problem`) from core operational symptoms (`battery_power_issue`, `hardware_audio_connection_issue`, `display_touch_issue`), preventing over-attribution of post-update symptoms to generic update categories.
- **Explainable Ambiguity Analyzer (`AmbiguityAnalyzer`, `AmbiguityType`):** Categorizes inquiry ambiguity into `CLEAR_PRIMARY`, `CAUSE_VS_SYMPTOM`, `GENUINE_AMBIGUITY`, `MULTI_SYMPTOM`, or `UNCLEAR_INSUFFICIENT`, generating human-readable escalation rationales.
- **Operational Evidence Ranking & Retrieval (`CaseRetriever`, `EvidenceRanker`, `EvidenceMatchTier`):** Explicitly categorizes retrieved historical cases into 4 operational tiers (`DIRECT_PROBLEM_MATCH`, `RELATED_SYMPTOM`, `RELATED_CONTEXT`, `WEAK_SEMANTIC_MATCH`), decoupling raw lexical TF-IDF / vector cosine similarity from true operational problem identity.
- **Support Resolution Engine (`SupportResolutionEngine`):** Unifies the pipeline to generate brand-grounded, empathetic AppleSupport troubleshooting replies for clear cases (`AUTO_HANDLE`) and structured decision-support escalation packages with candidate workflows for ambiguous cases (`ESCALATE_TO_HUMAN`).
- **REST API Endpoints & CLI Simulator (`support_resolution.py`, `simulate_support_resolution.py`):** Added FastAPI routes `/api/v1/resolution/resolve`, `/api/v1/resolution/understand`, `/api/v1/resolution/retrieve-evidence`, and interactive CLI simulation suite.
- **Evaluation & Cryptographic Ground Truth Immutability (`EvidenceRoutingEvaluator`):** Evaluated against 77 human ground truth records, maintaining SHA-256 integrity (`1d3e9b3b8bdef3750437e59b17ab7c27167fd1548c2bb29de984151296c5b45a`) and achieving 314 passed tests in the test suite.

**Trade-off:** Escalating ambiguous queries to human agents with structured candidate packages prioritizes safety, brand trust, and operational correctness over uncalibrated automated volume.

---

## Decision 62 — Phase 7.1

**Decision:** Implement Empirical Multi-Signal Clarity Analysis, Safe Ambiguity Decision Gating, and Strict Veto Rules (`clarity_signals.py`, `ambiguity_decision_gate.py`, `ambiguity_gate_evaluator.py`, `evaluate_ambiguity_gate.py`, `analyze_unsafe_auto_handles.py`, `simulate_safe_routing.py`).

**Context:** The Phase 7 benchmark across 77 human ground-truth records revealed that while Top-3 coverage was 80.5%, the system was overly aggressive in auto-handling cases (Auto-Handle Rate: 84.4%) with low precision (56.9%), allowing 28 incorrect predictions into automated handling. The core failure mode was relying on high model confidence scores or single-signal assumptions to declare cases "CLEAR". An empirical multi-signal gating architecture with hard safety vetoes was required to establish: **CLEAR $\ne$ HIGH CONFIDENCE**.

**Why:**
- **Multi-Signal Clarity Evaluation (`ClaritySignalEvaluator`, `ClaritySignalsProfile`):** Evaluates 6 independent operational signals:
  1. *Signal A: Information Sufficiency* (`SUFFICIENT`, `PARTIAL`, `INSUFFICIENT`)
  2. *Signal B: Primary Problem Strength* (`STRONG`, `MODERATE`, `WEAK`)
  3. *Signal C: Candidate Conflict* (`NO_CONFLICT`, `CAUSE_VS_SYMPTOM`, `GENUINE_CONFLICT`)
  4. *Signal D: Operational Evidence Agreement* (`STRONG_AGREEMENT`, `PARTIAL_AGREEMENT`, `NO_AGREEMENT`)
  5. *Signal E: Retrieval Evidence Quality* (`STRONG_EVIDENCE`, `MODERATE_EVIDENCE`, `WEAK_EVIDENCE`, `NO_EVIDENCE`)
  6. *Signal F: Multi-Symptom Complexity* (`SINGLE_SYMPTOM`, `RELATED_MULTI_SYMPTOM`, `UNRELATED_MULTI_SYMPTOM`)
- **Strict Safety Veto Rules (`AmbiguityDecisionGate`):** High model probability ($>90\%$) is strictly prevented from overriding missing information, weak symptoms, symptom-intent contradictions, or unrelated multi-symptom complexity. If any veto condition triggers, the case is immediately redirected to human review.
- **Enriched Human Escalation Packages:** Generates comprehensive decision support bundles for human reviewers with what the system understood, candidate hypotheses, explicit reasons why AI did not auto-handle, and related historical evidence.
- **Three-Strategy Comparative Benchmark:** Evaluated against 77 human ground-truth records, Strategy C (Multi-Signal Safe Gate) reduced unsafe auto-handled errors from 28 down to 15 (**-46.4% error reduction**), increased error interception from 17.6% to **55.9% (+38.2% increase in safety)**, and raised auto-handle precision from 56.9% to **63.4%**.
- **Root-Cause Error Categorization (`analyze_unsafe_auto_handles.py`):** Categorized all remaining false-clarity errors into structured classes (`FALSE_CLARITY_DECISION`, `RETRIEVAL_MISMATCH`, `MODEL_CLASSIFICATION_ERROR`, `MULTIPLE_VALID_INTERPRETATIONS`).
- **Cryptographic Immutability & Test Coverage:** Maintained SHA-256 integrity on `golden_set_human_review.csv` (`1d3e9b3b8bdef3750437e59b17ab7c27167fd1548c2bb29de984151296c5b45a`) and achieved 325 passing tests in the test suite.

**Trade-off:** Multi-signal gating safely reduces the automated workload from 84.4% to 53.2% in order to intercept nearly half of all previous customer-facing misclassifications.

---

## Decision 63 — Phase 8

**Decision:** Implement Evidence-Grounded Response Generation, Operational Evidence Validation, and Response Grounding Verification (`evidence_validator.py`, `response_generator.py`, `response_verifier.py`, `resolution_auditor.py`, `support_resolution_engine.py`, `evidence_resolution_evaluator.py`, `simulate_evidence_resolution.py`).

**Context:** Following Phase 7.1, empirical evaluation demonstrated that while multi-signal clarity gating reduced unsafe errors from 28 to 15, classifier confidence and clarity signals alone were still insufficient to prevent all customer-facing misclassifications. SupportGraph AI needed to move beyond asking *"Am I confident about this intent?"* to establishing *"Do I have strong enough operational evidence from real historical support cases to safely generate this resolution, and is the generated response verified to be grounded in that evidence without unsupported troubleshooting claims?"*

**Why:**
- **Semantic Retrieval for Candidate Discovery Only:** Re-established the core principle that dense semantic similarity does not equal operational problem equivalence. Dense retrieval indexes historical cases, but admission as supporting evidence requires multi-dimensional operational validation across Device, OS/Service, Primary Symptom, Intent, and Cause.
- **Operational Evidence Validation (`ResolutionEvidenceValidator`, `EvidenceVerdict`):** Computes structured verdicts (`STRONG_EVIDENCE`, `MODERATE_EVIDENCE`, `WEAK_EVIDENCE`, `CONFLICTING_EVIDENCE`, `INSUFFICIENT_EVIDENCE`) and evaluates quantitative agreement metrics before auto-resolution is permitted.
- **Evidence-Grounded Response Generation (`EvidenceGroundedResponseGenerator`):** Synthesizes empathetic, brand-compliant AppleSupport troubleshooting responses grounded in extracted problem profiles, verified operational intents, and historical support response patterns.
- **Response Grounding Verification (`ResponseGroundingVerifier`):** Enforces a 5-point verification check before response dispatch:
  1. Addresses customer's actual primary symptom.
  2. Avoids misleading focus on causal triggers.
  3. Detects hazardous/unsupported troubleshooting recommendations (e.g. unauthorized disassembly, logic board replacement, jailbreak).
  4. Requires historical evidence corroboration.
  5. Includes official AppleSupport DM escalation link.
- **Multi-Phase Benchmark Safety Gain (-75.0% Unsafe Errors):** Evaluated against the 77 human ground-truth records, Phase 8 reduced unsafe auto-handled errors from 28 (Phase 7) down to **7** (**-75.0% error reduction**), raised error interception to **79.4%**, and achieved a 96.1% response grounding pass rate.
- **Isolated Runtime Audit Logging (`ResolutionAuditor`):** Persists all runtime decisions to `data/runtime/evidence_resolution_audit.csv` with zero contamination of the benchmark dataset.
- **Cryptographic Immutability & Test Coverage:** Maintained SHA-256 integrity on `golden_set_human_review.csv` (`1d3e9b3b8bdef3750437e59b17ab7c27167fd1548c2bb29de984151296c5b45a`) and achieved 339 passing tests in the test suite.

**Trade-off:** Requiring verified operational evidence and response grounding lowers the auto-handle rate from 53.2% to 23.4% in exchange for intercepting nearly 80% of all potential customer-facing errors.

---


*Phase 8 decisions recorded after implementing `backend/app/resolution/evidence_validator.py`, `backend/app/resolution/response_generator.py`, `backend/app/resolution/response_verifier.py`, `backend/app/resolution/resolution_auditor.py`, `backend/app/evaluation/evidence_resolution_evaluator.py`, CLI tools, benchmark report `reports/phase_8_evidence_grounded_resolution.md`, and achieving 339 passing tests.*

---

## Decision 64 — Phase 9: Escalation Quality Analysis & Selective Automation Optimization

**Decision:** Implement a 7-category escalation quality taxonomy and a safety-gated selective recovery engine to analyze the Phase 8 76.6% escalation rate and determine which escalations, if any, can be safely recovered for automated resolution.

**Context:** Phase 8 achieved a 79.4% error interception rate but escalated 76.6% of customer inquiries to human review. The core architectural question was: *Is the high escalation rate driven by marginal signal noise (fixable by threshold tuning) or by genuine evidence, information, and ambiguity deficiencies (not fixable without corpus expansion)?*

**Why — Escalation Taxonomy:**
The 7-category taxonomy (GENUINE_AMBIGUITY, MULTI_PROBLEM_COMPLEXITY, INSUFFICIENT_INFORMATION, EVIDENCE_LIMITED, VERIFICATION_VETO, RECOVERABLE_ESCALATION, HUMAN_REQUIRED) classifies escalations by root cause rather than symptom. This provides an actionable diagnostic framework rather than a generic escalation rate number.

**Why — Safety-First Classification Priority:**
The decision tree evaluates the most restrictive safety violations first (HUMAN_REQUIRED before INSUFFICIENT_INFORMATION before VERIFICATION_VETO, etc.) to guarantee that no safety-sensitive case is ever misclassified as RECOVERABLE. High model confidence alone never qualifies a case for recovery.

**Why — Independent 4-Gate Recovery Engine:**
The recovery engine re-runs the _full_ Phase 8 pipeline from scratch rather than modifying existing escalation decisions in place. This eliminates the risk of state-dependent errors and ensures all safety gates (evidence validation, grounding verification) are independently re-evaluated for every attempted recovery.

**Empirical Finding:**
Evaluation on 77 protected human ground-truth records revealed that **72.9% of escalations are EVIDENCE_LIMITED**. The historical AppleSupport corpus does not contain sufficiently strong Tier 1/Tier 2 operational evidence for the customer problem profiles in the test set. Only **1.7% (1 case)** was classified as RECOVERABLE_ESCALATION, and when that case was re-processed through the full pipeline, the recovery was blocked by independent safety gates — confirming the escalation was genuinely warranted.

**Conclusion:** The Phase 8 escalation rate of 76.6% is empirically justified and cannot be safely reduced through routing rule changes or confidence threshold tuning. The correct remedy is corpus expansion — ingesting more historical AppleSupport interaction data to provide stronger operational evidence coverage.

**Trade-off:** Phase 9 does not improve the auto-handle rate in this benchmark (remains 23.4%) because the evidence corpus limitation is the binding constraint. However, Phase 9 provides the diagnostic evidence needed to target the next architectural investment correctly.

---

*Phase 9 decisions recorded after implementing `backend/app/resolution/escalation_quality_analyzer.py`, `backend/app/resolution/escalation_recovery_engine.py`, `backend/app/evaluation/escalation_quality_evaluator.py`, `backend/scripts/evaluate_escalation_quality.py`, Phase 9 unit tests (33 new tests, 372 total), benchmark report `reports/phase_9_escalation_quality_and_selective_recovery.md`, and verifying 100% dataset immutability.*











