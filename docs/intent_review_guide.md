# SupportGraph AI — Human Review Guide: Intent Discovery

> **Purpose:** Standard operating procedure for human experts to review unsupervised customer clusters in `data/interim/intent_cluster_review.csv` and transform them into a practical, grounded intent taxonomy.

---

## 1. What Is a Customer Support Intent?

An **intent** represents the customer's underlying goal or root problem that prompted them to reach out for assistance.

- An intent answers: *"What action is the customer trying to take, or what obstacle are they experiencing?"*
- An intent is **NOT** simply the product mentioned (e.g. *iPhone*, *MacBook*, *Apple Watch*).
- An intent is **NOT** a sentiment or generic emotion (e.g. *angry*, *confused*).

### Intent vs. Product/Topic (Crucial Distinction)

| Entity / Topic (What is NOT an intent) | True Intent (Customer Problem / Action) |
|---|---|
| `iPhone 11` | `battery_drain_issue` or `screen_freeze_crash` |
| `Apple ID` | `account_access_issue` or `password_reset_request` |
| `iOS 11.1 Update` | `software_update_failure` or `post_update_bug_report` |
| `AirPods` | `bluetooth_pairing_issue` or `audio_hardware_defect` |
| `Apple Music` | `subscription_billing_inquiry` or `media_sync_issue` |

---

## 2. Review Workflow: How to Inspect Clusters

Open the generated review file:
```
data/interim/intent_cluster_review.csv
```

For each cluster row:
1. **Examine `cluster_size` and `percentage`:** Understand how prevalent this topic is across the customer base.
2. **Review `top_terms`:** Scan the most distinctive TF-IDF keywords that separated this cluster in the vector space.
3. **Inspect `representative_examples`:** Read the 3–5 real customer messages positioned closest to the cluster centroid. These reflect the purest semantic center of the cluster.
4. **Evaluate `proposed_label`:** Review the automated heuristic suggestion.
5. **Decide Action:** Assign `review_status` and provide `final_intent_label`.

---

## 3. Decision Rules: When to Merge, Split, or Reject

### When to ACCEPT a Cluster
- **Criteria:** The representative messages describe a coherent, actionable problem with a consistent resolution path.
- **Action:** Set `review_status = accepted`. Provide a descriptive snake_case name in `final_intent_label` (e.g. `software_update_issue`).

### When to MERGE Clusters
- **Criteria:** Two or more clusters describe the same operational problem with slightly different wording (e.g. one cluster says *"can't login"* and another says *"locked out of account"*).
- **Action:** Set `review_status = merged` on both clusters. In `final_intent_label`, enter the identical target intent name (e.g. `account_access_issue`) and note `merged_with: cluster_X`.

### When to SPLIT a Cluster
- **Criteria:** A cluster combines two fundamentally different customer inquiries that happen to share a common vocabulary (e.g. hardware battery replacement combined with software app crashes).
- **Action:** Set `review_status = split`. In `final_intent_label`, document the target sub-intents for Phase 5 curation.

### When to REJECT a Cluster
- **Criteria:** The cluster consists purely of conversational noise, spam, single-word greetings (*"hello?"*), or unintelligible fragments that provide no support signal.
- **Action:** Set `review_status = rejected`. Leave `final_intent_label = null`.

---

## 4. Standardized Intent Naming Conventions

All finalized intent labels must adhere to these conventions:
1. **Lowercase `snake_case`:** E.g. `account_access_issue`, `battery_performance_complaint`.
2. **Action/Problem Formulation:** Combine the affected domain with the problem state (`[domain]_[problem_type]`).
3. **Target Cardinality:** The final reviewed taxonomy should contain **5 to 10 practical, discrete intents** to support robust downstream multi-class classification.

---

## 5. Review Record Template

When updating `data/interim/intent_cluster_review.csv`:

```csv
cluster_id,cluster_size,percentage,top_terms,representative_examples,proposed_label,final_intent_label,review_status
0,2841,14.2%,update; ios; phone; battery,"...@AppleSupport My battery drains fast after iOS 11...",software_update_bug,software_update_issue,accepted
```
