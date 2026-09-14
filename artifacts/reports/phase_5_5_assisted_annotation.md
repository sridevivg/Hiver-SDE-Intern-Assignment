# Phase 5.5 — AI-Assisted Human Annotation Workflow

> **SupportGraph AI — Scientific Annotation Architecture Report**  
> **Brand Under Analysis:** AppleSupport  
> **LLM Engine:** Groq API (`openai/gpt-oss-20b`)  
> **Date:** September 2026  
> **Status:** Phase 5.5 Implemented — AI Advisory Workflow Ready  

---

## 1. Executive Summary

Phase 5.5 introduces an **AI-assisted human-in-the-loop annotation workflow** for the 200-record SupportGraph AI Golden Evaluation Set. 

In applied natural language processing and customer support intelligence, manually annotating hundreds of customer inquiries is time-intensive. While Large Language Models (such as `openai/gpt-oss-20b` via Groq) can assist by drafting provisional category assignments, **treating AI suggestions as ground truth destroys scientific validity**.

Phase 5.5 establishes a rigorous, fail-safe architecture that leverages LLM efficiency while enforcing absolute human decision authority. The system generates structured advisory suggestions (`model_suggested_label`, `model_confidence`, `model_reasoning_summary`), but strictly isolates them from ground-truth annotation fields (`annotation_label`, `annotator`, `annotation_status`), which remain unpopulated until explicitly verified by a human annotator.

---

## 2. What AI Did vs. What AI Did NOT Do

To prevent ambiguity or exaggerated claims regarding AI autonomy, the responsibilities of the model and the human are strictly delineated:

### What AI Did:
- **Suggested Candidate Labels:** Generated a single suggested intent chosen strictly from the 9 approved operational intents or `unclear_needs_review`.
- **Provided Calibrated Confidence Estimates:** Produced a float score $[0.0, 1.0]$ reflecting model certainty based on semantic match with taxonomy definitions.
- **Provided Concise Reasoning Summaries:** Drafted a 1–2 sentence explanation citing explicit customer keywords or failure symptoms to assist human reviewers during evaluation.
- **Flagged Borderline Inquiries:** Raised `model_needs_human_review = True` when queries were ambiguous, mixed multiple issues, or had low confidence ($< 0.70$).

### What AI Did NOT Do:
- **Did NOT Create Ground Truth:** Model suggestions carry zero normative weight in the final golden evaluation benchmark.
- **Did NOT Automatically Populate Human Fields:** `annotation_label` and `annotator` remain strictly empty (`""`), and `annotation_status` is locked to `"pending_human_review"`.
- **Did NOT Finalize or Freeze the Golden Set:** Freezing requires 100% human verification via `freeze_golden_set.py`.
- **Did NOT Replace Human Judgment:** Reviewers can accept, override, or classify records as unclear with a single keystroke.

---

## 3. Scientific Integrity & The Perils of AI-Assisted Ground Truth

> **Core Principle:** AI-assisted annotation is NOT equivalent to independently human-labeled ground truth.

Deploying LLMs in the data annotation loop introduces severe methodological risks that must be actively countered:

### 1. The Anchoring Bias Hazard
When human reviewers are presented with a pre-selected AI recommendation, cognitive psychology demonstrates a strong tendency to accept the suggestion passively rather than independently scrutinize difficult or borderline cases. 
- *Countermeasure in SupportGraph AI:* The human review CLI (`review_golden_labels.py`) presents the full customer message and conversation context prominently, explicitly lists all 9 operational intents with numeric shortcuts `[1-9]`, and requires active keystrokes (`[A]` to accept, `[1-9]` to override, `[U]` for unclear). Overrides are explicitly tracked in the audit trail as `overridden_ai_suggestion`.

### 2. Circular Validation & Model Blindspots
If a model's suggestions are accepted uncritically to construct a test set, evaluating future classifiers against that test set will measure agreement with the assisting model's preconceptions, not real-world human intent. Systematic model blindspots (e.g. conflating an app crash after an OS update with general device slowness) become baked into the evaluation benchmark.
- *Countermeasure in SupportGraph AI:* All overrides and disagreements are logged in `data/golden/golden_set_human_review.csv` and audited via `annotation_audit.py`.

### 3. Model-Human Agreement vs. Inter-Annotator Agreement
A critical failure in ML reporting is calculating agreement between an LLM and a human and describing it as "inter-annotator agreement" or computing Cohen's Kappa to claim high reliability.
- *Countermeasure in SupportGraph AI:* We explicitly reject this conflation. The metric measuring model acceptance is formally designated:
  $$\text{MODEL\_HUMAN\_AGREEMENT} = \frac{\sum (\text{Model Suggestion} == \text{Human Decision})}{\text{Total Human Reviewed}}$$
  This metric reflects **model suggestion accuracy/alignment**, NOT inter-rater reliability. Inter-annotator agreement is reserved exclusively for two independent human reviewers (`annotation_agreement.py`).

---

## 4. Architecture & Pipeline Overview

The Phase 5.5 pipeline consists of four decoupled layers:

```
[ data/golden/golden_set_annotation_template.csv ] (200 records, labels empty)
                       │
                       ▼
    [ backend/scripts/suggest_golden_labels.py ] (Groq openai/gpt-oss-20b)
                       │
                       ▼
[ data/golden/golden_set_ai_suggestions.csv ] (model fields added; annotation_label STILL EMPTY)
                       │
                       ▼
    [ backend/scripts/review_golden_labels.py ] (Interactive Human CLI)
                       │
       ┌───────────────┴───────────────┐
       ▼                               ▼
[ Accept (A) ]                [ Override (1-9) / Unclear (U) ]
annotation_status="reviewed"   annotation_status="overridden_ai_suggestion"
       │                               │
       └───────────────┬───────────────┘
                       ▼
[ data/golden/golden_set_human_review.csv ] (Human ground truth established)
                       │
                       ▼
    [ backend/app/evaluation/annotation_audit.py ]
                       │
                       ▼
[ data/golden/golden_annotation_progress.json ] (MODEL_HUMAN_AGREEMENT audit)
```

---

## 5. Security & Fail-Safe Implementation

1. **API Key Protection:** `GROQ_API_KEY` is loaded from `backend/.env` through Pydantic Settings. Keys are never printed to terminal, written to logs, or embedded in generated CSVs or manifests. If an exception occurs, keys are automatically redacted (`[REDACTED]`).
2. **Interrupt & Resume Capability:** Both `suggest_golden_labels.py` and `review_golden_labels.py` save progress after every single record. If a process is terminated due to rate limits, network timeouts, or user exit (`Q`), re-running with `--resume` resumes execution seamlessly from the first uncompleted record.
3. **Dry-Run Validation:** The `--dry-run` flag validates input schema, taxonomy definitions, and environment variables without issuing API calls or modifying data files.

---

## 6. Current Status & Next Steps

- **Completed:** 
  - Groq annotation assistant service (`backend/app/evaluation/annotation_assistant.py`)
  - Batch suggestion CLI with resume & dry-run (`backend/scripts/suggest_golden_labels.py`)
  - Human review domain logic (`backend/app/evaluation/human_review.py`)
  - Interactive terminal review CLI (`backend/scripts/review_golden_labels.py`)
  - Model-human audit engine with scientific integrity checks (`backend/app/evaluation/annotation_audit.py`)
  - Progress tracking manifest (`data/golden/golden_annotation_progress.json`)
- **Status:** **Phase 5.5 Implemented — Human Review Ready**.
- **Next Step:** Execute unit test suite, perform dry-run verification, and run a 5-record smoke test batch to inspect model suggestions.
