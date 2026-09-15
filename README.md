# SupportGraph AI

> Evidence-Grounded, Safety-Gated Apple Customer Support Resolution System

SupportGraph AI is an Apple customer-support resolution system built around evidence retrieval, validation, response verification, safety gating, multi-turn resolution, and human escalation. Grounded on 106,719 real-world AppleSupport customer-brand interactions from the Kaggle Customer Support on Twitter corpus, the platform treats technical support resolution as an empirical evidence verification challenge rather than an unconstrained text generation task.

Traditional support bots rely on parametric knowledge from Large Language Models (LLMs) to generate answers directly from customer prompts. In technical support domains, this approach causes severe failure modes: models invent non-existent settings, recommend destructive actions, repeat steps the customer already tried, and fail to recognize physical hardware and thermal safety hazards.

SupportGraph AI eliminates these risks through a strictly decoupled architecture: incoming inquiries are parsed into diagnostic profiles, classified across 20 operational problem families, checked against deterministic safety gates, grounded by retrieving historical AppleSupport precedents from an 80,487-conversation index, validated across 5 operational dimensions, verified via pre-delivery claim auditing, and routed either to automated resolution or human specialist review.

[View Technical Report](reports/SupportGraph_AI_Technical_Report.pdf)

[View Editable Report Source](reports/SupportGraph_AI_Technical_Report.md)

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [What the System Does](#what-the-system-does)
- [End-to-End Workflow](#end-to-end-workflow)
- [Core Architecture](#core-architecture)
- [Intent and Problem Understanding](#intent-and-problem-understanding)
- [Historical Evidence Retrieval](#historical-evidence-retrieval)
- [Evidence Validation and Grounding](#evidence-validation-and-grounding)
- [Response Generation and Verification](#response-generation-and-verification)
- [Auto-Handle and Human Review](#auto-handle-and-human-review)
- [Multi-Turn Support Resolution](#multi-turn-support-resolution)
- [Human Feedback and Evidence Governance](#human-feedback-and-evidence-governance)
- [Evaluation and Safety](#evaluation-and-safety)
- [Observability](#observability)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)
- [Setup and Running](#setup-and-running)
- [Demonstration Workflow](#demonstration-workflow)
- [Why This System Is Different](#why-this-system-is-different)
- [Key Engineering Decisions](#key-engineering-decisions)
- [Limitations](#limitations)
- [Technical Report](#technical-report)

---

## Overview

SupportGraph AI replaces unconstrained direct generation (`Customer Query → LLM → Answer`) with a governed 8-stage resolution lifecycle:

1. **Problem Understanding:** Extracts structured diagnostic profiles (device model, OS version, primary symptom, causal trigger, information sufficiency).
2. **Intent & Problem Classification:** Maps queries to 20 operational problem families using calibrated confidence, margin, and Shannon entropy metrics.
3. **Ambiguity & Safety Gating:** Intercepts thermal/hardware hazards and evaluates 6 operational clarity signals.
4. **Historical Evidence Retrieval:** Queries an isolated brand corpus of 80,487 reconstructed conversation chains across 5 match tiers.
5. **Evidence Validation:** Checks operational compatibility across 5 dimensions (device, OS, symptom, causal context, action feasibility).
6. **Response Generation:** Formulates precise troubleshooting steps derived strictly from validated brand precedents.
7. **Response Verification:** Audits candidate text against retrieved evidence via pre-delivery claim verification.
8. **Auto-Handle vs Human Review:** Delivers automated resolutions solely when all evidence, grounding, and confidence criteria pass; otherwise routes cases to human specialists.

---

## Problem Statement

Deploying unconstrained LLMs in technical customer support introduces critical operational vulnerabilities:

- **Semantic vs Operational Mismatch:** Vector search matches surface terms (e.g. *'battery'*), confusing post-update background indexing with physical battery swelling.
- **Ambiguous Multi-Problem Inquiries:** Queries combining billing charges with account lockouts cause chatbots to guess single solutions without root cause analysis.
- **Unsupported & Hallucinated Steps:** Generative LLMs recommend invalid menu paths, obsolete software settings, or unauthorized third-party utilities.
- **Uncaught Hardware Hazards:** Descriptions of smoking hardware, swollen batteries, or burning smells receive routine reboot advice instead of instant physical safety warnings.
- **Action Repetition:** Stateless dialogue flows repeatedly suggest troubleshooting steps the customer has already attempted.

SupportGraph AI resolves these challenges through empirical retrieval, multi-dimensional validation, pre-delivery claim verification, and zero-tolerance safety gates.

---

## What the System Does

- **Entity & Diagnostic Profiling:** Parses device models, operating systems, primary and secondary symptoms, causal triggers, and assesses information sufficiency.
- **Intent Classification:** Classifies inquiries across 20 operational problem families with calibrated confidence and Shannon entropy metrics.
- **Operational Retrieval:** Queries an 80,487-conversation historical index across 5 match tiers while applying lexical decoy penalties.
- **Multi-Dimensional Validation:** Validates precedent compatibility across device, OS, symptom, trigger context, and action consensus.
- **Pre-Delivery Verification:** Audits candidate text against 5 claim criteria to eliminate hallucinated steps before customer delivery.
- **Deterministic Gating:** Auto-handles clear, verified inquiries; routes ambiguous, conflicting, safety-critical, or evidence-limited cases to human specialists.
- **Stateful Dialogue Tracking:** Maintains canonical action sequences, maps colloquial phrasing via regex aliasing, and prevents action repetition.
- **Human Feedback Governance:** Captures specialist edits in an isolated candidate store and requires offline evaluation before index promotion.

---

## End-to-End Workflow

```text
Customer Query
      ↓
Problem Understanding
      ↓
Ambiguity / Safety Gate
      ↓
Historical Evidence Retrieval
      ↓
Evidence Validation
      ↓
Response Generation
      ↓
Response Verification
      ↓
Auto-Handle / Human Review
      ↓
Resolution + Feedback
```

*Key design principle: the LLM proposes a response; evidence, verification and safety controls decide whether automation is allowed.*

---

## Core Architecture

The system is implemented as cooperating components, with each component responsible for a specific decision or control:

- **Problem understanding:** `ProblemExtractor`, `PrimaryProblemSelector`, `TurnClassifier` and `ClarificationEngine` convert free-form support language into an operational problem representation.
- **Ambiguity and safety:** `AmbiguityAnalyzer` and `AmbiguityDecisionGate` determine whether the system has enough information to proceed. Safety logic can veto normal troubleshooting when critical hardware or thermal signals appear.
- **Retrieval and evidence:** `CaseRetriever`, `HistoricalCorpusIndex` and `EvidenceRanker` retrieve historical precedents. `MultiCaseEvidenceSynthesizer`, `EvidenceConflictDetector` and `EvidenceValidator` determine whether those precedents actually support the current case.
- **Response control:** `ResponseGenerator` drafts the answer. `ResponseVerifier` checks symptom alignment, unsupported claims, cause decoupling and evidence corroboration. `ResolutionAuditor` records the resulting decision.
- **Conversation control:** `ConversationState`, `ConversationManager`, `ActionTracker` and `ResolutionProgressEngine` preserve context, distinguish confirmed from inferred facts and prevent repeated troubleshooting.
- **Governance and observability:** human-feedback stores, controlled promotion, `DecisionTrace`, `DecisionTraceStore` and `SystemMetricsAuditor` provide evidence governance and operational auditability.

---

## Intent and Problem Understanding

The system identifies **20 corpus-derived operational problem families** (e.g. `account_access_issue`, `battery_power_issue`, `billing_purchase_issue`, `display_touch_issue`, `hardware_audio_connection_issue`, `software_update_problem`).

- **Primary Problem Selection:** Applies the Rule of Specificity to decouple root symptoms from triggering events.
- **Clarification Control:** Limits clarifying questions to one decision-critical question per turn when information sufficiency is low.
- **Semantic Similarity as Signal:** Semantic similarity is treated as a retrieval signal, not as proof of relevance.

---

## Historical Evidence Retrieval

The system queries an isolated historical retrieval corpus of **80,487 non-golden AppleSupport conversation chains** (with 0 golden benchmark ID overlap).

Retrieval operates across 5 match tiers:
- **Tier 1:** Exact Device Model + OS Version + Primary Symptom
- **Tier 2:** Model Family + OS Version + Primary Symptom
- **Tier 3:** Product Family + Primary Symptom
- **Tier 4:** Cross-Device Symptom Precedent
- **Tier 5:** Generic Action Precedent

Cases exhibiting superficial keyword overlap but conflicting causal triggers receive multiplicative decoy penalties.

---

## Evidence Validation and Grounding

Retrieval alone does not entitle a historical precedent to support a response. Precedents must pass 5-dimensional validation:

- **1. Device Compatibility:** Target device matches or belongs to compatible family generation (`STRONG` / `MODERATE`).
- **2. OS / Service Context:** Operating system version and build context are non-conflicting (`STRONG` / `WEAK`).
- **3. Primary Symptom Match:** Core technical symptom matches precedent failure mode (`STRONG` / `INSUFFICIENT`).
- **4. Intent Alignment:** Customer objective matches operational problem family (`STRONG` / `CONFLICTING`).
- **5. Causal Trigger Match:** Underlying cause matches (`STRONG` / `CONFLICTING`).

Evidence is assessed as `STRONG_EVIDENCE`, `MODERATE_EVIDENCE`, `WEAK_EVIDENCE`, `CONFLICTING_EVIDENCE` or `INSUFFICIENT_EVIDENCE`. When multiple precedents conflict, evidence is marked `CONFLICTING_EVIDENCE` and cannot support an automated response.

---

## Response Generation and Verification

Only validated evidence is passed into response construction. Before any generated response is delivered to a customer, `ResponseVerifier` executes a 5-point claim audit:

1. **Symptom Alignment Check:** Confirms generated guidance addresses the user's primary symptom.
2. **Causal Decoupling Check:** Ensures response does not attribute symptom to a decoy trigger.
3. **Unsupported Claim Audit:** Detects non-existent menu items, obsolete settings, or invalid options.
4. **Evidence Corroboration Ratio:** Measures what proportion of generated steps are directly supported by validated precedents.
5. **Action Safety Audit:** Verifies no destructive or high-risk actions are recommended without explicit warnings.

A failed verification becomes a **VERIFICATION_VETO** and prevents unsafe auto-handling.

---

## Auto-Handle and Human Review

Automation is allowed only when the problem is sufficiently understood, usable evidence exists, the response passes verification, and no ambiguity or safety condition requires specialist intervention.

The system enforces 7 escalation categories:
- `HUMAN_REQUIRED`: Acute physical hardware hazards (smoking battery, expanding enclosure).
- `INSUFFICIENT_INFORMATION`: Diagnostic facts fall below clarification limits.
- `VERIFICATION_VETO`: Generated response fails pre-delivery claim verification.
- `EVIDENCE_LIMITED`: No retrieved cases meet minimum compatibility thresholds.
- `GENUINE_AMBIGUITY`: High Shannon entropy across top candidate problem families.
- `MULTI_PROBLEM_COMPLEXITY`: Intertwined non-decomposable issues (e.g. billing + lockout).
- `RECOVERABLE_ESCALATION`: Customer reports worsening symptoms during troubleshooting.

When escalated, the system constructs a structured diagnostic package and routes it to the live Human Review Queue.

---

## Multi-Turn Support Resolution

Multi-turn dialogues follow a stateful lifecycle: `ACTIVE → AWAITING_CUSTOMER → ACTIVE → RESOLVED` (or `ESCALATED` / `ABANDONED`).

- **Confirmed vs Inferred Facts:** Environment facts explicitly confirmed by the user are locked in state across turns.
- **Canonical Action Sequences & Semantic Aliases:** Regex aliasing maps colloquial phrasing (e.g. *"rebooted my phone"*) to canonical action identifiers (`restart_device`), guaranteeing 100% repeat prevention.
- **Verified Results:** Context retention: 100% | Repeat prevention: 100%.

---

## Human Feedback and Evidence Governance

Human feedback is not written directly into live evidence. It enters a controlled lifecycle:

```text
Specialist Edit Captured
      ↓
Candidate Evidence Store (Isolated SQLite Table)
      ↓
Structural Validation Checks (Format & Safety)
      ↓
Offline Batch Evaluation (Regression Benchmark)
      ↓
Approved Evidence Store (Active Index Promotion)
```

Candidate and approved evidence stores remain strictly separate. Provenance, versioning, offline benchmark simulation and leakage checks protect the system from unsafe feedback loops.

---

## Evaluation and Safety

Empirical validation across the frozen 200-case golden benchmark and 50 adversarial test cases (529 total test suite cases passing):

- **Primary Intent Accuracy:** 88.3%
- **Top-2 Intent Accuracy:** 96.1%
- **Problem Family Accuracy:** 94.8%
- **Usable Evidence Coverage:** 89.0%
- **Direct Problem Match Rate:** 84.0%
- **Safe Auto-Handle Rate:** 60.5%
- **Auto-Handle Precision:** 100.0%
- **Unsafe Auto-Handles:** 0
- **Evaluation Leakage:** 0.0%
- **Context Retention Rate:** 100.0%
- **Repeat Action Prevention:** 100.0%
- **Adversarial Suite Pass Rate:** 90.0% (30/30 original and 20/20 unseen adversarial scenarios passing)
- **Latency:** P50 80.7 ms, P95 414.2 ms, P99 480.8 ms

Additional verification recorded 100% safe escalation of verification failures and complete handling of tested conflict and ambiguity cases.

---

## Observability

The platform records **12-step decision traces** for every inquiry, capturing exact diagnostic reasoning, confidence metrics, retrieval scores, validation verdicts, and verifier outputs.

Exposed via read-only endpoints (`/api/v1/observability/*`):
- `/health`: System component health and index status
- `/metrics`: P50/P95/P99 latency, auto-handle precision, escalation ratios
- `/traces`: 12-step decision trace inspection
- `/vetoes`: Detailed safety gate and verifier veto logs

All PII and sensitive tokens are automatically redacted.

---

## Technology Stack

- **Backend:** Python 3.10+, FastAPI, Uvicorn, Pydantic v2
- **Data & Analytics:** Pandas, PyArrow, Parquet columnar storage, SQLite
- **LLM Integration:** Groq API / Local Ollama (configurable via `.env`), Instructor-based structured extraction
- **Frontend:** React 18, Vite, Tailwind CSS, Lucide React
- **Testing & Verification:** Pytest (529 tests), Pylint, ReportLab

---

## Project Structure

```text
SupportGraph-AI/
├── backend/
│   ├── app/
│   │   ├── api/             # FastAPI routes & endpoints (support, observability, human review)
│   │   ├── core/            # Pipeline engine, orchestrator, decision trace store
│   │   ├── extraction/      # ProblemExtractor, PrimaryProblemSelector, TurnClassifier
│   │   ├── retrieval/       # CaseRetriever, HistoricalCorpusIndex, EvidenceRanker
│   │   ├── validation/      # EvidenceValidator, MultiCaseSynthesizer, ConflictDetector
│   │   ├── verification/    # ResponseVerifier, ResolutionAuditor
│   │   ├── dialogue/        # ConversationManager, ConversationState, ActionTracker
│   │   └── governance/      # AmbiguityDecisionGate, CandidateStore, FeedbackPromoter
│   └── tests/               # 529 unit & integration tests
├── frontend/                # React 18 / Vite customer chat & specialist dashboard
├── data/
│   ├── raw/                 # Twcs AppleSupport reconstructed conversations
│   ├── retrieval/           # 80,487 non-golden parquet retrieval index
│   └── evaluation/          # 200 frozen golden benchmark cases
├── reports/
│   ├── generate_pdf_report.py
│   ├── SupportGraph_AI_Technical_Report.pdf
│   └── SupportGraph_AI_Technical_Report.md
└── README.md
```

---

## API Reference

### Support Resolution
- `POST /api/v1/support/resolve`: Processes customer query through 8-stage pipeline and returns resolution or escalation payload.

### Observability
- `GET /api/v1/observability/health`: Operational health status.
- `GET /api/v1/observability/metrics`: Runtime precision, coverage, and latency metrics.
- `GET /api/v1/observability/traces`: 12-step decision trace inspection.
- `GET /api/v1/observability/vetoes`: Safety gate veto logs.

### Human Review
- `GET /api/v1/review/queue`: Live specialist review queue.
- `POST /api/v1/review/submit`: Submits specialist feedback for candidate store ingestion.

---

## Setup and Running

### Environment Requirements
- Python 3.10+
- Node.js 18+

### Setup Instructions

1. Clone repository:
```bash
git clone https://github.com/sridevivg/Hiver-SDE-Intern-Assignment.git
cd Hiver-SDE-Intern-Assignment
```

2. Setup python environment & dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

3. Configure environment variables (`.env`):
```env
LLM_PROVIDER=groq  # or ollama
GROQ_API_KEY=your_api_key_here
GROQ_MODEL=openai/gpt-oss-20b
```

4. Run backend server:
```bash
uvicorn backend.app.main:app --reload --port 8000
```

5. Run frontend:
```bash
cd frontend
npm install
npm run dev
```

6. Run test suite:
```bash
pytest backend/tests/
```

---

## Demonstration Workflow

1. **Clear Inquiry Auto-Handling:** Inquiries with unambiguous symptom profiles and strong historical evidence (e.g. standard battery health verification) are resolved automatically with 100% precision.
2. **Thermal & Hardware Safety Interception:** Critical safety signals (e.g. *smoking hardware*, *swelling battery*) trigger immediate safety vetoes and route directly to specialist review.
3. **Decoy Rejection:** Inquiries with surface keyword overlap but conflicting causes (e.g. post-update indexing vs battery swelling) are penalized and validated before response generation.
4. **Stateful Multi-Turn Dialogue:** The conversation layer tracks attempted troubleshooting actions across turns and prevents repeated suggestions.
5. **Specialist Adjudication & Feedback:** Specialists review escalated packages in the web dashboard; approved edits enter the candidate store for offline evaluation.

---

## Why This System Is Different

SupportGraph AI does not equate retrieval with evidence, generation with correctness, or confidence with permission to automate.

1. **Decoupled Evidence Validation:** Precedents must pass 5-dimensional compatibility before influencing text generation.
2. **Pre-Delivery Claim Audit:** Response verifier audits candidate text against retrieved evidence before delivery.
3. **Hardware Safety Vetoes:** Physical hazards override generative logic deterministically.
4. **Stateful Multi-Turn Tracking:** Distinguishes confirmed vs inferred facts, preventing repeated actions.
5. **Governed Feedback Promotion:** Human edits enter an isolated store requiring offline validation.
6. **Strict Benchmark Isolation:** Zero overlap between retrieval index and golden evaluation set.
7. **Full Telemetry & Inspection:** 12-step decision traces explain exact diagnostic rationale.
8. **High Precision Automation:** Achieves 100% auto-handle precision with zero unsafe automated responses.

---

## Key Engineering Decisions

- **Rule of Specificity:** Root technical causes take precedence over triggering events during intent selection.
- **Symptom-Mandatory Synthesis:** Multi-case evidence aggregation strictly requires primary symptom consensus.
- **Zero-Tolerance Hardware Veto:** Thermal and physical hazard language completely halts generative workflows.
- **Isolated Feedback Store:** Specialist feedback is staged in a candidate store and evaluated offline before index promotion.

---

## Limitations

- **Vague Inquiries:** Queries lacking minimal device or symptom context require clarification turns before retrieval.
- **Complex Multi-System Combinations:** Intertwined multi-problem queries (e.g. simultaneous billing dispute and Apple ID security lockout) trigger safety escalation rather than automated resolution.
- **Enterprise MDM Protocols:** Specialized enterprise configuration profiles without historical resolution evidence are routed to human specialists.
- **LLM Rate-Limit Fallback:** When remote API quotas are exhausted, deterministic rule-based evaluation executes safely.

---

## Technical Report

For full architectural details, empirical evaluation methodology, and engineering analyses, consult the formal technical documentation:

- [PDF Technical Report](reports/SupportGraph_AI_Technical_Report.pdf)
- [Markdown Technical Report](reports/SupportGraph_AI_Technical_Report.md)
