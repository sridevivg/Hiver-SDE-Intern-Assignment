# Phase 7 Evidence-Aware Support Resolution Benchmark Report

## 1. Overall Classification & Disambiguation Performance
| Metric | Value | Description |
| :--- | :--- | :--- |
| **Evaluated Benchmark Records** | `77` | Completed human-reviewed ground truth cases |
| **Primary Intent Accuracy** | `55.8%` | Percentage where primary intent matches ground truth |
| **Top-2 Candidate Coverage** | `61.0%` | Percentage where ground truth is in Top-2 candidates |
| **Top-3 Candidate Coverage** | `80.5%` | Percentage where ground truth is in Top-3 candidates |
| **Macro F1 Score** | `0.4936` | Unweighted mean F1 across operational intents |

## 2. Evidence-Aware Routing & Escalation Safety
| Metric | Value | Target / Description |
| :--- | :--- | :--- |
| **Auto-Handle Count** | `65` | Total cases routed to automated brand resolution |
| **Auto-Handle Rate** | `84.4%` | Workload safely automated without human review |
| **Auto-Handle Accuracy** | `56.9%` | Precision of auto-handled support responses |
| **Auto-Handle Error Rate** | `43.1%` | Erroneous auto-handled decisions |
| **Escalation Count** | `12` | Cases packaged for human agent review |
| **Escalation Rate** | `15.6%` | Proportion redirected to human queue |
| **Escalation Safety Rate** | `17.6%` | Percentage of ambiguous/incorrect cases safely escalated |
| **Human Assistance (Top-2)** | `66.7%` | True intent in escalation package's Top-2 candidates |

## 3. Ambiguity Type Distribution
| Ambiguity Type | Count | Percentage |
| :--- | :--- | :--- |
| `CLEAR_PRIMARY` | `64` | `83.1%` |
| `GENUINE_AMBIGUITY` | `6` | `7.8%` |
| `CAUSE_VS_SYMPTOM` | `4` | `5.2%` |
| `MULTI_SYMPTOM` | `3` | `3.9%` |

## 4. Historical Evidence Match Tiers
| Evidence Match Tier | Count | Percentage |
| :--- | :--- | :--- |
| `WEAK_SEMANTIC_MATCH` | `127` | `55.0%` |
| `DIRECT_PROBLEM_MATCH` | `60` | `26.0%` |
| `RELATED_SYMPTOM` | `22` | `9.5%` |
| `RELATED_CONTEXT` | `22` | `9.5%` |
