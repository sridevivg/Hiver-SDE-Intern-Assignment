/**
 * SupportGraph AI — Frontend TypeScript Type Definitions
 * Directly aligned with FastAPI / Pydantic Backend Schemas.
 */

export type RoutingDecisionType = 'AUTO_HANDLE' | 'ESCALATE_TO_HUMAN'

export type AmbiguityType =
  | 'CLEAR_INTENT'
  | 'GENUINE_AMBIGUITY'
  | 'MULTI_PROBLEM_COMPLEXITY'
  | 'INSUFFICIENT_INFORMATION'
  | 'EVIDENCE_LIMITED'
  | 'SAFETY_RISK'
  | 'UNKNOWN'

export type EvidenceMatchTier =
  | 'DIRECT_PROBLEM_MATCH'
  | 'RELATED_SYMPTOM'
  | 'RELATED_CONTEXT'
  | 'WEAK_SEMANTIC_MATCH'

export type EvidenceVerdict =
  | 'STRONG_EVIDENCE'
  | 'MODERATE_EVIDENCE'
  | 'WEAK_EVIDENCE'
  | 'CONFLICTING_EVIDENCE'
  | 'INSUFFICIENT_EVIDENCE'

export type VerificationStatus =
  | 'PASSED'
  | 'FAILED'
  | 'VERIFICATION_VETO'
  | 'UNVERIFIED'

export interface CustomerProblemProfile {
  device_type: string
  os_or_service: string
  primary_symptom: string
  causal_trigger: string
  problem_family: string
  is_sufficient_for_resolution: boolean
  extracted_entities: Record<string, string>
  device_model?: string
  os_version?: string
}

export interface IntentPrediction {
  intent: string
  confidence: number
  rank?: number
}

export interface AmbiguityAnalysisResult {
  ambiguity_type: AmbiguityType
  is_ambiguous: boolean
  reason: string
  top_candidates: IntentPrediction[]
  clarification_question?: string | null
}

export interface RetrievedEvidenceCase {
  case_id?: string
  tweet_id?: string
  customer_query?: string
  brand_response?: string
  historical_customer_message?: string
  historical_brand_response?: string
  relevance_score?: number
  similarity_score?: number
  operational_similarity?: number
  match_tier: EvidenceMatchTier
  historical_intent?: string
  historical_problem_family?: string
  problem_family?: string
  retrieval_explanation?: string
  provenance?: string
  created_at?: string
}

export interface EvidenceValidationResult {
  is_valid: boolean
  evidence_verdict: EvidenceVerdict
  coverage_score: number
  agreement_score: number
  consistency_score: number
  direct_matches_count: number
  reasons: string[]
  unsupported_elements: string[]
}

export interface CompositeEvidencePackage {
  verdict: string
  overall_score: number
  covered_dimensions: string[]
  missing_dimensions: string[]
  conflicts_detected: boolean
  synthesis_summary: string
}

export interface ClaimVerificationItem {
  claim: string
  is_supported: boolean
  corroborating_case_ids: string[]
  verdict: string
}

export interface ResponseGroundingResult {
  is_grounded: boolean
  verification_status: VerificationStatus
  evidence_verdict: EvidenceVerdict
  grounding_score: number
  unsupported_claims: string[]
  verified_claims: ClaimVerificationItem[]
  verification_notes: string
  veto_reason?: string | null
}

export interface DecisionGateResult {
  can_auto_handle: boolean
  gate_passed: boolean
  veto_triggered: boolean
  veto_reason?: string | null
  confidence_passed: boolean
  clarity_passed: boolean
  evidence_passed: boolean
}

export interface DecisionExplanation {
  routing_decision: RoutingDecisionType
  primary_reason: string
  outcome_code: string
  confidence_score: number
  blocking_factors: string[]
  positive_factors: string[]
  recommended_human_action?: string
  verdict_summary?: string
}

export interface HumanEscalationPackage {
  customer_message: string
  problem_profile: CustomerProblemProfile
  ambiguity_type: AmbiguityType
  ambiguity_reason: string
  top_candidates: IntentPrediction[]
  related_evidence: RetrievedEvidenceCase[]
  evidence_verdict: EvidenceVerdict
  candidate_resolution_approaches: Record<string, string>
  decision_checklist: Record<string, boolean>
  why_not_auto_handled: string[]
  grounding_result?: ResponseGroundingResult | null
  recommended_human_action: string
  composite_evidence_verdict?: string | null
  composite_evidence_summary?: string | null
  covered_evidence_dimensions: string[]
  missing_evidence_dimensions: string[]
  conflicting_evidence_pairs: string[]
}

export interface SupportResolutionResult {
  routing_decision: RoutingDecisionType
  primary_intent: string
  confidence: number
  problem_summary: string
  ambiguity_analysis: AmbiguityAnalysisResult
  gate_result?: DecisionGateResult | null
  evidence_validation?: EvidenceValidationResult | null
  composite_evidence?: CompositeEvidencePackage | null
  evidence_cases: RetrievedEvidenceCase[]
  resolution_strategy: string
  grounded_response?: string | null
  response_grounding?: ResponseGroundingResult | null
  escalation_package?: HumanEscalationPackage | null
  explanation: string
  outcome?: string | null
  decision_explanation?: DecisionExplanation | null
}

// Multi-Turn Conversation Types
export type ConversationStatus = 'ACTIVE' | 'AWAITING_CUSTOMER' | 'RESOLVED' | 'ESCALATED' | 'ABANDONED'

export interface ConversationTurn {
  turn_index: number
  role: 'customer' | 'agent' | 'system'
  content: string
  timestamp: string
  role_type?: string
  intent?: string
  metadata?: Record<string, any>
}

export interface StartConversationResponse {
  conversation_id: string
  status: ConversationStatus
  stage: string
  problem_family?: string | null
  initial_agent_response?: string | null
  confirmed_facts: Record<string, any>
  turns_count: number
}

export interface TurnResponse {
  conversation_id: string
  turn_index: number
  agent_response: string
  role_type: string
  status: ConversationStatus
  stage: string
  confirmed_facts: Record<string, any>
  current_action?: Record<string, any> | null
  escalation_package?: Record<string, any> | null
}

export interface ConversationState {
  conversation_id: string
  customer_id?: string | null
  status: ConversationStatus
  stage: string
  problem_family?: string | null
  turns: ConversationTurn[]
  confirmed_facts: Record<string, any>
  created_at: string
  updated_at: string
  resolution_summary?: string | null
  escalation_package?: Record<string, any> | null
}

// Observability Types
export interface DecisionLogRecord {
  case_id: string
  timestamp: string
  routing_decision: RoutingDecisionType
  outcome: string
  problem_family: string
  total_latency_ms: number
  is_demo: boolean
}

export interface DecisionTraceStep {
  step: string
  status: string
  latency_ms: number
  data: Record<string, unknown>
}

export interface DecisionTraceRecord {
  case_id: string
  timestamp: string
  is_demo: boolean
  steps: DecisionTraceStep[]
  total_latency_ms: number
}

// Feedback & Human-in-the-Loop Types
export interface HumanResolution {
  resolution_id: string
  case_id?: string
  problem_family: string
  customer_query: string
  resolution_text: string
  troubleshooting_steps: string[]
  specialist_id: string
  status: string
  quality_score?: number
  created_at: string
}

export interface ReviewDecision {
  decision: 'APPROVE' | 'REVISE' | 'REJECT'
  passed_validation_gate: boolean
  quality_score: number
  validation_checks: Record<string, boolean>
  blocking_reasons: string[]
  warning_notes: string[]
}

export interface ApprovedEvidenceItem {
  evidence_id: string
  problem_family: string
  symptom_pattern: string
  resolution_text: string
  troubleshooting_steps: string[]
  quality_score: number
  version: string
  promoted_at: string
  authorizing_reviewer: string
}

// Live Human Review Queue Types
export type CaseSource = 'LIVE_SUPPORT' | 'DEMO' | 'EVALUATION' | 'BENCHMARK' | 'SYNTHETIC' | 'TEST'
export type ReviewCaseStatus = 'NEW' | 'UNDER_REVIEW' | 'APPROVED' | 'EDITED' | 'ESCALATED' | 'RESOLVED'

export interface LiveReviewCase {
  case_id: string
  created_at: string
  updated_at: string
  source: CaseSource
  status: ReviewCaseStatus
  customer_query: string
  conversation_id?: string | null
  priority?: 'NORMAL' | 'URGENT' | string
  decision: string
  outcome?: string | null
  escalation_reason: string
  decision_explanation?: Record<string, any> | null
  ai_suggested_response?: string | null
  problem_understanding?: Record<string, any> | null
  intent?: string | null
  problem_family?: string | null
  evidence_summary?: string | null
  related_historical_cases: RetrievedEvidenceCase[]
  reviewer_id?: string | null
  reviewer_notes?: string | null
  edited_response?: string | null
  completed_at?: string | null
}

export interface LiveReviewQueueResponse {
  cases: LiveReviewCase[]
  count: number
}
