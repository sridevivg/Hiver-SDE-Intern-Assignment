/**
 * SupportGraph AI — API Client Service
 *
 * Uses relative API URLs (/api/v1/...) via Vite proxy.
 * Communicates with FastAPI backend.
 */

import type {
  ApprovedEvidenceItem,
  ConversationState,
  DecisionLogRecord,
  DecisionTraceRecord,
  HumanResolution,
  LiveReviewCase,
  LiveReviewQueueResponse,
  RetrievedEvidenceCase,
  ReviewDecision,
  StartConversationResponse,
  SupportResolutionResult,
  TurnResponse,
} from '../types'

const BASE_URL = '' // Relative URL will be proxied by Vite

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let errorDetail = `HTTP ${res.status}: ${res.statusText}`
    try {
      const errJson = await res.json()
      if (errJson.detail) {
        errorDetail = typeof errJson.detail === 'string' ? errJson.detail : JSON.stringify(errJson.detail)
      }
    } catch {
      // ignore
    }
    throw new Error(errorDetail)
  }
  return res.json()
}

export const api = {
  // -------------------------------------------------------------------------
  // Support Resolution Pipeline (Primary)
  // -------------------------------------------------------------------------
  async resolveInquiry(
    customerMessage: string,
    topKEvidence = 3,
    caseId?: string,
    source = 'LIVE_SUPPORT'
  ): Promise<SupportResolutionResult> {
    const res = await fetch(`${BASE_URL}/api/v1/resolution/resolve`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        customer_message: customerMessage,
        top_k_evidence: topKEvidence,
        case_id: caseId,
        log_audit: true,
        source,
      }),
    })
    return handleResponse<SupportResolutionResult>(res)
  },

  async retrieveEvidence(
    queryText: string,
    queryIntent = '',
    topK = 3
  ): Promise<RetrievedEvidenceCase[]> {
    const res = await fetch(`${BASE_URL}/api/v1/resolution/retrieve-evidence`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query_text: queryText,
        query_intent: queryIntent,
        top_k: topK,
      }),
    })
    return handleResponse<RetrievedEvidenceCase[]>(res)
  },

  // -------------------------------------------------------------------------
  // Multi-Turn Conversations (Phase 11)
  // -------------------------------------------------------------------------
  async startConversation(
    customerId?: string,
    initialMessage?: string
  ): Promise<StartConversationResponse> {
    const res = await fetch(`${BASE_URL}/api/v1/conversations/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        customer_id: customerId,
        initial_message: initialMessage,
      }),
    })
    return handleResponse<StartConversationResponse>(res)
  },

  async sendConversationMessage(
    conversationId: string,
    message: string
  ): Promise<TurnResponse> {
    const res = await fetch(`${BASE_URL}/api/v1/conversations/${encodeURIComponent(conversationId)}/message`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message }),
    })
    return handleResponse<TurnResponse>(res)
  },

  async getConversationState(conversationId: string): Promise<ConversationState> {
    const res = await fetch(`${BASE_URL}/api/v1/conversations/${encodeURIComponent(conversationId)}/state`)
    return handleResponse<ConversationState>(res)
  },

  async getConversationHistory(conversationId: string): Promise<any[]> {
    const res = await fetch(`${BASE_URL}/api/v1/conversations/${encodeURIComponent(conversationId)}/history`)
    return handleResponse<any[]>(res)
  },

  async getConversationResolutionSummary(conversationId: string): Promise<Record<string, any>> {
    const res = await fetch(`${BASE_URL}/api/v1/conversations/${encodeURIComponent(conversationId)}/resolution-summary`)
    return handleResponse<Record<string, any>>(res)
  },

  async resolveConversation(
    conversationId: string,
    summary?: string
  ): Promise<Record<string, any>> {
    const res = await fetch(`${BASE_URL}/api/v1/conversations/${encodeURIComponent(conversationId)}/resolve`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ summary }),
    })
    return handleResponse<Record<string, any>>(res)
  },

  async getConversationAudit(conversationId: string): Promise<any[]> {
    const res = await fetch(`${BASE_URL}/api/v1/conversations/${encodeURIComponent(conversationId)}/audit`)
    return handleResponse<any[]>(res)
  },

  // -------------------------------------------------------------------------
  // Observability & Operations (Phase 14)
  // -------------------------------------------------------------------------
  async getObservabilitySummary(): Promise<Record<string, any>> {
    const res = await fetch(`${BASE_URL}/api/v1/observability/summary`)
    return handleResponse<Record<string, any>>(res)
  },

  async getObservabilityMetrics(): Promise<Record<string, any>> {
    const res = await fetch(`${BASE_URL}/api/v1/observability/metrics`)
    return handleResponse<Record<string, any>>(res)
  },

  async getDecisions(filter = '', limit = 50): Promise<{ decisions: DecisionLogRecord[]; count: number }> {
    const qs = filter ? `?decision=${encodeURIComponent(filter)}&limit=${limit}` : `?limit=${limit}`
    const res = await fetch(`${BASE_URL}/api/v1/observability/decisions${qs}`)
    return handleResponse<{ decisions: DecisionLogRecord[]; count: number }>(res)
  },

  async getDecisionTrace(caseId: string): Promise<DecisionTraceRecord> {
    const res = await fetch(`${BASE_URL}/api/v1/observability/decisions/${encodeURIComponent(caseId)}`)
    return handleResponse<DecisionTraceRecord>(res)
  },

  async getEvidenceStores(): Promise<Record<string, any>> {
    const res = await fetch(`${BASE_URL}/api/v1/observability/evidence`)
    return handleResponse<Record<string, any>>(res)
  },

  async getFeedbackMetrics(): Promise<Record<string, any>> {
    const res = await fetch(`${BASE_URL}/api/v1/observability/feedback`)
    return handleResponse<Record<string, any>>(res)
  },

  async getHealthDiagnostics(): Promise<Record<string, any>> {
    const res = await fetch(`${BASE_URL}/api/v1/observability/health`)
    return handleResponse<Record<string, any>>(res)
  },

  async getBasicHealth(): Promise<{ status: string; service: string }> {
    const res = await fetch(`${BASE_URL}/health`)
    return handleResponse<{ status: string; service: string }>(res)
  },

  // -------------------------------------------------------------------------
  // Feedback & Human-in-the-loop (Phase 13)
  // -------------------------------------------------------------------------
  async listCandidateResolutions(status?: string): Promise<HumanResolution[]> {
    const qs = status ? `?status=${encodeURIComponent(status)}` : ''
    const res = await fetch(`${BASE_URL}/api/v1/feedback/resolutions${qs}`)
    return handleResponse<HumanResolution[]>(res)
  },

  async validateResolution(resolutionId: string): Promise<ReviewDecision> {
    const res = await fetch(`${BASE_URL}/api/v1/feedback/resolutions/${encodeURIComponent(resolutionId)}/validation`)
    return handleResponse<ReviewDecision>(res)
  },

  async evaluateResolution(resolutionId: string, reviewerId = 'spec_01'): Promise<Record<string, any>> {
    const res = await fetch(`${BASE_URL}/api/v1/feedback/resolutions/${encodeURIComponent(resolutionId)}/evaluate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reviewer_id: reviewerId, notes: 'Automated evaluation trigger' }),
    })
    return handleResponse<Record<string, any>>(res)
  },

  async approveResolution(resolutionId: string, reviewerId = 'spec_01', notes = 'Approved for promotion'): Promise<HumanResolution> {
    const res = await fetch(`${BASE_URL}/api/v1/feedback/resolutions/${encodeURIComponent(resolutionId)}/approve`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reviewer_id: reviewerId, notes }),
    })
    return handleResponse<HumanResolution>(res)
  },

  async promoteResolution(resolutionId: string, reviewerId = 'lead_01'): Promise<ApprovedEvidenceItem> {
    const res = await fetch(`${BASE_URL}/api/v1/feedback/resolutions/${encodeURIComponent(resolutionId)}/promote`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reviewer_id: reviewerId, version: '1.0.0' }),
    })
    return handleResponse<ApprovedEvidenceItem>(res)
  },

  async listApprovedEvidence(problemFamily?: string): Promise<ApprovedEvidenceItem[]> {
    const qs = problemFamily ? `?problem_family=${encodeURIComponent(problemFamily)}` : ''
    const res = await fetch(`${BASE_URL}/api/v1/feedback/evidence${qs}`)
    return handleResponse<ApprovedEvidenceItem[]>(res)
  },

  // -------------------------------------------------------------------------
  // Live Human Review Queue (Production Agent Queue)
  // -------------------------------------------------------------------------
  async getHumanReviewQueue(status = 'active', limit = 50): Promise<LiveReviewQueueResponse> {
    const qs = `?status=${encodeURIComponent(status)}&limit=${limit}`
    const res = await fetch(`${BASE_URL}/api/v1/human-review/queue${qs}`)
    return handleResponse<LiveReviewQueueResponse>(res)
  },

  async getHumanReviewCase(caseId: string): Promise<LiveReviewCase> {
    const res = await fetch(`${BASE_URL}/api/v1/human-review/${encodeURIComponent(caseId)}`)
    return handleResponse<LiveReviewCase>(res)
  },

  async approveHumanReviewCase(
    caseId: string,
    reviewerId = 'human_specialist_1',
    notes = 'Approved by human reviewer'
  ): Promise<LiveReviewCase> {
    const res = await fetch(`${BASE_URL}/api/v1/human-review/${encodeURIComponent(caseId)}/approve`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reviewer_id: reviewerId, notes }),
    })
    return handleResponse<LiveReviewCase>(res)
  },

  async editHumanReviewCase(
    caseId: string,
    editedResponse: string,
    reviewerId = 'human_specialist_1',
    notes = 'Edited by human reviewer'
  ): Promise<LiveReviewCase> {
    const res = await fetch(`${BASE_URL}/api/v1/human-review/${encodeURIComponent(caseId)}/edit`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        reviewer_id: reviewerId,
        edited_response: editedResponse,
        notes,
      }),
    })
    return handleResponse<LiveReviewCase>(res)
  },

  async escalateHumanReviewCase(
    caseId: string,
    reviewerId = 'human_specialist_1',
    notes = 'Escalated to Tier 2 specialist'
  ): Promise<LiveReviewCase> {
    const res = await fetch(`${BASE_URL}/api/v1/human-review/${encodeURIComponent(caseId)}/escalate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reviewer_id: reviewerId, notes }),
    })
    return handleResponse<LiveReviewCase>(res)
  },
}
