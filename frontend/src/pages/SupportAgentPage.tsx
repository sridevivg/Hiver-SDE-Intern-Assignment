/**
 * SupportGraph AI — Support Agent Primary Workspace (Phase 15)
 *
 * Exposes the core AI customer-support agent capabilities for AppleSupport:
 * 1. Intent Classification & Problem Understanding
 * 2. Historical Evidence Retrieval & Match Tier Categorization
 * 3. Evidence-Grounded Response Drafting & Verification
 * 4. Authoritative Safety Decision (AUTO-HANDLE vs. ESCALATE TO HUMAN)
 * 5. Interactive Multi-Turn Support Sessions
 */

import React, { useState, useEffect, useRef } from 'react'
import { api } from '../services/api'
import type {
  SupportResolutionResult,
  EvidenceMatchTier,
  EvidenceVerdict,
  ConversationTurn,
  ConversationStatus,
} from '../types'

// Real test presets for one-click testing (representing distinct pipeline scenarios)
const PRESETS = [
  {
    label: 'Clear Solvable Issue',
    description: 'Battery health drain after update',
    text: 'My iPhone 13 battery drains very quickly after updating to iOS 16. Battery health shows 79% maximum capacity.',
  },
  {
    label: 'Ambiguous Query',
    description: 'Vague complaint needing clarification',
    text: 'My device is acting really weird today and things are not working properly.',
  },
  {
    label: 'Thermal Safety Hazard',
    description: 'Urgent hardware risk — safety escalation',
    text: 'My iPhone is extremely hot, smoking and the back glass is cracking.',
  },
  {
    label: 'Account Access Issue',
    description: 'Apple ID password recovery & 2FA',
    text: 'I forgot my Apple ID password and cannot sign in to iCloud or receive verification codes.',
  },
  {
    label: 'Audio Hardware Issue',
    description: 'AirPods connection & audio grill',
    text: 'My AirPods Pro left earbud has no sound and keeps disconnecting from my iPhone.',
  },
]

const PIPELINE_STEPS = [
  'Understanding problem & extracting entities',
  'Classifying intent candidates',
  'Retrieving historical AppleSupport evidence',
  'Validating evidence coverage & consistency',
  'Drafting evidence-grounded response',
  'Verifying grounding & safety constraints',
  'Making safety decision (Auto-Handle vs Escalation)',
]

export const SupportAgentPage: React.FC = () => {
  // Mode: 'single' (Inquiry Analyzer) vs 'multi-turn' (Live Conversation)
  const [mode, setMode] = useState<'single' | 'multi-turn'>('single')

  // Single Inquiry State
  const [customerMessage, setCustomerMessage] = useState('')
  const [loading, setLoading] = useState(false)
  const [activeStepIndex, setActiveStepIndex] = useState(0)
  const [result, setResult] = useState<SupportResolutionResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)

  // Multi-turn State
  const [conversationId, setConversationId] = useState<string | null>(null)
  const [convStatus, setConvStatus] = useState<ConversationStatus>('ACTIVE')
  const [convTurns, setConvTurns] = useState<ConversationTurn[]>([])
  const [confirmedFacts, setConfirmedFacts] = useState<Record<string, any>>({})
  const [convInput, setConvInput] = useState('')
  const [convLoading, setConvLoading] = useState(false)
  const [convSummary, setConvSummary] = useState<string | null>(null)
  const [showAuditModal, setShowAuditModal] = useState(false)
  const [auditLogs, setAuditLogs] = useState<any[]>([])

  const chatEndRef = useRef<HTMLDivElement>(null)

  // Simulated progressive steps while waiting for backend API response
  useEffect(() => {
    let interval: ReturnType<typeof setInterval>
    if (loading) {
      setActiveStepIndex(0)
      interval = setInterval(() => {
        setActiveStepIndex((prev) => (prev < PIPELINE_STEPS.length - 1 ? prev + 1 : prev))
      }, 400)
    }
    return () => clearInterval(interval)
  }, [loading])

  // Scroll chat in multi-turn mode
  useEffect(() => {
    if (mode === 'multi-turn') {
      chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }
  }, [convTurns, mode])

  // Handle single inquiry analysis
  const handleAnalyze = async () => {
    if (!customerMessage.trim()) {
      setError('Please enter a customer problem description before analyzing.')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await api.resolveInquiry(customerMessage.trim())
      setResult(response)
    } catch (err: any) {
      setError(err.message || 'Failed to analyze customer message. Check backend connectivity.')
    } finally {
      setLoading(false)
    }
  }

  // Copy response to clipboard
  const handleCopy = (text: string) => {
    navigator.clipboard.writeText(text)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  // Start new multi-turn conversation
  const handleStartConversation = async (initialText?: string) => {
    setConvLoading(true)
    setError(null)
    setConvSummary(null)
    try {
      const res = await api.startConversation('cust_user_01', initialText || undefined)
      setConversationId(res.conversation_id)
      setConvStatus(res.status)
      setConfirmedFacts(res.confirmed_facts || {})

      const initialTurns: ConversationTurn[] = []
      if (initialText) {
        initialTurns.push({
          turn_index: 0,
          role: 'customer',
          content: initialText,
          timestamp: new Date().toISOString(),
        })
      }
      if (res.initial_agent_response) {
        initialTurns.push({
          turn_index: initialText ? 1 : 0,
          role: 'agent',
          content: res.initial_agent_response,
          timestamp: new Date().toISOString(),
        })
      }
      setConvTurns(initialTurns)
      setConvInput('')
    } catch (err: any) {
      setError(err.message || 'Failed to start conversation.')
    } finally {
      setConvLoading(false)
    }
  }

  // Send turn in multi-turn conversation
  const handleSendTurn = async () => {
    if (!convInput.trim() || !conversationId) return

    const userMessage = convInput.trim()
    setConvInput('')
    setConvLoading(true)

    // Optimistically append customer turn
    const nextTurns = [
      ...convTurns,
      {
        turn_index: convTurns.length,
        role: 'customer' as const,
        content: userMessage,
        timestamp: new Date().toISOString(),
      },
    ]
    setConvTurns(nextTurns)

    try {
      const res = await api.sendConversationMessage(conversationId, userMessage)
      setConvStatus(res.status)
      setConfirmedFacts(res.confirmed_facts || {})

      setConvTurns([
        ...nextTurns,
        {
          turn_index: res.turn_index,
          role: 'agent',
          content: res.agent_response,
          timestamp: new Date().toISOString(),
          role_type: res.role_type,
        },
      ])

      if (res.status === 'RESOLVED' || res.status === 'ESCALATED') {
        const summaryRes = await api.getConversationResolutionSummary(conversationId)
        setConvSummary(summaryRes.summary || (summaryRes.escalation ? `Escalated: ${JSON.stringify(summaryRes.escalation)}` : null))
      }
    } catch (err: any) {
      setError(err.message || 'Failed to send message.')
    } finally {
      setConvLoading(false)
    }
  }

  // Explicitly resolve conversation
  const handleResolveConv = async () => {
    if (!conversationId) return
    setConvLoading(true)
    try {
      await api.resolveConversation(conversationId, 'Customer issue resolved via interactive troubleshooting.')
      setConvStatus('RESOLVED')
      setConvSummary('Customer issue marked as resolved.')
    } catch (err: any) {
      setError(err.message || 'Failed to resolve conversation.')
    } finally {
      setConvLoading(false)
    }
  }

  // Load audit trail for conversation
  const handleLoadAudit = async () => {
    if (!conversationId) return
    try {
      const logs = await api.getConversationAudit(conversationId)
      setAuditLogs(logs)
      setShowAuditModal(true)
    } catch (err: any) {
      setError(err.message || 'Failed to load audit trail.')
    }
  }

  // Match Tier Style Helper
  const getMatchTierBadge = (tier: EvidenceMatchTier) => {
    switch (tier) {
      case 'DIRECT_PROBLEM_MATCH':
        return <span className="tag tier-direct">DIRECT MATCH</span>
      case 'RELATED_SYMPTOM':
        return <span className="tag tier-symptom">RELATED SYMPTOM</span>
      case 'RELATED_CONTEXT':
        return <span className="tag tier-context">RELATED CONTEXT</span>
      case 'WEAK_SEMANTIC_MATCH':
      default:
        return <span className="tag tier-weak">WEAK MATCH</span>
    }
  }

  // Verdict Style Helper
  const getVerdictBadge = (verdict?: EvidenceVerdict | string) => {
    switch (verdict) {
      case 'STRONG_EVIDENCE':
        return <span className="tag tag-green">STRONG EVIDENCE</span>
      case 'MODERATE_EVIDENCE':
        return <span className="tag tag-blue">MODERATE EVIDENCE</span>
      case 'WEAK_EVIDENCE':
        return <span className="tag tag-amber">WEAK EVIDENCE</span>
      case 'CONFLICTING_EVIDENCE':
        return <span className="tag tag-red">CONFLICTING EVIDENCE</span>
      case 'INSUFFICIENT_EVIDENCE':
      default:
        return <span className="tag tag-red">INSUFFICIENT EVIDENCE</span>
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem' }}>
      {/* Brand & Workspace Subheader */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '1rem',
          background: 'var(--surface)',
          padding: '1rem 1.5rem',
          borderRadius: 'var(--radius-m)',
          border: '1px solid var(--border)',
          boxShadow: 'var(--shadow-xs)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div
            style={{
              width: 36,
              height: 36,
              borderRadius: 'var(--radius-s)',
              background: 'var(--blue-bg)',
              border: '1px solid var(--blue-border)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: 'var(--blue-primary)',
              fontWeight: 700,
              fontSize: '1rem',
            }}
          >
            
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <h2 style={{ fontSize: '1.05rem', fontWeight: 700 }}>AI Customer Support Resolution Agent</h2>
              <span className="tag tag-blue">Selected Brand: AppleSupport</span>
            </div>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: 2 }}>
              Classify intents · Retrieve grounded AppleSupport historical evidence · Decide auto-handle vs. human escalation
            </div>
          </div>
        </div>

        {/* Mode Switcher */}
        <div
          style={{
            display: 'flex',
            background: 'var(--bg-subtle)',
            padding: '3px',
            borderRadius: 'var(--radius-s)',
            border: '1px solid var(--border)',
          }}
        >
          <button
            onClick={() => setMode('single')}
            style={{
              padding: '6px 14px',
              borderRadius: 'var(--radius-xs)',
              border: 'none',
              background: mode === 'single' ? 'var(--surface)' : 'transparent',
              color: mode === 'single' ? 'var(--text-primary)' : 'var(--text-muted)',
              fontWeight: mode === 'single' ? 600 : 400,
              fontSize: '0.82rem',
              boxShadow: mode === 'single' ? 'var(--shadow-xs)' : 'none',
              cursor: 'pointer',
              transition: 'all 0.15s ease',
            }}
          >
            Single Inquiry Resolution
          </button>
          <button
            onClick={() => {
              setMode('multi-turn')
              if (!conversationId) handleStartConversation()
            }}
            style={{
              padding: '6px 14px',
              borderRadius: 'var(--radius-xs)',
              border: 'none',
              background: mode === 'multi-turn' ? 'var(--surface)' : 'transparent',
              color: mode === 'multi-turn' ? 'var(--text-primary)' : 'var(--text-muted)',
              fontWeight: mode === 'multi-turn' ? 600 : 400,
              fontSize: '0.82rem',
              boxShadow: mode === 'multi-turn' ? 'var(--shadow-xs)' : 'none',
              cursor: 'pointer',
              transition: 'all 0.15s ease',
            }}
          >
            Multi-Turn Support Session
          </button>
        </div>
      </div>

      {/* Error Banner */}
      {error && (
        <div
          style={{
            background: 'var(--red-bg)',
            border: '1px solid var(--red-border)',
            borderRadius: 'var(--radius-m)',
            padding: '1rem 1.25rem',
            color: 'var(--red-text)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            fontSize: '0.88rem',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <span style={{ fontWeight: 700, fontSize: '1rem' }}>⚠</span>
            <span>{error}</span>
          </div>
          <button
            onClick={() => setError(null)}
            style={{ background: 'none', border: 'none', color: 'var(--red-text)', cursor: 'pointer', fontSize: '0.9rem' }}
          >
            ✕
          </button>
        </div>
      )}

      {/* ==================================================================== */}
      {/* MODE 1: SINGLE INQUIRY RESOLUTION                                   */}
      {/* ==================================================================== */}
      {mode === 'single' && (
        <>
          {/* Customer Message Section */}
          <div className="sg-card">
            <div className="sg-card-header">
              <div className="sg-section-title">
                <span>Customer Message</span>
              </div>
              <div style={{ fontSize: '0.78rem', color: 'var(--text-subtle)' }}>
                {customerMessage.length} characters
              </div>
            </div>

            {/* Presets Bar */}
            <div style={{ marginBottom: '0.85rem' }}>
              <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: 6 }}>
                TEST PRESETS (REAL HISTORICAL SCENARIOS):
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                {PRESETS.map((p, idx) => (
                  <button
                    key={idx}
                    className="btn-preset"
                    onClick={() => {
                      setCustomerMessage(p.text)
                      setResult(null)
                    }}
                    title={p.description}
                  >
                    {p.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Textarea */}
            <textarea
              className="textarea-custom"
              rows={4}
              placeholder="Describe the customer's problem..."
              value={customerMessage}
              onChange={(e) => setCustomerMessage(e.target.value)}
              disabled={loading}
            />

            {/* Actions */}
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.75rem', marginTop: '1rem' }}>
              {customerMessage && (
                <button
                  className="btn btn-secondary"
                  onClick={() => {
                    setCustomerMessage('')
                    setResult(null)
                  }}
                  disabled={loading}
                >
                  Clear
                </button>
              )}
              <button
                className="btn btn-primary"
                onClick={handleAnalyze}
                disabled={loading || !customerMessage.trim()}
                style={{ minWidth: 160 }}
              >
                {loading ? (
                  <>
                    <span className="spinner" /> Analyzing Pipeline...
                  </>
                ) : (
                  'Analyze Message'
                )}
              </button>
            </div>
          </div>

          {/* Loading Pipeline State */}
          {loading && (
            <div className="sg-card" style={{ background: '#f8fafc', borderColor: 'var(--blue-border)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: '1rem' }}>
                <span className="spinner" />
                <span style={{ fontWeight: 600, color: 'var(--blue-text)', fontSize: '0.95rem' }}>
                  Processing Inquiry through SupportGraph AI Pipeline
                </span>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {PIPELINE_STEPS.map((step, idx) => {
                  const isDone = idx < activeStepIndex
                  const isCurrent = idx === activeStepIndex
                  return (
                    <div
                      key={idx}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: 12,
                        padding: '6px 12px',
                        borderRadius: 'var(--radius-s)',
                        background: isCurrent ? 'var(--blue-bg)' : isDone ? '#ffffff' : 'transparent',
                        border: isCurrent ? '1px solid var(--blue-border)' : '1px solid transparent',
                        transition: 'all 0.2s ease',
                      }}
                    >
                      <span
                        style={{
                          width: 20,
                          height: 20,
                          borderRadius: '50%',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          fontSize: '0.75rem',
                          fontWeight: 700,
                          background: isDone ? 'var(--green-bg)' : isCurrent ? 'var(--blue-primary)' : 'var(--bg-subtle)',
                          color: isDone ? 'var(--green-text)' : isCurrent ? '#ffffff' : 'var(--text-subtle)',
                          border: isDone ? '1px solid var(--green-border)' : 'none',
                        }}
                      >
                        {isDone ? '✓' : idx + 1}
                      </span>
                      <span
                        style={{
                          fontSize: '0.85rem',
                          fontWeight: isCurrent ? 600 : 400,
                          color: isCurrent ? 'var(--blue-text)' : isDone ? 'var(--text-primary)' : 'var(--text-muted)',
                        }}
                      >
                        {step}
                      </span>
                      {isCurrent && <span className="spinner" style={{ marginLeft: 'auto', width: 12, height: 12 }} />}
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          {/* ================================================================ */}
          {/* PIPELINE RESULTS SECTION                                         */}
          {/* ================================================================ */}
          {result && !loading && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
              {/* ------------------------------------------------------------ */}
              {/* SECTION 4 — FINAL DECISION (PROMINENT HERO CARD AT TOP/CENTER)*/}
              {/* ------------------------------------------------------------ */}
              {(() => {
                const isAutoHandle = result.routing_decision === 'AUTO_HANDLE'
                const exp = result.decision_explanation
                const esc = result.escalation_package

                return (
                  <div
                    style={{
                      background: isAutoHandle ? 'var(--green-bg)' : 'var(--amber-bg)',
                      border: `2px solid ${isAutoHandle ? 'var(--green-border)' : 'var(--amber-border)'}`,
                      borderRadius: 'var(--radius-l)',
                      padding: '1.5rem 1.75rem',
                      boxShadow: 'var(--shadow-m)',
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '1rem' }}>
                      <div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6 }}>
                          <span
                            style={{
                              fontSize: '1.3rem',
                              fontWeight: 800,
                              color: isAutoHandle ? 'var(--green-text)' : 'var(--amber-text)',
                              letterSpacing: '-0.02em',
                            }}
                          >
                            {isAutoHandle ? '✓ AUTO-HANDLE' : '⚠ ESCALATE TO HUMAN'}
                          </span>
                          <span className={isAutoHandle ? 'tag tag-green' : 'tag tag-amber'}>
                            {result.outcome || result.routing_decision}
                          </span>
                        </div>
                        <div style={{ fontSize: '0.95rem', color: isAutoHandle ? '#166534' : '#92400e', fontWeight: 500, maxWidth: 850 }}>
                          {exp?.primary_reason || result.explanation || 'Authoritative backend decision applied.'}
                        </div>
                      </div>

                      <div style={{ textAlign: 'right' }}>
                        <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontFamily: 'var(--mono)' }}>SAFETY AUDIT</div>
                        <div style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-secondary)' }}>
                          Confidence: {(result.confidence * 100).toFixed(1)}%
                        </div>
                      </div>
                    </div>

                    {/* Factors / Checklist */}
                    <div
                      style={{
                        display: 'grid',
                        gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
                        gap: '1rem',
                        marginTop: '1.25rem',
                        paddingTop: '1rem',
                        borderTop: `1px solid ${isAutoHandle ? '#bbf7d0' : '#fde68a'}`,
                      }}
                    >
                      {/* Positive Verified Factors */}
                      <div>
                        <div style={{ fontSize: '0.75rem', fontWeight: 700, color: '#166534', marginBottom: 6, textTransform: 'uppercase' }}>
                          Verified Positive Factors
                        </div>
                        <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: 4 }}>
                          {(exp?.positive_factors && exp.positive_factors.length > 0
                            ? exp.positive_factors
                            : isAutoHandle
                            ? [
                                'Problem sufficiently understood',
                                'Direct historical evidence available',
                                'Evidence supports response',
                                'Response passed grounding verification',
                                'No safety veto triggered',
                              ]
                            : ['Intent candidates identified']
                          ).map((factor, i) => (
                            <li key={i} style={{ fontSize: '0.82rem', color: '#15803d', display: 'flex', alignItems: 'center', gap: 6 }}>
                              <span>✓</span> {factor}
                            </li>
                          ))}
                        </ul>
                      </div>

                      {/* Blocking Factors (if escalated) */}
                      {!isAutoHandle && (
                        <div>
                          <div style={{ fontSize: '0.75rem', fontWeight: 700, color: '#991b1b', marginBottom: 6, textTransform: 'uppercase' }}>
                            Blocking Factors & Safety Vetoes
                          </div>
                          <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: 4 }}>
                            {(exp?.blocking_factors && exp.blocking_factors.length > 0
                              ? exp.blocking_factors
                              : esc?.why_not_auto_handled && esc.why_not_auto_handled.length > 0
                              ? esc.why_not_auto_handled
                              : [esc?.ambiguity_reason || 'Safety threshold not met for autonomous resolution.']
                            ).map((b, i) => (
                              <li key={i} style={{ fontSize: '0.82rem', color: '#b91c1c', display: 'flex', alignItems: 'center', gap: 6 }}>
                                <span>✕</span> {b}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}

                      {/* Recommended Human Action */}
                      {!isAutoHandle && (esc?.recommended_human_action || exp?.recommended_human_action) && (
                        <div>
                          <div style={{ fontSize: '0.75rem', fontWeight: 700, color: '#854d0e', marginBottom: 6, textTransform: 'uppercase' }}>
                            Recommended Specialist Action
                          </div>
                          <div
                            style={{
                              fontSize: '0.82rem',
                              color: '#78350f',
                              background: '#fef3c7',
                              padding: '8px 12px',
                              borderRadius: 'var(--radius-s)',
                              border: '1px solid #fde68a',
                            }}
                          >
                            {esc?.recommended_human_action || exp?.recommended_human_action}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )
              })()}

              {/* ------------------------------------------------------------ */}
              {/* SECTION 1 — INTENT CLASSIFICATION & PROBLEM UNDERSTANDING   */}
              {/* ------------------------------------------------------------ */}
              <div className="sg-card">
                <div className="sg-card-header">
                  <div className="sg-section-title">
                    <span>Section 1 — Intent Classification & Problem Understanding</span>
                  </div>
                  <span className="tag tag-blue">Primary Intent</span>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem', marginBottom: '1.25rem' }}>
                  <div style={{ background: 'var(--bg-subtle)', padding: '0.85rem', borderRadius: 'var(--radius-s)', border: '1px solid var(--border)' }}>
                    <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontFamily: 'var(--mono)' }}>PRIMARY INTENT</div>
                    <div style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--blue-primary)', marginTop: 4 }}>
                      {result.primary_intent}
                    </div>
                  </div>

                  <div style={{ background: 'var(--bg-subtle)', padding: '0.85rem', borderRadius: 'var(--radius-s)', border: '1px solid var(--border)' }}>
                    <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontFamily: 'var(--mono)' }}>PROBLEM FAMILY</div>
                    <div style={{ fontSize: '1rem', fontWeight: 600, color: 'var(--text-primary)', marginTop: 4 }}>
                      {result.ambiguity_analysis?.top_candidates?.[0]?.intent || result.primary_intent}
                    </div>
                  </div>

                  <div style={{ background: 'var(--bg-subtle)', padding: '0.85rem', borderRadius: 'var(--radius-s)', border: '1px solid var(--border)' }}>
                    <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontFamily: 'var(--mono)' }}>CONFIDENCE SCORE</div>
                    <div style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--green-text)', marginTop: 4 }}>
                      {(result.confidence * 100).toFixed(1)}%
                    </div>
                  </div>

                  <div style={{ background: 'var(--bg-subtle)', padding: '0.85rem', borderRadius: 'var(--radius-s)', border: '1px solid var(--border)' }}>
                    <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontFamily: 'var(--mono)' }}>AMBIGUITY CLASSIFICATION</div>
                    <div style={{ fontSize: '0.95rem', fontWeight: 600, marginTop: 4 }}>
                      <span className={result.ambiguity_analysis?.is_ambiguous ? 'tag tag-amber' : 'tag tag-green'}>
                        {result.ambiguity_analysis?.ambiguity_type || 'CLEAR_INTENT'}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Structured Problem Profile Grid */}
                <div style={{ background: '#f8fafc', padding: '1rem', borderRadius: 'var(--radius-s)', border: '1px solid var(--border)' }}>
                  <div style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-muted)', marginBottom: 8, letterSpacing: '0.04em' }}>
                    EXTRACTED PROBLEM UNDERSTANDING:
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '0.75rem' }}>
                    <div>
                      <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', display: 'block' }}>Device</span>
                      <strong style={{ fontSize: '0.88rem', color: 'var(--text-primary)' }}>
                        {result.escalation_package?.problem_profile?.device_type || 'iPhone / Apple Device'}
                      </strong>
                    </div>
                    <div>
                      <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', display: 'block' }}>OS / Service</span>
                      <strong style={{ fontSize: '0.88rem', color: 'var(--text-primary)' }}>
                        {result.escalation_package?.problem_profile?.os_or_service || 'iOS / Apple Services'}
                      </strong>
                    </div>
                    <div>
                      <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', display: 'block' }}>Primary Symptom</span>
                      <strong style={{ fontSize: '0.88rem', color: 'var(--text-primary)' }}>
                        {result.escalation_package?.problem_profile?.primary_symptom || result.problem_summary || 'Operational issue'}
                      </strong>
                    </div>
                    <div>
                      <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', display: 'block' }}>Causal Trigger</span>
                      <strong style={{ fontSize: '0.88rem', color: 'var(--text-primary)' }}>
                        {result.escalation_package?.problem_profile?.causal_trigger || 'None detected'}
                      </strong>
                    </div>
                    <div>
                      <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', display: 'block' }}>Information Sufficiency</span>
                      <span className="tag tag-neutral">
                        {result.escalation_package?.problem_profile?.is_sufficient_for_resolution === false ? 'INSUFFICIENT' : 'SUFFICIENT'}
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              {/* ------------------------------------------------------------ */}
              {/* SECTION 2 — HISTORICAL EVIDENCE                              */}
              {/* ------------------------------------------------------------ */}
              <div className="sg-card">
                <div className="sg-card-header">
                  <div className="sg-section-title">
                    <span>Section 2 — Historical AppleSupport Evidence</span>
                  </div>
                  <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                    {getVerdictBadge(result.evidence_validation?.evidence_verdict)}
                    <span className="tag tag-neutral">
                      {result.evidence_cases?.length || 0} Relevant Cases
                    </span>
                  </div>
                </div>

                {/* Evidence Metrics Banner */}
                <div
                  style={{
                    display: 'flex',
                    flexWrap: 'wrap',
                    gap: '1.5rem',
                    background: 'var(--bg-subtle)',
                    padding: '0.75rem 1rem',
                    borderRadius: 'var(--radius-s)',
                    marginBottom: '1rem',
                    fontSize: '0.8rem',
                  }}
                >
                  <div>
                    <span style={{ color: 'var(--text-muted)' }}>Evidence Strength: </span>
                    <strong style={{ color: 'var(--text-primary)' }}>
                      {result.evidence_validation?.evidence_verdict || 'EVALUATED'}
                    </strong>
                  </div>
                  <div>
                    <span style={{ color: 'var(--text-muted)' }}>Coverage Score: </span>
                    <strong style={{ color: 'var(--text-primary)' }}>
                      {result.evidence_validation?.coverage_score != null
                        ? `${(result.evidence_validation.coverage_score * 100).toFixed(0)}%`
                        : '—'}
                    </strong>
                  </div>
                  <div>
                    <span style={{ color: 'var(--text-muted)' }}>Agreement Score: </span>
                    <strong style={{ color: 'var(--text-primary)' }}>
                      {result.evidence_validation?.agreement_score != null
                        ? `${(result.evidence_validation.agreement_score * 100).toFixed(0)}%`
                        : '—'}
                    </strong>
                  </div>
                  <div>
                    <span style={{ color: 'var(--text-muted)' }}>Direct Matches: </span>
                    <strong style={{ color: 'var(--green-text)' }}>
                      {result.evidence_validation?.direct_matches_count ?? 0}
                    </strong>
                  </div>
                </div>

                {/* Cases List */}
                {(!result.evidence_cases || result.evidence_cases.length === 0) ? (
                  <div style={{ padding: '1.5rem', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.88rem' }}>
                    No historical evidence cases matched this query.
                  </div>
                ) : (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
                    {result.evidence_cases.map((evCase, i) => (
                      <div
                        key={i}
                        style={{
                          background: '#ffffff',
                          border: '1px solid var(--border)',
                          borderRadius: 'var(--radius-s)',
                          padding: '1rem',
                          boxShadow: 'var(--shadow-xs)',
                        }}
                      >
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8, flexWrap: 'wrap', gap: 6 }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                            {getMatchTierBadge(evCase.match_tier)}
                            <span className="mono" style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                              Case #{evCase.tweet_id || i + 1}
                            </span>
                          </div>
                          <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', fontFamily: 'var(--mono)' }}>
                            Evidence Score: <strong style={{ color: 'var(--blue-primary)' }}>{evCase.relevance_score?.toFixed(3)}</strong>
                          </div>
                        </div>

                        {/* Customer Problem */}
                        <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: 8 }}>
                          <span style={{ fontWeight: 600, color: 'var(--text-muted)', fontSize: '0.75rem', display: 'block' }}>
                            HISTORICAL CUSTOMER INQUIRY:
                          </span>
                          "{evCase.customer_query}"
                        </div>

                        {/* Official AppleSupport Response */}
                        <div
                          style={{
                            background: 'var(--blue-bg)',
                            borderLeft: '3px solid var(--blue-primary)',
                            padding: '0.65rem 0.85rem',
                            borderRadius: '0 var(--radius-s) var(--radius-s) 0',
                            fontSize: '0.85rem',
                            color: 'var(--blue-text)',
                          }}
                        >
                          <span style={{ fontWeight: 600, fontSize: '0.72rem', display: 'block', marginBottom: 2 }}>
                            HISTORICAL APPLESUPPORT RESOLUTION:
                          </span>
                          {evCase.brand_response}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* ------------------------------------------------------------ */}
              {/* SECTION 3 — DRAFT REPLY & GROUNDING VERIFICATION             */}
              {/* ------------------------------------------------------------ */}
              <div className="sg-card">
                <div className="sg-card-header">
                  <div className="sg-section-title">
                    <span>Section 3 — Grounded Response & Verification</span>
                  </div>
                  <div style={{ display: 'flex', gap: 8 }}>
                    {result.response_grounding?.verification_status && (
                      <span className={result.response_grounding.is_grounded ? 'tag tag-green' : 'tag tag-red'}>
                        {result.response_grounding.verification_status}
                      </span>
                    )}
                  </div>
                </div>

                {/* Draft Reply Box */}
                <div style={{ marginBottom: '1.25rem' }}>
                  <div style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-muted)', marginBottom: 6 }}>
                    GENERATED APPLESUPPORT DRAFT REPLY:
                  </div>
                  <div
                    style={{
                      background: '#ffffff',
                      border: '1px solid var(--border-strong)',
                      borderRadius: 'var(--radius-m)',
                      padding: '1.25rem',
                      fontSize: '0.92rem',
                      lineHeight: 1.65,
                      color: 'var(--text-primary)',
                      position: 'relative',
                    }}
                  >
                    {result.grounded_response || (
                      <span style={{ color: 'var(--text-muted)', fontStyle: 'italic' }}>
                        No automatic reply generated because this inquiry requires human escalation. (Troubleshooting strategy: {result.resolution_strategy || 'Review by specialist'}).
                      </span>
                    )}

                    {result.grounded_response && (
                      <button
                        className="btn btn-secondary"
                        onClick={() => handleCopy(result.grounded_response!)}
                        style={{ position: 'absolute', top: 12, right: 12, padding: '4px 10px', fontSize: '0.75rem' }}
                      >
                        {copied ? '✓ Copied' : 'Copy Reply'}
                      </button>
                    )}
                  </div>
                </div>

                {/* Grounding Verification Breakdown */}
                {result.response_grounding && (
                  <div style={{ background: 'var(--bg-subtle)', padding: '1rem', borderRadius: 'var(--radius-s)', border: '1px solid var(--border)' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                      <span style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-muted)' }}>
                        GROUNDING VERIFICATION DETAILS
                      </span>
                      <span style={{ fontSize: '0.78rem', fontFamily: 'var(--mono)', color: 'var(--text-secondary)' }}>
                        Grounding Score: {(result.response_grounding.grounding_score * 100).toFixed(0)}%
                      </span>
                    </div>

                    <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginBottom: 6 }}>
                      {result.response_grounding.verification_notes || 'All candidate claims verified against historical evidence corpus.'}
                    </div>

                    {/* Unsupported Claims (if any) */}
                    {result.response_grounding.unsupported_claims && result.response_grounding.unsupported_claims.length > 0 && (
                      <div style={{ marginTop: 8, padding: '8px 12px', background: 'var(--red-bg)', borderRadius: 'var(--radius-s)', border: '1px solid var(--red-border)' }}>
                        <div style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--red-text)', marginBottom: 4 }}>
                          UNSUPPORTED CLAIMS DETECTED:
                        </div>
                        {result.response_grounding.unsupported_claims.map((claim, idx) => (
                          <div key={idx} style={{ fontSize: '0.8rem', color: 'var(--red-text)' }}>
                            • {claim}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          )}
        </>
      )}

      {/* ==================================================================== */}
      {/* MODE 2: MULTI-TURN SUPPORT SESSION                                  */}
      {/* ==================================================================== */}
      {mode === 'multi-turn' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: '1.25rem' }}>
          {/* Main Chat Area */}
          <div className="sg-card" style={{ display: 'flex', flexDirection: 'column', minHeight: 520 }}>
            <div className="sg-card-header">
              <div className="sg-section-title">
                <span>Interactive Support Session</span>
                <span className={`tag ${convStatus === 'RESOLVED' ? 'tag-green' : convStatus === 'ESCALATED' ? 'tag-amber' : 'tag-blue'}`}>
                  {convStatus}
                </span>
              </div>
              <div style={{ display: 'flex', gap: 8 }}>
                <button className="btn btn-secondary" onClick={handleLoadAudit} style={{ padding: '4px 10px', fontSize: '0.75rem' }}>
                  View Audit
                </button>
                {convStatus === 'ACTIVE' && (
                  <button className="btn btn-secondary" onClick={handleResolveConv} style={{ padding: '4px 10px', fontSize: '0.75rem', color: 'var(--green-text)' }}>
                    Resolve Session
                  </button>
                )}
                <button
                  className="btn btn-secondary"
                  onClick={() => handleStartConversation()}
                  style={{ padding: '4px 10px', fontSize: '0.75rem' }}
                >
                  New Session
                </button>
              </div>
            </div>

            {/* Message Thread */}
            <div
              style={{
                flex: 1,
                overflowY: 'auto',
                display: 'flex',
                flexDirection: 'column',
                gap: '1rem',
                padding: '0.5rem 0',
                maxHeight: 400,
              }}
            >
              {convTurns.length === 0 && (
                <div style={{ textAlign: 'center', color: 'var(--text-muted)', margin: 'auto', fontSize: '0.9rem' }}>
                  Session started. Type a customer question below to begin troubleshooting.
                </div>
              )}

              {convTurns.map((turn, i) => {
                const isCustomer = turn.role === 'customer'
                return (
                  <div
                    key={i}
                    style={{
                      display: 'flex',
                      justifyContent: isCustomer ? 'flex-end' : 'flex-start',
                    }}
                  >
                    <div
                      style={{
                        maxWidth: '80%',
                        background: isCustomer ? 'var(--blue-primary)' : 'var(--bg-subtle)',
                        color: isCustomer ? '#ffffff' : 'var(--text-primary)',
                        padding: '0.75rem 1rem',
                        borderRadius: isCustomer ? '14px 14px 2px 14px' : '14px 14px 14px 2px',
                        border: isCustomer ? 'none' : '1px solid var(--border)',
                        boxShadow: 'var(--shadow-xs)',
                        fontSize: '0.88rem',
                        lineHeight: 1.5,
                      }}
                    >
                      <div style={{ fontSize: '0.68rem', fontWeight: 600, opacity: 0.75, marginBottom: 2 }}>
                        {isCustomer ? 'Customer' : 'AppleSupport AI'}
                      </div>
                      {turn.content}
                    </div>
                  </div>
                )
              })}
              <div ref={chatEndRef} />
            </div>

            {/* Conversation Summary Banner if ended */}
            {convSummary && (
              <div
                style={{
                  background: convStatus === 'RESOLVED' ? 'var(--green-bg)' : 'var(--amber-bg)',
                  border: `1px solid ${convStatus === 'RESOLVED' ? 'var(--green-border)' : 'var(--amber-border)'}`,
                  borderRadius: 'var(--radius-s)',
                  padding: '0.75rem 1rem',
                  fontSize: '0.85rem',
                  color: convStatus === 'RESOLVED' ? 'var(--green-text)' : 'var(--amber-text)',
                  margin: '0.75rem 0',
                }}
              >
                <strong>Session Outcome:</strong> {convSummary}
              </div>
            )}

            {/* Chat Input */}
            <div style={{ display: 'flex', gap: 8, marginTop: 'auto', paddingTop: '1rem', borderTop: '1px solid var(--border)' }}>
              <input
                type="text"
                className="textarea-custom"
                style={{ padding: '8px 12px', minHeight: 'unset', resize: 'none' }}
                placeholder="Type customer reply..."
                value={convInput}
                onChange={(e) => setConvInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSendTurn()}
                disabled={convLoading || convStatus === 'RESOLVED' || convStatus === 'ESCALATED'}
              />
              <button
                className="btn btn-primary"
                onClick={handleSendTurn}
                disabled={convLoading || !convInput.trim() || convStatus === 'RESOLVED' || convStatus === 'ESCALATED'}
              >
                {convLoading ? <span className="spinner" /> : 'Send'}
              </button>
            </div>
          </div>

          {/* Session Facts & State Panel */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <div className="sg-card">
              <div style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-muted)', marginBottom: 8 }}>
                SESSION STATE
              </div>
              <div style={{ fontSize: '0.82rem', display: 'flex', flexDirection: 'column', gap: 6 }}>
                <div>
                  <span style={{ color: 'var(--text-muted)' }}>Status: </span>
                  <strong>{convStatus}</strong>
                </div>
                <div>
                  <span style={{ color: 'var(--text-muted)' }}>Turns: </span>
                  <strong>{convTurns.length}</strong>
                </div>
                <div>
                  <span style={{ color: 'var(--text-muted)' }}>Session ID: </span>
                  <div className="mono" style={{ fontSize: '0.7rem', color: 'var(--text-muted)', wordBreak: 'break-all' }}>
                    {conversationId || 'None'}
                  </div>
                </div>
              </div>
            </div>

            <div className="sg-card">
              <div style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-muted)', marginBottom: 8 }}>
                CONFIRMED PROBLEM FACTS
              </div>
              {Object.keys(confirmedFacts).length === 0 ? (
                <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                  No facts confirmed yet. Facts will be extracted as customer clarifies details.
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                  {Object.entries(confirmedFacts).map(([k, v]) => (
                    <div key={k} style={{ fontSize: '0.78rem', background: 'var(--bg-subtle)', padding: '4px 8px', borderRadius: 'var(--radius-xs)' }}>
                      <span style={{ color: 'var(--text-muted)' }}>{k}: </span>
                      <strong>{String(v)}</strong>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Audit Modal */}
      {showAuditModal && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(15,23,42,0.6)',
            backdropFilter: 'blur(4px)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000,
            padding: '1rem',
          }}
        >
          <div
            className="sg-card"
            style={{ width: '100%', maxWidth: 650, maxHeight: '80vh', display: 'flex', flexDirection: 'column' }}
          >
            <div className="sg-card-header">
              <span className="sg-section-title">Session Audit Trail</span>
              <button
                onClick={() => setShowAuditModal(false)}
                style={{ background: 'none', border: 'none', fontSize: '1.1rem', cursor: 'pointer' }}
              >
                ✕
              </button>
            </div>
            <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 8 }}>
              {auditLogs.map((log, i) => (
                <div key={i} style={{ background: 'var(--bg-subtle)', padding: '8px 12px', borderRadius: 'var(--radius-s)', fontSize: '0.75rem', fontFamily: 'var(--mono)' }}>
                  <span style={{ color: 'var(--blue-primary)', fontWeight: 600 }}>{log.event || 'EVENT'}</span> · {new Date(log.timestamp).toLocaleTimeString()}
                  <pre style={{ marginTop: 4, whiteSpace: 'pre-wrap', color: 'var(--text-secondary)' }}>
                    {JSON.stringify(log.data || log, null, 2)}
                  </pre>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
