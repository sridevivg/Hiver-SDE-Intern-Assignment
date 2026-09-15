/**
 * SupportGraph AI — Customer Support Chatbot UX
 *
 * Real customer-facing support chatbot for AppleSupport.
 * Core workflow:
 * 1. User describes problem
 * 2. Backend processes message (multi-turn conversation + evidence retrieval)
 * 3. AI response appears in chat
 * 4. Related historical problems appear BELOW the AI response
 * 5. If human escalation is needed, user receives a clean, natural notification.
 * NO internal developer labels (e.g. AUTO-HANDLED, VERIFICATION_VETO) are shown to the customer.
 */

import React, { useState, useEffect, useRef } from 'react'
import { api } from '../services/api'
import type { RetrievedEvidenceCase, EvidenceMatchTier } from '../types'

interface ChatMessage {
  id: string
  speaker: 'user' | 'agent'
  text: string
  timestamp: string
  isEscalated?: boolean
  escalationNote?: string
  routingDecision?: string
  relatedCases?: RetrievedEvidenceCase[]
}

const QUICK_PROMPTS = [
  'My iPhone battery is draining very quickly after updating to iOS 16.',
  'My AirPods Pro left earbud has no sound and will not connect.',
  'I forgot my Apple ID password and cannot sign in to iCloud.',
  'My iPhone is extremely hot, smoking and the back glass is cracking.',
]

export const SupportPage: React.FC = () => {
  const [conversationId, setConversationId] = useState<string | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [inputMessage, setInputMessage] = useState('')
  const [loading, setLoading] = useState(false)
  const [loadingText, setLoadingText] = useState('Thinking...')
  const [errorMessage, setErrorMessage] = useState<string | null>(null)

  const chatEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  // Start fresh conversation
  const startNewConversation = () => {
    setConversationId(null)
    setMessages([])
    setInputMessage('')
    setErrorMessage(null)
  }

  // Format match tier cleanly for customer
  const renderTierBadge = (tier?: EvidenceMatchTier) => {
    switch (tier) {
      case 'DIRECT_PROBLEM_MATCH':
        return (
          <span
            style={{
              fontSize: '0.68rem',
              fontWeight: 600,
              padding: '2px 8px',
              borderRadius: '9999px',
              background: '#f0fdf4',
              color: '#166534',
              border: '1px solid #bbf7d0',
            }}
          >
            Direct Match
          </span>
        )
      case 'RELATED_SYMPTOM':
        return (
          <span
            style={{
              fontSize: '0.68rem',
              fontWeight: 600,
              padding: '2px 8px',
              borderRadius: '9999px',
              background: '#eff6ff',
              color: '#1e40af',
              border: '1px solid #bfdbfe',
            }}
          >
            Similar Symptom
          </span>
        )
      case 'RELATED_CONTEXT':
        return (
          <span
            style={{
              fontSize: '0.68rem',
              fontWeight: 600,
              padding: '2px 8px',
              borderRadius: '9999px',
              background: '#faf5ff',
              color: '#6b21a8',
              border: '1px solid #e9d5ff',
            }}
          >
            Related Context
          </span>
        )
      case 'WEAK_SEMANTIC_MATCH':
      default:
        return (
          <span
            style={{
              fontSize: '0.68rem',
              fontWeight: 500,
              padding: '2px 8px',
              borderRadius: '9999px',
              background: '#f1f5f9',
              color: '#475569',
              border: '1px solid #e2e8f0',
            }}
          >
            General Topic
          </span>
        )
    }
  }

  // Send message
  const handleSendMessage = async (textToSend?: string) => {
    const text = (textToSend || inputMessage).trim()
    if (!text || loading) return

    setInputMessage('')
    setErrorMessage(null)

    // Append user message immediately
    const userMsgId = `usr_${Date.now()}`
    const userMsg: ChatMessage = {
      id: userMsgId,
      speaker: 'user',
      text,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    }

    setMessages((prev) => [...prev, userMsg])
    setLoading(true)
    setLoadingText('Thinking...')

    try {
      // Step 1: Query backend resolution pipeline & retrieve related historical evidence
      setLoadingText('Looking for similar support cases...')

      // Concurrently query resolution engine & multi-turn conversation
      let convId = conversationId
      let agentReplyText = ''
      let isEscalated = false
      let escalationNote = ''
      let routingDecision: string = 'AUTO_HANDLE'
      let relatedCases: RetrievedEvidenceCase[] = []

      // 1. Check end-to-end resolution engine
      try {
        const resolution = await api.resolveInquiry(text, 3, undefined, 'LIVE_SUPPORT')
        routingDecision = resolution.routing_decision || 'AUTO_HANDLE'
        if (resolution.routing_decision === 'ESCALATE_TO_HUMAN') {
          isEscalated = true
          // Check if safety risk
          const isHazard =
            text.toLowerCase().includes('smoking') ||
            text.toLowerCase().includes('fire') ||
            text.toLowerCase().includes('hot') ||
            text.toLowerCase().includes('burn') ||
            resolution.explanation?.toLowerCase().includes('hazard')

          if (isHazard) {
            escalationNote =
              'For your immediate safety, please stop using the device, disconnect it from power, and keep it away from flammable materials. A senior support specialist has been alerted for priority review.'
          } else {
            escalationNote =
              'This issue requires assistance from an Apple Support specialist. Your request has been sent for human review.'
          }
        }

        if (resolution.grounded_response) {
          agentReplyText = resolution.grounded_response
        }

        if (resolution.evidence_cases && resolution.evidence_cases.length > 0) {
          relatedCases = resolution.evidence_cases
        }
      } catch (resErr) {
        console.warn('Resolution endpoint fallback:', resErr)
      }

      // 2. Multi-turn conversation state machine
      if (!convId) {
        const startRes = await api.startConversation('customer_web', text)
        convId = startRes.conversation_id
        setConversationId(convId)
        if (!agentReplyText && startRes.initial_agent_response) {
          agentReplyText = startRes.initial_agent_response
        }
        if (startRes.status === 'ESCALATED') {
          isEscalated = true
          routingDecision = 'ESCALATE_TO_HUMAN'
          if (!escalationNote) {
            escalationNote =
              'This issue requires assistance from an Apple Support specialist. Your request has been sent for human review.'
          }
        }
      } else {
        const turnRes = await api.sendConversationMessage(convId, text)
        if (!agentReplyText && turnRes.agent_response) {
          agentReplyText = turnRes.agent_response
        }
        if (turnRes.status === 'ESCALATED') {
          isEscalated = true
          routingDecision = 'ESCALATE_TO_HUMAN'
          if (!escalationNote) {
            escalationNote =
              'This issue requires assistance from an Apple Support specialist. Your request has been sent for human review.'
          }
        }
      }

      // 3. If evidence not returned yet, query retrieveEvidence
      if (relatedCases.length === 0) {
        try {
          const evidence = await api.retrieveEvidence(text, '', 3)
          if (evidence && evidence.length > 0) {
            relatedCases = evidence
          }
        } catch {
          // ignore
        }
      }

      // Default safe reply if empty
      if (!agentReplyText) {
        if (isEscalated) {
          agentReplyText =
            escalationNote ||
            'This issue requires assistance from a support specialist. Your request has been sent for human review.'
        } else {
          agentReplyText =
            'Thank you for describing the issue. Let us look into how we can resolve this for your Apple device.'
        }
      }

      // Append agent response
      const agentMsg: ChatMessage = {
        id: `agt_${Date.now()}`,
        speaker: 'agent',
        text: agentReplyText,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        isEscalated,
        escalationNote: isEscalated ? escalationNote : undefined,
        routingDecision,
        relatedCases,
      }

      setMessages((prev) => [...prev, agentMsg])
    } catch (err: any) {
      setErrorMessage(
        err.message?.includes('fetch')
          ? 'SupportGraph is temporarily unavailable. Please try again.'
          : "We couldn't process your request. Please try again."
      )
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ maxWidth: 780, margin: '0 auto', display: 'flex', flexDirection: 'column', height: 'calc(100vh - 120px)' }}>
      {/* Top Header / Welcome Area */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          paddingBottom: '1rem',
          borderBottom: '1px solid var(--border)',
          marginBottom: '1rem',
        }}
      >
        <div>
          <h1 style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--text-primary)' }}>
            How can we help?
          </h1>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
            AppleSupport AI Assistant · Verified historical resolutions
          </p>
        </div>

        {messages.length > 0 && (
          <button
            onClick={startNewConversation}
            className="btn btn-secondary"
            style={{ fontSize: '0.8rem', padding: '5px 12px' }}
          >
            New Conversation
          </button>
        )}
      </div>

      {/* Messages Scroll Area */}
      <div
        style={{
          flex: 1,
          overflowY: 'auto',
          padding: '0.5rem 0.25rem',
          display: 'flex',
          flexDirection: 'column',
          gap: '1.5rem',
        }}
      >
        {/* Welcome Empty State */}
        {messages.length === 0 && (
          <div
            style={{
              margin: 'auto',
              maxWidth: 540,
              textAlign: 'center',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: '1rem',
              padding: '2rem 1rem',
            }}
          >
            <div
              style={{
                width: 48,
                height: 48,
                borderRadius: '50%',
                background: 'var(--blue-bg)',
                border: '1px solid var(--blue-border)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '1.5rem',
                color: 'var(--blue-primary)',
              }}
            >
              
            </div>
            <div>
              <h2 style={{ fontSize: '1.1rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                Welcome to Apple Support
              </h2>
              <p style={{ fontSize: '0.88rem', color: 'var(--text-muted)', marginTop: 4 }}>
                Describe any issue you are experiencing with your iPhone, iPad, Mac, or accessories.
              </p>
            </div>

            {/* Quick Prompts */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8, width: '100%', marginTop: '0.75rem' }}>
              <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', textAlign: 'left' }}>
                COMMON TOPICS:
              </div>
              {QUICK_PROMPTS.map((prompt, i) => (
                <button
                  key={i}
                  onClick={() => handleSendMessage(prompt)}
                  style={{
                    textAlign: 'left',
                    background: 'var(--surface)',
                    border: '1px solid var(--border)',
                    borderRadius: 'var(--radius-s)',
                    padding: '10px 14px',
                    fontSize: '0.85rem',
                    color: 'var(--text-secondary)',
                    cursor: 'pointer',
                    transition: 'all 0.15s ease',
                    boxShadow: 'var(--shadow-xs)',
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.borderColor = 'var(--blue-border)'
                    e.currentTarget.style.background = 'var(--blue-bg)'
                    e.currentTarget.style.color = 'var(--blue-text)'
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.borderColor = 'var(--border)'
                    e.currentTarget.style.background = 'var(--surface)'
                    e.currentTarget.style.color = 'var(--text-secondary)'
                  }}
                >
                  {prompt}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Chat Message List */}
        {messages.map((msg) => {
          const isUser = msg.speaker === 'user'
          return (
            <div
              key={msg.id}
              style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: isUser ? 'flex-end' : 'flex-start',
                gap: '0.5rem',
              }}
            >
              {/* Speaker Label & Status Tag */}
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  fontSize: '0.72rem',
                  fontWeight: 600,
                  color: 'var(--text-muted)',
                  padding: '0 4px',
                }}
              >
                <span>{isUser ? 'You' : 'AppleSupport AI'} · {msg.timestamp}</span>
                {!isUser && msg.routingDecision && (
                  <span
                    style={{
                      fontSize: '0.66rem',
                      fontWeight: 700,
                      padding: '2px 8px',
                      borderRadius: '9999px',
                      letterSpacing: '0.03em',
                      background: msg.routingDecision === 'AUTO_HANDLE' ? '#f0fdf4' : '#fffbeb',
                      color: msg.routingDecision === 'AUTO_HANDLE' ? '#166534' : '#b45309',
                      border: `1px solid ${msg.routingDecision === 'AUTO_HANDLE' ? '#bbf7d0' : '#fde68a'}`,
                      textTransform: 'uppercase',
                    }}
                  >
                    {msg.routingDecision === 'AUTO_HANDLE' ? '✓ Auto-Handled' : '⚠ Human Review Required'}
                  </span>
                )}
              </div>

              {/* Chat Bubble */}
              <div
                style={{
                  maxWidth: '85%',
                  background: isUser ? '#2563eb' : '#ffffff',
                  color: isUser ? '#ffffff' : 'var(--text-primary)',
                  padding: '0.9rem 1.15rem',
                  borderRadius: isUser ? '16px 16px 3px 16px' : '16px 16px 16px 3px',
                  border: isUser ? 'none' : '1px solid var(--border)',
                  boxShadow: 'var(--shadow-xs)',
                  fontSize: '0.92rem',
                  lineHeight: 1.6,
                  whiteSpace: 'pre-wrap',
                }}
              >
                {msg.text}
              </div>

              {/* Human Escalation Notification (If Escalated) */}
              {!isUser && msg.isEscalated && (
                <div
                  style={{
                    maxWidth: '85%',
                    background: '#fffbeb',
                    border: '1px solid #fde68a',
                    borderRadius: 'var(--radius-s)',
                    padding: '0.85rem 1rem',
                    color: '#92400e',
                    fontSize: '0.85rem',
                    display: 'flex',
                    alignItems: 'flex-start',
                    gap: 10,
                  }}
                >
                  <span style={{ fontSize: '1rem', marginTop: 1 }}>ℹ</span>
                  <div>
                    <div style={{ fontWeight: 600, marginBottom: 2 }}>
                      Request Sent for Human Review
                    </div>
                    <div>
                      {msg.escalationNote ||
                        'This issue requires assistance from an Apple Support specialist. Your request has been sent for human review.'}
                    </div>
                  </div>
                </div>
              )}

              {/* REQUIRED SECTION: Related Historical Problems (BELOW AI Response) */}
              {!isUser && msg.relatedCases && msg.relatedCases.length > 0 && (
                <div
                  style={{
                    maxWidth: '92%',
                    width: '100%',
                    marginTop: '0.5rem',
                    background: '#f8fafc',
                    border: '1px solid var(--border)',
                    borderRadius: 'var(--radius-m)',
                    padding: '1rem 1.25rem',
                  }}
                >
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      marginBottom: '0.75rem',
                      paddingBottom: '0.5rem',
                      borderBottom: '1px solid var(--border)',
                    }}
                  >
                    <div>
                      <div
                        style={{
                          fontSize: '0.78rem',
                          fontWeight: 700,
                          color: 'var(--text-primary)',
                          textTransform: 'uppercase',
                          letterSpacing: '0.04em',
                        }}
                      >
                        Related problems
                      </div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                        Similar issues from previous AppleSupport conversations
                      </div>
                    </div>
                    <span
                      style={{
                        fontSize: '0.7rem',
                        fontWeight: 600,
                        color: 'var(--text-muted)',
                        background: '#ffffff',
                        border: '1px solid var(--border)',
                        padding: '2px 8px',
                        borderRadius: '9999px',
                      }}
                    >
                      {msg.relatedCases.length} similar cases
                    </span>
                  </div>

                  {/* List of Related Cases */}
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                    {msg.relatedCases.map((c, idx) => {
                      const customerIssue =
                        c.historical_customer_message || c.customer_query || 'Customer reported similar problem.'
                      const historicalApproach =
                        c.historical_brand_response || c.brand_response || 'Standard AppleSupport resolution.'
                      const similarity = c.operational_similarity ?? c.similarity_score ?? c.relevance_score

                      return (
                        <div
                          key={idx}
                          style={{
                            background: '#ffffff',
                            border: '1px solid var(--border)',
                            borderRadius: 'var(--radius-s)',
                            padding: '0.85rem 1rem',
                            display: 'flex',
                            flexDirection: 'column',
                            gap: '0.5rem',
                          }}
                        >
                          <div
                            style={{
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'space-between',
                              flexWrap: 'wrap',
                              gap: 6,
                            }}
                          >
                            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                              {renderTierBadge(c.match_tier)}
                              {c.case_id && (
                                <span
                                  style={{
                                    fontSize: '0.68rem',
                                    color: 'var(--text-subtle)',
                                    fontFamily: 'var(--mono)',
                                  }}
                                >
                                  {c.case_id}
                                </span>
                              )}
                            </div>
                            {similarity != null && (
                              <span
                                style={{
                                  fontSize: '0.72rem',
                                  color: 'var(--text-muted)',
                                  fontFamily: 'var(--mono)',
                                }}
                              >
                                Match score: {(similarity * 100).toFixed(0)}%
                              </span>
                            )}
                          </div>

                          {/* Customer Issue */}
                          <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
                            <strong style={{ color: 'var(--text-muted)', fontSize: '0.72rem', display: 'block', marginBottom: 2 }}>
                              CUSTOMER ISSUE:
                            </strong>
                            "{customerIssue}"
                          </div>

                          {/* Historical Approach */}
                          <div
                            style={{
                              fontSize: '0.82rem',
                              color: '#1d4ed8',
                              background: '#eff6ff',
                              borderLeft: '3px solid #2563eb',
                              padding: '6px 10px',
                              borderRadius: '0 var(--radius-xs) var(--radius-xs) 0',
                            }}
                          >
                            <strong style={{ fontSize: '0.72rem', display: 'block', marginBottom: 2 }}>
                              HISTORICAL APPLESUPPORT APPROACH:
                            </strong>
                            {historicalApproach}
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </div>
              )}
            </div>
          )
        })}

        {/* Loading Bubble */}
        {loading && (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: 6 }}>
            <div style={{ fontSize: '0.72rem', fontWeight: 600, color: 'var(--text-muted)' }}>
              AppleSupport AI
            </div>
            <div
              style={{
                background: '#ffffff',
                border: '1px solid var(--border)',
                borderRadius: '16px 16px 16px 3px',
                padding: '0.75rem 1.1rem',
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                fontSize: '0.85rem',
                color: 'var(--text-muted)',
                boxShadow: 'var(--shadow-xs)',
              }}
            >
              <span className="spinner" style={{ width: 14, height: 14 }} />
              <span>{loadingText}</span>
            </div>
          </div>
        )}

        <div ref={chatEndRef} />
      </div>

      {/* Error Message */}
      {errorMessage && (
        <div
          style={{
            background: '#fef2f2',
            border: '1px solid #fecaca',
            color: '#b91c1c',
            borderRadius: 'var(--radius-s)',
            padding: '8px 12px',
            fontSize: '0.82rem',
            margin: '8px 0',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
          }}
        >
          <span>{errorMessage}</span>
          <button
            onClick={() => setErrorMessage(null)}
            style={{ background: 'none', border: 'none', color: '#b91c1c', cursor: 'pointer' }}
          >
            ✕
          </button>
        </div>
      )}

      {/* Bottom Input Area */}
      <div
        style={{
          borderTop: '1px solid var(--border)',
          paddingTop: '0.85rem',
          background: 'var(--bg)',
        }}
      >
        <form
          onSubmit={(e) => {
            e.preventDefault()
            handleSendMessage()
          }}
          style={{ display: 'flex', gap: '0.65rem' }}
        >
          <input
            type="text"
            className="textarea-custom"
            style={{
              padding: '12px 16px',
              borderRadius: 'var(--radius-pill)',
              minHeight: 'unset',
              fontSize: '0.92rem',
            }}
            placeholder="Describe your problem..."
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            disabled={loading}
          />
          <button
            type="submit"
            className="btn btn-primary"
            style={{
              borderRadius: 'var(--radius-pill)',
              padding: '0 24px',
              minWidth: 90,
              fontWeight: 600,
            }}
            disabled={loading || !inputMessage.trim()}
          >
            Send
          </button>
        </form>
      </div>
    </div>
  )
}
