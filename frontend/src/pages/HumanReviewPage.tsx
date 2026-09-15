/**
 * SupportGraph AI — Live Human Review Queue
 *
 * Dedicated live work queue for human specialists to adjudicate escalated customer inquiries.
 * Strict Requirements:
 * - Displays ONLY live support escalations (source: LIVE_SUPPORT)
 * - Empty state when no live cases require human assistance
 * - Live auto-refresh polling (5s interval)
 * - Complete case adjudication:
 *   1. Customer Query
 *   2. AI Suggested Answer / Guidance
 *   3. Why Human Review is Required
 *   4. Supporting Context (Intent, Device, Problem Family)
 *   5. Related Historical Problems & Approaches
 *   6. Action Controls: Approve Response, Edit Response, Escalate to Tier 2
 * - Upon action, case transitions status and leaves the active queue.
 */

import React, { useState, useEffect, useCallback, useRef } from 'react'
import { api } from '../services/api'
import type { LiveReviewCase, EvidenceMatchTier } from '../types'

export const HumanReviewPage: React.FC = () => {
  const [cases, setCases] = useState<LiveReviewCase[]>([])
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [actionSuccess, setActionSuccess] = useState<string | null>(null)
  const [selectedCaseId, setSelectedCaseId] = useState<string | null>(null)

  // Edit response state
  const [isEditing, setIsEditing] = useState(false)
  const [editedResponseText, setEditedResponseText] = useState('')
  const [specialistNotes, setSpecialistNotes] = useState('')
  const [submittingAction, setSubmittingAction] = useState(false)

  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // Fetch live queue from backend
  const fetchQueue = useCallback(async (isSilent = false) => {
    if (!isSilent) setRefreshing(true)
    try {
      const res = await api.getHumanReviewQueue('active', 50)
      const liveCases = res.cases || []
      setCases(liveCases)
      setErrorMessage(null)

      // Ensure active selection remains valid or defaults to first case
      setSelectedCaseId((currentId) => {
        if (!currentId) return liveCases.length > 0 ? liveCases[0].case_id : null
        const exists = liveCases.some((c) => c.case_id === currentId)
        return exists ? currentId : liveCases.length > 0 ? liveCases[0].case_id : null
      })
    } catch (err: any) {
      if (!isSilent) {
        setErrorMessage(
          err.message?.includes('fetch')
            ? 'Backend queue is temporarily unreachable.'
            : 'Unable to load live review queue.'
        )
      }
    } finally {
      setLoading(false)
      if (!isSilent) setRefreshing(false)
    }
  }, [])

  // Initial load + Polling every 5 seconds
  useEffect(() => {
    fetchQueue(false)
    pollingRef.current = setInterval(() => {
      fetchQueue(true)
    }, 5000)

    return () => {
      if (pollingRef.current) clearInterval(pollingRef.current)
    }
  }, [fetchQueue])

  // Active case object
  const activeCase = cases.find((c) => c.case_id === selectedCaseId) || cases[0] || null

  // Reset editing mode when selected case changes
  useEffect(() => {
    if (activeCase) {
      setEditedResponseText(activeCase.ai_suggested_response || '')
      setIsEditing(false)
      setSpecialistNotes('')
    }
  }, [activeCase?.case_id])

  // Clear action alert after 4 seconds
  useEffect(() => {
    if (actionSuccess) {
      const timer = setTimeout(() => setActionSuccess(null), 4000)
      return () => clearTimeout(timer)
    }
  }, [actionSuccess])

  // Approve action
  const handleApprove = async () => {
    if (!activeCase || submittingAction) return
    setSubmittingAction(true)
    try {
      await api.approveHumanReviewCase(
        activeCase.case_id,
        'specialist_on_duty',
        specialistNotes || 'Approved AI suggested response'
      )
      setActionSuccess(`Case ${activeCase.case_id} successfully approved and resolved.`)
      // Optimistically remove from active list
      setCases((prev) => prev.filter((c) => c.case_id !== activeCase.case_id))
      setIsEditing(false)
      setSpecialistNotes('')
    } catch (err: any) {
      setErrorMessage(err.message || 'Failed to approve case.')
    } finally {
      setSubmittingAction(false)
    }
  }

  // Edit & submit action
  const handleEditSubmit = async () => {
    if (!activeCase || submittingAction) return
    if (!editedResponseText.trim()) {
      setErrorMessage('Response text cannot be empty.')
      return
    }
    setSubmittingAction(true)
    try {
      await api.editHumanReviewCase(
        activeCase.case_id,
        editedResponseText.trim(),
        'specialist_on_duty',
        specialistNotes || 'Edited and resolved by human specialist'
      )
      setActionSuccess(`Case ${activeCase.case_id} edited response submitted and resolved.`)
      setCases((prev) => prev.filter((c) => c.case_id !== activeCase.case_id))
      setIsEditing(false)
      setSpecialistNotes('')
    } catch (err: any) {
      setErrorMessage(err.message || 'Failed to submit edited response.')
    } finally {
      setSubmittingAction(false)
    }
  }

  // Escalate to Tier 2 action
  const handleEscalateTier2 = async () => {
    if (!activeCase || submittingAction) return
    setSubmittingAction(true)
    try {
      await api.escalateHumanReviewCase(
        activeCase.case_id,
        'specialist_on_duty',
        specialistNotes || 'Escalated to Tier 2 Senior Engineering Support'
      )
      setActionSuccess(`Case ${activeCase.case_id} routed to Tier 2 Senior Engineering.`)
      setCases((prev) => prev.filter((c) => c.case_id !== activeCase.case_id))
      setIsEditing(false)
      setSpecialistNotes('')
    } catch (err: any) {
      setErrorMessage(err.message || 'Failed to escalate case.')
    } finally {
      setSubmittingAction(false)
    }
  }

  // Tier badge formatter
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

  return (
    <div style={{ maxWidth: 1100, margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      {/* Header */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          paddingBottom: '1rem',
          borderBottom: '1px solid var(--border)',
          flexWrap: 'wrap',
          gap: '0.75rem',
        }}
      >
        <div>
          <h1 style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--text-primary)' }}>
            Human Review
          </h1>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
            Cases requiring human assistance
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <span
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: 6,
              fontSize: '0.75rem',
              color: 'var(--text-muted)',
              background: 'var(--bg-subtle)',
              padding: '4px 10px',
              borderRadius: '9999px',
              border: '1px solid var(--border)',
            }}
          >
            <span
              style={{
                width: 7,
                height: 7,
                borderRadius: '50%',
                background: cases.length > 0 ? '#ef4444' : '#10b981',
              }}
            />
            {cases.length} active {cases.length === 1 ? 'case' : 'cases'}
          </span>

          <button
            onClick={() => fetchQueue(false)}
            className="btn btn-secondary"
            style={{ fontSize: '0.8rem', padding: '6px 14px' }}
            disabled={refreshing}
          >
            {refreshing ? <span className="spinner" style={{ width: 12, height: 12 }} /> : 'Refresh'}
          </button>
        </div>
      </div>

      {/* Action Success Alert */}
      {actionSuccess && (
        <div
          style={{
            background: '#f0fdf4',
            border: '1px solid #bbf7d0',
            color: '#166534',
            borderRadius: 'var(--radius-s)',
            padding: '10px 14px',
            fontSize: '0.85rem',
            display: 'flex',
            alignItems: 'center',
            gap: 8,
          }}
        >
          <span>✓</span>
          <span>{actionSuccess}</span>
        </div>
      )}

      {/* Error Alert */}
      {errorMessage && (
        <div
          style={{
            background: '#fef2f2',
            border: '1px solid #fecaca',
            color: '#b91c1c',
            borderRadius: 'var(--radius-s)',
            padding: '10px 14px',
            fontSize: '0.85rem',
          }}
        >
          {errorMessage}
        </div>
      )}

      {/* Main Content Area */}
      {loading ? (
        <div style={{ padding: '3.5rem', textAlign: 'center', color: 'var(--text-muted)' }}>
          <span className="spinner" style={{ width: 16, height: 16, marginBottom: 8 }} />
          <div style={{ fontSize: '0.88rem' }}>Loading live human review queue...</div>
        </div>
      ) : cases.length === 0 ? (
        /* Clean Empty Queue State */
        <div
          style={{
            background: '#ffffff',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius-m)',
            padding: '3.5rem 2rem',
            textAlign: 'center',
            boxShadow: 'var(--shadow-xs)',
          }}
        >
          <div
            style={{
              width: 46,
              height: 46,
              borderRadius: '50%',
              background: '#f0fdf4',
              border: '1px solid #bbf7d0',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              margin: '0 auto 14px auto',
              color: '#16a34a',
              fontSize: '1.3rem',
            }}
          >
            ✓
          </div>
          <h2 style={{ fontSize: '1.1rem', fontWeight: 600, color: 'var(--text-primary)' }}>
            No cases require human assistance
          </h2>
          <p
            style={{
              fontSize: '0.86rem',
              color: 'var(--text-muted)',
              marginTop: 6,
              maxWidth: 480,
              margin: '8px auto 0 auto',
              lineHeight: 1.5,
            }}
          >
            All incoming customer support inquiries are being autonomously resolved or have met verified automated resolution criteria. Escalated cases will appear here automatically in real time.
          </p>
        </div>
      ) : (
        /* Workspace: Left Queue List + Right Adjudication Card */
        <div style={{ display: 'grid', gridTemplateColumns: '320px 1fr', gap: '1.25rem', alignItems: 'flex-start' }}>
          {/* Left Column: Live Queue List */}
          <div
            style={{
              background: '#ffffff',
              border: '1px solid var(--border)',
              borderRadius: 'var(--radius-m)',
              overflow: 'hidden',
              boxShadow: 'var(--shadow-xs)',
            }}
          >
            <div
              style={{
                padding: '10px 14px',
                background: 'var(--bg-subtle)',
                borderBottom: '1px solid var(--border)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
              }}
            >
              <span
                style={{
                  fontSize: '0.75rem',
                  fontWeight: 700,
                  color: 'var(--text-muted)',
                  letterSpacing: '0.04em',
                  textTransform: 'uppercase',
                }}
              >
                Active Queue ({cases.length})
              </span>
              <span style={{ fontSize: '0.7rem', color: 'var(--text-subtle)' }}>Newest First</span>
            </div>

            <div style={{ maxHeight: 680, overflowY: 'auto' }}>
              {cases.map((c) => {
                const isSelected = c.case_id === activeCase?.case_id
                const isUrgent = c.priority === 'URGENT'
                const formattedTime = new Date(c.created_at).toLocaleTimeString([], {
                  hour: '2-digit',
                  minute: '2-digit',
                })

                return (
                  <div
                    key={c.case_id}
                    onClick={() => setSelectedCaseId(c.case_id)}
                    style={{
                      padding: '12px 14px',
                      borderBottom: '1px solid var(--border)',
                      cursor: 'pointer',
                      background: isSelected ? '#eff6ff' : '#ffffff',
                      borderLeft: isSelected
                        ? isUrgent
                          ? '4px solid #ef4444'
                          : '4px solid #2563eb'
                        : '4px solid transparent',
                      transition: 'all 0.15s ease',
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                        <span className="mono" style={{ fontSize: '0.72rem', fontWeight: 600, color: 'var(--text-secondary)' }}>
                          {c.case_id}
                        </span>
                        {isUrgent && (
                          <span
                            style={{
                              fontSize: '0.62rem',
                              fontWeight: 700,
                              background: '#fef2f2',
                              color: '#dc2626',
                              border: '1px solid #fecaca',
                              padding: '1px 5px',
                              borderRadius: '4px',
                              textTransform: 'uppercase',
                            }}
                          >
                            URGENT
                          </span>
                        )}
                      </div>
                      <span style={{ fontSize: '0.68rem', color: 'var(--text-subtle)' }}>
                        {formattedTime}
                      </span>
                    </div>

                    <div
                      style={{
                        fontSize: '0.84rem',
                        fontWeight: 500,
                        color: isSelected ? '#1d4ed8' : 'var(--text-primary)',
                        display: '-webkit-box',
                        WebkitLineClamp: 2,
                        WebkitBoxOrient: 'vertical',
                        overflow: 'hidden',
                        lineHeight: 1.4,
                      }}
                    >
                      {c.customer_query}
                    </div>
                  </div>
                )
              })}
            </div>
          </div>

          {/* Right Column: Active Case Adjudication Detail */}
          {activeCase && (
            <div
              style={{
                background: '#ffffff',
                border: '1px solid var(--border)',
                borderRadius: 'var(--radius-m)',
                padding: '1.5rem',
                boxShadow: 'var(--shadow-s)',
                display: 'flex',
                flexDirection: 'column',
                gap: '1.25rem',
              }}
            >
              {/* Card Header */}
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  paddingBottom: '0.75rem',
                  borderBottom: '1px solid var(--border)',
                  flexWrap: 'wrap',
                  gap: 8,
                }}
              >
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span className="mono" style={{ fontSize: '0.92rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                      {activeCase.case_id}
                    </span>
                    <span
                      style={{
                        fontSize: '0.66rem',
                        fontWeight: 600,
                        padding: '2px 6px',
                        borderRadius: '4px',
                        background: '#eff6ff',
                        color: '#1d4ed8',
                        border: '1px solid #bfdbfe',
                      }}
                    >
                      LIVE SUPPORT
                    </span>
                  </div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 2 }}>
                    Received: {new Date(activeCase.created_at).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' })}
                  </div>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  {activeCase.priority === 'URGENT' && (
                    <span
                      style={{
                        background: '#fef2f2',
                        color: '#dc2626',
                        border: '1px solid #fecaca',
                        padding: '3px 10px',
                        borderRadius: '9999px',
                        fontSize: '0.74rem',
                        fontWeight: 700,
                      }}
                    >
                      ⚠ PRIORITY URGENT
                    </span>
                  )}
                  <span
                    style={{
                      background: '#fffbeb',
                      color: '#b45309',
                      border: '1px solid #fde68a',
                      padding: '3px 10px',
                      borderRadius: '9999px',
                      fontSize: '0.74rem',
                      fontWeight: 600,
                    }}
                  >
                    Requires Human Review
                  </span>
                </div>
              </div>

              {/* 1. CUSTOMER QUERY */}
              <div>
                <div
                  style={{
                    fontSize: '0.75rem',
                    fontWeight: 700,
                    color: 'var(--text-muted)',
                    marginBottom: 6,
                    textTransform: 'uppercase',
                    letterSpacing: '0.04em',
                  }}
                >
                  Customer Query
                </div>
                <div
                  style={{
                    background: '#f8fafc',
                    border: '1px solid var(--border)',
                    borderRadius: 'var(--radius-s)',
                    padding: '1rem',
                    fontSize: '0.94rem',
                    color: 'var(--text-primary)',
                    lineHeight: 1.55,
                    fontWeight: 500,
                  }}
                >
                  "{activeCase.customer_query}"
                </div>
              </div>

              {/* 2. WHY HUMAN REVIEW IS REQUIRED */}
              <div>
                <div
                  style={{
                    fontSize: '0.75rem',
                    fontWeight: 700,
                    color: 'var(--text-muted)',
                    marginBottom: 6,
                    textTransform: 'uppercase',
                    letterSpacing: '0.04em',
                  }}
                >
                  Why Human Review is Required
                </div>
                <div
                  style={{
                    background: activeCase.priority === 'URGENT' ? '#fef2f2' : '#fffbeb',
                    borderLeft: `4px solid ${activeCase.priority === 'URGENT' ? '#ef4444' : '#d97706'}`,
                    borderRadius: '0 var(--radius-s) var(--radius-s) 0',
                    padding: '0.85rem 1rem',
                    color: activeCase.priority === 'URGENT' ? '#991b1b' : '#92400e',
                    fontSize: '0.88rem',
                    lineHeight: 1.5,
                  }}
                >
                  <div style={{ fontWeight: 600, marginBottom: 3 }}>
                    {activeCase.priority === 'URGENT'
                      ? 'Critical Safety / Thermal Hazard Escalation'
                      : 'Automated Confidence Gate Threshold'}
                  </div>
                  <div>{activeCase.escalation_reason}</div>
                </div>
              </div>

              {/* 3. SUPPORTING INFORMATION */}
              {(activeCase.intent || activeCase.problem_family || activeCase.problem_understanding) && (
                <div>
                  <div
                    style={{
                      fontSize: '0.75rem',
                      fontWeight: 700,
                      color: 'var(--text-muted)',
                      marginBottom: 6,
                      textTransform: 'uppercase',
                      letterSpacing: '0.04em',
                    }}
                  >
                    Supporting Diagnostic Context
                  </div>
                  <div
                    style={{
                      display: 'grid',
                      gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
                      gap: '0.75rem',
                      background: 'var(--bg-subtle)',
                      padding: '0.85rem 1rem',
                      borderRadius: 'var(--radius-s)',
                      border: '1px solid var(--border)',
                      fontSize: '0.82rem',
                    }}
                  >
                    {activeCase.intent && (
                      <div>
                        <span style={{ color: 'var(--text-muted)', display: 'block', fontSize: '0.72rem' }}>
                          CLASSIFIED INTENT
                        </span>
                        <strong style={{ color: 'var(--text-primary)' }}>{activeCase.intent}</strong>
                      </div>
                    )}
                    {activeCase.problem_family && (
                      <div>
                        <span style={{ color: 'var(--text-muted)', display: 'block', fontSize: '0.72rem' }}>
                          PROBLEM FAMILY
                        </span>
                        <strong style={{ color: 'var(--text-primary)' }}>{activeCase.problem_family}</strong>
                      </div>
                    )}
                    {activeCase.problem_understanding?.device_type && (
                      <div>
                        <span style={{ color: 'var(--text-muted)', display: 'block', fontSize: '0.72rem' }}>
                          DETECTED DEVICE
                        </span>
                        <strong style={{ color: 'var(--text-primary)' }}>
                          {activeCase.problem_understanding.device_type}
                        </strong>
                      </div>
                    )}
                    {activeCase.problem_understanding?.primary_symptom && (
                      <div>
                        <span style={{ color: 'var(--text-muted)', display: 'block', fontSize: '0.72rem' }}>
                          PRIMARY SYMPTOM
                        </span>
                        <strong style={{ color: 'var(--text-primary)' }}>
                          {activeCase.problem_understanding.primary_symptom}
                        </strong>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* 4. AI SUGGESTED ANSWER / GUIDANCE */}
              <div>
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    marginBottom: 6,
                  }}
                >
                  <span
                    style={{
                      fontSize: '0.75rem',
                      fontWeight: 700,
                      color: 'var(--text-muted)',
                      textTransform: 'uppercase',
                      letterSpacing: '0.04em',
                    }}
                  >
                    AI Suggested Answer / Guidance
                  </span>
                  {!isEditing && (
                    <button
                      onClick={() => setIsEditing(true)}
                      style={{
                        fontSize: '0.72rem',
                        color: 'var(--blue-primary)',
                        background: 'none',
                        border: 'none',
                        cursor: 'pointer',
                        padding: 0,
                        textDecoration: 'underline',
                      }}
                    >
                      ✎ Edit draft
                    </button>
                  )}
                </div>

                {isEditing ? (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                    <textarea
                      value={editedResponseText}
                      onChange={(e) => setEditedResponseText(e.target.value)}
                      rows={4}
                      style={{
                        width: '100%',
                        padding: '0.75rem 1rem',
                        fontSize: '0.88rem',
                        lineHeight: 1.6,
                        borderRadius: 'var(--radius-s)',
                        border: '1px solid var(--blue-border)',
                        background: '#ffffff',
                        color: 'var(--text-primary)',
                        fontFamily: 'inherit',
                      }}
                    />
                    <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
                      <button
                        onClick={() => {
                          setIsEditing(false)
                          setEditedResponseText(activeCase.ai_suggested_response || '')
                        }}
                        className="btn btn-secondary"
                        style={{ fontSize: '0.75rem', padding: '4px 10px' }}
                      >
                        Cancel
                      </button>
                      <button
                        onClick={handleEditSubmit}
                        className="btn btn-primary"
                        style={{ fontSize: '0.75rem', padding: '4px 12px' }}
                        disabled={submittingAction}
                      >
                        {submittingAction ? 'Submitting...' : 'Submit Edited Response'}
                      </button>
                    </div>
                  </div>
                ) : (
                  <div
                    style={{
                      background: '#ffffff',
                      border: '1px solid var(--border)',
                      borderRadius: 'var(--radius-s)',
                      padding: '1rem',
                      fontSize: '0.9rem',
                      color: 'var(--text-secondary)',
                      lineHeight: 1.6,
                    }}
                  >
                    {activeCase.ai_suggested_response || 'No automated response drafted. Specialist review required.'}
                  </div>
                )}
              </div>

              {/* 5. RELATED HISTORICAL PROBLEMS */}
              <div>
                <div
                  style={{
                    fontSize: '0.75rem',
                    fontWeight: 700,
                    color: 'var(--text-muted)',
                    marginBottom: 8,
                    textTransform: 'uppercase',
                    letterSpacing: '0.04em',
                  }}
                >
                  Related Historical Problems & Historical Resolution
                </div>

                {activeCase.related_historical_cases.length === 0 ? (
                  <div
                    style={{
                      padding: '1rem',
                      background: '#f8fafc',
                      border: '1px solid var(--border)',
                      borderRadius: 'var(--radius-s)',
                      fontSize: '0.84rem',
                      color: 'var(--text-muted)',
                    }}
                  >
                    No historical precedent matches above similarity threshold in corpus.
                  </div>
                ) : (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                    {activeCase.related_historical_cases.map((rc, i) => {
                      const customerMsg =
                        rc.historical_customer_message || rc.customer_query || 'Customer reported similar problem.'
                      const brandMsg =
                        rc.historical_brand_response || rc.brand_response || 'Standard AppleSupport resolution.'
                      const similarity =
                        rc.operational_similarity ?? rc.similarity_score ?? rc.relevance_score

                      return (
                        <div
                          key={i}
                          style={{
                            background: '#f8fafc',
                            border: '1px solid var(--border)',
                            borderRadius: 'var(--radius-s)',
                            padding: '0.85rem 1rem',
                          }}
                        >
                          <div
                            style={{
                              display: 'flex',
                              justifyContent: 'space-between',
                              alignItems: 'center',
                              marginBottom: 6,
                              flexWrap: 'wrap',
                              gap: 6,
                            }}
                          >
                            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                              {renderTierBadge(rc.match_tier)}
                              {similarity !== undefined && (
                                <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                                  {(similarity * 100).toFixed(0)}% match
                                </span>
                              )}
                            </div>
                            {rc.case_id && (
                              <span className="mono" style={{ fontSize: '0.7rem', color: 'var(--text-subtle)' }}>
                                {rc.case_id}
                              </span>
                            )}
                          </div>

                          <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginBottom: 6 }}>
                            <strong style={{ fontSize: '0.7rem', color: 'var(--text-muted)', display: 'block' }}>
                              HISTORICAL PROBLEM:
                            </strong>
                            "{customerMsg}"
                          </div>

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
                            <strong style={{ fontSize: '0.7rem', display: 'block', marginBottom: 2 }}>
                              HISTORICAL BRAND RESOLUTION:
                            </strong>
                            {brandMsg}
                          </div>
                        </div>
                      )
                    })}
                  </div>
                )}
              </div>

              {/* 6. SPECIALIST ACTION CONTROLS */}
              <div
                style={{
                  paddingTop: '1rem',
                  borderTop: '1px solid var(--border)',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '0.75rem',
                }}
              >
                <div>
                  <label
                    style={{
                      fontSize: '0.75rem',
                      fontWeight: 700,
                      color: 'var(--text-muted)',
                      textTransform: 'uppercase',
                      letterSpacing: '0.04em',
                      display: 'block',
                      marginBottom: 4,
                    }}
                  >
                    Specialist Adjudication Notes (Optional)
                  </label>
                  <input
                    type="text"
                    placeholder="Enter review reason or internal audit note..."
                    value={specialistNotes}
                    onChange={(e) => setSpecialistNotes(e.target.value)}
                    style={{
                      width: '100%',
                      padding: '8px 12px',
                      fontSize: '0.84rem',
                      border: '1px solid var(--border)',
                      borderRadius: 'var(--radius-s)',
                      background: '#ffffff',
                    }}
                  />
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
                  <button
                    onClick={handleApprove}
                    disabled={submittingAction}
                    className="btn btn-primary"
                    style={{
                      fontSize: '0.85rem',
                      padding: '8px 16px',
                      background: '#16a34a',
                      borderColor: '#15803d',
                    }}
                  >
                    ✓ Approve Response
                  </button>

                  <button
                    onClick={() => setIsEditing(true)}
                    disabled={submittingAction}
                    className="btn btn-secondary"
                    style={{ fontSize: '0.85rem', padding: '8px 16px' }}
                  >
                    ✎ Edit Response
                  </button>

                  <button
                    onClick={handleEscalateTier2}
                    disabled={submittingAction}
                    className="btn btn-secondary"
                    style={{
                      fontSize: '0.85rem',
                      padding: '8px 16px',
                      color: '#b91c1c',
                      borderColor: '#fca5a5',
                    }}
                  >
                    ↑ Escalate to Tier 2
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
