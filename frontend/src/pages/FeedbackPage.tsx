/**
 * SupportGraph AI — Human Feedback & Promotion Pipeline Page (Phase 13)
 *
 * Exposes specialist candidate resolutions, 10-check validation gate,
 * offline simulation evaluation, and promotion to trusted evidence.
 */

import React, { useState, useEffect, useCallback } from 'react'
import { api } from '../services/api'
import type { HumanResolution, ReviewDecision } from '../types'

export const FeedbackPage: React.FC = () => {
  const [candidates, setCandidates] = useState<HumanResolution[]>([])
  const [metrics, setMetrics] = useState<Record<string, any> | null>(null)
  const [loading, setLoading] = useState(true)
  const [activeResId, setActiveResId] = useState<string | null>(null)
  const [validationResult, setValidationResult] = useState<ReviewDecision | null>(null)
  const [actionLoading, setActionLoading] = useState(false)
  const [actionMessage, setActionMessage] = useState<string | null>(null)

  const loadData = useCallback(async () => {
    setLoading(true)
    try {
      const [candList, fbMetrics] = await Promise.all([
        api.listCandidateResolutions(),
        api.getFeedbackMetrics(),
      ])
      setCandidates(candList || [])
      setMetrics(fbMetrics || null)
    } catch {
      // ignore
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    loadData()
  }, [loadData])

  const handleValidate = async (resId: string) => {
    setActiveResId(resId)
    setActionLoading(true)
    setActionMessage(null)
    try {
      const dec = await api.validateResolution(resId)
      setValidationResult(dec)
    } catch (err: any) {
      setActionMessage(`Validation error: ${err.message}`)
    } finally {
      setActionLoading(false)
    }
  }

  const handleApproveAndPromote = async (resId: string) => {
    setActionLoading(true)
    setActionMessage(null)
    try {
      await api.approveResolution(resId, 'specialist_lead')
      const item = await api.promoteResolution(resId, 'specialist_lead')
      setActionMessage(`Successfully promoted resolution to Approved Store as evidence ID: ${item.evidence_id}`)
      loadData()
    } catch (err: any) {
      setActionMessage(`Promotion error: ${err.message}`)
    } finally {
      setActionLoading(false)
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      <div>
        <h2 style={{ fontSize: '1.1rem', fontWeight: 700 }}>Human-in-the-Loop Feedback & Promotion</h2>
        <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
          Continuous evidence improvement with 10-check deterministic validation and offline simulation
        </div>
      </div>

      {loading ? (
        <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>
          <span className="spinner" /> Loading feedback pipeline...
        </div>
      ) : (
        <>
          {/* Top Metrics */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
            <div className="sg-card">
              <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontFamily: 'var(--mono)' }}>TOTAL FEEDBACK EVENTS</div>
              <div style={{ fontSize: '1.4rem', fontWeight: 700, marginTop: 4 }}>{metrics?.total_events ?? 0}</div>
            </div>
            <div className="sg-card">
              <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontFamily: 'var(--mono)' }}>PROMOTIONS</div>
              <div style={{ fontSize: '1.4rem', fontWeight: 700, color: 'var(--green-text)', marginTop: 4 }}>
                {metrics?.promotions ?? 0}
              </div>
            </div>
            <div className="sg-card">
              <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontFamily: 'var(--mono)' }}>PROMOTION BLOCKED</div>
              <div style={{ fontSize: '1.4rem', fontWeight: 700, color: 'var(--amber-text)', marginTop: 4 }}>
                {metrics?.promotion_blocked ?? 0}
              </div>
            </div>
          </div>

          {actionMessage && (
            <div style={{ background: 'var(--blue-bg)', color: 'var(--blue-text)', border: '1px solid var(--blue-border)', padding: '0.75rem 1rem', borderRadius: 'var(--radius-s)', fontSize: '0.85rem' }}>
              {actionMessage}
            </div>
          )}

          {/* Candidate Resolutions List */}
          <div className="sg-card">
            <div className="sg-card-header">
              <span className="sg-section-title">Specialist Resolution Queue</span>
              <span className="tag tag-neutral">{candidates.length} Candidates</span>
            </div>

            {candidates.length === 0 ? (
              <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                No candidate resolutions in queue. Resolutions captured from escalated interactions appear here.
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                {candidates.map((cand) => (
                  <div
                    key={cand.resolution_id}
                    style={{
                      border: '1px solid var(--border)',
                      borderRadius: 'var(--radius-s)',
                      padding: '1rem',
                      background: 'var(--bg-subtle)',
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6, flexWrap: 'wrap', gap: 6 }}>
                      <div>
                        <strong style={{ fontSize: '0.88rem' }}>{cand.problem_family}</strong>
                        <span className="mono" style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginLeft: 8 }}>
                          {cand.resolution_id}
                        </span>
                      </div>
                      <span className="tag tag-blue">{cand.status}</span>
                    </div>

                    <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginBottom: 6 }}>
                      <strong>Customer Query:</strong> "{cand.customer_query}"
                    </div>
                    <div style={{ fontSize: '0.82rem', color: 'var(--text-primary)', background: '#ffffff', padding: '8px 12px', borderRadius: 'var(--radius-xs)', border: '1px solid var(--border)', marginBottom: 10 }}>
                      <strong>Specialist Resolution:</strong> {cand.resolution_text}
                    </div>

                    <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
                      <button
                        className="btn btn-secondary"
                        onClick={() => handleValidate(cand.resolution_id)}
                        disabled={actionLoading}
                        style={{ padding: '4px 12px', fontSize: '0.78rem' }}
                      >
                        Run 10-Check Gate
                      </button>
                      <button
                        className="btn btn-primary"
                        onClick={() => handleApproveAndPromote(cand.resolution_id)}
                        disabled={actionLoading}
                        style={{ padding: '4px 12px', fontSize: '0.78rem' }}
                      >
                        Approve & Promote
                      </button>
                    </div>

                    {/* Validation Gate Inspector */}
                    {activeResId === cand.resolution_id && validationResult && (
                      <div style={{ marginTop: 12, padding: '10px 14px', background: '#ffffff', borderRadius: 'var(--radius-s)', border: '1px solid var(--border)' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                          <span style={{ fontSize: '0.75rem', fontWeight: 700 }}>VALIDATION GATE RESULT</span>
                          <span className={validationResult.passed_validation_gate ? 'tag tag-green' : 'tag tag-red'}>
                            {validationResult.decision} · Quality Score: {validationResult.quality_score?.toFixed(2)}
                          </span>
                        </div>
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 4 }}>
                          {Object.entries(validationResult.validation_checks || {}).map(([chk, passed]) => (
                            <div key={chk} style={{ fontSize: '0.72rem', color: passed ? 'var(--green-text)' : 'var(--red-text)' }}>
                              {passed ? '✓' : '✕'} {chk}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  )
}
