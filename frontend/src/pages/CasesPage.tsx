/**
 * SupportGraph AI — Decision Cases & Traces Page
 *
 * Exposes runtime audit decision cases with full 12-step structured tracing.
 */

import React, { useState, useEffect, useCallback } from 'react'
import { api } from '../services/api'
import type { DecisionLogRecord, DecisionTraceRecord } from '../types'

const STEP_LABELS: Record<string, string> = {
  customer_message: 'Customer Message',
  problem_understanding: 'Problem Understanding',
  intent_candidates: 'Intent Candidates',
  primary_problem_selection: 'Primary Problem Selection',
  ambiguity_analysis: 'Ambiguity Analysis',
  safety_gate: 'Safety Gate',
  evidence_retrieval: 'Evidence Retrieval',
  evidence_validation: 'Evidence Validation',
  conflict_check: 'Conflict Check',
  response_generation: 'Response Generation',
  grounding_verification: 'Grounding Verification',
  final_decision: 'Final Decision',
}

function stepColor(status: string) {
  if (['ok', 'received', 'PASS', 'AUTO_HANDLE', 'no_conflict', 'no_evidence'].includes(status)) return '#16a34a'
  if (['FAIL', 'conflict_detected', 'fallback', 'veto_triggered'].includes(status)) return '#dc2626'
  if (['ESCALATE_TO_HUMAN', 'human_required'].includes(status)) return '#d97706'
  return '#2563eb'
}

export const CasesPage: React.FC = () => {
  const [rows, setRows] = useState<DecisionLogRecord[]>([])
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState('')
  const [selectedCaseId, setSelectedCaseId] = useState<string | null>(null)
  const [trace, setTrace] = useState<DecisionTraceRecord | null>(null)
  const [traceLoading, setTraceLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const loadCases = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await api.getDecisions(filter, 50)
      setRows(res.decisions || [])
    } catch (err: any) {
      setError(err.message || 'Failed to load cases.')
    } finally {
      setLoading(false)
    }
  }, [filter])

  useEffect(() => {
    loadCases()
  }, [loadCases])

  const openTrace = async (caseId: string) => {
    setSelectedCaseId(caseId)
    setTraceLoading(true)
    try {
      const res = await api.getDecisionTrace(caseId)
      setTrace(res)
    } catch {
      setTrace(null)
    } finally {
      setTraceLoading(false)
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <h2 style={{ fontSize: '1.1rem', fontWeight: 700 }}>Resolution Decision Audit Log</h2>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            Real-time append-only decision records and end-to-end pipeline traces
          </div>
        </div>

        {/* Filter Pills */}
        <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
          {['', 'AUTO_HANDLE', 'ESCALATE_TO_HUMAN'].map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              style={{
                padding: '5px 12px',
                borderRadius: 'var(--radius-s)',
                border: '1px solid',
                borderColor: filter === f ? 'var(--blue-border)' : 'var(--border)',
                background: filter === f ? 'var(--blue-bg)' : 'var(--surface)',
                color: filter === f ? 'var(--blue-text)' : 'var(--text-secondary)',
                cursor: 'pointer',
                fontSize: '0.78rem',
                fontWeight: filter === f ? 600 : 400,
              }}
            >
              {f || 'ALL DECISIONS'}
            </button>
          ))}
          <button className="btn btn-secondary" onClick={loadCases} style={{ padding: '4px 10px', fontSize: '0.75rem' }}>
            Refresh
          </button>
        </div>
      </div>

      {error && (
        <div style={{ background: 'var(--red-bg)', color: 'var(--red-text)', padding: '0.75rem 1rem', borderRadius: 'var(--radius-s)' }}>
          {error}
        </div>
      )}

      {/* Main Grid: Cases Table + Side Trace Viewer */}
      <div style={{ display: 'grid', gridTemplateColumns: selectedCaseId ? '1fr 420px' : '1fr', gap: '1.25rem' }}>
        {/* Table / List */}
        <div className="sg-card" style={{ padding: 0, overflow: 'hidden' }}>
          {loading ? (
            <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>
              <span className="spinner" /> Loading runtime decisions...
            </div>
          ) : rows.length === 0 ? (
            <div style={{ padding: '2.5rem', textAlign: 'center', color: 'var(--text-muted)' }}>
              No decisions recorded yet in runtime audit log. Process inquiries on the Support Agent page to populate.
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column' }}>
              {/* Header */}
              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: '130px 140px 140px 1fr 100px',
                  padding: '10px 16px',
                  background: 'var(--bg-subtle)',
                  borderBottom: '1px solid var(--border)',
                  fontSize: '0.72rem',
                  fontWeight: 700,
                  color: 'var(--text-muted)',
                  letterSpacing: '0.04em',
                }}
              >
                <span>CASE ID</span>
                <span>DECISION</span>
                <span>OUTCOME</span>
                <span>PROBLEM FAMILY</span>
                <span style={{ textAlign: 'right' }}>LATENCY</span>
              </div>

              {/* Rows */}
              {rows.map((row) => {
                const isSelected = selectedCaseId === row.case_id
                const isAuto = row.routing_decision === 'AUTO_HANDLE'
                return (
                  <div
                    key={row.case_id}
                    onClick={() => openTrace(row.case_id)}
                    style={{
                      display: 'grid',
                      gridTemplateColumns: '130px 140px 140px 1fr 100px',
                      padding: '12px 16px',
                      alignItems: 'center',
                      borderBottom: '1px solid var(--border)',
                      background: isSelected ? 'var(--blue-bg)' : 'transparent',
                      cursor: 'pointer',
                      fontSize: '0.82rem',
                      transition: 'background 0.15s ease',
                    }}
                  >
                    <span className="mono" style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                      {row.case_id}
                    </span>
                    <div>
                      <span className={`tag ${isAuto ? 'tag-green' : 'tag-amber'}`}>
                        {row.routing_decision}
                      </span>
                    </div>
                    <div>
                      <span className="tag tag-neutral" style={{ fontSize: '0.68rem' }}>
                        {row.outcome || '—'}
                      </span>
                    </div>
                    <span style={{ color: 'var(--text-primary)', fontWeight: 500 }}>
                      {row.problem_family || '—'}
                    </span>
                    <span className="mono" style={{ textAlign: 'right', color: 'var(--text-muted)', fontSize: '0.75rem' }}>
                      {row.total_latency_ms != null ? `${row.total_latency_ms.toFixed(0)} ms` : '—'}
                    </span>
                  </div>
                )
              })}
            </div>
          )}
        </div>

        {/* Trace Inspector */}
        {selectedCaseId && (
          <div className="sg-card" style={{ maxHeight: 'calc(100vh - 180px)', overflowY: 'auto' }}>
            <div className="sg-card-header">
              <div>
                <span className="sg-section-title">Step-by-Step Decision Trace</span>
                <div className="mono" style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 2 }}>
                  {selectedCaseId}
                </div>
              </div>
              <button
                onClick={() => {
                  setSelectedCaseId(null)
                  setTrace(null)
                }}
                style={{ background: 'none', border: 'none', cursor: 'pointer', fontSize: '1rem', color: 'var(--text-muted)' }}
              >
                ✕
              </button>
            </div>

            {traceLoading ? (
              <div style={{ padding: '1.5rem', textAlign: 'center', color: 'var(--text-muted)' }}>
                <span className="spinner" /> Loading trace steps...
              </div>
            ) : !trace ? (
              <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>Trace details not found.</div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {trace.steps?.map((step, idx) => {
                  const label = STEP_LABELS[step.step] || step.step
                  const col = stepColor(step.status)
                  return (
                    <div
                      key={idx}
                      style={{
                        padding: '8px 12px',
                        borderRadius: 'var(--radius-s)',
                        background: 'var(--bg-subtle)',
                        borderLeft: `4px solid ${col}`,
                        fontSize: '0.8rem',
                      }}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                        <strong style={{ color: 'var(--text-primary)' }}>{label}</strong>
                        <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                          <span className="mono" style={{ color: col, fontSize: '0.72rem', fontWeight: 600 }}>
                            {step.status}
                          </span>
                          {step.latency_ms > 0 && (
                            <span className="mono" style={{ color: 'var(--text-muted)', fontSize: '0.68rem' }}>
                              {step.latency_ms.toFixed(0)}ms
                            </span>
                          )}
                        </div>
                      </div>
                      {Object.entries(step.data || {})
                        .filter(([, v]) => v != null && v !== '' && v !== false)
                        .map(([k, v]) => (
                          <div key={k} className="mono" style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: 2 }}>
                            <span style={{ color: 'var(--text-secondary)' }}>{k}: </span>
                            <span>{typeof v === 'object' ? JSON.stringify(v) : String(v)}</span>
                          </div>
                        ))}
                    </div>
                  )
                })}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
