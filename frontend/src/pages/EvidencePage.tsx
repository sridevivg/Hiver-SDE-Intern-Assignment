/**
 * SupportGraph AI — Evidence & Corpus Browser Page
 *
 * Exposes the immutable Golden Benchmark, Approved Evidence Store,
 * and live historical AppleSupport retrieval search.
 */

import React, { useState, useEffect } from 'react'
import { api } from '../services/api'
import type { ApprovedEvidenceItem, RetrievedEvidenceCase } from '../types'

export const EvidencePage: React.FC = () => {
  const [storesData, setStoresData] = useState<Record<string, any> | null>(null)
  const [approvedItems, setApprovedItems] = useState<ApprovedEvidenceItem[]>([])
  const [loading, setLoading] = useState(true)

  // Live Retriever Tester
  const [searchQuery, setSearchQuery] = useState('')
  const [searchLoading, setSearchLoading] = useState(false)
  const [searchResults, setSearchResults] = useState<RetrievedEvidenceCase[]>([])
  const [searchError, setSearchError] = useState<string | null>(null)

  useEffect(() => {
    Promise.all([api.getEvidenceStores(), api.listApprovedEvidence()])
      .then(([stores, approved]) => {
        setStoresData(stores)
        setApprovedItems(approved || [])
      })
      .catch(() => {
        setStoresData(null)
      })
      .finally(() => setLoading(false))
  }, [])

  const handleSearch = async () => {
    if (!searchQuery.trim()) return
    setSearchLoading(true)
    setSearchError(null)
    try {
      const results = await api.retrieveEvidence(searchQuery.trim(), '', 5)
      setSearchResults(results)
    } catch (err: any) {
      setSearchError(err.message || 'Failed to retrieve evidence.')
    } finally {
      setSearchLoading(false)
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      <div>
        <h2 style={{ fontSize: '1.1rem', fontWeight: 700 }}>Historical Support Evidence & Provenance</h2>
        <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
          Strictly segregated stores: Immutable Golden Benchmark, Approved Evidence, and Candidate Store
        </div>
      </div>

      {loading ? (
        <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>
          <span className="spinner" /> Loading evidence stores...
        </div>
      ) : (
        <>
          {/* Store Statistics Cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '1rem' }}>
            <div className="sg-card">
              <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontFamily: 'var(--mono)', marginBottom: 4 }}>
                GOLDEN BENCHMARK
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, margin: '6px 0' }}>
                <span className="tag tag-green">IMMUTABLE</span>
                <span style={{ fontSize: '1.1rem', fontWeight: 700 }}>50 Cases</span>
              </div>
              <div className="mono" style={{ fontSize: '0.68rem', color: 'var(--text-muted)', wordBreak: 'break-all' }}>
                SHA: {storesData?.golden_benchmark?.sha256?.substring(0, 24)}...
              </div>
            </div>

            <div className="sg-card">
              <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontFamily: 'var(--mono)', marginBottom: 4 }}>
                APPROVED EVIDENCE STORE
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, margin: '6px 0' }}>
                <span className="tag tag-blue">RETRIEVABLE</span>
                <span style={{ fontSize: '1.1rem', fontWeight: 700 }}>
                  {storesData?.evidence_stores?.APPROVED_STORE?.count ?? approvedItems.length} Items
                </span>
              </div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                Validated via 10-check gate & authorized for grounding
              </div>
            </div>

            <div className="sg-card">
              <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontFamily: 'var(--mono)', marginBottom: 4 }}>
                CANDIDATE STORE
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, margin: '6px 0' }}>
                <span className="tag tag-amber">ISOLATED</span>
                <span style={{ fontSize: '1.1rem', fontWeight: 700 }}>
                  {storesData?.evidence_stores?.CANDIDATE_STORE?.count ?? 0} Pending
                </span>
              </div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                Zero retrieval access until verified & promoted
              </div>
            </div>
          </div>

          {/* Live Evidence Retrieval Tool */}
          <div className="sg-card">
            <div className="sg-card-header">
              <span className="sg-section-title">Live Historical Evidence Index Search</span>
              <span className="tag tag-blue">AppleSupport Corpus</span>
            </div>

            <div style={{ display: 'flex', gap: 8, marginBottom: '1rem' }}>
              <input
                type="text"
                className="textarea-custom"
                style={{ padding: '8px 12px', minHeight: 'unset', resize: 'none' }}
                placeholder="Enter a symptom or issue (e.g. 'iPhone battery draining fast after update')..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
              />
              <button
                className="btn btn-primary"
                onClick={handleSearch}
                disabled={searchLoading || !searchQuery.trim()}
                style={{ minWidth: 120 }}
              >
                {searchLoading ? <span className="spinner" /> : 'Search'}
              </button>
            </div>

            {searchError && (
              <div style={{ color: 'var(--red-text)', fontSize: '0.85rem', marginBottom: 12 }}>
                {searchError}
              </div>
            )}

            {searchResults.length > 0 && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {searchResults.map((item, idx) => (
                  <div
                    key={idx}
                    style={{
                      background: 'var(--bg-subtle)',
                      border: '1px solid var(--border)',
                      borderRadius: 'var(--radius-s)',
                      padding: '0.85rem',
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                      <span className="tag tag-green">{item.match_tier}</span>
                      <span className="mono" style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                        Relevance: <strong style={{ color: 'var(--blue-primary)' }}>{item.relevance_score?.toFixed(3)}</strong>
                      </span>
                    </div>
                    <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginBottom: 6 }}>
                      <strong>Customer:</strong> "{item.customer_query}"
                    </div>
                    <div style={{ fontSize: '0.82rem', color: 'var(--blue-text)', background: 'var(--blue-bg)', padding: '6px 10px', borderRadius: 'var(--radius-xs)' }}>
                      <strong>AppleSupport:</strong> {item.brand_response}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Approved Evidence Table */}
          <div className="sg-card">
            <div className="sg-card-header">
              <span className="sg-section-title">Approved Evidence Items for Grounding</span>
              <span className="tag tag-green">{approvedItems.length} Promoted</span>
            </div>

            {approvedItems.length === 0 ? (
              <div style={{ padding: '1.5rem', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                No specialist-promoted evidence items in Approved Store yet. Promote candidate resolutions on the Feedback tab.
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {approvedItems.map((item) => (
                  <div
                    key={item.evidence_id}
                    style={{
                      padding: '10px 14px',
                      background: 'var(--bg-subtle)',
                      border: '1px solid var(--border)',
                      borderRadius: 'var(--radius-s)',
                      fontSize: '0.82rem',
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                      <span className="mono" style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                        {item.evidence_id} · <span style={{ color: 'var(--blue-primary)' }}>{item.problem_family}</span>
                      </span>
                      <div style={{ display: 'flex', gap: 6 }}>
                        <span className="tag tag-green">Score: {item.quality_score?.toFixed(2)}</span>
                        <span className="tag tag-neutral">v{item.version}</span>
                      </div>
                    </div>
                    <div style={{ color: 'var(--text-secondary)', marginTop: 4 }}>
                      {item.resolution_text}
                    </div>
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
