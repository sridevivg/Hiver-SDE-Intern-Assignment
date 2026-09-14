import type { FC } from 'react'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
interface PhaseInfo {
  id: number
  name: string
  status: 'complete' | 'active' | 'planned'
}

// ---------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------
const PHASES: PhaseInfo[] = [
  { id: 0, name: 'Project Foundation', status: 'complete' },
  { id: 1, name: 'Dataset Exploration', status: 'active' },
  { id: 2, name: 'Brand Selection', status: 'planned' },
  { id: 3, name: 'Conversation Reconstruction', status: 'planned' },
  { id: 4, name: 'Data-Driven Intent Discovery', status: 'planned' },
  { id: 5, name: 'Golden Evaluation Set', status: 'planned' },
  { id: 6, name: 'Baselines', status: 'planned' },
  { id: 7, name: 'LangGraph Agent System', status: 'planned' },
  { id: 8, name: 'Historical RAG', status: 'planned' },
  { id: 9, name: 'Escalation Policy', status: 'planned' },
  { id: 10, name: 'Evaluation Harness', status: 'planned' },
  { id: 11, name: 'LLM-as-a-Judge', status: 'planned' },
  { id: 12, name: 'Failure Analysis', status: 'planned' },
  { id: 13, name: 'Frontend Dashboard', status: 'planned' },
  { id: 14, name: 'Final Report', status: 'planned' },
]

const CAPABILITIES = [
  {
    icon: '🔍',
    title: 'Intent Classification',
    description: 'Data-driven discovery of customer intent clusters from raw, noisy support conversations.',
    status: 'PLANNED',
  },
  {
    icon: '📚',
    title: 'Historical RAG',
    description: 'Retrieve historically similar resolutions to ground generated replies in real agent behavior.',
    status: 'PLANNED',
  },
  {
    icon: '🤖',
    title: 'Agentic Orchestration',
    description: 'LangGraph-powered agent graph that routes, generates, and decides escalation.',
    status: 'PLANNED',
  },
  {
    icon: '📊',
    title: 'Rigorous Evaluation',
    description: 'Automated metrics, LLM-as-a-judge scoring, and a human-curated golden test set.',
    status: 'PLANNED',
  },
]

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------
const StatusBadge: FC<{ status: PhaseInfo['status'] }> = ({ status }) => {
  const styles: Record<PhaseInfo['status'], React.CSSProperties> = {
    complete: {
      background: 'rgba(63, 185, 80, 0.15)',
      border: '1px solid rgba(63, 185, 80, 0.4)',
      color: '#3fb950',
    },
    active: {
      background: 'rgba(88, 166, 255, 0.15)',
      border: '1px solid rgba(88, 166, 255, 0.4)',
      color: '#58a6ff',
    },
    planned: {
      background: 'rgba(139, 148, 158, 0.1)',
      border: '1px solid rgba(139, 148, 158, 0.2)',
      color: '#8b949e',
    },
  }

  const labels: Record<PhaseInfo['status'], string> = {
    complete: '✓ Done',
    active:   '◉ Active',
    planned:  '○ Planned',
  }

  return (
    <span
      style={{
        ...styles[status],
        fontSize: '0.7rem',
        fontWeight: 600,
        padding: '2px 8px',
        borderRadius: '20px',
        letterSpacing: '0.05em',
        fontFamily: 'var(--font-mono)',
        whiteSpace: 'nowrap',
      }}
    >
      {labels[status]}
    </span>
  )
}

// ---------------------------------------------------------------------------
// Main App
// ---------------------------------------------------------------------------
const App: FC = () => {
  return (
    <div style={{ minHeight: '100vh', background: 'var(--color-bg)', color: 'var(--color-text)' }}>

      {/* ------------------------------------------------------------------ */}
      {/* Header / Nav                                                        */}
      {/* ------------------------------------------------------------------ */}
      <header
        style={{
          borderBottom: '1px solid var(--color-border)',
          background: 'rgba(13, 17, 23, 0.8)',
          backdropFilter: 'blur(12px)',
          position: 'sticky',
          top: 0,
          zIndex: 100,
        }}
      >
        <div
          style={{
            maxWidth: '1100px',
            margin: '0 auto',
            padding: '0 1.5rem',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            height: '56px',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <span style={{ fontSize: '1.25rem' }}>🧠</span>
            <span
              style={{
                fontWeight: 700,
                fontSize: '1rem',
                background: 'linear-gradient(90deg, #58a6ff, #3fb950)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                letterSpacing: '-0.02em',
              }}
            >
              SupportGraph AI
            </span>
          </div>
          <span
            style={{
              fontSize: '0.75rem',
              color: 'var(--color-text-muted)',
              fontFamily: 'var(--font-mono)',
            }}
          >
            v0.1.0 — Phase 1
          </span>
        </div>
      </header>

      {/* ------------------------------------------------------------------ */}
      {/* Hero                                                                */}
      {/* ------------------------------------------------------------------ */}
      <section
        style={{
          background: 'var(--gradient-hero)',
          position: 'relative',
          overflow: 'hidden',
          padding: '5rem 1.5rem',
          textAlign: 'center',
        }}
      >
        {/* Glow */}
        <div
          style={{
            position: 'absolute',
            inset: 0,
            background: 'var(--gradient-glow)',
            pointerEvents: 'none',
          }}
        />

        <div style={{ maxWidth: '760px', margin: '0 auto', position: 'relative' }}>
          {/* Phase tag */}
          <div
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '6px',
              background: 'var(--color-tag-bg)',
              border: '1px solid var(--color-tag-border)',
              borderRadius: '20px',
              padding: '4px 14px',
              fontSize: '0.78rem',
              color: 'var(--color-primary)',
              fontFamily: 'var(--font-mono)',
              marginBottom: '1.5rem',
              fontWeight: 500,
            }}
          >
            <span style={{ animation: 'pulse 2s infinite' }}>◉</span>
            <span>Current Phase: Dataset Exploration</span>
          </div>

          {/* Title */}
          <h1
            style={{
              fontSize: 'clamp(2rem, 5vw, 3.25rem)',
              fontWeight: 800,
              letterSpacing: '-0.04em',
              lineHeight: 1.1,
              marginBottom: '1rem',
              background: 'linear-gradient(135deg, #e6edf3 0%, #8b949e 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}
          >
            SupportGraph AI
          </h1>

          {/* Subtitle */}
          <p
            style={{
              fontSize: 'clamp(1rem, 2vw, 1.2rem)',
              color: 'var(--color-text-muted)',
              marginBottom: '2.5rem',
              lineHeight: 1.7,
            }}
          >
            Agentic Customer Support Intelligence System
          </p>

          {/* Active phase card */}
          <div
            style={{
              display: 'inline-block',
              background: 'var(--color-surface)',
              border: '1px solid var(--color-border)',
              borderRadius: 'var(--radius-lg)',
              padding: '1.5rem 2.5rem',
              textAlign: 'left',
            }}
          >
            <div style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)', marginBottom: '4px', fontFamily: 'var(--font-mono)' }}>
              CURRENT PHASE
            </div>
            <div style={{ fontSize: '1.4rem', fontWeight: 700, color: 'var(--color-primary)' }}>
              Dataset Exploration
            </div>
            <div style={{ fontSize: '0.85rem', color: 'var(--color-text-muted)', marginTop: '6px' }}>
              Phase 1 — Analyzing the Customer Support on Twitter dataset
            </div>
          </div>
        </div>
      </section>

      {/* ------------------------------------------------------------------ */}
      {/* Capabilities                                                        */}
      {/* ------------------------------------------------------------------ */}
      <section style={{ maxWidth: '1100px', margin: '0 auto', padding: '4rem 1.5rem' }}>
        <h2
          style={{
            fontSize: '1.4rem',
            fontWeight: 700,
            marginBottom: '1.75rem',
            color: 'var(--color-text)',
            letterSpacing: '-0.02em',
          }}
        >
          System Capabilities
        </h2>
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
            gap: '1rem',
          }}
        >
          {CAPABILITIES.map((cap) => (
            <div
              key={cap.title}
              id={`capability-${cap.title.toLowerCase().replace(/\s+/g, '-')}`}
              style={{
                background: 'var(--color-surface)',
                border: '1px solid var(--color-border)',
                borderRadius: 'var(--radius-md)',
                padding: '1.5rem',
                transition: 'border-color var(--transition)',
              }}
              onMouseEnter={(e) => {
                ;(e.currentTarget as HTMLDivElement).style.borderColor = 'var(--color-primary-dim)'
              }}
              onMouseLeave={(e) => {
                ;(e.currentTarget as HTMLDivElement).style.borderColor = 'var(--color-border)'
              }}
            >
              <div style={{ fontSize: '1.75rem', marginBottom: '0.75rem' }}>{cap.icon}</div>
              <div style={{ fontWeight: 600, fontSize: '0.95rem', marginBottom: '0.5rem' }}>
                {cap.title}
              </div>
              <p style={{ fontSize: '0.82rem', color: 'var(--color-text-muted)', lineHeight: 1.6, marginBottom: '0.75rem' }}>
                {cap.description}
              </p>
              <span
                style={{
                  fontSize: '0.68rem',
                  fontWeight: 600,
                  fontFamily: 'var(--font-mono)',
                  color: 'var(--color-text-muted)',
                  background: 'var(--color-surface-2)',
                  padding: '2px 6px',
                  borderRadius: '4px',
                  letterSpacing: '0.05em',
                }}
              >
                {cap.status}
              </span>
            </div>
          ))}
        </div>
      </section>

      {/* ------------------------------------------------------------------ */}
      {/* Phase roadmap                                                        */}
      {/* ------------------------------------------------------------------ */}
      <section
        style={{
          background: 'var(--color-surface)',
          borderTop: '1px solid var(--color-border)',
          borderBottom: '1px solid var(--color-border)',
        }}
      >
        <div style={{ maxWidth: '1100px', margin: '0 auto', padding: '4rem 1.5rem' }}>
          <h2
            style={{
              fontSize: '1.4rem',
              fontWeight: 700,
              marginBottom: '1.75rem',
              letterSpacing: '-0.02em',
            }}
          >
            Development Roadmap
          </h2>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            {PHASES.map((phase) => (
              <div
                key={phase.id}
                id={`phase-${phase.id}`}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: '0.75rem 1rem',
                  borderRadius: 'var(--radius-sm)',
                  background: phase.status === 'active'
                    ? 'rgba(88, 166, 255, 0.05)'
                    : 'transparent',
                  border: phase.status === 'active'
                    ? '1px solid rgba(88, 166, 255, 0.2)'
                    : '1px solid transparent',
                  transition: 'background var(--transition)',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                  <span
                    style={{
                      fontFamily: 'var(--font-mono)',
                      fontSize: '0.72rem',
                      color: 'var(--color-text-muted)',
                      minWidth: '60px',
                    }}
                  >
                    Phase {phase.id}
                  </span>
                  <span
                    style={{
                      fontSize: '0.9rem',
                      fontWeight: phase.status === 'active' ? 600 : 400,
                      color: phase.status === 'planned'
                        ? 'var(--color-text-muted)'
                        : 'var(--color-text)',
                    }}
                  >
                    {phase.name}
                  </span>
                </div>
                <StatusBadge status={phase.status} />
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ------------------------------------------------------------------ */}
      {/* Footer                                                              */}
      {/* ------------------------------------------------------------------ */}
      <footer
        style={{
          padding: '2rem 1.5rem',
          textAlign: 'center',
          color: 'var(--color-text-muted)',
          fontSize: '0.8rem',
          fontFamily: 'var(--font-mono)',
          borderTop: '1px solid var(--color-border)',
        }}
      >
        SupportGraph AI — Phase 1: Dataset Exploration
        <br />
        <span style={{ opacity: 0.5 }}>Built with React + TypeScript + Vite</span>
      </footer>

      {/* Pulse animation */}
      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50%       { opacity: 0.4; }
        }
      `}</style>
    </div>
  )
}

export default App
