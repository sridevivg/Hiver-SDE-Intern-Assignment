/**
 * SupportGraph AI — Customer Support Product Application Shell
 *
 * ONLY TWO USER-FACING SECTIONS:
 * 1. Support (Default customer chatbot UX with related historical problems)
 * 2. Human Review (Internal support specialist adjudication view)
 */

import React, { useState } from 'react'
import { SupportPage } from './pages/SupportPage'
import { HumanReviewPage } from './pages/HumanReviewPage'

type TabType = 'support' | 'review'

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabType>('support')

  return (
    <div
      style={{
        minHeight: '100vh',
        background: '#f8fafc',
        color: '#0f172a',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      {/* Product Header */}
      <header
        style={{
          background: '#ffffff',
          borderBottom: '1px solid #e2e8f0',
          position: 'sticky',
          top: 0,
          zIndex: 100,
          boxShadow: '0 1px 2px 0 rgba(0, 0, 0, 0.03)',
        }}
      >
        <div
          style={{
            maxWidth: 1060,
            margin: '0 auto',
            padding: '0 1.5rem',
            height: 60,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}
        >
          {/* Brand Logo & Product Name */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '1.25rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <div
                style={{
                  width: 32,
                  height: 32,
                  borderRadius: 8,
                  background: '#2563eb',
                  color: '#ffffff',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontWeight: 700,
                  fontSize: '0.95rem',
                  letterSpacing: '-0.02em',
                }}
              >
                SG
              </div>
              <div style={{ fontWeight: 700, fontSize: '1rem', letterSpacing: '-0.02em', color: '#0f172a' }}>
                SupportGraph AI
              </div>
            </div>

            <div style={{ height: 20, width: 1, background: '#e2e8f0' }} />

            {/* Main Navigation: ONLY Support & Human Review */}
            <nav style={{ display: 'flex', gap: 4 }}>
              <button
                id="nav-tab-support"
                onClick={() => setActiveTab('support')}
                style={{
                  padding: '6px 14px',
                  borderRadius: 6,
                  border: 'none',
                  background: activeTab === 'support' ? '#eff6ff' : 'transparent',
                  color: activeTab === 'support' ? '#1d4ed8' : '#475569',
                  fontSize: '0.88rem',
                  fontWeight: activeTab === 'support' ? 600 : 500,
                  cursor: 'pointer',
                  transition: 'all 0.15s ease',
                }}
              >
                Support
              </button>
              <button
                id="nav-tab-review"
                onClick={() => setActiveTab('review')}
                style={{
                  padding: '6px 14px',
                  borderRadius: 6,
                  border: 'none',
                  background: activeTab === 'review' ? '#eff6ff' : 'transparent',
                  color: activeTab === 'review' ? '#1d4ed8' : '#475569',
                  fontSize: '0.88rem',
                  fontWeight: activeTab === 'review' ? 600 : 500,
                  cursor: 'pointer',
                  transition: 'all 0.15s ease',
                }}
              >
                Human Review
              </button>
            </nav>
          </div>

          {/* Right Brand Indicator */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span
              style={{
                fontSize: '0.82rem',
                fontWeight: 600,
                color: '#334155',
                display: 'flex',
                alignItems: 'center',
                gap: 5,
              }}
            >
              <span style={{ fontSize: '0.95rem' }}></span> AppleSupport
            </span>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main
        style={{
          maxWidth: 1060,
          width: '100%',
          margin: '0 auto',
          padding: '1.5rem 1.5rem',
          flex: 1,
        }}
      >
        {activeTab === 'support' && <SupportPage />}
        {activeTab === 'review' && <HumanReviewPage />}
      </main>

      {/* Subtle Footer */}
      <footer
        style={{
          borderTop: '1px solid #e2e8f0',
          background: '#ffffff',
          padding: '1rem 1.5rem',
          textAlign: 'center',
          fontSize: '0.75rem',
          color: '#64748b',
        }}
      >
        SupportGraph AI · AppleSupport Resolution Assistant
      </footer>
    </div>
  )
}

export default App
