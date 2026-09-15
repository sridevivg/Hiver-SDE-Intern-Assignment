#!/usr/bin/env python3
"""
Render professional README banner for SupportGraph AI using Headless Chrome.
Outputs docs/assets/supportgraph-banner.png.
"""

from pathlib import Path
import subprocess
import tempfile

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    width: 1200px;
    height: 500px;
    background: #f8fafc;
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #0f172a;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
  }
  .banner-container {
    width: 1160px;
    height: 460px;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 32px 40px;
    display: flex;
    gap: 36px;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.05);
    position: relative;
  }
  /* Left Column */
  .left-col {
    flex: 1.05;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
  }
  .badge-row {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .system-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #eff6ff;
    color: #1d4ed8;
    border: 1px solid #bfdbfe;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    padding: 4px 10px;
    border-radius: 6px;
  }
  .title-group h1 {
    font-size: 34px;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: #0f172a;
    line-height: 1.15;
    margin-top: 10px;
    margin-bottom: 8px;
  }
  .title-group p {
    font-size: 14px;
    line-height: 1.45;
    color: #475569;
    font-weight: 450;
  }
  .feature-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
    margin: 14px 0;
  }
  .feature-item {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 12.5px;
    color: #334155;
    font-weight: 500;
  }
  .icon-check {
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: #ecfdf5;
    color: #059669;
    border: 1px solid #a7f3d0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 11px;
    font-weight: bold;
    flex-shrink: 0;
  }
  .stats-bar {
    display: flex;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 10px 16px;
    justify-content: space-between;
  }
  .stat-unit {
    display: flex;
    flex-direction: column;
  }
  .stat-val {
    font-size: 17px;
    font-weight: 800;
    color: #0f172a;
    letter-spacing: -0.02em;
  }
  .stat-lbl {
    font-size: 10.5px;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.02em;
    font-weight: 600;
  }

  /* Right Column (Chat UI Mockup) */
  .right-col {
    flex: 1.35;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  .chat-header {
    background: #ffffff;
    border-bottom: 1px solid #e2e8f0;
    padding: 10px 16px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .chat-title {
    font-size: 12px;
    font-weight: 700;
    color: #1e293b;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .live-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #10b981;
  }
  .grounding-tag {
    font-size: 10.5px;
    font-weight: 600;
    background: #eff6ff;
    color: #2563eb;
    padding: 2px 8px;
    border-radius: 4px;
    border: 1px solid #dbeafe;
  }
  .chat-body {
    padding: 14px 16px;
    display: flex;
    flex-direction: column;
    gap: 12px;
    flex: 1;
    justify-content: center;
  }
  .user-bubble {
    align-self: flex-end;
    background: #2563eb;
    color: #ffffff;
    padding: 8px 12px;
    border-radius: 12px 12px 2px 12px;
    font-size: 12px;
    max-width: 90%;
    line-height: 1.4;
  }
  .agent-bubble {
    align-self: flex-start;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    color: #0f172a;
    padding: 10px 12px;
    border-radius: 12px 12px 12px 2px;
    font-size: 11.5px;
    max-width: 98%;
    line-height: 1.45;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
  }
  .action-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: #ecfdf5;
    color: #047857;
    border: 1px solid #a7f3d0;
    font-size: 9.5px;
    font-weight: 700;
    padding: 2px 6px;
    border-radius: 4px;
    margin-bottom: 6px;
  }
  .evidence-card {
    background: #f1f5f9;
    border-left: 3px solid #2563eb;
    border-radius: 4px;
    padding: 8px 10px;
    margin-top: 6px;
  }
  .evidence-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 10px;
    font-weight: 700;
    color: #1e3a8a;
    margin-bottom: 3px;
  }
  .evidence-text {
    font-size: 10.5px;
    color: #334155;
    line-height: 1.35;
  }
  .chat-footer-alert {
    background: #ffffff;
    border-top: 1px solid #e2e8f0;
    padding: 8px 14px;
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 10.5px;
    color: #64748b;
  }
  .alert-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #f59e0b;
    flex-shrink: 0;
  }
</style>
</head>
<body>
  <div class="banner-container">
    <!-- Left Column: Branding and System Metadata -->
    <div class="left-col">
      <div class="badge-row">
        <span class="system-badge">🛡️ Safety-Gated Support Architecture</span>
      </div>

      <div class="title-group">
        <h1>SupportGraph AI</h1>
        <p>Evidence-Grounded, Safety-Gated Apple Customer Support Resolution Platform</p>
      </div>

      <div class="feature-list">
        <div class="feature-item">
          <span class="icon-check">✓</span>
          <span><b>80,487</b> Historical AppleSupport Conversation Chains</span>
        </div>
        <div class="feature-item">
          <span class="icon-check">✓</span>
          <span><b>5-Dimension</b> Operational Precedent Validation & Synthesis</span>
        </div>
        <div class="feature-item">
          <span class="icon-check">✓</span>
          <span><b>Pre-Delivery</b> Claim Grounding & Unauthorized Action Verifier</span>
        </div>
        <div class="feature-item">
          <span class="icon-check">✓</span>
          <span><b>Deterministic Safety Vetoes:</b> Thermal & Hazard Escalation</span>
        </div>
        <div class="feature-item">
          <span class="icon-check">✓</span>
          <span><b>Live Human Review Queue:</b> Real-Time Specialist Adjudication</span>
        </div>
      </div>

      <div class="stats-bar">
        <div class="stat-unit">
          <span class="stat-val">0</span>
          <span class="stat-lbl">Unsafe Auto-Handles</span>
        </div>
        <div class="stat-unit">
          <span class="stat-val">100%</span>
          <span class="stat-lbl">Auto-Handle Precision</span>
        </div>
        <div class="stat-unit">
          <span class="stat-val">529</span>
          <span class="stat-lbl">Verified Tests</span>
        </div>
        <div class="stat-unit">
          <span class="stat-val">0</span>
          <span class="stat-lbl">Golden Leakage</span>
        </div>
      </div>
    </div>

    <!-- Right Column: Interactive Chatbot Grounding Representation -->
    <div class="right-col">
      <div class="chat-header">
        <div class="chat-title">
          <span class="live-dot"></span>
          <span>SupportGraph AI · AppleSupport Resolution</span>
        </div>
        <span class="grounding-tag">Evidence Grounded</span>
      </div>

      <div class="chat-body">
        <div class="user-bubble">
          My iPhone battery is draining very quickly after updating to iOS 16.
        </div>

        <div class="agent-bubble">
          <div class="action-badge">✓ AUTO-HANDLED · VERIFIED RESOLUTION</div>
          <div>
            Battery drain following an iOS update is typically caused by background photo indexing and Spotlight library rebuilds. These finish within 48 hours. Here are official optimization steps...
          </div>
          <div class="evidence-card">
            <div class="evidence-head">
              <span>DIRECT MATCH · AppleSupport Precedent #106719</span>
              <span>100% Grounded</span>
            </div>
            <div class="evidence-text">
              "Restoring apps & system indexing can cause temporary battery usage for 24-48 hrs after updating. Settings > Battery will show background activity."
            </div>
          </div>
        </div>
      </div>

      <div class="chat-footer-alert">
        <span class="alert-dot"></span>
        <span><b>Safety Gate:</b> Thermal hazards, swollen batteries & ambiguous queries automatically route to Live Human Review.</span>
      </div>
    </div>
  </div>
</body>
</html>
"""

def main():
    out_dir = Path("docs/assets")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "supportgraph-banner.png"

    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as f:
        f.write(HTML_CONTENT)
        html_path = f.name

    cmd = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "--headless",
        "--disable-gpu",
        "--window-size=1200,500",
        "--hide-scrollbars",
        "--force-device-scale-factor=2",
        f"--screenshot={out_png.resolve()}",
        f"file://{Path(html_path).resolve()}"
    ]
    subprocess.run(cmd, check=True)
    print(f"Banner successfully generated at: {out_png} ({out_png.stat().st_size} bytes)")

if __name__ == "__main__":
    main()
