"""
AQRA Dashboard Server
======================
Run this file to launch the web dashboard.
It connects to all 5 agents and serves a live UI.

Usage:
    pip install flask flask-cors
    python dashboard.py
    Open: http://localhost:5000
"""
import os
from flask import Flask, jsonify, request, render_template_string
from flask_cors import CORS
import json
import traceback

# Import all agents
from agent1 import DataIngestionAgent
from agent2 import SignalResearchAgent
from agent3 import RiskManagerAgent
from agent4 import ExecutionAgent
from agent5 import OrchestratorAgent

app = Flask(__name__)
CORS(app)

# Global state — in production you'd use a database
state = {
    "positions": [],
    "activity" : [],
    "last_memo": "",
    "last_goal": "",
}

# Shared agents (persist across requests)
data_agent   = DataIngestionAgent()
signal_agent = SignalResearchAgent()
risk_agent   = RiskManagerAgent(capital=1_000_000, min_confidence=0.30)
exec_agent   = ExecutionAgent(slippage_pct=0.001)


# ── API Routes ──────────────────────────────────

@app.route("/api/scan", methods=["POST"])
def scan():
    """Run full pipeline on a goal."""
    try:
        goal    = request.json.get("goal", "Scan Indian IT stocks")
        tickers = request.json.get("tickers", [])

        orch = OrchestratorAgent(capital=1_000_000, min_confidence=0.30)

        if not tickers:
            tickers = orch._llm_select_tickers(goal)

        snapshots = orch._run_data_agent(tickers)
        reports   = orch._run_signal_agent(snapshots)
        decisions = orch._run_risk_agent(reports)
        orch._run_execution_agent(decisions)
        memo      = orch._llm_generate_memo(goal, reports, decisions, orch.exec_agent.get_report())

        # Serialize for frontend
        signal_data = []
        for ticker, r in reports.items():
            d = decisions[ticker]
            signal_data.append({
                "ticker"    : ticker,
                "price"     : r.current_price,
                "currency"  : r.currency,
                "exchange"  : r.exchange,
                "direction" : r.final_direction,
                "confidence": round(r.confidence * 100),
                "approved"  : d.approved,
                "action"    : d.action,
                "stop_loss" : d.stop_loss,
                "take_profit": d.take_profit,
                "position_size": d.position_size,
                "capital_risk" : d.capital_to_risk,
                "rr_ratio"  : d.risk_reward_ratio,
                "rejection" : d.rejection_reason,
                "signals"   : [
                    {"name": s.name, "direction": s.direction,
                     "strength": s.strength, "reason": s.reason}
                    for s in r.signals
                ]
            })

        state["last_memo"] = memo
        state["last_goal"] = goal
        state["activity"].insert(0, {
            "time"  : __import__("datetime").datetime.now().strftime("%H:%M:%S"),
            "agent" : "Orchestrator",
            "event" : f"Scan complete — {len(tickers)} tickers, {sum(1 for s in signal_data if s['approved'])} approved",
            "type"  : "info"
        })

        return jsonify({
            "success"    : True,
            "goal"       : goal,
            "tickers"    : tickers,
            "signals"    : signal_data,
            "memo"       : memo,
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/portfolio", methods=["GET"])
def portfolio():
    """Get current portfolio status."""
    positions = []
    for p in exec_agent.positions:
        positions.append({
            "ticker"     : p.order.ticker,
            "action"     : p.order.action,
            "units"      : p.order.units,
            "entry"      : p.order.fill_price,
            "current"    : p.current_price,
            "pnl"        : p.unrealized_pnl if p.status.value == "OPEN" else p.realized_pnl,
            "status"     : p.status.value,
            "exit_reason": p.exit_reason,
            "currency"   : p.order.currency,
            "stop_loss"  : p.order.stop_loss,
            "take_profit": p.order.take_profit,
        })
    return jsonify({"positions": positions})


@app.route("/api/quickscan", methods=["GET"])
def quickscan():
    """Quick scan of default tickers — no LLM needed."""
    try:
        tickers   = ["TCS.NS", "INFY.NS", "RELIANCE.NS", "AAPL", "NVDA"]
        snapshots = data_agent.get_multiple(tickers, period="6mo")
        reports   = signal_agent.analyze_multiple(snapshots)

        result = []
        for ticker, r in reports.items():
            result.append({
                "ticker"    : ticker,
                "price"     : r.current_price,
                "currency"  : r.currency,
                "direction" : r.final_direction,
                "confidence": round(r.confidence * 100),
                "signals"   : [{"name": s.name, "direction": s.direction, "strength": s.strength} for s in r.signals]
            })
        return jsonify({"success": True, "data": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/")
def index():
    return render_template_string(HTML)


# ── HTML Dashboard ──────────────────────────────

HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AQRA — Autonomous Quantitative Research Architecture</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #05080f;
    --surface: #0a0f1a;
    --border: #111827;
    --border2: #1f2937;
    --text: #e2e8f0;
    --muted: #4b5563;
    --muted2: #6b7280;
    --gold: #f59e0b;
    --green: #10b981;
    --red: #ef4444;
    --blue: #3b82f6;
    --purple: #8b5cf6;
    --cyan: #06b6d4;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
    font-size: 13px;
    min-height: 100vh;
    overflow-x: hidden;
  }
  /* Grid noise texture */
  body::before {
    content: '';
    position: fixed; inset: 0;
    background-image:
      linear-gradient(rgba(245,158,11,0.02) 1px, transparent 1px),
      linear-gradient(90deg, rgba(245,158,11,0.02) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none; z-index: 0;
  }

  /* ── Header ── */
  header {
    position: sticky; top: 0; z-index: 100;
    background: rgba(5,8,15,0.95);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid var(--border);
    padding: 0 28px;
    height: 56px;
    display: flex; align-items: center; justify-content: space-between;
  }
  .logo {
    display: flex; align-items: center; gap: 10px;
  }
  .logo-icon {
    width: 32px; height: 32px;
    background: linear-gradient(135deg, var(--gold), #d97706);
    border-radius: 7px;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px;
    box-shadow: 0 0 16px rgba(245,158,11,0.3);
  }
  .logo-text {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 22px; letter-spacing: 0.08em;
    color: var(--text);
  }
  .logo-sub {
    font-family: 'DM Mono', monospace;
    font-size: 9px; color: var(--muted); letter-spacing: 0.14em;
    text-transform: uppercase;
  }
  .header-stats {
    display: flex; gap: 24px; align-items: center;
  }
  .hstat { text-align: right; }
  .hstat-label { font-size: 9px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; }
  .hstat-value { font-family: 'DM Mono', monospace; font-size: 12px; font-weight: 500; }
  .live-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--green);
    box-shadow: 0 0 8px var(--green);
    animation: pulse 1.8s infinite;
  }
  @keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.3;} }

  /* ── Layout ── */
  .main { position: relative; z-index: 1; padding: 20px 24px; display: flex; flex-direction: column; gap: 16px; }

  /* ── Scan Bar ── */
  .scan-bar {
    background: var(--surface);
    border: 1px solid var(--border2);
    border-radius: 12px;
    padding: 14px 18px;
    display: flex; gap: 10px; align-items: center;
  }
  .scan-input {
    flex: 1;
    background: transparent;
    border: none; outline: none;
    color: var(--text);
    font-family: 'DM Mono', monospace;
    font-size: 13px;
  }
  .scan-input::placeholder { color: var(--muted); }
  .btn {
    background: var(--gold);
    color: #000;
    border: none; cursor: pointer;
    padding: 8px 20px;
    border-radius: 7px;
    font-family: 'DM Mono', monospace;
    font-size: 12px; font-weight: 500;
    letter-spacing: 0.04em;
    transition: all 0.15s;
    white-space: nowrap;
  }
  .btn:hover { background: #fbbf24; transform: translateY(-1px); }
  .btn:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }
  .btn-ghost {
    background: transparent;
    border: 1px solid var(--border2);
    color: var(--muted2);
    padding: 8px 16px;
    border-radius: 7px;
    cursor: pointer;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    transition: all 0.15s;
  }
  .btn-ghost:hover { border-color: var(--gold); color: var(--gold); }

  /* ── Top row metrics ── */
  .metrics-row {
    display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px;
  }
  .metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 16px;
    position: relative; overflow: hidden;
    transition: border-color 0.2s;
  }
  .metric-card:hover { border-color: var(--border2); }
  .metric-card::after {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
  }
  .metric-card.gold::after { background: var(--gold); }
  .metric-card.green::after { background: var(--green); }
  .metric-card.red::after { background: var(--red); }
  .metric-card.blue::after { background: var(--blue); }
  .metric-card.purple::after { background: var(--purple); }
  .metric-label { font-size: 9px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px; }
  .metric-value { font-family: 'Bebas Neue', sans-serif; font-size: 28px; letter-spacing: 0.04em; line-height: 1; }
  .metric-sub { font-size: 10px; color: var(--muted2); margin-top: 4px; font-family: 'DM Mono', monospace; }

  /* ── Middle row ── */
  .mid-row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }

  /* ── Cards ── */
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
  }
  .card-header {
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; justify-content: space-between;
  }
  .card-title {
    font-family: 'DM Mono', monospace;
    font-size: 10px; font-weight: 500;
    text-transform: uppercase; letter-spacing: 0.12em;
    color: var(--muted2);
  }
  .card-body { padding: 14px 16px; }

  /* ── Signal Table ── */
  .signal-table { width: 100%; border-collapse: collapse; }
  .signal-table th {
    font-family: 'DM Mono', monospace;
    font-size: 9px; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.1em;
    text-align: left; padding: 0 10px 8px;
    border-bottom: 1px solid var(--border);
  }
  .signal-table td {
    padding: 9px 10px;
    border-bottom: 1px solid var(--border);
    vertical-align: middle;
  }
  .signal-table tr:last-child td { border-bottom: none; }
  .signal-table tr { transition: background 0.15s; }
  .signal-table tr:hover td { background: rgba(255,255,255,0.02); }
  .ticker-badge {
    font-family: 'DM Mono', monospace;
    font-size: 12px; font-weight: 500;
    color: var(--text);
  }
  .exch-badge {
    font-size: 9px; color: var(--muted);
    font-family: 'DM Mono', monospace;
  }
  .dir-badge {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 3px 8px; border-radius: 4px;
    font-family: 'DM Mono', monospace;
    font-size: 10px; font-weight: 500; letter-spacing: 0.06em;
  }
  .dir-buy  { background: rgba(16,185,129,0.15); color: var(--green); }
  .dir-sell { background: rgba(239,68,68,0.15);  color: var(--red); }
  .dir-neutral { background: rgba(107,114,128,0.15); color: var(--muted2); }
  .conf-bar-wrap { display: flex; align-items: center; gap: 6px; }
  .conf-bar { height: 3px; background: var(--border2); border-radius: 2px; width: 60px; }
  .conf-fill { height: 100%; border-radius: 2px; transition: width 0.4s; }
  .approved-badge {
    font-size: 9px; font-family: 'DM Mono', monospace;
    padding: 2px 7px; border-radius: 3px; letter-spacing: 0.06em;
  }
  .approved-yes { background: rgba(16,185,129,0.15); color: var(--green); }
  .approved-no  { background: rgba(107,114,128,0.1);  color: var(--muted); }

  /* ── Agent Pipeline ── */
  .pipeline {
    display: flex; flex-direction: column; gap: 8px;
  }
  .agent-row {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 12px;
    background: rgba(255,255,255,0.02);
    border: 1px solid var(--border);
    border-radius: 8px;
    transition: border-color 0.2s;
    cursor: default;
  }
  .agent-row:hover { border-color: var(--border2); }
  .agent-icon {
    width: 32px; height: 32px; border-radius: 7px;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; flex-shrink: 0;
  }
  .agent-name { font-family: 'DM Mono', monospace; font-size: 11px; font-weight: 500; }
  .agent-desc { font-size: 10px; color: var(--muted2); margin-top: 1px; }
  .agent-status {
    margin-left: auto;
    font-size: 9px; font-family: 'DM Mono', monospace;
    padding: 2px 8px; border-radius: 10px; letter-spacing: 0.06em;
    text-transform: uppercase;
  }
  .status-active   { background: rgba(16,185,129,0.15);  color: var(--green); }
  .status-idle     { background: rgba(107,114,128,0.1);   color: var(--muted); }
  .status-thinking { background: rgba(245,158,11,0.15);   color: var(--gold); animation: pulse 1s infinite; }

  /* ── Memo ── */
  .memo-box {
    font-size: 12px; line-height: 1.7; color: #94a3b8;
    white-space: pre-wrap;
    font-family: 'DM Sans', sans-serif;
    max-height: 260px; overflow-y: auto;
  }
  .memo-box strong, .memo-box b { color: var(--text); }

  /* ── Positions ── */
  .positions-list { display: flex; flex-direction: column; gap: 8px; }
  .pos-card {
    display: grid; grid-template-columns: 1fr auto;
    gap: 8px; align-items: center;
    padding: 10px 12px;
    background: rgba(255,255,255,0.02);
    border: 1px solid var(--border);
    border-radius: 8px;
  }
  .pos-ticker { font-family: 'DM Mono', monospace; font-size: 12px; font-weight: 500; }
  .pos-detail { font-size: 10px; color: var(--muted2); margin-top: 2px; font-family: 'DM Mono', monospace; }
  .pos-pnl {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 18px; text-align: right;
  }
  .pnl-pos { color: var(--green); }
  .pnl-neg { color: var(--red); }
  .pnl-zero { color: var(--muted2); }

  /* ── Bottom row ── */
  .bottom-row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }

  /* ── Activity feed ── */
  .activity-feed { display: flex; flex-direction: column; gap: 6px; max-height: 220px; overflow-y: auto; }
  .activity-item {
    display: flex; gap: 10px; align-items: flex-start;
    padding: 8px 10px;
    border-radius: 6px;
    background: rgba(255,255,255,0.015);
    animation: fadeIn 0.3s ease;
  }
  @keyframes fadeIn { from { opacity:0; transform:translateY(-4px); } to { opacity:1; transform:translateY(0); } }
  .act-time { font-family: 'DM Mono', monospace; font-size: 9px; color: var(--muted); flex-shrink: 0; margin-top: 1px; }
  .act-dot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; margin-top: 4px; }
  .act-text { font-size: 11px; color: var(--muted2); line-height: 1.4; }

  /* ── Signals mini bars ── */
  .signal-bars { display: flex; gap: 4px; align-items: flex-end; height: 24px; }
  .signal-bar-item { width: 8px; border-radius: 2px 2px 0 0; transition: height 0.4s; }

  /* ── Loading ── */
  .loading {
    display: flex; align-items: center; gap: 8px;
    color: var(--gold); font-family: 'DM Mono', monospace; font-size: 11px;
  }
  .spinner {
    width: 14px; height: 14px; border: 2px solid var(--border2);
    border-top-color: var(--gold); border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  .empty { color: var(--muted); font-family: 'DM Mono', monospace; font-size: 11px; text-align: center; padding: 20px; }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 3px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }
</style>
</head>
<body>

<header>
  <div class="logo">
    <div class="logo-icon">⬡</div>
    <div>
      <div class="logo-text">AQRA</div>
      <div class="logo-sub">Autonomous Quantitative Research Architecture</div>
    </div>
  </div>
  <div class="header-stats">
    <div class="hstat">
      <div class="hstat-label">Capital</div>
      <div class="hstat-value" style="color:var(--gold)">₹10,00,000</div>
    </div>
    <div class="hstat">
      <div class="hstat-label">Agents</div>
      <div class="hstat-value" style="color:var(--green)">5 / 5</div>
    </div>
    <div class="hstat">
      <div class="hstat-label">Mode</div>
      <div class="hstat-value" style="color:var(--blue)">PAPER</div>
    </div>
    <div class="hstat">
      <div class="hstat-label">Status</div>
      <div class="hstat-value" style="color:var(--green)">LIVE</div>
    </div>
    <div class="live-dot"></div>
  </div>
</header>

<div class="main">

  <!-- Scan Bar -->
  <div class="scan-bar">
    <span style="color:var(--gold);font-family:'DM Mono',monospace;font-size:12px;flex-shrink:0">⬡ GOAL</span>
    <input class="scan-input" id="goalInput" placeholder="e.g. Find opportunities in Indian IT stocks today..." value="Find the best trading opportunities in Indian IT stocks today" />
    <button class="btn-ghost" onclick="quickScan()">Quick Scan</button>
    <button class="btn" id="scanBtn" onclick="runScan()">▶ Run Orchestrator</button>
  </div>

  <!-- Metrics Row -->
  <div class="metrics-row">
    <div class="metric-card gold">
      <div class="metric-label">Tickers Scanned</div>
      <div class="metric-value" id="m-tickers" style="color:var(--gold)">—</div>
      <div class="metric-sub" id="m-tickers-sub">awaiting scan</div>
    </div>
    <div class="metric-card green">
      <div class="metric-label">Approved Trades</div>
      <div class="metric-value" id="m-approved" style="color:var(--green)">—</div>
      <div class="metric-sub" id="m-approved-sub">risk-filtered</div>
    </div>
    <div class="metric-card red">
      <div class="metric-label">Rejected</div>
      <div class="metric-value" id="m-rejected" style="color:var(--red)">—</div>
      <div class="metric-sub">no directional edge</div>
    </div>
    <div class="metric-card blue">
      <div class="metric-label">Capital at Risk</div>
      <div class="metric-value" id="m-risk" style="color:var(--blue);font-size:20px">—</div>
      <div class="metric-sub">half-kelly sizing</div>
    </div>
    <div class="metric-card purple">
      <div class="metric-label">Avg Confidence</div>
      <div class="metric-value" id="m-conf" style="color:var(--purple)">—</div>
      <div class="metric-sub">signal agreement</div>
    </div>
  </div>

  <!-- Middle Row -->
  <div class="mid-row">

    <!-- Signal Table -->
    <div class="card">
      <div class="card-header">
        <span class="card-title">Signal Research</span>
        <span id="signal-status" style="font-family:'DM Mono',monospace;font-size:9px;color:var(--muted)">—</span>
      </div>
      <div class="card-body" style="padding:0">
        <table class="signal-table" id="signalTable">
          <thead>
            <tr>
              <th>Ticker</th>
              <th>Price</th>
              <th>Signal</th>
              <th>Conf</th>
              <th>Decision</th>
            </tr>
          </thead>
          <tbody id="signalBody">
            <tr><td colspan="5" class="empty">Run a scan to see signals</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- Agent Pipeline + Positions -->
    <div style="display:flex;flex-direction:column;gap:16px">

      <!-- Agent Pipeline -->
      <div class="card">
        <div class="card-header">
          <span class="card-title">Agent Pipeline</span>
          <span id="pipeline-status" style="font-family:'DM Mono',monospace;font-size:9px;color:var(--muted)">IDLE</span>
        </div>
        <div class="card-body">
          <div class="pipeline" id="pipeline">
            <div class="agent-row">
              <div class="agent-icon" style="background:rgba(245,158,11,0.15)">⬡</div>
              <div><div class="agent-name">Orchestrator</div><div class="agent-desc">Gemini LLM · goal decomposition</div></div>
              <span class="agent-status status-idle" id="s-orch">IDLE</span>
            </div>
            <div class="agent-row">
              <div class="agent-icon" style="background:rgba(6,182,212,0.15)">◈</div>
              <div><div class="agent-name">Data Ingestion</div><div class="agent-desc">yfinance · NSE + NYSE + CBOE</div></div>
              <span class="agent-status status-idle" id="s-data">IDLE</span>
            </div>
            <div class="agent-row">
              <div class="agent-icon" style="background:rgba(16,185,129,0.15)">◇</div>
              <div><div class="agent-name">Signal Research</div><div class="agent-desc">RSI · SMA · Bollinger Bands</div></div>
              <span class="agent-status status-idle" id="s-signal">IDLE</span>
            </div>
            <div class="agent-row">
              <div class="agent-icon" style="background:rgba(239,68,68,0.15)">◉</div>
              <div><div class="agent-name">Risk Manager</div><div class="agent-desc">Kelly criterion · portfolio gates</div></div>
              <span class="agent-status status-idle" id="s-risk">IDLE</span>
            </div>
            <div class="agent-row">
              <div class="agent-icon" style="background:rgba(139,92,246,0.15)">◐</div>
              <div><div class="agent-name">Execution</div><div class="agent-desc">paper trading · stop/TP auto-exit</div></div>
              <span class="agent-status status-idle" id="s-exec">IDLE</span>
            </div>
          </div>
        </div>
      </div>

    </div>
  </div>

  <!-- Bottom Row -->
  <div class="bottom-row">

    <!-- Investment Memo -->
    <div class="card">
      <div class="card-header">
        <span class="card-title">Gemini Investment Memo</span>
        <span id="memo-time" style="font-family:'DM Mono',monospace;font-size:9px;color:var(--muted)">—</span>
      </div>
      <div class="card-body">
        <div class="memo-box" id="memoBox">
          <span style="color:var(--muted)">Investment memo will appear here after running the orchestrator...</span>
        </div>
      </div>
    </div>

    <!-- Positions -->
    <div class="card">
      <div class="card-header">
        <span class="card-title">Open Positions</span>
        <span id="pos-count" style="font-family:'DM Mono',monospace;font-size:9px;color:var(--muted)">0 OPEN</span>
      </div>
      <div class="card-body">
        <div class="positions-list" id="positionsList">
          <div class="empty">No open positions</div>
        </div>
      </div>
    </div>

  </div>

</div>

<script>
  const API = '';

  function setAgentStatus(id, status) {
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = status;
    el.className = 'agent-status ' + (
      status === 'ACTIVE'   ? 'status-active' :
      status === 'THINKING' ? 'status-thinking' :
      'status-idle'
    );
  }

  function resetAgents() {
    ['s-orch','s-data','s-signal','s-risk','s-exec'].forEach(id => setAgentStatus(id, 'IDLE'));
    document.getElementById('pipeline-status').textContent = 'IDLE';
  }

  async function runScan() {
    const goal = document.getElementById('goalInput').value.trim();
    if (!goal) return;

    const btn = document.getElementById('scanBtn');
    btn.disabled = true;
    btn.textContent = '⟳ Running...';
    resetAgents();

    // Animate agents sequentially
    const delay = ms => new Promise(r => setTimeout(r, ms));
    document.getElementById('pipeline-status').textContent = 'RUNNING';

    setAgentStatus('s-orch', 'THINKING');
    await delay(400);
    setAgentStatus('s-orch', 'ACTIVE');
    setAgentStatus('s-data', 'THINKING');
    await delay(600);
    setAgentStatus('s-data', 'ACTIVE');
    setAgentStatus('s-signal', 'THINKING');
    await delay(500);
    setAgentStatus('s-signal', 'ACTIVE');
    setAgentStatus('s-risk', 'THINKING');
    await delay(400);
    setAgentStatus('s-risk', 'ACTIVE');
    setAgentStatus('s-exec', 'THINKING');

    try {
      const res = await fetch(`${API}/api/scan`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ goal })
      });
      const data = await res.json();

      if (!data.success) throw new Error(data.error);

      setAgentStatus('s-exec', 'ACTIVE');
      document.getElementById('pipeline-status').textContent = 'COMPLETE';

      updateSignalTable(data.signals);
      updateMetrics(data.signals);
      updateMemo(data.memo, goal);
      loadPositions();

    } catch (err) {
      document.getElementById('memoBox').innerHTML = `<span style="color:var(--red)">Error: ${err.message}</span>`;
      resetAgents();
    }

    btn.disabled = false;
    btn.textContent = '▶ Run Orchestrator';
  }

  async function quickScan() {
    const btn = document.querySelector('.btn-ghost');
    btn.disabled = true;
    btn.textContent = '⟳ Scanning...';

    try {
      const res = await fetch(`${API}/api/quickscan`);
      const data = await res.json();
      if (!data.success) throw new Error(data.error);
      updateSignalTable(data.data);
      updateMetrics(data.data);
      document.getElementById('signal-status').textContent = 'QUICK SCAN';
    } catch(err) {
      alert('Quick scan failed: ' + err.message);
    }

    btn.disabled = false;
    btn.textContent = 'Quick Scan';
  }

  function updateSignalTable(signals) {
    const tbody = document.getElementById('signalBody');
    if (!signals || signals.length === 0) {
      tbody.innerHTML = '<tr><td colspan="5" class="empty">No signals</td></tr>';
      return;
    }

    tbody.innerHTML = signals.map(s => {
      const dirClass = s.direction === 'BUY' ? 'dir-buy' : s.direction === 'SELL' ? 'dir-sell' : 'dir-neutral';
      const dirIcon  = s.direction === 'BUY' ? '▲' : s.direction === 'SELL' ? '▼' : '—';
      const confColor = s.confidence >= 67 ? 'var(--green)' : s.confidence >= 50 ? 'var(--gold)' : 'var(--red)';
      const approved = s.approved !== undefined;
      const appClass = s.approved ? 'approved-yes' : 'approved-no';
      const appText  = s.approved ? '✓ TRADE' : '✗ SKIP';
      const price    = s.currency === 'INR' ? `₹${s.price?.toLocaleString('en-IN', {maximumFractionDigits:2})}` : `$${s.price?.toFixed(2)}`;

      return `<tr>
        <td>
          <div class="ticker-badge">${s.ticker}</div>
          <div class="exch-badge">${s.exchange || ''}</div>
        </td>
        <td style="font-family:'DM Mono',monospace;font-size:11px">${price}</td>
        <td><span class="dir-badge ${dirClass}">${dirIcon} ${s.direction}</span></td>
        <td>
          <div class="conf-bar-wrap">
            <div class="conf-bar"><div class="conf-fill" style="width:${s.confidence}%;background:${confColor}"></div></div>
            <span style="font-family:'DM Mono',monospace;font-size:10px;color:${confColor}">${s.confidence}%</span>
          </div>
        </td>
        <td>${approved ? `<span class="approved-badge ${appClass}">${appText}</span>` : '—'}</td>
      </tr>`;
    }).join('');

    document.getElementById('signal-status').textContent = `${signals.length} TICKERS`;
  }

  function updateMetrics(signals) {
    if (!signals || signals.length === 0) return;
    const approved = signals.filter(s => s.approved);
    const rejected = signals.filter(s => s.approved === false);
    const totalRisk = approved.reduce((sum, s) => sum + (s.capital_risk || 0), 0);
    const avgConf   = Math.round(signals.reduce((sum, s) => sum + s.confidence, 0) / signals.length);

    document.getElementById('m-tickers').textContent = signals.length;
    document.getElementById('m-tickers-sub').textContent = signals.map(s=>s.ticker.replace('.NS','')).join(' · ');
    document.getElementById('m-approved').textContent = approved.length;
    document.getElementById('m-approved-sub').textContent = approved.length ? approved.map(s=>s.ticker.replace('.NS','')).join(' · ') : 'none passed gates';
    document.getElementById('m-rejected').textContent = rejected.length;
    document.getElementById('m-risk').textContent = totalRisk > 0 ? `₹${(totalRisk/1000).toFixed(0)}K` : '₹0';
    document.getElementById('m-conf').textContent = `${avgConf}%`;
  }

  function updateMemo(memo, goal) {
    const box = document.getElementById('memoBox');
    // Convert **bold** markdown to HTML
    const html = memo
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>');
    box.innerHTML = html;
    document.getElementById('memo-time').textContent = new Date().toLocaleTimeString();
  }

  async function loadPositions() {
    try {
      const res  = await fetch(`${API}/api/portfolio`);
      const data = await res.json();
      const list = document.getElementById('positionsList');
      const positions = data.positions || [];

      document.getElementById('pos-count').textContent = `${positions.filter(p=>p.status==='OPEN').length} OPEN`;

      if (positions.length === 0) {
        list.innerHTML = '<div class="empty">No positions yet</div>';
        return;
      }

      list.innerHTML = positions.map(p => {
        const pnl = p.pnl || 0;
        const pnlClass = pnl > 0 ? 'pnl-pos' : pnl < 0 ? 'pnl-neg' : 'pnl-zero';
        const pnlSign  = pnl > 0 ? '+' : '';
        const curr     = p.currency === 'INR' ? '₹' : '$';
        return `<div class="pos-card">
          <div>
            <div class="pos-ticker">${p.ticker} <span style="font-size:10px;color:var(--muted)">${p.action}</span></div>
            <div class="pos-detail">${p.units?.toFixed(2)} units @ ${curr}${p.entry?.toLocaleString()} · SL: ${curr}${p.stop_loss?.toFixed(0)} · TP: ${curr}${p.take_profit?.toFixed(0)}</div>
            <div class="pos-detail" style="color:${p.status==='OPEN'?'var(--green)':'var(--muted)'}">${p.status} ${p.exit_reason ? '· '+p.exit_reason : ''}</div>
          </div>
          <div class="pos-pnl ${pnlClass}">${pnlSign}${curr}${Math.abs(pnl).toLocaleString('en-IN',{maximumFractionDigits:0})}</div>
        </div>`;
      }).join('');
    } catch(e) { console.error(e); }
  }

  // Load positions on page load
  loadPositions();
</script>
</body>
</html>'''


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════╗
║   AQRA Dashboard starting...                ║
║   Open: http://localhost:5000               ║
╚══════════════════════════════════════════════╝
""")
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
