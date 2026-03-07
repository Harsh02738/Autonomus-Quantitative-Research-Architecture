# AQRA — Autonomous Quantitative Research Architecture

> A 5-agent trading system for US equities and Indian NSE markets, validated across 17 years (2008–2024) including the Global Financial Crisis.

---

## The Problem This Solves

Most retail quantitative systems are either (a) a single strategy optimized on in-sample data, or (b) a manually operated signal system requiring constant human intervention. Both fail the same test: they are not robust to regime change.

AQRA was built around a different thesis — that a system with genuine edge across market regimes should be able to identify *what kind of market it is in* before deciding *how to trade*. Every design decision flows from this.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    AQRA Pipeline                        │
│                                                         │
│  Goal → [Agent 5: Orchestrator]                         │
│           ↓  (Selects ticker universe via LLM)          │
│         [Agent 1: Data Ingestion]                       │
│           ↓  (yfinance — NSE + NYSE EOD)                │
│         [Agent 6: Macro Sentinel]  ←── NEW              │
│           ↓  (Regime flip probability)                  │
│         [Agent 2: Signal Research]                      │
│           ↓  (5-signal weighted vote)                   │
│         [Agent 3: Risk Manager]                         │
│           ↓  (Kelly sizing + ATR stops)                 │
│         [Agent 4: Execution]                            │
│           ↓  (Paper fills + P&L tracking)              │
│         [Dashboard + Telegram Alerts]                   │
└─────────────────────────────────────────────────────────┘
```

### Agent Responsibilities

| Agent | Role | Key Output |
|-------|------|------------|
| Agent 1 — Data | Fetches OHLCV for 11 tickers (NSE + NYSE), downloads macro series (VIX, HYG/LQD, Treasuries, India VIX) | Clean aligned DataFrames |
| Agent 2 — Signal Research | Computes 5 indicators with regime-weighted scoring; fires BUY/SELL when weighted confidence ≥ 45% | Signal + confidence score |
| Agent 3 — Risk Manager | Half-Kelly position sizing scaled by regime strength × VIX multiplier × macro modifier; trailing ATR stops | Units, stop price, target price |
| Agent 4 — Execution | Paper trading simulator; tracks fills, open P&L, stop/target hits using daily high/low for realism | Trade log, equity curve |
| Agent 5 — Orchestrator | Gemini LLM selects tickers based on market conditions; generates investment memo | Ticker universe |
| Agent 6 — Macro Sentinel | Computes daily Regime Flip Probability from VIX term structure, credit spreads, yield curve | Flip probability (0–1) |

---

## Signal Engine

Five indicators with regime-dependent weights:

```python
signals  = ["SMA crossover", "RSI", "Bollinger Bands", "MACD", "OBV"]
base_wts = [1.0,              1.5,   1.0,               2.0,    2.0  ]

# Weights shift by regime — example: BULL regime
bull_wts = {"sma": 2.0, "macd": 2.5, "obv": 1.8, "rsi": 0.8, "bb": 0.8}
bear_wts = {"sma": 0.6, "macd": 0.7, "obv": 1.5, "rsi": 2.2, "bb": 2.2}
```

**Why this works:** In bull regimes, trend-following signals (SMA, MACD, OBV) have higher predictive power. In bear regimes, mean-reversion signals (RSI, BB) dominate. Static equal-weighting would underperform in both.

**Per-ticker classification:** Momentum tickers (NVDA, MSFT, JPM, TCS, HCLTECH, BAJFINANCE) receive widened RSI thresholds in bull regimes to avoid premature exits. Mean-reversion tickers (AAPL, GOOGL, INFY, HDFCBANK, RELIANCE) use standard thresholds always.

---

## Regime Detection (3-Layer)

```
Layer 1 — Trend      : Price vs 200-day SMA → BULL / NEUTRAL / BEAR
Layer 2 — Volatility : 14-day ATR vs 60-day ATR mean → LOW / NORMAL / HIGH
Layer 3 — Breadth    : % universe above 50-day SMA → RISK_ON / NEUTRAL / RISK_OFF

Combinations → 9 named regime labels:
  OPTIMAL_BULL, BULL_RISK_ON, BULL_NEUTRAL,
  NEUTRAL_NORMAL, NEUTRAL_CAUTIOUS,
  BEAR_RISK_ON, BEAR_RISK_OFF,
  HIGH_VOL_BULL, HIGH_VOL_BEAR

Special case: BEAR + HIGH_VOL + RISK_OFF → BLACKOUT (position size = 0)
```

Each regime label maps to specific RSI thresholds, ADX filters, and ATR stop multipliers — 45 total parameter combinations, none optimized post-hoc.

---

## Macro Sentinel (Agent 6)

The Macro Sentinel addresses the lag inherent in price-based regime detection. By the time price signals confirm a regime flip, significant drawdown has already occurred. The Sentinel uses three leading macro indicators that historically anticipate regime changes by days to weeks:

**Signal 1 — VIX Term Structure (weight: 30%)**
```
Ratio = VIX9D / VIX3M
Normal market (contango):     ratio < 1.0  →  score near 0
Stressed market (backwardation): ratio > 1.0  →  score rises toward 1
```

**Signal 2 — Credit Spread Proxy (weight: 40%)**
```
HYG (High Yield ETF) / LQD (Investment Grade ETF) — 20-day rolling return
Falling ratio = credit conditions tightening = rising stress score
```

**Signal 3 — Yield Curve (weight: 30%)**
```
10Y Treasury yield − 2Y Treasury yield
Inversion + rapid flattening → high stress score
```

Combined `flip_prob` (0–1) modifies position sizing:

| flip_prob | Action |
|-----------|--------|
| < 0.50 | No change — v8 logic unchanged |
| 0.50 – 0.70 | Position size × 0.65, ATR stop tightened |
| 0.70 – 0.85 | Position size × 0.40, stop tightened further |
| > 0.85 | New positions blocked entirely |

For NSE tickers: global macro signal blended 50/50 with India VIX stress score.

---

## Risk Management

**Position Sizing**
```
risk_per_trade = 2% of capital
stop_distance  = ATR(14) × regime_atr_multiplier
units          = (capital × 0.02 × regime_mult × vix_mult × macro_mult) / stop_distance
target         = entry ± 2 × stop_distance  (2:1 R/R enforced always)
```

**Trailing ATR Stops (v8 innovation)**
```
atr_ratio = current 14-day ATR / 60-day mean ATR

ratio < 1.2  →  stop stays at ATR × 3.0  (normal volatility)
ratio 1.2–1.5 →  stop tightens to ATR × 2.0  (early warning)
ratio > 1.5  →  stop tightens to ATR × 1.5  (high volatility, protect gains)

Ratchet rule: stop can only move in direction of the trade — never widens
```
78% of trades had their stops tightened at least once in backtesting.

**Additional Gates**
- VIX gate: India VIX > 35 → zero new positions
- Correlation limit: skip trade if 60-day rolling correlation with any open position > 0.70
- Cross-sectional momentum filter: skip BUY if ticker ranks in bottom 50% of 63-day momentum
- Circuit breaker: rolling Sharpe < −0.5 for 2 consecutive 25-trade windows → 20-day pause

---

## Validation Results

### In-Sample (2015–2024)

| Metric | Value |
|--------|-------|
| CAGR | 10.69% |
| Sharpe Ratio | 2.139 |
| Max Drawdown | 17.37% |
| Win Rate | 42.8% |
| Profit Factor | 1.482 |
| Total Trades | 649 |
| P&L on $1M | +$1.76M |

### Monte Carlo (1,000 simulations, IS period)

| Metric | p5 | Median | p95 |
|--------|----|--------|-----|
| CAGR | 7.4% | 10.75% | 13.4% |
| Sharpe | 1.46 | 2.36 | 3.19 |
| Max Drawdown | 6.95% | 11.47% | 22.1% |

P(CAGR > 0%) = **100%** | P(Sharpe > 2) = **74.5%** | P(MaxDD < 20%) = **92.2%**

### Out-of-Sample Stress Test (2008–2014)

Parameters were **frozen** after v8 development. The 2008–2014 period was never used in any design decision.

| Metric | OOS (2008–14) | IS (2015–24) | Degradation |
|--------|--------------|--------------|-------------|
| CAGR | 10.12% | 11.77% | −1.65% |
| Sharpe | 2.19 | 2.43 | −0.24 |
| Max Drawdown | 16.01% | 16.16% | −0.15% |

**OOS degradation < 15% across all metrics** — indicating absence of overfitting.

### GFC Specific (Sep 2008 – Mar 2009)

- Total trades: 3 (VIX gate + BLACKOUT regime nearly shut system down)
- P&L: −$5,492 on $1,000,000 capital = **0.55% drawdown during peak GFC**
- The system's self-preservation mechanism functioned exactly as designed

### Year-by-Year OOS

| Year | P&L | Sharpe | Event |
|------|-----|--------|-------|
| 2008 | −$32,100 | −0.98 | Global Financial Crisis |
| 2009 | +$4,514 | +0.27 | Recovery begins |
| 2010 | +$116,597 | +1.98 | Post-GFC rally |
| 2011 | −$10,907 | −0.16 | European debt crisis |
| 2012 | +$478,961 | +7.06 | QE-driven rally |
| 2013 | +$269,024 | +2.79 | Taper tantrum |
| 2014 | +$136,877 | +1.64 | Oil crash |

5 of 7 years positive. The two negative years lost < 4% of capital combined.

---

## Development Progression

AQRA was built iteratively over 9 versions, each adding one principled layer:

| Version | Key Addition | Sharpe | Max DD |
|---------|-------------|--------|--------|
| v1 | Baseline (RSI + SMA + BB) | 1.631 | 36.94% |
| v2 | 3-layer regime detection | 1.869 | 22.94% |
| v3 | MACD + OBV + VIX proxy + ATR stops | 1.613 | 23.09% |
| v4 | Regime-weighted signal scoring + circuit breaker | 1.822 | 24.37% |
| v5 | Factor model regression | 0.372 | 54.89% — **dropped** |
| v6 | Walk-forward validation + correlation limits | 2.030 | 22.44% |
| v7 | Regime-specific RSI/ADX/ATR parameters | 1.797 | 32.33% |
| v8 | Trailing ATR ratchet stops + per-ticker override | **2.139** | **17.71%** |
| v9 | Macro Sentinel (Agent 6) | 2.139 | 17.37% |

**v5 was deliberately discarded** — a 2-year training window produced degenerate factor classifications. This failure is documented as a learning outcome: factor models require substantially longer lookback periods than signal-based systems.

---

## Universe

| Ticker | Exchange | Classification | Sector |
|--------|----------|---------------|--------|
| TCS.NS | NSE | Momentum | IT Services |
| INFY.NS | NSE | Mean-Reversion | IT Services |
| RELIANCE.NS | NSE | Mean-Reversion | Conglomerate |
| HDFCBANK.NS | NSE | Mean-Reversion | Banking |
| BAJFINANCE.NS | NSE | Momentum | NBFC |
| HCLTECH.NS | NSE | Momentum | IT Services |
| AAPL | NYSE | Mean-Reversion | Technology |
| MSFT | NYSE | Momentum | Technology |
| NVDA | NYSE | Momentum | Semiconductors |
| GOOGL | NYSE | Mean-Reversion | Technology |
| JPM | NYSE | Momentum | Banking |

---

## Paper Trading

The system runs live EOD paper trading on all 11 tickers.

**Daily workflow:**
```
3:30 PM IST — NSE closes
4:00 PM IST — Run: python paper_trading.py

  → Fetches final OHLCV for NSE (today) + NYSE (previous close)
  → Fetches live USD/INR rate
  → Updates trailing stops on open positions
  → Checks stop/target hits using today's high/low
  → Scans all 11 tickers for new signals
  → Saves daily HTML report to reports/YYYY-MM-DD.html
  → Sends Telegram summary with positions + P&L
  → Updates dashboard at http://localhost:5001
```

All P&L displayed in INR. USD positions converted at live USD/INR rate.

---

## Installation

```bash
# Clone
git clone https://github.com/yourusername/AQRA.git
cd AQRA

# Install dependencies
pip install yfinance pandas numpy flask python-dotenv requests

# Configure Telegram notifications (optional)
cp .env.example .env
# Edit .env with your Telegram bot token + chat ID

# Run backtests
python backtest_v9.py          # Full 2015–2024 backtest
python stress_test.py          # OOS 2008–2014 stress test

# Run paper trading (after 4 PM IST on trading days)
python paper_trading.py
```

---

## File Structure

```
AQRA/
├── agent1_data_ingestion.py   # Data fetching (NSE + NYSE)
├── agent2_signal_research.py  # 5-indicator signal engine
├── agent3_risk_manager.py     # Kelly sizing + ATR stop logic
├── agent4_execution.py        # Paper trading simulator
├── agent5_orchestrator.py     # Gemini LLM ticker selection
├── backtest_v8.py             # Best backtest version (IS period)
├── backtest_v9.py             # + Macro Sentinel
├── stress_test.py             # OOS 2008–2014 validation
├── paper_trading.py           # Live EOD paper trading engine
├── dashboard.py               # Flask web dashboard
├── paper_data/
│   └── state.json             # Persistent paper trading state
├── reports/
│   └── YYYY-MM-DD.html        # Daily HTML reports
└── .env.example               # Telegram configuration template
```

---

## Key Design Decisions and Why

**Why daily bars, not intraday?**
The regime detection layer requires 200-day SMA, 60-day ATR lookback, and 50-day breadth calculations. These are inherently daily-bar metrics. Intraday execution would add noise without adding signal — the system's edge comes from regime awareness, not order flow.

**Why not optimize parameters per ticker?**
Per-ticker optimization on historical data creates the illusion of performance through overfitting. The regime-specific parameter sets (9 regime labels × ~4 parameters each) were set using economic logic, not optimization. Walk-forward validation with 7 non-overlapping windows confirmed the parameters generalize.

**Why keep RELIANCE despite negative Sharpe?**
RELIANCE.NS is India's largest company by market cap and a core component of any NSE portfolio. Its negative Sharpe reflects the system's signals being poorly suited to a macro-driven conglomerate — not that the stock itself is untradeable. The Macro Sentinel's credit spread and yield curve signals are theoretically better aligned with RELIANCE's drivers (oil, debt markets), which future iterations will explore.

**Why was v5 (factor model) dropped?**
A 2-year rolling window produced insufficient variance in returns for PCA-based factor classification. All 11 tickers were classified as MOMENTUM, collapsing the system to a single regime. The lesson: factor models require 5+ years of returns data to produce stable factor loadings. Documented as an explicit failure case.

---

## What This Demonstrates for MFE Applications

This project is the result of a deliberate learning agenda, not a polished finished product:

1. **Regime awareness over signal optimization** — the largest performance gains came from regime detection (v1→v2: MaxDD 36%→22%) not from adding more signals

2. **Principled failure** — v5 was built, tested, found to be degenerate, and discarded. The commit history shows this. Knowing *when to stop* is as important as knowing what to build

3. **Overfitting discipline** — the 2008–2014 period was sealed off before any development began and opened only once for final validation. Degradation < 15% confirms the signal stack generalizes

4. **Production thinking** — trailing stops, correlation limits, circuit breakers, and the VIX gate were not added to improve backtest metrics. They were added because live trading requires self-protection mechanisms that backtests cannot fully capture

---

## Limitations and Future Work

- **US-centric macro signals** — VIX term structure, HYG/LQD, and Treasury yields are US market indicators. For NSE tickers, these are blended 50/50 with India VIX, but a dedicated India macro layer (G-Sec yield curve, INR/USD vol, FII flow data) would be more precise

- **No execution alpha** — paper trading uses next-open fills with 0.1% slippage. Real execution on illiquid NSE mid-caps would face wider spreads

- **RELIANCE signal mismatch** — the system's technical indicators are poorly suited to a stock driven by commodity prices and regulatory events. A macro-factor overlay specific to energy and telecom would address this

- **Zerodha Kite API integration** — the paper trading engine is structured to accept real fills with minimal code changes. Live execution is the natural next step after one quarter of paper trading

---

*Built over 9 development sessions as a structured learning project in quantitative systems design.*
*Contact: [your email] | [LinkedIn]*
