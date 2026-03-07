"""
AQRA — Agent 1: Data Ingestion Agent
=====================================
Responsibility: Fetch, normalize, and serve market data.
Other agents NEVER call yfinance directly — they ask this agent.

Supported:
  - US Equities (NYSE/NASDAQ) via yfinance
  - Indian Equities (NSE) via yfinance  e.g. "RELIANCE.NS"
  - Basic options chain (US only via yfinance)

Output format: Always a normalized Python dict (MarketSnapshot)
so downstream agents don't care about the data source.
"""

import yfinance as yf
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime , timezone
from typing import Optional


# ─────────────────────────────────────────────
#  Data Structures  (what this agent outputs)
# ─────────────────────────────────────────────

@dataclass
class OHLCV:
    """One row of price data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int

@dataclass
class MarketSnapshot:
    """
    Normalized output of this agent.
    Every other agent works with this — never raw yfinance objects.
    """
    ticker: str
    exchange: str                        # "NSE" | "NYSE" | "NASDAQ"
    currency: str                        # "INR" | "USD"
    current_price: float
    day_change_pct: float
    history: list[OHLCV]                 # sorted oldest → newest
    metadata: dict = field(default_factory=dict)  # PE, market cap, etc.
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ─────────────────────────────────────────────
#  The Agent Class
# ─────────────────────────────────────────────

class DataIngestionAgent:
    """
    Agent 1 — Data Ingestion.

    Usage:
        agent = DataIngestionAgent()
        snapshot = agent.get_snapshot("RELIANCE.NS", period="3mo")
        snapshot = agent.get_snapshot("AAPL", period="1y")
    """

    # Map yfinance exchange codes → readable names
    EXCHANGE_MAP = {
        "NSI": "NSE", "BSE": "BSE",
        "NMS": "NASDAQ", "NYQ": "NYSE",
        "NGM": "NASDAQ", "PCX": "NYSE",
    }

    def get_snapshot(self, ticker: str, period: str = "3mo", interval: str = "1d") -> MarketSnapshot:
        """
        Main method. Fetch + normalize data for a ticker.

        Args:
            ticker:   e.g. "AAPL", "RELIANCE.NS", "TCS.NS", "INFY.NS"
            period:   "1mo", "3mo", "6mo", "1y", "2y", "5y"
            interval: "1d", "1wk", "1mo"  (use "1d" for most cases)

        Returns:
            MarketSnapshot  — ready for other agents to consume
        """
        print(f"[DataAgent] Fetching {ticker}  period={period}  interval={interval}")

        tkr = yf.Ticker(ticker)
        info = tkr.info
        hist = tkr.history(period=period, interval=interval)

        if hist.empty:
            raise ValueError(f"[DataAgent] No data returned for {ticker}. Check the ticker symbol.")

        # ── Detect exchange & currency ──
        raw_exchange = info.get("exchange", "")
        exchange = self.EXCHANGE_MAP.get(raw_exchange, raw_exchange or "UNKNOWN")
        currency = info.get("currency", "USD")

        # ── Current price & day change ──
        current_price = info.get("currentPrice") or info.get("regularMarketPrice") or float(hist["Close"].iloc[-1])
        prev_close    = info.get("previousClose") or info.get("regularMarketPreviousClose") or float(hist["Close"].iloc[-2])
        day_change_pct = ((current_price - prev_close) / prev_close) * 100 if prev_close else 0.0

        # ── Build OHLCV list ──
        history = []
        for ts, row in hist.iterrows():
            history.append(OHLCV(
                timestamp = ts.to_pydatetime(),
                open      = round(float(row["Open"]), 4),
                high      = round(float(row["High"]), 4),
                low       = round(float(row["Low"]), 4),
                close     = round(float(row["Close"]), 4),
                volume    = int(row["Volume"]),
            ))

        # ── Metadata (useful for Signal Agent later) ──
        metadata = {
            "company_name" : info.get("longName", ticker),
            "sector"       : info.get("sector", "N/A"),
            "industry"     : info.get("industry", "N/A"),
            "market_cap"   : info.get("marketCap"),
            "pe_ratio"     : info.get("trailingPE"),
            "52w_high"     : info.get("fiftyTwoWeekHigh"),
            "52w_low"      : info.get("fiftyTwoWeekLow"),
            "avg_volume"   : info.get("averageVolume"),
        }

        snapshot = MarketSnapshot(
            ticker        = ticker.upper(),
            exchange      = exchange,
            currency      = currency,
            current_price = round(current_price, 4),
            day_change_pct= round(day_change_pct, 3),
            history       = history,
            metadata      = metadata,
        )

        print(f"[DataAgent] ✓ {ticker} | {exchange} | {currency} {current_price:.2f} | {day_change_pct:+.2f}% | {len(history)} candles")
        return snapshot


    def get_multiple(self, tickers: list[str], period: str = "3mo") -> dict[str, MarketSnapshot]:
        """
        Fetch several tickers at once.
        Returns a dict: { ticker: MarketSnapshot }
        Failed tickers are skipped with a warning.
        """
        results = {}
        for ticker in tickers:
            try:
                results[ticker] = self.get_snapshot(ticker, period=period)
            except Exception as e:
                print(f"[DataAgent] ✗ Skipping {ticker}: {e}")
        return results


    def get_options_chain(self, ticker: str) -> dict:
        """
        Fetch options chain for a US ticker (NSE options not supported by yfinance).
        Returns: { expiry_date: { "calls": DataFrame, "puts": DataFrame } }
        """
        print(f"[DataAgent] Fetching options chain for {ticker}")
        tkr = yf.Ticker(ticker)
        expiries = tkr.options

        if not expiries:
            raise ValueError(f"[DataAgent] No options data for {ticker}.")

        chain = {}
        for exp in expiries[:3]:  # first 3 expiries to avoid hammering the API
            opt = tkr.option_chain(exp)
            chain[exp] = {
                "calls": opt.calls[["strike", "lastPrice", "bid", "ask", "impliedVolatility", "openInterest", "volume"]],
                "puts" : opt.puts[["strike", "lastPrice", "bid", "ask", "impliedVolatility", "openInterest", "volume"]],
            }
            print(f"[DataAgent] ✓ Options {ticker} expiry {exp} — {len(opt.calls)} calls, {len(opt.puts)} puts")

        return chain


    def summary(self, snapshot: MarketSnapshot) -> None:
        """Pretty-print a snapshot. Useful during development."""
        m = snapshot.metadata
        direction = "▲" if snapshot.day_change_pct >= 0 else "▼"
        print(f"""
┌─────────────────────────────────────────┐
│  {snapshot.ticker:<10}  {m.get('company_name','')[:26]:<26} │
│  Exchange : {snapshot.exchange:<10} Currency : {snapshot.currency}   │
│  Price    : {snapshot.current_price:<12,.2f}  {direction} {abs(snapshot.day_change_pct):.2f}%      │
│  Sector   : {str(m.get('sector','N/A'))[:30]:<30} │
│  PE Ratio : {str(m.get('pe_ratio','N/A')):<10}  MCap: {str(m.get('market_cap','N/A'))[:14]}  │
│  52W H/L  : {str(m.get('52w_high','N/A'))[:8]} / {str(m.get('52w_low','N/A'))[:10]}              │
│  Candles  : {len(snapshot.history)} ({snapshot.history[0].timestamp.date()} → {snapshot.history[-1].timestamp.date()}) │
└─────────────────────────────────────────┘""")


# ─────────────────────────────────────────────
#  Run this file directly to test the agent
# ─────────────────────────────────────────────

if __name__ == "__main__":

    agent = DataIngestionAgent()

    print("\n" + "="*50)
    print("TEST 1: Indian stock (NSE)")
    print("="*50)
    snap = agent.get_snapshot("RELIANCE.NS", period="3mo")
    agent.summary(snap)

    print("\n" + "="*50)
    print("TEST 2: US stock")
    print("="*50)
    snap2 = agent.get_snapshot("AAPL", period="3mo")
    agent.summary(snap2)

    print("\n" + "="*50)
    print("TEST 3: Fetch multiple tickers at once")
    print("="*50)
    portfolio = agent.get_multiple(["TCS.NS", "INFY.NS", "HDFCBANK.NS"], period="1mo")
    for ticker, s in portfolio.items():
        print(f"  {ticker}: {s.currency} {s.current_price:.2f}  ({s.day_change_pct:+.2f}%)")

    print("\n" + "="*50)
    print("TEST 4: Options chain (US only)")
    print("="*50)
    chain = agent.get_options_chain("SPY")
    for expiry, data in chain.items():
        print(f"  Expiry {expiry}: {len(data['calls'])} calls | {len(data['puts'])} puts")
        print(data["calls"].head(3).to_string(index=False))
