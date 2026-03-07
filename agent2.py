"""
AQRA — Agent 2: Signal Research Agent
=======================================
Responsibility: Take a MarketSnapshot from Agent 1 and compute
trading signals. Output a SignalReport that Agent 3 (Risk) will use.

Signals implemented:
  1. Momentum       — is price trending?        (SMA crossover)
  2. Mean Reversion — is price overstretched?   (RSI)
  3. Volatility     — is price in a breakout?   (Bollinger Bands)

This agent does NOT decide position size or place orders.
Its only job: look at price history → produce a signal.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timezone

# Import Agent 1's output type — Agent 2 depends on Agent 1's contract
from agent1 import DataIngestionAgent, MarketSnapshot


# ─────────────────────────────────────────────
#  Data Structures (what this agent outputs)
# ─────────────────────────────────────────────

@dataclass
class Signal:
    """A single trading signal from one strategy."""
    name: str           # "momentum" | "mean_reversion" | "volatility"
    direction: str      # "BUY" | "SELL" | "NEUTRAL"
    strength: float     # 0.0 (weak) → 1.0 (strong)
    reason: str         # human-readable explanation
    value: float        # the raw indicator value (RSI score, etc.)


@dataclass
class SignalReport:
    """
    Normalized output of Agent 2.
    Agent 3 (Risk) will consume this — it never looks at raw price data.
    """
    ticker: str
    exchange: str
    current_price: float
    currency: str
    signals: list[Signal]
    final_direction: str    # majority vote across all signals
    confidence: float       # 0.0 → 1.0, how many signals agree
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def summary(self):
        bar = "█" * int(self.confidence * 20) + "░" * (20 - int(self.confidence * 20))
        print(f"""
┌──────────────────────────────────────────────┐
│  Signal Report: {self.ticker:<10} @ {self.currency} {self.current_price:<10,.2f} │
│  Final Call : {self.final_direction:<8}  Confidence: {self.confidence:.0%}          │
│  [{bar}]          │
├──────────────────────────────────────────────┤""")
        for s in self.signals:
            icon = "▲" if s.direction == "BUY" else ("▼" if s.direction == "SELL" else "─")
            print(f"│  {icon} {s.name:<18} {s.direction:<8} strength={s.strength:.2f}  │")
            print(f"│    {s.reason[:50]:<50} │")
        print("└──────────────────────────────────────────────┘")


# ─────────────────────────────────────────────
#  The Agent Class
# ─────────────────────────────────────────────

class SignalResearchAgent:
    """
    Agent 2 — Signal Research.

    Usage:
        agent = SignalResearchAgent()
        report = agent.analyze(snapshot)   # snapshot from Agent 1
    """

    def analyze(self, snapshot: MarketSnapshot) -> SignalReport:
        """
        Main method. Run all signal strategies on a MarketSnapshot.
        Returns a SignalReport.
        """
        print(f"[SignalAgent] Analyzing {snapshot.ticker}  ({len(snapshot.history)} candles)")

        # Convert history to a pandas DataFrame — easier to work with
        df = self._to_dataframe(snapshot)

        if len(df) < 30:
            raise ValueError(f"[SignalAgent] Need at least 30 candles, got {len(df)}. Use a longer period.")

        # Run each signal strategy
        signals = [
            self._momentum_signal(df),
            self._mean_reversion_signal(df),
            self._volatility_signal(df),
        ]

        # Aggregate: majority vote for final direction
        final_direction, confidence = self._aggregate(signals)

        report = SignalReport(
            ticker          = snapshot.ticker,
            exchange        = snapshot.exchange,
            current_price   = snapshot.current_price,
            currency        = snapshot.currency,
            signals         = signals,
            final_direction = final_direction,
            confidence      = confidence,
        )

        print(f"[SignalAgent] ✓ {snapshot.ticker} → {final_direction}  confidence={confidence:.0%}")
        return report


    def analyze_multiple(self, snapshots: dict[str, MarketSnapshot]) -> dict[str, SignalReport]:
        """Analyze a batch of snapshots. Skips failed tickers."""
        results = {}
        for ticker, snapshot in snapshots.items():
            try:
                results[ticker] = self.analyze(snapshot)
            except Exception as e:
                print(f"[SignalAgent] ✗ Skipping {ticker}: {e}")
        return results


    # ─────────────────────────────────────────
    #  Signal Strategies
    # ─────────────────────────────────────────

    def _momentum_signal(self, df: pd.DataFrame) -> Signal:
        """
        Strategy: SMA Crossover (20-day vs 50-day)
        Logic:
          - If 20 SMA > 50 SMA → price is trending UP → BUY
          - If 20 SMA < 50 SMA → price is trending DOWN → SELL
          - Strength = how far apart the SMAs are (normalized)
        """
        sma20 = df["close"].rolling(20).mean().iloc[-1]
        sma50 = df["close"].rolling(50).mean().iloc[-1] if len(df) >= 50 else df["close"].rolling(20).mean().iloc[-1]

        gap_pct = (sma20 - sma50) / sma50       # positive = bullish
        strength = min(abs(gap_pct) * 20, 1.0)  # scale to 0–1

        if gap_pct > 0.005:
            direction = "BUY"
            reason = f"20-SMA ({sma20:.2f}) above 50-SMA ({sma50:.2f}) by {gap_pct:.2%}"
        elif gap_pct < -0.005:
            direction = "SELL"
            reason = f"20-SMA ({sma20:.2f}) below 50-SMA ({sma50:.2f}) by {abs(gap_pct):.2%}"
        else:
            direction = "NEUTRAL"
            reason = f"20-SMA and 50-SMA nearly equal (gap {gap_pct:.2%})"

        return Signal("momentum", direction, round(strength, 3), reason, round(sma20, 4))


    def _mean_reversion_signal(self, df: pd.DataFrame) -> Signal:
        """
        Strategy: RSI (Relative Strength Index, 14-day)
        Logic:
          - RSI > 70 → overbought → SELL (expect pullback)
          - RSI < 30 → oversold  → BUY  (expect bounce)
          - 30–70   → NEUTRAL
        """
        delta  = df["close"].diff()
        gain   = delta.clip(lower=0).rolling(14).mean()
        loss   = (-delta.clip(upper=0)).rolling(14).mean()
        rs     = gain / loss.replace(0, np.nan)
        rsi    = (100 - (100 / (1 + rs))).iloc[-1]

        if rsi > 70:
            direction = "SELL"
            strength  = min((rsi - 70) / 30, 1.0)
            reason    = f"RSI={rsi:.1f} — overbought (>70), mean reversion likely"
        elif rsi < 30:
            direction = "BUY"
            strength  = min((30 - rsi) / 30, 1.0)
            reason    = f"RSI={rsi:.1f} — oversold (<30), bounce expected"
        else:
            direction = "NEUTRAL"
            strength  = 0.2
            reason    = f"RSI={rsi:.1f} — neutral zone (30–70)"

        return Signal("mean_reversion", direction, round(strength, 3), reason, round(rsi, 2))


    def _volatility_signal(self, df: pd.DataFrame) -> Signal:
        """
        Strategy: Bollinger Bands (20-day, 2 std deviations)
        Logic:
          - Price above upper band → breakout UP → BUY
          - Price below lower band → breakout DOWN → SELL
          - Price inside bands    → NEUTRAL
        Strength = how far outside the band the price is
        """
        sma    = df["close"].rolling(20).mean()
        std    = df["close"].rolling(20).std()
        upper  = (sma + 2 * std).iloc[-1]
        lower  = (sma - 2 * std).iloc[-1]
        price  = df["close"].iloc[-1]
        mid    = sma.iloc[-1]
        band_width = upper - lower

        if price > upper:
            direction = "BUY"
            strength  = min((price - upper) / (band_width * 0.1 + 1e-9), 1.0)
            reason    = f"Price {price:.2f} above upper band {upper:.2f} — upside breakout"
        elif price < lower:
            direction = "SELL"
            strength  = min((lower - price) / (band_width * 0.1 + 1e-9), 1.0)
            reason    = f"Price {price:.2f} below lower band {lower:.2f} — downside breakout"
        else:
            direction = "NEUTRAL"
            # Position within band: 0 = at lower, 1 = at upper
            pos_in_band = (price - lower) / (band_width + 1e-9)
            strength = 0.1
            reason = f"Price inside bands [{lower:.2f} – {upper:.2f}], position={pos_in_band:.0%}"

        return Signal("volatility", direction, round(strength, 3), reason, round(price, 4))


    # ─────────────────────────────────────────
    #  Helpers
    # ─────────────────────────────────────────

    def _aggregate(self, signals: list[Signal]) -> tuple[str, float]:
        """
        Majority vote across all signals.
        Confidence = fraction of signals that agree with the majority.
        """
        counts = {"BUY": 0, "SELL": 0, "NEUTRAL": 0}
        for s in signals:
            counts[s.direction] += 1

        final = max(counts, key=counts.get)
        confidence = counts[final] / len(signals)
        return final, round(confidence, 3)


    def _to_dataframe(self, snapshot: MarketSnapshot) -> pd.DataFrame:
        """Convert MarketSnapshot.history → clean pandas DataFrame."""
        rows = [{
            "timestamp": c.timestamp,
            "open":  c.open,
            "high":  c.high,
            "low":   c.low,
            "close": c.close,
            "volume":c.volume,
        } for c in snapshot.history]
        df = pd.DataFrame(rows).set_index("timestamp").sort_index()
        return df


# ─────────────────────────────────────────────
#  Run directly to test
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # Agent 1 feeds Agent 2 — this is the pipeline starting to form
    data_agent   = DataIngestionAgent()
    signal_agent = SignalResearchAgent()

    print("\n" + "="*50)
    print("TEST 1: Analyze a single NSE stock")
    print("="*50)
    snap   = data_agent.get_snapshot("HDFCBANK.NS", period="6mo")
    report = signal_agent.analyze(snap)
    report.summary()

    print("\n" + "="*50)
    print("TEST 2: Analyze a US stock")
    print("="*50)
    snap2   = data_agent.get_snapshot("NVDA", period="6mo")
    report2 = signal_agent.analyze(snap2)
    report2.summary()

    print("\n" + "="*50)
    print("TEST 3: Batch analyze a mini portfolio")
    print("="*50)
    tickers   = ["TCS.NS", "INFY.NS", "AAPL", "MSFT"]
    snapshots = data_agent.get_multiple(tickers, period="6mo")
    reports   = signal_agent.analyze_multiple(snapshots)

    print("\n  PORTFOLIO SUMMARY:")
    print(f"  {'TICKER':<15} {'SIGNAL':<8} {'CONFIDENCE':<12} {'PRICE'}")
    print("  " + "-"*50)
    for ticker, r in reports.items():
        icon = "▲" if r.final_direction == "BUY" else ("▼" if r.final_direction == "SELL" else "─")
        print(f"  {ticker:<15} {icon} {r.final_direction:<6}  {f'{r.confidence:.0%}':<12} {r.currency} {r.current_price:,.2f}")