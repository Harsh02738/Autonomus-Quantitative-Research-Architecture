"""
AQRA — Agent 3: Risk Manager Agent
=====================================
Responsibility: Take a SignalReport from Agent 2 and decide:
  1. Should we trade this at all? (risk gate)
  2. If yes — how much capital to allocate? (position sizing)
  3. Where is the stop-loss and take-profit? (risk/reward)

Methods used:
  - Kelly Criterion        → mathematically optimal bet sizing
  - Portfolio heat check   → total capital at risk across all positions
  - Volatility-adjusted sizing → scale down in high-vol regimes

Output: TradeDecision — consumed by Agent 4 (Execution)
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone

# Agent 3 depends on Agent 2's output contract
from agent2 import SignalResearchAgent, SignalReport, Signal
from agent1 import DataIngestionAgent, MarketSnapshot


# ─────────────────────────────────────────────
#  Data Structures
# ─────────────────────────────────────────────

@dataclass
class TradeDecision:
    """
    Normalized output of Agent 3.
    Agent 4 (Execution) consumes this — it never sees raw signals.
    """
    ticker: str
    exchange: str
    currency: str
    action: str             # "BUY" | "SELL" | "HOLD"
    approved: bool          # False = risk gate rejected the trade

    entry_price: float
    stop_loss: float        # price at which we exit if wrong
    take_profit: float      # target price

    capital_total: float    # total portfolio capital (INR or USD)
    capital_to_risk: float  # INR/USD amount we're willing to lose
    position_size: float    # number of shares/units to buy
    position_value: float   # total value of the position

    kelly_fraction: float   # raw kelly output (for reference)
    risk_reward_ratio: float
    rejection_reason: str = ""   # why it was rejected (if approved=False)
    decided_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def summary(self):
        status = "✓ APPROVED" if self.approved else f"✗ REJECTED — {self.rejection_reason}"
        rr = f"{self.risk_reward_ratio:.1f}:1"
        print(f"""
┌──────────────────────────────────────────────────┐
│  Trade Decision: {self.ticker:<10}  [{status}]
│  Action     : {self.action}
│  Entry      : {self.currency} {self.entry_price:>10,.2f}
│  Stop Loss  : {self.currency} {self.stop_loss:>10,.2f}   (exit if wrong)
│  Take Profit: {self.currency} {self.take_profit:>10,.2f}   (target)
│  R/R Ratio  : {rr}
├──────────────────────────────────────────────────┤
│  Portfolio Capital : {self.currency} {self.capital_total:>12,.2f}
│  Capital at Risk   : {self.currency} {self.capital_to_risk:>12,.2f}  ({self.capital_to_risk/self.capital_total:.1%} of portfolio)
│  Position Size     : {self.position_size:>8.2f} units
│  Position Value    : {self.currency} {self.position_value:>12,.2f}
│  Kelly Fraction    : {self.kelly_fraction:.3f}  (raw, before caps)
└──────────────────────────────────────────────────┘""")


# ─────────────────────────────────────────────
#  The Agent Class
# ─────────────────────────────────────────────

class RiskManagerAgent:
    """
    Agent 3 — Risk Manager.

    Usage:
        agent = RiskManagerAgent(capital=1_000_000)
        decision = agent.evaluate(signal_report)

    Args:
        capital          : total portfolio value in base currency
        max_risk_per_trade: max % of portfolio to risk on one trade (default 2%)
        max_portfolio_heat: max % of portfolio at risk across ALL open trades (default 10%)
        min_confidence   : minimum signal confidence to even consider trading (default 50%)
        min_rr_ratio     : minimum risk/reward ratio required (default 1.5)
    """

    def __init__(
        self,
        capital: float = 1_000_000,
        max_risk_per_trade: float = 0.02,
        max_portfolio_heat: float = 0.10,
        min_confidence: float = 0.50,
        min_rr_ratio: float = 1.5,
    ):
        self.capital            = capital
        self.max_risk_per_trade = max_risk_per_trade
        self.max_portfolio_heat = max_portfolio_heat
        self.min_confidence     = min_confidence
        self.min_rr_ratio       = min_rr_ratio
        self.open_positions: list[TradeDecision] = []  # track live trades

        print(f"[RiskAgent] Initialized  capital={capital:,.0f}  max_risk/trade={max_risk_per_trade:.0%}  min_confidence={min_confidence:.0%}")


    def evaluate(self, report: SignalReport) -> TradeDecision:
        """
        Main method. Takes a SignalReport → returns a TradeDecision.
        """
        print(f"[RiskAgent] Evaluating {report.ticker}  signal={report.final_direction}  confidence={report.confidence:.0%}")

        price = report.current_price

        # ── Step 1: Hard gates — don't even calculate if these fail ──
        rejection = self._check_gates(report)
        if rejection:
            return self._rejected_decision(report, rejection)

        # ── Step 2: Compute stop-loss and take-profit levels ──
        stop_loss, take_profit = self._compute_levels(report)
        risk_per_share  = abs(price - stop_loss)
        reward_per_share= abs(take_profit - price)
        rr_ratio        = reward_per_share / risk_per_share if risk_per_share > 0 else 0

        # ── Step 3: Check risk/reward is acceptable ──
        if rr_ratio < self.min_rr_ratio:
            return self._rejected_decision(report, f"R/R ratio {rr_ratio:.2f} below minimum {self.min_rr_ratio}")

        # ── Step 4: Kelly Criterion for position sizing ──
        kelly_fraction = self._kelly(report)

        # ── Step 5: Apply caps (never bet the full Kelly — use half Kelly) ──
        # Half-Kelly is standard practice: reduces variance significantly
        capped_fraction = min(kelly_fraction * 0.5, self.max_risk_per_trade)

        # ── Step 6: Portfolio heat check ──
        current_heat = self._portfolio_heat()
        if current_heat + capped_fraction > self.max_portfolio_heat:
            return self._rejected_decision(
                report,
                f"Portfolio heat {current_heat:.1%} + {capped_fraction:.1%} exceeds max {self.max_portfolio_heat:.1%}"
            )

        # ── Step 7: Compute actual position ──
        capital_to_risk = self.capital * capped_fraction
        position_size   = capital_to_risk / risk_per_share   # shares we can buy
        position_value  = position_size * price

        decision = TradeDecision(
            ticker            = report.ticker,
            exchange          = report.exchange,
            currency          = report.currency,
            action            = report.final_direction,
            approved          = True,
            entry_price       = price,
            stop_loss         = round(stop_loss, 4),
            take_profit       = round(take_profit, 4),
            capital_total     = self.capital,
            capital_to_risk   = round(capital_to_risk, 2),
            position_size     = round(position_size, 4),
            position_value    = round(position_value, 2),
            kelly_fraction    = round(kelly_fraction, 4),
            risk_reward_ratio = round(rr_ratio, 3),
        )

        self.open_positions.append(decision)
        print(f"[RiskAgent] ✓ APPROVED  {report.ticker}  size={position_size:.2f} units  risk={capital_to_risk:,.0f}  R/R={rr_ratio:.1f}:1")
        return decision


    def evaluate_multiple(self, reports: dict[str, SignalReport]) -> dict[str, TradeDecision]:
        """Evaluate a batch of SignalReports."""
        decisions = {}
        for ticker, report in reports.items():
            decisions[ticker] = self.evaluate(report)
        return decisions


    def portfolio_status(self):
        """Print current portfolio risk summary."""
        approved = [p for p in self.open_positions if p.approved]
        total_risk = sum(p.capital_to_risk for p in approved)
        total_value = sum(p.position_value for p in approved)
        heat = total_risk / self.capital if self.capital else 0

        print(f"""
┌─────────────────────────────────────────────┐
│  PORTFOLIO STATUS                           │
│  Total Capital  : {self.capital:>12,.0f}             │
│  Open Positions : {len(approved):>3}                         │
│  Total Exposure : {total_value:>12,.0f}             │
│  Capital at Risk: {total_risk:>12,.0f}  ({heat:.1%})     │
├─────────────────────────────────────────────┤""")
        for p in approved:
            print(f"│  {p.ticker:<12} {p.action:<5} {p.position_size:>8.2f} units  risk={p.capital_to_risk:>10,.0f}  │")
        print("└─────────────────────────────────────────────┘")


    # ─────────────────────────────────────────
    #  Internal Methods
    # ─────────────────────────────────────────

    def _check_gates(self, report: SignalReport) -> str:
        """Hard filters. Returns rejection reason string, or '' if pass."""

        # Gate 1: Don't trade NEUTRAL signals
        if report.final_direction == "NEUTRAL":
            return "Signal is NEUTRAL — no directional edge"

        # Gate 2: Confidence too low
        if report.confidence < self.min_confidence:
            return f"Confidence {report.confidence:.0%} below minimum {self.min_confidence:.0%}"

        # Gate 3: Don't trade the same ticker twice
        existing = [p.ticker for p in self.open_positions if p.approved]
        if report.ticker in existing:
            return f"Already have an open position in {report.ticker}"

        return ""   # all gates passed


    def _compute_levels(self, report: SignalReport) -> tuple[float, float]:
        """
        Compute stop-loss and take-profit using ATR-style fixed % levels.
        Simple but effective for daily timeframe.

        BUY:  stop = entry - 3%,  target = entry + 6%  (2:1 R/R)
        SELL: stop = entry + 3%,  target = entry - 6%
        """
        price = report.current_price
        stop_pct   = 0.03   # 3% stop loss
        target_pct = 0.06   # 6% take profit → 2:1 R/R

        if report.final_direction == "BUY":
            stop_loss   = price * (1 - stop_pct)
            take_profit = price * (1 + target_pct)
        else:  # SELL
            stop_loss   = price * (1 + stop_pct)
            take_profit = price * (1 - target_pct)

        return stop_loss, take_profit


    def _kelly(self, report: SignalReport) -> float:
        """
        Kelly Criterion: f* = (bp - q) / b
        where:
          p = probability of winning (we use signal confidence as proxy)
          q = 1 - p (probability of losing)
          b = reward/risk ratio (we use 2.0 for our fixed 3%/6% levels)

        Kelly tells us the mathematically optimal fraction of capital to bet.
        We then apply half-Kelly in practice (standard risk management).
        """
        p = report.confidence          # win probability proxy
        q = 1 - p
        b = 2.0                        # reward/risk = 6% / 3%

        kelly = (b * p - q) / b
        return max(kelly, 0.01)         # Kelly can be negative — floor at 0.01 


    def _portfolio_heat(self) -> float:
        """Current fraction of capital at risk across all open positions."""
        approved = [p for p in self.open_positions if p.approved]
        total_risk = sum(p.capital_to_risk for p in approved)
        return total_risk / self.capital if self.capital else 0.0


    def _rejected_decision(self, report: SignalReport, reason: str) -> TradeDecision:
        print(f"[RiskAgent] ✗ REJECTED  {report.ticker} — {reason}")
        return TradeDecision(
            ticker            = report.ticker,
            exchange          = report.exchange,
            currency          = report.currency,
            action            = report.final_direction,
            approved          = False,
            entry_price       = report.current_price,
            stop_loss         = 0.0,
            take_profit       = 0.0,
            capital_total     = self.capital,
            capital_to_risk   = 0.0,
            position_size     = 0.0,
            position_value    = 0.0,
            kelly_fraction    = 0.0,
            risk_reward_ratio = 0.0,
            rejection_reason  = reason,
        )


# ─────────────────────────────────────────────
#  Run directly to test
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # Full pipeline: Agent 1 → Agent 2 → Agent 3
    data_agent   = DataIngestionAgent()
    signal_agent = SignalResearchAgent()
    risk_agent = RiskManagerAgent(capital=1_000_000, min_confidence=0.30) # 10 lakh INR / $1M USD

    print("\n" + "="*55)
    print("TEST 1: Full pipeline on a single stock")
    print("="*55)
    snap     = data_agent.get_snapshot("HDFCBANK.NS", period="6mo")
    report   = signal_agent.analyze(snap)
    decision = risk_agent.evaluate(report)
    decision.summary()

    print("\n" + "="*55)
    print("TEST 2: Batch pipeline — portfolio of 5 stocks")
    print("="*55)
    tickers   = ["TCS.NS", "INFY.NS", "RELIANCE.NS", "AAPL", "NVDA"]
    snapshots = data_agent.get_multiple(tickers, period="6mo")
    reports   = signal_agent.analyze_multiple(snapshots)
    decisions = risk_agent.evaluate_multiple(reports)

    print("\n  FINAL DECISIONS:")
    print(f"  {'TICKER':<15} {'ACTION':<8} {'APPROVED':<10} {'UNITS':>8}  {'RISK':>12}  REASON")
    print("  " + "-"*75)
    for ticker, d in decisions.items():
        approved_str = "✓ YES" if d.approved else "✗ NO"
        reason = "" if d.approved else d.rejection_reason[:35]
        print(f"  {ticker:<15} {d.action:<8} {approved_str:<10} {d.position_size:>8.2f}  {d.currency} {d.capital_to_risk:>10,.0f}  {reason}")

    print()
    risk_agent.portfolio_status()
