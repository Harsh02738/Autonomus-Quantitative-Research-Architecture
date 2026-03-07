"""
AQRA — Agent 4: Execution Agent
=================================
Responsibility: Take a TradeDecision from Agent 3 and:
  1. Simulate order fill (paper trading)
  2. Track open positions over time
  3. Mark-to-market P&L as prices update
  4. Trigger stop-loss / take-profit exits automatically
  5. Produce a full trade log and performance summary

This agent does NOT generate signals or size positions.
Its only job: execute what Agent 3 approved, track it, report it.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from agent3 import RiskManagerAgent, TradeDecision
from agent2 import SignalResearchAgent
from agent1 import DataIngestionAgent


# ─────────────────────────────────────────────
#  Enums & Data Structures
# ─────────────────────────────────────────────

class OrderStatus(Enum):
    FILLED    = "FILLED"
    REJECTED  = "REJECTED"
    OPEN      = "OPEN"
    CLOSED    = "CLOSED"


@dataclass
class Order:
    """Represents a single order placed by the execution agent."""
    order_id: str
    ticker: str
    exchange: str
    currency: str
    action: str               # "BUY" | "SELL"
    requested_price: float    # price when decision was made
    fill_price: float         # actual fill (with slippage)
    slippage_pct: float       # how much we paid above/below
    units: float
    position_value: float
    stop_loss: float
    take_profit: float
    status: OrderStatus
    placed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Position:
    """A live open position being tracked."""
    order: Order
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    status: OrderStatus       # OPEN | CLOSED
    exit_price: float = 0.0
    exit_reason: str = ""     # "STOP_LOSS" | "TAKE_PROFIT" | "MANUAL"
    realized_pnl: float = 0.0
    closed_at: datetime = None


@dataclass
class ExecutionReport:
    """Full summary produced after running the execution agent."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_realized_pnl: float
    total_unrealized_pnl: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float      # gross wins / gross losses
    positions: list[Position]

    def summary(self):
        pf = f"{self.profit_factor:.2f}" if self.profit_factor != float('inf') else "∞"
        print(f"""
╔══════════════════════════════════════════════════╗
║           EXECUTION REPORT                      ║
╠══════════════════════════════════════════════════╣
║  Total Trades     : {self.total_trades:<5}                        ║
║  Winning Trades   : {self.winning_trades:<5}  Losing: {self.losing_trades:<5}            ║
║  Win Rate         : {self.win_rate:.1%}                         ║
║  Realized P&L     : {self.total_realized_pnl:>+12,.2f}                  ║
║  Unrealized P&L   : {self.total_unrealized_pnl:>+12,.2f}                  ║
║  Avg Win          : {self.avg_win:>+12,.2f}                  ║
║  Avg Loss         : {self.avg_loss:>+12,.2f}                  ║
║  Profit Factor    : {pf:<8}                        ║
╚══════════════════════════════════════════════════╝""")

        print("\n  POSITION LOG:")
        print(f"  {'TICKER':<14} {'ACTION':<6} {'UNITS':>7}  {'ENTRY':>10}  {'CURRENT':>10}  {'P&L':>10}  {'STATUS'}")
        print("  " + "─" * 75)
        for p in self.positions:
            o = p.order
            pnl = p.realized_pnl if p.status == OrderStatus.CLOSED else p.unrealized_pnl
            pnl_str = f"{pnl:>+10,.2f}"
            status = f"{p.status.value}"
            if p.status == OrderStatus.CLOSED:
                status += f" ({p.exit_reason})"
            price = p.exit_price if p.status == OrderStatus.CLOSED else p.current_price
            print(f"  {o.ticker:<14} {o.action:<6} {o.units:>7.2f}  {o.fill_price:>10,.2f}  {price:>10,.2f}  {pnl_str}  {status}")


# ─────────────────────────────────────────────
#  The Agent Class
# ─────────────────────────────────────────────

class ExecutionAgent:
    """
    Agent 4 — Execution (Paper Trading).

    Usage:
        agent = ExecutionAgent(slippage_pct=0.001)
        order = agent.execute(trade_decision)
        agent.update_prices({"TCS.NS": 2600.0})
        report = agent.get_report()

    Args:
        slippage_pct: simulated market impact (0.1% default — realistic for liquid stocks)
    """

    def __init__(self, slippage_pct: float = 0.001):
        self.slippage_pct = slippage_pct
        self.positions: list[Position] = []
        self._order_counter = 0
        print(f"[ExecAgent] Initialized  mode=PAPER  slippage={slippage_pct:.2%}")


    def execute(self, decision: TradeDecision) -> Order | None:
        """
        Main method. Takes a TradeDecision → simulates a fill → opens a position.
        Returns None if the decision was rejected by Agent 3.
        """
        if not decision.approved:
            print(f"[ExecAgent] Skipping {decision.ticker} — not approved by Risk Agent")
            return None

        self._order_counter += 1
        order_id = f"ORD-{self._order_counter:04d}"

        # Simulate slippage — BUY fills slightly above, SELL fills slightly below
        if decision.action == "BUY":
            fill_price = decision.entry_price * (1 + self.slippage_pct)
        else:
            fill_price = decision.entry_price * (1 - self.slippage_pct)

        slippage_cost = abs(fill_price - decision.entry_price) * decision.position_size

        order = Order(
            order_id        = order_id,
            ticker          = decision.ticker,
            exchange        = decision.exchange,
            currency        = decision.currency,
            action          = decision.action,
            requested_price = decision.entry_price,
            fill_price      = round(fill_price, 4),
            slippage_pct    = self.slippage_pct,
            units           = decision.position_size,
            position_value  = round(fill_price * decision.position_size, 2),
            stop_loss       = decision.stop_loss,
            take_profit     = decision.take_profit,
            status          = OrderStatus.FILLED,
        )

        position = Position(
            order               = order,
            current_price       = fill_price,
            unrealized_pnl      = 0.0,
            unrealized_pnl_pct  = 0.0,
            status              = OrderStatus.OPEN,
        )

        self.positions.append(position)

        print(f"[ExecAgent] ✓ FILLED  {order_id}  {decision.ticker}  {decision.action}  "
              f"{decision.position_size:.2f} units @ {order.currency} {fill_price:,.2f}  "
              f"(slippage cost: {order.currency} {slippage_cost:.2f})")

        return order


    def execute_multiple(self, decisions: dict[str, TradeDecision]) -> list[Order]:
        """Execute a batch of decisions."""
        orders = []
        for ticker, decision in decisions.items():
            order = self.execute(decision)
            if order:
                orders.append(order)
        return orders


    def update_prices(self, prices: dict[str, float]) -> None:
        """
        Feed new prices into the agent.
        It will:
          - Update unrealized P&L for each open position
          - Auto-trigger stop-loss or take-profit if hit
        """
        for position in self.positions:
            if position.status != OrderStatus.OPEN:
                continue

            ticker = position.order.ticker
            if ticker not in prices:
                continue

            new_price = prices[ticker]
            position.current_price = new_price
            o = position.order

            # ── Mark to market ──
            if o.action == "BUY":
                position.unrealized_pnl = (new_price - o.fill_price) * o.units
            else:  # SELL
                position.unrealized_pnl = (o.fill_price - new_price) * o.units

            position.unrealized_pnl_pct = position.unrealized_pnl / o.position_value

            # ── Check stop-loss ──
            stop_hit = (o.action == "BUY"  and new_price <= o.stop_loss) or \
                       (o.action == "SELL" and new_price >= o.stop_loss)

            # ── Check take-profit ──
            tp_hit   = (o.action == "BUY"  and new_price >= o.take_profit) or \
                       (o.action == "SELL" and new_price <= o.take_profit)

            if stop_hit:
                self._close_position(position, new_price, "STOP_LOSS")
            elif tp_hit:
                self._close_position(position, new_price, "TAKE_PROFIT")
            else:
                print(f"[ExecAgent] MTM  {ticker}  {o.currency} {new_price:,.2f}  "
                      f"P&L: {o.currency} {position.unrealized_pnl:+,.2f}  ({position.unrealized_pnl_pct:+.2%})")


    def _close_position(self, position: Position, exit_price: float, reason: str) -> None:
        """Close a position and lock in realized P&L."""
        o = position.order
        if o.action == "BUY":
            realized = (exit_price - o.fill_price) * o.units
        else:
            realized = (o.fill_price - exit_price) * o.units

        position.exit_price    = exit_price
        position.exit_reason   = reason
        position.realized_pnl  = round(realized, 2)
        position.unrealized_pnl= 0.0
        position.status        = OrderStatus.CLOSED
        position.closed_at     = datetime.now(timezone.utc)

        icon = "✓" if realized >= 0 else "✗"
        print(f"[ExecAgent] {icon} CLOSED  {o.ticker}  reason={reason}  "
              f"exit={o.currency} {exit_price:,.2f}  "
              f"realized P&L={o.currency} {realized:+,.2f}")


    def get_report(self) -> ExecutionReport:
        """Generate a full performance report across all positions."""
        closed   = [p for p in self.positions if p.status == OrderStatus.CLOSED]
        open_pos = [p for p in self.positions if p.status == OrderStatus.OPEN]

        winners = [p for p in closed if p.realized_pnl > 0]
        losers  = [p for p in closed if p.realized_pnl <= 0]

        total_realized   = sum(p.realized_pnl for p in closed)
        total_unrealized = sum(p.unrealized_pnl for p in open_pos)

        avg_win  = sum(p.realized_pnl for p in winners) / len(winners) if winners else 0
        avg_loss = sum(p.realized_pnl for p in losers)  / len(losers)  if losers  else 0

        gross_wins   = sum(p.realized_pnl for p in winners)
        gross_losses = abs(sum(p.realized_pnl for p in losers))
        profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')

        win_rate = len(winners) / len(closed) if closed else 0.0

        return ExecutionReport(
            total_trades        = len(self.positions),
            winning_trades      = len(winners),
            losing_trades       = len(losers),
            total_realized_pnl  = round(total_realized, 2),
            total_unrealized_pnl= round(total_unrealized, 2),
            win_rate            = win_rate,
            avg_win             = round(avg_win, 2),
            avg_loss            = round(avg_loss, 2),
            profit_factor       = round(profit_factor, 3),
            positions           = self.positions,
        )


# ─────────────────────────────────────────────
#  Run directly to test
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # Full pipeline: Agent 1 → 2 → 3 → 4
    data_agent  = DataIngestionAgent()
    signal_agent= SignalResearchAgent()
    risk_agent  = RiskManagerAgent(capital=1_000_000, min_confidence=0.30)
    exec_agent  = ExecutionAgent(slippage_pct=0.001)

    # ── Fetch + analyze + risk-check ──
    tickers   = ["TCS.NS", "INFY.NS", "RELIANCE.NS", "AAPL", "NVDA"]
    snapshots = data_agent.get_multiple(tickers, period="6mo")
    reports   = signal_agent.analyze_multiple(snapshots)
    decisions = risk_agent.evaluate_multiple(reports)

    print("\n" + "="*55)
    print("STEP 1: Execute all approved trades")
    print("="*55)
    orders = exec_agent.execute_multiple(decisions)

    # ── Simulate price moves ──
    # We'll test 3 scenarios: price goes up, hits stop, hits target

    print("\n" + "="*55)
    print("STEP 2: Simulate price update — small move up")
    print("="*55)
    # TCS entry was ~2578. Let's move it up 1%
    exec_agent.update_prices({"TCS.NS": 2604.0})

    print("\n" + "="*55)
    print("STEP 3: Simulate stop-loss being hit")
    print("="*55)
    # TCS stop is at entry - 3% = ~2501. Let's crash it there
    tcs_stop = snapshots["TCS.NS"].current_price * 0.97
    exec_agent.update_prices({"TCS.NS": round(tcs_stop - 5, 2)})

    print("\n" + "="*55)
    print("STEP 4: Final execution report")
    print("="*55)
    report = exec_agent.get_report()
    report.summary()

    # ── Now test a take-profit scenario with a fresh agent ──
    print("\n" + "="*55)
    print("STEP 5: Fresh trade — simulate take-profit hit")
    print("="*55)
    exec_agent2 = ExecutionAgent(slippage_pct=0.001)
    snap2    = data_agent.get_snapshot("TCS.NS", period="6mo")
    report2  = signal_agent.analyze(snap2)
    decision2= risk_agent.evaluate(report2)
    exec_agent2.execute(decision2)

    # TCS take profit = entry + 6% = ~2733
    tcs_tp = snapshots["TCS.NS"].current_price * 1.06
    print(f"\n  Simulating price hitting take-profit: {tcs_tp:,.2f}")
    exec_agent2.update_prices({"TCS.NS": round(tcs_tp + 5, 2)})

    report3 = exec_agent2.get_report()
    report3.summary()
