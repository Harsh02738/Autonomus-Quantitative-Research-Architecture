"""
AQRA — Agent 5: Orchestrator (Gemini LLM Brain)
=================================================
Responsibility: Central intelligence of the system.
Uses Google Gemini API to:
  1. Decide which tickers to research based on a plain-English goal
  2. Coordinate Agents 1-4 in the right sequence
  3. Reason about results — not just execute rules
  4. Produce a plain-English investment memo explaining every decision
"""
import os 
from dotenv import load_dotenv
import json
from google import genai
from datetime import datetime, timezone

from agent1 import DataIngestionAgent, MarketSnapshot
from agent2 import SignalResearchAgent, SignalReport
from agent3 import RiskManagerAgent, TradeDecision
from agent4 import ExecutionAgent, ExecutionReport
load_dotenv()


class OrchestratorAgent:
    """
    Agent 5 — LLM Orchestrator (Gemini-powered).

    Usage:
        orch = OrchestratorAgent(capital=1_000_000)
        result = orch.run("Find opportunities in Indian IT stocks")
    """

    UNIVERSE = {
        "indian_it"     : ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS"],
        "indian_banking": ["HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS", "SBIN.NS"],
        "indian_large"  : ["RELIANCE.NS", "TATAMOTORS.NS", "MARUTI.NS", "BAJFINANCE.NS"],
        "us_tech"       : ["AAPL", "MSFT", "NVDA", "GOOGL", "META"],
        "us_finance"    : ["JPM", "GS", "MS", "BAC"],
        "mixed_best"    : ["TCS.NS", "INFY.NS", "RELIANCE.NS", "AAPL", "NVDA"],
    }

    def __init__(self, capital: float = 1_000_000, min_confidence: float = 0.30):
        self.capital        = capital
        self.min_confidence = min_confidence

        # ── Configure Gemini ──
        # Paste your key from aistudio.google.com here

        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        # ── Spin up the 4 specialist agents ──
        self.data_agent   = DataIngestionAgent()
        self.signal_agent = SignalResearchAgent()
        self.risk_agent   = RiskManagerAgent(capital=capital, min_confidence=min_confidence)
        self.exec_agent   = ExecutionAgent(slippage_pct=0.001)

        print(f"[Orchestrator] Initialized  capital={capital:,.0f}  model=gemini-2.5-flash")
        print(f"[Orchestrator] Agents loaded: Data · Signal · Risk · Execution")


    def run(self, goal: str) -> dict:
        """
        Main entry point. Give it a goal in plain English.
        Returns full result dict with decisions + LLM memo.
        """
        print(f"\n{'='*60}")
        print(f"[Orchestrator] Goal: {goal}")
        print(f"{'='*60}")

        # Step 1: Gemini picks which tickers to scan
        tickers = self._llm_select_tickers(goal)
        print(f"[Orchestrator] Tickers selected: {tickers}")

        # Step 2: Run the 4-agent pipeline
        snapshots = self._run_data_agent(tickers)
        reports   = self._run_signal_agent(snapshots)
        decisions = self._run_risk_agent(reports)
        orders    = self._run_execution_agent(decisions)
        ex_report = self.exec_agent.get_report()

        # Step 3: Gemini reasons about results → investment memo
        memo = self._llm_generate_memo(goal, reports, decisions, ex_report)

        # Step 4: Print everything
        self._print_memo(memo, ex_report)

        return {
            "goal"     : goal,
            "tickers"  : tickers,
            "snapshots": snapshots,
            "reports"  : reports,
            "decisions": decisions,
            "orders"   : orders,
            "memo"     : memo,
        }


    # ─────────────────────────────────────────
    #  Gemini LLM Calls
    # ─────────────────────────────────────────

    def _llm_select_tickers(self, goal: str) -> list[str]:
        """Ask Gemini which tickers to scan for this goal."""
        print(f"[Orchestrator] Asking Gemini to select tickers...")

        prompt = f"""You are the orchestrator of a quantitative trading system.

The user's research goal is: "{goal}"

Available ticker universe:
{json.dumps(self.UNIVERSE, indent=2)}

Select 3-6 tickers most relevant to this goal.
Respond with ONLY a raw JSON array. No explanation, no markdown, no code fences.
Example: ["TCS.NS", "INFY.NS", "WIPRO.NS"]"""

        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        raw = response.text.strip()

        # Clean up any markdown fences Gemini might add
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        tickers = json.loads(raw)
        return tickers


    def _llm_generate_memo(
        self,
        goal     : str,
        reports  : dict[str, SignalReport],
        decisions: dict[str, TradeDecision],
        ex_report: ExecutionReport,
    ) -> str:
        """Ask Gemini to reason about results and write an investment memo."""
        print(f"\n[Orchestrator] Asking Gemini to generate investment memo...")

        signal_summary = []
        for ticker, r in reports.items():
            signal_summary.append({
                "ticker"    : ticker,
                "direction" : r.final_direction,
                "confidence": f"{r.confidence:.0%}",
                "signals"   : [
                    {
                        "name"     : s.name,
                        "direction": s.direction,
                        "strength" : s.strength,
                        "reason"   : s.reason,
                    }
                    for s in r.signals
                ],
            })

        decision_summary = []
        for ticker, d in decisions.items():
            decision_summary.append({
                "ticker"          : ticker,
                "approved"        : d.approved,
                "action"          : d.action,
                "entry_price"     : d.entry_price,
                "stop_loss"       : d.stop_loss,
                "take_profit"     : d.take_profit,
                "position_size"   : d.position_size,
                "capital_at_risk" : d.capital_to_risk,
                "kelly_fraction"  : d.kelly_fraction,
                "rr_ratio"        : d.risk_reward_ratio,
                "rejection_reason": d.rejection_reason,
            })

        prompt = f"""You are the head of quantitative research at a systematic trading fund.

Research goal: "{goal}"

Signal analysis results:
{json.dumps(signal_summary, indent=2)}

Risk manager decisions:
{json.dumps(decision_summary, indent=2)}

Write a concise investment memo (under 300 words) with these sections:

1. MARKET CONTEXT: What do the signals tell us about current market regime?
2. TRADE RATIONALE: For approved trades, why does each make sense?
3. REJECTIONS: Were the rejected trades the right call? Why?
4. RISK ASSESSMENT: Key risks in the current portfolio.
5. NEXT STEPS: What should the portfolio manager watch?

Be precise and data-driven. Reference actual numbers from the data.
Write like a quant analyst — no fluff."""

        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text.strip()


    # ─────────────────────────────────────────
    #  Pipeline Steps
    # ─────────────────────────────────────────

    def _run_data_agent(self, tickers: list[str]) -> dict[str, MarketSnapshot]:
        print(f"\n[Orchestrator] → Calling Data Agent  ({len(tickers)} tickers)")
        return self.data_agent.get_multiple(tickers, period="6mo")

    def _run_signal_agent(self, snapshots: dict) -> dict[str, SignalReport]:
        print(f"\n[Orchestrator] → Calling Signal Agent")
        return self.signal_agent.analyze_multiple(snapshots)

    def _run_risk_agent(self, reports: dict) -> dict[str, TradeDecision]:
        print(f"\n[Orchestrator] → Calling Risk Agent")
        return self.risk_agent.evaluate_multiple(reports)

    def _run_execution_agent(self, decisions: dict) -> list:
        print(f"\n[Orchestrator] → Calling Execution Agent")
        return self.exec_agent.execute_multiple(decisions)


    # ─────────────────────────────────────────
    #  Output
    # ─────────────────────────────────────────

    def _print_memo(self, memo: str, ex_report: ExecutionReport) -> None:
        print(f"""
{'='*60}
  AQRA INVESTMENT MEMO
  Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
{'='*60}
{memo}
{'='*60}""")
        ex_report.summary()


# ─────────────────────────────────────────────
#  Run directly to test
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("\n" + "="*60)
    print("TEST 1: Indian IT stocks")
    print("="*60)
    orch = OrchestratorAgent(capital=1_000_000, min_confidence=0.30)
    orch.run("Find the best trading opportunities in Indian IT stocks today")

    print("\n" + "="*60)
    print("TEST 2: Broad market scan")
    print("="*60)
    orch2 = OrchestratorAgent(capital=1_000_000, min_confidence=0.30)
    orch2.run("Scan US tech and Indian large cap, find any strong directional signals")
