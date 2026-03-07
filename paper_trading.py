"""
AQRA — Paper Trading Engine
============================
Run every evening after 4:00 PM IST (NSE close).

    python paper_trading.py

Outputs:
  1. Terminal printout  — positions, signals, P&L
  2. HTML report        — saved to reports/YYYY-MM-DD.html
  3. Flask dashboard    — http://localhost:5001  (auto-refresh 30s)
  4. Telegram Bot       — instant message on phone + desktop (configure .env)

Capital  : ₹50,000
Universe : TCS.NS, INFY.NS, RELIANCE.NS, HDFCBANK.NS, BAJFINANCE.NS, HCLTECH.NS
Logic    : Full v9 signal stack (regime + macro sentinel + trailing ATR + per-ticker override)

Setup (one-time):
  pip install yfinance pandas numpy flask python-dotenv requests --break-system-packages
  1. Open Telegram → search @BotFather → /newbot → copy Bot Token
  2. Message your bot once, then open:
     https://api.telegram.org/botYOUR_TOKEN/getUpdates
     Copy the "id" value inside "chat" — that is your Chat ID
  3. Copy .env.example to .env and fill TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID
  Run: python paper_trading.py
"""

import os, json, time, threading
from datetime import datetime, date
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import deque

import pandas as pd
import numpy as np
import yfinance as yf
from flask import Flask, render_template_string, jsonify

# optional — graceful if missing
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import requests   # for Telegram — stdlib-adjacent, always available

# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────

CAPITAL_INR     = 50_000
RISK_PCT        = 0.02          # 2% risk per trade = ₹1,000
SLIPPAGE_PCT    = 0.001
DATA_DIR        = Path("paper_data")
REPORTS_DIR     = Path("reports")
STATE_FILE      = DATA_DIR / "state.json"
DATA_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

NSE_TICKERS = ["TCS.NS","INFY.NS","RELIANCE.NS","HDFCBANK.NS","BAJFINANCE.NS","HCLTECH.NS"]
US_TICKERS  = ["AAPL","MSFT","NVDA","GOOGL","JPM"]
ALL_TICKERS = NSE_TICKERS + US_TICKERS
MOMENTUM_T  = {"TCS.NS","HCLTECH.NS","BAJFINANCE.NS","NVDA","MSFT","JPM"}
MEAN_REV_T  = {"INFY.NS","HDFCBANK.NS","RELIANCE.NS","AAPL","GOOGL"}

# USD → INR for unified P&L display (fetched live, fallback hardcoded)
USD_INR_FALLBACK = 84.0

# v9 params (identical to backtest)
ATR_RATIO_WARN  = 1.2
ATR_RATIO_HIGH  = 1.5
ATR_MULT_NORMAL = 3.0
ATR_MULT_WARN   = 2.0
ATR_MULT_HIGH   = 1.5
ATR_LOOKBACK    = 60
CORR_LIMIT      = 0.70
CORR_LOOKBACK   = 60
CS_MOM_LOOKBACK = 63
CS_FILTER_PCT   = 0.50
VIX_RULES       = [(15,1.2),(20,1.0),(28,0.6),(35,0.3),(999,0.0)]

FLIP_MODERATE   = 0.50
FLIP_HIGH       = 0.70
FLIP_BLOCK      = 0.85

MACRO_TICKERS = {
    "^VIX9D":"vix9d","^VIX3M":"vix3m",
    "HYG":"hyg","LQD":"lqd",
    "^TNX":"tnx","^IRX":"irx","^INDIAVIX":"indiavix",
}

# Telegram Bot
TG_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ─────────────────────────────────────────────
#  State Management
# ─────────────────────────────────────────────

def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "capital"       : CAPITAL_INR,
        "open_positions": {},   # ticker → {direction, entry_date, entry_price, units, stop, target, regime, flip_prob}
        "closed_trades" : [],   # list of completed trade dicts
        "daily_logs"    : [],   # list of daily run summaries
        "start_date"    : str(date.today()),
        "version"       : "v9",
    }

def save_state(state: dict):
    with open(STATE_FILE,"w") as f:
        json.dump(state, f, indent=2, default=str)


# ─────────────────────────────────────────────
#  Data Fetching
# ─────────────────────────────────────────────

def fetch_usd_inr() -> float:
    """Fetch live USD/INR rate via yfinance. Fallback to hardcoded."""
    try:
        d = yf.download("USDINR=X", period="5d", auto_adjust=True, progress=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        if len(d) > 0:
            rate = float(d["Close"].iloc[-1])
            print(f"  ✓ USD/INR = {rate:.2f}")
            return rate
    except:
        pass
    print(f"  ✗ USD/INR fetch failed — using fallback {USD_INR_FALLBACK}")
    return USD_INR_FALLBACK


def fetch_data(lookback_days=400) -> dict:
    """
    Fetch EOD data for all 11 tickers + macro.

    Timezone note:
      NSE closes 3:30 PM IST — data available by 4:00 PM IST (today)
      NYSE closes 4:00 PM EST = 1:30 AM IST — data available next morning IST
      → When you run at 4:00 PM IST, US data = yesterday's close (clearly labelled)
      → This is fine: our signals are daily-bar, not intraday
    """
    print("\n[Data] Fetching EOD data...")
    # UTC-normalized date strings — avoids DST nonexistent-time errors on US changeover days
    end_str   = pd.Timestamp.now('UTC').strftime("%Y-%m-%d")
    start_str = (pd.Timestamp.now('UTC') - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    data  = {}

    print("  [NSE] Today's close (3:30 PM IST):")
    for ticker in NSE_TICKERS:
        try:
            df = yf.download(ticker, start=start_str, end=end_str,
                             auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > 50:
                data[ticker] = df
                last_date = str(df.index[-1].date())
                print(f"    ✓ {ticker:<15} {len(df)} days  "
                      f"close=₹{df['Close'].iloc[-1]:.2f}  ({last_date})")
        except Exception as e:
            print(f"    ✗ {ticker}: {e}")

    print("  [NYSE] Previous close (data lags ~12h behind IST):")
    for ticker in US_TICKERS:
        try:
            df = yf.download(ticker, start=start_str, end=end_str,
                             auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > 50:
                data[ticker] = df
                last_date = str(df.index[-1].date())
                print(f"    ✓ {ticker:<15} {len(df)} days  "
                      f"close=${df['Close'].iloc[-1]:.2f}  ({last_date})")
        except Exception as e:
            print(f"    ✗ {ticker}: {e}")

    macro = {}
    print("  [Macro]")
    for sym, key in MACRO_TICKERS.items():
        try:
            d = yf.download(sym, start=start_str, end=end_str,
                            auto_adjust=True, progress=False)
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            if len(d) > 0:
                macro[key] = d["Close"]
        except:
            macro[key] = None

    usd_inr = fetch_usd_inr()
    return data, macro, usd_inr


# ─────────────────────────────────────────────
#  Macro Sentinel (same logic as v9)
# ─────────────────────────────────────────────

def compute_flip_prob(macro: dict, today: pd.Timestamp) -> float:
    scores = []
    weights = []

    # VIX term structure
    v9d = macro.get("vix9d"); v3m = macro.get("vix3m")
    if v9d is not None and v3m is not None:
        try:
            al = pd.concat([v9d,v3m],axis=1).ffill().dropna()
            al.columns=["v9d","v3m"]
            ratio = (al["v9d"]/al["v3m"])
            r = float(ratio.iloc[-1]) if len(ratio)>0 else 1.0
            score = np.clip((r-0.80)/(1.20-0.80),0,1)
            scores.append(score*0.30); weights.append(0.30)
        except: pass

    # Credit spread
    hyg = macro.get("hyg"); lqd = macro.get("lqd")
    if hyg is not None and lqd is not None:
        try:
            al = pd.concat([hyg,lqd],axis=1).ffill().dropna()
            al.columns=["hyg","lqd"]
            ratio = al["hyg"]/al["lqd"]
            ret20 = float(ratio.pct_change(20).iloc[-1]) if len(ratio)>20 else 0.0
            score = np.clip((-ret20+0.05)/0.10,0,1)
            scores.append(score*0.40); weights.append(0.40)
        except: pass

    # Yield curve
    tnx = macro.get("tnx"); irx = macro.get("irx")
    if tnx is not None and irx is not None:
        try:
            al = pd.concat([tnx,irx],axis=1).ffill().dropna()
            al.columns=["t10","t2"]
            spread = al["t10"]/100 - al["t2"]/100
            spread_chg = float(spread.diff(20).iloc[-1]) if len(spread)>20 else 0.0
            spread_val = float(spread.iloc[-1])
            inv_score  = np.clip((-spread_val+0.02)/0.05,0,1)
            flat_score = np.clip((-spread_chg+0.005)/0.015,0,1)
            score = 0.5*inv_score + 0.5*flat_score
            scores.append(score*0.30); weights.append(0.30)
        except: pass

    global_prob = sum(scores)/sum(weights) if weights else 0.0

    # India VIX blend for NSE
    indiavix = macro.get("indiavix")
    if indiavix is not None and len(indiavix)>0:
        try:
            iv = float(indiavix.iloc[-1])
            india_stress = np.clip((iv-15)/(35-15),0,1)
            return float(global_prob*0.50 + india_stress*0.50)
        except: pass

    return float(global_prob)


def get_size_modifier(flip_prob: float):
    if flip_prob >= FLIP_BLOCK:    return 0.0, ATR_MULT_HIGH, True
    elif flip_prob >= FLIP_HIGH:   return 0.40, ATR_MULT_WARN, False
    elif flip_prob >= FLIP_MODERATE: return 0.65, ATR_MULT_NORMAL*0.85, False
    else:                           return 1.0, ATR_MULT_NORMAL, False


# ─────────────────────────────────────────────
#  Indicators (identical to v9)
# ─────────────────────────────────────────────

def compute_atr(df, period=14):
    h=df["High"].values; l=df["Low"].values; c=df["Close"].values
    tr=np.maximum(h[1:]-l[1:],np.maximum(abs(h[1:]-c[:-1]),abs(l[1:]-c[:-1])))
    return float(tr[-period:].mean()) if len(tr)>=period else float(c[-1])*0.03

def compute_atr_ratio(df):
    h=df["High"].values; l=df["Low"].values; c=df["Close"].values
    tr=np.maximum(h[1:]-l[1:],np.maximum(abs(h[1:]-c[:-1]),abs(l[1:]-c[:-1])))
    if len(tr)<ATR_LOOKBACK+14: return 1.0
    curr=tr[-14:].mean(); avg=tr[-ATR_LOOKBACK-14:-14].mean()
    return float(curr/avg) if avg>0 else 1.0

def compute_adx(df, period=14):
    h=df["High"].values; l=df["Low"].values; c=df["Close"].values
    if len(h)<period*2+1: return 0.0
    dmp=np.where((h[1:]-h[:-1])>(l[:-1]-l[1:]),np.maximum(h[1:]-h[:-1],0),0)
    dmm=np.where((l[:-1]-l[1:])>(h[1:]-h[:-1]),np.maximum(l[:-1]-l[1:],0),0)
    tr =np.maximum(h[1:]-l[1:],np.maximum(abs(h[1:]-c[:-1]),abs(l[1:]-c[:-1])))
    def sm(a,n):
        s=np.zeros(len(a))
        if len(a)>=n:
            s[n-1]=a[:n].sum()
            for i in range(n,len(a)): s[i]=s[i-1]-s[i-1]/n+a[i]
        return s
    at=sm(tr,period); dp=sm(dmp,period); dm=sm(dmm,period)
    with np.errstate(divide='ignore',invalid='ignore'):
        dip=np.where(at>0,100*dp/at,0); dim=np.where(at>0,100*dm/at,0)
        dx=np.where((dip+dim)>0,100*abs(dip-dim)/(dip+dim),0)
    v=sm(dx,period); return float(v[-1]) if len(v)>0 else 0.0

def get_regime_params(regime_label, ticker):
    is_mom = ticker in MOMENTUM_T
    if "HIGH_VOL" in regime_label: return {"rsi":25,"adx":22,"atr":2.0}
    if "BEAR" in regime_label:     return {"rsi":28,"adx":20,"atr":2.0}
    if regime_label in ("OPTIMAL_BULL","BULL_RISK_ON"):
        return {"rsi":35 if is_mom else 30,"adx":15,"atr":ATR_MULT_NORMAL}
    if "BULL" in regime_label:
        return {"rsi":32 if is_mom else 30,"adx":15,"atr":2.5}
    return {"rsi":30,"adx":18,"atr":2.5}

def detect_regime(df, all_dfs):
    close = df["Close"]
    if len(close)<200: return 0.5,True,True,"WARMING_UP"
    price=float(close.iloc[-1]); sma200=float(close.iloc[-200:].mean())
    trend="BULL" if price>sma200*1.02 else "BEAR" if price<sma200*0.98 else "NEUTRAL"
    ac=close.diff().abs().iloc[-14:].mean()
    aa=close.diff().abs().iloc[-74:-14].mean()
    ar=float(ac/aa) if aa>0 else 1.0
    vol="HIGH" if ar>1.5 else "LOW" if ar<0.7 else "NORMAL"
    above=0
    for t,d in all_dfs.items():
        c2=d["Close"]
        if len(c2)>=50:
            above += 1 if float(c2.iloc[-1])>float(c2.iloc[-50:].mean()) else 0
    b=above/len(all_dfs) if all_dfs else 0.5
    br="RISK_ON" if b>0.6 else "RISK_OFF" if b<0.4 else "NEUTRAL"
    if trend=="BEAR" and vol=="HIGH" and br=="RISK_OFF": return 0.0,False,False,"BLACKOUT"
    if vol=="HIGH": return 0.35,trend!="BEAR",True,f"HIGH_VOL_{trend}"
    if trend=="BEAR":
        s=0.5 if br=="RISK_OFF" else 0.65; return s,False,True,f"BEAR_{br}"
    if trend=="BULL":
        if vol=="LOW" and br=="RISK_ON": return 1.0,True,True,"OPTIMAL_BULL"
        if br=="RISK_ON": return 0.85,True,True,"BULL_RISK_ON"
        return 0.70,True,True,"BULL_NEUTRAL"
    return (0.50,True,True,"NEUTRAL_CAUTIOUS") if br=="RISK_OFF" \
        else (0.75,True,True,"NEUTRAL_NORMAL")

def compute_signals(df, regime_label, ticker):
    rp=get_regime_params(regime_label,ticker)
    rsi_t=rp["rsi"]; adx_t=rp["adx"]; atr_m=rp["atr"]
    close=df["Close"]; high=df["High"]; low=df["Low"]
    if len(close)<60: return "NEUTRAL",0.0,atr_m,{}

    adx=compute_adx(df)
    if adx<adx_t: return "NEUTRAL",0.0,atr_m,{"adx":round(adx,1),"blocked":"ADX too low"}

    price=float(close.iloc[-1])
    sma20=float(close.iloc[-20:].mean()); sma50=float(close.iloc[-50:].mean())
    gap=(sma20-sma50)/sma50
    sma_s="BUY" if gap>0.005 else "SELL" if gap<-0.005 else "NEUTRAL"

    w=close.diff().iloc[-15:]; g=w.clip(lower=0).mean(); lv=(-w.clip(upper=0)).mean()
    rsi=100-(100/(1+g/lv)) if lv!=0 else 100.0
    rsi_s="BUY" if rsi<rsi_t else "SELL" if rsi>(100-rsi_t) else "NEUTRAL"

    sb=float(close.iloc[-20:].mean()); stdb=float(close.iloc[-20:].std())
    bb_s="BUY" if price<sb-2*stdb else "SELL" if price>sb+2*stdb else "NEUTRAL"

    w2=close.iloc[-51:]
    ef=w2.ewm(span=12,adjust=False).mean(); es=w2.ewm(span=26,adjust=False).mean()
    ml=ef-es; sl2=ml.ewm(span=9,adjust=False).mean()
    macd_h=float(ml.iloc[-1]-sl2.iloc[-1])
    macd_s="BUY" if macd_h>0 else "SELL" if macd_h<0 else "NEUTRAL"

    vol=df.get("Volume")
    if vol is not None and len(vol)>20:
        obv=(np.sign(close.diff().fillna(0))*vol).cumsum()
        osma=obv.rolling(20).mean()
        obv_s="BUY" if float(obv.iloc[-1])>float(osma.iloc[-1])*1.02 else \
              "SELL" if float(obv.iloc[-1])<float(osma.iloc[-1])*0.98 else "NEUTRAL"
    else: obv_s="NEUTRAL"

    signals={"sma":sma_s,"rsi":rsi_s,"bb":bb_s,"macd":macd_s,"obv":obv_s}

    if "BULL" in regime_label and "HIGH_VOL" not in regime_label:
        weights={"sma":2.0,"macd":2.5,"obv":1.8,"rsi":0.8,"bb":0.8}
    elif "BEAR" in regime_label:
        weights={"sma":0.6,"macd":0.7,"obv":1.5,"rsi":2.2,"bb":2.2}
    elif "HIGH_VOL" in regime_label:
        weights={"sma":0.5,"macd":1.2,"obv":2.5,"rsi":0.9,"bb":0.9}
    else:
        weights={"sma":1.0,"macd":2.0,"obv":2.0,"rsi":1.5,"bb":1.0}

    tw=sum(weights.values())
    bw=sum(weights[k] for k,v in signals.items() if v=="BUY")
    sw=sum(weights[k] for k,v in signals.items() if v=="SELL")

    detail={"adx":round(adx,1),"rsi":round(rsi,1),"macd":round(macd_h,4),
            "signals":signals,"weights":{k:round(v,1) for k,v in weights.items()},
            "buy_weight":round(bw/tw,3),"sell_weight":round(sw/tw,3)}

    if bw>sw and bw/tw>=0.45: return "BUY",round(bw/tw,3),atr_m,detail
    elif sw>bw and sw/tw>=0.45: return "SELL",round(sw/tw,3),atr_m,detail
    return "NEUTRAL",0.0,atr_m,detail

def get_vix_mult(macro, ticker=""):
    """Use India VIX for NSE tickers, US VIX for NYSE tickers."""
    is_nse = ticker in NSE_TICKERS or ticker == ""
    if is_nse:
        iv = macro.get("indiavix")
    else:
        # US VIX: prefer vix9d, fall back to vix3m, then None
        iv = macro.get("vix9d")
        if iv is None or (hasattr(iv, "__len__") and len(iv) == 0):
            iv = macro.get("vix3m")
    try:
        v = float(iv.iloc[-1]) if iv is not None and len(iv) > 0 else 20.0
    except Exception:
        v = 20.0
    if np.isnan(v): v = 20.0
    for t, m in VIX_RULES:
        if v < t: return m, round(v, 1)
    return 0.0, round(v, 1)

def get_cs_rank(ticker, all_dfs):
    rets={}
    for t,d in all_dfs.items():
        c=d["Close"]
        if len(c)>=CS_MOM_LOOKBACK:
            p=float(c.iloc[-CS_MOM_LOOKBACK]); curr=float(c.iloc[-1])
            if p>0: rets[t]=(curr-p)/p
    if not rets or ticker not in rets: return 0.5
    srt=sorted(rets.keys(),key=lambda x:rets[x])
    return srt.index(ticker)/max(len(srt)-1,1)


# ─────────────────────────────────────────────
#  Core Daily Run
# ─────────────────────────────────────────────

def run_daily(state: dict, data: dict, macro: dict, usd_inr: float) -> dict:
    today     = str(date.today())
    today_ts  = pd.Timestamp.today()
    signals_log = []
    actions     = []

    flip_prob = compute_flip_prob(macro, today_ts)
    size_mod, atr_override, macro_blocked = get_size_modifier(flip_prob)

    # Global VIX values for header display
    india_vix_mult, india_vix_val = get_vix_mult(macro, "TCS.NS")
    us_vix_mult,    us_vix_val    = get_vix_mult(macro, "AAPL")

    print(f"\n{'═'*60}")
    print(f"  AQRA Paper Trading  —  {today}")
    print(f"  Capital: ₹{state['capital']:,.0f}  |  Flip Prob: {flip_prob:.3f}")
    print(f"  India VIX: {india_vix_val}  |  US VIX: {us_vix_val}  |  USD/INR: {usd_inr:.2f}")
    print(f"  Macro: {'⛔ BLOCKED' if macro_blocked else '🟡 CAUTIOUS' if flip_prob>FLIP_MODERATE else '🟢 CALM'}")
    print(f"  Universe: {len([t for t in ALL_TICKERS if t in data])}/11 tickers loaded")
    print(f"{'═'*60}")

    # ── 1. Check open positions for stop/target hits ──────────────
    print(f"\n[Positions] Checking {len(state['open_positions'])} open positions...")
    to_close = []
    for ticker, pos in state["open_positions"].items():
        if ticker not in data: continue
        df    = data[ticker]
        curr  = float(df["Close"].iloc[-1])
        high  = float(df["High"].iloc[-1])
        low   = float(df["Low"].iloc[-1])
        direc = pos["direction"]

        # Trailing stop update
        atr_now   = compute_atr(df)
        atr_ratio = compute_atr_ratio(df)
        if   atr_ratio >= ATR_RATIO_HIGH: tm = ATR_MULT_HIGH
        elif atr_ratio >= ATR_RATIO_WARN: tm = ATR_MULT_WARN
        else:                              tm = ATR_MULT_NORMAL

        if direc == "BUY":
            new_stop = max(pos["stop"], curr - atr_now*tm)
        else:
            new_stop = min(pos["stop"], curr + atr_now*tm)

        if new_stop != pos["stop"]:
            pos["stop"] = round(new_stop, 2)
            print(f"  ↑ {ticker} stop tightened → ₹{new_stop:.2f}  (ATR ratio={atr_ratio:.2f})")

        # Check hits (use high/low for realism)
        hit_stop   = (low  <= pos["stop"]   if direc=="BUY" else high >= pos["stop"])
        hit_target = (high >= pos["target"] if direc=="BUY" else low  <= pos["target"])

        exit_price = exit_reason = None
        if hit_target:
            exit_price  = pos["target"]; exit_reason = "TAKE_PROFIT"
        elif hit_stop:
            exit_price  = pos["stop"];   exit_reason = "STOP_LOSS"

        if exit_reason:
            pnl = (exit_price - pos["entry_price"]) * pos["units"] \
                  if direc=="BUY" else \
                  (pos["entry_price"] - exit_price) * pos["units"]
            pnl_pct = pnl / (pos["entry_price"] * pos["units"]) * 100
            trade = {
                "ticker"      : ticker,
                "direction"   : direc,
                "entry_date"  : pos["entry_date"],
                "exit_date"   : today,
                "entry_price" : pos["entry_price"],
                "exit_price"  : round(exit_price, 2),
                "units"       : pos["units"],
                "pnl"         : round(pnl, 2),
                "pnl_pct"     : round(pnl_pct, 2),
                "exit_reason" : exit_reason,
                "regime"      : pos.get("regime","—"),
            }
            state["closed_trades"].append(trade)
            state["capital"] = round(state["capital"] + pnl, 2)
            to_close.append(ticker)
            icon = "✅" if pnl >= 0 else "❌"
            print(f"  {icon} CLOSED {ticker}  {exit_reason}  "
                  f"P&L=₹{pnl:+.0f} ({pnl_pct:+.1f}%)")
            actions.append(f"CLOSED {ticker} {exit_reason} P&L=₹{pnl:+.0f}")

    for t in to_close:
        del state["open_positions"][t]

    # ── 2. Scan for new signals ───────────────────────────────────
    print(f"\n[Signals] Scanning {len(data)} tickers...")
    print(f"  {'Ticker':<14} {'Exchange':<6} {'Price':>10} {'Regime':<22} {'Signal':<8} {'Conf':>6} {'Action'}")
    print(f"  {'─'*85}")
    open_tickers = set(state["open_positions"].keys())

    for ticker in ALL_TICKERS:
        if ticker not in data:
            continue
        df         = data[ticker]
        is_nse     = ticker in NSE_TICKERS
        exchange   = "NSE" if is_nse else "NYSE"
        curr_price = float(df["Close"].iloc[-1])
        price_str  = f"₹{curr_price:.2f}" if is_nse else f"${curr_price:.2f}"

        r_mult, allow_buy, allow_sell, r_lbl = detect_regime(df, data)
        sig, conf, atr_m, detail = compute_signals(df, r_lbl, ticker)

        # Per-ticker VIX
        v_mult, vix_val = get_vix_mult(macro, ticker)

        log_entry = {
            "ticker"   : ticker,
            "exchange" : exchange,
            "price"    : round(curr_price, 2),
            "price_str": price_str,
            "regime"   : r_lbl,
            "signal"   : sig,
            "conf"     : conf,
            "flip_prob": round(flip_prob, 3),
            "vix"      : vix_val,
            "detail"   : detail,
            "action"   : "NONE",
            "reason"   : "",
        }

        # Already in position
        if ticker in open_tickers:
            pos = state["open_positions"][ticker]
            log_entry["action"] = f"HOLDING {pos['direction']}"
            log_entry["reason"] = f"stop={price_str[0]}{pos['stop']}  target={price_str[0]}{pos['target']}"
            print(f"  {ticker.replace('.NS',''):<14} {exchange:<6} {price_str:>10} "
                  f"{r_lbl:<22} {'HOLD':<8} {'—':>6}  holding {pos['direction']}")
            signals_log.append(log_entry)
            continue

        # Gate checks
        if sig == "NEUTRAL":
            log_entry["reason"] = "No signal"
        elif r_mult == 0.0:
            log_entry["reason"] = "BLACKOUT regime"
        elif sig=="BUY" and not allow_buy:
            log_entry["reason"] = "Buy blocked in BEAR regime"
        elif sig=="SELL" and not allow_sell:
            log_entry["reason"] = "Sell blocked in BULL regime"
        elif v_mult == 0.0:
            log_entry["reason"] = f"VIX gate blocked (VIX={vix_val})"
        elif macro_blocked:
            log_entry["reason"] = f"Macro Sentinel blocked (flip_prob={flip_prob:.3f})"
        else:
            cs_rank = get_cs_rank(ticker, data)
            if sig=="BUY" and cs_rank < CS_FILTER_PCT:
                log_entry["reason"] = f"CS momentum filter (rank={cs_rank:.2f})"
            elif sig=="SELL" and cs_rank > (1-CS_FILTER_PCT):
                log_entry["reason"] = f"CS momentum filter (rank={cs_rank:.2f})"
            else:
                # Position sizing — risk in native currency, display in INR
                atr       = compute_atr(df)
                stop_dist = atr * min(atr_m, atr_override)
                tgt_dist  = stop_dist * 2.0
                combined  = r_mult * v_mult * size_mod

                # Risk amount always in INR
                risk_inr  = state["capital"] * RISK_PCT * combined
                # For US stocks: convert stop_dist to INR for sizing
                stop_dist_inr = stop_dist * usd_inr if not is_nse else stop_dist
                units = risk_inr / stop_dist_inr if stop_dist_inr > 0 else 0

                if units < 1:
                    log_entry["reason"] = f"Position too small (units={units:.2f})"
                else:
                    units = int(units)
                    ep    = curr_price * (1+SLIPPAGE_PCT) if sig=="BUY" \
                            else curr_price * (1-SLIPPAGE_PCT)
                    stop  = ep - stop_dist if sig=="BUY" else ep + stop_dist
                    tgt   = ep + tgt_dist  if sig=="BUY" else ep - tgt_dist

                    # Cost in INR
                    cost_inr = ep * units * (usd_inr if not is_nse else 1.0)
                    if cost_inr > state["capital"] * 0.40:
                        units = max(1, int(state["capital"] * 0.40 /
                                          (ep * (usd_inr if not is_nse else 1.0))))

                    state["open_positions"][ticker] = {
                        "direction"   : sig,
                        "entry_date"  : today,
                        "entry_price" : round(ep, 2),
                        "units"       : units,
                        "stop"        : round(stop, 2),
                        "target"      : round(tgt, 2),
                        "regime"      : r_lbl,
                        "flip_prob"   : round(flip_prob, 3),
                        "r_mult"      : round(r_mult, 3),
                        "conf"        : round(conf, 3),
                        "atr"         : round(atr, 2),
                        "is_nse"      : is_nse,
                        "currency"    : "INR" if is_nse else "USD",
                        "exchange"    : exchange,
                    }
                    cur_sym = "₹" if is_nse else "$"
                    log_entry["action"] = f"🚀 OPENED {sig}"
                    log_entry["reason"] = (f"units={units}  entry={cur_sym}{ep:.2f}  "
                                           f"stop={cur_sym}{stop:.2f}  target={cur_sym}{tgt:.2f}")
                    print(f"  {ticker.replace('.NS',''):<14} {exchange:<6} {price_str:>10} "
                          f"{r_lbl:<22} {sig:<8} {conf:>6.3f}  "
                          f"🚀 ENTRY {cur_sym}{ep:.2f}  stop={cur_sym}{stop:.2f}  "
                          f"target={cur_sym}{tgt:.2f}  units={units}")
                    actions.append(f"OPENED {ticker.replace('.NS','')} "
                                   f"[{exchange}] {sig} @ {cur_sym}{ep:.2f}")
                    signals_log.append(log_entry)
                    continue

        print(f"  {ticker.replace('.NS',''):<14} {exchange:<6} {price_str:>10} "
              f"{r_lbl:<22} {sig:<8} {conf:>6.3f}  {log_entry['reason'][:35]}")

        signals_log.append(log_entry)

    # ── 3. Portfolio summary ──────────────────────────────────────
    total_pnl = sum(t["pnl"] for t in state["closed_trades"])
    open_pnl  = 0.0
    for ticker, pos in state["open_positions"].items():
        if ticker not in data: continue
        curr = float(data[ticker]["Close"].iloc[-1])
        is_nse = pos.get("is_nse", ticker in NSE_TICKERS)
        raw_pnl = ((curr - pos["entry_price"]) * pos["units"]
                   if pos["direction"]=="BUY"
                   else (pos["entry_price"] - curr) * pos["units"])
        open_pnl += raw_pnl * (usd_inr if not is_nse else 1.0)

    total_return_pct = (state["capital"] + open_pnl - CAPITAL_INR) / CAPITAL_INR * 100

    print(f"\n{'─'*60}")
    print(f"  Portfolio Summary  (all values in ₹ INR)")
    print(f"  Cash          : ₹{state['capital']:>10,.0f}")
    print(f"  Open P&L      : ₹{open_pnl:>+10,.0f}  (USD converted @ {usd_inr:.2f})")
    print(f"  Closed P&L    : ₹{total_pnl:>+10,.0f}")
    print(f"  Total Return  : {total_return_pct:>+.2f}%")
    print(f"  Open Trades   : {len(state['open_positions'])}  "
          f"(NSE={sum(1 for t in state['open_positions'] if t in NSE_TICKERS)}  "
          f"US={sum(1 for t in state['open_positions'] if t not in NSE_TICKERS)})")
    print(f"  Closed Trades : {len(state['closed_trades'])}")
    print(f"{'─'*60}")

    # ── 4. Log daily run ─────────────────────────────────────────
    daily_log = {
        "date"        : today,
        "capital"     : state["capital"],
        "open_pnl"    : round(open_pnl, 2),
        "total_pnl"   : round(total_pnl, 2),
        "total_return": round(total_return_pct, 2),
        "open_trades" : len(state["open_positions"]),
        "flip_prob"   : round(flip_prob, 3),
        "india_vix"   : india_vix_val,
        "us_vix"      : us_vix_val,
        "usd_inr"     : round(usd_inr, 2),
        "actions"     : actions,
        "signals"     : signals_log,
    }
    state["daily_logs"].append(daily_log)
    save_state(state)

    return daily_log


# ─────────────────────────────────────────────
#  HTML Report Generator
# ─────────────────────────────────────────────

def generate_html_report(state: dict, daily_log: dict) -> str:
    today      = daily_log["date"]
    total_pnl  = daily_log["total_pnl"]
    open_pnl   = daily_log["open_pnl"]
    ret_pct    = daily_log["total_return"]
    flip_prob  = daily_log["flip_prob"]
    india_vix  = daily_log.get("india_vix", "—")
    us_vix     = daily_log.get("us_vix", "—")
    usd_inr    = daily_log.get("usd_inr", USD_INR_FALLBACK)

    dates_eq = [d["date"] for d in state["daily_logs"]]
    equity_eq= [CAPITAL_INR + d["total_pnl"] + d["open_pnl"] for d in state["daily_logs"]]
    equity_js = ",".join([f'{{x:"{d}",y:{e:.0f}}}' for d,e in zip(dates_eq,equity_eq)])

    pos_rows = ""
    for ticker, pos in state["open_positions"].items():
        is_nse   = pos.get("is_nse", ticker in NSE_TICKERS)
        cur      = "₹" if is_nse else "$"
        exchange = pos.get("exchange","NSE" if is_nse else "NYSE")
        pos_rows += f"""
        <tr>
          <td>{ticker.replace('.NS','')}</td>
          <td><span style="color:#6b7280;font-size:11px">{exchange}</span></td>
          <td class="{'buy' if pos['direction']=='BUY' else 'sell'}">{pos['direction']}</td>
          <td>{cur}{pos['entry_price']:.2f}</td>
          <td>{cur}{pos['stop']:.2f}</td>
          <td>{cur}{pos['target']:.2f}</td>
          <td>{pos['units']}</td>
          <td style="font-size:11px">{pos['regime']}</td>
          <td>{pos['entry_date']}</td>
        </tr>"""

    sig_rows = ""
    for s in daily_log.get("signals", []):
        color    = "buy" if "BUY" in s.get("action","") else \
                   "sell" if "SELL" in s.get("action","") else \
                   "neutral" if s.get("signal")=="NEUTRAL" else ""
        is_nse   = s.get("exchange","NSE") == "NSE"
        cur      = "₹" if is_nse else "$"
        sig_rows += f"""
        <tr class="{color}">
          <td>{s['ticker'].replace('.NS','')}</td>
          <td style="color:#6b7280;font-size:11px">{s.get('exchange','—')}</td>
          <td>{cur}{s['price']}</td>
          <td style="font-size:11px">{s['regime']}</td>
          <td class="{color}">{s['signal']}</td>
          <td>{s['conf']:.3f}</td>
          <td>{s['flip_prob']:.3f}</td>
          <td>{s.get('action','—')}</td>
          <td style="font-size:11px">{s.get('reason','')}</td>
        </tr>"""

    closed_rows = ""
    for t in reversed(state["closed_trades"][-10:]):
        pnl_class = "profit" if t["pnl"] >= 0 else "loss"
        is_nse    = t.get("is_nse", ".NS" in t["ticker"])
        cur       = "₹" if is_nse else "$"
        closed_rows += f"""
        <tr>
          <td>{t['ticker'].replace('.NS','')}</td>
          <td>{t['direction']}</td>
          <td>{t['entry_date']}</td>
          <td>{t['exit_date']}</td>
          <td>{cur}{t['entry_price']}</td>
          <td>{cur}{t['exit_price']}</td>
          <td class="{pnl_class}">₹{t['pnl']:+.0f}</td>
          <td class="{pnl_class}">{t['pnl_pct']:+.1f}%</td>
          <td>{t['exit_reason']}</td>
        </tr>"""

    sentiment_color = "#ef4444" if flip_prob>=FLIP_HIGH else \
                      "#f59e0b" if flip_prob>=FLIP_MODERATE else "#10b981"
    ret_color = "#10b981" if ret_pct >= 0 else "#ef4444"

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta http-equiv="refresh" content="30">
  <title>AQRA Paper Trading — {today}</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    * {{ box-sizing:border-box; margin:0; padding:0; }}
    body {{ background:#05080f; color:#e2e8f0; font-family:'Segoe UI',sans-serif; padding:20px; }}
    h1 {{ color:#f59e0b; font-size:22px; margin-bottom:4px; }}
    .sub {{ color:#6b7280; font-size:13px; margin-bottom:20px; }}
    .cards {{ display:flex; gap:16px; flex-wrap:wrap; margin-bottom:24px; }}
    .card {{ background:#0a0f1a; border:1px solid #1f2937; border-radius:10px;
             padding:16px 20px; min-width:150px; flex:1; }}
    .card .label {{ color:#6b7280; font-size:12px; margin-bottom:6px; }}
    .card .value {{ font-size:22px; font-weight:700; }}
    .profit {{ color:#10b981; }} .loss {{ color:#ef4444; }}
    .buy {{ color:#3b82f6; }} .sell {{ color:#ef4444; }} .neutral {{ color:#6b7280; }}
    section {{ background:#0a0f1a; border:1px solid #1f2937; border-radius:10px;
               padding:16px; margin-bottom:20px; }}
    section h2 {{ color:#f59e0b; font-size:15px; margin-bottom:14px; border-bottom:1px solid #1f2937; padding-bottom:8px; }}
    table {{ width:100%; border-collapse:collapse; font-size:13px; }}
    th {{ color:#6b7280; font-weight:600; padding:8px 10px; text-align:left;
          border-bottom:1px solid #1f2937; }}
    td {{ padding:7px 10px; border-bottom:1px solid #111827; }}
    tr:hover {{ background:#111827; }}
    .bar-outer {{ background:#1f2937; border-radius:4px; height:8px; width:100px; display:inline-block; vertical-align:middle; }}
    .bar-inner {{ background:#f59e0b; border-radius:4px; height:8px; }}
    canvas {{ max-height:200px; }}
    .tag {{ display:inline-block; padding:2px 8px; border-radius:12px; font-size:11px; font-weight:600; }}
    .tag-calm {{ background:#064e3b; color:#10b981; }}
    .tag-warn {{ background:#78350f; color:#f59e0b; }}
    .tag-high {{ background:#7f1d1d; color:#ef4444; }}
  </style>
</head>
<body>
  <h1>⚡ AQRA — Paper Trading Dashboard</h1>
  <div class="sub">NSE + NYSE Universe (11 tickers)  ·  ₹50,000 Starting Capital  ·  v9 Signal Stack  ·  Auto-refresh: 30s</div>

  <div class="cards">
    <div class="card">
      <div class="label">Cash Available</div>
      <div class="value" style="color:#e2e8f0">₹{state['capital']:,.0f}</div>
    </div>
    <div class="card">
      <div class="label">Open P&L (INR)</div>
      <div class="value {'profit' if open_pnl>=0 else 'loss'}">₹{open_pnl:+,.0f}</div>
      <div style="color:#6b7280;font-size:11px;margin-top:4px">USD @ ₹{usd_inr:.2f}</div>
    </div>
    <div class="card">
      <div class="label">Closed P&L</div>
      <div class="value {'profit' if total_pnl>=0 else 'loss'}">₹{total_pnl:+,.0f}</div>
    </div>
    <div class="card">
      <div class="label">Total Return</div>
      <div class="value" style="color:{ret_color}">{ret_pct:+.2f}%</div>
    </div>
    <div class="card">
      <div class="label">Open Positions</div>
      <div class="value" style="color:#3b82f6">{len(state['open_positions'])}/11</div>
    </div>
    <div class="card">
      <div class="label">Macro Sentiment</div>
      <div class="value" style="color:{sentiment_color}">{flip_prob:.3f}</div>
      <div style="margin-top:6px">
        <span class="tag {'tag-high' if flip_prob>=FLIP_HIGH else 'tag-warn' if flip_prob>=FLIP_MODERATE else 'tag-calm'}">
          {'🔴 HIGH' if flip_prob>=FLIP_HIGH else '🟡 MODERATE' if flip_prob>=FLIP_MODERATE else '🟢 CALM'}
        </span>
      </div>
    </div>
    <div class="card">
      <div class="label">India VIX</div>
      <div class="value" style="color:#8b5cf6">{india_vix}</div>
    </div>
    <div class="card">
      <div class="label">US VIX</div>
      <div class="value" style="color:#06b6d4">{us_vix}</div>
    </div>
    <div class="card">
      <div class="label">Date</div>
      <div class="value" style="color:#6b7280;font-size:15px">{today}</div>
    </div>
  </div>

  <section>
    <h2>📈 Equity Curve</h2>
    <canvas id="equity"></canvas>
  </section>

  <section>
    <h2>📂 Open Positions ({len(state['open_positions'])})</h2>
    <table>
      <tr><th>Ticker</th><th>Exch</th><th>Dir</th><th>Entry</th><th>Stop</th>
          <th>Target</th><th>Units</th><th>Regime</th><th>Entry Date</th></tr>
      {pos_rows if pos_rows else '<tr><td colspan="8" style="color:#6b7280;text-align:center">No open positions</td></tr>'}
    </table>
  </section>

  <section>
    <h2>🔍 Today's Signals — {today}</h2>
    <table>
      <tr><th>Ticker</th><th>Exch</th><th>Price</th><th>Regime</th><th>Signal</th>
          <th>Conf</th><th>Flip Prob</th><th>Action</th><th>Notes</th></tr>
      {sig_rows}
    </table>
  </section>

  <section>
    <h2>📋 Recent Closed Trades (last 10)</h2>
    <table>
      <tr><th>Ticker</th><th>Dir</th><th>Entry</th><th>Exit</th>
          <th>Entry ₹</th><th>Exit ₹</th><th>P&L</th><th>Return</th><th>Reason</th></tr>
      {closed_rows if closed_rows else '<tr><td colspan="9" style="color:#6b7280;text-align:center">No closed trades yet</td></tr>'}
    </table>
  </section>

  <script>
    const ctx = document.getElementById('equity').getContext('2d');
    const eq  = [{equity_js}];
    new Chart(ctx, {{
      type: 'line',
      data: {{
        labels: eq.map(d=>d.x),
        datasets: [{{
          label: 'Portfolio Value (₹)',
          data: eq.map(d=>d.y),
          borderColor: '#f59e0b', backgroundColor: 'rgba(245,158,11,0.1)',
          borderWidth: 2, fill: true, tension: 0.3, pointRadius: 3,
        }}]
      }},
      options: {{
        responsive:true, maintainAspectRatio:false,
        plugins:{{ legend:{{ labels:{{ color:'#e2e8f0' }} }} }},
        scales:{{
          x:{{ ticks:{{ color:'#6b7280' }}, grid:{{ color:'#1f2937' }} }},
          y:{{ ticks:{{ color:'#6b7280', callback: v=>'₹'+v.toLocaleString('en-IN') }},
               grid:{{ color:'#1f2937' }} }}
        }}
      }}
    }});
  </script>
</body>
</html>"""

    report_path = REPORTS_DIR / f"{today}.html"
    with open(report_path,"w",encoding="utf-8") as f:
        f.write(html)
    print(f"\n[Report] Saved → {report_path}")
    return html


# ─────────────────────────────────────────────
#  WhatsApp / Email Notification
# ─────────────────────────────────────────────

def send_telegram(state: dict, daily_log: dict):
    """Send daily summary to Telegram bot."""
    if not TG_TOKEN or not TG_CHAT_ID:
        print("[Notify] Telegram not configured — skipping")
        print("         Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env")
        return

    actions    = daily_log.get("actions", [])
    action_str = "\n".join(f"  • {a}" for a in actions) if actions else "  • No new trades today"

    # Build open positions summary
    open_lines = ""
    for ticker, pos in state["open_positions"].items():
        t = ticker.replace(".NS","")
        open_lines += f"  • {t} {pos['direction']} @ ₹{pos['entry_price']}  stop=₹{pos['stop']}\n"
    if not open_lines:
        open_lines = "  • None\n"

    ret      = daily_log["total_return"]
    fp       = daily_log["flip_prob"]
    ret_icon = "📈" if ret >= 0 else "📉"
    macro_icon = "🔴" if fp >= FLIP_HIGH else "🟡" if fp >= FLIP_MODERATE else "🟢"

    msg = (
        f"⚡ *AQRA Daily Report — {daily_log['date']}*\n\n"
        f"💰 Cash: ₹{state['capital']:,.0f}\n"
        f"📊 Open P\\&L: ₹{daily_log['open_pnl']:+,.0f}\n"
        f"✅ Closed P\\&L: ₹{daily_log['total_pnl']:+,.0f}\n"
        f"{ret_icon} Total Return: *{ret:+.2f}%*\n\n"
        f"{macro_icon} Macro Flip Prob: {fp:.3f}\n"
        f"🇮🇳 India VIX: {daily_log.get('india_vix','—')}  "
        f"🇺🇸 US VIX: {daily_log.get('us_vix','—')}\n"
        f"💱 USD/INR: {daily_log.get('usd_inr', USD_INR_FALLBACK):.2f}\n\n"
        f"*Actions today:*\n{action_str}\n\n"
        f"*Open positions:*\n{open_lines}"
        f"\n_Dashboard: http://localhost:5001_"
    )

    url  = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    resp = requests.post(url, json={
        "chat_id"    : TG_CHAT_ID,
        "text"       : msg,
        "parse_mode" : "Markdown",
    }, timeout=10)

    if resp.status_code == 200:
        print(f"[Notify] ✅ Telegram message sent")
    else:
        print(f"[Notify] ❌ Telegram failed: {resp.status_code} — {resp.text[:120]}")


# ─────────────────────────────────────────────
#  Flask Dashboard
# ─────────────────────────────────────────────

app      = Flask(__name__)
_state   = {}
_html    = "<h1 style='color:white;background:#05080f;padding:20px'>Run the daily update first...</h1>"

@app.route("/")
def dashboard():
    return _html

@app.route("/api/state")
def api_state():
    return jsonify(_state)

@app.route("/api/refresh")
def api_refresh():
    return jsonify({"status":"ok","capital":_state.get("capital",0),
                    "open_positions":len(_state.get("open_positions",{}))})

def start_dashboard():
    print(f"\n[Dashboard] Starting at http://localhost:5001")
    app.run(host="0.0.0.0", port=5001, debug=False, use_reloader=False)


# ─────────────────────────────────────────────
#  Main Entry Point
# ─────────────────────────────────────────────

def main():
    global _state, _html

    print(f"\n{'═'*60}")
    print(f"  AQRA Paper Trading Engine")
    print(f"  Capital: ₹{CAPITAL_INR:,}  |  Universe: {len(ALL_TICKERS)} tickers (NSE + NYSE)")
    print(f"  NSE : {', '.join(t.replace('.NS','') for t in NSE_TICKERS)}")
    print(f"  NYSE: {', '.join(US_TICKERS)}")
    print(f"  Signal stack: v9 (regime + macro sentinel + trailing ATR)")
    print(f"{'═'*60}")

    # Start dashboard in background thread
    t = threading.Thread(target=start_dashboard, daemon=True)
    t.start()
    time.sleep(1)

    # Load state
    state = load_state()
    print(f"\n[State] Loaded  —  "
          f"cash=₹{state['capital']:,.0f}  "
          f"open={len(state['open_positions'])}  "
          f"closed={len(state['closed_trades'])}  "
          f"days_run={len(state['daily_logs'])}")

    # Fetch data
    data, macro, usd_inr = fetch_data()

    if not data:
        print("[Error] No data fetched. Check internet connection.")
        return

    # Run daily logic
    daily_log = run_daily(state, data, macro, usd_inr)

    # Generate report
    html = generate_html_report(state, daily_log)
    _state = state
    _html  = html

    # Send notifications
    send_telegram(state, daily_log)

    print(f"\n{'═'*60}")
    print(f"  ✓ Done for {daily_log['date']}")
    print(f"  Dashboard : http://localhost:5001")
    print(f"  Report    : reports/{daily_log['date']}.html")
    print(f"  State     : {STATE_FILE}")
    print(f"\n  Press Ctrl+C to stop the dashboard server")
    print(f"  Or leave it running — it auto-refreshes every 30s")
    print(f"{'═'*60}\n")

    # Keep dashboard alive
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("\n[AQRA] Dashboard stopped.")


if __name__ == "__main__":
    main()