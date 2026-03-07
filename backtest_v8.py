"""
AQRA — Backtest v8  (Day 7 Upgrade)
=====================================
Base    : v7  (regime-specific params + BAJFINANCE + correlation limits)
Changes :

  1. Trailing ATR Stops
     - Once in a trade, monitor ATR ratio continuously
     - ATR ratio = current_14d_ATR / avg_60d_ATR
     - Thresholds:
         ratio < 1.2  → keep original stop (full room)
         ratio 1.2-1.5 → tighten stop to 2.0x ATR  (early warning)
         ratio > 1.5   → tighten stop to 1.5x ATR  (HIGH_VOL, protect profits)
     - Stop can only MOVE IN FAVOR of position (trailing = ratchet)
       Never widen the stop once tightened
     - For BUY trades: stop can only move UP
     - For SELL trades: stop can only move DOWN

  2. Per-Ticker Regime Override
     - Small lookup: momentum tickers keep bull-widened RSI
       mean-reversion tickers keep standard RSI even in bull regime
     MOMENTUM tickers (use widened RSI in bull):
         NVDA, MSFT, JPM, TCS, HCLTECH, BAJFINANCE
     MEAN_REV tickers (keep standard RSI always):
         INFY, HDFCBANK, RELIANCE, AAPL, GOOGL

Run:
    python backtest_v8.py
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass
from datetime import datetime
from collections import deque

# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────

TICKERS = {
    "NSE": ["TCS.NS","INFY.NS","RELIANCE.NS","HDFCBANK.NS","BAJFINANCE.NS","HCLTECH.NS"],
    "US" : ["AAPL","MSFT","NVDA","GOOGL","JPM"],
}
ALL_TICKERS    = TICKERS["NSE"] + TICKERS["US"]
NSE_TICKERS    = set(TICKERS["NSE"])
START_DATE     = "2015-01-01"
END_DATE       = "2024-12-31"
CAPITAL        = 1_000_000
RISK_PER_TRADE = 0.02
SLIPPAGE_PCT   = 0.001
MC_RUNS        = 1000

# Trailing ATR stop thresholds
ATR_RATIO_WARN  = 1.2   # start tightening
ATR_RATIO_HIGH  = 1.5   # full tighten
ATR_MULT_NORMAL = 3.0   # matches OPTIMAL_BULL from v7
ATR_MULT_WARN   = 2.0   # early warning tighten
ATR_MULT_HIGH   = 1.5   # high-vol tighten
ATR_LOOKBACK    = 60    # days for ATR average

# Per-ticker regime override
MOMENTUM_TICKERS  = {"NVDA","MSFT","JPM","TCS.NS","HCLTECH.NS","BAJFINANCE.NS"}
MEAN_REV_TICKERS  = {"INFY.NS","HDFCBANK.NS","RELIANCE.NS","AAPL","GOOGL"}

# Correlation limit
CORR_LIMIT    = 0.70
CORR_LOOKBACK = 60

# Circuit breaker
CB_SHARPE_THRESH = -0.5
CB_PAUSE_DAYS    = 20

# CS momentum
CS_MOM_LOOKBACK = 63
CS_FILTER_PCT   = 0.50

# VIX
VIX_RULES = [(15,1.2),(20,1.0),(28,0.6),(35,0.3),(999,0.0)]

# ─────────────────────────────────────────────
#  Per-Ticker Regime Override
# ─────────────────────────────────────────────

def get_regime_params_v8(regime_label: str, ticker: str) -> dict:
    """
    Regime params with per-ticker override.
    Momentum tickers: use widened RSI in bull regimes.
    Mean-reversion tickers: always use standard RSI.
    """
    is_momentum = ticker in MOMENTUM_TICKERS

    if "HIGH_VOL" in regime_label:
        return {"rsi": 25, "adx": 22, "atr": 2.0}

    if "BEAR" in regime_label:
        return {"rsi": 28, "adx": 20, "atr": 2.0}

    if "OPTIMAL_BULL" == regime_label or "BULL_RISK_ON" == regime_label:
        if is_momentum:
            return {"rsi": 35, "adx": 15, "atr": ATR_MULT_NORMAL}
        else:
            return {"rsi": 30, "adx": 15, "atr": ATR_MULT_NORMAL}  # keep standard RSI

    if "BULL" in regime_label:
        if is_momentum:
            return {"rsi": 32, "adx": 15, "atr": 2.5}
        else:
            return {"rsi": 30, "adx": 15, "atr": 2.5}

    # Neutral
    return {"rsi": 30, "adx": 18, "atr": 2.5}


# ─────────────────────────────────────────────
#  Trailing ATR Stop Manager
# ─────────────────────────────────────────────

def get_atr_ratio(high, low, close, idx):
    """
    Returns current ATR / avg ATR over lookback.
    Values > 1.2 signal rising volatility (early warning).
    Values > 1.5 signal high volatility (tighten stop).
    """
    if idx < ATR_LOOKBACK + 14:
        return 1.0

    # Current 14-day ATR
    h  = high.iloc[idx-13:idx+1].values
    l  = low.iloc[idx-13:idx+1].values
    cp = close.iloc[idx-14:idx].values
    tr_curr = np.maximum(h-l, np.maximum(abs(h-cp), abs(l-cp)))
    atr_curr = float(tr_curr.mean())

    # Average ATR over lookback
    h2  = high.iloc[idx-ATR_LOOKBACK-13:idx+1].values
    l2  = low.iloc[idx-ATR_LOOKBACK-13:idx+1].values
    cp2 = close.iloc[idx-ATR_LOOKBACK-14:idx].values
    n   = min(len(h2)-1, len(l2)-1, len(cp2))
    if n < 10: return 1.0
    tr_all = np.maximum(h2[1:n+1]-l2[1:n+1],
                        np.maximum(abs(h2[1:n+1]-cp2[:n]),
                                   abs(l2[1:n+1]-cp2[:n])))
    atr_avg = float(tr_all.mean())

    return atr_curr / atr_avg if atr_avg > 0 else 1.0


def update_trailing_stop(
    current_stop : float,
    current_price: float,
    direction    : str,
    atr          : float,
    atr_ratio    : float,
    entry_price  : float,
) -> tuple[float, str]:
    """
    Ratchet the stop based on current ATR ratio.
    Stop can only move in favor of the position — never widen.
    Returns (new_stop, reason)
    """
    # Determine target multiplier based on ATR ratio
    if   atr_ratio >= ATR_RATIO_HIGH: target_mult = ATR_MULT_HIGH
    elif atr_ratio >= ATR_RATIO_WARN: target_mult = ATR_MULT_WARN
    else:                              target_mult = ATR_MULT_NORMAL

    # Compute candidate stop at target multiplier from current price
    if direction == "BUY":
        candidate = current_price - atr * target_mult
        # Ratchet: only move stop UP (never down)
        new_stop = max(current_stop, candidate)
        reason   = f"ATR_ratio={atr_ratio:.2f}_mult={target_mult}"
    else:
        candidate = current_price + atr * target_mult
        # Ratchet: only move stop DOWN (never up)
        new_stop = min(current_stop, candidate)
        reason   = f"ATR_ratio={atr_ratio:.2f}_mult={target_mult}"

    return new_stop, reason


# ─────────────────────────────────────────────
#  Indicators
# ─────────────────────────────────────────────

def compute_atr(high, low, close, idx, period=14):
    if idx < period+1: return float(close.iloc[idx])*0.03
    h  = high.iloc[idx-period:idx+1].values
    l  = low.iloc[idx-period:idx+1].values
    cp = close.iloc[idx-period-1:idx].values
    return float(np.maximum(h-l, np.maximum(abs(h-cp), abs(l-cp))).mean())


def compute_adx(high, low, close, idx, period=14):
    if idx < period*2: return 0.0
    h=high.iloc[idx-period*2:idx+1].values
    l=low.iloc[idx-period*2:idx+1].values
    c=close.iloc[idx-period*2:idx+1].values
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
        dip=np.where(at>0,100*dp/at,0)
        dim=np.where(at>0,100*dm/at,0)
        dx=np.where((dip+dim)>0,100*abs(dip-dim)/(dip+dim),0)
    v=sm(dx,period)
    return float(v[-1]) if len(v)>0 else 0.0


def compute_signals_v8(df, idx, regime_label, ticker):
    """Signals with per-ticker regime override."""
    rp         = get_regime_params_v8(regime_label, ticker)
    rsi_thresh = rp["rsi"]
    adx_thresh = rp["adx"]
    atr_mult   = rp["atr"]

    close  = df["Close"]; high=df["High"]; low=df["Low"]
    volume = df.get("Volume")

    if idx < 60: return "NEUTRAL", 0.0, atr_mult

    adx = compute_adx(high, low, close, idx)
    if adx < adx_thresh: return "NEUTRAL", 0.0, atr_mult

    window = close.iloc[:idx+1]; price = float(close.iloc[idx])

    sma20=window.rolling(20).mean().iloc[-1]
    sma50=window.rolling(50).mean().iloc[-1]
    gap=(sma20-sma50)/sma50
    sma_s="BUY" if gap>0.005 else "SELL" if gap<-0.005 else "NEUTRAL"

    w  = close.iloc[idx-14:idx+1].diff()
    g  = w.clip(lower=0).mean(); l=(-w.clip(upper=0)).mean()
    rsi= 100-(100/(1+g/l)) if l!=0 else 100.0
    rsi_s="BUY" if rsi<rsi_thresh else "SELL" if rsi>(100-rsi_thresh) else "NEUTRAL"

    sb  = window.rolling(20).mean().iloc[-1]
    stdb= window.rolling(20).std().iloc[-1]
    bb_s="BUY" if price<sb-2*stdb else "SELL" if price>sb+2*stdb else "NEUTRAL"

    w2  = close.iloc[max(0,idx-50):idx+1]
    ef  = w2.ewm(span=12,adjust=False).mean()
    es  = w2.ewm(span=26,adjust=False).mean()
    ml  = ef-es; sl2=ml.ewm(span=9,adjust=False).mean()
    macd_s="BUY" if float(ml.iloc[-1]-sl2.iloc[-1])>0 else \
           "SELL" if float(ml.iloc[-1]-sl2.iloc[-1])<0 else "NEUTRAL"

    if volume is not None and len(volume)>20:
        obv  =(np.sign(close.iloc[:idx+1].diff().fillna(0))*volume.iloc[:idx+1]).cumsum()
        osma = obv.rolling(20).mean()
        ov,osv=float(obv.iloc[-1]),float(osma.iloc[-1])
        obv_s="BUY" if ov>osv*1.02 else "SELL" if ov<osv*0.98 else "NEUTRAL"
    else:
        obv_s="NEUTRAL"

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

    if   bw>sw and bw/tw>=0.45: return "BUY",  bw/tw, atr_mult
    elif sw>bw and sw/tw>=0.45: return "SELL", sw/tw, atr_mult
    return "NEUTRAL", 0.0, atr_mult


# ─────────────────────────────────────────────
#  Regime Detection
# ─────────────────────────────────────────────

def get_regime(close, universe_df, idx):
    if idx<200: return 0.5,True,True,"WARMING_UP"
    price=float(close.iloc[idx]); sma200=float(close.iloc[idx-199:idx+1].mean())
    trend="BULL" if price>sma200*1.02 else "BEAR" if price<sma200*0.98 else "NEUTRAL"
    ac=close.iloc[idx-13:idx+1].diff().abs().mean()
    aa=close.iloc[idx-73:idx-13].diff().abs().mean()
    ar=ac/aa if aa>0 else 1.0
    vol="HIGH" if ar>1.5 else "LOW" if ar<0.7 else "NORMAL"
    above=sum(1 for c in universe_df.columns
              if idx<len(universe_df[c]) and
              float(universe_df[c].iloc[idx])>
              float(universe_df[c].iloc[max(0,idx-49):idx+1].mean()))
    b=above/len(universe_df.columns)
    br="RISK_ON" if b>0.6 else "RISK_OFF" if b<0.4 else "NEUTRAL"
    if trend=="BEAR" and vol=="HIGH" and br=="RISK_OFF": return 0.0,False,False,"BLACKOUT"
    if vol=="HIGH": return 0.35,trend!="BEAR",True,f"HIGH_VOL_{trend}"
    if trend=="BEAR":
        s=0.5 if br=="RISK_OFF" else 0.65; return s,False,True,f"BEAR_{br}"
    if trend=="BULL":
        if vol=="LOW" and br=="RISK_ON": return 1.0,True,True,"OPTIMAL_BULL"
        if br=="RISK_ON":               return 0.85,True,True,"BULL_RISK_ON"
        return 0.70,True,True,"BULL_NEUTRAL"
    return (0.50,True,True,"NEUTRAL_CAUTIOUS") if br=="RISK_OFF" \
        else (0.75,True,True,"NEUTRAL_NORMAL")


def get_vix_mult(vix, idx):
    if vix is None or idx>=len(vix): return 1.0,20.0
    v=float(vix.iloc[idx])
    if np.isnan(v): return 1.0,20.0
    for t,m in VIX_RULES:
        if v<t: return m,v
    return 0.0,v


def get_cs_rank(ticker, universe_df, idx):
    if idx<CS_MOM_LOOKBACK: return 0.5
    rets={}
    for col in universe_df.columns:
        s=universe_df[col].dropna()
        if idx<len(s) and idx>=CS_MOM_LOOKBACK:
            p=float(s.iloc[idx-CS_MOM_LOOKBACK]); c=float(s.iloc[idx])
            if p>0: rets[col]=(c-p)/p
    if not rets or ticker not in rets: return 0.5
    srt=sorted(rets.keys(),key=lambda t:rets[t])
    return srt.index(ticker)/max(len(srt)-1,1)


# ─────────────────────────────────────────────
#  Position Manager
# ─────────────────────────────────────────────

class PositionManager:
    def __init__(self, universe_close):
        self.universe=universe_close; self.open_pos={}

    def can_open(self, ticker, idx):
        for ot in list(self.open_pos.keys()):
            if ot==ticker: continue
            if self._corr(ticker,ot,idx)>CORR_LIMIT: return False,ot
        return True,""

    def _corr(self, t1, t2, idx):
        if idx<CORR_LOOKBACK+1: return 0.0
        if t1 not in self.universe.columns or t2 not in self.universe.columns: return 0.0
        try:
            s1=self.universe[t1].iloc[idx-CORR_LOOKBACK:idx].pct_change().dropna()
            s2=self.universe[t2].iloc[idx-CORR_LOOKBACK:idx].pct_change().dropna()
            al=pd.concat([s1,s2],axis=1).dropna()
            return float(al.iloc[:,0].corr(al.iloc[:,1])) if len(al)>=10 else 0.0
        except: return 0.0

    def open(self, ticker, idx, direction):
        self.open_pos[ticker]={"direction":direction,"entry_idx":idx}

    def close(self, ticker): self.open_pos.pop(ticker,None)


# ─────────────────────────────────────────────
#  Circuit Breaker
# ─────────────────────────────────────────────

class CircuitBreaker:
    def __init__(self):
        self.pnls={}; self.paused_until={}; self.triggers={}; self.consec={}

    def record(self, ticker, pnl, idx):
        if ticker not in self.pnls: self.pnls[ticker]=deque(maxlen=25)
        self.pnls[ticker].append(pnl)
        p=list(self.pnls[ticker])
        if len(p)>=8:
            a=np.array(p); s=np.std(a)
            rs=(np.mean(a)/s)*np.sqrt(50) if s>0 else 0
            if rs<CB_SHARPE_THRESH:
                self.consec[ticker]=self.consec.get(ticker,0)+1
                if self.consec[ticker]>=2:
                    self.paused_until[ticker]=idx+CB_PAUSE_DAYS
                    self.triggers[ticker]=self.triggers.get(ticker,0)+1
                    self.consec[ticker]=0
            else: self.consec[ticker]=0

    def paused(self, ticker, idx):
        return idx<self.paused_until.get(ticker,0)


# ─────────────────────────────────────────────
#  Trade Dataclass
# ─────────────────────────────────────────────

@dataclass
class Trade:
    ticker:str; direction:str; entry_date:str; exit_date:str
    entry_price:float; exit_price:float; units:float; pnl:float
    pnl_pct:float; exit_reason:str; confidence:float
    regime_label:str; size_mult:float; vix_entry:float
    ticker_type:str; stop_tightened:bool; max_atr_ratio:float


# ─────────────────────────────────────────────
#  Main Backtest Loop v8
# ─────────────────────────────────────────────

def backtest_v8(ticker, df, universe_df, vix_series, cb, pos_mgr):
    trades   = []
    close    = df["Close"]; high=df["High"]; low=df["Low"]
    open_    = df["Open"];  dates=df.index;  n=len(close)
    in_trade = False
    tick_type= "MOM" if ticker in MOMENTUM_TICKERS else "MR"

    e_idx=e_px=direc=units=conf=rlbl=smult=vix_v=atr_v=s_px=t_px=None
    stop_tightened=False; max_atr_ratio=1.0

    for i in range(200, n):
        if not in_trade:
            if cb.paused(ticker, i): continue

            r_mult,allow_buy,allow_sell,r_lbl=get_regime(close,universe_df,i-1)
            if r_mult==0.0: continue

            sig,conf_v,atr_m=compute_signals_v8(df,i-1,r_lbl,ticker)
            if sig=="NEUTRAL" or conf_v<0.45: continue
            if sig=="BUY"  and not allow_buy:  continue
            if sig=="SELL" and not allow_sell: continue

            v_mult,vix_val=get_vix_mult(vix_series,i-1)
            if v_mult==0.0: continue

            cs_rank=get_cs_rank(ticker,universe_df,i-1)
            if sig=="BUY"  and cs_rank<CS_FILTER_PCT: continue
            if sig=="SELL" and cs_rank>(1-CS_FILTER_PCT): continue

            can_open,_=pos_mgr.can_open(ticker,i-1)
            if not can_open: continue

            atr      =compute_atr(high,low,close,i-1)
            stop_dist=atr*atr_m
            tgt_dist =stop_dist*2.0

            raw_e=float(open_.iloc[i])
            ep   =raw_e*(1+SLIPPAGE_PCT) if sig=="BUY" else raw_e*(1-SLIPPAGE_PCT)

            combined=r_mult*v_mult
            risk_amt =CAPITAL*RISK_PER_TRADE*combined
            u=risk_amt/stop_dist if stop_dist>0 else 0
            if u<=0: continue

            in_trade=True; e_idx=i; e_px=ep; direc=sig; units=u
            conf=conf_v; rlbl=r_lbl; smult=combined; vix_v=vix_val; atr_v=atr
            s_px=ep-stop_dist if sig=="BUY" else ep+stop_dist
            t_px=ep+tgt_dist  if sig=="BUY" else ep-tgt_dist
            stop_tightened=False; max_atr_ratio=1.0
            pos_mgr.open(ticker,i,sig)

        else:
            curr=float(close.iloc[i])

            # ── Trailing ATR stop update ──────────────────────────
            atr_now   = compute_atr(high,low,close,i)
            atr_ratio = get_atr_ratio(high,low,close,i)
            max_atr_ratio = max(max_atr_ratio, atr_ratio)

            new_stop, _ = update_trailing_stop(
                current_stop =s_px,
                current_price=curr,
                direction    =direc,
                atr          =atr_now,
                atr_ratio    =atr_ratio,
                entry_price  =e_px,
            )
            if new_stop != s_px:
                stop_tightened = True
                s_px = new_stop
            # ─────────────────────────────────────────────────────

            hs=curr<=s_px if direc=="BUY" else curr>=s_px
            ht=curr>=t_px if direc=="BUY" else curr<=t_px
            pfn=(lambda x:(x-e_px)*units) if direc=="BUY" else (lambda x:(e_px-x)*units)
            xp=xr=None
            if   hs: xp,xr=s_px,"STOP_LOSS"
            elif ht: xp,xr=t_px,"TAKE_PROFIT"
            elif i==n-1: xp,xr=curr,"END_OF_DATA"
            if xr:
                pnl=pfn(xp)
                trades.append(Trade(
                    ticker=ticker,direction=direc,
                    entry_date=str(dates[e_idx].date()),
                    exit_date=str(dates[i].date()),
                    entry_price=round(e_px,4),exit_price=round(xp,4),
                    units=round(units,4),pnl=round(pnl,2),
                    pnl_pct=round(pnl/(e_px*units)*100 if e_px*units>0 else 0,3),
                    exit_reason=xr,confidence=conf,regime_label=rlbl,
                    size_mult=smult,vix_entry=vix_v,
                    ticker_type=tick_type,
                    stop_tightened=stop_tightened,
                    max_atr_ratio=round(max_atr_ratio,3),
                ))
                cb.record(ticker,pnl,i)
                pos_mgr.close(ticker)
                in_trade=False

    return trades


# ─────────────────────────────────────────────
#  Metrics + Monte Carlo
# ─────────────────────────────────────────────

def compute_metrics(trades, label=""):
    if not trades:
        return {"label":label,"trades":0,"cagr":0,"sharpe":0,
                "max_drawdown":0,"win_rate":0,"profit_factor":0,
                "total_pnl":0,"equity_curve":[CAPITAL]}
    pnls=[t.pnl for t in trades]
    winners=[t for t in trades if t.pnl>0]
    losers =[t for t in trades if t.pnl<=0]
    tp=sum(pnls); wr=len(winners)/len(trades)*100
    gw=sum(t.pnl for t in winners); gl=abs(sum(t.pnl for t in losers))
    pf=gw/gl if gl>0 else float('inf')
    eq=np.array([CAPITAL+sum(pnls[:i]) for i in range(len(pnls)+1)])
    r=np.diff(eq)/eq[:-1]; r=r[np.isfinite(r)&(r!=0)]
    sh=(np.mean(r)/np.std(r))*np.sqrt(252) if len(r)>1 and np.std(r)>0 else 0
    pk=eq[0]; md=0
    for v in eq:
        if v>pk: pk=v
        d=(pk-v)/pk
        if d>md: md=d
    yrs=(datetime.strptime(END_DATE,"%Y-%m-%d")-
         datetime.strptime(START_DATE,"%Y-%m-%d")).days/365.25
    fn=CAPITAL+tp; cagr=(fn/CAPITAL)**(1/yrs)-1 if yrs>0 and fn>0 else 0
    return {"label":label,"trades":len(trades),"win_rate":round(wr,1),
            "total_pnl":round(tp,2),"cagr":round(cagr*100,2),"sharpe":round(sh,3),
            "max_drawdown":round(md*100,2),"profit_factor":round(pf,3),
            "avg_win":round(np.mean([t.pnl for t in winners]),2) if winners else 0,
            "avg_loss":round(np.mean([t.pnl for t in losers]),2) if losers else 0,
            "equity_curve":eq.tolist()}


def monte_carlo(trades, n_runs=MC_RUNS):
    print(f"\n[MC] {n_runs} simulations on {len(trades)} trades...")
    pnls=np.array([t.pnl for t in trades]); n=len(pnls)
    yrs=(datetime.strptime(END_DATE,"%Y-%m-%d")-
         datetime.strptime(START_DATE,"%Y-%m-%d")).days/365.25
    cagrs=[]; sharpes=[]; dds=[]; wrs=[]; finals=[]; eqs=[]
    rng=np.random.default_rng(42)
    for run in range(n_runs):
        s=rng.choice(pnls,size=n,replace=True)
        eq=np.array([CAPITAL+s[:i].sum() for i in range(n+1)])
        r=np.diff(eq)/eq[:-1]; r=r[np.isfinite(r)&(r!=0)]
        fn=float(eq[-1])
        cagrs.append(((fn/CAPITAL)**(1/yrs)-1)*100 if fn>0 else -100)
        sharpes.append((np.mean(r)/np.std(r))*np.sqrt(252) if len(r)>1 and np.std(r)>0 else 0)
        pk=eq[0]; md=0
        for v in eq:
            if v>pk: pk=v
            d=(pk-v)/pk
            if d>md: md=d
        dds.append(md*100); wrs.append(float(np.sum(s>0))/n*100); finals.append(fn)
        if run%100==0: eqs.append(eq)
    res={"cagrs":np.array(cagrs),"sharpes":np.array(sharpes),
         "max_drawdowns":np.array(dds),"win_rates":np.array(wrs),
         "final_equities":np.array(finals),"equity_curves":eqs}
    print(f"\n{'='*65}")
    print(f"  MONTE CARLO  ({n_runs} runs)")
    print(f"{'='*65}")
    for lbl,arr in [("CAGR %",cagrs),("Sharpe",sharpes),("Max DD %",dds),("Win Rate %",wrs)]:
        a=np.array(arr)
        print(f"  {lbl:<14} p5={np.percentile(a,5):>6.2f}  p25={np.percentile(a,25):>6.2f}  "
              f"median={np.median(a):>6.2f}  p75={np.percentile(a,75):>6.2f}  "
              f"p95={np.percentile(a,95):>6.2f}")
    ca=np.array(cagrs); sa=np.array(sharpes); da=np.array(dds)
    print(f"\n  P(CAGR > 0%)   = {np.mean(ca>0)*100:.1f}%")
    print(f"  P(CAGR > 10%)  = {np.mean(ca>10)*100:.1f}%")
    print(f"  P(Sharpe > 1)  = {np.mean(sa>1)*100:.1f}%")
    print(f"  P(Sharpe > 2)  = {np.mean(sa>2)*100:.1f}%")
    print(f"  P(MaxDD < 20%) = {np.mean(da<20)*100:.1f}%")
    print(f"{'='*65}")
    return res


# ─────────────────────────────────────────────
#  Charts
# ─────────────────────────────────────────────

def plot_v8(all_trades_v7, all_trades_v8, mv7, mv8, mc, ticker_metrics):
    BG,SURF='#05080f','#0a0f1a'
    TEXT,MUT='#e2e8f0','#6b7280'
    GOLD,GRN,RED='#f59e0b','#10b981','#ef4444'
    BLUE,PUR,CYN='#3b82f6','#8b5cf6','#06b6d4'

    fig=plt.figure(figsize=(20,22),facecolor=BG)
    gs =gridspec.GridSpec(4,3,figure=fig,hspace=0.50,wspace=0.35)

    def sax(ax,t):
        ax.set_facecolor(SURF); ax.tick_params(colors=MUT,labelsize=8)
        ax.set_title(t,color=TEXT,fontsize=10,fontweight='bold',pad=8)
        for s in ax.spines.values(): s.set_edgecolor('#1f2937')

    # ── Row 0: Equity + MC fan ──
    ax0=fig.add_subplot(gs[0,:2])
    eq7=mv7["equity_curve"]; eq8=mv8["equity_curve"]
    eqs=np.array(mc["equity_curves"])
    if len(eqs)>0:
        tl=len(eq8)
        rs=[np.interp(np.linspace(0,1,tl),np.linspace(0,1,len(e)),e)
            for e in eqs if len(e)>1]
        if rs:
            m=np.array(rs)
            ax0.fill_between(range(tl),np.percentile(m,5,0),np.percentile(m,95,0),
                             color=GOLD,alpha=0.08,label='MC 5–95%')
            ax0.fill_between(range(tl),np.percentile(m,25,0),np.percentile(m,75,0),
                             color=GOLD,alpha=0.15,label='MC 25–75%')
    ax0.plot(range(len(eq7)),eq7,color=RED, lw=1.0,alpha=0.6,label='v7')
    ax0.plot(range(len(eq8)),eq8,color=GOLD,lw=1.5,label='v8 Trailing Stops')
    ax0.axhline(CAPITAL,color=MUT,lw=0.5,ls='--',alpha=0.4)
    ax0.legend(facecolor=SURF,edgecolor='#1f2937',labelcolor=TEXT,fontsize=8)
    sax(ax0,'Equity Curve: v7 vs v8 with Monte Carlo Fan')
    ax0.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f'{x/1e6:.1f}M'))

    # ── Row 0: Metrics table ──
    axm=fig.add_subplot(gs[0,2]); axm.set_facecolor(SURF); axm.axis('off')
    axm.set_title('v7 → v8 Improvement',color=TEXT,fontsize=10,fontweight='bold',pad=8)
    rows=[("CAGR",      f"{mv7['cagr']}%",        f"{mv8['cagr']}%"),
          ("Sharpe",    f"{mv7['sharpe']}",         f"{mv8['sharpe']}"),
          ("Max DD",    f"-{mv7['max_drawdown']}%", f"-{mv8['max_drawdown']}%"),
          ("Win Rate",  f"{mv7['win_rate']}%",      f"{mv8['win_rate']}%"),
          ("Prof Fact", f"{mv7['profit_factor']}",  f"{mv8['profit_factor']}"),
          ("Trades",    f"{mv7['trades']}",          f"{mv8['trades']}"),
          ("P&L",       f"{mv7['total_pnl']/1e6:.2f}M", f"{mv8['total_pnl']/1e6:.2f}M")]
    axm.text(0.35,0.96,"v7",transform=axm.transAxes,color=RED, fontsize=9,fontweight='bold',ha='center')
    axm.text(0.75,0.96,"v8",transform=axm.transAxes,color=GOLD,fontsize=9,fontweight='bold',ha='center')
    for j,(l,v7,v8) in enumerate(rows):
        y=0.84-j*0.12
        axm.text(0.02,y,l, transform=axm.transAxes,color=MUT, fontsize=9)
        axm.text(0.35,y,v7,transform=axm.transAxes,color=RED, fontsize=9,ha='center',fontweight='bold')
        axm.text(0.75,y,v8,transform=axm.transAxes,color=GOLD,fontsize=9,ha='center',fontweight='bold')
    for s in axm.spines.values(): s.set_edgecolor('#1f2937')

    # ── Row 1: Trailing stop effectiveness ──
    ax1=fig.add_subplot(gs[1,0])
    tightened    =[t for t in all_trades_v8 if t.stop_tightened]
    not_tightened=[t for t in all_trades_v8 if not t.stop_tightened]
    cats  =['Stop Tightened','No Tighten']
    wr_t  =len([t for t in tightened     if t.pnl>0])/len(tightened)     *100 if tightened     else 0
    wr_nt =len([t for t in not_tightened if t.pnl>0])/len(not_tightened) *100 if not_tightened else 0
    apnl_t =np.mean([t.pnl for t in tightened])     if tightened     else 0
    apnl_nt=np.mean([t.pnl for t in not_tightened]) if not_tightened else 0
    x=np.arange(2); w=0.35
    ax1b=ax1.twinx()
    ax1.bar(x-w/2,[wr_t,wr_nt],w,color=[GRN,BLUE],alpha=0.7,label='Win Rate %')
    ax1b.bar(x+w/2,[apnl_t,apnl_nt],w,color=[GOLD,PUR],alpha=0.7,label='Avg P&L')
    ax1.set_xticks(x); ax1.set_xticklabels(cats,fontsize=8)
    ax1.set_ylabel('Win Rate %',color=MUT,fontsize=8)
    ax1b.set_ylabel('Avg P&L',color=MUT,fontsize=8)
    ax1.tick_params(colors=MUT); ax1b.tick_params(colors=MUT)
    sax(ax1,'Trailing Stop: Tightened vs Not')
    pct_tight=len(tightened)/len(all_trades_v8)*100 if all_trades_v8 else 0
    ax1.set_title(f'Trailing Stop Effectiveness  ({pct_tight:.0f}% trades tightened)',
                  color=TEXT,fontsize=9,fontweight='bold',pad=8)

    # ── Row 1: ATR ratio distribution at exit ──
    ax2=fig.add_subplot(gs[1,1])
    atr_ratios=[t.max_atr_ratio for t in all_trades_v8]
    ax2.hist(atr_ratios,bins=40,color=CYN,alpha=0.75,edgecolor='none')
    ax2.axvline(ATR_RATIO_WARN,color=GOLD,lw=1.2,ls='--',label=f'Warn ({ATR_RATIO_WARN})')
    ax2.axvline(ATR_RATIO_HIGH,color=RED, lw=1.2,ls='--',label=f'High ({ATR_RATIO_HIGH})')
    ax2.legend(facecolor=SURF,edgecolor='#1f2937',labelcolor=TEXT,fontsize=8)
    sax(ax2,'Max ATR Ratio Seen During Trade')
    ax2.set_xlabel('Max ATR Ratio',color=MUT,fontsize=8)

    # ── Row 1: MOM vs MR ticker performance ──
    ax3=fig.add_subplot(gs[1,2])
    mom_trades=[t for t in all_trades_v8 if t.ticker_type=="MOM"]
    mr_trades =[t for t in all_trades_v8 if t.ticker_type=="MR"]
    mom_m=compute_metrics(mom_trades,"MOM")
    mr_m =compute_metrics(mr_trades, "MR")
    cats2=['Momentum\nTickers','Mean-Rev\nTickers']
    sharpes2=[mom_m['sharpe'],mr_m['sharpe']]
    cagrs2  =[mom_m['cagr'],  mr_m['cagr']]
    colors2 =[GRN if s>0 else RED for s in sharpes2]
    x2=np.arange(2)
    ax3b=ax3.twinx()
    ax3.bar(x2-w/2,sharpes2,w,color=colors2,alpha=0.8,label='Sharpe')
    ax3b.bar(x2+w/2,cagrs2,w,color=[GOLD,PUR],alpha=0.7,label='CAGR %')
    ax3.set_xticks(x2); ax3.set_xticklabels(cats2,fontsize=9)
    ax3.axhline(0,color=MUT,lw=0.5)
    ax3.set_ylabel('Sharpe',color=MUT,fontsize=8)
    ax3b.set_ylabel('CAGR %',color=MUT,fontsize=8)
    ax3.tick_params(colors=MUT); ax3b.tick_params(colors=MUT)
    sax(ax3,'MOM vs MR Ticker Performance')

    # ── Row 2: MC distributions ──
    ax4=fig.add_subplot(gs[2,0])
    ax4.hist(mc["cagrs"],bins=60,color=GOLD,alpha=0.75,edgecolor='none')
    ax4.axvline(np.percentile(mc["cagrs"],5), color=RED,lw=1.2,ls='--',label='p5')
    ax4.axvline(np.median(mc["cagrs"]),        color=GRN,lw=1.5,label='median')
    ax4.axvline(np.percentile(mc["cagrs"],95),color=CYN,lw=1.2,ls='--',label='p95')
    ax4.axvline(0,color=MUT,lw=0.8)
    ax4.legend(facecolor=SURF,edgecolor='#1f2937',labelcolor=TEXT,fontsize=7)
    sax(ax4,f'MC CAGR  (n={MC_RUNS})')

    ax5=fig.add_subplot(gs[2,1])
    ax5.hist(mc["sharpes"],bins=60,color=PUR,alpha=0.75,edgecolor='none')
    ax5.axvline(np.percentile(mc["sharpes"],5), color=RED,lw=1.2,ls='--',label='p5')
    ax5.axvline(np.median(mc["sharpes"]),        color=GRN,lw=1.5,label='median')
    ax5.axvline(1.0,color=GOLD,lw=1.0,ls=':',label='Sharpe=1')
    ax5.axvline(2.0,color=CYN, lw=1.0,ls=':',label='Sharpe=2')
    ax5.legend(facecolor=SURF,edgecolor='#1f2937',labelcolor=TEXT,fontsize=7)
    sax(ax5,'MC Sharpe Distribution')

    ax6=fig.add_subplot(gs[2,2])
    def dd_s(eq):
        eq=np.array(eq); pk=eq[0]; dd=[]
        for v in eq:
            if v>pk: pk=v
            dd.append((v-pk)/pk*100)
        return dd
    dd7=dd_s(mv7["equity_curve"]); dd8=dd_s(mv8["equity_curve"])
    ax6.fill_between(range(len(dd7)),dd7,0,color=RED, alpha=0.3,label='v7')
    ax6.fill_between(range(len(dd8)),dd8,0,color=GOLD,alpha=0.3,label='v8')
    ax6.plot(range(len(dd7)),dd7,color=RED, lw=0.8,alpha=0.7)
    ax6.plot(range(len(dd8)),dd8,color=GOLD,lw=1.0)
    ax6.legend(facecolor=SURF,edgecolor='#1f2937',labelcolor=TEXT,fontsize=8)
    sax(ax6,'Drawdown: v7 vs v8')

    # ── Row 3: Sharpe per ticker ──
    ax7=fig.add_subplot(gs[3,:])
    ts=[t.replace('.NS','') for t in ticker_metrics]
    s7=[ticker_metrics[t].get('sharpe_v7',0) for t in ticker_metrics]
    s8=[ticker_metrics[t].get('sharpe_v8',0) for t in ticker_metrics]
    x=np.arange(len(ts)); w=0.35
    ax7.bar(x-w/2,s7,w,color=RED, alpha=0.7,label='v7')
    ax7.bar(x+w/2,s8,w,color=GOLD,alpha=0.8,label='v8 (trailing stops + per-ticker override)')
    ax7.axhline(1.0,color=GRN,lw=0.8,ls='--',alpha=0.6,label='Sharpe=1')
    ax7.axhline(2.0,color=CYN,lw=0.6,ls=':' ,alpha=0.5,label='Sharpe=2')
    ax7.axhline(0,  color=MUT,lw=0.5)
    # Label MOM vs MR
    for i,t in enumerate(ticker_metrics):
        tp="MOM" if t in MOMENTUM_TICKERS else "MR"
        ax7.text(i,-0.3,tp,ha='center',fontsize=6,color=CYN if tp=="MOM" else GOLD)
    ax7.set_xticks(x); ax7.set_xticklabels(ts,fontsize=9)
    ax7.legend(facecolor=SURF,edgecolor='#1f2937',labelcolor=TEXT,fontsize=9)
    sax(ax7,'Sharpe per Ticker: v7 vs v8  (MOM/MR labels below)')

    fig.suptitle('AQRA  ·  Day 7: Trailing ATR Stops + Per-Ticker Regime Override  ·  2015–2024',
                 color=TEXT,fontsize=14,fontweight='bold',y=0.998)
    plt.savefig('backtest_v8.png',dpi=150,bbox_inches='tight',facecolor=BG,edgecolor='none')
    print("[Chart] Saved → backtest_v8.png")


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def run():
    print(f"\n{'='*65}")
    print(f"  AQRA BACKTEST v8  —  Day 7")
    print(f"  Trailing ATR Stops + Per-Ticker Regime Override")
    print(f"  MOM tickers: {', '.join(t.replace('.NS','') for t in MOMENTUM_TICKERS)}")
    print(f"  MR  tickers: {', '.join(t.replace('.NS','') for t in MEAN_REV_TICKERS)}")
    print(f"{'='*65}\n")

    raw_data={}
    print("[Data] Downloading...")
    for ticker in ALL_TICKERS:
        try:
            df=yf.download(ticker,start=START_DATE,end=END_DATE,
                           auto_adjust=True,progress=False)
            if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(0)
            if not df.empty and len(df)>200:
                raw_data[ticker]=df; print(f"  ✓ {ticker:<15} {len(df)} days")
            else: print(f"  ✗ {ticker} insufficient data")
        except Exception as e: print(f"  ✗ {ticker}: {e}")

    vix_us=vix_in=None
    print("\n[Data] VIX...")
    for sym,name in [("^VIX","US"),("^INDIAVIX","NSE")]:
        try:
            d=yf.download(sym,start=START_DATE,end=END_DATE,auto_adjust=True,progress=False)
            if isinstance(d.columns,pd.MultiIndex): d.columns=d.columns.get_level_values(0)
            v=d["Close"]
            if name=="US": vix_us=v
            else:          vix_in=v
            print(f"  ✓ {sym} {len(v)} days")
        except: print(f"  ✗ {sym}")
    if vix_in is None: vix_in=vix_us

    universe_close=pd.DataFrame({t:raw_data[t]["Close"] for t in raw_data}).ffill()

    all_v7=[]; all_v8=[]; ticker_metrics={}
    cb8=CircuitBreaker(); pm8=PositionManager(universe_close)

    print("\n[Backtest] Running v7 and v8...")
    for ticker,df in raw_data.items():
        is_nse=ticker in NSE_TICKERS
        vix_s =vix_in if is_nse else vix_us
        vix_a =vix_s.reindex(df.index).ffill() if vix_s is not None else None

        from backtest7 import backtest_v7, CircuitBreaker as CB7, PositionManager as PM7
        cb7=CB7(); pm7=PM7(universe_close)
        tv7=backtest_v7(ticker,df,universe_close,vix_a,cb7,pm7)

        tv8=backtest_v8(ticker,df,universe_close,vix_a,cb8,pm8)

        all_v7.extend(tv7); all_v8.extend(tv8)
        m7=compute_metrics(tv7,ticker); m8=compute_metrics(tv8,ticker)
        ticker_metrics[ticker]={
            "sharpe_v7":m7.get("sharpe",0),"sharpe_v8":m8.get("sharpe",0),
            "cagr_v7"  :m7.get("cagr",0),  "cagr_v8"  :m8.get("cagr",0),
            "maxdd_v7" :m7.get("max_drawdown",0),"maxdd_v8":m8.get("max_drawdown",0),
        }
        tick_type="MOM" if ticker in MOMENTUM_TICKERS else "MR"
        tight_count=len([t for t in tv8 if t.stop_tightened])
        print(f"  {ticker.replace('.NS',''):<12} [{tick_type}]  "
              f"v7: S={m7.get('sharpe',0):>5.2f} DD={m7.get('max_drawdown',0):.1f}%  │  "
              f"v8: S={m8.get('sharpe',0):>5.2f} DD={m8.get('max_drawdown',0):.1f}%  "
              f"trades={m8.get('trades',0)}  tightened={tight_count}")

    mv7=compute_metrics(all_v7,"v7"); mv8=compute_metrics(all_v8,"v8")

    print(f"\n{'='*65}")
    print(f"  {'METRIC':<20} {'v7':>12}  {'v8':>12}  {'CHANGE':>10}")
    print(f"  {'─'*57}")
    for lbl,k in [("CAGR","cagr"),("Sharpe","sharpe"),("Max Drawdown","max_drawdown"),
                  ("Win Rate","win_rate"),("Profit Factor","profit_factor"),
                  ("Trades","trades"),("Total P&L","total_pnl")]:
        v7v=mv7.get(k,0); v8v=mv8.get(k,0)
        if k=="max_drawdown":
            print(f"  {lbl:<20} {f'-{v7v}%':>12}  {f'-{v8v}%':>12}  {f'{v7v-v8v:+.2f}%':>10}")
        elif k=="total_pnl":
            print(f"  {lbl:<20} {f'{v7v:+,.0f}':>12}  {f'{v8v:+,.0f}':>12}  {f'{v8v-v7v:+,.0f}':>10}")
        elif k=="trades":
            print(f"  {lbl:<20} {v7v:>12}  {v8v:>12}  {f'{v8v-v7v:+d}':>10}")
        else:
            sfx="%" if k in ("cagr","win_rate") else ""
            print(f"  {lbl:<20} {f'{v7v}{sfx}':>12}  {f'{v8v}{sfx}':>12}  {f'{v8v-v7v:+.3f}':>10}")
    print(f"{'='*65}")

    mc=monte_carlo(all_v8)
    print("\n[Chart] Generating...")
    plot_v8(all_v7,all_v8,mv7,mv8,mc,ticker_metrics)
    print(f"\n[AQRA v8] ✓ Done. Open backtest_v8.png")


if __name__=="__main__":
    run()
