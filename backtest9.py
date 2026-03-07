"""
AQRA — Backtest v9  (Day 9 Upgrade)
=====================================
Base    : v8  (trailing ATR stops + per-ticker override + correlation limits)
New     : Agent 6 — Macro Sentinel

The Macro Sentinel computes a daily Regime Flip Probability (0-1)
using three leading macro indicators:

  Signal 1 — VIX Term Structure (weight 0.30)
      ^VIX9D / ^VIX3M ratio
      > 1.00 = backwardation (immediate fear > future fear) = stress
      > 1.15 = high stress

  Signal 2 — Credit Spread Proxy (weight 0.40)
      HYG (High Yield ETF) / LQD (Investment Grade ETF) ratio
      20-day rolling return of the ratio
      Falling = credit conditions tightening = stress

  Signal 3 — Yield Curve (weight 0.30)
      10Y Treasury (^TNX) minus 2Y Treasury (^IRX)
      Rate of change of spread
      Rapidly flattening/inverting = stress

Combined flip_prob = weighted sum of normalized scores

Position sizing modification:
  flip_prob > 0.70  →  size ×0.40, tighten ATR stop, block new trades if >0.85
  flip_prob > 0.50  →  size ×0.65, slightly tighten ATR stop
  flip_prob ≤ 0.50  →  no modification (v8 behavior unchanged)

Note: Macro signals are US-centric. For NSE tickers we use
India VIX (already in v8) as the primary stress signal and
blend with global credit/yield curve at reduced weight.

Run:
    python backtest_v9.py
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
#  Config — identical to v8 except macro layer
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

MOMENTUM_TICKERS = {"NVDA","MSFT","JPM","TCS.NS","HCLTECH.NS","BAJFINANCE.NS"}
MEAN_REV_TICKERS = {"INFY.NS","HDFCBANK.NS","RELIANCE.NS","AAPL","GOOGL"}

ATR_RATIO_WARN   = 1.2
ATR_RATIO_HIGH   = 1.5
ATR_MULT_NORMAL  = 3.0
ATR_MULT_WARN    = 2.0
ATR_MULT_HIGH    = 1.5
ATR_LOOKBACK     = 60

CORR_LIMIT    = 0.70
CORR_LOOKBACK = 60
CB_SHARPE_THRESH = -0.5
CB_PAUSE_DAYS    = 20
CS_MOM_LOOKBACK  = 63
CS_FILTER_PCT    = 0.50
VIX_RULES = [(15,1.2),(20,1.0),(28,0.6),(35,0.3),(999,0.0)]

# Macro Sentinel config
MACRO_WEIGHTS = {
    "vix_term"   : 0.30,
    "credit"     : 0.40,
    "yield_curve": 0.30,
}
FLIP_PROB_MODERATE = 0.50
FLIP_PROB_HIGH     = 0.70
FLIP_PROB_BLOCK    = 0.85

# NSE blending: global macro at 50% weight (India has own dynamics)
NSE_MACRO_BLEND = 0.50


# ─────────────────────────────────────────────
#  Macro Sentinel
# ─────────────────────────────────────────────

class MacroSentinel:
    """
    Computes daily Regime Flip Probability from macro leading indicators.
    All signals normalized to 0-1 (0=calm, 1=high stress).
    """

    def __init__(self, vix9d, vix3m, hyg, lqd, tnx, irx, indiavix=None):
        self.vix9d    = vix9d
        self.vix3m    = vix3m
        self.hyg      = hyg
        self.lqd      = lqd
        self.tnx      = tnx     # 10Y yield
        self.irx      = irx     # 2Y yield (^IRX quotes as annualized %)
        self.indiavix = indiavix

        # Pre-compute daily series
        self._vix_term_series   = self._compute_vix_term()
        self._credit_series     = self._compute_credit()
        self._yield_curve_series= self._compute_yield_curve()
        self._flip_prob_series  = self._compute_flip_prob()

        print(f"  [MacroSentinel] Built flip_prob series: {len(self._flip_prob_series)} days")
        fp = self._flip_prob_series.dropna()
        if len(fp) > 0:
            print(f"  [MacroSentinel] flip_prob range: "
                  f"min={fp.min():.3f}  mean={fp.mean():.3f}  "
                  f"max={fp.max():.3f}")
            print(f"  [MacroSentinel] Days >0.50: {(fp>0.50).sum()}  "
                  f"Days >0.70: {(fp>0.70).sum()}  "
                  f"Days >0.85: {(fp>0.85).sum()}")

    def _compute_vix_term(self) -> pd.Series:
        """
        VIX9D / VIX3M ratio normalized to 0-1 stress score.
        Ratio > 1.0 = backwardation = stress.
        Score = sigmoid-like mapping centered at 1.0.
        """
        if self.vix9d is None or self.vix3m is None:
            return pd.Series(dtype=float)
        try:
            aligned = pd.concat([self.vix9d, self.vix3m], axis=1).ffill().dropna()
            aligned.columns = ["v9d","v3m"]
            ratio = aligned["v9d"] / aligned["v3m"]
            # Normalize: ratio 0.8 → score 0.0, ratio 1.2 → score 1.0
            score = (ratio - 0.80) / (1.20 - 0.80)
            return score.clip(0, 1)
        except:
            return pd.Series(dtype=float)

    def _compute_credit(self) -> pd.Series:
        """
        HYG/LQD ratio 20-day rolling return.
        Falling credit ratio = tightening = stress.
        Score = 1 when credit is deteriorating, 0 when improving.
        """
        if self.hyg is None or self.lqd is None:
            return pd.Series(dtype=float)
        try:
            aligned = pd.concat([self.hyg, self.lqd], axis=1).ffill().dropna()
            aligned.columns = ["hyg","lqd"]
            ratio   = aligned["hyg"] / aligned["lqd"]
            ret_20  = ratio.pct_change(20)
            # Normalize: -5% over 20 days → score 1.0, +5% → score 0.0
            score = (-ret_20 - (-0.05)) / (0.05 - (-0.05))
            return score.clip(0, 1)
        except:
            return pd.Series(dtype=float)

    def _compute_yield_curve(self) -> pd.Series:
        """
        10Y - 2Y spread, rate of change.
        Rapidly flattening/inverting = stress.
        Absolute inversion also scores high.
        """
        if self.tnx is None or self.irx is None:
            return pd.Series(dtype=float)
        try:
            aligned = pd.concat([self.tnx, self.irx], axis=1).ffill().dropna()
            aligned.columns = ["t10","t2"]
            # IRX is quoted as annualized discount rate, convert to yield
            t2_yield = aligned["t2"] / 100
            t10_yield = aligned["t10"] / 100
            spread  = t10_yield - t2_yield
            # Rate of change over 20 days (flattening = stress)
            spread_chg = spread.diff(20)
            # Score 1: spread inverted AND falling fast
            inversion_score = (-spread - (-0.02)) / (0.03 - (-0.02))
            flattening_score= (-spread_chg - (-0.005)) / (0.01 - (-0.005))
            score = 0.5 * inversion_score + 0.5 * flattening_score
            return score.clip(0, 1)
        except:
            return pd.Series(dtype=float)

    def _compute_flip_prob(self) -> pd.Series:
        """Weighted combination of all three signals."""
        series = []
        weights_used = {}

        if len(self._vix_term_series) > 0:
            series.append(self._vix_term_series * MACRO_WEIGHTS["vix_term"])
            weights_used["vix_term"] = MACRO_WEIGHTS["vix_term"]

        if len(self._credit_series) > 0:
            series.append(self._credit_series * MACRO_WEIGHTS["credit"])
            weights_used["credit"] = MACRO_WEIGHTS["credit"]

        if len(self._yield_curve_series) > 0:
            series.append(self._yield_curve_series * MACRO_WEIGHTS["yield_curve"])
            weights_used["yield_curve"] = MACRO_WEIGHTS["yield_curve"]

        if not series:
            return pd.Series(dtype=float)

        total_w = sum(weights_used.values())
        combined = pd.concat(series, axis=1).sum(axis=1) / total_w
        # Smooth with 5-day EMA to reduce noise
        return combined.ewm(span=5, adjust=False).mean()

    def get_flip_prob(self, date, is_nse=False) -> float:
        """
        Get flip probability for a given date.
        For NSE tickers, blend global macro with India VIX stress.
        """
        if len(self._flip_prob_series) == 0:
            return 0.0

        try:
            ts = pd.Timestamp(date)
            # Find nearest available date
            if ts in self._flip_prob_series.index:
                global_prob = float(self._flip_prob_series[ts])
            else:
                idx = self._flip_prob_series.index.get_indexer([ts], method='ffill')[0]
                if idx < 0:
                    return 0.0
                global_prob = float(self._flip_prob_series.iloc[idx])

            if np.isnan(global_prob):
                global_prob = 0.0

            # For NSE: blend with India VIX stress
            if is_nse and self.indiavix is not None:
                try:
                    if ts in self.indiavix.index:
                        india_vix = float(self.indiavix[ts])
                    else:
                        idx2 = self.indiavix.index.get_indexer([ts], method='ffill')[0]
                        india_vix = float(self.indiavix.iloc[idx2]) if idx2>=0 else 20.0

                    if not np.isnan(india_vix):
                        # India VIX stress score
                        india_stress = (india_vix - 15) / (35 - 15)
                        india_stress = float(np.clip(india_stress, 0, 1))
                        # Blend: 50% global, 50% India
                        return global_prob*(1-NSE_MACRO_BLEND) + india_stress*NSE_MACRO_BLEND
                except:
                    pass

            return global_prob

        except:
            return 0.0

    def get_size_modifier(self, flip_prob: float) -> tuple[float, float, bool]:
        """
        Returns (size_mult_modifier, atr_mult_override, block_new_trades)
        """
        if flip_prob >= FLIP_PROB_BLOCK:
            return 0.0, ATR_MULT_HIGH, True      # full block
        elif flip_prob >= FLIP_PROB_HIGH:
            return 0.40, ATR_MULT_WARN, False    # severe reduction
        elif flip_prob >= FLIP_PROB_MODERATE:
            return 0.65, ATR_MULT_NORMAL*0.85, False  # moderate reduction
        else:
            return 1.0, ATR_MULT_NORMAL, False   # no change


# ─────────────────────────────────────────────
#  v8 Core Logic (verbatim)
# ─────────────────────────────────────────────

def get_regime_params_v8(regime_label, ticker):
    is_momentum = ticker in MOMENTUM_TICKERS
    if "HIGH_VOL" in regime_label: return {"rsi":25,"adx":22,"atr":2.0}
    if "BEAR" in regime_label:     return {"rsi":28,"adx":20,"atr":2.0}
    if regime_label in ("OPTIMAL_BULL","BULL_RISK_ON"):
        return {"rsi":35 if is_momentum else 30,"adx":15,"atr":ATR_MULT_NORMAL}
    if "BULL" in regime_label:
        return {"rsi":32 if is_momentum else 30,"adx":15,"atr":2.5}
    return {"rsi":30,"adx":18,"atr":2.5}


def compute_atr(high,low,close,idx,period=14):
    if idx<period+1: return float(close.iloc[idx])*0.03
    h=high.iloc[idx-period:idx+1].values; l=low.iloc[idx-period:idx+1].values
    cp=close.iloc[idx-period-1:idx].values
    return float(np.maximum(h-l,np.maximum(abs(h-cp),abs(l-cp))).mean())


def compute_adx(high,low,close,idx,period=14):
    if idx<period*2: return 0.0
    h=high.iloc[idx-period*2:idx+1].values; l=low.iloc[idx-period*2:idx+1].values
    c=close.iloc[idx-period*2:idx+1].values
    dmp=np.where((h[1:]-h[:-1])>(l[:-1]-l[1:]),np.maximum(h[1:]-h[:-1],0),0)
    dmm=np.where((l[:-1]-l[1:])>(h[1:]-h[:-1]),np.maximum(l[:-1]-l[1:],0),0)
    tr=np.maximum(h[1:]-l[1:],np.maximum(abs(h[1:]-c[:-1]),abs(l[1:]-c[:-1])))
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


def get_atr_ratio(high,low,close,idx):
    if idx<ATR_LOOKBACK+14: return 1.0
    h=high.iloc[idx-13:idx+1].values; l=low.iloc[idx-13:idx+1].values
    cp=close.iloc[idx-14:idx].values
    tr_c=np.maximum(h-l,np.maximum(abs(h-cp),abs(l-cp)))
    atr_c=float(tr_c.mean())
    h2=high.iloc[idx-ATR_LOOKBACK-13:idx+1].values
    l2=low.iloc[idx-ATR_LOOKBACK-13:idx+1].values
    cp2=close.iloc[idx-ATR_LOOKBACK-14:idx].values
    n=min(len(h2)-1,len(l2)-1,len(cp2))
    if n<10: return 1.0
    tr_a=np.maximum(h2[1:n+1]-l2[1:n+1],
                    np.maximum(abs(h2[1:n+1]-cp2[:n]),abs(l2[1:n+1]-cp2[:n])))
    atr_a=float(tr_a.mean())
    return atr_c/atr_a if atr_a>0 else 1.0


def update_trailing_stop(s_px,curr,direc,atr,atr_ratio):
    if   atr_ratio>=ATR_RATIO_HIGH: m=ATR_MULT_HIGH
    elif atr_ratio>=ATR_RATIO_WARN: m=ATR_MULT_WARN
    else:                            m=ATR_MULT_NORMAL
    if direc=="BUY":
        return max(s_px, curr-atr*m)
    else:
        return min(s_px, curr+atr*m)


def compute_signals(df,idx,regime_label,ticker):
    rp=get_regime_params_v8(regime_label,ticker)
    rsi_t=rp["rsi"]; adx_t=rp["adx"]; atr_m=rp["atr"]
    close=df["Close"]; high=df["High"]; low=df["Low"]; volume=df.get("Volume")
    if idx<60: return "NEUTRAL",0.0,atr_m
    adx=compute_adx(high,low,close,idx)
    if adx<adx_t: return "NEUTRAL",0.0,atr_m
    window=close.iloc[:idx+1]; price=float(close.iloc[idx])
    sma20=window.rolling(20).mean().iloc[-1]; sma50=window.rolling(50).mean().iloc[-1]
    gap=(sma20-sma50)/sma50
    sma_s="BUY" if gap>0.005 else "SELL" if gap<-0.005 else "NEUTRAL"
    w=close.iloc[idx-14:idx+1].diff(); g=w.clip(lower=0).mean(); l=(-w.clip(upper=0)).mean()
    rsi=100-(100/(1+g/l)) if l!=0 else 100.0
    rsi_s="BUY" if rsi<rsi_t else "SELL" if rsi>(100-rsi_t) else "NEUTRAL"
    sb=window.rolling(20).mean().iloc[-1]; stdb=window.rolling(20).std().iloc[-1]
    bb_s="BUY" if price<sb-2*stdb else "SELL" if price>sb+2*stdb else "NEUTRAL"
    w2=close.iloc[max(0,idx-50):idx+1]
    ef=w2.ewm(span=12,adjust=False).mean(); es=w2.ewm(span=26,adjust=False).mean()
    ml=ef-es; sl2=ml.ewm(span=9,adjust=False).mean()
    macd_s="BUY" if float(ml.iloc[-1]-sl2.iloc[-1])>0 else \
           "SELL" if float(ml.iloc[-1]-sl2.iloc[-1])<0 else "NEUTRAL"
    if volume is not None and len(volume)>20:
        obv=(np.sign(close.iloc[:idx+1].diff().fillna(0))*volume.iloc[:idx+1]).cumsum()
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
    if bw>sw and bw/tw>=0.45: return "BUY",bw/tw,atr_m
    elif sw>bw and sw/tw>=0.45: return "SELL",sw/tw,atr_m
    return "NEUTRAL",0.0,atr_m


def get_regime(close,universe_df,idx):
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
        if br=="RISK_ON": return 0.85,True,True,"BULL_RISK_ON"
        return 0.70,True,True,"BULL_NEUTRAL"
    return (0.50,True,True,"NEUTRAL_CAUTIOUS") if br=="RISK_OFF" \
        else (0.75,True,True,"NEUTRAL_NORMAL")


def get_vix_mult(vix,idx):
    if vix is None or idx>=len(vix): return 1.0,20.0
    v=float(vix.iloc[idx])
    if np.isnan(v): return 1.0,20.0
    for t,m in VIX_RULES:
        if v<t: return m,v
    return 0.0,v


def get_cs_rank(ticker,universe_df,idx):
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


class PositionManager:
    def __init__(self,u): self.universe=u; self.open_pos={}
    def can_open(self,ticker,idx):
        for ot in list(self.open_pos.keys()):
            if ot==ticker: continue
            if self._corr(ticker,ot,idx)>CORR_LIMIT: return False,ot
        return True,""
    def _corr(self,t1,t2,idx):
        if idx<CORR_LOOKBACK+1: return 0.0
        if t1 not in self.universe.columns or t2 not in self.universe.columns: return 0.0
        try:
            s1=self.universe[t1].iloc[idx-CORR_LOOKBACK:idx].pct_change().dropna()
            s2=self.universe[t2].iloc[idx-CORR_LOOKBACK:idx].pct_change().dropna()
            al=pd.concat([s1,s2],axis=1).dropna()
            return float(al.iloc[:,0].corr(al.iloc[:,1])) if len(al)>=10 else 0.0
        except: return 0.0
    def open(self,t,i,d): self.open_pos[t]={"direction":d,"entry_idx":i}
    def close(self,t): self.open_pos.pop(t,None)


class CircuitBreaker:
    def __init__(self): self.pnls={}; self.paused_until={}; self.triggers={}; self.consec={}
    def record(self,ticker,pnl,idx):
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
    def paused(self,ticker,idx): return idx<self.paused_until.get(ticker,0)


@dataclass
class Trade:
    ticker:str; direction:str; entry_date:str; exit_date:str
    entry_price:float; exit_price:float; units:float; pnl:float
    pnl_pct:float; exit_reason:str; regime_label:str
    size_mult:float; flip_prob_at_entry:float; macro_blocked:bool


# ─────────────────────────────────────────────
#  Main Backtest Loop v9
# ─────────────────────────────────────────────

def backtest_v9(ticker, df, universe_df, vix_series, sentinel, cb, pos_mgr):
    trades=[]; close=df["Close"]; high=df["High"]; low=df["Low"]
    open_=df["Open"]; dates=df.index; n=len(close)
    in_trade=False; is_nse=ticker in NSE_TICKERS
    e_idx=e_px=direc=units=rlbl=smult=s_px=t_px=fp_entry=None

    for i in range(200,n):
        if not in_trade:
            if cb.paused(ticker,i): continue

            r_mult,ab,as_,r_lbl=get_regime(close,universe_df,i-1)
            if r_mult==0.0: continue

            sig,conf_v,atr_m=compute_signals(df,i-1,r_lbl,ticker)
            if sig=="NEUTRAL" or conf_v<0.45: continue
            if sig=="BUY" and not ab: continue
            if sig=="SELL" and not as_: continue

            v_mult,vix_val=get_vix_mult(vix_series,i-1)
            if v_mult==0.0: continue

            cs=get_cs_rank(ticker,universe_df,i-1)
            if sig=="BUY"  and cs<CS_FILTER_PCT: continue
            if sig=="SELL" and cs>(1-CS_FILTER_PCT): continue

            ok,_=pos_mgr.can_open(ticker,i-1)
            if not ok: continue

            # ── Macro Sentinel gate ──────────────────────────────
            current_date = dates[i-1]
            flip_prob    = sentinel.get_flip_prob(current_date, is_nse=is_nse)
            size_mod, atr_override, blocked = sentinel.get_size_modifier(flip_prob)

            if blocked:
                continue  # Macro says: don't open new positions

            # Override ATR mult with macro-adjusted value if more conservative
            atr_m = min(atr_m, atr_override) if atr_override < atr_m else atr_m
            # ────────────────────────────────────────────────────

            atr=compute_atr(high,low,close,i-1)
            sd =atr*atr_m; td=sd*2.0

            raw_e=float(open_.iloc[i])
            ep   =raw_e*(1+SLIPPAGE_PCT) if sig=="BUY" else raw_e*(1-SLIPPAGE_PCT)

            # Apply macro size modifier on top of regime+vix sizing
            combined=r_mult*v_mult*size_mod
            u=CAPITAL*RISK_PER_TRADE*combined/sd if sd>0 else 0
            if u<=0: continue

            in_trade=True; e_idx=i; e_px=ep; direc=sig; units=u
            rlbl=r_lbl; smult=combined; fp_entry=flip_prob
            s_px=ep-sd if sig=="BUY" else ep+sd
            t_px=ep+td if sig=="BUY" else ep-td
            pos_mgr.open(ticker,i,sig)

        else:
            curr=float(close.iloc[i])
            atr_now=compute_atr(high,low,close,i)
            atr_r  =get_atr_ratio(high,low,close,i)
            s_px   =update_trailing_stop(s_px,curr,direc,atr_now,atr_r)
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
                    exit_reason=xr,regime_label=rlbl,size_mult=smult,
                    flip_prob_at_entry=round(fp_entry,3),
                    macro_blocked=False,
                ))
                cb.record(ticker,pnl,i)
                pos_mgr.close(ticker)
                in_trade=False

    return trades


# ─────────────────────────────────────────────
#  Metrics + Monte Carlo
# ─────────────────────────────────────────────

def compute_metrics(trades,label=""):
    if not trades:
        return {"label":label,"trades":0,"cagr":0,"sharpe":0,
                "max_drawdown":0,"win_rate":0,"profit_factor":0,
                "total_pnl":0,"equity_curve":[CAPITAL]}
    pnls=[t.pnl for t in trades]
    winners=[t for t in trades if t.pnl>0]; losers=[t for t in trades if t.pnl<=0]
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
            "equity_curve":eq.tolist()}


def monte_carlo(trades,n_runs=MC_RUNS):
    print(f"\n[MC] {n_runs} simulations on {len(trades)} trades...")
    pnls=np.array([t.pnl for t in trades]); n=len(pnls)
    yrs=(datetime.strptime(END_DATE,"%Y-%m-%d")-
         datetime.strptime(START_DATE,"%Y-%m-%d")).days/365.25
    cagrs=[]; sharpes=[]; dds=[]; wrs=[]; eqs=[]
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
        dds.append(md*100); wrs.append(float(np.sum(s>0))/n*100)
        if run%100==0: eqs.append(eq)
    res={"cagrs":np.array(cagrs),"sharpes":np.array(sharpes),
         "max_drawdowns":np.array(dds),"equity_curves":eqs}
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

def plot_v9(all_v8, all_v9, mv8, mv9, mc, ticker_metrics, sentinel, flip_prob_series):
    BG,SURF='#05080f','#0a0f1a'
    TEXT,MUT='#e2e8f0','#6b7280'
    GOLD,GRN,RED='#f59e0b','#10b981','#ef4444'
    BLUE,PUR,CYN='#3b82f6','#8b5cf6','#06b6d4'

    fig=plt.figure(figsize=(20,24),facecolor=BG)
    gs =gridspec.GridSpec(4,3,figure=fig,hspace=0.52,wspace=0.35)

    def sax(ax,t):
        ax.set_facecolor(SURF); ax.tick_params(colors=MUT,labelsize=8)
        ax.set_title(t,color=TEXT,fontsize=10,fontweight='bold',pad=8)
        for s in ax.spines.values(): s.set_edgecolor('#1f2937')

    # ── Row 0: Flip probability time series ──
    ax0=fig.add_subplot(gs[0,:])
    if len(flip_prob_series)>0:
        fp=flip_prob_series.dropna()
        fp_plot=fp[fp.index>=START_DATE]
        ax0.fill_between(fp_plot.index, fp_plot.values, 0,
                         where=fp_plot.values>=FLIP_PROB_HIGH,
                         color=RED, alpha=0.5, label=f'High stress (>{FLIP_PROB_HIGH})')
        ax0.fill_between(fp_plot.index, fp_plot.values, 0,
                         where=(fp_plot.values>=FLIP_PROB_MODERATE)&(fp_plot.values<FLIP_PROB_HIGH),
                         color=GOLD, alpha=0.4, label=f'Moderate stress ({FLIP_PROB_MODERATE}-{FLIP_PROB_HIGH})')
        ax0.fill_between(fp_plot.index, fp_plot.values, 0,
                         where=fp_plot.values<FLIP_PROB_MODERATE,
                         color=GRN, alpha=0.2, label=f'Calm (<{FLIP_PROB_MODERATE})')
        ax0.plot(fp_plot.index, fp_plot.values, color=TEXT, lw=0.8, alpha=0.6)
        ax0.axhline(FLIP_PROB_HIGH,     color=RED,  lw=1.0, ls='--', alpha=0.7)
        ax0.axhline(FLIP_PROB_MODERATE, color=GOLD, lw=1.0, ls='--', alpha=0.7)
        ax0.axhline(FLIP_PROB_BLOCK,    color=PUR,  lw=0.8, ls=':',  alpha=0.5,
                    label=f'Block threshold ({FLIP_PROB_BLOCK})')
    ax0.set_ylim(0,1.05)
    ax0.legend(facecolor=SURF,edgecolor='#1f2937',labelcolor=TEXT,fontsize=8)
    sax(ax0,'Macro Sentinel — Daily Regime Flip Probability  (VIX Term + Credit + Yield Curve)')

    # ── Row 1: Equity curve ──
    ax1=fig.add_subplot(gs[1,:2])
    eq8=mv8["equity_curve"]; eq9=mv9["equity_curve"]
    eqs=np.array(mc["equity_curves"])
    if len(eqs)>0:
        tl=len(eq9)
        rs=[np.interp(np.linspace(0,1,tl),np.linspace(0,1,len(e)),e) for e in eqs if len(e)>1]
        if rs:
            m=np.array(rs)
            ax1.fill_between(range(tl),np.percentile(m,5,0),np.percentile(m,95,0),
                             color=GOLD,alpha=0.08,label='MC 5–95%')
            ax1.fill_between(range(tl),np.percentile(m,25,0),np.percentile(m,75,0),
                             color=GOLD,alpha=0.15,label='MC 25–75%')
    ax1.plot(range(len(eq8)),eq8,color=RED, lw=1.0,alpha=0.6,label='v8')
    ax1.plot(range(len(eq9)),eq9,color=GOLD,lw=1.5,label='v9 Macro Sentinel')
    ax1.axhline(CAPITAL,color=MUT,lw=0.5,ls='--',alpha=0.4)
    ax1.legend(facecolor=SURF,edgecolor='#1f2937',labelcolor=TEXT,fontsize=8)
    sax(ax1,'Equity Curve: v8 vs v9 with MC Fan')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f'{x/1e6:.1f}M'))

    # ── Row 1: Metrics table ──
    axm=fig.add_subplot(gs[1,2]); axm.set_facecolor(SURF); axm.axis('off')
    axm.set_title('v8 → v9 Improvement',color=TEXT,fontsize=10,fontweight='bold',pad=8)
    rows=[("CAGR",     f"{mv8['cagr']}%",        f"{mv9['cagr']}%"),
          ("Sharpe",   f"{mv8['sharpe']}",         f"{mv9['sharpe']}"),
          ("Max DD",   f"-{mv8['max_drawdown']}%", f"-{mv9['max_drawdown']}%"),
          ("Win Rate", f"{mv8['win_rate']}%",      f"{mv9['win_rate']}%"),
          ("PF",       f"{mv8['profit_factor']}",  f"{mv9['profit_factor']}"),
          ("Trades",   f"{mv8['trades']}",          f"{mv9['trades']}"),
          ("P&L",      f"{mv8['total_pnl']/1e6:.2f}M",f"{mv9['total_pnl']/1e6:.2f}M")]
    axm.text(0.35,0.96,"v8",transform=axm.transAxes,color=RED, fontsize=9,fontweight='bold',ha='center')
    axm.text(0.75,0.96,"v9",transform=axm.transAxes,color=GOLD,fontsize=9,fontweight='bold',ha='center')
    for j,(l,v8,v9) in enumerate(rows):
        y=0.84-j*0.12
        axm.text(0.02,y,l, transform=axm.transAxes,color=MUT, fontsize=9)
        axm.text(0.35,y,v8,transform=axm.transAxes,color=RED, fontsize=9,ha='center',fontweight='bold')
        axm.text(0.75,y,v9,transform=axm.transAxes,color=GOLD,fontsize=9,ha='center',fontweight='bold')
    for s in axm.spines.values(): s.set_edgecolor('#1f2937')

    # ── Row 2: flip_prob vs trade outcomes ──
    ax2=fig.add_subplot(gs[2,0])
    fp_bins=[(0,0.3,"Calm"),(0.3,0.5,"Low"),(0.5,0.7,"Moderate"),(0.7,1.0,"High")]
    bin_wr=[]; bin_apnl=[]; bin_labels=[]
    for lo,hi,lbl in fp_bins:
        bt=[t for t in all_v9 if lo<=t.flip_prob_at_entry<hi]
        if bt:
            wr=len([t for t in bt if t.pnl>0])/len(bt)*100
            ap=np.mean([t.pnl for t in bt])
            bin_wr.append(wr); bin_apnl.append(ap); bin_labels.append(f"{lbl}\n({lo}-{hi})")
        else:
            bin_wr.append(0); bin_apnl.append(0); bin_labels.append(lbl)
    x=np.arange(len(bin_labels)); w=0.35
    ax2b=ax2.twinx()
    ax2.bar(x-w/2,bin_wr,w,color=[GRN,GOLD,GOLD,RED],alpha=0.7,label='Win Rate %')
    ax2b.bar(x+w/2,bin_apnl,w,color=BLUE,alpha=0.6,label='Avg P&L')
    ax2.set_xticks(x); ax2.set_xticklabels(bin_labels,fontsize=7)
    ax2.set_ylabel('Win Rate %',color=MUT,fontsize=8)
    ax2b.set_ylabel('Avg P&L',color=MUT,fontsize=8)
    ax2.tick_params(colors=MUT); ax2b.tick_params(colors=MUT)
    ax2b.axhline(0,color=MUT,lw=0.5)
    sax(ax2,'Trade Outcome vs Flip Probability at Entry')

    # ── Row 2: MC Sharpe ──
    ax3=fig.add_subplot(gs[2,1])
    ax3.hist(mc["sharpes"],bins=60,color=PUR,alpha=0.75,edgecolor='none')
    ax3.axvline(np.percentile(mc["sharpes"],5), color=RED,lw=1.2,ls='--',label='p5')
    ax3.axvline(np.median(mc["sharpes"]),        color=GRN,lw=1.5,label='median')
    ax3.axvline(1.0,color=GOLD,lw=1.0,ls=':',label='Sharpe=1')
    ax3.axvline(2.0,color=CYN, lw=1.0,ls=':',label='Sharpe=2')
    ax3.legend(facecolor=SURF,edgecolor='#1f2937',labelcolor=TEXT,fontsize=7)
    sax(ax3,'MC Sharpe Distribution')

    # ── Row 2: Drawdown ──
    ax4=fig.add_subplot(gs[2,2])
    def dd_s(eq):
        eq=np.array(eq); pk=eq[0]; dd=[]
        for v in eq:
            if v>pk: pk=v
            dd.append((v-pk)/pk*100)
        return dd
    dd8=dd_s(mv8["equity_curve"]); dd9=dd_s(mv9["equity_curve"])
    ax4.fill_between(range(len(dd8)),dd8,0,color=RED, alpha=0.3,label='v8')
    ax4.fill_between(range(len(dd9)),dd9,0,color=GOLD,alpha=0.3,label='v9')
    ax4.plot(range(len(dd8)),dd8,color=RED, lw=0.8,alpha=0.7)
    ax4.plot(range(len(dd9)),dd9,color=GOLD,lw=1.0)
    ax4.legend(facecolor=SURF,edgecolor='#1f2937',labelcolor=TEXT,fontsize=8)
    sax(ax4,'Drawdown: v8 vs v9')

    # ── Row 3: Sharpe per ticker ──
    ax5=fig.add_subplot(gs[3,:])
    ts=[t.replace('.NS','') for t in ticker_metrics]
    s8=[ticker_metrics[t].get('sharpe_v8',0) for t in ticker_metrics]
    s9=[ticker_metrics[t].get('sharpe_v9',0) for t in ticker_metrics]
    x=np.arange(len(ts)); w=0.35
    ax5.bar(x-w/2,s8,w,color=RED, alpha=0.7,label='v8')
    ax5.bar(x+w/2,s9,w,color=GOLD,alpha=0.8,label='v9 Macro Sentinel')
    ax5.axhline(1.0,color=GRN,lw=0.8,ls='--',alpha=0.6,label='Sharpe=1')
    ax5.axhline(2.0,color=CYN,lw=0.6,ls=':' ,alpha=0.5,label='Sharpe=2')
    ax5.axhline(0,  color=MUT,lw=0.5)
    ax5.set_xticks(x); ax5.set_xticklabels(ts,fontsize=9)
    ax5.legend(facecolor=SURF,edgecolor='#1f2937',labelcolor=TEXT,fontsize=9)
    sax(ax5,'Sharpe per Ticker: v8 vs v9')

    fig.suptitle('AQRA  ·  Day 9: Macro Sentinel (Agent 6)  ·  2015–2024',
                 color=TEXT,fontsize=14,fontweight='bold',y=0.999)
    plt.savefig('backtest_v9.png',dpi=150,bbox_inches='tight',facecolor=BG,edgecolor='none')
    print("[Chart] Saved → backtest_v9.png")


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def run():
    print(f"\n{'='*65}")
    print(f"  AQRA BACKTEST v9  —  Day 9")
    print(f"  Macro Sentinel: VIX Term Structure + Credit + Yield Curve")
    print(f"{'='*65}\n")

    raw_data={}
    print("[Data] Downloading price data...")
    for ticker in ALL_TICKERS:
        try:
            df=yf.download(ticker,start=START_DATE,end=END_DATE,
                           auto_adjust=True,progress=False)
            if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(0)
            if not df.empty and len(df)>200:
                raw_data[ticker]=df; print(f"  ✓ {ticker:<15} {len(df)} days")
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

    # ── Download Macro Data ──────────────────────────────────────
    print("\n[Data] Macro Sentinel inputs...")
    macro = {}
    macro_tickers = {
        "^VIX9D" : "VIX9D (short-term fear)",
        "^VIX3M" : "VIX3M (medium-term fear)",
        "HYG"    : "High Yield Bond ETF",
        "LQD"    : "Investment Grade Bond ETF",
        "^TNX"   : "10Y Treasury Yield",
        "^IRX"   : "3M Treasury Yield (2Y proxy)",
    }
    for sym,desc in macro_tickers.items():
        try:
            d=yf.download(sym,start=START_DATE,end=END_DATE,auto_adjust=True,progress=False)
            if isinstance(d.columns,pd.MultiIndex): d.columns=d.columns.get_level_values(0)
            macro[sym]=d["Close"]
            print(f"  ✓ {sym:<10} {desc}  ({len(d)} days)")
        except Exception as e:
            print(f"  ✗ {sym:<10} {desc}  → {e}")
            macro[sym]=None

    # Build Macro Sentinel
    print("\n[MacroSentinel] Building...")
    sentinel = MacroSentinel(
        vix9d    = macro.get("^VIX9D"),
        vix3m    = macro.get("^VIX3M"),
        hyg      = macro.get("HYG"),
        lqd      = macro.get("LQD"),
        tnx      = macro.get("^TNX"),
        irx      = macro.get("^IRX"),
        indiavix = vix_in,
    )

    universe_close=pd.DataFrame({t:raw_data[t]["Close"] for t in raw_data}).ffill()

    all_v8=[]; all_v9=[]; ticker_metrics={}
    cb9=CircuitBreaker(); pm9=PositionManager(universe_close)

    print("\n[Backtest] Running v8 and v9...")
    for ticker,df in raw_data.items():
        is_nse=ticker in NSE_TICKERS
        vix_s =vix_in if is_nse else vix_us
        vix_a =vix_s.reindex(df.index).ffill() if vix_s is not None else None

        # v8 reference
        from backtest8 import backtest_v8, CircuitBreaker as CB8, PositionManager as PM8
        cb8=CB8(); pm8=PM8(universe_close)
        tv8=backtest_v8(ticker,df,universe_close,vix_a,cb8,pm8)

        # v9
        tv9=backtest_v9(ticker,df,universe_close,vix_a,sentinel,cb9,pm9)

        all_v8.extend(tv8); all_v9.extend(tv9)
        m8=compute_metrics(tv8,ticker); m9=compute_metrics(tv9,ticker)
        ticker_metrics[ticker]={
            "sharpe_v8":m8.get("sharpe",0),"sharpe_v9":m9.get("sharpe",0),
            "cagr_v8"  :m8.get("cagr",0),  "cagr_v9"  :m9.get("cagr",0),
            "maxdd_v8" :m8.get("max_drawdown",0),"maxdd_v9":m9.get("max_drawdown",0),
        }
        fp_avg=np.mean([t.flip_prob_at_entry for t in tv9]) if tv9 else 0
        print(f"  {ticker.replace('.NS',''):<12}  "
              f"v8: S={m8.get('sharpe',0):>5.2f}  │  "
              f"v9: S={m9.get('sharpe',0):>5.2f} DD={m9.get('max_drawdown',0):.1f}%  "
              f"trades={m9.get('trades',0)}  avg_flip_prob={fp_avg:.3f}")

    mv8=compute_metrics(all_v8,"v8"); mv9=compute_metrics(all_v9,"v9")

    print(f"\n{'='*65}")
    print(f"  {'METRIC':<20} {'v8':>12}  {'v9':>12}  {'CHANGE':>10}")
    print(f"  {'─'*57}")
    for lbl,k in [("CAGR","cagr"),("Sharpe","sharpe"),("Max Drawdown","max_drawdown"),
                  ("Win Rate","win_rate"),("Profit Factor","profit_factor"),
                  ("Trades","trades"),("Total P&L","total_pnl")]:
        v8v=mv8.get(k,0); v9v=mv9.get(k,0)
        if k=="max_drawdown":
            print(f"  {lbl:<20} {f'-{v8v}%':>12}  {f'-{v9v}%':>12}  {f'{v8v-v9v:+.2f}%':>10}")
        elif k=="total_pnl":
            print(f"  {lbl:<20} {f'{v8v:+,.0f}':>12}  {f'{v9v:+,.0f}':>12}  {f'{v9v-v8v:+,.0f}':>10}")
        elif k=="trades":
            print(f"  {lbl:<20} {v8v:>12}  {v9v:>12}  {f'{v9v-v8v:+d}':>10}")
        else:
            sfx="%" if k in ("cagr","win_rate") else ""
            print(f"  {lbl:<20} {f'{v8v}{sfx}':>12}  {f'{v9v}{sfx}':>12}  {f'{v9v-v8v:+.3f}':>10}")
    print(f"{'='*65}")

    mc=monte_carlo(all_v9)
    print("\n[Chart] Generating...")
    plot_v9(all_v8,all_v9,mv8,mv9,mc,ticker_metrics,
            sentinel,sentinel._flip_prob_series)
    print(f"\n[AQRA v9] ✓ Done. Open backtest_v9.png")


if __name__=="__main__":
    run()