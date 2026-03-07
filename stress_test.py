"""
AQRA — Out-of-Sample Stress Test  (Day 8)
==========================================
Take v8 (the best system, zero parameter changes) and run it
on 2008-2014 — data it has NEVER seen.

This period contains:
  2008-09  Global Financial Crisis  (S&P -57%, VIX hit 89)
  2010     Flash Crash (May 6)
  2011     Euro sovereign debt crisis, US debt ceiling crisis
  2012     Greek default fears, LTRO by ECB
  2013     Taper tantrum (Bernanke hints at tapering)
  2014     Oil crash begins, Russia/Ukraine, Ebola fear

Why this matters:
  - v8 was designed and all parameters were chosen on 2015-2024
  - Running on 2008-2014 with ZERO changes = true OOS test
  - If the system makes money here, the edge is real
  - If it blows up, we learn where the gaps are

What we report:
  1. Full 2008-2014 backtest metrics vs 2015-2024
  2. Year-by-year breakdown (which years worked, which didn't)
  3. GFC-specific analysis (Sep 2008 - Mar 2009)
  4. Recovery period analysis (Mar 2009 - Dec 2010)
  5. Monte Carlo on OOS trades
  6. Combined 2008-2024 metrics (the full picture)

Run:
    python stress_test.py
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
#  Config — IDENTICAL to v8, zero changes
# ─────────────────────────────────────────────

TICKERS = {
    "NSE": ["TCS.NS","INFY.NS","RELIANCE.NS","HDFCBANK.NS","BAJFINANCE.NS","HCLTECH.NS"],
    "US" : ["AAPL","MSFT","NVDA","GOOGL","JPM"],
}
ALL_TICKERS  = TICKERS["NSE"] + TICKERS["US"]
NSE_TICKERS  = set(TICKERS["NSE"])

# Two periods
OOS_START    = "2007-01-01"   # extra year for indicator warmup
OOS_END      = "2014-12-31"
IS_START     = "2015-01-01"
IS_END       = "2024-12-31"
COMBINED_START = "2007-01-01"
COMBINED_END   = "2024-12-31"

CAPITAL        = 1_000_000
RISK_PER_TRADE = 0.02
SLIPPAGE_PCT   = 0.001
MC_RUNS        = 1000

# v8 parameters — identical, no tuning
MOMENTUM_TICKERS = {"NVDA","MSFT","JPM","TCS.NS","HCLTECH.NS","BAJFINANCE.NS"}
MEAN_REV_TICKERS = {"INFY.NS","HDFCBANK.NS","RELIANCE.NS","AAPL","GOOGL"}
ATR_RATIO_WARN   = 1.2
ATR_RATIO_HIGH   = 1.5
ATR_MULT_NORMAL  = 3.0
ATR_MULT_WARN    = 2.0
ATR_MULT_HIGH    = 1.5
ATR_LOOKBACK     = 60
CORR_LIMIT       = 0.70
CORR_LOOKBACK    = 60
CB_SHARPE_THRESH = -0.5
CB_PAUSE_DAYS    = 20
CS_MOM_LOOKBACK  = 63
CS_FILTER_PCT    = 0.50
VIX_RULES        = [(15,1.2),(20,1.0),(28,0.6),(35,0.3),(999,0.0)]

# Crisis period labels for annotation
CRISIS_PERIODS = [
    ("2008-09-15", "2009-03-09",  "GFC",          "#ef4444"),
    ("2010-05-06", "2010-07-01",  "Flash Crash",  "#f59e0b"),
    ("2011-07-01", "2011-10-01",  "Euro Crisis",  "#8b5cf6"),
    ("2013-05-22", "2013-09-01",  "Taper Tantrum","#06b6d4"),
    ("2014-06-01", "2014-12-31",  "Oil Crash",    "#10b981"),
]


# ─────────────────────────────────────────────
#  All v8 logic copied verbatim
# ─────────────────────────────────────────────

def get_regime_params_v8(regime_label, ticker):
    is_momentum = ticker in MOMENTUM_TICKERS
    if "HIGH_VOL" in regime_label:
        return {"rsi":25,"adx":22,"atr":2.0}
    if "BEAR" in regime_label:
        return {"rsi":28,"adx":20,"atr":2.0}
    if regime_label in ("OPTIMAL_BULL","BULL_RISK_ON"):
        return {"rsi":35 if is_momentum else 30,"adx":15,"atr":ATR_MULT_NORMAL}
    if "BULL" in regime_label:
        return {"rsi":32 if is_momentum else 30,"adx":15,"atr":2.5}
    return {"rsi":30,"adx":18,"atr":2.5}


def compute_atr(high, low, close, idx, period=14):
    if idx<period+1: return float(close.iloc[idx])*0.03
    h=high.iloc[idx-period:idx+1].values; l=low.iloc[idx-period:idx+1].values
    cp=close.iloc[idx-period-1:idx].values
    return float(np.maximum(h-l,np.maximum(abs(h-cp),abs(l-cp))).mean())


def compute_adx(high, low, close, idx, period=14):
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


def get_atr_ratio(high, low, close, idx):
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


def update_trailing_stop(s_px, curr, direc, atr, atr_ratio, e_px):
    if   atr_ratio>=ATR_RATIO_HIGH: m=ATR_MULT_HIGH
    elif atr_ratio>=ATR_RATIO_WARN: m=ATR_MULT_WARN
    else:                            m=ATR_MULT_NORMAL
    if direc=="BUY":
        cand=curr-atr*m; return max(s_px,cand)
    else:
        cand=curr+atr*m; return min(s_px,cand)


def compute_signals(df, idx, regime_label, ticker):
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
        ov,osv=float(obv.iloc[-1]),float(osma.iloc[-1])
        obv_s="BUY" if ov>osv*1.02 else "SELL" if ov<osv*0.98 else "NEUTRAL"
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
        if br=="RISK_ON": return 0.85,True,True,"BULL_RISK_ON"
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


class PositionManager:
    def __init__(self, u): self.universe=u; self.open_pos={}
    def can_open(self, ticker, idx):
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
    def open(self,ticker,idx,d): self.open_pos[ticker]={"direction":d,"entry_idx":idx}
    def close(self,ticker): self.open_pos.pop(ticker,None)


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
    entry_price:float; exit_price:float; units:float
    pnl:float; pnl_pct:float; exit_reason:str
    regime_label:str; size_mult:float; year:int


def run_backtest(ticker, df, universe_df, vix_series, cb, pos_mgr,
                 start_date, end_date):
    trades=[]; close=df["Close"]; high=df["High"]; low=df["Low"]
    open_=df["Open"]; dates=df.index; n=len(close)
    in_trade=False
    e_idx=e_px=direc=units=conf=rlbl=smult=vix_v=s_px=t_px=None

    # Find index range
    mask=(df.index>=start_date)&(df.index<=end_date)
    if mask.sum()==0: return []
    start_i=df.index.get_loc(df[mask].index[0])
    end_i  =df.index.get_loc(df[mask].index[-1])

    for i in range(max(200,start_i), end_i+1):
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
            if sig=="BUY" and cs<CS_FILTER_PCT: continue
            if sig=="SELL" and cs>(1-CS_FILTER_PCT): continue
            ok,_=pos_mgr.can_open(ticker,i-1)
            if not ok: continue
            atr=compute_atr(high,low,close,i-1)
            sd=atr*atr_m; td=sd*2.0
            raw_e=float(open_.iloc[i])
            ep=raw_e*(1+SLIPPAGE_PCT) if sig=="BUY" else raw_e*(1-SLIPPAGE_PCT)
            combined=r_mult*v_mult
            u=CAPITAL*RISK_PER_TRADE*combined/sd if sd>0 else 0
            if u<=0: continue
            in_trade=True; e_idx=i; e_px=ep; direc=sig; units=u
            rlbl=r_lbl; smult=combined; vix_v=vix_val
            s_px=ep-sd if sig=="BUY" else ep+sd
            t_px=ep+td if sig=="BUY" else ep-td
            pos_mgr.open(ticker,i,sig)
        else:
            curr=float(close.iloc[i])
            # Trailing stop
            atr_now=compute_atr(high,low,close,i)
            atr_r  =get_atr_ratio(high,low,close,i)
            s_px   =update_trailing_stop(s_px,curr,direc,atr_now,atr_r,e_px)
            hs=curr<=s_px if direc=="BUY" else curr>=s_px
            ht=curr>=t_px if direc=="BUY" else curr<=t_px
            pfn=(lambda x:(x-e_px)*units) if direc=="BUY" else (lambda x:(e_px-x)*units)
            xp=xr=None
            if   hs: xp,xr=s_px,"STOP_LOSS"
            elif ht: xp,xr=t_px,"TAKE_PROFIT"
            elif i>=end_i: xp,xr=curr,"END_OF_PERIOD"
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
                    year=dates[e_idx].year,
                ))
                cb.record(ticker,pnl,i)
                pos_mgr.close(ticker)
                in_trade=False
    return trades


# ─────────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────────

def compute_metrics(trades, label="", start=OOS_START, end=OOS_END):
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
    yrs=(datetime.strptime(end,"%Y-%m-%d")-datetime.strptime(start,"%Y-%m-%d")).days/365.25
    fn=CAPITAL+tp; cagr=(fn/CAPITAL)**(1/yrs)-1 if yrs>0 and fn>0 else 0
    return {"label":label,"trades":len(trades),"win_rate":round(wr,1),
            "total_pnl":round(tp,2),"cagr":round(cagr*100,2),"sharpe":round(sh,3),
            "max_drawdown":round(md*100,2),"profit_factor":round(pf,3),
            "equity_curve":eq.tolist()}


def year_by_year(trades, start_year=2008, end_year=2014):
    results={}
    for yr in range(start_year, end_year+1):
        yr_trades=[t for t in trades if t.year==yr]
        if not yr_trades:
            results[yr]={"trades":0,"pnl":0,"win_rate":0,"sharpe":0}
            continue
        pnls=[t.pnl for t in yr_trades]
        wr=len([p for p in pnls if p>0])/len(pnls)*100
        eq=np.array([CAPITAL+sum(pnls[:i]) for i in range(len(pnls)+1)])
        r=np.diff(eq)/eq[:-1]; r=r[np.isfinite(r)&(r!=0)]
        sh=(np.mean(r)/np.std(r))*np.sqrt(252) if len(r)>1 and np.std(r)>0 else 0
        results[yr]={"trades":len(yr_trades),"pnl":round(sum(pnls),2),
                     "win_rate":round(wr,1),"sharpe":round(sh,3)}
    return results


def monte_carlo(trades, start, end, n_runs=MC_RUNS):
    if not trades: return {}
    print(f"\n[MC] {n_runs} simulations on {len(trades)} trades ({start[:4]}–{end[:4]})...")
    pnls=np.array([t.pnl for t in trades]); n=len(pnls)
    yrs=(datetime.strptime(end,"%Y-%m-%d")-datetime.strptime(start,"%Y-%m-%d")).days/365.25
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
    return {"cagrs":np.array(cagrs),"sharpes":np.array(sharpes),
            "max_drawdowns":np.array(dds),"equity_curves":eqs}


# ─────────────────────────────────────────────
#  Charts
# ─────────────────────────────────────────────

def plot_stress(oos_trades, is_trades, m_oos, m_is, mc_oos, mc_is,
                yr_breakdown, ticker_oos, ticker_is):
    BG,SURF='#05080f','#0a0f1a'
    TEXT,MUT='#e2e8f0','#6b7280'
    GOLD,GRN,RED='#f59e0b','#10b981','#ef4444'
    BLUE,PUR,CYN='#3b82f6','#8b5cf6','#06b6d4'

    fig=plt.figure(figsize=(20,26),facecolor=BG)
    gs =gridspec.GridSpec(5,3,figure=fig,hspace=0.50,wspace=0.35)

    def sax(ax,t):
        ax.set_facecolor(SURF); ax.tick_params(colors=MUT,labelsize=8)
        ax.set_title(t,color=TEXT,fontsize=10,fontweight='bold',pad=8)
        for s in ax.spines.values(): s.set_edgecolor('#1f2937')

    # ── Row 0: OOS equity + MC fan ──
    ax0=fig.add_subplot(gs[0,:2])
    eq_oos=m_oos["equity_curve"]
    if mc_oos and "equity_curves" in mc_oos:
        eqs=np.array(mc_oos["equity_curves"])
        if len(eqs)>0:
            tl=len(eq_oos)
            rs=[np.interp(np.linspace(0,1,tl),np.linspace(0,1,len(e)),e)
                for e in eqs if len(e)>1]
            if rs:
                m=np.array(rs)
                ax0.fill_between(range(tl),np.percentile(m,5,0),np.percentile(m,95,0),
                                 color=RED,alpha=0.08,label='MC OOS 5-95%')
                ax0.fill_between(range(tl),np.percentile(m,25,0),np.percentile(m,75,0),
                                 color=RED,alpha=0.15,label='MC OOS 25-75%')
    ax0.plot(range(len(eq_oos)),eq_oos,color=RED,lw=1.5,label='OOS 2008–2014')
    ax0.axhline(CAPITAL,color=MUT,lw=0.5,ls='--',alpha=0.4)
    # Annotate crisis periods
    trade_dates=[t.entry_date for t in oos_trades]
    if trade_dates:
        first_date=pd.Timestamp(min(trade_dates))
        for cs,ce,clbl,ccol in CRISIS_PERIODS:
            try:
                cs_t=pd.Timestamp(cs); ce_t=pd.Timestamp(ce)
                # Estimate x position from trade index
                n_trades=len(oos_trades)
                ax0.axvspan(0,min(n_trades,50),alpha=0.0)  # placeholder
            except: pass
    ax0.legend(facecolor=SURF,edgecolor='#1f2937',labelcolor=TEXT,fontsize=8)
    sax(ax0,'OOS Equity Curve: 2008–2014  (System NEVER saw this data)')
    ax0.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f'{x/1e6:.2f}M'))

    # ── Row 0: OOS vs IS comparison table ──
    axm=fig.add_subplot(gs[0,2]); axm.set_facecolor(SURF); axm.axis('off')
    axm.set_title('OOS vs In-Sample',color=TEXT,fontsize=10,fontweight='bold',pad=8)
    rows=[("Period",      "2008–2014",       "2015–2024"),
          ("CAGR",        f"{m_oos['cagr']}%",     f"{m_is['cagr']}%"),
          ("Sharpe",      f"{m_oos['sharpe']}",      f"{m_is['sharpe']}"),
          ("Max DD",      f"-{m_oos['max_drawdown']}%",f"-{m_is['max_drawdown']}%"),
          ("Win Rate",    f"{m_oos['win_rate']}%",   f"{m_is['win_rate']}%"),
          ("Prof Factor", f"{m_oos['profit_factor']}",f"{m_is['profit_factor']}"),
          ("Trades",      f"{m_oos['trades']}",       f"{m_is['trades']}"),
          ("Total P&L",   f"{m_oos['total_pnl']/1e6:.2f}M",f"{m_is['total_pnl']/1e6:.2f}M")]
    axm.text(0.40,0.96,"OOS",transform=axm.transAxes,color=RED, fontsize=9,fontweight='bold',ha='center')
    axm.text(0.80,0.96,"IS" ,transform=axm.transAxes,color=GRN, fontsize=9,fontweight='bold',ha='center')
    for j,(l,oos,ins) in enumerate(rows):
        y=0.86-j*0.105
        axm.text(0.02,y,l,  transform=axm.transAxes,color=MUT,fontsize=8)
        axm.text(0.40,y,oos,transform=axm.transAxes,color=RED,fontsize=9,ha='center',fontweight='bold')
        axm.text(0.80,y,ins,transform=axm.transAxes,color=GRN,fontsize=9,ha='center',fontweight='bold')
    for s in axm.spines.values(): s.set_edgecolor('#1f2937')

    # ── Row 1: Year-by-year breakdown ──
    ax1=fig.add_subplot(gs[1,:2])
    years=sorted(yr_breakdown.keys())
    yr_pnls=[yr_breakdown[y]["pnl"] for y in years]
    yr_sharpes=[yr_breakdown[y]["sharpe"] for y in years]
    yr_wr=[yr_breakdown[y]["win_rate"] for y in years]
    colors_y=[GRN if p>0 else RED for p in yr_pnls]
    x=np.arange(len(years))
    ax1b=ax1.twinx()
    bars=ax1.bar(x,yr_pnls,color=colors_y,alpha=0.7,label='Annual P&L')
    ax1b.plot(x,yr_sharpes,color=GOLD,lw=2,marker='o',ms=6,label='Annual Sharpe')
    ax1b.axhline(0,color=MUT,lw=0.5,ls='--')
    ax1b.axhline(1,color=GRN,lw=0.8,ls=':',alpha=0.5)
    ax1.set_xticks(x); ax1.set_xticklabels([str(y) for y in years],fontsize=10)
    ax1.set_ylabel('Annual P&L ($)',color=MUT,fontsize=8)
    ax1b.set_ylabel('Annual Sharpe',color=GOLD,fontsize=8)
    ax1.tick_params(colors=MUT); ax1b.tick_params(colors=MUT)
    # Add trade count labels
    for i,y in enumerate(years):
        ax1.text(i,yr_pnls[i]+(5000 if yr_pnls[i]>=0 else -15000),
                 f"n={yr_breakdown[y]['trades']}\nWR={yr_breakdown[y]['win_rate']:.0f}%",
                 ha='center',fontsize=7,color=TEXT)
    ax1.legend(facecolor=SURF,edgecolor='#1f2937',labelcolor=TEXT,fontsize=8,loc='upper left')
    ax1b.legend(facecolor=SURF,edgecolor='#1f2937',labelcolor=TEXT,fontsize=8,loc='upper right')
    sax(ax1,'Year-by-Year OOS Performance  (2008–2014)')

    # ── Row 1: Regime distribution OOS ──
    ax2=fig.add_subplot(gs[1,2])
    regime_counts={}
    for t in oos_trades: regime_counts[t.regime_label]=regime_counts.get(t.regime_label,0)+1
    top8=sorted(regime_counts.items(),key=lambda x:-x[1])[:8]
    if top8:
        rnames=[x[0].replace('_','\n') for x in top8]
        rvals=[x[1] for x in top8]
        ax2.barh(range(len(rnames)),rvals,color=BLUE,alpha=0.8)
        ax2.set_yticks(range(len(rnames)))
        ax2.set_yticklabels(rnames,fontsize=7)
    sax(ax2,'OOS Regime Distribution')

    # ── Row 2: MC OOS vs MC IS distributions ──
    ax3=fig.add_subplot(gs[2,0])
    if mc_oos and "cagrs" in mc_oos:
        ax3.hist(mc_oos["cagrs"],bins=50,color=RED, alpha=0.6,label='OOS 2008-14',density=True)
    if mc_is and "cagrs" in mc_is:
        ax3.hist(mc_is["cagrs"], bins=50,color=GRN, alpha=0.6,label='IS 2015-24', density=True)
    ax3.axvline(0,color=MUT,lw=0.8)
    ax3.legend(facecolor=SURF,edgecolor='#1f2937',labelcolor=TEXT,fontsize=8)
    sax(ax3,'MC CAGR: OOS vs IS Distributions')
    ax3.set_xlabel('CAGR %',color=MUT,fontsize=8)

    ax4=fig.add_subplot(gs[2,1])
    if mc_oos and "sharpes" in mc_oos:
        ax4.hist(mc_oos["sharpes"],bins=50,color=RED, alpha=0.6,label='OOS',density=True)
    if mc_is and "sharpes" in mc_is:
        ax4.hist(mc_is["sharpes"], bins=50,color=GRN, alpha=0.6,label='IS',  density=True)
    ax4.axvline(1.0,color=GOLD,lw=1.0,ls=':',label='Sharpe=1')
    ax4.legend(facecolor=SURF,edgecolor='#1f2937',labelcolor=TEXT,fontsize=8)
    sax(ax4,'MC Sharpe: OOS vs IS')

    ax5=fig.add_subplot(gs[2,2])
    if mc_oos and "max_drawdowns" in mc_oos:
        ax5.hist(mc_oos["max_drawdowns"],bins=50,color=RED, alpha=0.6,label='OOS',density=True)
    if mc_is and "max_drawdowns" in mc_is:
        ax5.hist(mc_is["max_drawdowns"], bins=50,color=GRN, alpha=0.6,label='IS',  density=True)
    ax5.axvline(20,color=GOLD,lw=1.0,ls=':',label='20% line')
    ax5.legend(facecolor=SURF,edgecolor='#1f2937',labelcolor=TEXT,fontsize=8)
    sax(ax5,'MC Max Drawdown: OOS vs IS')

    # ── Row 3: Combined equity curve 2008-2024 ──
    ax6=fig.add_subplot(gs[3,:2])
    eq_combined=np.array(m_oos["equity_curve"].copy())
    oos_end_val =float(eq_combined[-1])
    # Chain IS equity curve starting from OOS end value
    eq_is_arr=np.array(m_is["equity_curve"])
    scale=oos_end_val/CAPITAL
    eq_is_scaled=eq_is_arr*scale
    eq_full=np.concatenate([eq_combined, eq_is_scaled[1:]])
    oos_len=len(eq_combined)
    ax6.plot(range(oos_len),eq_combined,color=RED, lw=1.5,label='OOS 2008–2014')
    ax6.plot(range(oos_len-1,len(eq_full)),eq_full[oos_len-1:],
             color=GRN,lw=1.5,label='IS 2015–2024')
    ax6.axvline(oos_len,color=GOLD,lw=1.5,ls='--',alpha=0.7,label='OOS/IS boundary')
    ax6.axhline(CAPITAL,color=MUT,lw=0.5,ls='--',alpha=0.4)
    ax6.legend(facecolor=SURF,edgecolor='#1f2937',labelcolor=TEXT,fontsize=8)
    sax(ax6,'Combined Equity Curve: 2008–2024  (red=OOS, green=IS)')
    ax6.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f'{x/1e6:.2f}M'))

    # ── Row 3: GFC specific ──
    ax7=fig.add_subplot(gs[3,2])
    gfc_trades=[t for t in oos_trades
                if "2008-09-01"<=t.entry_date<="2009-03-31"]
    gfc_regime={}
    for t in gfc_trades: gfc_regime[t.regime_label]=gfc_regime.get(t.regime_label,0)+1
    if gfc_regime:
        ax7.pie(gfc_regime.values(),labels=[k.replace('_','\n') for k in gfc_regime.keys()],
                colors=[RED,GOLD,GRN,BLUE,PUR][:len(gfc_regime)],autopct='%1.0f%%',
                textprops={'color':MUT,'fontsize':7})
    gfc_pnl=sum(t.pnl for t in gfc_trades)
    gfc_wr =len([t for t in gfc_trades if t.pnl>0])/max(len(gfc_trades),1)*100
    ax7.set_title(f'GFC Period (Sep08–Mar09)\n'
                  f'Trades={len(gfc_trades)}  P&L=${gfc_pnl:+,.0f}  WR={gfc_wr:.0f}%',
                  color=TEXT,fontsize=9,fontweight='bold',pad=8)
    ax7.set_facecolor(SURF)
    for s in ax7.spines.values(): s.set_edgecolor('#1f2937')

    # ── Row 4: Per-ticker OOS vs IS Sharpe ──
    ax8=fig.add_subplot(gs[4,:])
    ts=[t.replace('.NS','') for t in ticker_oos]
    s_oos=[ticker_oos[t].get('sharpe',0) for t in ticker_oos]
    s_is =[ticker_is.get(t,{}).get('sharpe',0) for t in ticker_oos]
    x=np.arange(len(ts)); w=0.35
    ax8.bar(x-w/2,s_oos,w,color=RED, alpha=0.7,label='OOS 2008–2014')
    ax8.bar(x+w/2,s_is, w,color=GRN, alpha=0.8,label='IS 2015–2024')
    ax8.axhline(1.0,color=GOLD,lw=0.8,ls='--',alpha=0.6,label='Sharpe=1')
    ax8.axhline(0,  color=MUT, lw=0.5)
    ax8.set_xticks(x); ax8.set_xticklabels(ts,fontsize=9)
    ax8.legend(facecolor=SURF,edgecolor='#1f2937',labelcolor=TEXT,fontsize=9)
    sax(ax8,'Per-Ticker Sharpe: OOS 2008–2014 vs IS 2015–2024')

    fig.suptitle('AQRA  ·  Day 8: Out-of-Sample Stress Test  ·  2008–2014 vs 2015–2024',
                 color=TEXT,fontsize=14,fontweight='bold',y=0.999)
    plt.savefig('stress_test.png',dpi=150,bbox_inches='tight',facecolor=BG,edgecolor='none')
    print("[Chart] Saved → stress_test.png")


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def run():
    print(f"\n{'='*65}")
    print(f"  AQRA STRESS TEST  —  Day 8")
    print(f"  Running v8 (zero changes) on 2008–2014")
    print(f"  This data was NEVER used in any design decision")
    print(f"{'='*65}\n")

    raw_data={}
    print("[Data] Downloading 2007–2024...")
    for ticker in ALL_TICKERS:
        try:
            df=yf.download(ticker,start="2007-01-01",end=IS_END,
                           auto_adjust=True,progress=False)
            if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(0)
            if not df.empty and len(df)>200:
                raw_data[ticker]=df
                print(f"  ✓ {ticker:<15} {len(df)} days")
        except Exception as e: print(f"  ✗ {ticker}: {e}")

    vix_us=vix_in=None
    print("\n[Data] VIX...")
    for sym,name in [("^VIX","US"),("^INDIAVIX","NSE")]:
        try:
            d=yf.download(sym,start="2007-01-01",end=IS_END,
                          auto_adjust=True,progress=False)
            if isinstance(d.columns,pd.MultiIndex): d.columns=d.columns.get_level_values(0)
            v=d["Close"]
            if name=="US": vix_us=v
            else:          vix_in=v
            print(f"  ✓ {sym} {len(v)} days")
        except: print(f"  ✗ {sym}")
    if vix_in is None: vix_in=vix_us

    universe_close=pd.DataFrame({t:raw_data[t]["Close"] for t in raw_data}).ffill()

    # ── OOS: 2008-2014 ──
    print(f"\n[OOS] Running on 2008–2014 (never-seen data)...")
    cb_oos=CircuitBreaker(); pm_oos=PositionManager(universe_close)
    all_oos=[]; ticker_oos_metrics={}

    for ticker,df in raw_data.items():
        is_nse=ticker in NSE_TICKERS
        vix_s =vix_in if is_nse else vix_us
        vix_a =vix_s.reindex(df.index).ffill() if vix_s is not None else None
        t=run_backtest(ticker,df,universe_close,vix_a,cb_oos,pm_oos,
                       "2008-01-01","2014-12-31")
        all_oos.extend(t)
        m=compute_metrics(t,ticker,start="2008-01-01",end="2014-12-31")
        ticker_oos_metrics[ticker]={"sharpe":m.get("sharpe",0),"cagr":m.get("cagr",0),
                                    "maxdd":m.get("max_drawdown",0)}
        print(f"  {ticker.replace('.NS',''):<12}  "
              f"Sharpe={m.get('sharpe',0):>5.2f}  "
              f"CAGR={m.get('cagr',0):>+6.1f}%  "
              f"MaxDD={m.get('max_drawdown',0):>5.1f}%  "
              f"Trades={m.get('trades',0)}")

    m_oos=compute_metrics(all_oos,"OOS",start="2008-01-01",end="2014-12-31")
    yr_bk=year_by_year(all_oos,2008,2014)

    print(f"\n  ── OOS Summary ──")
    print(f"  CAGR         : {m_oos['cagr']}%")
    print(f"  Sharpe       : {m_oos['sharpe']}")
    print(f"  Max Drawdown : -{m_oos['max_drawdown']}%")
    print(f"  Win Rate     : {m_oos['win_rate']}%")
    print(f"  Trades       : {m_oos['trades']}")
    print(f"  Total P&L    : ${m_oos['total_pnl']:+,.0f}")

    print(f"\n  ── Year by Year ──")
    for yr,d in yr_bk.items():
        bar="█"*max(0,int(d['pnl']/10000)) if d['pnl']>0 else "▓"*max(0,int(-d['pnl']/10000))
        col="+" if d['pnl']>=0 else "-"
        print(f"  {yr}  P&L={col}${abs(d['pnl']):>9,.0f}  "
              f"Sharpe={d['sharpe']:>5.2f}  WR={d['win_rate']:>4.1f}%  "
              f"n={d['trades']:>3}  {bar}")

    # ── IS: 2015-2024 (v8 reference) ──
    print(f"\n[IS] Running on 2015–2024 (in-sample reference)...")
    cb_is=CircuitBreaker(); pm_is=PositionManager(universe_close)
    all_is=[]; ticker_is_metrics={}

    for ticker,df in raw_data.items():
        is_nse=ticker in NSE_TICKERS
        vix_s =vix_in if is_nse else vix_us
        vix_a =vix_s.reindex(df.index).ffill() if vix_s is not None else None
        t=run_backtest(ticker,df,universe_close,vix_a,cb_is,pm_is,IS_START,IS_END)
        all_is.extend(t)
        m=compute_metrics(t,ticker,start=IS_START,end=IS_END)
        ticker_is_metrics[ticker]={"sharpe":m.get("sharpe",0),"cagr":m.get("cagr",0)}

    m_is=compute_metrics(all_is,"IS",start=IS_START,end=IS_END)

    # GFC analysis
    gfc_trades=[t for t in all_oos if "2008-09-01"<=t.entry_date<="2009-03-31"]
    gfc_pnl=sum(t.pnl for t in gfc_trades)
    print(f"\n  ── GFC Analysis (Sep 2008 – Mar 2009) ──")
    print(f"  Trades during GFC : {len(gfc_trades)}")
    print(f"  P&L during GFC    : ${gfc_pnl:+,.0f}")
    print(f"  Win Rate          : "
          f"{len([t for t in gfc_trades if t.pnl>0])/max(len(gfc_trades),1)*100:.1f}%")
    print(f"  Regime breakdown  :")
    rc={}
    for t in gfc_trades: rc[t.regime_label]=rc.get(t.regime_label,0)+1
    for r,c in sorted(rc.items(),key=lambda x:-x[1]):
        print(f"    {r:<25} {c} trades")

    # Monte Carlo both periods
    mc_oos=monte_carlo(all_oos,"2008-01-01","2014-12-31")
    mc_is =monte_carlo(all_is, IS_START, IS_END)

    # Print MC comparison
    print(f"\n{'='*65}")
    print(f"  MONTE CARLO COMPARISON")
    print(f"{'='*65}")
    print(f"  {'Metric':<20} {'OOS 2008-14':>14}  {'IS 2015-24':>14}")
    print(f"  {'─'*50}")
    if mc_oos and mc_is:
        for lbl,k in [("CAGR median","cagrs"),("Sharpe median","sharpes"),
                      ("MaxDD median","max_drawdowns")]:
            oos_v=np.median(mc_oos[k]); is_v=np.median(mc_is[k])
            print(f"  {lbl:<20} {oos_v:>14.2f}  {is_v:>14.2f}")
        print(f"  {'P(CAGR>0%)':<20} "
              f"{np.mean(mc_oos['cagrs']>0)*100:>13.1f}%  "
              f"{np.mean(mc_is['cagrs']>0)*100:>13.1f}%")
        print(f"  {'P(Sharpe>1)':<20} "
              f"{np.mean(mc_oos['sharpes']>1)*100:>13.1f}%  "
              f"{np.mean(mc_is['sharpes']>1)*100:>13.1f}%")
    print(f"{'='*65}")

    print("\n[Chart] Generating...")
    plot_stress(all_oos,all_is,m_oos,m_is,mc_oos,mc_is,
                yr_bk,ticker_oos_metrics,ticker_is_metrics)
    print(f"\n[AQRA Stress Test] ✓ Done. Open stress_test.png")


if __name__=="__main__":
    run()
