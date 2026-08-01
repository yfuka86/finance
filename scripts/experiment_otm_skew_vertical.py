#!/usr/bin/env python3
"""Frozen settlement-price diagnostic for OTM put vertical relative value."""
from __future__ import annotations
import glob,json
from pathlib import Path
import numpy as np
import pandas as pd
from trading.jp_intraday.daily_model import annualized_stats

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"data/jp_otm_skew_vertical"
EVAL=pd.Timestamp("2024-01-01"); HOLD=5

def load_options():
    d=pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(str(ROOT/"data/jp_options/opt225_*.parquet")))],ignore_index=True)
    d.Date=pd.to_datetime(d.Date); d.Code=d.Code.astype(str)
    for c in ["Settle","Vo","OI","Strike","UnderPx","IV","dte"]: d[c]=pd.to_numeric(d[c],errors="coerce")
    return d.drop_duplicates(["Date","Code"]).reset_index(drop=True)

def daily_pairs(d):
    rows=[]
    for date,x in d.groupby("Date",sort=True):
        x=x[(x.PCDiv.astype(str)=="1")&x.dte.between(20,45)&x.Settle.gt(0)&x.Vo.gt(0)&x.OI.ge(100)].copy()
        if x.empty: continue
        # Nearest eligible expiry, then choose fixed moneyness legs within it.
        expiry=x.groupby(["CM","LTD"]).dte.median().sort_values().index[0]
        x=x[(x.CM==expiry[0])&(x.LTD==expiry[1])].copy().reset_index(drop=True)
        spot=float(x.UnderPx.median()); x["moneyness"]=x.Strike/spot
        near=x.loc[(x.moneyness-.95).abs().idxmin()]
        wing=x.loc[(x.moneyness-.85).abs().idxmin()]
        if near.Code==wing.Code or near.Strike<=wing.Strike: continue
        rows.append({"date":date,"near_code":near.Code,"wing_code":wing.Code,
          "near_strike":near.Strike,"wing_strike":wing.Strike,"near_px":near.Settle,
          "wing_px":wing.Settle,"skew":wing.IV-near.IV})
    p=pd.DataFrame(rows).sort_values("date")
    mu=p["skew"].rolling(252,min_periods=126).mean(); sd=p["skew"].rolling(252,min_periods=126).std()
    p["z"]=((p["skew"]-mu)/sd).shift(1)
    return p

def trades(d,p):
    sessions=pd.Index(sorted(d.Date.unique())); pos=pd.Series(range(len(sessions)),index=sessions)
    rows=[]; last_exit=-1
    for _,r in p.iterrows():
        i=int(pos[r.date])
        if i<=last_exit or i+HOLD>=len(sessions) or not np.isfinite(r.z) or abs(r.z)<.5: continue
        if i>0 and sessions[i-1].isocalendar()[:2]==r.date.isocalendar()[:2]: continue
        exit_date=sessions[i+HOLD]
        ex=d[(d.Date==exit_date)&d.Code.isin([r.near_code,r.wing_code])&d.Settle.gt(0)]
        if len(ex)!=2: continue
        prices=ex.set_index("Code").Settle
        entry=float(r.near_px-r.wing_px); exitv=float(prices[r.near_code]-prices[r.wing_code])
        width=float(r.near_strike-r.wing_strike); side=-1 if r.z>=.5 else 1
        gross=side*(exitv-entry)/width
        rows.append({"entry_date":r.date,"date":exit_date,"side":"SELL" if side<0 else "BUY",
                     "z":r.z,"width":width,"entry_spread":entry,"exit_spread":exitv,"gross":gross})
        last_exit=i+HOLD
    return pd.DataFrame(rows)

def main():
    d=load_options(); t=trades(d,daily_pairs(d)); ev=t[t.date>=EVAL].copy()
    cal=pd.date_range(EVAL,ev.date.max(),freq="B")
    daily=pd.DataFrame({"date":cal}).merge(ev[["date","gross"]],how="left").fillna({"gross":0})
    stats=annualized_stats(daily,"gross"); yearly={str(y):annualized_stats(g,"gross") for y,g in daily.groupby(daily.date.dt.year)}
    top=float(ev.gross.max()/ev.gross.sum()) if ev.gross.sum()>0 else None
    passed=(stats["sharpe"]>=1 and stats["max_drawdown"]>-.2 and all(v["ann_return"]>0 for v in yearly.values()) and top is not None and top<.2)
    summary={"spec":{"near_moneyness":.95,"wing_moneyness":.85,"z_threshold":.5,"hold_sessions":5},
             "evaluation":stats,"yearly":yearly,"trades":len(ev),"top_trade_profit_share":top,
             "decision":"PAPER ONLY" if passed else "NO-GO","execution_status":"SETTLEMENT_ONLY"}
    OUT.mkdir(parents=True,exist_ok=True); t.to_csv(OUT/"trades.csv",index=False); daily.to_csv(OUT/"daily.csv",index=False)
    (OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
