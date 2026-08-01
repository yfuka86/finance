#!/usr/bin/env python3
"""One-shot OOS check of frozen long-only buyback execution states."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from trading.jp_intraday.daily_gap import load_existing_daily
from trading.jp_intraday.daily_model import annualized_stats

ROOT=Path(__file__).resolve().parents[1]; PANEL=ROOT/"data/jp_buybacks/edinet/pressure_panel.parquet"
OUT=ROOT/"results/buyback_steady_oos_20260801.json"; CUTOFF=pd.Timestamp("2026-03-01")

def candidates(panel):
    x=panel.copy()
    x["steady"]=(x.remaining_pressure.ge(.10)&x.pace_surprise.ge(0)&x.purchase_day_ratio.ge(.50)
      &x.max_daily_concentration.le(.25)&x.remaining_sessions.ge(10))
    x["acceleration"]=x.remaining_pressure.ge(.20)&x.pace_surprise.ge(.55)
    return x[x.steady|x.acceleration].copy()

def run_variant(panel,daily,variant,cost_bps=20):
    x=panel[panel[variant]].sort_values(["submit_at","remaining_pressure"],ascending=[True,False]).copy()
    sessions=pd.Index(sorted(daily.date.unique())); by={s:g.set_index("date") for s,g in daily.groupby("symbol")}
    active=[]; trades=[]
    for _,r in x.iterrows():
        entry_i=sessions.searchsorted(pd.Timestamp(r.submit_at).normalize(),side="right"); exit_i=entry_i+19
        if exit_i>=len(sessions):continue
        entry_date,exit_date=sessions[entry_i],sessions[exit_i]
        # The historical confirmation was preregistered for entries from March 2026.
        if entry_date<CUTOFF:continue
        active=[z for z in active if z[0]>=entry_date]
        if len(active)>=10 or any(z[1]==r.symbol for z in active):continue
        b=by.get(r.symbol)
        if b is None or entry_date not in b.index or exit_date not in b.index:continue
        prior_i=sessions.searchsorted(entry_date)-1
        if prior_i<0 or sessions[prior_i] not in b.index:continue
        prior=b.loc[sessions[prior_i]]; ent=b.loc[entry_date]; ext=b.loc[exit_date]
        if prior.value<1e9 or ent.raw_open<=0 or ent.raw_open*100>600_000:continue
        ret=ext.close/ent.open-1-cost_bps/10000
        trades.append({"symbol":r.symbol,"program_id":r.program_id,"entry_date":entry_date,
          "exit_date":exit_date,"case_return":ret,"remaining_pressure":r.remaining_pressure})
        active.append((exit_date,r.symbol))
    t=pd.DataFrame(trades)
    cal=pd.DataFrame({"date":sessions[sessions>=CUTOFF]}); cal["net"]=0.0
    if not t.empty:
        for _,z in t.iterrows():
            b=by[z.symbol]; ds=sessions[(sessions>=z.entry_date)&(sessions<=z.exit_date)]
            px=b.reindex(ds)
            if px[["open","close"]].isna().any().any():continue
            # Enter at the first session's open, then mark at each close.
            rr=px.close.pct_change()
            rr.iloc[0]=px.close.iloc[0]/px.open.iloc[0]-1
            rr=rr*.03
            cal_idx=cal.set_index("date").index
            cal.loc[cal.date.isin(ds),"net"]+=rr.reindex(cal_idx.intersection(ds)).values
        # Charge the full round-trip assumption on entry for NAV-return accounting.
        for d,n in t.groupby("entry_date").size().items():cal.loc[cal.date.eq(d),"net"]-=n*.03*cost_bps/10000
    stats=annualized_stats(cal,"net"); total=t.case_return.sum() if not t.empty else 0
    top=(t.case_return.max()/total) if total>0 else None
    return {"stats":stats,"trades":len(t),"median_case":float(t.case_return.median()) if len(t) else None,
      "win_rate_case":float(t.case_return.gt(0).mean()) if len(t) else None,"top_case_profit_share":top},t

def main():
    p=pd.read_parquet(PANEL);p.submit_at=pd.to_datetime(p.submit_at);p=candidates(p)
    d=load_existing_daily().rename(columns={"Date":"date","Code":"code","AdjO":"open","AdjC":"close","Va":"value"})
    d.date=pd.to_datetime(d.date);d["symbol"]=d.code.astype(str).str[:4];d=d.sort_values(["symbol","date"])
    results={}
    all_trades=[]
    for name,mask in [("steady","steady"),("acceleration","acceleration")]:
        results[name],tr=run_variant(p,d,mask)
        if not tr.empty:tr["variant"]=name;all_trades.append(tr)
    p["union"]=p.steady|p.acceleration;results["union"],trades=run_variant(p,d,"union")
    if not trades.empty:trades["variant"]="union";all_trades.append(trades)
    results["steady_40bps"],tr40=run_variant(p,d,"steady",cost_bps=40)
    if not tr40.empty:tr40["variant"]="steady_40bps";all_trades.append(tr40)
    s=results["steady"]; st=s["stats"]
    robust=results["steady_40bps"]["stats"]["total_return"]>0
    passed=(s["trades"]>=20 and st["sharpe"]>=1 and st["max_drawdown"]>-.10
            and s["top_case_profit_share"] is not None and s["top_case_profit_share"]<.20 and robust)
    result={"evaluation_start":"2026-03-01","primary":"steady","cost_bps_roundtrip":20,
      "results":results,"decision":"FORWARD CANDIDATE" if passed else "NO-GO"}
    if OUT.exists():raise SystemExit(f"append-only: {OUT} exists")
    OUT.parent.mkdir(exist_ok=True);OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    pd.concat(all_trades,ignore_index=True).to_csv(OUT.with_suffix(".trades.csv"),index=False)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=="__main__":main()
