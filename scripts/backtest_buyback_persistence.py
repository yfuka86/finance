#!/usr/bin/env python3
"""One-shot OOS evaluation of frozen realized-buyback persistence."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from trading.jp_intraday.daily_gap import load_existing_daily
from trading.jp_intraday.daily_model import annualized_stats

ROOT=Path(__file__).resolve().parents[1]; PANEL=ROOT/"data/jp_buybacks/edinet/pressure_panel.parquet"
OUT=ROOT/"results/buyback_persistence_oos_20260801.json"; CUTOFF=pd.Timestamp("2026-01-01")

def persistence_candidates(panel):
    x=(panel.sort_values(["program_id","submit_at","doc_id"])
       .drop_duplicates(["program_id","submit_at"],keep="last").copy())
    x["realized_daily_pressure"]=x.month_shares/x.purchase_month_sessions/x.adv20_shares
    x["prev_realized_daily_pressure"]=x.groupby("program_id").realized_daily_pressure.shift()
    keep=(x.realized_daily_pressure.ge(.05)&x.prev_realized_daily_pressure.ge(.01)
          &x.purchase_day_ratio.ge(.50)&x.max_daily_concentration.le(.25)
          &x.remaining_pressure.ge(.05)&x.remaining_sessions.ge(10)
          &x.daily_detail_consistent.fillna(False))
    return x[keep].copy()

def next_session(sessions,ts):
    i=sessions.searchsorted(pd.Timestamp(ts).normalize(),side="right")
    return sessions[i] if i<len(sessions) else None

def build_trades(candidates,panel,daily):
    sessions=pd.Index(sorted(daily.date.unique())); by={s:g.set_index("date") for s,g in daily.groupby("symbol")}
    reports=panel.sort_values("submit_at"); trades=[]; used=set()
    for _,r in candidates.sort_values(["submit_at","realized_daily_pressure"],ascending=[True,False]).iterrows():
        if r.program_id in used:continue
        entry=next_session(sessions,r.submit_at)
        if entry is None or entry<CUTOFF:continue
        ei=sessions.get_loc(entry)
        if ei+19>=len(sessions):continue
        exit_date=sessions[ei+19]; exit_open=False
        u=reports[(reports.program_id==r.program_id)&(reports.submit_at>r.submit_at)]
        progress=u.cumulative_shares/u.max_shares
        stop=u[u.remaining_pressure.lt(.02)|progress.ge(.95)]
        if not stop.empty:
            early=next_session(sessions,stop.iloc[0].submit_at)
            if early is not None and early<exit_date:exit_date,exit_open=early,True
        b=by.get(r.symbol)
        if b is None or entry not in b.index or exit_date not in b.index or ei<1:continue
        prior_date=sessions[ei-1]
        if prior_date not in b.index:continue
        prior,ent,ext=b.loc[prior_date],b.loc[entry],b.loc[exit_date]
        if prior.value<1e9 or ent.raw_open<=0 or ent.raw_open*100>600_000:continue
        exit_price=ext.open if exit_open else ext.close
        if ent.open<=0 or exit_price<=0:continue
        trades.append({"symbol":r.symbol,"program_id":r.program_id,"entry_date":entry,
          "exit_date":exit_date,"exit_at_open":exit_open,"gross_case_return":exit_price/ent.open-1,
          "realized_daily_pressure":r.realized_daily_pressure})
        used.add(r.program_id)
    return pd.DataFrame(trades)

def evaluate(trades,daily,cost_bps):
    sessions=pd.Index(sorted(daily.date.unique())); cal=pd.DataFrame(index=sessions[sessions>=CUTOFF],data={"net":0.})
    by={s:g.set_index("date") for s,g in daily.groupby("symbol")}; cases=[]
    for _,t in trades.iterrows():
        ds=sessions[(sessions>=t.entry_date)&(sessions<=t.exit_date)]; px=by[t.symbol].reindex(ds)
        rr=px.close.pct_change();rr.iloc[0]=px.close.iloc[0]/px.open.iloc[0]-1
        if t.exit_at_open and len(rr)>1:rr.iloc[-1]=px.open.iloc[-1]/px.close.iloc[-2]-1
        rr.iloc[0]-=cost_bps/10000;cal.loc[ds,"net"]+=rr.values*.03
        cases.append(t.gross_case_return-cost_bps/10000)
    case=pd.Series(cases,dtype=float);stats=annualized_stats(cal.reset_index(drop=True),"net")
    profit=case[case>0].sum()
    return {"stats":stats,"trades":len(case),"median_case":float(case.median()) if len(case) else None,
      "win_rate_case":float(case.gt(0).mean()) if len(case) else None,
      "top_case_profit_share":float(case.max()/profit) if profit>0 else None}

def main():
    p=pd.read_parquet(PANEL);p.submit_at=pd.to_datetime(p.submit_at);c=persistence_candidates(p)
    d=load_existing_daily().rename(columns={"Date":"date","Code":"code","AdjO":"open","AdjC":"close","Va":"value"})
    d.date=pd.to_datetime(d.date);code=d.code.astype(str)
    # J-Quants ordinary-share codes end in 0. A fifth character such as 9 is a
    # distinct security and must not be collapsed onto the same four characters.
    d["symbol"]=code.where(~code.str.endswith("0"),code.str[:-1])
    d=d.sort_values(["symbol","date"])
    trades=build_trades(c,p,d);r20=evaluate(trades,d,20);r40=evaluate(trades,d,40);s=r20["stats"]
    enough=r20["trades"]>=20
    passed=(enough and s["sharpe"]>=1 and s["max_drawdown"]>-.10
      and r20["top_case_profit_share"] is not None and r20["top_case_profit_share"]<.20
      and r40["stats"]["total_return"]>0)
    result={"evaluation_start":str(CUTOFF.date()),"rule":"realized_execution_persistence_v2",
      "results":{"20bps":r20,"40bps":r40},
      "decision":"FORWARD CANDIDATE" if passed else ("INSUFFICIENT SAMPLE" if not enough else "NO-GO")}
    if OUT.exists():raise SystemExit(f"append-only: {OUT} exists")
    OUT.parent.mkdir(exist_ok=True);OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    trades.to_csv(OUT.with_suffix(".trades.csv"),index=False);print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
