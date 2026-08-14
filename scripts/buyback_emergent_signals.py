#!/usr/bin/env python3
"""Append-only PIT signals for the buyback corporate-put hypothesis; no PnL."""
from __future__ import annotations
import argparse, hashlib, json, subprocess
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];PANEL=ROOT/"data/jp_buybacks/edinet/pressure_panel.parquet"
OUT=ROOT/"data/jp_buybacks/forward"

def candidates(panel,asof):
    x=(panel[panel.submit_at.lt(asof)].sort_values(["program_id","submit_at","doc_id"])
       .drop_duplicates(["program_id","submit_at"],keep="last").copy())
    x["realized_daily_pressure"]=x.month_shares/x.purchase_month_sessions/x.adv20_shares
    x["prev_realized_daily_pressure"]=x.groupby("program_id").realized_daily_pressure.shift()
    x["execution_acceleration_ratio"]=x.realized_daily_pressure/x.prev_realized_daily_pressure.replace(0,np.nan)
    x["average_acquisition_price"]=x.cumulative_yen/x.cumulative_shares.replace(0,np.nan)
    x["anchor_gap"]=x.prior_close/x.average_acquisition_price-1
    x["completion_ratio"]=x.cumulative_shares/x.max_shares.replace(0,np.nan)
    x=x.groupby("program_id",as_index=False).tail(1).copy()
    x["report_age_days"]=(asof-x.submit_at.dt.normalize()).dt.days
    x["calendar_days_to_end"]=(pd.to_datetime(x.period_end)-asof).dt.days
    x["estimated_adv_yen"]=x.prior_close*x.adv20_shares;x["unit_lot_yen"]=x.prior_close*100
    base=(x.report_age_days.between(0,45)&x.calendar_days_to_end.ge(14)
      &x.estimated_adv_yen.ge(1e9)&x.unit_lot_yen.le(600_000)&x.remaining_pressure.ge(.05)
      &x.completion_ratio.between(.10,.90)&x.daily_detail_consistent.fillna(False))
    x["support_anchor"]=base&x.anchor_gap.le(.02)
    x["execution_acceleration"]=base&x.execution_acceleration_ratio.ge(2)
    x["state"]="OBSERVE"
    x.loc[x.support_anchor&x.execution_acceleration,"state"]="CORPORATE_PUT_LONG"
    return x[x.support_anchor|x.execution_acceleration].copy()

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--asof",required=True);args=ap.parse_args()
    asof=pd.Timestamp(args.asof).normalize();today=pd.Timestamp.now(tz="Asia/Tokyo").tz_localize(None).normalize()
    if asof>today:raise SystemExit(f"future asof forbidden: {asof.date()} > {today.date()}")
    out=OUT/f"emergent_v1_{asof:%Y%m%d}.parquet"
    if out.exists():raise SystemExit(f"append-only: {out} exists")
    p=pd.read_parquet(PANEL);p.submit_at=pd.to_datetime(p.submit_at);sig=candidates(p,asof);sig["asof"]=asof
    OUT.mkdir(parents=True,exist_ok=True);sig.to_parquet(out,index=False)
    try:commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    except Exception:commit="unknown"
    manifest={"asof":str(asof.date()),"schema":"buyback_emergent_v1","rows":len(sig),
      "states":sig.state.value_counts().to_dict(),"primary":int(sig.state.eq("CORPORATE_PUT_LONG").sum()),
      "panel_sha256":hashlib.sha256(PANEL.read_bytes()).hexdigest(),"git_commit":commit,
      "evaluation_allowed_from":"2027-08-03","pnl_opened":False}
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(manifest,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
