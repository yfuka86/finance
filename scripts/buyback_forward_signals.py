#!/usr/bin/env python3
"""Append-only forward candidates from frozen buyback-pressure thresholds."""
from __future__ import annotations
import argparse,hashlib,json,subprocess
from pathlib import Path
import pandas as pd
from scripts.backtest_buyback_persistence import persistence_candidates

ROOT=Path(__file__).resolve().parents[1]; PANEL=ROOT/"data/jp_buybacks/edinet/pressure_panel.parquet"
OUT=ROOT/"data/jp_buybacks/forward"

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--asof",required=True);args=ap.parse_args()
    asof=pd.Timestamp(args.asof).normalize()
    today=pd.Timestamp.now(tz="Asia/Tokyo").tz_localize(None).normalize()
    if asof>today:raise SystemExit(f"future asof forbidden: {asof.date()} > {today.date()}")
    out=OUT/f"signals_v33_{asof:%Y%m%d}.parquet"
    if out.exists():raise SystemExit(f"append-only: {out} exists")
    x=pd.read_parquet(PANEL);x.submit_at=pd.to_datetime(x.submit_at);x=x[x.submit_at<asof].copy()
    x=x.sort_values("submit_at"); latest=x.groupby("program_id",as_index=False).tail(1).copy()
    latest["state"]="NONE"
    persistence=persistence_candidates(x)
    latest["estimated_adv_yen"]=latest.prior_close*latest.adv20_shares
    latest["unit_lot_yen"]=latest.prior_close*100
    latest["report_age_days"]=(asof-latest.submit_at.dt.normalize()).dt.days
    latest["calendar_days_to_end"]=(pd.to_datetime(latest.period_end)-asof).dt.days
    latest["program_current"]=(latest.report_age_days.between(0,45)
                               &latest.calendar_days_to_end.ge(14))
    latest["long_execution_eligible"]=(latest.estimated_adv_yen.ge(1e9)
                                       &latest.unit_lot_yen.le(600_000)
                                       &latest.program_current)
    long_ok=latest.doc_id.isin(persistence.doc_id)&latest.long_execution_eligible
    latest.loc[long_ok,"state"]="PERSISTENCE_LONG"
    under=(latest.remaining_pressure.ge(.20)&latest.pace_surprise.le(-.60)&latest.program_current)
    latest.loc[under,"state"]="UNDEREXECUTION_SHORT_CANDIDATE"
    x["completion_ratio"]=x.cumulative_shares/x.max_shares
    x["prior_completion_ratio"]=x.groupby("program_id").completion_ratio.shift(1)
    x["prior_pressure"]=x.groupby("program_id").remaining_pressure.shift(1)
    cliff=x[x.completion_ratio.ge(.95)&x.prior_completion_ratio.lt(.95)&x.prior_pressure.ge(.10)]
    cliff=cliff.groupby("program_id",as_index=False).tail(1)
    fresh_cliff=cliff[cliff.submit_at.dt.normalize().between(asof-pd.Timedelta(days=3),asof)]
    latest.loc[latest.program_id.isin(fresh_cliff.program_id),"state"]="COMPLETION_CLIFF_SHORT_CANDIDATE"
    sig=latest[latest.state.ne("NONE")].copy();sig["asof"]=asof
    sig["short_execution_status"]=sig.state.str.contains("SHORT").map({True:"REQUIRES_LENDING_CHECK",False:"NOT_APPLICABLE"})
    OUT.mkdir(parents=True,exist_ok=True);sig.to_parquet(out,index=False)
    try:commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    except Exception:commit="unknown"
    manifest={"asof":str(asof.date()),"signals":len(sig),"states":sig.state.value_counts().to_dict(),
      "panel_sha256":hashlib.sha256(PANEL.read_bytes()).hexdigest(),"git_commit":commit,
      "schema":"buyback_forward_v3.3","evaluation_allowed_from":"2027-08-03","pnl_opened":False}
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(manifest,ensure_ascii=False,indent=2))

if __name__=="__main__":main()
