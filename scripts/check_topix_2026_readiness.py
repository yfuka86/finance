#!/usr/bin/env python3
"""Fail-closed readiness audit for the TOPIX 2026 forward experiment."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from trading.jp_intraday.reference import extract_share_snapshots

ROOT=Path(__file__).resolve().parents[1]
def main():
    wpath=ROOT/"data/jp_intraday_reference/topixweight_current.csv"
    w=pd.read_csv(wpath,encoding="cp932"); weight_asof=pd.to_datetime(
        w["日付"].astype(str),format="%Y%m%d",errors="coerce").max()
    shares=extract_share_snapshots(ROOT/"data/cache"); share_asof=pd.to_datetime(shares.known_at).max()
    ffw=list((ROOT/"data/topix_2026_forward/official_ffw").glob("*")) if (ROOT/"data/topix_2026_forward/official_ffw").exists() else []
    report={"checked_at":"2026-08-01","current_weight_asof":str(weight_asof.date()),
      "share_snapshot_asof":str(share_asof.date()),"official_ffw_files":len(ffw),
      "weight_fresh_for_august":bool(weight_asof>=pd.Timestamp("2026-07-31")),
      "shares_fresh_for_august":bool(share_asof>=pd.Timestamp("2026-07-31")),
      "official_ffw_ready":bool(ffw)}
    report["prediction_ready"]=all([report["weight_fresh_for_august"],report["shares_fresh_for_august"],report["official_ffw_ready"]])
    out=ROOT/"data/topix_2026_forward/readiness_20260801.json";out.parent.mkdir(parents=True,exist_ok=True)
    if not out.exists():out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=="__main__":main()
