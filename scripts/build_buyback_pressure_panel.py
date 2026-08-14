#!/usr/bin/env python3
"""Build a PIT feature panel from latest EDINET report revisions; no returns."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from trading.jp_intraday.buyback_pressure import pressure_features
from trading.jp_intraday.daily_gap import load_existing_daily
from trading.jp_intraday.edinet_buyback import parse_ixbrl_zip

ROOT=Path("data/jp_buybacks/edinet"); DOCS=ROOT/"documents.jsonl"; OUT=ROOT/"pressure_panel.parquet"

def main():
    records=[json.loads(x) for x in DOCS.read_text(encoding="utf-8").splitlines() if x.strip()]
    latest={r["doc_id"]:r for r in records}; rows=[]
    for r in latest.values():
        t=r.get("terms") or {}
        if not r.get("parse_error") and "tostnet3_mentioned" not in t:
            t=parse_ixbrl_zip(Path(r["raw_file"]).read_bytes())
        if r.get("parse_error") or not r.get("sec_code") or t.get("tostnet3_mentioned"):continue
        max_shares=pd.to_numeric(t.get("max_shares"),errors="coerce")
        daily_valid=[z for z in t.get("daily_acquisitions",[]) if pd.notna(max_shares)
                     and pd.notna(pd.to_numeric(z.get("shares"),errors="coerce"))
                     and 0<=float(z["shares"])<=float(max_shares)]
        rows.append({"doc_id":r["doc_id"],"symbol":r["sec_code"],"submit_at":r["submit_datetime"],
          "report_date":t.get("report_date"),
          "board_meeting_date":t.get("board_meeting_date"),"period_start":t.get("period_start"),
          "period_end":t.get("period_end"),"max_shares":t.get("max_shares"),"max_yen":t.get("max_yen"),
          "cumulative_shares":t.get("cumulative_shares"),"cumulative_yen":t.get("cumulative_yen"),
          "month_shares":t.get("month_shares"),"month_yen":t.get("month_yen"),
          "daily_purchase_days":len(daily_valid),
          "daily_shares_sum":float(sum(float(z.get("shares",0)) for z in daily_valid)),
          "daily_yen_sum":float(sum(float(z.get("yen",0) or 0) for z in daily_valid)),
          "max_daily_shares":float(max([z.get("shares",0) for z in daily_valid],default=0)),
          "first_daily_date":min([z.get("date") for z in daily_valid],default=None),
          "last_daily_date":max([z.get("date") for z in daily_valid],default=None)})
    x=pd.DataFrame(rows); x["submit_at"]=pd.to_datetime(x.submit_at); x["event_date"]=x.submit_at.dt.normalize()
    for c in ["report_date","period_start","period_end","board_meeting_date"]:x[c]=pd.to_datetime(x[c],errors="coerce")
    daily=load_existing_daily().rename(columns={"Date":"date","Code":"code","raw_close":"close","raw_volume":"volume"})
    daily["date"]=pd.to_datetime(daily.date);code=daily.code.astype(str)
    daily["symbol"]=code.where(~code.str.endswith("0"),code.str[:-1])
    daily=daily.sort_values(["symbol","date"]);daily["adv20_shares"]=daily.groupby("symbol").volume.transform(
        lambda s:s.rolling(20,min_periods=20).mean())
    market=daily[["date","symbol","close","adv20_shares"]].dropna(subset=["close","adv20_shares"]).sort_values("date")
    x=pd.merge_asof(x.sort_values("event_date"),market,left_on="event_date",right_on="date",by="symbol",
                    direction="backward",allow_exact_matches=False).rename(columns={"close":"prior_close"})
    sessions=pd.Index(sorted(daily.date.unique())); spos=pd.Series(range(len(sessions)),index=sessions)
    def session_counts(r):
        if pd.isna(r.period_start) or pd.isna(r.period_end):return pd.Series([np.nan,np.nan,np.nan])
        report=sessions.searchsorted(r.event_date,"left")-1; start=sessions.searchsorted(r.period_start,"left"); end=sessions.searchsorted(r.period_end,"right")-1
        total=max(end-start+1,0); elapsed=max(min(report,end)-start+1,0); remaining=max(end-report,0)
        return pd.Series([total,elapsed,remaining])
    x[["total_sessions","elapsed_sessions","remaining_sessions"]]=x.apply(session_counts,axis=1)
    x=pressure_features(x)
    x["first_daily_date"]=pd.to_datetime(x.first_daily_date,errors="coerce")
    x["last_daily_date"]=pd.to_datetime(x.last_daily_date,errors="coerce")
    def purchase_month_sessions(r):
        # Denominator is the complete acquisition-report month, including days
        # with zero purchases. Using first..last purchase dates conditions on the
        # realized path and materially overstates cadence and daily pressure.
        month=r.report_date if pd.notna(r.report_date) else r.last_daily_date
        if pd.isna(month):return np.nan
        same_month=sessions[(sessions.year==month.year)&(sessions.month==month.month)]
        return int(len(same_month))
    x["purchase_month_sessions"]=x.apply(purchase_month_sessions,axis=1)
    x["purchase_day_ratio"]=x.daily_purchase_days/x.purchase_month_sessions.replace(0,np.nan)
    x["max_daily_concentration"]=x.max_daily_shares/pd.to_numeric(x.month_shares,errors="coerce").replace(0,np.nan)
    x["daily_shares_coverage"]=x.daily_shares_sum/pd.to_numeric(x.month_shares,errors="coerce").replace(0,np.nan)
    x["daily_yen_coverage"]=x.daily_yen_sum/pd.to_numeric(x.month_yen,errors="coerce").replace(0,np.nan)
    x["daily_detail_consistent"]=(x.daily_shares_coverage.between(.995,1.005)
                                  &x.daily_yen_coverage.between(.99,1.01))
    x["program_id"]=x.symbol+":"+x.board_meeting_date.dt.strftime("%Y-%m-%d")
    x=x.sort_values(["submit_at","doc_id"]).drop_duplicates(["doc_id"],keep="last")
    x.to_parquet(OUT,index=False)
    complete=x[["remaining_pressure","pace_surprise"]].notna().all(axis=1)
    summary={"documents_latest":len(latest),"market_program_reports":len(x),"complete_features":int(complete.sum()),
      "programs":int(x.loc[complete,"program_id"].nunique()),"symbols":int(x.loc[complete,"symbol"].nunique()),
      "date_min":str(x.loc[complete,"event_date"].min().date()) if complete.any() else None,
      "date_max":str(x.loc[complete,"event_date"].max().date()) if complete.any() else None,
      "remaining_pressure_quantiles":x.loc[complete,"remaining_pressure"].quantile([.01,.1,.5,.9,.99]).to_dict(),
      "pace_surprise_quantiles":x.loc[complete,"pace_surprise"].quantile([.01,.1,.5,.9,.99]).to_dict()}
    (ROOT/"pressure_panel_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=="__main__":main()
