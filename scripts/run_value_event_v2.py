#!/usr/bin/env python3
"""Run the two preregistered independent value-unlock V2 hypotheses once."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import pandas as pd

from scripts.run_value_event_v1 import load_fins
from trading.jp_intraday.daily_gap import load_existing_daily
from trading.jp_intraday.value_event_model import (
    CANCEL_FEATURES, RECOVERY_FEATURES, attach_market_and_features, case_report,
    dividend_resumption_events, fit_and_select_oos, treasury_cancellation_events,
)

OUT=Path("results/value_event_v2")


def evaluate(name,events,features):
    cases=attach_market_and_features(events,load_existing_daily())
    common={"model":name,"detected_events":len(events),"eligible_cases":len(cases),
            "training_cases":int((cases.exit_date<pd.Timestamp("2024-01-01")).sum()) if len(cases) else 0,
            "holding_sessions":60,"round_trip_cost_bps":20}
    try:
        _,oos=fit_and_select_oos(cases,features=features)
    except ValueError as exc:
        return common|{"status":"INSUFFICIENT_SAMPLE","decision":"NO_GO",
                       "failed_criteria":["insufficient_training_cases"],"reason":str(exc)}
    report=case_report(oos); selected=oos[oos.selected].copy()
    yearly=selected.groupby(selected.entry_date.dt.year).net_case_return.sum().to_dict()
    failed=[]
    if report.get("median",-1)<=0: failed.append("median_not_positive")
    if report.get("max_loss",-1)<=-.30: failed.append("max_loss_le_minus_30pct")
    if report.get("top_case_profit_share",1)>=.20: failed.append("top_case_profit_share_ge_20pct")
    if any(v<=0 for v in yearly.values()): failed.append("nonpositive_calendar_year")
    return common|report|{"status":"TESTED","decision":"NO_GO" if failed else "PENDING_BOOTSTRAP",
                          "failed_criteria":failed,"yearly_case_sum":yearly,
                          "oos_candidates":len(oos)}


def main():
    f=load_fins()
    reports=[evaluate("dividend_resumption",dividend_resumption_events(f),RECOVERY_FEATURES),
             evaluate("treasury_cancellation_proxy",treasury_cancellation_events(f),CANCEL_FEATURES)]
    payload=json.dumps({"registered_at":"2026-08-01","multiple_testing_ci":.975,
                        "models":reports},ensure_ascii=False,indent=2)
    OUT.mkdir(parents=True,exist_ok=True)
    stamp=dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    digest=hashlib.sha256(payload.encode()).hexdigest()[:12]
    path=OUT/f"run_{stamp}_{digest}.json"
    with path.open("x",encoding="utf-8") as fh: fh.write(payload)
    print(payload); print(path)


if __name__=="__main__": main()
