#!/usr/bin/env python3
"""Run the frozen dividend-raise x value event model once."""
from __future__ import annotations

import gzip
import hashlib
import json
import datetime as dt
from pathlib import Path

import pandas as pd

from trading.jp_intraday.daily_gap import load_existing_daily
from trading.jp_intraday.value_event_model import (
    attach_market_and_features, case_report, dividend_raise_events, fit_and_select_oos,
)


def load_fins() -> pd.DataFrame:
    records=[]
    for path in sorted(Path("data/cache").glob("fins_*.json*")):
        try:
            opener=gzip.open if path.suffix == ".gz" else open
            with opener(path,"rt",encoding="utf-8") as fh: payload=json.load(fh)
        except (OSError,json.JSONDecodeError):
            continue
        if isinstance(payload,dict):
            for value in payload.values(): records.extend(value if isinstance(value,list) else [value])
        elif isinstance(payload,list): records.extend(payload)
    f=pd.DataFrame([r for r in records if isinstance(r,dict)])
    keys=[c for c in ["Code","DiscDate","DiscTime","DiscNo"] if c in f]
    return f.drop_duplicates(keys,keep="last")


def main():
    events=dividend_raise_events(load_fins())
    cases=attach_market_and_features(events,load_existing_daily())
    training_cases=int((cases.exit_date < pd.Timestamp("2024-01-01")).sum()) if len(cases) else 0
    common={"detected_events":len(events),"eligible_cases_with_raw_price":len(cases),
            "training_cases":training_cases,"oos_start":"2024-01-01",
            "holding_sessions":60,"round_trip_cost_bps":20}
    try:
        _,oos=fit_and_select_oos(cases)
        report=case_report(oos)
        failed=[]
        if report.get("median",float("-inf")) <= 0: failed.append("median_not_positive")
        if report.get("top_case_profit_share",float("inf")) >= .20:
            failed.append("top_case_profit_share_ge_20pct")
        report.update(common | {"status":"TESTED", "decision":"NO_GO" if failed else "PENDING_FULL_RISK_TESTS",
                                "failed_criteria":failed,"oos_candidates":len(oos)})
    except ValueError as exc:
        report=common | {"status":"DATA_BLOCKED", "reason":str(exc),
                         "required_data":"raw historical close at each disclosure date"}
    out_dir=Path("results/value_event_v1")
    out_dir.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(report,ensure_ascii=False,indent=2)
    digest=hashlib.sha256(payload.encode()).hexdigest()[:12]
    stamp=dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out=out_dir/f"run_{stamp}_{digest}.json"
    # Research runs are append-only: a collision must never overwrite evidence.
    with out.open("x",encoding="utf-8") as fh:
        fh.write(payload)
    latest=Path("results/value_event_v1_20260801.json")
    if not latest.exists():
        latest.write_text(payload,encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))
    print(out)

if __name__ == "__main__": main()
