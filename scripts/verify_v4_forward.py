#!/usr/bin/env python3
"""V4 dividend-raise FORWARD verdict -- sealed until 2027-12-01. Run ONCE.

Preregistered in docs/PREREGISTER_VALUE_EVENT_V4_FORWARD.md. Before the
judgment date this script refuses to run; there is no override flag by design
(the gotobi seal set the precedent). It recomputes everything from raw fins +
daily bars with the frozen V4 spec -- the ledger is an integrity cross-check,
not the data source.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd

JUDGMENT_DATE = dt.date(2027, 12, 1)
EVENT_WINDOW = ("2026-05-01", "2027-08-31")
FLOOR, COST = 1e8, .004
CRITERIA = "selected>=30 AND median_net_40bps>0 AND top_case_share<20%"


def main() -> None:
    if dt.date.today() < JUDGMENT_DATE:
        raise SystemExit(
            f"SEALED until {JUDGMENT_DATE}. Today is {dt.date.today()}. "
            "No override exists; opening early would unseal the forward test.")
    from scripts.run_value_event_v1 import load_fins
    from trading.jp_intraday.daily_gap import load_existing_daily
    from trading.jp_intraday.value_event_model import (
        attach_market_and_features, case_report, dividend_raise_events,
        fit_and_select_oos)
    lo, hi = map(pd.Timestamp, EVENT_WINDOW)
    cases = attach_market_and_features(
        dividend_raise_events(load_fins()), load_existing_daily(),
        min_value_yen=FLOOR)
    _, oos = fit_and_select_oos(cases, cutoff="2024-01-01", cost=COST)
    fwd = oos[oos["event_date"].between(lo, hi)].copy()
    rep = case_report(fwd.assign(selected=fwd["selected"]))
    # Secondary: pure-forward subset (events sealed before they occurred).
    ledger = Path("data/value_event_v4_forward/events.jsonl")
    backfilled_keys = set()
    if ledger.exists():
        for line in ledger.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("backfilled"):
                    backfilled_keys.add((r["symbol"], r["event_date"][:10]))
    pure = fwd[~fwd.apply(lambda r: (r["symbol"],
                                     r["event_date"].strftime("%Y-%m-%d"))
                          in backfilled_keys, axis=1)]
    rep_pure = case_report(pure)
    failed = []
    if rep.get("cases", 0) < 30:
        failed.append("cases_lt_30")
    if rep.get("median", -9) <= 0:
        failed.append("median_40bps_not_positive")
    if rep.get("top_case_profit_share", 9) >= .20:
        failed.append("top_share_ge_20pct")
    verdict = {"judged_at": dt.datetime.now(dt.timezone.utc).isoformat(),
               "event_window": EVENT_WINDOW, "floor_yen": FLOOR,
               "cost_round_trip": COST, "criteria": CRITERIA,
               "primary": rep, "pure_forward_secondary": rep_pure,
               "failed_criteria": failed,
               "decision": "NO_GO" if failed else "GO_PENDING_USER_APPROVAL"}
    out = Path("data/value_event_v4_forward/verdict.json")
    out.open("x", encoding="utf-8").write(          # "x": a verdict is written once
        json.dumps(verdict, ensure_ascii=False, indent=2, default=str))
    print(json.dumps(verdict, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
