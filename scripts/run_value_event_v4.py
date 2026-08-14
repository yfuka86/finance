#!/usr/bin/env python3
"""V4: dividend-raise x low-PBR with the liquidity floor redefined for 60-day holds.

Frozen in docs/PREREGISTER_VALUE_EVENT_V4_LIQUIDITY.md. The ONLY change vs the
frozen V1 spec is the floor (¥1e9 -> ¥1e8, justified by participation math for a
60-session hold). Signal, model, features, holding and selection are untouched.
Primary cost 40bps round trip (small-cap auction, conservative); 20bps reported
for V1 comparability. Criterion 4 requires the never-before-judged increment
(¥1e8..1e9) to stand on its own.
"""
from __future__ import annotations
import datetime as dt, hashlib, json
from pathlib import Path
import pandas as pd
from scripts.run_value_event_v1 import load_fins
from trading.jp_intraday.daily_gap import load_existing_daily
from trading.jp_intraday.value_event_model import (
    attach_market_and_features, case_report, dividend_raise_events, fit_and_select_oos)

FLOOR, COST = 1e8, .004

def main():
    events = dividend_raise_events(load_fins())
    daily = load_existing_daily()
    cases = attach_market_and_features(events, daily, min_value_yen=FLOOR)
    common = {"floor_yen": FLOOR, "cost_round_trip": COST,
              "eligible_cases": len(cases),
              "training_cases": int((cases.exit_date < pd.Timestamp("2024-01-01")).sum()),
              "oos_start": "2024-01-01", "holding_sessions": 60}
    try:
        _, oos = fit_and_select_oos(cases, cost=COST)
        rep = case_report(oos)
        sel = oos[oos["selected"]]
        inc = sel[sel["prior_value"] < 1e9]          # V1未評価の増分のみ
        inc_med = float(inc["net_case_return"].median()) if len(inc) else None
        rep20 = case_report(fit_and_select_oos(cases, cost=.002)[1])
        failed = []
        if rep.get("cases", 0) < 30: failed.append("cases_lt_30")
        if rep.get("median", -9) <= 0: failed.append("median_40bps_not_positive")
        if rep.get("top_case_profit_share", 9) >= .20: failed.append("top_share_ge_20pct")
        if inc_med is None or inc_med <= 0: failed.append("increment_median_not_positive")
        rep.update(common | {
            "increment_cases": int(len(inc)), "increment_median": inc_med,
            "cost20bps_comparison": {k: rep20.get(k) for k in
                                     ("cases", "median", "mean", "top_case_profit_share")},
            "status": "TESTED", "failed_criteria": failed,
            "decision": "NO_GO" if failed else "PENDING_FULL_RISK_TESTS"})
    except ValueError as exc:
        rep = common | {"status": "DATA_BLOCKED", "reason": str(exc)}
    out_dir = Path("results/value_event_v4"); out_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(rep, ensure_ascii=False, indent=2, default=str)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (out_dir / f"run_{stamp}_{hashlib.sha256(payload.encode()).hexdigest()[:12]}.json"
     ).open("x", encoding="utf-8").write(payload)
    print(payload)

if __name__ == "__main__":
    main()
