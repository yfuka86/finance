#!/usr/bin/env python3
"""Run the frozen policy-shareholding-unwind (ownership) event model once.

Specification is fixed in docs/PREREGISTER_VALUE_EVENT_V3_OWNERSHIP.md and must
not be edited after the OOS numbers are read.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import pandas as pd

from scripts.run_value_event_v1 import load_fins
from trading.jp_intraday.daily_gap import load_existing_daily
from trading.jp_intraday.ownership_features import (
    filing_panel, load_filings, ownership_release_events,
)
from trading.jp_intraday.value_event_model import (
    OWNERSHIP_FEATURES, attach_market_and_features, case_report, fit_and_select_oos,
)

STATE_COLS = ("BPS", "EPS", "OP", "Sales", "Eq", "TA", "CashEq", "NP")
MIN_CASES, MAX_TOP_SHARE = 20, .20


def attach_fundamentals(events: pd.DataFrame, fins: pd.DataFrame) -> pd.DataFrame:
    """Join the latest financial disclosure published strictly before the filing."""
    f = fins.copy()
    f["disc_date"] = pd.to_datetime(f["DiscDate"], errors="coerce")
    f["symbol"] = f["Code"].astype(str).str[:4]
    for c in STATE_COLS:
        f[c] = pd.to_numeric(f.get(c), errors="coerce")
    f = f.dropna(subset=["disc_date", "symbol"]).sort_values(
        [c for c in ["symbol", "disc_date", "DiscTime", "DiscNo"] if c in f])
    # Quarterly rows omit balance-sheet fields; carry forward only what this
    # issuer had already disclosed. Never bfill (that would be look-ahead).
    f[list(STATE_COLS)] = f.groupby("symbol", sort=False)[list(STATE_COLS)].ffill()
    f = f.drop_duplicates(["symbol", "disc_date"], keep="last")
    # merge_asof requires both frames sorted by the join key, not by the `by` group.
    f = f.sort_values("disc_date")
    merged = pd.merge_asof(
        events.sort_values("event_date"), f[["symbol", "disc_date", *STATE_COLS]],
        left_on="event_date", right_on="disc_date", by="symbol",
        direction="backward", allow_exact_matches=False)
    return merged.dropna(subset=["BPS"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols-out", help="write price-gated candidate symbols and exit")
    args = ap.parse_args()

    panel = filing_panel(load_filings())
    events = ownership_release_events(panel)
    cases_in = attach_fundamentals(events, load_fins())
    # Price-independent preregistered gates first, so the raw-price backfill
    # never has to look at outcomes.
    cases_in = cases_in[cases_in.BPS.gt(0) & cases_in.EPS.gt(0) & cases_in.OP.gt(0)]
    if args.symbols_out:
        Path(args.symbols_out).write_text(
            "\n".join(sorted(cases_in.symbol.unique())), encoding="utf-8")
        print(f"filings={len(panel)} events={len(events)} "
              f"fundamental_gated={len(cases_in)} symbols={cases_in.symbol.nunique()}")
        return

    cases = attach_market_and_features(cases_in, load_existing_daily())
    common = {"filings": len(panel), "detected_events": len(events),
              "fundamental_gated": len(cases_in),
              "eligible_cases_with_raw_price": len(cases),
              "training_cases": int((cases.exit_date < pd.Timestamp("2024-01-01")).sum())
              if len(cases) else 0,
              "oos_start": "2024-01-01", "holding_sessions": 60,
              "round_trip_cost_bps": 20, "min_decline": .02,
              "features": OWNERSHIP_FEATURES}
    try:
        _, oos = fit_and_select_oos(cases, features=OWNERSHIP_FEATURES)
        report = case_report(oos)
        failed = []
        if report.get("cases", 0) < MIN_CASES:
            failed.append("cases_lt_20")
        if report.get("median", float("-inf")) <= 0:
            failed.append("median_not_positive")
        if report.get("top_case_profit_share", float("inf")) >= MAX_TOP_SHARE:
            failed.append("top_case_profit_share_ge_20pct")
        report.update(common | {"status": "TESTED",
                                "decision": "NO_GO" if failed else "PENDING_FULL_RISK_TESTS",
                                "failed_criteria": failed, "oos_candidates": len(oos)})
    except ValueError as exc:
        report = common | {"status": "DATA_BLOCKED", "reason": str(exc)}

    out_dir = Path("results/value_event_v3")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    digest = hashlib.sha256(payload.encode()).hexdigest()[:12]
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    # Research runs are append-only: a collision must never overwrite evidence.
    with (out_dir / f"run_{stamp}_{digest}.json").open("x", encoding="utf-8") as fh:
        fh.write(payload)
    print(payload)


if __name__ == "__main__":
    main()
