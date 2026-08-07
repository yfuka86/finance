#!/usr/bin/env python3
"""V4 forward-seal event ledger (append-only, no outcome computation).

Detects dividend-raise events under the FROZEN V4 spec (+20% forecast raise,
PBR<=1 on raw prior close, EPS>0, OP>0, prior-day value >= 1e8 yen) for
event_date > 2026-04-24 and appends them to
data/value_event_v4_forward/events.jsonl.

What this script never does: compute forward/exit returns, read outcomes, or
rewrite existing rows. The frozen Ridge (trained on exits < 2024-01-01, exactly
as in run_value_event_v4) scores each event at detection time -- that is what a
live desk would know, so recording it is not peeking. Judgment happens once, in
scripts/verify_v4_forward.py, on 2027-12-01+.

Rows detected after their entry day already passed carry "backfilled": true
(catch-up at seal creation; disclosed in the preregistration).
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd

from scripts.run_value_event_v1 import load_fins
from trading.jp_intraday.daily_gap import load_existing_daily
from trading.jp_intraday.value_event_model import (
    FEATURES, dividend_raise_events)

LEDGER = Path("data/value_event_v4_forward/events.jsonl")
FORWARD_START = pd.Timestamp("2026-05-01")   # first day after the fins snapshot
FLOOR = 1e8
KEY_FIELDS = ("symbol", "CurFYEn", "event_date")


def _frozen_model(fins: pd.DataFrame, daily: pd.DataFrame):
    """Refit the deterministic frozen model (same code path as V4's judgment)."""
    from trading.jp_intraday.value_event_model import attach_market_and_features
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    cases = attach_market_and_features(dividend_raise_events(fins), daily,
                                       min_value_yen=FLOOR)
    train = cases[cases["exit_date"].lt(pd.Timestamp("2024-01-01"))].dropna(
        subset=["forward_return"])
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          Ridge(alpha=10.0))
    model.fit(train[FEATURES], train["forward_return"])
    return model, len(train)


def forward_eligible(events: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Frozen V4 eligibility using only pre-disclosure data (no exits)."""
    d = daily.rename(columns={"Date": "date", "Code": "code",
                              "raw_close": "valuation_close", "Va": "value"}).copy()
    d["date"] = pd.to_datetime(d["date"])
    d["symbol"] = d["code"].astype(str).str[:4]
    d = d.sort_values(["symbol", "date"]).drop_duplicates(["symbol", "date"])
    by_symbol = {s: g.reset_index(drop=True) for s, g in d.groupby("symbol", sort=False)}
    rows = []
    for _, e in events.iterrows():
        bars = by_symbol.get(e["symbol"])
        if bars is None:
            continue
        prior_idx = bars["date"].searchsorted(e["event_date"], side="left") - 1
        if prior_idx < 0:
            continue
        prior = bars.iloc[prior_idx]
        r = e.to_dict()
        r.update({"prior_close": prior.valuation_close, "prior_value": prior.value,
                  "prior_date": prior.date})
        rows.append(r)
    x = pd.DataFrame(rows)
    if x.empty:
        return x
    x["pbr"] = x["prior_close"] / x["BPS"]
    x["book_to_price"] = x["BPS"] / x["prior_close"]
    x["earnings_yield"] = x["EPS"] / x["prior_close"]
    x["op_margin"] = x["OP"] / x["Sales"].replace(0, float("nan"))
    x["equity_ratio"] = x["Eq"] / x["TA"].replace(0, float("nan"))
    x["cash_assets"] = x["CashEq"] / x["TA"].replace(0, float("nan"))
    x["roe"] = x["NP"] / x["Eq"].replace(0, float("nan"))
    ok = (x["pbr"].gt(0) & x["pbr"].le(1) & x["EPS"].gt(0) & x["OP"].gt(0)
          & x["prior_value"].ge(FLOOR))
    return x.loc[ok].reset_index(drop=True)


def _existing_keys() -> set:
    if not LEDGER.exists():
        return set()
    keys = set()
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            keys.add((r["symbol"], r["CurFYEn"], r["event_date"][:10]))
    return keys


def main() -> None:
    fins = load_fins()
    daily = load_existing_daily()
    events = dividend_raise_events(fins)
    events = events[events["event_date"].ge(FORWARD_START)]
    eligible = forward_eligible(events, daily)
    model, n_train = _frozen_model(fins, daily)
    seen = _existing_keys()
    today = pd.Timestamp(dt.date.today())
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    appended = 0
    with LEDGER.open("a", encoding="utf-8") as fh:
        for _, e in eligible.sort_values(["event_date", "symbol"]).iterrows():
            key = (e["symbol"], e["CurFYEn"], e["event_date"].strftime("%Y-%m-%d"))
            if key in seen:
                continue
            pred = float(model.predict(pd.DataFrame([e[FEATURES]])[FEATURES])[0])
            row = {"symbol": e["symbol"], "CurFYEn": e["CurFYEn"],
                   "event_date": e["event_date"].strftime("%Y-%m-%d"),
                   "div_raise": round(float(e["div_raise"]), 6),
                   "pbr": round(float(e["pbr"]), 4),
                   "prior_close": float(e["prior_close"]),
                   "prior_value": float(e["prior_value"]),
                   "features": {k: (None if pd.isna(e[k]) else round(float(e[k]), 6))
                                for k in FEATURES},
                   "prediction": round(pred, 6),
                   "model_training_cases": n_train,
                   "backfilled": bool((today - e["event_date"]).days > 3),
                   "detected_at": dt.datetime.now(dt.timezone.utc).isoformat()}
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            appended += 1
    print(json.dumps({"forward_events_detected": int(len(eligible)),
                      "appended": appended, "ledger": str(LEDGER),
                      "training_cases": n_train}))


if __name__ == "__main__":
    main()
