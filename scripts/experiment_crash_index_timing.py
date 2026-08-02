#!/usr/bin/env python3
"""Post-crash index timing on 1306 (TOPIX ETF), judged against buy-and-hold.

Frozen in docs/PREREGISTER_CRASH_INDEX_TIMING.md. The dip-buy decomposition showed
the rebound is real but lives in the market, not in stock selection, so this drops
selection entirely and only times exposure. Long-only cash ETF: no borrow wall, no
futures/equity tax split.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from trading.jp_intraday.daily_gap import load_existing_daily
from trading.jp_intraday.daily_model import load_panel_cached

ETF, TRIGGER, HOLD = "13060", -0.030, 10
CAPITAL, COST_ROUND_TRIP = 2e7, 0.0020
OUT = Path("data/jp_crash_index_timing")


def stats(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 50 or r.std() == 0:
        return {"sharpe": None}
    eq = (1 + r).cumprod()
    return {"sharpe": round(float(r.mean() / r.std() * 252 ** .5), 3),
            "ann_return": round(float(r.mean() * 252), 4),
            "ann_vol": round(float(r.std() * 252 ** .5), 4),
            "max_drawdown": round(float((eq / eq.cummax() - 1).min()), 4),
            "total_return": round(float(eq.iloc[-1] - 1), 4), "days": int(len(r))}


def run(etf: pd.DataFrame, mkt: pd.Series, trigger: float, hold: int) -> tuple[pd.Series, dict]:
    """Binary in/out. A fresh trigger while held extends the exit, never adds size."""
    dates = etf.index
    triggers = {d for d in dates if mkt.get(d, 0.0) <= trigger}
    held_until, entry_i, ret, events = -1, None, pd.Series(0.0, index=dates), 0
    for i, d in enumerate(dates):
        if i - 1 >= 0 and dates[i - 1] in triggers:
            if entry_i is None:                     # enter at this session's open
                entry_i, events = i, events + 1
                held_until = i + hold
            else:
                held_until = max(held_until, i + hold)
        if entry_i is None:
            continue
        # open->close on the entry session, then close->close, exit at held_until close
        if i == entry_i:
            ret.iloc[i] = etf["close"].iloc[i] / etf["open"].iloc[i] - 1 - COST_ROUND_TRIP
        elif i <= held_until:
            ret.iloc[i] = etf["close"].iloc[i] / etf["close"].iloc[i - 1] - 1
        if i >= held_until:
            entry_i, held_until = None, -1
    return ret, {"events": events, "days_in_market": int((ret != 0).sum()),
                 "time_in_market": round(float((ret != 0).mean()), 4)}


def main() -> None:
    daily = load_existing_daily()
    e = daily[daily["Code"].astype(str).eq(ETF)].copy()
    e["Date"] = pd.to_datetime(e["Date"])
    etf = e.set_index("Date").sort_index()[["AdjO", "AdjC", "raw_close"]].rename(
        columns={"AdjO": "open", "AdjC": "close"})
    panel = load_panel_cached(min_value_yen=1e9)
    mkt = panel.groupby("date")["ret"].mean()
    etf = etf[etf.index.isin(mkt.index)]

    bh = etf["close"].pct_change()
    strat, meta = run(etf, mkt, TRIGGER, HOLD)
    out = {"spec": {"etf": ETF, "trigger": TRIGGER, "hold": HOLD,
                    "cost_round_trip": COST_ROUND_TRIP, "capital": CAPITAL},
           "buy_and_hold": stats(bh), "strategy": stats(strat) | meta,
           "lot_impact": round(float(etf["raw_close"].max() * 100 / CAPITAL), 5),
           "sensitivity": {}}

    s, b = out["strategy"], out["buy_and_hold"]
    failed = []
    if (s.get("sharpe") or -9) < 1.0:
        failed.append("sharpe_lt_1.0")
    if (s.get("sharpe") or -9) <= (b.get("sharpe") or 9):
        failed.append("sharpe_not_above_buy_and_hold")
    if (s.get("max_drawdown") or -9) <= (b.get("max_drawdown") or 0):
        failed.append("drawdown_not_shallower_than_buy_and_hold")
    out["failed_criteria"] = failed
    out["decision"] = "NO_GO" if failed else "PENDING_FULL_RISK_TESTS"

    for label, t, h in [("trigger_-4pct", -.04, HOLD), ("trigger_-5pct", -.05, HOLD),
                        ("hold_5", TRIGGER, 5), ("hold_20", TRIGGER, 20)]:
        r, m = run(etf, mkt, t, h)
        out["sensitivity"][label] = stats(r) | m
    yr = strat.groupby(strat.index.year).apply(lambda x: float((1 + x).prod() - 1))
    ybh = bh.groupby(bh.index.year).apply(lambda x: float((1 + x).prod() - 1))
    out["by_year"] = {int(k): {"strategy": round(yr[k], 4), "buy_and_hold": round(ybh[k], 4)}
                      for k in yr.index}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
