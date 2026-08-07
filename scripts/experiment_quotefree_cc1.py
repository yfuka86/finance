#!/usr/bin/env python3
"""Plan B: quote-free ML retargeted to cc1 (close->next close), close-auction only.

Frozen in docs/PREREGISTER_quotefree_cc1.md. Single cell, no sweeps:
v2 frozen features -> Ridge(a=30) yearly WF on demeaned ret_cc_fwd ->
signal at D close -> enter D+1 close auction -> exit D+2 close auction
(1-day delay, dynamic_cc_lasso executability rules; lots sized on D close).
Costs: 0.5bps/side auction + margin interest (2.8% long / 4.2% short, /245).

The no-delay same-bar variant is reported as a DIAGNOSTIC upper bound only
(signal and entry share the close print -> not executable).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from trading.jp_intraday.daily_model import BASE_FEATURES, load_panel_cached, \
    walk_forward_predictions
from trading.jp_intraday.extra_features import EXTRA_FEATURES, attach_extra_features
from trading.jp_intraday.flow_features import FLOW_FEATURES, attach_flow_features
from trading.jp_intraday.strategies import TODAY_OPEN_COLS, unit_lot_backtest

SELECTION_END = pd.Timestamp("2024-12-31")
COST_BPS_SIDE = 0.5
RATE_LONG, RATE_SHORT, SESSIONS = .028, .042, 245
OUT = Path("data/jp_quotefree_cc1")


def stats(daily: pd.DataFrame) -> dict:
    if daily.empty or "net" not in daily:
        return {"sharpe": None}
    r = daily["net"].dropna()
    if len(r) < 50 or r.std() == 0:
        return {"sharpe": None, "days": int(len(r))}
    eq = (1 + r).cumprod()
    yearly = r.groupby(r.index.year).sum()
    top5 = r.nlargest(5).sum() / r.sum() if r.sum() > 0 else np.nan
    ex10 = r.drop(r.nlargest(10).index)
    return {"sharpe": round(float(r.mean() / r.std() * 252 ** .5), 3),
            "ann_return": round(float(r.mean() * 252), 4),
            "max_drawdown": round(float((eq / eq.cummax() - 1).min()), 4),
            "days": int(len(r)),
            "negative_years": int((yearly < 0).sum()), "years": int(len(yearly)),
            "top5_day_share": None if np.isnan(top5) else round(float(top5), 3),
            "sharpe_ex_top10": round(float(ex10.mean() / ex10.std() * 252 ** .5), 3)}


def interest_adjusted(daily: pd.DataFrame) -> pd.DataFrame:
    """Charge margin interest per held session on the executed notionals."""
    d = daily.copy()
    if d.empty:
        return d
    d = d.set_index(pd.to_datetime(d["date"]))   # unit_lot returns date as a column
    carry = (d["long_yen"] * RATE_LONG + d["short_yen"] * RATE_SHORT) / SESSIONS
    d["net_yen"] = d["net_yen"] - carry
    d["net"] = d["net_yen"] / 2e7
    return d


def build_exec_frame(panel: pd.DataFrame, preds: pd.DataFrame,
                     delay: int) -> pd.DataFrame:
    """Map D predictions onto the true next global session (delay=1) or D itself.

    Stale predictions never carry: the target execution date must be exactly
    the next session on the exchange calendar, and the symbol must trade there.
    """
    sessions = np.sort(panel["date"].unique())
    nxt = dict(zip(sessions[:-1], sessions[1:]))
    p = preds.rename(columns={"intraday_ret": "_sig_day_ret"}).copy()
    if delay == 0:
        p["exec_date"] = p["date"]
    else:
        p["exec_date"] = p["date"].map(nxt)
    p = p.dropna(subset=["exec_date"])
    # Sizing price = last close known when the order is placed (= signal-day close).
    size_px = panel[["date", "symbol", "raw_close"]].rename(
        columns={"raw_close": "size_close"})
    p = p.merge(size_px, on=["date", "symbol"], how="left")
    cols = ["date", "symbol", "ret_cc_fwd", "shortable", "short_restricted",
            "prev_value", "prev_close", "ivol"]
    ex = panel[[c for c in cols if c in panel.columns]].rename(
        columns={"date": "exec_date"})
    f = p.merge(ex, on=["exec_date", "symbol"], how="inner")
    f = f.dropna(subset=["ret_cc_fwd", "size_close"])
    # unit_lot semantics: price column raw_open sizes lots, intraday_ret is
    # harvested. Alias AFTER scoring (scores were computed on signal-day data).
    f["raw_open"] = f["size_close"]
    f["open"] = f["size_close"]
    f["intraday_ret"] = f["ret_cc_fwd"]
    f["date"] = f["exec_date"]
    return f


def run_cell(panel: pd.DataFrame, feats: list[str], delay: int,
             end: pd.Timestamp | None,
             stats_from: pd.Timestamp | None = None) -> dict:
    sub = panel if end is None else panel[panel["date"].le(end)]
    preds = walk_forward_predictions(sub, feats, alpha=30.0, target="demeaned")
    frame = build_exec_frame(sub, preds, delay=delay)
    frame = frame.assign(
        _s=frame["pred"] - frame.groupby("date")["pred"].transform("mean"))
    frame = frame[frame["_s"].notna()]
    daily, _ = unit_lot_backtest(frame, capital_yen=2e7, names_per_side=8,
                                 margin_ratio=2.0, cost_bps_side=COST_BPS_SIDE,
                                 construction="magnitude")
    daily = interest_adjusted(daily)
    if stats_from is not None and len(daily):
        daily = daily[daily.index >= stats_from]
    gross_bps = float((daily["pnl_yen"] / (daily["long_yen"] + daily["short_yen"])
                       ).mean() * 1e4) if len(daily) else None
    cost_bps = float(((daily["cost_yen"]
                       + (daily["long_yen"] * RATE_LONG
                          + daily["short_yen"] * RATE_SHORT) / SESSIONS)
                      / (daily["long_yen"] + daily["short_yen"])).mean() * 1e4) \
        if len(daily) else None
    return {"stats": stats(daily),
            "gross_bps_per_day": None if gross_bps is None else round(gross_bps, 2),
            "all_in_cost_bps_per_day": None if cost_bps is None else round(cost_bps, 2)}


def main() -> None:
    panel = attach_extra_features(attach_flow_features(
        load_panel_cached(min_value_yen=1e9)))
    sparse = [c for c in EXTRA_FEATURES if c in panel.columns]
    panel[sparse] = panel[sparse].fillna(0.0)
    # Retarget the frozen v2 features to cc1: demeaned ret_cc_fwd.
    panel["target"] = panel["ret_cc_fwd"] - panel.groupby("date")[
        "ret_cc_fwd"].transform("mean")
    feats = [f for f in [f for f in BASE_FEATURES if f not in TODAY_OPEN_COLS]
             + FLOW_FEATURES + ["prev_value"] + EXTRA_FEATURES if f in panel.columns]

    summary = {"spec": "docs/PREREGISTER_quotefree_cc1.md",
               "n_features": len(feats),
               "selection": {}, "confirmation": "UNOPENED"}
    summary["selection"]["executable_delay1"] = run_cell(
        panel, feats, delay=1, end=SELECTION_END)
    summary["selection"]["diagnostic_delay0_not_executable"] = run_cell(
        panel, feats, delay=0, end=SELECTION_END)

    s = summary["selection"]["executable_delay1"]["stats"]
    crit = {"net_sharpe_ge_1": bool((s.get("sharpe") or -9) >= 1.0),
            "neg_years_le_third": bool(s.get("years", 0) > 0 and
                                       s.get("negative_years", 9) * 3 <= s.get("years", 0)),
            "top5_share_lt_20pct": bool((s.get("top5_day_share") or 9) < .20),
            "sharpe_ex_top10_ge_05": bool((s.get("sharpe_ex_top10") or -9) >= .5),
            "gross_ge_2x_cost": bool(
                (summary["selection"]["executable_delay1"]["gross_bps_per_day"] or -9)
                >= 2 * (summary["selection"]["executable_delay1"]["all_in_cost_bps_per_day"] or 9))}
    summary["selection_criteria"] = crit
    if all(crit.values()):
        summary["confirmation"] = run_cell(
            panel, feats, delay=1, end=None,
            stats_from=pd.Timestamp("2025-01-01"))
    summary["decision"] = ("SELECTION_PASSED_SEE_CONFIRMATION"
                           if all(crit.values()) else "NO_GO_AT_SELECTION")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
