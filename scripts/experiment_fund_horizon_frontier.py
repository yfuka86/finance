#!/usr/bin/env python3
"""Holding-horizon frontier for PIT fundamentals x flow (all close auctions).

Frozen in docs/PREREGISTER_FUND_HORIZON_FRONTIER.md. Twelve cells
(signal x horizon), h-tranche overlapped books, measured daily turnover
costed at 0.5bps/side, short notional pays 4.2%/245 per day. Selection
2018-2024; a passing cell must then survive the Y20M unit-lot form before
the (partially consumed, disclosed) 2025+ confirmation opens once.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experiment_flow_fund_sector_on import build_panel, zsec
from trading.jp_intraday.strategies import unit_lot_backtest

SELECTION_END = pd.Timestamp("2024-12-31")
Q = 0.10
COST_SIDE = 0.5e-4
SHORT_RATE, SESSIONS = .042, 245
OUT = Path("data/jp_fund_horizon")

CELLS = [("V", h) for h in (5, 20, 60)] + [("Q", h) for h in (5, 20, 60)] \
    + [("VQ", h) for h in (5, 20, 60)] + [("F", 1), ("F", 5), ("FVQ", 5)]


def signal(p: pd.DataFrame, name: str) -> pd.Series:
    if name == "V":
        return p["z_bp"]
    if name == "Q":
        return p["z_roe"]
    if name == "VQ":
        return .5 * p["z_bp"] + .5 * p["z_roe"]
    if name == "F":
        return p["z_flow"]
    if name == "FVQ":
        return .5 * p["z_flow"] + .25 * p["z_bp"] + .25 * p["z_roe"]
    raise ValueError(name)


def battery(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 100 or r.std() == 0:
        return {"sharpe": None, "days": int(len(r))}
    eq = (1 + r).cumprod()
    yearly = r.groupby(r.index.year).sum()
    top5 = float(r.nlargest(5).sum() / r.sum()) if r.sum() > 0 else None
    ex10 = r.drop(r.nlargest(10).index)
    return {"sharpe": round(float(r.mean() / r.std() * 252 ** .5), 3),
            "ann_return_pct": round(float(r.mean() * 252 * 100), 2),
            "max_drawdown_pct": round(float((eq / eq.cummax() - 1).min() * 100), 2),
            "negative_years": int((yearly < 0).sum()), "years": int(len(yearly)),
            "top5_day_share": None if top5 is None else round(top5, 3),
            "sharpe_ex_top10": round(float(ex10.mean() / ex10.std() * 252 ** .5), 3),
            "days": int(len(r))}


def judge(s: dict) -> dict:
    return {"net_sharpe_ge_1": bool((s.get("sharpe") or -9) >= 1.0),
            "neg_years_le_third": bool(s.get("years", 0) > 0
                                       and s.get("negative_years", 9) * 3 <= s["years"]),
            "top5_share_lt_20pct": bool((s.get("top5_day_share") or 9) < .20),
            "sharpe_ex_top10_ge_05": bool((s.get("sharpe_ex_top10") or -9) >= .5),
            "gross_ge_2x_cost": bool((s.get("gross_bps_per_day") or -9)
                                     >= 2 * (s.get("all_in_cost_bps_per_day") or 9))}


def run_cell(p: pd.DataFrame, sig_name: str, h: int,
             end: pd.Timestamp | None, start: pd.Timestamp | None = None) -> dict:
    sub = p if end is None else p[p["date"].le(end)]
    if start is not None:
        sub = sub[sub["date"].ge(start)]
    f = sub[["date", "symbol", "sector", "shortable", "short_restricted",
             "z_bp", "z_roe", "z_flow", "ret_cc_fwd"]].copy()
    f["score"] = signal(f, sig_name)
    f = f.dropna(subset=["score"])
    # daily long/short books: top/bottom decile of the sector-z score
    hi = f.groupby("date")["score"].transform("quantile", 1 - Q)
    lo = f.groupby("date")["score"].transform("quantile", Q)
    f["side"] = 0.0
    f.loc[f["score"].ge(hi), "side"] = 1.0
    short_ok = f["shortable"].fillna(False) & ~f["short_restricted"].fillna(False)
    f.loc[f["score"].le(lo) & short_ok, "side"] = -1.0
    b = f[f["side"] != 0].copy()
    n_side = b.groupby(["date", "side"])["symbol"].transform("count")
    b["w"] = b["side"] * .5 / n_side
    W = b.pivot_table(index="date", columns="symbol", values="w",
                      aggfunc="sum").fillna(0.0)
    # book formed at s is live (earning cc) on sessions s+2 .. s+1+h
    W_eff = sum(W.shift(k) for k in range(2, h + 2)) / h
    CC = f.pivot_table(index="date", columns="symbol", values="ret_cc_fwd",
                       aggfunc="last").reindex(index=W_eff.index,
                                               columns=W_eff.columns)
    live = W_eff.fillna(0.0) * CC.notna()
    gross = (live * CC.fillna(0.0)).sum(axis=1)
    turnover = W_eff.fillna(0.0).diff().abs().sum(axis=1)
    short_notional = live.clip(upper=0).abs().sum(axis=1)
    cost = turnover * COST_SIDE + short_notional * SHORT_RATE / SESSIONS
    net = (gross - cost).iloc[h + 2:]
    s = battery(net)
    dep = live.abs().sum(axis=1).iloc[h + 2:]
    s["gross_bps_per_day"] = round(float((gross.iloc[h + 2:] / dep.replace(0, np.nan)
                                          ).mean() * 1e4), 2)
    s["all_in_cost_bps_per_day"] = round(float((cost.iloc[h + 2:]
                                                / dep.replace(0, np.nan)).mean() * 1e4), 2)
    s["turnover_per_day"] = round(float(turnover.iloc[h + 2:].mean()), 3)
    s["criteria"] = judge(s)
    return s


def unit_lot_check(p: pd.DataFrame, sig_name: str, h: int) -> dict:
    """Concentrated Y20M unit-lot form of the winning cell (magnitude, 8/side).

    Approximation for h>1: hold-to-horizon compounded cc returns per tranche
    are not expressible in the daily unit-lot engine; we validate the h=1-style
    daily-rebalanced magnitude book on the same score with amortized costs.
    """
    sub = p[p["date"].le(SELECTION_END)].copy()
    sub["score"] = signal(sub, sig_name)
    f = sub.dropna(subset=["score", "ret_cc_fwd"]).copy()
    f["_s"] = f["score"] - f.groupby("date")["score"].transform("mean")
    f["raw_open"] = f["raw_close"]
    f["open"] = f["raw_close"]
    f["intraday_ret"] = f["ret_cc_fwd"]
    daily, _ = unit_lot_backtest(f, capital_yen=2e7, names_per_side=8,
                                 margin_ratio=2.0, cost_bps_side=0.5 / h,
                                 construction="magnitude")
    if not len(daily):
        return {"sharpe": None}
    carry = daily["short_yen"] * SHORT_RATE / SESSIONS
    r = ((daily["net_yen"] - carry) / 2e7)
    r.index = pd.to_datetime(daily["date"])
    return battery(r)


def main() -> None:
    p = build_panel()
    summary = {"spec": "docs/PREREGISTER_FUND_HORIZON_FRONTIER.md",
               "selection": {}, "confirmation": "UNOPENED"}
    passing = []
    for sig_name, h in CELLS:
        key = f"{sig_name}_h{h}"
        s = run_cell(p, sig_name, h, SELECTION_END)
        summary["selection"][key] = s
        if all(s["criteria"].values()):
            passing.append((sig_name, h))
    summary["passing_cells"] = [f"{s}_h{h}" for s, h in passing]
    if passing:
        order = {"V": 0, "Q": 1, "F": 2, "VQ": 3, "FVQ": 4}
        sig_name, h = sorted(passing, key=lambda x: (order[x[0]], x[1]))[0]
        summary["frozen_cell"] = f"{sig_name}_h{h}"
        ul = unit_lot_check(p, sig_name, h)
        summary["unit_lot_check"] = ul
        if (ul.get("sharpe") or -9) >= 1.0:
            conf = run_cell(p, sig_name, h, None, pd.Timestamp("2025-01-01"))
            summary["confirmation"] = {summary["frozen_cell"]: conf}
            summary["decision"] = ("GO_PENDING_USER_APPROVAL"
                                   if all(conf["criteria"].values())
                                   else "NO_GO_AT_CONFIRMATION")
        else:
            summary["decision"] = "NO_GO_AT_UNIT_LOT"
    else:
        summary["decision"] = "NO_GO_AT_SELECTION"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
