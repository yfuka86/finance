#!/usr/bin/env python3
"""Nikkei 225 option calendar spread on the IV term-structure slope.

Frozen in docs/PREREGISTER_OPTION_CALENDAR.md.

Sell the near expiry, buy the far expiry at the same ATM strike — a net debit, so
the loss is bounded by the debit paid. That is deliberate: the straddle study in
this family died on a naked short leg (-144% of premium in one trade), and the
vertical study died because same-month skew had no structure. This uses neither
the level of IV nor same-month skew, but the slope between expiries.

Settlement prices only (no bid/ask), so a pass is PAPER ONLY by the family rule.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

Z_ENTRY, HOLD, NEAR_DTE, FAR_DTE = 0.5, 5, (7, 25), (35, 80)
MIN_OI, TICKS_COST, MIN_TRADES, MAX_TOP_SHARE = 100, 4, 30, .20
OUT = Path("data/jp_option_calendar")


def load() -> pd.DataFrame:
    d = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("data/jp_options/opt225_*.parquet"))],
                  ignore_index=True)
    d["Date"] = pd.to_datetime(d["Date"])
    for c in ("Settle", "IV", "OI", "Vo", "Strike", "dte", "UnderPx"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["is_call"] = d["PCDiv"].astype(str).eq("1")
    return d.dropna(subset=["Settle", "IV", "Strike", "dte", "UnderPx"])


def term_slope(d: pd.DataFrame) -> pd.Series:
    """Per session: ATM near IV minus ATM far IV."""
    rows = {}
    for day, g in d[d["is_call"]].groupby("Date"):
        near = g[g["dte"].between(*NEAR_DTE)]
        far = g[g["dte"].between(*FAR_DTE)]
        if near.empty or far.empty:
            continue
        n = near.loc[(near["Strike"] - near["UnderPx"]).abs().idxmin()]
        f = far.loc[(far["Strike"] - far["UnderPx"]).abs().idxmin()]
        rows[day] = n["IV"] - f["IV"]
    return pd.Series(rows).sort_index()


def trades(d: pd.DataFrame, z: pd.Series, z_entry: float, hold: int,
           is_call: bool = True) -> pd.DataFrame:
    sessions = pd.Index(sorted(d["Date"].unique()))
    side = d[d["is_call"].eq(is_call)]
    by_day = {day: g for day, g in side.groupby("Date")}
    week_seen, out = set(), []
    for day in sessions:
        wk = (day.isocalendar().year, day.isocalendar().week)
        zv = z.get(day, np.nan)
        # NaN < x は False なので `not (zv >= x)` で書く（z の無い日を素通りさせない）。
        if wk in week_seen or not (zv >= z_entry):
            continue
        i = sessions.get_loc(day)
        if i + hold >= len(sessions):
            continue
        exit_day = sessions[i + hold]
        g = by_day.get(day)
        if g is None:
            continue
        near = g[g["dte"].between(*NEAR_DTE) & g["Vo"].gt(0) & g["OI"].ge(MIN_OI)]
        far = g[g["dte"].between(*FAR_DTE) & g["Vo"].gt(0) & g["OI"].ge(MIN_OI)]
        if near.empty or far.empty:
            continue
        strike = near.loc[(near["Strike"] - near["UnderPx"]).abs().idxmin(), "Strike"]
        f_same = far[far["Strike"].eq(strike)]
        n_same = near[near["Strike"].eq(strike)]
        if f_same.empty or n_same.empty:
            continue
        n0, f0 = n_same.iloc[0], f_same.iloc[0]
        debit = f0["Settle"] - n0["Settle"]
        if debit <= 0:                      # not a debit calendar; skip
            continue
        ex = by_day.get(exit_day)
        if ex is None:
            continue
        n1 = ex[ex["Code"].eq(n0["Code"])]
        f1 = ex[ex["Code"].eq(f0["Code"])]
        if n1.empty or f1.empty:
            continue
        value = f1.iloc[0]["Settle"] - n1.iloc[0]["Settle"]
        pnl = (value - debit - TICKS_COST) / debit
        out.append({"entry": day, "exit": exit_day, "strike": strike, "z": float(zv),
                    "debit": debit, "exit_value": value, "ret": pnl,
                    "near_dte": n0["dte"], "far_dte": f0["dte"]})
        week_seen.add(wk)
    return pd.DataFrame(out)


def report(t: pd.DataFrame) -> dict:
    if t.empty:
        return {"trades": 0}
    r = t["ret"]
    pos = r[r > 0].sum()
    per_year = 52 / HOLD * HOLD / 5      # weekly cadence -> ~52 trades/yr equivalent
    sh = float(r.mean() / r.std() * np.sqrt(52)) if r.std() else None
    by = t.assign(y=t["entry"].dt.year).groupby("y")["ret"].mean()
    return {"trades": int(len(r)), "mean": round(float(r.mean()), 4),
            "median": round(float(r.median()), 4),
            "win_rate": round(float(r.gt(0).mean()), 4),
            "sharpe": round(sh, 3) if sh else None,
            "worst": round(float(r.min()), 4), "best": round(float(r.max()), 4),
            "top_trade_profit_share": round(float(r.max() / pos), 4) if pos > 0 else None,
            "negative_years": int((by < 0).sum()),
            "by_year": {int(k): round(float(v), 4) for k, v in by.items()}}


def main() -> None:
    d = load()
    slope = term_slope(d)
    z = ((slope - slope.rolling(252, min_periods=120).mean())
         / slope.rolling(252, min_periods=120).std()).dropna()
    out = {"spec": {"z_entry": Z_ENTRY, "hold": HOLD, "near_dte": NEAR_DTE,
                    "far_dte": FAR_DTE, "cost_ticks": TICKS_COST},
           "note": "settlement prices only -> PAPER ONLY even if it passes",
           "sessions_with_slope": int(len(slope)), "sessions_with_z": int(len(z))}
    main_t = trades(d, z, Z_ENTRY, HOLD)
    out["primary"] = report(main_t)
    r = out["primary"]
    failed = []
    if r.get("trades", 0) < MIN_TRADES:
        failed.append("trades_lt_30")
    if (r.get("sharpe") or -9) < 1.0:
        failed.append("sharpe_lt_1.0")
    if (r.get("top_trade_profit_share") or 1) >= MAX_TOP_SHARE:
        failed.append("top_trade_share_ge_20pct")
    if (r.get("worst") or -9) <= -1.0:
        failed.append("loss_exceeded_the_debit")
    if r.get("negative_years", 9) > 3:
        failed.append("negative_years_gt_3")
    out["failed_criteria"] = failed
    out["decision"] = "NO_GO" if failed else "PAPER_ONLY_PENDING_REAL_QUOTES"

    out["sensitivity"] = {}
    for label, kw in [("z0.0", dict(z_entry=0.0)), ("z1.0", dict(z_entry=1.0)),
                      ("hold3", dict(hold=3)), ("hold10", dict(hold=10)),
                      ("put", dict(is_call=False))]:
        kw = {"z_entry": Z_ENTRY, "hold": HOLD, **kw}
        out["sensitivity"][label] = report(trades(d, z, **kw))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    if not main_t.empty:
        main_t.to_csv(OUT / "trades.csv", index=False)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
