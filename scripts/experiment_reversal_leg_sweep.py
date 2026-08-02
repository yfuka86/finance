#!/usr/bin/env python3
"""Quote-free cross-sectional reversal, swept by leg and by borrow feasibility.

Everything that died in this repo died on one of two walls: the alpha sat in
shorts that cannot be borrowed, or the long leg was pure market beta. So this
does not report a single headline Sharpe. For every reversal cell it reports:

  long_excess  : long leg minus the equal-weight market (survives the borrow wall)
  short_free   : short leg with no borrow constraint    (the theoretical number)
  short_borrow : short leg restricted to 貸借 and unrestricted names (tradable)
  ls_borrow    : both legs, short side borrow-constrained

A cell only matters if `long_excess` clears the bar, because that is the only
column that is executable at ¥20M without borrowing what does not exist.

Signals are quote-free: formation uses closes through the prior session, so the
book can be sent before the open. Formation is computed on the **unfiltered**
history and merged in, never shifted inside the liquidity-filtered panel.

Window is the 2021-2024 selection window, same as the momentum reproduction.
Any surviving cell needs a single confirmation run on 2025+ and must not be
promoted on the strength of this sweep alone.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from trading.jp_intraday.daily_gap import load_existing_daily
from trading.jp_intraday.daily_model import load_panel_cached

FORMATIONS = (1, 5, 20, 60)       # sessions of prior return that we fade
HOLDINGS = (1, 5, 20)
RESIDUALS = ("raw", "sector")
WINDOW, QUANTILE, COST_BPS = ("2021-01-01", "2024-12-31"), .10, 1.0
OUT = Path("data/jp_reversal_leg_sweep")


def formation_frame() -> pd.DataFrame:
    """Prior-return formations on the full history, guarded against stale stitching."""
    d = load_existing_daily().rename(columns={"Date": "date", "Code": "symbol"})
    d["date"] = pd.to_datetime(d["date"])
    d = d.dropna(subset=["AdjC"]).sort_values(["symbol", "date"])
    d = d[d["AdjC"] > 0].drop_duplicates(["symbol", "date"])
    logc, g = np.log(d["AdjC"]), d.groupby("symbol", sort=False)
    out = d[["date", "symbol"]].copy()
    for length in FORMATIONS:
        near = logc.groupby(d["symbol"], sort=False).shift(1)
        far = logc.groupby(d["symbol"], sort=False).shift(1 + length)
        span = (d["date"] - g["date"].shift(1 + length)).dt.days
        out[f"form{length}"] = (near - far).where(span.between(1 + length, (1 + length) * 2.2))
    return out


def sharpe(r: pd.Series) -> float | None:
    r = r.dropna()
    if len(r) < 50 or r.std() == 0:
        return None
    return round(float(r.mean() / r.std() * 252 ** .5), 3)


def held(weights: pd.DataFrame, hold: int) -> pd.DataFrame:
    return sum(weights.shift(k) for k in range(1, hold + 1)) / hold


def main() -> None:
    p = load_panel_cached(min_value_yen=1e9)
    p = p.merge(formation_frame(), on=["date", "symbol"], how="left")
    p = p[p["date"].between(*WINDOW)].copy()
    p["borrowable"] = p["shortable"].fillna(True) & ~p["short_restricted"].fillna(False)
    mkt = p.groupby("date")["ret"].mean()

    results = {}
    for length in FORMATIONS:
        col = f"form{length}"
        for resid in RESIDUALS:
            f = p.dropna(subset=[col, "ret"]).copy()
            score = -f[col]                       # fade the prior move
            if resid == "sector":
                score = score - score.groupby([f["date"], f["s33_code"]]).transform("mean")
            f["_s"] = score
            rank = f.groupby("date")["_s"].rank(pct=True)
            longs, shorts = rank.ge(1 - QUANTILE), rank.le(QUANTILE)
            ret = f.pivot_table(index="date", columns="symbol", values="ret", aggfunc="last")

            def book(mask, sign):
                w = pd.Series(0.0, index=f.index)
                w[mask] = float(sign)
                w = w / w.abs().groupby(f["date"]).transform("sum").replace(0, np.nan)
                return f.assign(w=w).pivot_table(index="date", columns="symbol",
                                                 values="w", aggfunc="last").fillna(0.0)

            wl, ws = book(longs, +1), book(shorts, -1)
            ws_b = book(shorts & f["borrowable"], -1)
            for hold in HOLDINGS:
                cost = 2.0 * COST_BPS / 1e4 / hold        # full book replaced every `hold`
                hl, hs, hb = held(wl, hold), held(ws, hold), held(ws_b, hold)
                # Only score sessions where the book is actually on (the first
                # `hold` rows have no tranche yet and would otherwise read as 0%).
                active = hl.abs().sum(axis=1).gt(1e-9)
                lr = ((hl * ret).sum(axis=1) - cost).where(active)
                sr = ((hs * ret).sum(axis=1) - cost).where(active)
                sb = ((hb * ret).sum(axis=1) - cost).where(active)
                mkt_leg = mkt.reindex(lr.index).where(active)
                ls = (((hl + hb) / 2 * ret).sum(axis=1) - cost).where(active)
                results[f"form{length}/{resid}/hold{hold}"] = {
                    "long_excess": sharpe(lr - mkt_leg),
                    "long_raw": sharpe(lr),
                    "short_free": sharpe(sr),
                    "short_borrow": sharpe(sb),
                    "ls_borrow": sharpe(ls),
                }

    best = max((v["long_excess"] for v in results.values() if v["long_excess"] is not None),
               default=None)
    out = {"window": WINDOW, "quantile": QUANTILE, "cost_bps_side": COST_BPS,
           "cells": results, "best_long_excess": best,
           "note": "long_excess is the only executable column; any positive cell "
                   "needs one confirmation run on 2025+ before it means anything"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    hdr = f"{'cell':30s} {'long_exc':>9s} {'long_raw':>9s} {'short_free':>11s} {'short_borrow':>13s} {'ls_borrow':>10s}"
    print(hdr)
    for k, v in results.items():
        print(f"{k:30s} {str(v['long_excess']):>9s} {str(v['long_raw']):>9s} "
              f"{str(v['short_free']):>11s} {str(v['short_borrow']):>13s} {str(v['ls_borrow']):>10s}")
    print("\nbest long_excess:", best)


if __name__ == "__main__":
    main()
