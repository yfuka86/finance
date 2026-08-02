#!/usr/bin/env python3
"""JP cross-sectional momentum — a *reproduction*, not a new selection.

AGENTS records (2026-07-30) that momentum was tested on the 2021-2024 selection
window and died "全滅・再検証不要", with gross itself negative. No script for
that run exists in the repository, so the claim cannot currently be checked.
This runs the standard variants on **the same documented window** so the entry
becomes reproducible. It is not a fresh selection and must not be used to pick a
surviving variant: the direction (long winners / short losers), the formation
grid, and the holding grid are all fixed below before any number is read.

PIT: momentum at date D uses adjusted closes through D-1 only. It is computed on
the **unfiltered** daily history and merged in afterwards — shifting inside the
liquidity-filtered panel would stitch across removed rows (the schema-v7 bug).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from trading.jp_intraday.daily_gap import load_existing_daily
from trading.jp_intraday.daily_model import load_panel_cached
from trading.jp_intraday.strategies import unit_lot_backtest

# (name, formation length in sessions, skip in sessions) — standard definitions.
VARIANTS = [("mom12_1", 252, 21), ("mom6_1", 126, 21), ("mom3_1", 60, 21),
            ("mom60_5", 60, 5), ("mom20_1", 20, 1)]
HOLDINGS = (1, 5, 10, 20)
WINDOW = ("2021-01-01", "2024-12-31")   # the documented selection window
COSTS = (1.0, 2.0)
OUT = Path("data/jp_momentum")


def momentum_frame() -> pd.DataFrame:
    """Momentum on the full history, with a calendar guard against stale stitching."""
    d = load_existing_daily().rename(columns={"Date": "date", "Code": "symbol"})
    d["date"] = pd.to_datetime(d["date"])
    d = d.dropna(subset=["AdjC"]).sort_values(["symbol", "date"])
    d = d[d["AdjC"] > 0].drop_duplicates(["symbol", "date"])
    logc = np.log(d["AdjC"])
    g = d.groupby("symbol", sort=False)
    out = d[["date", "symbol"]].copy()
    for name, length, skip in VARIANTS:
        near, far = 1 + skip, 1 + skip + length
        near_v = logc.groupby(d["symbol"], sort=False).shift(near)
        far_v = logc.groupby(d["symbol"], sort=False).shift(far)
        # Reject windows stitched across suspensions/delisting gaps: the far leg
        # must sit within a plausible calendar span of the formation length.
        far_d = g["date"].shift(far)
        span_ok = (d["date"] - far_d).dt.days.between(far * 1.0, far * 2.2)
        out[name] = (near_v - far_v).where(span_ok)
    return out


def stats(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 50 or r.std() == 0:
        return {"sharpe": None}
    eq = (1 + r).cumprod()
    return {"sharpe": round(float(r.mean() / r.std() * 252 ** .5), 3),
            "ann_return": round(float(r.mean() * 252), 4),
            "max_drawdown": round(float((eq / eq.cummax() - 1).min()), 4),
            "days": int(len(r))}


def gross_tranche(panel: pd.DataFrame, col: str, hold: int, q: float = .2) -> pd.Series:
    """Equal-weight long-top / short-bottom quintile, averaged over `hold` tranches."""
    p = panel.dropna(subset=[col, "ret"])[["date", "symbol", col, "ret"]].copy()
    rank = p.groupby("date")[col].rank(pct=True)
    w = pd.Series(0.0, index=p.index)
    w[rank >= 1 - q] = 1.0
    w[rank <= q] = -1.0
    p["w"] = w / w.abs().groupby(p["date"]).transform("sum").replace(0, np.nan)
    wide = p.pivot_table(index="date", columns="symbol", values="w", aggfunc="last")
    rets = p.pivot_table(index="date", columns="symbol", values="ret", aggfunc="last")
    # Hold each day's book for `hold` sessions: average the overlapping tranches.
    held = sum(wide.shift(k) for k in range(1, hold + 1)) / hold
    return (held * rets).sum(axis=1).reindex(rets.index)


def main() -> None:
    panel = load_panel_cached(min_value_yen=1e9)
    panel = panel.merge(momentum_frame(), on=["date", "symbol"], how="left")
    panel = panel[panel["date"].between(*WINDOW)].copy()

    summary = {"window": WINDOW, "note": "reproduction of the 2026-07-30 rejection",
               "rows": len(panel), "variants": {}}
    for name, _, _ in VARIANTS:
        cov = float(panel[name].notna().mean())
        entry = {"coverage": round(cov, 3), "daily_flat": {}, "gross_tranche": {}}
        # Family A: 日中実行（寄成→引成・本番制約・単元）— long winners / short losers.
        frame = panel.dropna(subset=[name]).copy()
        frame["_s"] = frame[name] - frame.groupby("date")[name].transform("mean")
        for cost in COSTS:
            daily, _ = unit_lot_backtest(frame, capital_yen=2e7, names_per_side=8,
                                         margin_ratio=2.0, cost_bps_side=cost,
                                         construction="dollar_neutral")
            entry["daily_flat"][f"{cost}bps"] = stats(daily["net"]) if len(daily) else {"sharpe": None}
        # Family B: gross のみ（コストをゼロにしても α があるか）
        for hold in HOLDINGS:
            entry["gross_tranche"][f"h{hold}"] = stats(gross_tranche(panel, name, hold))
        summary["variants"][name] = entry
        print(name, json.dumps(entry, ensure_ascii=False), flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
