#!/usr/bin/env python3
"""Long-only frontier: cash buys of sector-z top decile, judged on MARKET EXCESS.

Frozen in docs/PREREGISTER_LONG_ONLY_FRONTIER.md. No shorts -> no interest
floor, no borrow wall; the only question is whether selection beats (a) the
same-universe equal-weight market on the same cc legs and (b) TOPIX buy&hold.
Selection 2018-2024; confirmation 2025+ opens once on a full pass.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experiment_flow_fund_sector_on import build_panel
from scripts.experiment_fund_horizon_frontier import signal

SELECTION_END = pd.Timestamp("2024-12-31")
Q = 0.10
COST_SIDE = 0.5e-4
OUT = Path("data/jp_long_only_frontier")
CELLS = [(s, h) for s in ("V", "Q", "VQ") for h in (5, 20, 60)]


def topix_bh_sharpe(lo, hi) -> float:
    t = pd.read_parquet("data/jp_derivatives/topix_index_2008_2026.parquet")
    t["Date"] = pd.to_datetime(t["Date"])
    r = t.set_index("Date")["C"].sort_index().loc[lo:hi].pct_change().dropna()
    return float(r.mean() / r.std() * 252 ** .5)


def run_cell(p: pd.DataFrame, sig_name: str, h: int, lo, hi) -> dict:
    sub = p[(p["date"] >= lo) & (p["date"] <= hi)]
    f = sub[["date", "symbol", "z_bp", "z_roe", "z_flow", "ret_cc_fwd"]].copy()
    f["score"] = signal(f, sig_name)
    f = f.dropna(subset=["score"])
    thr = f.groupby("date")["score"].transform("quantile", 1 - Q)
    b = f[f["score"].ge(thr)].copy()
    n = b.groupby("date")["symbol"].transform("count")
    b["w"] = 1.0 / n
    W = b.pivot_table(index="date", columns="symbol", values="w",
                      aggfunc="sum").fillna(0.0)
    W_eff = sum(W.shift(k) for k in range(2, h + 2)) / h
    CC = f.pivot_table(index="date", columns="symbol", values="ret_cc_fwd",
                       aggfunc="last").reindex(index=W_eff.index, columns=W_eff.columns)
    live = W_eff.fillna(0.0) * CC.notna()
    scale = live.sum(axis=1).replace(0, np.nan)
    port_gross = (live * CC.fillna(0.0)).sum(axis=1) / scale       # fully invested
    turnover = W_eff.fillna(0.0).diff().abs().sum(axis=1) / scale
    net = (port_gross - turnover * COST_SIDE).iloc[h + 2:]
    # equal-weight same-universe market on the same cc legs, cost-free
    mkt = f.groupby("date")["ret_cc_fwd"].mean().reindex(net.index)
    ex = (net - mkt).dropna()
    def sh(r):
        return round(float(r.mean() / r.std() * 252 ** .5), 3) if len(r) > 100 and r.std() > 0 else None
    yearly = ex.groupby(ex.index.year).sum()
    top5 = float(ex.nlargest(5).sum() / ex.sum()) if ex.sum() > 0 else None
    ex10 = ex.drop(ex.nlargest(10).index)
    bh = topix_bh_sharpe(lo, hi)
    out = {"net_sharpe": sh(net), "ann_return_pct": round(float(net.mean() * 252 * 100), 2),
           "excess_ir": sh(ex), "excess_ann_pct": round(float(ex.mean() * 252 * 100), 2),
           "excess_neg_years": int((yearly < 0).sum()), "years": int(len(yearly)),
           "excess_top5_share": None if top5 is None else round(top5, 3),
           "excess_ir_ex_top10": sh(ex10),
           "topix_bh_sharpe": round(bh, 3),
           "turnover_per_day": round(float(turnover.iloc[h + 2:].mean()), 4),
           "names_avg": round(float((live > 0).sum(axis=1).mean()), 1),
           "days": int(len(net))}
    out["criteria"] = {
        "excess_ir_ge_05": bool((out["excess_ir"] or -9) >= .5),
        "sharpe_beats_bh_plus_02": bool((out["net_sharpe"] or -9) >= bh + .2),
        "excess_neg_years_le_third": bool(out["years"] > 0
                                          and out["excess_neg_years"] * 3 <= out["years"]),
        "excess_top5_lt_20pct": bool((out["excess_top5_share"] or 9) < .20),
        "excess_ir_ex_top10_ge_025": bool((out["excess_ir_ex_top10"] or -9) >= .25)}
    return out


def main() -> None:
    p = build_panel()
    summary = {"spec": "docs/PREREGISTER_LONG_ONLY_FRONTIER.md", "selection": {},
               "confirmation": "UNOPENED"}
    passing = []
    for s_name, h in CELLS:
        r = run_cell(p, s_name, h, pd.Timestamp("2018-01-01"), SELECTION_END)
        summary["selection"][f"{s_name}_h{h}"] = r
        if all(r["criteria"].values()):
            passing.append((s_name, h))
    summary["passing_cells"] = [f"{s}_h{h}" for s, h in passing]
    if passing:
        order = {"V": 0, "Q": 1, "VQ": 2}
        s_name, h = sorted(passing, key=lambda x: (order[x[0]], x[1]))[0]
        summary["frozen_cell"] = f"{s_name}_h{h}"
        conf = run_cell(p, s_name, h, pd.Timestamp("2025-01-01"),
                        pd.Timestamp("2026-12-31"))
        summary["confirmation"] = {summary["frozen_cell"]: conf}
        summary["decision"] = ("GO_PENDING_UNIT_LOT_AND_USER_APPROVAL"
                               if all(conf["criteria"].values())
                               else "NO_GO_AT_CONFIRMATION")
    else:
        summary["decision"] = "NO_GO_AT_SELECTION"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
