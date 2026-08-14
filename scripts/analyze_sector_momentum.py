#!/usr/bin/env python3
"""How much of long-horizon JP stock performance is sector, and does sector momentum persist?

DIAGNOSTIC, not a strategy: no execution, no costs, no selection discipline. If
anything here looks tradable it must go through its own preregistration before
any strategy claim. Motivated by the user's observation that long-term stock
screens seem dominated by sector momentum.

Three questions:
  A. Variance share: at 1/3/6/12-month horizons, what fraction of cross-sectional
     stock return variance is explained by S33 sector membership? (stocks 2018-2026)
  B. Screen contamination: if you rank stocks by 12-1 momentum, how correlated is
     that ranking with the stock's *sector* momentum? (= how sector-driven a
     momentum screen is)
  C. Sector momentum persistence: with 18 years of sector indices (2008-2026),
     does past sector return predict future sector return? Rank IC + top-vs-
     bottom-quartile spread, formation x holding grid, overlapping tranches.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from trading.jp_intraday.daily_gap import load_existing_daily
from trading.jp_intraday.daily_model import load_master

OUT = Path("data/jp_sector_momentum")
HORIZONS = {"1m": 21, "3m": 63, "6m": 126, "12m": 252}


def stock_panel() -> pd.DataFrame:
    d = load_existing_daily()
    d["Date"] = pd.to_datetime(d["Date"])
    d["symbol"] = d["Code"].astype(str)
    m = load_master()
    d = d.merge(m[["symbol", "s33_code", "is_fund"]], on="symbol", how="left")
    d = d[(d["is_fund"] != True) & d["s33_code"].notna()]          # noqa: E712
    # 月末値に絞る（長期ホライズンの分析に日次は不要）
    d = d.sort_values(["symbol", "Date"])
    d["ym"] = d["Date"].dt.to_period("M")
    eom = d.groupby(["symbol", "ym"]).tail(1)
    px = eom.pivot_table(index="ym", columns="symbol", values="AdjC", aggfunc="last")
    sec = m.set_index("symbol")["s33_code"]
    return px, sec


def sector_indices() -> pd.DataFrame:
    d = pd.read_parquet("data/jp_derivatives/sector_indices_2008_2026.parquet")
    d["Date"] = pd.to_datetime(d["Date"])
    d["ym"] = d["Date"].dt.to_period("M")
    eom = d.sort_values("Date").groupby(["Code", "ym"]).tail(1)
    return eom.pivot_table(index="ym", columns="Code", values="C", aggfunc="last")


def var_share(px: pd.DataFrame, sec: pd.Series) -> dict:
    """A: fraction of cross-sectional H-month stock return variance from sector means."""
    out = {}
    for lab, _ in HORIZONS.items():
        months = int(lab.rstrip("m"))
        r = px.pct_change(months, fill_method=None)
        shares = []
        for ym in r.index[months::months]:                     # 非重複期間
            row = r.loc[ym].dropna()
            row = row[(row > -0.9) & (row < 4.0)]              # 上場直後等の異常値を除外
            if len(row) < 300:
                continue
            g = row.groupby(sec.reindex(row.index))
            between = g.transform("mean")
            if row.var() > 0:
                shares.append(float(between.var() / row.var()))
        out[lab] = {"periods": len(shares), "mean_share": round(float(np.mean(shares)), 4),
                    "p25": round(float(np.percentile(shares, 25)), 4),
                    "p75": round(float(np.percentile(shares, 75)), 4)}
    return out


def screen_contamination(px: pd.DataFrame, sec: pd.Series) -> dict:
    """B: corr between a stock's 12-1 momentum and its sector's 12-1 momentum."""
    r12 = px.shift(1) / px.shift(12) - 1                       # 12-1 モメンタム
    cors, topshare = [], []
    for ym in r12.index[13::3]:
        row = r12.loc[ym].dropna()
        row = row[(row > -0.9) & (row < 4.0)]
        if len(row) < 300:
            continue
        smom = row.groupby(sec.reindex(row.index)).transform("mean")
        cors.append(float(pd.Series(row.values).corr(pd.Series(smom.values), method="spearman")))
        top = row.nlargest(max(30, len(row) // 10))
        topshare.append(float(smom.reindex(top.index).mean() / top.mean())
                        if top.mean() != 0 else np.nan)
    return {"rank_corr_stockmom_vs_sectormom": {
                "mean": round(float(np.nanmean(cors)), 3),
                "p25": round(float(np.nanpercentile(cors, 25)), 3),
                "p75": round(float(np.nanpercentile(cors, 75)), 3)},
            "top_decile_momentum_share_from_sector": round(float(np.nanmean(topshare)), 3),
            "periods": len(cors)}


def sector_persistence(idx: pd.DataFrame) -> dict:
    """C: does past sector return predict future sector return? (2008-2026)"""
    out = {}
    for F in (3, 6, 12):
        past = idx.shift(1) / idx.shift(1 + F) - 1             # skip 1m
        for H in (1, 3, 6, 12):
            fut = idx.shift(-H) / idx - 1
            ics, spreads = [], []
            for ym in past.index[F + 1:-H]:
                p, f = past.loc[ym].dropna(), fut.loc[ym].dropna()
                common = p.index.intersection(f.index)
                if len(common) < 25:
                    continue
                ics.append(float(p[common].corr(f[common], method="spearman")))
                rk = p[common].rank()
                top = f[common][rk > len(common) * .75].mean()
                bot = f[common][rk <= len(common) * .25].mean()
                spreads.append(float(top - bot))
            n_eff = max(1, len(ics) // H)                       # 重複補正の保守近似
            ic = np.mean(ics)
            t = ic / (np.std(ics) / np.sqrt(n_eff)) if np.std(ics) else np.nan
            out[f"F{F}m_H{H}m"] = {
                "mean_rank_ic": round(float(ic), 3),
                "t_overlap_adj": round(float(t), 2),
                "mean_top_minus_bottom_pct": round(float(np.mean(spreads)) * 100, 2),
                "periods": len(ics)}
    return out


def main() -> None:
    px, sec = stock_panel()
    idx = sector_indices()
    out = {"note": "DIAGNOSTIC only — no costs, no execution, no strategy claim. "
                   "Stocks 2018-2026 (survivorship-safe panel), sector indices 2008-2026.",
           "A_variance_share_by_horizon": var_share(px, sec),
           "B_momentum_screen_sector_contamination": screen_contamination(px, sec),
           "C_sector_momentum_persistence_2008_2026": sector_persistence(idx)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
