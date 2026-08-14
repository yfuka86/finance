#!/usr/bin/env python3
"""Institutional footprint timing: weekly investor-type flows -> TOPIX.

Frozen in docs/PREREGISTER_JP_INVESTOR_FLOW.md. Primary cell: long/flat on the
sign of the trailing 4 published weeks of foreign net buying (FrgnBal), executed
the business day after publication at 2bps/side. Diagnostic: IC table of all 12
investor categories (52w z, PIT) vs next-week TOPIX, with and without a
past-return control. Selection 2009-2019; confirmation 2020+ only on a pass.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experiment_fx_micro3 import judge

CATS = ["Prop", "Brk", "Ind", "Frgn", "SecCo", "InvTr", "BusCo", "OthCo",
        "InsCo", "Bank", "TrstBnk", "OthFin"]
COST = 2e-4
SHORT_RATE = .042
SEL = ("2009-01-01", "2019-12-31")
CONF = ("2020-01-01", "2026-12-31")
OUT = Path("data/jp_investor_flow")


def load_flows() -> pd.DataFrame:
    d = pd.read_parquet("data/jp_daily_history/investor_types_2008_2026.parquet")
    d = d[d["Section"].isin(["TSE1st", "TSEPrime"])].copy()
    # overlap month at the 2022-04 segment change: prefer the newer section
    d = d.sort_values(["EnDate", "Section"]).drop_duplicates("EnDate", keep="last")
    return d.set_index("EnDate")


def load_topix() -> pd.Series:
    t = pd.read_parquet("data/jp_derivatives/topix_index_2008_2026.parquet")
    t["Date"] = pd.to_datetime(t["Date"])
    return t.set_index("Date")["C"].sort_index()


def battery_vs_bh(r: pd.Series, px: pd.Series, lo: str, hi: str) -> dict:
    r = r.loc[lo:hi]
    cal = pd.date_range(r.index.min(), r.index.max(), freq="B")
    r = r.reindex(cal).fillna(0.0)
    if len(r) < 100 or r.std() == 0:
        return {"sharpe": None}
    eq = (1 + r).cumprod()
    yearly = r.groupby(r.index.year).sum()
    top5 = float(r.nlargest(5).sum() / r.sum()) if r.sum() > 0 else None
    ex10 = r.drop(r.nlargest(10).index)
    bh = px.loc[lo:hi].pct_change().dropna()
    bh_sh = float(bh.mean() / bh.std() * 252 ** .5)
    sh = float(r.mean() / r.std() * 252 ** .5)
    return {"sharpe": round(sh, 3),
            "ann_return_pct": round(float(r.mean() * 252 * 100), 2),
            "max_drawdown_pct": round(float((eq / eq.cummax() - 1).min() * 100), 2),
            "negative_years": int((yearly < 0).sum()), "years": int(len(yearly)),
            "top5_day_share": None if top5 is None else round(top5, 3),
            "sharpe_ex_top10": round(float(ex10.mean() / ex10.std() * 252 ** .5), 3),
            "bh_sharpe": round(bh_sh, 3), "beats_bh": bool(sh > bh_sh),
            "exposure": round(float((r != 0).mean()), 3)}


def run_cell(flows: pd.DataFrame, px: pd.Series, window: int,
             long_short: bool) -> pd.Series:
    sig = flows["FrgnBal"].rolling(window).sum()
    # tradable from the business day after publication
    eff = pd.DataFrame({"sig": sig.to_numpy()},
                       index=pd.DatetimeIndex(flows["PubDate"])
                       + pd.offsets.BusinessDay(1)).dropna()
    eff = eff[~eff.index.duplicated(keep="last")].sort_index()
    pos_w = np.sign(eff["sig"]) if long_short else (eff["sig"] > 0).astype(float)
    pos = pos_w.reindex(px.index, method="ffill").fillna(0.0)
    ret = px.pct_change().fillna(0.0)
    strat = pos.shift(1) * ret                      # position set at close, earns next day
    cost = pos.diff().abs().fillna(0.0) * COST
    carry = np.where(pos.shift(1) < 0, SHORT_RATE / 245, 0.0)
    return strat - cost - carry


def diagnostics(flows: pd.DataFrame, px: pd.Series, lo: str, hi: str) -> dict:
    wk_px = px.reindex(pd.DatetimeIndex(flows["PubDate"])
                       + pd.offsets.BusinessDay(1), method="ffill")
    fwd = pd.Series(wk_px.to_numpy(), index=flows.index)
    fwd_ret = fwd.shift(-1) / fwd - 1               # pub-to-pub forward return
    past4 = fwd / fwd.shift(4) - 1                  # trailing 4-week price control
    out = {}
    mask = (flows.index >= lo) & (flows.index <= hi)
    for c in CATS:
        bal = flows[f"{c}Bal"]
        z = (bal - bal.rolling(52).mean()) / bal.rolling(52).std()
        v = pd.DataFrame({"z": z, "fwd": fwd_ret, "past": past4})[mask].dropna()
        if len(v) < 100:
            continue
        ic = v["z"].corr(v["fwd"], method="spearman")
        beta = np.polyfit(v["past"], v["z"], 1)
        resid = v["z"] - (beta[0] * v["past"] + beta[1])
        pic = resid.corr(v["fwd"], method="spearman")
        n = len(v)
        out[c] = {"ic": round(float(ic), 4), "t": round(float(ic * np.sqrt(n)), 2),
                  "partial_ic_price_controlled": round(float(pic), 4),
                  "partial_t": round(float(pic * np.sqrt(n)), 2),
                  "corr_with_past4w_ret": round(float(v["z"].corr(v["past"])), 3)}
    return out


def main() -> None:
    flows = load_flows()
    px = load_topix()
    summary = {"spec": "docs/PREREGISTER_JP_INVESTOR_FLOW.md",
               "weeks": int(len(flows))}
    a1 = run_cell(flows, px, 4, False)
    sel = battery_vs_bh(a1, px, *SEL)
    crit = judge(sel)
    crit["beats_buy_and_hold"] = sel.get("beats_bh", False)
    summary["A1_foreign_4w_long_flat_selection"] = sel
    summary["selection_criteria"] = crit
    summary["sensitivity_selection"] = {
        "A2_long_short": battery_vs_bh(run_cell(flows, px, 4, True), px, *SEL),
        "A3_13w": battery_vs_bh(run_cell(flows, px, 13, False), px, *SEL)}
    summary["diagnostic_ic_selection"] = diagnostics(flows, px, *SEL)
    if all(crit.values()):
        summary["confirmation"] = battery_vs_bh(a1, px, *CONF)
    else:
        summary["confirmation"] = "UNOPENED"
    summary["decision"] = ("SELECTION_PASSED_SEE_CONFIRMATION"
                           if all(crit.values()) else "NO_GO_AT_SELECTION")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
