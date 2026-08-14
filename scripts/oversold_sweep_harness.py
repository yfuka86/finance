#!/usr/bin/env python3
"""Parameterized sweep harness for the oversold-reversal family (2nd sweep).

Frozen judgment in docs/PREREGISTER_OVERSOLD_SWEEP2.md. Selection window
2018-2024 ONLY -- the 2025+ window is consumed for this family and is never
touched here. Cells are passed as JSON; every requested cell is reported.

CLI:
  PYTHONPATH=. python3 scripts/oversold_sweep_harness.py \
      --cells '[{"name":"base","floor":1e9,"mcap":"all","flow":"none",...}]' \
      --out data/jp_oversold_sweep2/axisA.json

Cell keys (all optional, defaults = base spec):
  floor: 5e8|1e9          mcap: all|small|mid|large (per-date terciles)
  dip:   z10|z20|rsi30    gapdip: none|gap|intra    ivol: all|hi|lo
  flow:  none|pos|neg|pos1|neg1     turn: none|hi
  market: none|m0         h: 1|3|5|10               tp: 0|1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experiment_oversold_interaction import simulate, battery
from trading.jp_intraday.daily_model import load_panel_cached
from trading.jp_intraday.flow_features import attach_flow_features

LO, HI = pd.Timestamp("2018-01-01"), pd.Timestamp("2024-12-31")
_CACHE: dict = {}


def build(floor: float) -> dict:
    if floor in _CACHE:
        return _CACHE[floor]
    p = load_panel_cached(min_value_yen=floor)[
        ["date", "symbol", "intraday_ret", "ret", "ret_on_fwd", "ivol",
         "overnight_gap", "raw_volume", "mktcap_yen"]].copy()
    p = attach_flow_features(p, lag=2)
    dates_all = pd.Index(sorted(p["date"].unique()))
    syms_all = pd.Index(sorted(p["symbol"].astype(str).unique()))
    piv = lambda c: p.pivot_table(index="date", columns="symbol", values=c,
                                  aggfunc="last").reindex(index=dates_all,
                                                          columns=syms_all)
    A = {"INTRA": piv("intraday_ret"), "CC": piv("ret"), "ONF": piv("ret_on_fwd"),
         "IVOL": piv("ivol"), "dates": dates_all}
    CC = A["CC"]
    A["mkt_intra"] = A["INTRA"].mean(axis=1)
    A["mkt_cc"] = CC.mean(axis=1)
    A["mkt_onf"] = A["ONF"].mean(axis=1)
    logcc = np.log1p(CC)
    r5 = np.expm1(logcc.rolling(5).sum())
    z = lambda M: M.sub(M.mean(axis=1), axis=0).div(M.std(axis=1), axis=0)
    A["Z5"] = z(r5)
    d = CC.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-CC.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    A["RSI"] = 100 - 100 / (1 + d / dn.replace(0, np.nan))
    GAP5 = piv("overnight_gap").rolling(5).sum()
    INT5 = A["INTRA"].rolling(5).sum()
    A["GAPLED"] = GAP5 < INT5                      # decline mostly overnight
    lv = np.log(piv("raw_volume").replace(0, np.nan))
    A["VSPIKE"] = z(lv - lv.rolling(20).mean())
    A["MCAP"] = piv("mktcap_yen")
    A["FLOW"] = piv("flow_close_z")
    tp = pd.read_parquet("data/jp_derivatives/topix_index_2008_2026.parquet")
    tp["Date"] = pd.to_datetime(tp["Date"])
    A["M"] = tp.set_index("Date")["C"].sort_index().pct_change(5).reindex(dates_all)
    _CACHE[floor] = A
    return A


def members_for(A: dict, c: dict) -> pd.DataFrame:
    dip = c.get("dip", "z10")
    if dip == "z10":
        m = A["Z5"].le(A["Z5"].quantile(.10, axis=1), axis=0)
    elif dip == "z20":
        m = A["Z5"].le(A["Z5"].quantile(.20, axis=1), axis=0)
    elif dip == "rsi30":
        m = A["RSI"] < 30
    else:
        raise ValueError(dip)
    mc = c.get("mcap", "all")
    if mc != "all":
        lo_t = A["MCAP"].quantile(1 / 3, axis=1)
        hi_t = A["MCAP"].quantile(2 / 3, axis=1)
        if mc == "small":
            m &= A["MCAP"].le(lo_t, axis=0)
        elif mc == "large":
            m &= A["MCAP"].gt(hi_t, axis=0)
        else:
            m &= A["MCAP"].gt(lo_t, axis=0) & A["MCAP"].le(hi_t, axis=0)
    fl = c.get("flow", "none")
    if fl != "none":
        thr = {"pos": 0, "neg": 0, "pos1": 1, "neg1": -1}[fl]
        m &= (A["FLOW"] > thr) if fl.startswith("pos") else (A["FLOW"] < thr)
    if c.get("turn", "none") == "hi":
        m &= A["VSPIKE"] > 1
    gd = c.get("gapdip", "none")
    if gd == "gap":
        m &= A["GAPLED"]
    elif gd == "intra":
        m &= ~A["GAPLED"]
    iv = c.get("ivol", "all")
    if iv != "all":
        med = A["IVOL"].quantile(.5, axis=1)
        m &= A["IVOL"].gt(med, axis=0) if iv == "hi" else A["IVOL"].le(med, axis=0)
    if c.get("market", "m0") == "m0":
        mm = A["M"].le(0.0)
        m = m[mm.reindex(m.index).fillna(False)]
    return m.fillna(False)


def run_cell(c: dict) -> dict:
    A = build(float(c.get("floor", 1e9)))
    mem = members_for(A, c)
    ex = simulate(A, mem, int(c.get("h", 5)), bool(int(c.get("tp", 1))))
    b = battery(ex, LO, HI)
    b["name"] = c.get("name", json.dumps(c, ensure_ascii=False))
    b["spec"] = {k: v for k, v in c.items() if k != "name"}
    if b.get("ir") is not None:
        e = ex.loc[LO:HI]
        h1 = e.loc[:"2021-12-31"]
        h2 = e.loc["2022-01-01":]
        sh = lambda r: (round(float(r.mean() / r.std() * 252 ** .5), 3)
                        if len(r) > 100 and r.std() > 0 else None)
        b["half_irs"] = [sh(h1), sh(h2)]
        yearly = e.groupby(e.index.year).sum()
        b["pos_years"] = int((yearly > 0).sum())
    return b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cells = json.loads(a.cells) if a.cells.strip().startswith("[") \
        else json.loads(Path(a.cells).read_text())
    out = [run_cell(c) for c in cells]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1),
                           encoding="utf-8")
    ok = [r for r in out if r.get("criteria") and all(r["criteria"].values())]
    print(json.dumps({"cells": len(out), "criteria_pass": [r["name"] for r in ok],
                      "best_ir": max((r.get("ir") or -9 for r in out), default=None)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
