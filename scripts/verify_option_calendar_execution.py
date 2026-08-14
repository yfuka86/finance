#!/usr/bin/env python3
"""Adversarial verification of the put calendar: is same-day settle entry the edge?

The pre-registered run (Sharpe 1.97) computes the signal z from day D's settlement
IV and *enters at day D's settlement price*. Those are the same numbers: the
settlement is fixed after the close, so you cannot observe it and then trade at it.

Worse, the two are mechanically coupled through settlement noise: if the near
leg's settle prints too high by ε, the slope IV_near − IV_far is inflated (which
*triggers* the signal) and the debit far − near is deflated (which *cheapens the
entry*) — so same-bar entry harvests pure noise reversion. The measured put/call
parity violation (std ≈ 23% of the debit) says that noise is large.

The executable form is: signal from D's settle, enter at D+1's settle
(contracts and strike chosen on D). This script measures that form, plus a
parity-averaged variant (mean of the put and call calendar on the identical
(day, strike, expiry-pair), which cancels leg-specific settlement noise).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experiment_option_calendar import (
    FAR_DTE, HOLD, MIN_OI, NEAR_DTE, TICKS_COST, Z_ENTRY, load, term_slope,
)

OUT = Path("data/jp_option_calendar")


def build(d, z, sessions, by, entry_lag=0, parity_avg=False) -> pd.DataFrame:
    week, rows = set(), []
    for day in sessions:
        wk = (day.isocalendar().year, day.isocalendar().week)
        zv = z.get(day, np.nan)
        if wk in week or not (zv >= Z_ENTRY):
            continue
        i = sessions.get_loc(day)
        if i + entry_lag + HOLD >= len(sessions):
            continue
        g = by.get(day)
        if g is None:
            continue
        gp = g[g["is_put"]]
        n = gp[gp.dte.between(*NEAR_DTE) & gp.Vo.gt(0) & gp.OI.ge(MIN_OI)]
        f = gp[gp.dte.between(*FAR_DTE) & gp.Vo.gt(0) & gp.OI.ge(MIN_OI)]
        if n.empty or f.empty:
            continue
        K = n.loc[(n.Strike - n.UnderPx).abs().idxmin(), "Strike"]
        ns, fs = n[n.Strike.eq(K)], f[f.Strike.eq(K)]
        if ns.empty or fs.empty:
            continue
        nd, fd = ns.iloc[0], fs.iloc[0]
        codes = [(nd.Code, fd.Code)]
        if parity_avg:
            gc = g[~g["is_put"] & g.Strike.eq(K)]
            nc, fc = gc[gc.dte.eq(nd.dte)], gc[gc.dte.eq(fd.dte)]
            if nc.empty or fc.empty:
                continue
            codes.append((nc.iloc[0].Code, fc.iloc[0].Code))
        eday, xday = sessions[i + entry_lag], sessions[i + entry_lag + HOLD]
        ge, gx = by.get(eday), by.get(xday)
        if ge is None or gx is None:
            continue
        rets = []
        for ncode, fcode in codes:
            n0, f0 = ge[ge.Code.eq(ncode)], ge[ge.Code.eq(fcode)]
            n1, f1 = gx[gx.Code.eq(ncode)], gx[gx.Code.eq(fcode)]
            if any(x.empty for x in (n0, f0, n1, f1)):
                continue
            debit = f0.iloc[0].Settle - n0.iloc[0].Settle
            if debit <= 0:
                continue
            rets.append((f1.iloc[0].Settle - n1.iloc[0].Settle - debit - TICKS_COST) / debit)
        if not rets:
            continue
        rows.append({"entry": day, "ret": float(np.mean(rets))})
        week.add(wk)
    return pd.DataFrame(rows)


def rep(t: pd.DataFrame, lo=None, hi=None) -> dict:
    r = t["ret"] if lo is None else t[t["entry"].between(lo, hi)]["ret"]
    if len(r) < 8:
        return {"trades": int(len(r))}
    pos = r[r > 0].sum()
    return {"trades": int(len(r)),
            "sharpe": round(float(r.mean() / r.std() * np.sqrt(52)), 3),
            "mean": round(float(r.mean()), 4),
            "win_rate": round(float(r.gt(0).mean()), 4),
            "worst": round(float(r.min()), 4),
            "top_share": round(float(r.max() / pos), 4) if pos > 0 else None}


def main() -> None:
    d = load()
    slope = term_slope(d)
    z = ((slope - slope.rolling(252, min_periods=120).mean())
         / slope.rolling(252, min_periods=120).std()).dropna()
    sessions = pd.Index(sorted(d["Date"].unique()))
    by = {day: g for day, g in d.groupby("Date")}
    out = {"note": "signal from D settle; entry_lag=1 = enter at D+1 settle (executable form)"}
    for label, kw in [("base_same_day", {}), ("t1_executable", dict(entry_lag=1)),
                      ("parity_avg_same_day", dict(parity_avg=True)),
                      ("parity_avg_t1", dict(entry_lag=1, parity_avg=True))]:
        t = build(d, z, sessions, by, **kw)
        out[label] = {"all": rep(t), "first_half": rep(t, "2019-01-01", "2022-12-31"),
                      "second_half": rep(t, "2023-01-01", "2026-12-31")}
    ex = out["t1_executable"]["all"]
    out["verdict"] = ("SAME_BAR_ARTIFACT_CONFIRMED"
                      if (ex.get("sharpe") or 9) < 1.0 else "EXECUTABLE_FORM_SURVIVES")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "execution_verification.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
