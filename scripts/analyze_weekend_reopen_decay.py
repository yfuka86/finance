#!/usr/bin/env python3
"""Diagnostic: intra-hour decay of the Sunday-reopen liquidity premium.

Answers "もっと短い時間足でやったら？" empirically. A taker's cost is per round
trip, so shorter bars per se change nothing; the two questions only minute data
can answer are (1) does the premium decay slower than the spread tightens
(a late-entry pocket), and (2) is the premium independent enough of the entry
spread that tight-spread Sundays are net-positive (PIT-conditionable)?

Selection window 2011-2019 only. Firing rule identical to H1/V2
(|gap| > 2x reopen half-spread, <=3%). Direct 7 pairs (crosses pay double
spread -- hopeless by construction). Exit fixed at Monday 12:00 ET (the cheap
exit; 18:00 exit costs 2.4bps for zero extra premium).
"""
from __future__ import annotations

import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from scripts.experiment_fx_micro3 import battery
from scripts.experiment_fx_session import load_pair

NY = ZoneInfo("America/New_York")
SEL_YEARS = tuple(range(2011, 2020))
PAIRS = ["EUR_USD", "GBP_USD", "AUD_USD", "NZD_USD", "USD_JPY", "USD_CHF", "USD_CAD"]
ENTRY_MIN = [0, 2, 5, 10, 15, 20, 30, 45]        # minutes after Sun 17:00 ET
OUT = Path("data/fx_weekend_gap_v2")


def weekend_frames(pair: str) -> pd.DataFrame:
    d = load_pair(pair, SEL_YEARS)
    ny = d["ts"].dt.tz_convert(NY)
    dd = d.assign(ny=ny, week=ny.dt.normalize() - pd.to_timedelta(ny.dt.weekday, unit="D"))
    fri_cut = dd["week"] + pd.Timedelta(days=4, hours=17)
    fri = dd[dd["ny"] <= fri_cut].groupby("week").last()
    fri_ok = (fri_cut.groupby(dd["week"]).first() - fri["ny"]) <= pd.Timedelta(hours=2)
    out = pd.DataFrame({"fri_mid": fri["mid"].where(fri_ok)})
    for m in ENTRY_MIN:
        t = dd["week"] + pd.Timedelta(days=6, hours=17, minutes=m)
        sub = dd[dd["ny"] >= t].groupby("week")
        s = sub.first()
        ok = (s["ny"] - t.groupby(dd["week"]).first()) <= pd.Timedelta(minutes=3)
        out[f"mid_{m}"] = s["mid"].where(ok)
        out[f"hs_{m}"] = s["half_spread"].where(ok)
    mon_cut = dd["week"] + pd.Timedelta(hours=12)
    mon = dd[dd["ny"] <= mon_cut].groupby("week").last()
    mon_ok = (mon_cut.groupby(dd["week"]).first() - mon["ny"]) <= pd.Timedelta(hours=2)
    mon = mon.where(mon_ok)
    mon.index = mon.index - pd.Timedelta(days=7)
    out = out.join(mon[["mid", "half_spread"]].rename(
        columns={"mid": "exit_mid", "half_spread": "exit_hs"}))
    out["pair"] = pair
    out.index = out.index.tz_localize(None)
    return out


def main() -> None:
    frames = pd.concat([weekend_frames(p) for p in PAIRS])
    f = frames.dropna(subset=["fri_mid", "mid_0", "exit_mid"]).copy()
    f["gap"] = f["mid_0"] / f["fri_mid"] - 1
    f["rel_hs0"] = f["hs_0"] / f["mid_0"]
    fired = f[(f["gap"].abs() > 2 * f["rel_hs0"]) & (f["gap"].abs() <= .03)].copy()
    sign = -np.sign(fired["gap"])
    res = {"n_fired": int(len(fired)), "note": "selection window only; exit Mon 12:00 ET"}

    decay = {}
    for m in ENTRY_MIN:
        mid, hs = fired[f"mid_{m}"], fired[f"hs_{m}"]
        ok = mid.notna()
        g = (sign * (fired["exit_mid"] / mid - 1))[ok]
        entry_px = mid + sign * hs
        exit_px = fired["exit_mid"] - sign * fired["exit_hs"]
        n = (sign * (exit_px / entry_px - 1))[ok]
        decay[f"t+{m}min"] = {
            "n": int(ok.sum()),
            "gross_to_mon_bps": round(float(g.mean() * 1e4), 2),
            "entry_hs_bps": round(float((hs / mid)[ok].mean() * 1e4), 2),
            "net_bps": round(float(n.mean() * 1e4), 2),
            "net_t": round(float(n.mean() / n.std() * np.sqrt(len(n))), 2)}
    res["decay_curve"] = decay

    buckets = {}
    prev = 0
    for m in ENTRY_MIN[1:] + ["exit"]:
        cur = fired["exit_mid"] if m == "exit" else fired[f"mid_{m}"]
        g = sign * (cur / fired[f"mid_{prev}"] - 1)
        buckets[f"{prev}->{m}"] = round(float(g.mean() * 1e4), 2)
        prev = m
    res["premium_per_bucket_bps"] = buckets

    # spread-conditioned taker (PIT: the entry spread is visible before trading)
    cond = {}
    fired["net0"] = sign * ((fired["exit_mid"] - sign * fired["exit_hs"])
                            / (fired["mid_0"] + sign * fired["hs_0"]) - 1)
    fired["gross0"] = sign * (fired["exit_mid"] / fired["mid_0"] - 1)
    q = fired["rel_hs0"] * 1e4
    for label, mask in [("hs_q1_tightest", q <= q.quantile(.25)),
                        ("hs_q2", (q > q.quantile(.25)) & (q <= q.quantile(.5))),
                        ("hs_q3", (q > q.quantile(.5)) & (q <= q.quantile(.75))),
                        ("hs_q4_widest", q > q.quantile(.75))]:
        sub = fired[mask]
        cond[label] = {
            "n": int(len(sub)),
            "entry_hs_bps": round(float(q[mask].mean()), 2),
            "gross_bps": round(float(sub["gross0"].mean() * 1e4), 2),
            "net_bps": round(float(sub["net0"].mean() * 1e4), 2),
            "net_t": round(float(sub["net0"].mean() / sub["net0"].std()
                                 * np.sqrt(len(sub))), 2)}
    res["spread_quartiles"] = cond
    res["corr_gross_vs_entry_hs"] = round(float(
        fired["gross0"].mul(1e4).corr(q)), 3)

    thr_cells = {}
    for x in (1.0, 1.5, 2.0, 2.5, 3.0):
        sub = fired[q <= x]
        if len(sub) < 30:
            thr_cells[f"hs<={x}bps"] = {"n": int(len(sub))}
            continue
        daily = sub.assign(date=sub.index + pd.Timedelta(days=7),
                           r=sub["net0"] / 7).groupby("date")["r"].sum()
        b = battery(daily)
        thr_cells[f"hs<={x}bps"] = {
            "n": int(len(sub)), "per_year": round(len(sub) / 9, 1),
            "net_bps": round(float(sub["net0"].mean() * 1e4), 2),
            "net_t": round(float(sub["net0"].mean() / sub["net0"].std()
                                 * np.sqrt(len(sub))), 2),
            "portfolio_sharpe": b.get("sharpe")}
    res["pit_threshold_cells"] = thr_cells

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "reopen_decay_diagnostic.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
