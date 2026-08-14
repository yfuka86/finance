#!/usr/bin/env python3
"""Weekend-gap fade V2: multi-currency + conditioning (commodity/equity gaps,
Sunday early action, Friday alignment). Frozen in
docs/PREREGISTER_FX_WEEKEND_GAP_V2.md.

Second pass over the consumed 2011-2019 selection window (disclosed). Entry is
moved to the first bar >= Sun 18:00 ET because CME-linked proxies (gold, oil,
equity index) reopen at 18:00 ET -- conditioning earlier would be look-ahead.
Every examined cell is reported; the freeze rule in the doc decides whether the
2020+ confirmation opens (once) or the family closes permanently.
"""
from __future__ import annotations

import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from scripts.experiment_fx_micro3 import battery, judge
from scripts.experiment_fx_session import load_pair

NY = ZoneInfo("America/New_York")
SEL_YEARS = tuple(range(2011, 2020))
CONF_YEARS = tuple(range(2020, 2027))
N_INSTR = 13
OUT = Path("data/fx_weekend_gap_v2")

CROSSES = {  # cross -> (numerator pair, denominator/multiplier pair, op)
    "EUR_JPY": ("EUR_USD", "USD_JPY", "*"), "GBP_JPY": ("GBP_USD", "USD_JPY", "*"),
    "AUD_JPY": ("AUD_USD", "USD_JPY", "*"), "AUD_NZD": ("AUD_USD", "NZD_USD", "/"),
    "EUR_GBP": ("EUR_USD", "GBP_USD", "/"), "EUR_CHF": ("EUR_USD", "USD_CHF", "*"),
}
DIRECT = ["EUR_USD", "GBP_USD", "AUD_USD", "NZD_USD", "USD_JPY", "USD_CHF", "USD_CAD"]
# risk_dir: sign of the pair's move in a risk-on weekend; proxies checked at 18:00 ET
PROXY = {"USD_JPY": [("spx", +1)], "USD_CHF": [("spx", +1)],
         "AUD_USD": [("spx", +1), ("gold", +1)], "NZD_USD": [("spx", +1)],
         "USD_CAD": [("spx", -1), ("oil", -1)],
         "EUR_USD": [("spx", +1)], "GBP_USD": [("spx", +1)],
         "EUR_JPY": [("spx", +1)], "GBP_JPY": [("spx", +1)], "AUD_JPY": [("spx", +1)]}


def weekend_snaps(d: pd.DataFrame, reopen_hour: int = 17) -> pd.DataFrame:
    """Per-weekend snapshots keyed by the Monday of the gap week."""
    ny = d["ts"].dt.tz_convert(NY)
    dd = d.assign(ny=ny, week=ny.dt.normalize() - pd.to_timedelta(ny.dt.weekday, unit="D"))
    def snap(offset, last, tol_h):
        t = dd["week"] + offset
        sub = dd[dd["ny"] <= t] if last else dd[dd["ny"] >= t]
        g = sub.groupby("week")
        s = (g.last() if last else g.first())
        ok = (t.groupby(dd["week"]).first() - s["ny"]).abs() <= pd.Timedelta(hours=tol_h)
        return s[["mid", "half_spread"]].where(ok.reindex(s.index, fill_value=False))
    thu = snap(pd.Timedelta(days=3, hours=17), True, 3)
    fri = snap(pd.Timedelta(days=4, hours=17), True, 2)
    reo = snap(pd.Timedelta(days=6, hours=reopen_hour), False, 2)
    ent = snap(pd.Timedelta(days=6, hours=18), False, 1)
    # Monday-noon exit lives in the NEXT week's group; shift its anchor back.
    exi = snap(pd.Timedelta(hours=12), True, 2)
    exi.index = exi.index - pd.Timedelta(days=7)
    out = pd.DataFrame({
        "thu_mid": thu["mid"], "fri_mid": fri["mid"],
        "reo_mid": reo["mid"], "reo_hs": reo["half_spread"],
        "ent_mid": ent["mid"], "ent_hs": ent["half_spread"],
        "exi_mid": exi["mid"], "exi_hs": exi["half_spread"]})
    out.index = out.index.tz_localize(None)
    return out


def build_instruments(years) -> dict[str, pd.DataFrame]:
    base = {p: weekend_snaps(load_pair(p, years)) for p in DIRECT}
    instr = dict(base)
    for cross, (a, b, op) in CROSSES.items():
        A, B = base[a].copy(), base[b].copy()
        j = A.join(B, lsuffix="_a", rsuffix="_b", how="inner")
        o = pd.DataFrame(index=j.index)
        for c in ("thu_mid", "fri_mid", "reo_mid", "ent_mid", "exi_mid"):
            o[c] = j[f"{c}_a"] * j[f"{c}_b"] if op == "*" else j[f"{c}_a"] / j[f"{c}_b"]
        # two-leg execution: relative half-spreads add
        for m, h in (("reo", "reo_hs"), ("ent", "ent_hs"), ("exi", "exi_hs")):
            rel = j[f"{h}_a"] / j[f"{m}_mid_a"] + j[f"{h}_b"] / j[f"{m}_mid_b"]
            o[h] = rel * o[f"{m}_mid"]
        instr[cross] = o
    return instr


def proxy_gaps(years) -> pd.DataFrame:
    def cmdty(name):
        parts = []
        for y in list(years) + [min(years) - 1]:
            f = Path(f"data/fx_oanda_cmdty/parts/{name}_{y}_H1.parquet")
            if f.exists():
                parts.append(pd.read_parquet(f, columns=["ts", "open", "close"]))
        d = pd.concat(parts).sort_values("ts")
        ny = d["ts"].dt.tz_convert(NY)
        dd = d.assign(ny=ny, week=ny.dt.normalize()
                      - pd.to_timedelta(ny.dt.weekday, unit="D"))
        fri = dd[dd["ny"] <= dd["week"] + pd.Timedelta(days=4, hours=17)].groupby(
            "week")["close"].last()
        sun = dd[dd["ny"] >= dd["week"] + pd.Timedelta(days=6, hours=18)].groupby(
            "week")["open"].first()
        g = (sun / fri - 1).rename(None)
        g.index = g.index.tz_localize(None)
        return g
    def spx():
        parts = []
        for y in years:
            f = Path(f"data/fx_oanda_us/parts/SPX500_USD_{y}.parquet")
            if f.exists():
                parts.append(pd.read_parquet(f, columns=["ts", "close_bid", "close_ask"]))
        d = pd.concat(parts).sort_values("ts")
        d["mid"] = (d["close_bid"] + d["close_ask"]) / 2
        ny = d["ts"].dt.tz_convert(NY)
        dd = d.assign(ny=ny, week=ny.dt.normalize()
                      - pd.to_timedelta(ny.dt.weekday, unit="D"))
        fri = dd[dd["ny"] <= dd["week"] + pd.Timedelta(days=4, hours=17)].groupby(
            "week")["mid"].last()
        sun = dd[dd["ny"] >= dd["week"] + pd.Timedelta(days=6, hours=18)].groupby(
            "week")["mid"].first()
        g = sun / fri - 1
        g.index = g.index.tz_localize(None)
        return g
    px = pd.DataFrame({"spx": spx(), "gold": cmdty("XAU_USD"), "oil": cmdty("WTICO_USD")})
    for c in px.columns:
        px[f"{c}_med"] = px[c].abs().rolling(52, min_periods=20).median().shift(1)
    return px


def build_trades(instr: dict, px: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, s in instr.items():
        s = s.dropna(subset=["fri_mid", "reo_mid", "ent_mid", "exi_mid"])
        gap = s["reo_mid"] / s["fri_mid"] - 1
        thr = 2 * s["reo_hs"] / s["reo_mid"]
        fire = (gap.abs() > thr) & (gap.abs() <= .03)
        t = s[fire].copy()
        if t.empty:
            continue
        g = gap[fire]
        sign = -np.sign(g)
        entry = t["ent_mid"] + sign * t["ent_hs"]
        exit_ = t["exi_mid"] - sign * t["exi_hs"]
        t["ret"] = (sign * (exit_ / entry - 1)).astype(float)
        t["gap"] = g
        t["gap_x_hs"] = (g.abs() / (t["reo_hs"] / t["reo_mid"])).astype(float)
        early = t["ent_mid"] / t["reo_mid"] - 1
        t["toward_fill"] = np.sign(early) == -np.sign(g)
        t["fri_ret"] = t["fri_mid"] / t["thu_mid"] - 1
        t["continuation"] = np.sign(t["fri_ret"]) == np.sign(g)
        j = t.join(px, how="left")
        confirmed = pd.Series(False, index=t.index)
        for pname, rdir in PROXY.get(name, []):
            pg = j[pname] * rdir * np.sign(j["gap"])
            big = j[pname].abs() > j[f"{pname}_med"]
            confirmed |= (pg > 0) & big & j[pname].notna()
        t["news_confirmed"] = confirmed.to_numpy()
        t["instr"] = name
        t["is_cross"] = name in CROSSES
        rows.append(t.reset_index().rename(columns={"week": "monday"}))
    return pd.concat(rows, ignore_index=True)


def cell_stats(tr: pd.DataFrame, years_n: int) -> dict:
    if tr.empty:
        return {"trades": 0}
    daily = tr.assign(date=tr["monday"] + pd.Timedelta(days=7),
                      r=tr["ret"] / N_INSTR).groupby("date")["r"].sum()
    b = battery(daily)
    b["trades"] = int(len(tr))
    b["trades_per_year"] = round(len(tr) / years_n, 1)
    b["mean_bps_per_trade"] = round(float(tr["ret"].mean() * 1e4), 2)
    b["hit_rate"] = round(float((tr["ret"] > 0).mean()), 3)
    return b


def halves(tr: pd.DataFrame) -> dict:
    h1 = tr[tr["monday"].dt.year <= 2015]
    h2 = tr[tr["monday"].dt.year >= 2016]
    return {"2011_2015_sharpe": cell_stats(h1, 5).get("sharpe"),
            "2016_2019_sharpe": cell_stats(h2, 4).get("sharpe")}


def run_grid(tr: pd.DataFrame, years_n: int) -> dict:
    cells = {
        "C0_all13": tr,
        "C0_direct7": tr[~tr["is_cross"]],
        "C0_crosses6": tr[tr["is_cross"]],
        "C2_size_2_4": tr[tr["gap_x_hs"] <= 4],
        "C2_size_4_8": tr[(tr["gap_x_hs"] > 4) & (tr["gap_x_hs"] <= 8)],
        "C2_size_gt8": tr[tr["gap_x_hs"] > 8],
        "C3_toward_fill": tr[tr["toward_fill"]],
        "C3_extending": tr[~tr["toward_fill"]],
        "C1_not_news_confirmed": tr[~tr["news_confirmed"]],
        "C1_news_confirmed_skip_these": tr[tr["news_confirmed"]],
        "C4_continuation": tr[tr["continuation"]],
        "C4_reversal_gap": tr[~tr["continuation"]],
        "X_fill_and_not_news": tr[tr["toward_fill"] & ~tr["news_confirmed"]],
        "X_fill_not_news_size4plus": tr[tr["toward_fill"] & ~tr["news_confirmed"]
                                        & (tr["gap_x_hs"] > 4)],
    }
    out = {}
    for k, sub in cells.items():
        out[k] = cell_stats(sub, years_n)
        if out[k].get("trades", 0) > 50:
            out[k]["halves"] = halves(sub)
    per_instr = {}
    for name, sub in tr.groupby("instr"):
        per_instr[name] = {"trades": int(len(sub)),
                           "mean_bps": round(float(sub["ret"].mean() * 1e4), 2),
                           "hit": round(float((sub["ret"] > 0).mean()), 3)}
    out["per_instrument_diagnostic"] = per_instr
    return out


def freeze_check(grid: dict) -> list[str]:
    """Cells satisfying the frozen rule (battery + years + halves + fires)."""
    ok = []
    for k, v in grid.items():
        if not isinstance(v, dict) or "sharpe" not in v or v.get("sharpe") is None:
            continue
        crit = judge(v)
        years_ok = v.get("years", 0) - v.get("negative_years", 9) >= 7
        h = v.get("halves", {})
        halves_ok = (h.get("2011_2015_sharpe") or -9) > .5 \
            and (h.get("2016_2019_sharpe") or -9) > .5
        fires_ok = v.get("trades_per_year", 0) >= 40
        if all(crit.values()) and years_ok and halves_ok and fires_ok:
            ok.append(k)
    return ok


def main() -> None:
    instr = build_instruments(SEL_YEARS)
    px = proxy_gaps(SEL_YEARS)
    tr = build_trades(instr, px)
    grid = run_grid(tr, 9)
    passing = freeze_check(grid)
    summary = {"spec": "docs/PREREGISTER_FX_WEEKEND_GAP_V2.md",
               "selection_grid": grid, "freeze_rule_passing_cells": passing}
    if passing:
        frozen = min(passing, key=lambda k: (k.count("_"), -grid[k]["sharpe"]))
        summary["frozen_cell"] = frozen
        instr_c = build_instruments(CONF_YEARS)
        px_c = proxy_gaps(CONF_YEARS)
        tr_c = build_trades(instr_c, px_c)
        grid_c = run_grid(tr_c, 7)
        summary["confirmation"] = {frozen: grid_c.get(frozen)}
        conf = grid_c.get(frozen, {})
        summary["decision"] = ("GO_PENDING_USER_APPROVAL"
                               if conf.get("sharpe") and all(judge(conf).values())
                               else "NO_GO_AT_CONFIRMATION")
    else:
        summary["confirmation"] = "UNOPENED"
        summary["decision"] = "NO_GO_FAMILY_CLOSED"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
