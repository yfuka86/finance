#!/usr/bin/env python3
"""FX session effect (Ranaldo: home currency depreciates in home hours).

Frozen in docs/PREREGISTER_FX_SESSION.md. Seven daily legs, each selling a
currency during its home business hours; equal-weight portfolio; measured
half-spread costs; no rollover crossing. Selection 2011-2019; confirmation
2020+ opened only if all portfolio criteria pass.
"""
from __future__ import annotations

import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

LEGS = [  # (leg, pair, sign(+1 buy pair), tz, start "HH:MM", end "HH:MM")
    ("JPY", "USD_JPY", +1, "Asia/Tokyo", "09:00", "15:00"),
    ("EUR", "EUR_USD", -1, "Europe/Berlin", "08:00", "16:00"),
    ("GBP", "GBP_USD", -1, "Europe/London", "08:00", "16:00"),
    ("AUD", "AUD_USD", -1, "Australia/Sydney", "09:00", "17:00"),
    ("NZD", "NZD_USD", -1, "Pacific/Auckland", "09:00", "17:00"),
    ("CHF", "USD_CHF", +1, "Europe/Zurich", "08:00", "16:00"),
    ("USD", "EUR_USD", +1, "America/New_York", "11:00", "16:30"),
]
SEL_YEARS = tuple(range(2011, 2020))
CONF_YEARS = tuple(range(2020, 2027))
SNAP_TOL = pd.Timedelta("30min")
OUT = Path("data/fx_session")


def load_pair(pair: str, years) -> pd.DataFrame:
    parts = []
    for y in years:
        f = Path(f"data/fx_oanda_min/parts/{pair}_{y}.parquet")
        if f.exists():
            parts.append(pd.read_parquet(f, columns=["ts", "close_bid", "close_ask"]))
    d = pd.concat(parts, ignore_index=True).sort_values("ts")
    d["mid"] = (d["close_bid"] + d["close_ask"]) / 2
    hs = ((d["close_ask"] - d["close_bid"]).clip(lower=0) / 2)
    d["half_spread"] = hs.clip(upper=hs.quantile(.99))
    return d[["ts", "mid", "half_spread"]].reset_index(drop=True)


def leg_daily(pair_df: pd.DataFrame, sign: int, tz: str, start: str, end: str,
              morning_only: bool = False) -> pd.DataFrame:
    """One row per local calendar day: net/gross session return of the leg."""
    local = pair_df["ts"].dt.tz_convert(ZoneInfo(tz))
    d = pair_df.assign(local=local, day=local.dt.normalize())
    sh, sm = map(int, start.split(":"))
    eh, em = map(int, end.split(":"))
    if morning_only:
        end_off = pd.Timedelta(hours=sh, minutes=sm) + pd.Timedelta(hours=3)
    else:
        end_off = pd.Timedelta(hours=eh, minutes=em)
    start_off = pd.Timedelta(hours=sh, minutes=sm)
    t0 = d["day"] + start_off
    t1 = d["day"] + end_off
    in_win = (d["local"] >= t0) & (d["local"] <= t1)
    w = d[in_win]
    if w.empty:
        return pd.DataFrame()
    g = w.groupby("day")
    first = g.first()
    last = g.last()
    ok = ((first["local"] - (first.index + start_off)) <= SNAP_TOL) \
        & (((last.index + end_off) - last["local"]) <= SNAP_TOL)
    entry_px = first["mid"] + sign * first["half_spread"]
    exit_px = last["mid"] - sign * last["half_spread"]
    net = sign * (exit_px / entry_px - 1)
    gross = sign * (last["mid"] / first["mid"] - 1)
    out = pd.DataFrame({"net": net, "gross": gross,
                        "cost": gross - net}).loc[ok]
    out.index = out.index.tz_localize(None).normalize()
    return out


def battery(r: pd.Series) -> dict:
    cal = pd.date_range(r.index.min(), r.index.max(), freq="B")
    r = r.reindex(cal).fillna(0.0)
    if len(r) < 100 or r.std() == 0:
        return {"sharpe": None, "days": int(len(r))}
    eq = (1 + r).cumprod()
    yearly = r.groupby(r.index.year).sum()
    top5 = float(r.nlargest(5).sum() / r.sum()) if r.sum() > 0 else None
    ex10 = r.drop(r.nlargest(10).index)
    return {"sharpe": round(float(r.mean() / r.std() * 252 ** .5), 3),
            "ann_return_pct": round(float(r.mean() * 252 * 100), 3),
            "max_drawdown_pct": round(float((eq / eq.cummax() - 1).min() * 100), 2),
            "negative_years": int((yearly < 0).sum()), "years": int(len(yearly)),
            "top5_day_share": None if top5 is None else round(top5, 3),
            "sharpe_ex_top10": round(float(ex10.mean() / ex10.std() * 252 ** .5), 3),
            "mean_daily_bps": round(float(r.mean() * 1e4), 3),
            "days": int(len(r))}


def run_window(years, morning_only=False, gmo_cost=False) -> dict:
    legs_daily = {}
    for leg, pair, sign, tz, start, end in LEGS:
        df = load_pair(pair, years)
        if gmo_cost:
            if pair == "USD_JPY":
                df = df.assign(half_spread=0.001)          # 0.2銭 full -> 0.1銭 half
            else:
                df = df.assign(half_spread=df["half_spread"] * 0.6)
        legs_daily[leg] = leg_daily(df, sign, tz, start, end, morning_only)
    nets = pd.DataFrame({k: v["net"] for k, v in legs_daily.items() if len(v)})
    port = nets.mean(axis=1, skipna=True) * (nets.notna().sum(axis=1) / len(LEGS))
    res = {"portfolio": battery(port.dropna())}
    res["per_leg_diagnostic"] = {
        k: {"net": battery(v["net"]), "gross_mean_daily_bps":
            round(float(v["gross"].mean() * 1e4), 3),
            "cost_mean_daily_bps": round(float(v["cost"].mean() * 1e4), 3)}
        for k, v in legs_daily.items() if len(v)}
    return res


def judge(s: dict) -> dict:
    return {"net_sharpe_ge_1": bool((s.get("sharpe") or -9) >= 1.0),
            "neg_years_le_third": bool(s.get("years", 0) > 0
                                       and s.get("negative_years", 9) * 3 <= s["years"]),
            "top5_share_lt_20pct": bool((s.get("top5_day_share") or 9) < .20),
            "sharpe_ex_top10_ge_05": bool((s.get("sharpe_ex_top10") or -9) >= .5)}


def main() -> None:
    sel = run_window(SEL_YEARS)
    crit = judge(sel["portfolio"])
    summary = {"spec": "docs/PREREGISTER_FX_SESSION.md",
               "S_primary_selection": sel, "selection_criteria": crit,
               "sensitivity_selection": {
                   "a_morning_3h": run_window(SEL_YEARS, morning_only=True)["portfolio"],
                   "b_gmo_cost": run_window(SEL_YEARS, gmo_cost=True)["portfolio"]}}
    if all(crit.values()):
        summary["confirmation"] = run_window(CONF_YEARS)
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
