#!/usr/bin/env python3
"""FX residual 3-cell sweep: weekend gap fade / Wednesday triple-swap rollover /
time-series weekly reversal. Frozen in docs/PREREGISTER_FX_MICRO3.md.

Three hypotheses tested simultaneously (multiplicity disclosed). Selection
2011-2019; each cell's confirmation 2020+ opens only on a full selection pass.
"""
from __future__ import annotations

import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from scripts.experiment_fx_session import load_pair
from trading.fx.swap import load_short_rates, rollover_days_series

PAIRS = {"EUR_USD": ("EUR", "USD"), "GBP_USD": ("GBP", "USD"),
         "AUD_USD": ("AUD", "USD"), "NZD_USD": ("NZD", "USD"),
         "USD_JPY": ("USD", "JPY"), "USD_CHF": ("USD", "CHF"),
         "USD_CAD": ("USD", "CAD")}
SEL_YEARS = tuple(range(2011, 2020))
CONF_YEARS = tuple(range(2020, 2027))
NY = ZoneInfo("America/New_York")
LON = ZoneInfo("Europe/London")
OUT = Path("data/fx_micro3")


def battery(r: pd.Series) -> dict:
    if r.empty:
        return {"sharpe": None, "days": 0}
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


def judge(s: dict) -> dict:
    return {"net_sharpe_ge_1": bool((s.get("sharpe") or -9) >= 1.0),
            "neg_years_le_third": bool(s.get("years", 0) > 0
                                       and s.get("negative_years", 9) * 3 <= s["years"]),
            "top5_share_lt_20pct": bool((s.get("top5_day_share") or 9) < .20),
            "sharpe_ex_top10_ge_05": bool((s.get("sharpe_ex_top10") or -9) >= .5)}


# ---------- H1 weekend gap fade ----------

def h1_weekend_gap(data: dict) -> tuple[pd.Series, dict]:
    trades = []
    for pair, d in data.items():
        ny = d["ts"].dt.tz_convert(NY)
        dd = d.assign(ny=ny, dow=ny.dt.weekday, week=ny.dt.normalize()
                      - pd.to_timedelta(ny.dt.weekday, unit="D"))
        fri_cut = dd["week"] + pd.Timedelta(days=4, hours=17)
        sun_open = dd["week"] + pd.Timedelta(days=6, hours=17)
        fri = dd[(dd["ny"] <= fri_cut)].groupby("week").last()
        fri_ok = (fri_cut.groupby(dd["week"]).first() - fri["ny"]) \
            <= pd.Timedelta(hours=2)
        sun = dd[(dd["ny"] >= sun_open)].groupby("week").first()
        sun_ok = (sun["ny"] - sun_open.groupby(dd["week"]).first()) \
            <= pd.Timedelta(hours=2)
        # ★2026-08-08 fix: Monday-noon bars belong to the NEXT week's anchor.
        # The original `week + 7d12h` cutoff selected the week's own last bar
        # (= next Sunday evening) => an accidental ~1-week hold. Take Monday
        # noon from the following week's group and shift its anchor back.
        mon_cut = dd["week"] + pd.Timedelta(hours=12)
        mon = dd[dd["ny"] <= mon_cut].groupby("week").last()
        mon_ok = (mon_cut.groupby(dd["week"]).first() - mon["ny"]) \
            <= pd.Timedelta(hours=2)
        mon = mon.where(mon_ok)
        j = fri[["mid"]].where(fri_ok).rename(columns={"mid": "fri_mid"}).join(
            sun[["mid", "half_spread", "ny"]].where(sun_ok).rename(
                columns={"mid": "sun_mid", "half_spread": "sun_hs"}), how="inner")
        nxt = mon[["mid", "half_spread"]].rename(
            columns={"mid": "mon_mid", "half_spread": "mon_hs"})
        nxt.index = nxt.index - pd.Timedelta(days=7)     # exit Monday = next week row
        j = j.join(nxt, how="inner").dropna()
        j["gap"] = j["sun_mid"] / j["fri_mid"] - 1
        thr = 2 * j["sun_hs"] / j["sun_mid"]
        fire = (j["gap"].abs() > thr) & (j["gap"].abs() <= .03)
        j = j[fire]
        sign = -np.sign(j["gap"])
        entry = j["sun_mid"] + sign * j["sun_hs"]
        exit_ = j["mon_mid"] - sign * j["mon_hs"]
        ret = sign * (exit_ / entry - 1)
        trades.append(pd.DataFrame({
            "date": (j.index + pd.Timedelta(days=7)).tz_localize(None),
            "ret": ret.to_numpy() / len(PAIRS), "pair": pair}))
    t = pd.concat(trades)
    daily = t.groupby("date")["ret"].sum()
    diag = {"trades": int(len(t)), "trades_per_year": round(len(t) / 9, 1),
            "hit_rate": round(float((t["ret"] > 0).mean()), 3),
            "mean_bps_per_trade": round(float(t["ret"].mean() * len(PAIRS) * 1e4), 3)}
    return daily, diag


# ---------- H2 Wednesday triple-swap rollover ----------

def h2_rollover(data: dict, rates: pd.DataFrame) -> tuple[pd.Series, dict]:
    lag = rates.shift(1)                                  # 1-month PIT lag for direction
    trades, ctrl = [], []
    for pair, d in data.items():
        base, quote = PAIRS[pair]
        ny = d["ts"].dt.tz_convert(NY)
        dd = d.assign(ny=ny, day=ny.dt.normalize(), dow=ny.dt.weekday)
        t_in = dd["day"] + pd.Timedelta(hours=16, minutes=30)
        t_out = dd["day"] + pd.Timedelta(hours=17, minutes=30)
        ent = dd[dd["ny"] <= t_in].groupby("day").last()
        exi = dd[dd["ny"] >= t_out].groupby("day").first()
        j = ent[["mid", "half_spread", "dow"]].rename(
            columns={"mid": "in_mid", "half_spread": "in_hs"}).join(
            exi[["mid", "half_spread"]].rename(
                columns={"mid": "out_mid", "half_spread": "out_hs"}),
            how="inner").dropna()
        ridx = pd.DatetimeIndex(j.index.tz_localize(None))
        diff_lag = (lag[base] - lag[quote]).reindex(
            ridx.union(lag.index)).ffill().reindex(ridx).to_numpy()
        diff_now = (rates[base] - rates[quote]).reindex(
            ridx.union(rates.index)).ffill().reindex(ridx).to_numpy()
        days = rollover_days_series(ridx).to_numpy()
        sign = np.sign(diff_lag)
        elig = np.abs(diff_lag) >= .01
        px_ret = sign * (j["out_mid"].to_numpy() / j["in_mid"].to_numpy() - 1)
        cost = ((j["in_hs"] + j["out_hs"]) / j["in_mid"]).to_numpy()
        swap = sign * diff_now * days / 365.0
        net = px_ret + swap - cost
        f = pd.DataFrame({"date": ridx, "dow": j["dow"].to_numpy(),
                          "net": net, "elig": elig})
        wed = f[(f["dow"] == 2) & f["elig"]]
        oth = f[(f["dow"].isin([0, 1, 3])) & f["elig"]]
        trades.append(pd.DataFrame({"date": wed["date"],
                                    "ret": wed["net"].to_numpy() / len(PAIRS)}))
        ctrl.append(oth["net"])
    t = pd.concat(trades)
    daily = t.groupby("date")["ret"].sum()
    co = pd.concat(ctrl)
    diag = {"trades": int(len(t)),
            "mean_bps_per_trade": round(float(t["ret"].mean() * len(PAIRS) * 1e4), 3),
            "control_mon_tue_thu_mean_bps": round(float(co.mean() * 1e4), 3),
            "control_n": int(len(co))}
    return daily, diag


# ---------- H3 time-series weekly reversal ----------

def h3_ts_reversal(data: dict, rates: pd.DataFrame) -> tuple[pd.Series, dict]:
    legs, switch_count = [], 0
    for pair, d in data.items():
        base, quote = PAIRS[pair]
        lon = d["ts"].dt.tz_convert(LON)
        dd = d.assign(lon=lon, day=lon.dt.normalize(), dow=lon.dt.weekday)
        mon = dd[(dd["dow"] == 0) & (dd["lon"] >= dd["day"] + pd.Timedelta(hours=8))]
        wk = mon.groupby("day").first()
        wk_ret = wk["mid"].pct_change()
        sigma = wk_ret.rolling(52, min_periods=40).std()
        z = wk_ret / sigma
        pos = np.where(z.abs() >= 1, -np.sign(z), 0.0)    # decided at week start
        pos = pd.Series(pos, index=wk.index)
        # weekly leg return: hold pos from this Monday to next Monday
        nxt_mid = wk["mid"].shift(-1)
        px_ret = pos * (nxt_mid / wk["mid"] - 1)
        turns = pos.diff().abs().fillna(pos.abs())
        cost = turns * (wk["half_spread"] / wk["mid"])
        # swap accrued over the held week
        ridx = pd.DatetimeIndex(wk.index.tz_localize(None))
        diff = (rates[base] - rates[quote]).reindex(
            ridx.union(rates.index)).ffill().reindex(ridx)
        swap = pos.to_numpy() * diff.to_numpy() * 7 / 365.0
        net = (px_ret - cost + swap).dropna()
        net.index = pd.DatetimeIndex(net.index.tz_localize(None)) + pd.Timedelta(days=7)
        legs.append(net / len(PAIRS))
        switch_count += int((turns > 0).sum())
    daily = pd.concat(legs).groupby(level=0).sum()
    active = pd.concat([l[l != 0] for l in legs])
    diag = {"weeks_active_leg_count": int(len(active)), "position_switches": switch_count,
            "mean_bps_per_active_leg_week": round(float(active.mean() * len(PAIRS) * 1e4), 3)}
    return daily, diag


def run(years) -> dict:
    data = {p: load_pair(p, years) for p in PAIRS}
    rates = load_short_rates()
    out = {}
    for name, fn in (("H1_weekend_gap", lambda: h1_weekend_gap(data)),
                     ("H2_wed_rollover", lambda: h2_rollover(data, rates)),
                     ("H3_ts_weekly_reversal", lambda: h3_ts_reversal(data, rates))):
        daily, diag = fn()
        out[name] = {"stats": battery(daily), "diag": diag}
    return out


def main() -> None:
    sel = run(SEL_YEARS)
    summary = {"spec": "docs/PREREGISTER_FX_MICRO3.md", "selection": sel}
    decisions = {}
    for cell, res in sel.items():
        crit = judge(res["stats"])
        res["criteria"] = crit
        decisions[cell] = "PASS_SELECTION" if all(crit.values()) else "NO_GO_AT_SELECTION"
    if any(v == "PASS_SELECTION" for v in decisions.values()):
        conf = run(CONF_YEARS)
        summary["confirmation"] = {c: conf[c] for c, v in decisions.items()
                                   if v == "PASS_SELECTION"}
    else:
        summary["confirmation"] = "UNOPENED"
    summary["decisions"] = decisions
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
