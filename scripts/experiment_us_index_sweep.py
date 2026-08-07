#!/usr/bin/env python3
"""US index CFD sweep: intraday momentum / TOM / overnight drift.

Frozen in docs/PREREGISTER_US_INDEX_SWEEP.md. Chosen for operability: the OANDA
account can execute these today, and the intraday cell avoids CFD financing
entirely. Costs are the measured bid/ask at the executed minute; financing for
overnight-crossing cells is (USD 3M interbank + 2.5%)/365 per night from FRED.
All timestamps converted UTC -> America/New_York with proper DST.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

NY = ZoneInfo("America/New_York")
SEL_MIN, CONF = ("2011-01-01", "2019-12-31"), ("2020-01-01", "2026-12-31")
SEL_DAY = ("2005-01-01", "2019-12-31")
R1_GATE = 5e-4
OUT = Path("data/us_index_sweep")


def load_minute(inst: str) -> pd.DataFrame:
    fs = sorted(f for f in glob.glob(f"data/fx_oanda_us/parts/{inst}_*.parquet")
                if not f.endswith("_D.parquet"))
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    d["ts"] = pd.to_datetime(d["ts"], utc=True).dt.tz_convert(NY)
    d["day"] = d["ts"].dt.date
    d["hm"] = d["ts"].dt.strftime("%H:%M")
    return d


def load_daily(inst: str) -> pd.DataFrame:
    fs = sorted(glob.glob(f"data/fx_oanda_us/parts/{inst}_*_D.parquet"))
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    d["ts"] = pd.to_datetime(d["ts"], utc=True)
    return d.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)


def usd_rate(idx: pd.DatetimeIndex) -> pd.Series:
    r = pd.read_parquet("data/fx_rates/short_rates_monthly.parquet")["USD"]
    r.index = pd.to_datetime(r.index)
    return r.reindex(r.index.union(idx)).ffill().reindex(idx)


def stats(net: pd.Series, lo: str, hi: str, ep: pd.Series | None = None) -> dict:
    r = net.loc[lo:hi]
    if len(r) < 100 or r.std() == 0:
        return {"sharpe": None, "days": int(len(r))}
    e = (ep.loc[lo:hi] if ep is not None else r[r != 0])
    pos = e[e > 0].sum()
    ex10 = r.drop(e.nlargest(min(10, len(e))).index, errors="ignore")
    by = r.groupby(r.index.year).sum()
    return {"days": int(len(r)), "n": int((r != 0).sum()),
            "sharpe": round(float(r.mean() / r.std() * 252 ** .5), 3),
            "ann_return": round(float(r.mean() * 252), 4),
            "top5_share": round(float(e.nlargest(5).sum() / pos), 4) if pos > 0 else None,
            "sharpe_ex_top10": round(float(ex10.mean() / ex10.std() * 252 ** .5), 3)
            if ex10.std() else None,
            "negative_years": int((by < 0).sum()), "years": int(len(by)),
            "by_year_pct": {int(k): round(float(v) * 100, 2) for k, v in by.items()}}


def passes(w: dict) -> list[str]:
    f = []
    if (w.get("sharpe") or -9) < 1.0:
        f.append("sharpe_lt_1.0")
    if w.get("negative_years", 9) > max(1, w.get("years", 1) // 3):
        f.append("too_many_negative_years")
    if (w.get("top5_share") or 1) >= .20:
        f.append("top5_share_ge_20pct")
    if (w.get("sharpe_ex_top10") or -9) < 0.5:
        f.append("sharpe_ex_top10_lt_0.5")
    return f


def snap(d: pd.DataFrame, hm: str) -> pd.DataFrame:
    """Last bar at or before hm per day (columns: mid/bid/ask)."""
    s = d[d["hm"] <= hm].groupby("day").tail(1)
    return s.set_index("day")[["close", "close_bid", "close_ask"]]


def h1_intraday_momentum(inst: str) -> pd.Series:
    d = load_minute(inst)
    rth = d[(d["hm"] >= "09:25") & (d["hm"] <= "16:05")]
    c16 = snap(rth, "16:00")
    c10 = snap(rth, "10:00")
    e1530 = snap(rth, "15:30")
    days = sorted(set(c16.index) & set(c10.index) & set(e1530.index))
    rows = {}
    prev = None
    for day in days:
        if prev is not None:
            r1 = c10.loc[day, "close"] / c16.loc[prev, "close"] - 1.0
            if np.isfinite(r1) and abs(r1) >= R1_GATE:
                sgn = 1.0 if r1 > 0 else -1.0
                if sgn > 0:
                    entry, exitp = e1530.loc[day, "close_ask"], c16.loc[day, "close_bid"]
                else:
                    entry, exitp = e1530.loc[day, "close_bid"], c16.loc[day, "close_ask"]
                rows[pd.Timestamp(day)] = sgn * (exitp / entry - 1.0)
        prev = day
    tr = pd.Series(rows)
    cal = pd.date_range(tr.index.min(), tr.index.max())
    return tr.reindex(cal).fillna(0.0)


def h3_overnight(inst: str) -> pd.Series:
    d = load_minute(inst)
    rth = d[(d["hm"] >= "09:25") & (d["hm"] <= "16:05")]
    c16 = snap(rth, "16:00")
    o930 = snap(rth, "09:31")
    days = sorted(set(c16.index) & set(o930.index))
    fin = usd_rate(pd.DatetimeIndex([pd.Timestamp(x) for x in days]))
    rows = {}
    for i in range(len(days) - 1):
        d0, d1 = days[i], days[i + 1]
        nights = (pd.Timestamp(d1) - pd.Timestamp(d0)).days
        cost_fin = (float(fin.iloc[i]) + .025) / 365 * nights
        entry = c16.loc[d0, "close_ask"]
        exitp = o930.loc[d1, "close_bid"]
        rows[pd.Timestamp(d1)] = exitp / entry - 1.0 - cost_fin
    tr = pd.Series(rows)
    cal = pd.date_range(tr.index.min(), tr.index.max())
    return tr.reindex(cal).fillna(0.0)


def h2_tom(inst: str) -> tuple[pd.Series, pd.Series]:
    d = load_daily(inst)
    d["day"] = d["ts"].dt.normalize().dt.tz_localize(None)
    d = d.drop_duplicates("day").set_index("day")
    days = d.index
    month = days.month
    is_last = pd.Series(month != np.roll(month, -1), index=days)
    is_last.iloc[-1] = False
    fin = usd_rate(days)
    ep_rows, daily_rows = {}, {}
    for i in range(len(days) - 4):
        if not is_last.iloc[i]:
            continue
        entry_ask, exit_bid = d["close_ask"].iloc[i], d["close_bid"].iloc[i + 3]
        nights = (days[i + 3] - days[i]).days
        net = exit_bid / entry_ask - 1.0 - (float(fin.iloc[i]) + .025) / 365 * nights
        ep_rows[days[i + 3]] = net
        mid_sum = 0.0
        for k in range(1, 4):
            r = d["close"].iloc[i + k] / d["close"].iloc[i + k - 1] - 1.0
            daily_rows[days[i + k]] = daily_rows.get(days[i + k], 0.0) + r
            mid_sum += r
        # スプレッド+金利ぶんはエグジット日に減算（日次分布は近似・集中判定はエピソードで）
        daily_rows[days[i + 3]] -= (mid_sum - net)
    ep = pd.Series(ep_rows)
    daily = pd.Series(daily_rows)
    cal = pd.date_range(daily.index.min(), daily.index.max())
    return daily.reindex(cal).fillna(0.0), ep


def h4_drawdown_entry(inst: str, dd_th: float = .15, max_hold: int = 250,
                      cost_rt: float = 4e-4) -> tuple[pd.Series, pd.Series, pd.Series]:
    """ATH−15%で買い、新高値回復 or 250営業日で手仕舞い（現物ETF前提・金利なし）."""
    d = load_daily(inst)
    d["day"] = d["ts"].dt.normalize().dt.tz_localize(None)
    d = d.drop_duplicates("day").set_index("day")
    px = d["close"]
    ath = px.cummax()
    daily = pd.Series(0.0, index=px.index)
    ep_rows = {}
    in_pos, entry_i = False, None
    for i in range(1, len(px)):
        if in_pos:
            daily.iloc[i] += px.iloc[i] / px.iloc[i - 1] - 1.0
            if px.iloc[i] >= ath.iloc[entry_i] or i - entry_i >= max_hold:
                ep_rows[px.index[i]] = px.iloc[i] / px.iloc[entry_i] - 1.0 - cost_rt
                daily.iloc[i] -= cost_rt
                in_pos = False
        else:
            if px.iloc[i] <= ath.iloc[i] * (1 - dd_th) and                px.iloc[i - 1] > ath.iloc[i - 1] * (1 - dd_th):
                in_pos, entry_i = True, i
    bh = px.pct_change().fillna(0.0)
    cal = pd.date_range(px.index.min(), px.index.max())
    return (daily.reindex(cal).fillna(0.0), pd.Series(ep_rows),
            bh.reindex(cal).fillna(0.0))


def main() -> None:
    out = {"note": "3 preregistered cells; SPX500 primary, NAS100 secondary (no promotion)"}
    for inst in ("SPX500_USD", "NAS100_USD"):
        r = {}
        h1 = h1_intraday_momentum(inst)
        e = {"selection": stats(h1, *SEL_MIN)}
        e["selection_failed"] = passes(e["selection"])
        if not e["selection_failed"]:
            e["confirmation"] = stats(h1, *CONF)
            e["confirmation_failed"] = passes(e["confirmation"])
            e["decision"] = "NO_GO" if e["confirmation_failed"] else "PENDING_OPERATIONAL_REVIEW"
        else:
            e["confirmation"] = "NOT_OPENED"
            e["decision"] = "NO_GO"
        r["H1_intraday_momentum"] = e

        h3 = h3_overnight(inst)
        e3 = {"selection": stats(h3, *SEL_MIN)}
        e3["selection_failed"] = passes(e3["selection"])
        if not e3["selection_failed"]:
            e3["confirmation"] = stats(h3, *CONF)
            e3["confirmation_failed"] = passes(e3["confirmation"])
            e3["decision"] = "NO_GO" if e3["confirmation_failed"] else "PENDING_OPERATIONAL_REVIEW"
        else:
            e3["confirmation"] = "NOT_OPENED"
            e3["decision"] = "NO_GO"
        r["H3_overnight_net_financing"] = e3

        dser, ep = h2_tom(inst)
        e2 = {"selection": stats(dser, *SEL_DAY, ep=ep),
              "episode_mean_net_bps": round(float(ep.mean() * 1e4), 1),
              "episodes": int(len(ep))}
        e2["selection_failed"] = passes(e2["selection"])
        if not e2["selection_failed"]:
            e2["confirmation"] = stats(dser, *CONF, ep=ep)
            e2["confirmation_failed"] = passes(e2["confirmation"])
            e2["decision"] = "NO_GO" if e2["confirmation_failed"] else "PENDING_OPERATIONAL_REVIEW"
        else:
            e2["confirmation"] = "NOT_OPENED"
            e2["decision"] = "NO_GO"
        r["H2_turn_of_month"] = e2

        h4, ep4, bh = h4_drawdown_entry(inst)
        s4 = stats(h4, "2005-01-01", "2026-12-31", ep=ep4)
        sbh = stats(bh, "2005-01-01", "2026-12-31")
        top1 = (float(ep4.max() / ep4[ep4 > 0].sum()) if len(ep4) and ep4[ep4 > 0].sum() > 0
                else None)
        f4 = []
        if (s4.get("sharpe") or -9) < (sbh.get("sharpe") or 9):
            f4.append("sharpe_below_buy_and_hold")
        eq_s = (1 + h4).cumprod(); eq_b = (1 + bh).cumprod()
        dd_s = float((eq_s / eq_s.cummax() - 1).min())
        dd_b = float((eq_b / eq_b.cummax() - 1).min())
        if dd_s <= dd_b:
            f4.append("drawdown_not_shallower")
        if len(ep4) < 6:
            f4.append("episodes_lt_6")
        if top1 is not None and top1 >= .50:
            f4.append("top1_share_ge_50pct")
        r["H4_drawdown_entry_longterm"] = {
            "strategy": s4 | {"max_dd": round(dd_s, 4), "episodes": int(len(ep4)),
                              "episode_mean_pct": round(float(ep4.mean() * 100), 2)
                              if len(ep4) else None, "top1_share": top1},
            "buy_and_hold": sbh | {"max_dd": round(dd_b, 4)},
            "failed": f4,
            "decision": "NO_GO" if f4 else "PROVISIONAL_PASS_LOW_N"}
        out[inst] = r
        print(inst, json.dumps({k: v["decision"] for k, v in r.items()}), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str),
                                      encoding="utf-8")
    print("saved")


if __name__ == "__main__":
    main()
