#!/usr/bin/env python3
"""Multi-currency L/S grid (trap-repeat style) + variance-ratio mechanism diagnostic.

Frozen in docs/PREREGISTER_FX_GRID.md. Two deliverables:
  D) VR(q) per pair -- is there any mean-reversion premium at grid scales?
  S) Frozen grid sim: annual anchor, dx=0.5*median daily range (prior year, PIT),
     TRUE hysteresis grid (buy limit at line k<0, TP one line up; short mirror
     above the anchor), +/-10 lines per side, fills at line prices, measured
     half-spread per fill, daily swap on net inventory (interbank, passthrough
     1.0, haircut 0 = broker-best; makes hedged inventory swap-free, generous).

A state-function grid (position = f(price)) degenerates to buying and selling
at the same boundary; real grids carry hysteresis, so the sim is event-driven.
Selection 2012-2019; confirmation 2020+ opened only if all criteria pass.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from trading.fx.swap import load_short_rates, rollover_days_series

PAIRS = {  # name -> (base, quote)
    "EUR_USD": ("EUR", "USD"), "GBP_USD": ("GBP", "USD"),
    "AUD_USD": ("AUD", "USD"), "NZD_USD": ("NZD", "USD"),
    "USD_JPY": ("USD", "JPY"), "USD_CHF": ("USD", "CHF"),
    "USD_CAD": ("USD", "CAD"),
}
SEL_YEARS = tuple(range(2012, 2020))
CONF_YEARS = tuple(range(2020, 2027))
CAP = 10
CAPITAL = 1.0                      # USD; unit notional = CAPITAL/(7*CAP)
OUT = Path("data/fx_grid")


@lru_cache(maxsize=64)
def load_minute(pair: str, year: int) -> pd.DataFrame | None:
    f = Path(f"data/fx_oanda_min/parts/{pair}_{year}.parquet")
    if not f.exists():
        return None
    d = pd.read_parquet(f)
    d["mid"] = (d["close_bid"] + d["close_ask"]) / 2
    hs = ((d["close_ask"] - d["close_bid"]).clip(lower=0) / 2)
    d["half_spread"] = hs.clip(upper=hs.quantile(.99))
    return d[["ts", "open", "high", "low", "mid", "half_spread"]].reset_index(drop=True)


@lru_cache(maxsize=64)
def grid_spacing(pair: str, year: int) -> float | None:
    prev = load_minute(pair, year - 1)
    if prev is None or prev.empty:
        return None
    day = prev.set_index("ts")["mid"].resample("1D")
    rng = (day.max() - day.min()).dropna()
    if len(rng) < 100:
        return None
    return float(rng.median()) * 0.5


def simulate_pair_year(pair: str, year: int, dx_scale: float, carry_only: bool,
                       cost_scale: float, rates: pd.DataFrame) -> pd.DataFrame:
    """Event-driven hysteresis grid; returns daily USD P&L components."""
    d = load_minute(pair, year)
    dx0 = grid_spacing(pair, year)
    if d is None or d.empty or dx0 is None:
        return pd.DataFrame()
    dx = dx0 * dx_scale / 0.5
    base, quote = PAIRS[pair]
    anchor = float(d["mid"].iloc[0])
    unit = CAPITAL / (len(PAIRS) * CAP)
    units_base = unit / anchor if quote == "USD" else unit

    mids = d["mid"].to_numpy()
    opens = d["open"].to_numpy()
    highs = d["high"].to_numpy()
    lows = d["low"].to_numpy()
    hs = d["half_spread"].to_numpy() * cost_scale
    band = np.floor((mids - anchor) / dx).astype(np.int64)
    band_o = np.floor((opens - anchor) / dx).astype(np.int64)
    band_h = np.floor((highs - anchor) / dx).astype(np.int64)
    band_l = np.floor((lows - anchor) / dx).astype(np.int64)
    prev_bc = np.concatenate(([0], band[:-1]))
    # touch-based detection: any bar whose H/L range or open gap crosses a line.
    # (close-only detection conditions entries on adverse continuation = biased;
    #  limit orders fill on touch, so use the OHLC monotone 3-leg path.)
    ev_mask = (band_h != band_l) | (band_o != prev_bc) | (band != prev_bc)
    events = np.nonzero(ev_mask)[0]

    # carry gate: which side may accumulate (re-fixed monthly, PIT)
    month_key = d["ts"].dt.tz_localize(None).dt.to_period("M").to_numpy()
    if carry_only:
        rr = (rates[base] - rates[quote]).dropna()
        month_sign = {}
        for mth in pd.unique(month_key):
            prior = rr[rr.index < mth.to_timestamp()]
            month_sign[mth] = 1 if (len(prior) and prior.iloc[-1] >= 0) else -1

    hold_long = np.zeros(CAP + 1, dtype=bool)          # index -k for line k in [-CAP,-1]
    hold_short = np.zeros(CAP + 1, dtype=bool)         # index k for line k in [1,CAP]
    line_px = lambda j: anchor + j * dx
    fills = 0
    pos = 0
    cash_path = np.zeros(len(mids))
    ev_i, ev_pos = [], []
    prev_b = 0                                          # first mid == anchor -> band 0

    def process_move(b_from: int, b_to: int, i: int, allow_long: bool,
                     allow_short: bool) -> float:
        """Process all line crossings of one monotone leg; returns cash delta."""
        nonlocal pos, fills
        delta = 0.0
        if b_to < b_from:   # down through boundaries b_from, ..., b_to+1
            for j in range(b_from, b_to, -1):
                if -CAP <= j <= -1 and not hold_long[-j] and allow_long:
                    delta -= units_base * (line_px(j) + hs[i])   # buy entry
                    hold_long[-j] = True; pos += 1; fills += 1
                k = j + 1                              # short line k TPs at line k-1 == j
                if 1 <= k <= CAP and hold_short[k]:
                    delta -= units_base * (line_px(j) + hs[i])   # buy back
                    hold_short[k] = False; pos += 1; fills += 1
        elif b_to > b_from:  # up through boundaries b_from+1, ..., b_to
            for j in range(b_from + 1, b_to + 1):
                if 1 <= j <= CAP and not hold_short[j] and allow_short:
                    delta += units_base * (line_px(j) - hs[i])   # short entry
                    hold_short[j] = True; pos -= 1; fills += 1
                k = j - 1                              # long line k TPs at line k+1 == j
                if -CAP <= k <= -1 and hold_long[-k]:
                    delta += units_base * (line_px(j) - hs[i])   # sell TP
                    hold_long[-k] = False; pos -= 1; fills += 1
        return delta

    for i in events:
        allow_long = (not carry_only) or month_sign[month_key[i]] > 0
        allow_short = (not carry_only) or month_sign[month_key[i]] < 0
        up_bar = mids[i] >= opens[i]
        legs = ([band_o[i], band_l[i], band_h[i], band[i]] if up_bar
                else [band_o[i], band_h[i], band_l[i], band[i]])
        delta = 0.0
        cur = prev_b
        for nxt in legs:
            delta += process_move(cur, nxt, i, allow_long, allow_short)
            cur = nxt
        cash_path[i] = delta
        ev_i.append(i); ev_pos.append(pos)
        prev_b = band[i]

    cash = np.cumsum(cash_path)
    if ev_i:
        loc = np.searchsorted(np.asarray(ev_i), np.arange(len(mids)), side="right")
        pos_path = np.concatenate(([0], np.asarray(ev_pos)))[loc]
    else:
        pos_path = np.zeros(len(mids), dtype=np.int64)

    eq_q = cash + pos_path * units_base * mids
    day_key = d["ts"].dt.normalize()
    eq_daily = pd.Series(eq_q, index=day_key).groupby(level=0).last()
    mid_daily = pd.Series(mids, index=day_key).groupby(level=0).last()
    pos_daily = pd.Series(pos_path, index=day_key).groupby(level=0).last()
    pnl_q = eq_daily.diff()
    pnl_q.iloc[0] = eq_daily.iloc[0]
    ridx = pd.DatetimeIndex(eq_daily.index.tz_localize(None))
    rb = rates[base].reindex(ridx.union(rates.index)).ffill().reindex(ridx)
    rq = rates[quote].reindex(ridx.union(rates.index)).ffill().reindex(ridx)
    days = rollover_days_series(ridx)
    swap_q = (pos_daily.to_numpy() * units_base * mid_daily.to_numpy()
              * (rb.to_numpy() - rq.to_numpy()) * days.to_numpy() / 365.0)
    to_usd = np.ones(len(ridx)) if quote == "USD" else 1.0 / mid_daily.to_numpy()
    out = pd.DataFrame({"date": ridx, "pnl_usd": pnl_q.to_numpy() * to_usd,
                        "swap_usd": swap_q * to_usd,
                        "inv_units": np.abs(pos_daily.to_numpy())})
    out["fills"] = 0
    if len(out):
        out.iloc[0, out.columns.get_loc("fills")] = fills
    return out


def portfolio(years, dx_scale=0.5, carry_only=False, cost_scale=1.0) -> dict:
    rates = load_short_rates()
    parts = []
    for pair in PAIRS:
        for y in years:
            r = simulate_pair_year(pair, y, dx_scale, carry_only, cost_scale, rates)
            if len(r):
                parts.append(r)
    if not parts:
        return {"sharpe": None}
    allp = pd.concat(parts)
    daily = allp.groupby("date")[["pnl_usd", "swap_usd", "inv_units"]].sum()
    r = (daily["pnl_usd"] + daily["swap_usd"]) / CAPITAL
    cal = pd.date_range(r.index.min(), r.index.max(), freq="B")
    r = r.reindex(cal).fillna(0.0)
    if r.std() == 0:
        return {"sharpe": None, "days": int(len(r))}
    eq = (1 + r).cumprod()
    yearly = r.groupby(r.index.year).sum()
    top5 = float(r.nlargest(5).sum() / r.sum()) if r.sum() > 0 else None
    ex10 = r.drop(r.nlargest(10).index)
    net_total = float(r.sum())
    return {"sharpe": round(float(r.mean() / r.std() * 252 ** .5), 3),
            "ann_return_pct": round(float(r.mean() * 252 * 100), 3),
            "max_drawdown_pct": round(float((eq / eq.cummax() - 1).min() * 100), 2),
            "negative_years": int((yearly < 0).sum()), "years": int(len(yearly)),
            "top5_day_share": None if top5 is None else round(top5, 3),
            "sharpe_ex_top10": round(float(ex10.mean() / ex10.std() * 252 ** .5), 3),
            "worst5_days_sum_pct": round(float(r.nsmallest(5).sum() * 100), 4),
            "es5_daily_pct": round(float(r[r <= r.quantile(.05)].mean() * 100), 4),
            "gross_usd": round(float(daily["pnl_usd"].sum()), 5),
            "swap_usd": round(float(daily["swap_usd"].sum()), 5),
            "net_total_pct_of_capital": round(net_total * 100, 3),
            "mean_abs_inventory_units": round(float(daily["inv_units"].mean()), 2),
            "max_inventory_units": int(daily["inv_units"].max()),
            "total_fills": int(allp["fills"].sum()),
            "days": int(len(r))}


def variance_ratios() -> dict:
    """VR(q) on selection years only. VR<1 = mean reversion a grid could harvest."""
    out = {}
    for pair in PAIRS:
        parts = [load_minute(pair, y) for y in SEL_YEARS]
        parts = [p for p in parts if p is not None and len(p)]
        if not parts:
            continue
        m = pd.concat(parts).set_index("ts")["mid"]
        res = {}
        hourly = np.log(m.resample("1h").last().dropna())
        daily = np.log(m.resample("1D").last().dropna())
        for label, series, q in (("1h->1d", hourly, 24), ("1h->1w", hourly, 120),
                                 ("1h->1m", hourly, 480), ("1d->1w", daily, 5),
                                 ("1d->1m", daily, 20)):
            r1 = series.diff().dropna()
            rq = series.diff(q).dropna()[::q]          # non-overlapping
            if len(rq) < 30 or r1.var() == 0:
                continue
            res[label] = round(float(rq.var() / (q * r1.var())), 3)
        out[pair] = res
    med = {}
    for label in ("1h->1d", "1h->1w", "1h->1m", "1d->1w", "1d->1m"):
        vals = [v[label] for v in out.values() if label in v]
        if vals:
            med[label] = round(float(np.median(vals)), 3)
    out["median_across_pairs"] = med
    return out


def judge(s: dict) -> dict:
    return {"net_sharpe_ge_1": bool((s.get("sharpe") or -9) >= 1.0),
            "neg_years_le_third": bool(s.get("years", 0) > 0
                                       and s.get("negative_years", 9) * 3 <= s["years"]),
            "top5_share_lt_20pct": bool((s.get("top5_day_share") or 9) < .20),
            "sharpe_ex_top10_ge_05": bool((s.get("sharpe_ex_top10") or -9) >= .5)}


def main() -> None:
    summary = {"spec": "docs/PREREGISTER_FX_GRID.md",
               "diagnostic_variance_ratios_selection": variance_ratios()}
    sel = portfolio(SEL_YEARS)
    summary["S_primary_selection"] = sel
    crit = judge(sel)
    summary["selection_criteria"] = crit
    summary["sensitivity_selection"] = {
        "a_dx_1.0": portfolio(SEL_YEARS, dx_scale=1.0),
        "b_carry_aligned": portfolio(SEL_YEARS, carry_only=True),
        "c_half_cost": portfolio(SEL_YEARS, cost_scale=0.5)}
    if all(crit.values()):
        summary["confirmation"] = portfolio(CONF_YEARS)
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
