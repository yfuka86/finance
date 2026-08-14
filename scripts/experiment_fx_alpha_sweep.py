#!/usr/bin/env python3
"""Four independent FX factors, judged on both windows. See PREREGISTER_FX_ALPHA_SWEEP.md.

A dollar timing (LRV dollar carry), B long-term reversal (nominal 5y, ECB history),
C rate-change momentum, D weekly basket reversal. Carry levels and price momentum
are NOT retested — they are dead. Confirmation window opens only for hypotheses
that pass every selection-window criterion; the others' 2020+ stays unconsumed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from trading.fx.swap import load_short_rates, rollover_days_series

PAIRS = {"EUR": ("EURUSD", +1), "JPY": ("USDJPY", -1), "GBP": ("GBPUSD", +1),
         "CHF": ("USDCHF", -1), "AUD": ("AUDUSD", +1), "CAD": ("USDCAD", -1),
         "NZD": ("NZDUSD", +1)}
CURRENCIES = ["USD", *PAIRS]
N_SIDE = 2
SELECTION, CONFIRM = ("2011-01-01", "2019-12-31"), ("2020-01-01", "2025-12-31")
OUT = Path("data/fx_alpha_sweep")


def load_duka():
    mid, hs = {}, {}
    for ccy, (pair, _) in PAIRS.items():
        d = pd.read_parquet(f"data/fx_dukascopy/{pair}_day.parquet")
        d["date"] = pd.to_datetime(d["ts"]).dt.tz_localize(None).dt.normalize()
        d = d.drop_duplicates("date").set_index("date").sort_index()
        mid[pair], hs[pair] = d["mid"], d["spread"] / d["mid"] / 2.0
    return pd.DataFrame(mid).dropna(how="all"), pd.DataFrame(hs)


def duka_strengths(mid: pd.DataFrame) -> pd.DataFrame:
    s = pd.DataFrame(index=mid.index)
    s["USD"] = 0.0
    for ccy, (pair, sign) in PAIRS.items():
        s[ccy] = sign * np.log(mid[pair])
    return s[CURRENCIES]


def ecb_strengths() -> pd.DataFrame:
    """USD-based log strengths from ECB fixings, 1999+. Signal history only."""
    e = pd.read_parquet("data/fx_ecb/eurofxref_hist.parquet").set_index("Date")
    s = pd.DataFrame(index=e.index)
    s["EUR"] = 0.0
    for c in ("USD", "JPY", "GBP", "CHF", "AUD", "CAD", "NZD"):
        s[c] = -np.log(pd.to_numeric(e[c], errors="coerce"))
    s = s.sub(s["USD"], axis=0)          # USD 基準へ
    return s[CURRENCIES].dropna()


def top_bottom(sig: pd.Series) -> pd.Series:
    r = sig.dropna()
    w = pd.Series(0.0, index=CURRENCIES)
    if len(r) < 2 * N_SIDE:
        return w
    n_other = len(CURRENCIES) - 1
    for c in r.nlargest(N_SIDE).index:
        w[c] += 1 / N_SIDE
        w[[x for x in CURRENCIES if x != c]] -= (1 / N_SIDE) / n_other
    for c in r.nsmallest(N_SIDE).index:
        w[c] -= 1 / N_SIDE
        w[[x for x in CURRENCIES if x != c]] += (1 / N_SIDE) / n_other
    return w


def engine(mid, hs, rates_accr, weight_fn) -> pd.DataFrame:
    s = duka_strengths(mid)
    ds = s.diff()
    idx = s.index
    roll = rollover_days_series(idx)
    w = pd.Series(0.0, index=CURRENCIES)
    rows = []
    for i, day in enumerate(idx):
        if i == 0:
            continue
        cost = 0.0
        if day.weekday() == 4:                       # 金曜リバランス
            neww = weight_fn(day)
            if neww is not None:
                for ccy, (pair, _) in PAIRS.items():
                    dv = abs(neww[ccy] - w[ccy])
                    h = hs[pair].get(day, np.nan)
                    if np.isfinite(h):
                        cost += dv * h
                w = neww
        px = float((w * ds.loc[day]).sum())
        sw = float((w * rates_accr.loc[day]).sum()) * roll.loc[day] / 365.0
        rows.append({"date": day, "net": px + sw - cost})
    return pd.DataFrame(rows)


def stats(df, lo, hi) -> dict:
    r = df.set_index("date")["net"].loc[lo:hi]
    if len(r) < 100 or r.std() == 0:
        return {"days": int(len(r)), "sharpe": None}
    eq = (1 + r).cumprod()
    wk = r.resample("W").sum()
    pos = wk[wk > 0].sum()
    d10 = wk.drop(wk.nlargest(10).index)
    by = r.groupby(r.index.year).sum()
    return {"days": int(len(r)),
            "sharpe": round(float(r.mean() / r.std() * 252 ** .5), 3),
            "ann_return": round(float(r.mean() * 252), 4),
            "max_drawdown": round(float((eq / eq.cummax() - 1).min()), 4),
            "top5_week_share": round(float(wk.nlargest(5).sum() / pos), 4) if pos > 0 else None,
            "sharpe_ex_top10w": round(float(d10.mean() / d10.std() * 52 ** .5), 3)
            if d10.std() else None,
            "negative_years": int((by < 0).sum()), "years": int(len(by)),
            "by_year": {int(k): round(float(v), 4) for k, v in by.items()}}


def passes(w: dict) -> list[str]:
    f = []
    if (w.get("sharpe") or -9) < 1.0:
        f.append("sharpe_lt_1.0")
    if w.get("negative_years", 9) > max(1, w.get("years", 1) // 3):
        f.append("too_many_negative_years")
    if (w.get("top5_week_share") or 1) >= .20:
        f.append("top5_week_share_ge_20pct")
    if (w.get("sharpe_ex_top10w") or -9) < 0.5:
        f.append("sharpe_ex_top10w_lt_0.5")
    return f


def main() -> None:
    mid, hs = load_duka()
    idx = mid.index
    rates_accr = load_short_rates(index=idx)                       # 発生額: 現行値ffill
    monthly = load_short_rates()
    lag = monthly.copy()
    lag.index = lag.index + pd.offsets.MonthBegin(1)               # ★シグナルは1か月ラグ
    rates_sig = lag.reindex(lag.index.union(idx)).ffill().reindex(idx)
    ecb = ecb_strengths()
    ecb_dev = ecb - ecb.rolling(5 * 261, min_periods=3 * 261).mean()
    ecb_dev = ecb_dev.reindex(ecb_dev.index.union(idx)).ffill().reindex(idx)
    duka_s = duka_strengths(mid)
    wk_ret = duka_s.diff(5).shift(1)                               # 直近5営業日（t-1まで）

    def sig_A(day):
        r = rates_sig.loc[day]
        if r.isna().any():
            return None
        d = r["USD"] - r[[c for c in CURRENCIES if c != "USD"]].mean()
        w = pd.Series(0.0, index=CURRENCIES)
        sgn = 1.0 if d > 0 else -1.0
        w["USD"] = sgn
        for c in CURRENCIES:
            if c != "USD":
                w[c] = -sgn / (len(CURRENCIES) - 1)
        return w

    def sig_B(day):
        d = ecb_dev.loc[day]
        return top_bottom(-d) if d.notna().sum() >= 2 * N_SIDE + 1 else None

    def sig_C(day):
        r0, r3 = rates_sig.loc[day], None
        j = rates_sig.index.get_loc(day)
        if j < 63:
            return None
        r3 = rates_sig.iloc[j - 63]
        chg = r0 - r3
        return top_bottom(chg) if chg.notna().sum() >= 2 * N_SIDE + 1 else None

    def sig_D(day):
        r = wk_ret.loc[day] if day in wk_ret.index else None
        return top_bottom(-r) if r is not None and r.notna().sum() >= 2 * N_SIDE + 1 else None

    out = {"note": "4 hypotheses tested simultaneously — multiplicity acknowledged; "
                   "confirmation opened only for selection passes"}
    results = {}
    for label, fn in [("A_dollar_timing", sig_A), ("B_longterm_reversal", sig_B),
                      ("C_rate_momentum", sig_C), ("D_weekly_reversal", sig_D)]:
        d = engine(mid, hs, rates_accr, fn)
        sel = stats(d, *SELECTION)
        entry = {"selection": sel, "selection_failed": passes(sel)}
        if not entry["selection_failed"]:
            con = stats(d, *CONFIRM)
            entry["confirmation"] = con
            entry["confirmation_failed"] = passes(con)
            entry["decision"] = ("NO_GO" if entry["confirmation_failed"]
                                 else "PENDING_FULL_RISK_TESTS")
        else:
            entry["confirmation"] = "NOT_OPENED (selection failed; window preserved)"
            entry["decision"] = "NO_GO"
        results[label] = entry
        print(label, json.dumps(entry, ensure_ascii=False, default=str), flush=True)
    out["results"] = results
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str),
                                      encoding="utf-8")
    print(json.dumps({k: v["decision"] for k, v in results.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
