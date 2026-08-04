#!/usr/bin/env python3
"""FX currency-basket carry and momentum, with real spreads and modelled swap.

Frozen in docs/PREREGISTER_FX_BASKET.md.

The referenced article supplies the *expression* (hold a currency against all
seven others so the counter-currency noise averages out) but not a signal — its
headline result is a volatility reduction, which is mechanical, not alpha. The
signal is supplied here: cross-sectional carry as the primary, momentum as a
secondary that must not be promoted.

Everything is expressed in the seven USD pairs, because with USD as the numeraire
a currency weight *is* a position in that currency's USD pair — so turnover and
cost use the measured bid/ask directly rather than an approximation.
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
N_SIDE, MOM_WEEKS = 2, 12
SELECTION = ("2011-01-01", "2019-12-31")
CONFIRM = ("2020-01-01", "2099-12-31")
OUT = Path("data/fx_basket")


def load_prices() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mid price and relative half-spread per USD pair, indexed by date."""
    mid, hs = {}, {}
    for ccy, (pair, _) in PAIRS.items():
        d = pd.read_parquet(f"data/fx_dukascopy/{pair}_day.parquet")
        d["date"] = pd.to_datetime(d["ts"]).dt.tz_localize(None).dt.normalize()
        d = d.drop_duplicates("date").set_index("date").sort_index()
        mid[pair] = d["mid"]
        hs[pair] = (d["spread"] / d["mid"] / 2.0)      # 片道の相対コスト
    return pd.DataFrame(mid).dropna(how="all"), pd.DataFrame(hs)


def strengths(mid: pd.DataFrame) -> pd.DataFrame:
    """Log strength per currency with USD as the numeraire (USD ≡ 0)."""
    s = pd.DataFrame(index=mid.index)
    s["USD"] = 0.0
    for ccy, (pair, sign) in PAIRS.items():
        s[ccy] = sign * np.log(mid[pair])
    return s[CURRENCIES]


def basket_weights(rank_series: pd.Series) -> pd.Series:
    """Long the top N currencies' baskets, short the bottom N."""
    r = rank_series.dropna()
    if len(r) < 2 * N_SIDE:
        return pd.Series(0.0, index=CURRENCIES)
    top, bot = r.nlargest(N_SIDE).index, r.nsmallest(N_SIDE).index
    w = pd.Series(0.0, index=CURRENCIES)
    n_other = len(CURRENCIES) - 1
    for c in top:
        w[c] += 1 / N_SIDE
        w[[x for x in CURRENCIES if x != c]] -= (1 / N_SIDE) / n_other
    for c in bot:
        w[c] -= 1 / N_SIDE
        w[[x for x in CURRENCIES if x != c]] += (1 / N_SIDE) / n_other
    return w


def run(mid: pd.DataFrame, half_spread: pd.DataFrame, rates: pd.DataFrame,
        mode: str, haircut_bp: float = 0.0, hold_weeks: int = 1) -> pd.DataFrame:
    s = strengths(mid)
    ds = s.diff()
    idx = s.index
    roll = rollover_days_series(idx)
    r_daily = rates.reindex(idx).ffill()
    # 週次: 金曜にリバランス（該当日が無ければ直前営業日）
    fridays = [d for d in idx if d.weekday() == 4]
    fridays = fridays[::hold_weeks]
    bask = ds.sub(ds[CURRENCIES].mean(axis=1), axis=0) * len(CURRENCIES) / (len(CURRENCIES) - 1)
    w = pd.Series(0.0, index=CURRENCIES)
    rows, prev_w = [], w.copy()
    reb = set(fridays)
    for i, day in enumerate(idx):
        if i == 0:
            continue
        if day in reb:
            if mode == "carry":
                sig = r_daily.loc[day]
            else:
                past = bask.loc[:day].tail(MOM_WEEKS * 5)
                sig = past.sum() if len(past) >= MOM_WEEKS * 5 else pd.Series(dtype=float)
            neww = basket_weights(sig) if len(sig.dropna()) == len(CURRENCIES) else w
            # コストは USD ペアの建玉変化 × 実測ハーフスプレッド
            cost = 0.0
            for ccy, (pair, _) in PAIRS.items():
                dv = abs(neww.get(ccy, 0.0) - w.get(ccy, 0.0))
                hsv = half_spread[pair].get(day, np.nan)
                if np.isfinite(hsv):
                    cost += dv * hsv
            w = neww
        else:
            cost = 0.0
        px = float((w * ds.loc[day]).sum())
        sw = float((w * r_daily.loc[day]).sum() - abs(w).sum() * haircut_bp / 1e4)
        sw *= roll.loc[day] / 365.0
        rows.append({"date": day, "px": px, "swap": sw, "cost": cost,
                     "net": px + sw - cost, "gross": px + sw})
    return pd.DataFrame(rows)


def stats(df: pd.DataFrame, lo: str, hi: str) -> dict:
    d = df[df["date"].between(lo, hi)]
    r = d.set_index("date")["net"]
    if len(r) < 100 or r.std() == 0:
        return {"days": int(len(r)), "sharpe": None}
    eq = (1 + r).cumprod()
    wk = r.resample("W").sum()
    pos = wk[wk > 0].sum()
    top5 = wk.nlargest(5).sum()
    drop10 = wk.drop(wk.nlargest(10).index)
    by = r.groupby(r.index.year).sum()
    return {"days": int(len(r)),
            "sharpe": round(float(r.mean() / r.std() * 252 ** .5), 3),
            "ann_return": round(float(r.mean() * 252), 4),
            "max_drawdown": round(float((eq / eq.cummax() - 1).min()), 4),
            "ann_swap": round(float(d["swap"].mean() * 252), 4),
            "ann_cost": round(float(d["cost"].mean() * 252), 4),
            "top5_week_profit_share": round(float(top5 / pos), 4) if pos > 0 else None,
            "sharpe_ex_top10_weeks": round(float(drop10.mean() / drop10.std() * 52 ** .5), 3)
            if drop10.std() else None,
            "negative_years": int((by < 0).sum()), "years": int(len(by)),
            "by_year": {int(k): round(float(v), 4) for k, v in by.items()}}


def verdict(sel: dict, con: dict) -> list[str]:
    f = []
    for name, w in (("selection", sel), ("confirmation", con)):
        if (w.get("sharpe") or -9) < 1.0:
            f.append(f"{name}_sharpe_lt_1.0")
        if w.get("negative_years", 9) > max(1, w.get("years", 1) // 3):
            f.append(f"{name}_too_many_negative_years")
        if (w.get("top5_week_profit_share") or 1) >= .20:
            f.append(f"{name}_top5_week_share_ge_20pct")
        if (w.get("sharpe_ex_top10_weeks") or -9) < 0.5:
            f.append(f"{name}_sharpe_ex_top10_lt_0.5")
    return f


def main() -> None:
    mid, hs = load_prices()
    rates = load_short_rates(index=mid.index)
    out = {"spec": {"n_side": N_SIDE, "mom_weeks": MOM_WEEKS, "rebalance": "weekly Friday",
                    "selection": SELECTION, "confirmation": CONFIRM},
           "rows": len(mid), "start": str(mid.index.min().date()), "end": str(mid.index.max().date())}
    for mode, label in [("carry", "A_carry"), ("momentum", "B_momentum")]:
        d = run(mid, hs, rates, mode)
        out[label] = {"selection": stats(d, *SELECTION), "confirmation": stats(d, *CONFIRM)}
        print(label, json.dumps(out[label], ensure_ascii=False), flush=True)
    out["failed_criteria"] = verdict(out["A_carry"]["selection"], out["A_carry"]["confirmation"])
    out["decision"] = "NO_GO" if out["failed_criteria"] else "PENDING_FULL_RISK_TESTS"
    out["sensitivity"] = {}
    for lab, kw in [("haircut25", dict(haircut_bp=25)), ("haircut50", dict(haircut_bp=50)),
                    ("haircut100", dict(haircut_bp=100)), ("hold2w", dict(hold_weeks=2))]:
        d = run(mid, hs, rates, "carry", **kw)
        out["sensitivity"][lab] = {"selection": stats(d, *SELECTION),
                                   "confirmation": stats(d, *CONFIRM)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str),
                                      encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("decision", "failed_criteria")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
