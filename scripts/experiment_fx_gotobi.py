#!/usr/bin/env python3
"""Tokyo-fix gotobi anomaly in USDJPY. Frozen in docs/PREREGISTER_FX_GOTOBI.md.

Buy USDJPY at 09:00 JST (00:00 UTC bar open, ask), sell at 10:00 JST (same bar
close, bid) on gotobi settlement days (5/10/15/20/25/30, rolled BACK to the prior
business day). JST has no DST, so the fix (9:55) is always inside the 00:00 UTC
bar. Costs are inherent (ask in, bid out). No overnight, no swap.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.collect_dukascopy_fx_hourly import ROOT as HOUR_ROOT

SELECTION, CONFIRM = ("2011-01-01", "2019-12-31"), ("2020-01-01", "2026-12-31")
OUT = Path("data/fx_gotobi")


def load_hour(pair: str = "USDJPY") -> pd.DataFrame:
    import glob
    fs = sorted(glob.glob(str(HOUR_ROOT / "parts" / f"{pair}_*.parquet")))
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    d["ts"] = pd.to_datetime(d["ts"])
    return d.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)


def jp_business_days() -> pd.DatetimeIndex:
    """TSE sessions as a proxy for JP bank business days."""
    from trading.jp_intraday.daily_gap import load_existing_daily
    d = load_existing_daily()
    return pd.DatetimeIndex(sorted(pd.to_datetime(d["Date"]).unique()))


def gotobi_days(bdays: pd.DatetimeIndex, lo: str, hi: str) -> set:
    """Calendar 5/10/15/20/25/30, rolled back to the prior business day."""
    bset = set(bdays.date)
    out = set()
    for day in pd.date_range(lo, hi, freq="D"):
        if day.day not in (5, 10, 15, 20, 25, 30):
            continue
        d = day.date()
        while d not in bset:
            d = d - pd.Timedelta(days=1)
            if (day.date() - d).days > 10:
                d = None
                break
        if d is not None:
            out.add(d)
    return out


def fix_bar_trades(hour: pd.DataFrame) -> pd.DataFrame:
    """One row per session day: the 00:00 UTC bar's ask-open -> bid-close return."""
    b = hour[hour["ts"].dt.hour == 0].copy()
    b["day"] = b["ts"].dt.date
    b["ret"] = b["close_bid"] / b["open_ask"] - 1.0
    return b[["day", "ret"]].dropna()


def stats(daily: pd.Series, trades: pd.Series, lo: str, hi: str) -> dict:
    r = daily.loc[lo:hi]
    t = trades.loc[lo:hi] if len(trades) else trades
    if len(r) < 100 or r.std() == 0:
        return {"days": int(len(r)), "sharpe": None}
    pos = t[t > 0].sum() if len(t) else 0
    ex10 = r.copy()
    if len(t) >= 10:
        ex10 = r.drop(t.nlargest(10).index, errors="ignore")
    by = r.groupby(r.index.year).sum()
    return {"days": int(len(r)), "trades": int(len(t)),
            "sharpe": round(float(r.mean() / r.std() * 252 ** .5), 3),
            "ann_return": round(float(r.mean() * 252), 4),
            "trade_mean_bps": round(float(t.mean() * 1e4), 2) if len(t) else None,
            "trade_win_rate": round(float(t.gt(0).mean()), 4) if len(t) else None,
            "top5_trade_share": round(float(t.nlargest(5).sum() / pos), 4) if pos > 0 else None,
            "sharpe_ex_top10": round(float(ex10.mean() / ex10.std() * 252 ** .5), 3)
            if ex10.std() else None,
            "negative_years": int((by < 0).sum()), "years": int(len(by)),
            "by_year_bps": {int(k): round(float(v) * 1e4, 1) for k, v in by.items()}}


def passes(w: dict) -> list[str]:
    f = []
    if (w.get("sharpe") or -9) < 1.0:
        f.append("sharpe_lt_1.0")
    if w.get("negative_years", 9) > max(1, w.get("years", 1) // 3):
        f.append("too_many_negative_years")
    if (w.get("top5_trade_share") or 1) >= .20:
        f.append("top5_trade_share_ge_20pct")
    if (w.get("sharpe_ex_top10") or -9) < 0.5:
        f.append("sharpe_ex_top10_lt_0.5")
    return f


def main() -> None:
    hour = load_hour("USDJPY")
    bdays = jp_business_days()
    trades = fix_bar_trades(hour)
    trades["day"] = pd.to_datetime(trades["day"])
    trades = trades.set_index("day")["ret"]
    got = gotobi_days(bdays, "2011-01-01", "2026-07-31")
    is_got = pd.Series([d.date() in got for d in trades.index], index=trades.index)
    # 全暦日シリーズ（非トレード日は0）
    cal = pd.date_range(trades.index.min(), trades.index.max(), freq="D")
    strat = trades.where(is_got, 0.0).reindex(cal).fillna(0.0)
    control = trades.where(~is_got, 0.0).reindex(cal).fillna(0.0)
    got_tr = trades[is_got]
    ctl_tr = trades[~is_got]

    out = {"spec": "gotobi(5/10/15/20/25/30, roll-back) 09:00->10:00 JST long USDJPY, ask in / bid out",
           "n_gotobi_days": int(len(got_tr)), "n_control_days": int(len(ctl_tr))}
    sel = stats(strat, got_tr, *SELECTION)
    out["primary"] = {"selection": sel, "selection_failed": passes(sel)}
    # 真正性: ゴトー vs 非ゴトーの平均差（選択窓）
    gsel, csel = got_tr.loc[slice(*SELECTION)], ctl_tr.loc[slice(*SELECTION)]
    out["authenticity_selection"] = {
        "gotobi_mean_bps": round(float(gsel.mean() * 1e4), 2),
        "control_mean_bps": round(float(csel.mean() * 1e4), 2),
        "t_diff": round(float((gsel.mean() - csel.mean())
                        / np.sqrt(gsel.var() / len(gsel) + csel.var() / len(csel))), 2)}
    if not out["primary"]["selection_failed"]:
        con = stats(strat, got_tr, *CONFIRM)
        out["primary"]["confirmation"] = con
        out["primary"]["confirmation_failed"] = passes(con)
        gc, cc = got_tr.loc[slice(*CONFIRM)], ctl_tr.loc[slice(*CONFIRM)]
        out["authenticity_confirmation"] = {
            "gotobi_mean_bps": round(float(gc.mean() * 1e4), 2),
            "control_mean_bps": round(float(cc.mean() * 1e4), 2)}
        out["decision"] = ("NO_GO" if out["primary"]["confirmation_failed"]
                           else "PENDING_FULL_RISK_TESTS")
    else:
        out["primary"]["confirmation"] = "NOT_OPENED"
        out["decision"] = "NO_GO"
    # 対照(診断): 非ゴトー日の同トレード
    out["control_diagnostic"] = {"selection": stats(control, ctl_tr, *SELECTION)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str),
                                      encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
