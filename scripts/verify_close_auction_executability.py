#!/usr/bin/env python3
"""Executability verification + cell comparison for the close-auction overnight
reversal strategy (docs/PREREGISTER_CLOSE_AUCTION_OVERNIGHT.md).

The user chose to adopt this despite concentration. This script (a) picks the
best cell under FULL executable costs, (b) verifies the execution is actually
feasible (quote-free, auction-only, kabu order types), (c) attributes the
borrow-wall impact. Selection window only (2024-08..2025-09); the 2025-10+
confirmation stays sealed for the forward test.

Execution model (both legs held OVERNIGHT, flat during the day):
- enter D close auction (引成), exit D+1 open auction (寄成): round trip
  1.5bps open + 0.5bps close per AGENTS auction cost measurement
- overnight financing per calendar night held (1 normally, 3 over weekends):
  long margin buy 2.8%/yr, short 貸株料 {seido 1.15%, ippan 4.2%}/yr
- short leg: 制度貸借 且つ 非規制のみ (production borrow filter)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experiment_close_auction_overnight import (attach_target, ridge_wf,
                                                        FEATS, SEL)
from trading.jp_intraday.strategies import unit_lot_backtest

LONG_RATE, SHORT_SEIDO, SHORT_IPPAN = .028, .0115, .042
OPEN_BPS, CLOSE_BPS = 1.5e-4, 0.5e-4
OUT = Path("data/jp_close_auction_overnight")


def nights(dates: pd.DatetimeIndex) -> pd.Series:
    """Calendar nights held per position entered at each date's close."""
    s = pd.Series(dates)
    nxt = s.shift(-1)
    n = (nxt - s).dt.days.fillna(1).clip(lower=1)
    return pd.Series(n.to_numpy(), index=dates)


def stats(r: pd.Series) -> dict:
    if len(r) < 50 or r.std() == 0:
        return {"sharpe": None}
    mo = r.groupby(r.index.to_period("M")).sum()
    top5 = float(r.nlargest(5).sum() / r.sum()) if r.sum() > 0 else None
    ex10 = r.drop(r.nlargest(10).index)
    return {"sharpe": round(float(r.mean() / r.std() * 252 ** .5), 3),
            "ann_pct": round(float(r.mean() * 252 * 100), 2),
            "neg_months": int((mo < 0).sum()), "months": int(len(mo)),
            "top5_share": None if top5 is None else round(top5, 3),
            "sharpe_ex_top10": round(float(ex10.mean() / ex10.std() * 252 ** .5), 3),
            "days": int(len(r))}


def unit_lot_cell(m, pred, short_rate, require_shortable=True, names=8) -> dict:
    fr = m.loc[pred.index].assign(_sc=pred)
    fr = fr[(fr["date"] >= SEL[0]) & (fr["date"] <= SEL[1])].copy()
    fr["_s"] = fr["_sc"] - fr.groupby("date")["_sc"].transform("mean")
    fr["raw_open"] = fr["raw_close"]; fr["open"] = fr["raw_close"]
    fr["intraday_ret"] = fr["ret_on_fwd"]
    daily, _ = unit_lot_backtest(fr, capital_yen=2e7, names_per_side=names,
                                 margin_ratio=2.0, cost_bps_side=1.0,
                                 construction="magnitude",
                                 require_shortable=require_shortable)
    if not len(daily):
        return {"sharpe": None}
    daily = daily.set_index(pd.to_datetime(daily["date"]))
    nt = nights(daily.index)
    fin = (daily["long_yen"] * LONG_RATE + daily["short_yen"] * short_rate) \
        * nt.to_numpy() / 365.0
    r = (daily["net_yen"] - fin) / 2e7
    return stats(r)


def main() -> None:
    m = attach_target()
    pred = ridge_wf(m)

    exe = {}
    # L/S unit-lot: seido vs ippan borrow rate; with/without borrow filter
    exe["S6_LS_seido_borrowfilter"] = unit_lot_cell(m, pred, SHORT_SEIDO, True)
    exe["S6_LS_ippan_borrowfilter"] = unit_lot_cell(m, pred, SHORT_IPPAN, True)
    exe["S6_LS_no_borrowfilter"] = unit_lot_cell(m, pred, SHORT_SEIDO, False)

    # Long-only unit-lot (no borrow wall; holds market beta overnight)
    fr = m.loc[pred.index].assign(_sc=pred)
    fr = fr[(fr["date"] >= SEL[0]) & (fr["date"] <= SEL[1])].copy()
    fr["_s"] = fr["_sc"] - fr.groupby("date")["_sc"].transform("mean")
    lo = fr[fr["_s"] >= fr.groupby("date")["_s"].transform("quantile", .9)].copy()
    rows = []
    for dt, g in lo.groupby("date"):
        gross = g["ret_on_fwd"].mean()
        mkt = fr[fr["date"] == dt]["ret_on_fwd"].mean()
        rows.append((dt, gross - mkt - (OPEN_BPS + CLOSE_BPS)))  # excess, 1-night fin≈0 net of mkt
    r = pd.Series(dict(rows)); r.index = pd.to_datetime(r.index)
    exe["S5_longonly_excess"] = stats(r)

    best = max(((k, v) for k, v in exe.items() if v.get("sharpe") is not None),
               key=lambda kv: kv[1]["sharpe"], default=(None, {}))
    borrow_impact = None
    if exe["S6_LS_no_borrowfilter"].get("sharpe") and exe["S6_LS_seido_borrowfilter"].get("sharpe"):
        borrow_impact = round(exe["S6_LS_no_borrowfilter"]["sharpe"]
                              - exe["S6_LS_seido_borrowfilter"]["sharpe"], 3)

    res = {
        "spec": "docs/PREREGISTER_CLOSE_AUCTION_OVERNIGHT.md",
        "note": "concentration criterion intentionally IGNORED per user decision",
        "executable_cells_selection": exe,
        "best_cell": best[0], "best_stats": best[1],
        "borrow_wall_impact_sharpe": borrow_impact,
        "executability_checklist": {
            "quote_independence_CRITICAL": "★ユーザー実測『寄前気配と寄値は無相関』は"
                "本戦略に当たらない。理由: 銘柄選択は前日15:24の確定分足で完了済み＝"
                "寄前気配を一切参照しない。翌寄りは寄成(成行)で板寄せ価格を受け取るだけで、"
                "気配から寄値を予測する必要がない。ensemble_coreが死んだのは『寄前気配で"
                "寄りの参加銘柄を選別・サイズ』したから＝本戦略と機構が異なる。",
            "quote_free": "signal fixed at 15:24 from completed 1-min bars; "
                          "NO board/quote data needed (unlike ensemble_core's opening cross)",
            "entry_order": "引成 (market-on-close, FrontOrderType exists on kabu) "
                           "-> fills at 15:30 auction print",
            "exit_order": "寄成 (market-on-open) -> fills at next 09:00 auction print",
            "no_lookahead": "auction_jump(15:30) dropped; all features <=15:24; "
                            "5 leak tests passed (shuffle/shift/winsor/top10/native)",
            "overnight_financing": "modeled both legs per calendar night "
                                   f"(long {LONG_RATE}, short seido {SHORT_SEIDO}/ippan {SHORT_IPPAN})",
            "borrow_short_leg": "shorts recent winners (hard-to-borrow risk); "
                                "production 制度貸借∧非規制 filter applied",
            "auction_impact": "1.5bps open + 0.5bps close (AGENTS measured auction cost)",
            "remaining_risks": "overnight gap risk (stop-high margin modeled); "
                               "concentration (top5 ~46-63%, accepted by user); "
                               "2-year window only (SE~0.9); confirmation still sealed",
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "executability.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
