#!/usr/bin/env python3
"""Authoritative filtered-vs-unfiltered + year breakdown for the close-auction
overnight LONG-ONLY strategy. Single clean methodology (top-decile of ridge
prediction, excess over the predicted-universe daily mean, 2bps).

Minute data 2024-08 earliest (JQ limit). This OPENS the confirmation window
(2025-10+) for display; the forward seal judges on 2026-08-12+ (после the
current minute end) so the clean forward test is unaffected.

★Key result: the edge lives in the weird small caps the user asked to exclude.
Filtering to clean large caps (Prime/Standard, >=Y30bn, no margin-reg) collapses
the strategy (SEL 4.06->1.64, ex-top10 +2.59->-1.07; FULL 2.57->0.83->neg).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experiment_close_auction_overnight import attach_target, ridge_wf, SEL

OUT = Path("data/jp_close_auction_overnight")


def longonly_series(quality: bool) -> pd.Series:
    m = attach_target(quality=quality)
    pred = ridge_wf(m)
    fr = m.loc[pred.index].assign(pred=pred)
    fr["_s"] = fr["pred"] - fr.groupby("date")["pred"].transform("mean")
    thr = fr.groupby("date")["_s"].transform("quantile", .9)
    book = fr[fr["_s"] >= thr]
    rows = []
    for dt, g in book.groupby("date"):
        mkt = fr[fr["date"] == dt]["ret_on_fwd"].mean()
        rows.append((dt, g["ret_on_fwd"].mean() - mkt - 2e-4))
    r = pd.Series(dict(rows)).sort_index()
    r.index = pd.to_datetime(r.index)
    return r


def stat(r: pd.Series) -> dict:
    if len(r) < 20 or r.std() == 0:
        return {"days": int(len(r)), "sharpe": None}
    ex10 = r.drop(r.nlargest(10).index)
    return {"days": int(len(r)),
            "sharpe": round(float(r.mean() / r.std() * 252 ** .5), 2),
            "ann_pct": round(float(r.mean() * 252 * 100), 1),
            "ex_top10_sharpe": round(float(ex10.mean() / ex10.std() * 252 ** .5), 2)}


def main() -> None:
    res = {"note": "minute data 2024-08 earliest; ★edge is in the excluded small caps",
           "compare": {}, "filtered_by_year": {}, "filtered_by_half": {}}
    for q, lab in ((False, "unfiltered_junk_included"), (True, "filtered_clean")):
        r = longonly_series(q)
        sel = r[(r.index >= SEL[0]) & (r.index <= SEL[1])]
        res["compare"][lab] = {"selection": stat(sel), "full": stat(r)}
        if q:
            for y, g in r.groupby(r.index.year):
                res["filtered_by_year"][str(y)] = stat(g)
            for h, g in r.groupby(r.index.map(lambda d: f"{d.year}H{1 if d.month<=6 else 2}")):
                res["filtered_by_half"][h] = stat(g)
    res["conclusion"] = (
        "クリーン母集団(ユーザー要望)では選択窓Sh1.64/ex-top10−1.07・全窓Sh0.83/−0.83で"
        "頑健性喪失＝NO-GO。αは除外対象の変な小型株(直近安値の小型/グロース/増担保)の"
        "オーバーナイト反発に宿っていた。実運用品質の母集団では成立しない。")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "by_year.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
