#!/usr/bin/env python3
"""Export detail + today's candidates for the close-auction overnight LONG-ONLY
strategy dashboard. Selection window shown; 2025-10+ confirmation stays SEALED.

Best cell (verified): long-only top-decile of the ridge overnight-residual
prediction, bought at D close auction (引成), sold at D+1 open auction (寄成),
cash (no borrow/financing), quote-free.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experiment_close_auction_overnight import (attach_target, ridge_wf,
                                                        SEL)
from scripts.export_oversold_detail import fundamentals_for, names_map

OPEN_BPS, CLOSE_BPS = 1.5e-4, 0.5e-4
OUT = Path("data/jp_close_auction_overnight")
SEAL = pd.Timestamp("2026-08-11")   # display frozen at seal date


def main() -> None:
    m = attach_target()
    pred = ridge_wf(m)
    fr = m.loc[pred.index].assign(pred=pred)
    fr["_s"] = fr["pred"] - fr.groupby("date")["pred"].transform("mean")
    thr = fr.groupby("date")["_s"].transform("quantile", .9)
    book = fr[fr["_s"] >= thr].copy()
    nm = names_map()

    # selection-window daily excess series (display)
    disp = book[book["date"] <= SEAL]
    rows = []
    for dt, g in disp.groupby("date"):
        mkt = fr[fr["date"] == dt]["ret_on_fwd"].mean()
        rows.append((dt, g["ret_on_fwd"].mean() - mkt - (OPEN_BPS + CLOSE_BPS)))
    r = pd.Series(dict(rows)).sort_index(); r.index = pd.to_datetime(r.index)
    r = r[(r.index >= SEL[0]) & (r.index <= SEL[1])]
    cum = (1 + r).cumprod()
    monthly = r.groupby(r.index.to_period("M")).sum()

    # today's candidates = latest signal date's book
    last = book["date"].max()
    cand = book[book["date"] == last].sort_values("_s", ascending=False)
    syms5 = [s if len(str(s)) == 5 else str(s) for s in cand["symbol"].astype(str)]
    fund = fundamentals_for(sorted(set(syms5)))
    names = [{"sym": str(row.symbol), "name": nm.get(str(row.symbol), ("", ""))[0],
              "sector": nm.get(str(row.symbol), ("", ""))[1],
              "pred_rank": int(i + 1), **fund.get(str(row.symbol), {})}
             for i, row in enumerate(cand.itertuples())]

    detail = {
        "sealed_note": "成績表示は封印日2026-08-11で凍結（判定はフォワード）",
        "sel_sharpe": round(float(r.mean() / r.std() * 252 ** .5), 2),
        "sel_ann_pct": round(float(r.mean() * 252 * 100), 1),
        "sel_ir_ex_top10": round(float(
            r.drop(r.nlargest(10).index).mean()
            / r.drop(r.nlargest(10).index).std() * 252 ** .5), 2),
        "n_book_days": int(r.shape[0]),
        "daily_cum": [[str(d.date()), round(float(v), 5)] for d, v in cum.items()],
        "monthly": [[str(p), round(float(v) * 100, 2)] for p, v in monthly.items()],
        "candidates": {"signal_date": str(pd.Timestamp(last).date()),
                       "entry": "翌営業日ではなく当日15:30引成で買い→翌09:00寄成で売り（現物・借株不要）",
                       "names": names},
    }
    exe = json.loads((OUT / "executability.json").read_text(encoding="utf-8"))
    detail["cell_comparison"] = exe["executable_cells_selection"]
    detail["executability"] = exe["executability_checklist"]
    (OUT / "detail.json").write_text(json.dumps(detail, ensure_ascii=False),
                                     encoding="utf-8")
    print(json.dumps({"sel_sharpe": detail["sel_sharpe"],
                      "ir_ex_top10": detail["sel_ir_ex_top10"],
                      "candidates": len(names)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
