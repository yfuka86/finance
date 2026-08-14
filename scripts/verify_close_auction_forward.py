#!/usr/bin/env python3
"""Close-auction overnight LONG-ONLY forward verdict -- SEALED until 2028-08-14.

Preregistered in docs/PREREGISTER_CLOSE_AUCTION_FORWARD.md. No override flag.
Recomputes the long-only excess from raw data with the frozen ridge spec.
User accepted the concentration criterion is waived; the surviving bars are
excess IR, ex-top10 robustness, and both-years-positive.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd

JUDGMENT_DATE = dt.date(2028, 8, 14)
SEAL = (pd.Timestamp("2026-08-12"), pd.Timestamp("2028-08-13"))
OPEN_BPS, CLOSE_BPS = 1.5e-4, 0.5e-4


def main() -> None:
    if dt.date.today() < JUDGMENT_DATE:
        raise SystemExit(
            f"SEALED until {JUDGMENT_DATE}. Today is {dt.date.today()}. "
            "No override exists; opening early would unseal the forward test.")
    from scripts.experiment_close_auction_overnight import attach_target, ridge_wf
    m = attach_target()
    pred = ridge_wf(m)
    fr = m.loc[pred.index].assign(pred=pred)
    fr["_s"] = fr["pred"] - fr.groupby("date")["pred"].transform("mean")
    thr = fr.groupby("date")["_s"].transform("quantile", .9)
    book = fr[fr["_s"] >= thr]
    rows = []
    for day, g in book.groupby("date"):
        mkt = fr[fr["date"] == day]["ret_on_fwd"].mean()
        rows.append((day, g["ret_on_fwd"].mean() - mkt - (OPEN_BPS + CLOSE_BPS)))
    r = pd.Series(dict(rows)).sort_index()
    r.index = pd.to_datetime(r.index)
    r = r[(r.index >= SEAL[0]) & (r.index <= SEAL[1])]
    ir = float(r.mean() / r.std() * 252 ** .5)
    ex10 = r.drop(r.nlargest(10).index)
    ir10 = float(ex10.mean() / ex10.std() * 252 ** .5)
    yearly = r.groupby(r.index.year).sum()
    crit = {"excess_ir_ge_05": ir >= .5,
            "ex_top10_ir_ge_04": ir10 >= .4,
            "both_years_positive": bool((yearly > 0).all()),
            "freq_ge_60py": bool(len(r) / (len(r) and (r.index[-1] - r.index[0]).days / 365 or 1) >= 60)}
    verdict = {"judged_at": dt.datetime.now(dt.timezone.utc).isoformat(),
               "window": [str(SEAL[0].date()), str(SEAL[1].date())],
               "excess_ir": round(ir, 3), "ex_top10_ir": round(ir10, 3),
               "excess_ann_pct": round(float(r.mean() * 252 * 100), 2),
               "yearly_pct": {int(y): round(float(v) * 100, 2) for y, v in yearly.items()},
               "days": int(len(r)), "criteria": crit,
               "decision": "GO_PENDING_USER_APPROVAL" if all(crit.values())
               else "NO_GO_PERMANENT_CLOSE"}
    out = Path("data/jp_close_auction_overnight/forward_verdict.json")
    out.open("x", encoding="utf-8").write(json.dumps(verdict, ensure_ascii=False, indent=2))
    print(json.dumps(verdict, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
