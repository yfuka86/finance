#!/usr/bin/env python3
"""One-shot sealed evaluation of the gotobi tick form. DO NOT RUN BEFORE 2027-08-06.

Evaluates the frozen 9:00->9:55 JST spec on the sealed 2020-2026 window plus the
forward accumulation, per docs/PREREGISTER_FX_GOTOBI_FORWARD.md. The date guard
below is part of the preregistration: running early consumes the window and voids
the judgment. There is no override flag on purpose.
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

JUDGMENT_DATE = dt.date(2027, 8, 6)
GMO = 0.002
OUT = Path("data/fx_gotobi")


def main() -> None:
    if dt.date.today() < JUDGMENT_DATE:
        print(f"SEALED until {JUDGMENT_DATE}. Refusing to run "
              f"(preregistration: docs/PREREGISTER_FX_GOTOBI_FORWARD.md).")
        sys.exit(1)
    from scripts.experiment_fx_gotobi import gotobi_days, jp_business_days, passes
    fs = sorted(glob.glob("data/fx_ticks_fix/USDJPY_*.parquet")) + \
        sorted(glob.glob("data/fx_ticks_fix/forward/USDJPY_*.parquet"))
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    d["ts"] = pd.to_datetime(d["ts"])
    d["day"] = d["ts"].dt.date
    bdays = jp_business_days()
    got = gotobi_days(bdays, "2020-01-01", "2027-12-31")
    rows = []
    for day, g in d[d["ts"].dt.year >= 2020].groupby("day"):
        w = g[(g["ts"].dt.hour == 0) & (g["ts"].dt.minute < 55)].sort_values("ts")
        if len(w) < 30:
            continue
        e, x = w.iloc[0], w.iloc[-1]
        mid_e, mid_x = (e.ask + e.bid) / 2, (x.ask + x.bid) / 2
        rows.append({"day": pd.Timestamp(day), "kind": "gotobi" if day in got else "other",
                     "net_gmo": (mid_x - GMO) / mid_e - 1.0,
                     "net_duka": x.bid / e.ask - 1.0})
    T = pd.DataFrame(rows)
    tr = T[T["kind"].eq("gotobi")].set_index("day")["net_gmo"]
    ctl = T[T["kind"].ne("gotobi")].set_index("day")["net_gmo"]
    cal = pd.date_range(tr.index.min(), tr.index.max(), freq="D")
    daily = tr.reindex(cal).fillna(0.0)
    pos = tr[tr > 0].sum()
    ex10 = daily.drop(tr.nlargest(10).index, errors="ignore")
    by = daily.groupby(daily.index.year).sum()
    w = {"trades": int(len(tr)),
         "sharpe": round(float(daily.mean() / daily.std() * 252 ** .5), 3),
         "trade_mean_bps": round(float(tr.mean() * 1e4), 2),
         "negative_years": int((by < 0).sum()), "years": int(len(by)),
         "top5_trade_share": round(float(tr.nlargest(5).sum() / pos), 4) if pos > 0 else None,
         "sharpe_ex_top10": round(float(ex10.mean() / ex10.std() * 252 ** .5), 3),
         "control_mean_bps": round(float(ctl.mean() * 1e4), 2),
         "by_year_bps": {int(k): round(float(v) * 1e4, 1) for k, v in by.items()}}
    failed = passes(w)
    if (w["trade_mean_bps"] or -9) <= (w["control_mean_bps"] or 9):
        failed.append("not_better_than_control")
    verdict = {"evaluated_at": str(dt.date.today()), "stats": w,
               "failed_criteria": failed,
               "decision": "LIVE_CANDIDATE" if not failed else "PERMANENTLY_CLOSED"}
    (OUT / "final_verdict.json").write_text(
        json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(verdict, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
