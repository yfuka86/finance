#!/usr/bin/env python3
"""Forward tick accumulation for the sealed gotobi re-judgment (no-peek).

Fetches the 00:00-01:00 UTC (09:00-10:00 JST) USDJPY tick file for every JP
business day from 2026-08-01 up to yesterday, in catch-up fashion — run it at any
cadence (daily cron, weekly, ad hoc) and it fills whatever is missing.

★No-peek design: stores RAW ticks only. It never computes, stores or prints a
return, win rate or anything performance-like. The evaluation happens once, on
2027-08-06+, via verify_gotobi_final.py (date-guarded). See
docs/PREREGISTER_FX_GOTOBI_FORWARD.md.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd

from scripts.collect_dukascopy_fix_ticks import _get, parse_hour, BASE, PAIR
from scripts.experiment_fx_gotobi import gotobi_days, jp_business_days

ROOT = Path("data/fx_ticks_fix/forward")
SCANS = ROOT / "scanned.jsonl"
START = dt.date(2026, 8, 1)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    done = set()
    if SCANS.exists():
        done = {json.loads(l)["day"] for l in SCANS.read_text().splitlines() if l.strip()}
    bdays = jp_business_days()
    got = gotobi_days(bdays, "2026-08-01", "2027-12-31")
    end = dt.date.today() - dt.timedelta(days=1)
    targets = [d.date() for d in bdays
               if START <= d.date() <= end and d.date().isoformat() not in done]
    print(f"pending={len(targets)} done={len(done)}", flush=True)
    for day in targets:
        raw = _get(f"{BASE}/{PAIR}/{day.year}/{day.month - 1:02d}/{day.day:02d}/00h_ticks.bi5")
        if raw == b"__UNAVAILABLE__":
            print(f"  retry-later {day}", flush=True)
            continue                          # scanned に書かない＝次回埋まる
        status = "missing"
        if raw:
            df = parse_hour(raw, day)
            if len(df):
                df["kind"] = "gotobi" if day in got else "other"
                out = ROOT / f"{PAIR}_{day.year}-{day.month:02d}.parquet"
                old = [pd.read_parquet(out)] if out.exists() else []
                pd.concat(old + [df], ignore_index=True).drop_duplicates(
                    ["ts"]).sort_values("ts").to_parquet(out, index=False)
                status = f"ok:{len(df)}"
        with SCANS.open("a") as fh:
            fh.write(json.dumps({"day": day.isoformat(), "status": status}) + "\n")
    # データ完全性のみ報告（成績は一切出さない）
    n_parts = len(list(ROOT.glob("*.parquet")))
    print(f"integrity: scanned={len(done) + len(targets)} parts={n_parts}", flush=True)


if __name__ == "__main__":
    main()
