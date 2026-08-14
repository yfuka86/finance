#!/usr/bin/env python3
"""Extend the 2y 1-minute bulk file forward, one business day at a time.

Fixes the old collect_jp_minutes_2y.py (broken scratchpad path, hardcoded end
date). The universe is the symbols already present in the 2y file. Fetches only
dates strictly after the file's last date, up to today, then rewrites the
parquet. Idempotent; run SOLO (J-Quants bulk minute add-on). This feeds the
close-auction overnight forward-seal candidate ledger.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from trading.jp_intraday.collector import collect_jquants_minutes_bulk

MIN_PARQUET = Path("data/jp_minutes_2y/jp_1m_2024-08-01_2026-07-24.parquet")
OUT_DIR = Path("data/jp_minutes_2y")


def main() -> None:
    if not MIN_PARQUET.exists():
        raise SystemExit(f"base file missing: {MIN_PARQUET}")
    existing = pd.read_parquet(MIN_PARQUET, columns=["timestamp", "symbol"])
    last = pd.to_datetime(existing["timestamp"]).dt.tz_convert(
        "Asia/Tokyo").dt.normalize().max()
    syms = sorted(existing["symbol"].astype(str).unique())
    start = (last + pd.Timedelta(days=1)).date()
    today = dt.date.today()
    if start > today:
        print(f"up to date (last={last.date()})")
        return
    add = collect_jquants_minutes_bulk(syms, start.isoformat(), today.isoformat(),
                                       OUT_DIR / "_daily_tmp")
    if add is None or len(add) == 0:
        print(f"no new bars ({start}..{today})")
        return
    add = add[add["symbol"].astype(str).isin(set(syms))]
    merged = pd.concat([pd.read_parquet(MIN_PARQUET), add], ignore_index=True)
    merged = merged.drop_duplicates(["timestamp", "symbol"]).sort_values(
        ["symbol", "timestamp"])
    merged.to_parquet(MIN_PARQUET, index=False)
    print(f"appended {len(add):,} rows; last now "
          f"{pd.to_datetime(merged['timestamp']).max()}")


if __name__ == "__main__":
    main()
