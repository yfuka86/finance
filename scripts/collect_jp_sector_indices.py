#!/usr/bin/env python3
"""Collect the 33 S33 sector indices (codes 0040-0060, hex-style) from 2008.

The existing data/jp_derivatives/indices_*.parquet only start in 2018; the API
serves sector indices from 2008-05-07 (same as TOPIX). 18 years is what a
long-horizon sector-momentum analysis actually needs.
"""
from __future__ import annotations
import json, time, datetime as dt
from pathlib import Path
import pandas as pd
from data.collectors.jquants import _paginated_get

CODES = [f"00{a}{b}" for a in "456" for b in "0123456789ABCDEF"]
CODES = [c for c in CODES if "0040" <= c <= "0060"][:33]
OUT = Path("data/jp_derivatives/sector_indices_2008_2026.parquet")

def main() -> None:
    frames = []
    for i, code in enumerate(CODES, 1):
        d = _paginated_get("/v2/indices/bars/daily",
                           {"code": code, "from": "20080101", "to": dt.date.today().strftime("%Y%m%d")})
        if len(d):
            frames.append(d)
        print(f"{i}/33 {code}: {len(d)} rows", flush=True)
        time.sleep(0.2)
    all_ = pd.concat(frames, ignore_index=True).drop_duplicates(["Date", "Code"])
    all_.to_parquet(OUT, index=False)
    print(f"saved {len(all_):,} rows -> {OUT}")

if __name__ == "__main__":
    main()
