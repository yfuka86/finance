#!/usr/bin/env python3
"""Collect Dukascopy hourly bid/ask candles for the major USD pairs.

Dukascopy is an ECN broker, so these are **actual tradable quotes with bid and
ask** — unlike the ECB reference rates, which are a 14:15 CET mid fixing.

Endpoint (verified 2026-08-04, no key required):
    https://datafeed.dukascopy.com/datafeed/{PAIR}/{YYYY}/{MM}/BID_candles_hour_1.bi5
    ...                                                     /ASK_candles_hour_1.bi5
`MM` is **0-indexed** (00 = January). One monthly file holds 720 hourly records
of 24 bytes: `>iiiiif` = (seconds from month start, open, close, low, high, volume),
prices as integers scaled by the pair's point value.

Only the seven USD pairs are collected: from `s_i - s_USD` for every i, any cross
is a difference, so these recover the full eight-currency strength matrix without
downloading 28 pairs.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import lzma
import struct
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE = "https://datafeed.dukascopy.com/datafeed"
PAIRS = ("EURUSD", "USDJPY", "GBPUSD", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD")
# 価格は整数。JPY 建ては小数3桁、その他は5桁でスケールされる。
SCALE = {p: (1e3 if p.endswith("JPY") else 1e5) for p in PAIRS}
ROOT = Path("data/fx_dukascopy")
REC = struct.Struct(">iiiiif")


def _get(url: str, tries: int = 6) -> bytes | None:
    for i in range(tries):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 404:
                return None                      # その月のデータ無し（上場前など）
            if r.status_code == 200:
                return r.content
        except requests.RequestException:
            pass
        time.sleep(min(2 ** i, 30))
    raise RuntimeError(f"dukascopy retry exhausted: {url}")


def month_frame(pair: str, year: int, month0: int, side: str) -> pd.DataFrame:
    """One monthly file -> hourly OHLC for that side (bid or ask)."""
    raw = _get(f"{BASE}/{pair}/{year}/{month0:02d}/{side}_candles_hour_1.bi5")
    if not raw:
        return pd.DataFrame()
    try:
        blob = lzma.LZMADecompressor(format=lzma.FORMAT_ALONE).decompress(raw)
    except lzma.LZMAError:
        return pd.DataFrame()
    start = dt.datetime(year, month0 + 1, 1, tzinfo=dt.timezone.utc)
    sc = SCALE[pair]
    rows = []
    for i in range(len(blob) // REC.size):
        t, o, c, l, h, v = REC.unpack_from(blob, i * REC.size)
        if o == 0 and c == 0 and v == 0:         # 週末など取引の無い時間帯
            continue
        rows.append((start + dt.timedelta(seconds=t), o / sc, h / sc, l / sc, c / sc, v))
    return pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, default=2011)
    ap.add_argument("--end-year", type=int, default=2026)
    args = ap.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for pair in PAIRS:
        out = ROOT / f"{pair}_hour.parquet"
        if out.exists():
            print(f"{pair}: skip (exists)", flush=True)
            manifest[pair] = {"rows": len(pd.read_parquet(out)), "file": str(out)}
            continue
        frames = []
        for year in range(args.start_year, args.end_year + 1):
            for m0 in range(12):
                if dt.date(year, m0 + 1, 1) > dt.date.today():
                    break
                bid = month_frame(pair, year, m0, "BID")
                ask = month_frame(pair, year, m0, "ASK")
                if bid.empty or ask.empty:
                    continue
                m = bid.merge(ask, on="ts", suffixes=("_bid", "_ask"))
                m["spread"] = m["close_ask"] - m["close_bid"]
                m["mid"] = (m["close_ask"] + m["close_bid"]) / 2
                frames.append(m)
                time.sleep(0.25)                 # 連続取得で接続を切られるため
            print(f"  {pair} {year} done ({sum(len(f) for f in frames):,} rows)", flush=True)
        if not frames:
            continue
        d = pd.concat(frames, ignore_index=True).drop_duplicates("ts").sort_values("ts")
        d.to_parquet(out, index=False)
        manifest[pair] = {"rows": len(d), "start": str(d.ts.min()), "end": str(d.ts.max()),
                          "median_spread": float(d.spread.median()), "file": str(out)}
        print(f"{pair}: {len(d):,} rows {d.ts.min()} .. {d.ts.max()}", flush=True)
    (ROOT / "manifest.json").write_text(json.dumps(
        {"source": BASE, "pairs": list(PAIRS), "granularity": "hour",
         "note": "Dukascopy ECN bid/ask candles; MM is 0-indexed in the URL",
         "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(), "pairs_detail": manifest},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print("manifest written")


if __name__ == "__main__":
    main()
