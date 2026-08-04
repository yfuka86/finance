#!/usr/bin/env python3
"""Collect Dukascopy 1-minute bid/ask candles (day files) for the major USD pairs.

Minute data only exists as per-day files (monthly minute files 404):
    {PAIR}/{YYYY}/{MM}/{DD}/BID_candles_min_1.bi5   (MM 0-indexed, 1440 recs/day)

Scale: 7 pairs x 15y x ~313 days x 2 sides ≈ 65k requests ≈ 10+ hours, so this is
resumable at (pair, month) granularity and processes pairs in priority order —
USDJPY first (Tokyo-fix / gotobi research needs it), then EURUSD, then the rest.
Saturdays are skipped (FX closed all day UTC). Rate-limit hits are recorded as
gaps and a rerun fills them.

Storage: data/fx_dukascopy_min/parts/{PAIR}_{YYYY-MM}.parquet (~2GB total — kept
out of git; see .gitignore). Use load_minutes() to read.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import lzma
import struct
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://datafeed.dukascopy.com/datafeed"
PRIORITY = ("USDJPY", "EURUSD", "GBPUSD", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD")
SCALE = {p: (1e3 if p.endswith("JPY") else 1e5) for p in PRIORITY}
ROOT = Path("data/fx_dukascopy_min")
REC = struct.Struct(">iiiiif")


class Unavailable(Exception):
    pass


def _get(url: str, tries: int = 8) -> bytes | None:
    import random
    for i in range(tries):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 404:
                return None
            if r.status_code == 200:
                return r.content
        except requests.RequestException:
            pass
        time.sleep(min(2 * 2 ** i, 90) + random.uniform(0, 1.5))
    raise Unavailable(url)


def day_frame(pair: str, day: dt.date, side: str) -> pd.DataFrame:
    raw = _get(f"{BASE}/{pair}/{day.year}/{day.month - 1:02d}/{day.day:02d}/"
               f"{side}_candles_min_1.bi5")
    if not raw:
        return pd.DataFrame()
    try:
        blob = lzma.LZMADecompressor(format=lzma.FORMAT_ALONE).decompress(raw)
    except lzma.LZMAError:
        return pd.DataFrame()
    start = dt.datetime(day.year, day.month, day.day, tzinfo=dt.timezone.utc)
    sc = SCALE[pair]
    rows = []
    for i in range(len(blob) // REC.size):
        t, o, c, l, h, v = REC.unpack_from(blob, i * REC.size)
        if o == 0 and c == 0 and v == 0:
            continue
        rows.append((start + dt.timedelta(seconds=t), o / sc, h / sc, l / sc, c / sc, v))
    return pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])


def collect_month(pair: str, year: int, month: int, sleep: float, gaps: list) -> pd.DataFrame | None:
    frames = []
    d = dt.date(year, month, 1)
    today = dt.date.today()
    while d.month == month:
        if d >= today:
            break
        if d.weekday() != 5:                      # 土曜はFX全休（UTC）
            try:
                bid = day_frame(pair, d, "BID")
                ask = day_frame(pair, d, "ASK")
                if not bid.empty and not ask.empty:
                    m = bid.merge(ask, on="ts", suffixes=("_bid", "_ask"))
                    m["mid"] = (m["close_bid"] + m["close_ask"]) / 2
                    m["spread"] = m["close_ask"] - m["close_bid"]
                    frames.append(m)
            except Unavailable as e:
                gaps.append(str(e))
            time.sleep(sleep)
        d += dt.timedelta(days=1)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def load_minutes(pair: str, root: Path = ROOT) -> pd.DataFrame:
    fs = sorted((root / "parts").glob(f"{pair}_*.parquet"))
    if not fs:
        raise FileNotFoundError(f"no minute parts for {pair}")
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    return d.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, default=2011)
    ap.add_argument("--end-year", type=int, default=2026)
    ap.add_argument("--pairs", default=",".join(PRIORITY))
    ap.add_argument("--sleep", type=float, default=0.45)
    args = ap.parse_args()
    parts = ROOT / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    gaps: list[str] = []
    done = 0
    for pair in [p.strip().upper() for p in args.pairs.split(",") if p.strip()]:
        for year in range(args.start_year, args.end_year + 1):
            for month in range(1, 13):
                if dt.date(year, month, 1) >= dt.date.today():
                    break
                out = parts / f"{pair}_{year}-{month:02d}.parquet"
                if out.exists():
                    continue
                m = collect_month(pair, year, month, args.sleep, gaps)
                if m is not None:
                    m.to_parquet(out, index=False)
                done += 1
                if done % 6 == 0:
                    print(f"{pair} {year}-{month:02d} done "
                          f"(months={done}, gaps={len(gaps)})", flush=True)
        print(f"== {pair} complete (gaps so far {len(gaps)})", flush=True)
        (ROOT / "manifest.json").write_text(json.dumps({
            "source": BASE, "granularity": "min_1",
            "parts": len(list(parts.glob("*.parquet"))), "gaps_unavailable": gaps,
            "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"all done: months={done} gaps={len(gaps)}")


if __name__ == "__main__":
    main()
