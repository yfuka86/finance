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


class Unavailable(Exception):
    """レート制限などで取得できなかった（データが無いのとは区別する）."""


def _get(url: str, tries: int = 9) -> bytes | None:
    """★Dukascopy は連続取得で接続を切ってくる。落とさずギャップとして記録する。

    以前は6回で例外を投げて**全体が停止し、途中経過も保存されない**設計だった
    （EURUSD 2017/07 で実際に落ちた）。ここは長いバックオフで粘り、それでも
    駄目なら Unavailable を投げて呼び出し側がスキップ・記録できるようにする。
    """
    import random
    for i in range(tries):
        try:
            r = requests.get(url, timeout=90)
            if r.status_code == 404:
                return None                      # その月のデータ無し（上場前など）
            if r.status_code == 200:
                return r.content
        except requests.RequestException:
            pass
        time.sleep(min(3 * 2 ** i, 120) + random.uniform(0, 2))
    raise Unavailable(url)


def year_frame(pair: str, year: int, side: str) -> pd.DataFrame:
    """One yearly file -> daily OHLC for that side.

    ★週次リバランスの戦略に時間足は過剰で、Dukascopy のレート制限に正面から
    当たるだけだった（月次×2×7ペア×16年＝2,520リクエスト）。日足は年次ファイル
    なので **224リクエスト**で済み、同じ bid/ask が得られる。
    """
    raw = _get(f"{BASE}/{pair}/{year}/{side}_candles_day_1.bi5")
    if not raw:
        return pd.DataFrame()
    try:
        blob = lzma.LZMADecompressor(format=lzma.FORMAT_ALONE).decompress(raw)
    except lzma.LZMAError:
        return pd.DataFrame()
    start = dt.datetime(year, 1, 1, tzinfo=dt.timezone.utc)
    sc = SCALE[pair]
    rows = []
    for i in range(len(blob) // REC.size):
        t, o, c, l, h, v = REC.unpack_from(blob, i * REC.size)
        if o == 0 and c == 0 and v == 0:
            continue
        rows.append((start + dt.timedelta(seconds=t), o / sc, h / sc, l / sc, c / sc, v))
    return pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])


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
    ap.add_argument("--sleep", type=float, default=0.6)
    args = ap.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    parts = ROOT / "parts"
    parts.mkdir(exist_ok=True)
    gaps = []
    for pair in PAIRS:
        for year in range(args.start_year, args.end_year + 1):
            out = parts / f"{pair}_{year}.parquet"
            if out.exists():
                continue                          # ★年単位で再開可能
            frames = []
            try:
                bid = year_frame(pair, year, "BID")
                ask = year_frame(pair, year, "ASK")
            except Unavailable as e:
                gaps.append(str(e)); print(f"  GAP {pair} {year}", flush=True); continue
            if not bid.empty and not ask.empty:
                m = bid.merge(ask, on="ts", suffixes=("_bid", "_ask"))
                m["spread"] = m["close_ask"] - m["close_bid"]
                m["mid"] = (m["close_ask"] + m["close_bid"]) / 2
                frames.append(m)
            time.sleep(args.sleep)
            if frames:
                pd.concat(frames, ignore_index=True).to_parquet(out, index=False)
                print(f"{pair} {year}: saved {sum(len(f) for f in frames):,} rows", flush=True)
    # 年別パーツを結合
    manifest = {}
    for pair in PAIRS:
        fs = sorted(parts.glob(f"{pair}_*.parquet"))
        if not fs:
            continue
        d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
        d = d.drop_duplicates("ts").sort_values("ts")
        d.to_parquet(ROOT / f"{pair}_day.parquet", index=False)
        manifest[pair] = {"rows": len(d), "start": str(d.ts.min()), "end": str(d.ts.max()),
                          "median_spread": float(d.spread.median()),
                          "years": len(fs)}
        print(f"{pair}: {len(d):,} rows {d.ts.min()} .. {d.ts.max()}", flush=True)
    (ROOT / "manifest.json").write_text(json.dumps(
        {"source": BASE, "pairs": list(PAIRS), "granularity": "day",
         "note": "Dukascopy ECN bid/ask candles; MM is 0-indexed in the URL",
         "gaps_unavailable": gaps,
         "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(),
         "pairs_detail": manifest}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"manifest written / gaps={len(gaps)}")


if __name__ == "__main__":
    main()
