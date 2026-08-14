#!/usr/bin/env python3
"""Collect USDJPY hour-00-UTC (09:00-10:00 JST) tick files for gotobi + control days.

Why ticks, and why only this hour: the hourly test showed the 09:00->10:00 JST
mid-to-mid drift is ~zero on gotobi days — but the classic fix pattern is a rise
INTO 9:55 followed by a reversal, which nets out inside one hourly bar. Only
sub-hourly data can test the classic trade (buy 09:00, sell at the 09:55 fix).
Full minute collection is ~65k requests under the current throttle (~15 days);
this fetches exactly one tick file per relevant day:

  gotobi days (5/10/15/20/25/30 rolled back)  ≈ 1,100 days over 2011-2026
  matched controls (nearest prior non-gotobi business day) ≈ 1,100 days
  → ~2,200 requests ≈ 9-18h at the current 11-28s/request throttle.

Tick record (verified 2026-08-05): 20 bytes '>3i2f' = (ms offset, ask*1000,
bid*1000, askVol, bidVol) for JPY pairs. Resumable via scanned.jsonl.
"""
from __future__ import annotations

import datetime as dt
import json
import lzma
import struct
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://datafeed.dukascopy.com/datafeed"
PAIR, SCALE = "USDJPY", 1e3
ROOT = Path("data/fx_ticks_fix")
SCANS = ROOT / "scanned.jsonl"
REC = struct.Struct(">3i2f")


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
    return b"__UNAVAILABLE__"


def target_days() -> list[tuple[str, str]]:
    """(day, kind) — gotobi settlement days and their matched controls."""
    from scripts.experiment_fx_gotobi import gotobi_days, jp_business_days
    bdays = jp_business_days()
    got = gotobi_days(bdays, "2011-01-01", "2026-07-31")
    bset = sorted(d for d in bdays.date if dt.date(2011, 1, 1) <= d <= dt.date(2026, 7, 31))
    out = {}
    for g in sorted(got):
        out[g] = "gotobi"
        # 対照: 直前の非ゴトー営業日（決定論的・同一週になりやすい）
        i = max(j for j, d in enumerate(bset) if d <= g)
        for j in range(i - 1, -1, -1):
            if bset[j] not in got and bset[j] not in out:
                out[bset[j]] = "control"
                break
    return [(d.isoformat(), k) for d, k in sorted(out.items())]


def parse_hour(raw: bytes, day: dt.date) -> pd.DataFrame:
    try:
        blob = lzma.LZMADecompressor(format=lzma.FORMAT_ALONE).decompress(raw)
    except lzma.LZMAError:
        return pd.DataFrame()
    base = dt.datetime(day.year, day.month, day.day, 0, 0, tzinfo=dt.timezone.utc)
    rows = []
    for i in range(len(blob) // REC.size):
        ms, a, b, av, bv = REC.unpack_from(blob, i * REC.size)
        rows.append((base + dt.timedelta(milliseconds=ms), a / SCALE, b / SCALE))
    return pd.DataFrame(rows, columns=["ts", "ask", "bid"])


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    done = set()
    if SCANS.exists():
        done = {json.loads(l)["day"] for l in SCANS.read_text().splitlines() if l.strip()}
    targets = [(d, k) for d, k in target_days() if d not in done]
    print(f"pending={len(targets)} done={len(done)}", flush=True)
    buf: dict[str, list] = {}
    for n, (ds, kind) in enumerate(targets, 1):
        day = dt.date.fromisoformat(ds)
        raw = _get(f"{BASE}/{PAIR}/{day.year}/{day.month - 1:02d}/{day.day:02d}/00h_ticks.bi5")
        status = "missing"
        if raw == b"__UNAVAILABLE__":
            status = "unavailable"          # スキャン済みにしない＝再実行で埋まる
        elif raw:
            df = parse_hour(raw, day)
            if len(df):
                df["kind"] = kind
                key = f"{day.year}-{day.month:02d}"
                buf.setdefault(key, []).append(df)
                status = f"ok:{len(df)}"
        if status != "unavailable":
            with SCANS.open("a") as fh:
                fh.write(json.dumps({"day": ds, "kind": kind, "status": status}) + "\n")
        if n % 20 == 0 or n == len(targets):
            for key, frames in list(buf.items()):
                out = ROOT / f"{PAIR}_{key}.parquet"
                old = [pd.read_parquet(out)] if out.exists() else []
                pd.concat(old + frames, ignore_index=True).drop_duplicates(
                    ["ts"]).sort_values("ts").to_parquet(out, index=False)
                del buf[key]
            print(f"{n}/{len(targets)} {ds} {status}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
