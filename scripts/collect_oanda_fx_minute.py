#!/usr/bin/env python3
"""Collect M1 bid/ask candles for the major USD pairs from the OANDA v20 API.

READ-ONLY: uses only /v3/instruments/{pair}/candles. Never touches account or
order endpoints — the configured token belongs to a live account.

Why OANDA: Dukascopy throttles this IP (11-28s/request → minute data ≈ 15 days);
OANDA serves 5,000 candles per request with no comparable throttle, so the same
15.5 years x 7 pairs is ~8k requests ≈ a couple of hours. History verified back
to 2011-01-03 with both bid and ask.

Storage: data/fx_oanda_min/parts/{PAIR}_{YYYY}.parquet (resumable per pair-year,
~2GB total, git-ignored). Columns: ts, open/high/low/close (mid), close_bid,
close_ask, volume (tick count).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests

from data.collectors.config import _load_local_env

_load_local_env()
BASE = "https://api-fxtrade.oanda.com"          # live host; candles は read-only
PAIRS = ("USD_JPY", "EUR_USD", "GBP_USD", "AUD_USD", "USD_CHF", "USD_CAD", "NZD_USD")
ROOT = Path("data/fx_oanda_min")


def _get(path: str, params: dict, tries: int = 6) -> dict:
    H = {"Authorization": f"Bearer {os.environ['OANDA_TOKEN']}"}
    for i in range(tries):
        try:
            r = requests.get(BASE + path, params=params, headers=H, timeout=60)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 502, 503):
                time.sleep(2 ** i)
                continue
            r.raise_for_status()
        except requests.RequestException:
            if i == tries - 1:
                raise
            time.sleep(2 ** i)
    raise RuntimeError("oanda retry exhausted")


def collect_year(pair: str, year: int) -> pd.DataFrame:
    frames, cur = [], f"{year}-01-01T00:00:00Z"
    end = f"{year + 1}-01-01T00:00:00Z"
    while True:
        j = _get(f"/v3/instruments/{pair}/candles",
                 {"granularity": "M1", "price": "BA", "count": 5000, "from": cur})
        cs = [c for c in j.get("candles", []) if c.get("complete")]
        if not cs:
            break
        rows = []
        for c in cs:
            if c["time"] >= end:
                break
            b, a = c["bid"], c["ask"]
            rows.append((c["time"],
                         (float(b["o"]) + float(a["o"])) / 2,
                         (float(b["h"]) + float(a["h"])) / 2,
                         (float(b["l"]) + float(a["l"])) / 2,
                         (float(b["c"]) + float(a["c"])) / 2,
                         float(b["c"]), float(a["c"]), int(c["volume"])))
        if rows:
            frames.append(pd.DataFrame(rows, columns=[
                "ts", "open", "high", "low", "close", "close_bid", "close_ask", "volume"]))
        last = cs[-1]["time"]
        if last >= end or len(cs) < 2:
            break
        cur = last
        time.sleep(0.08)
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True).drop_duplicates("ts")
    d["ts"] = pd.to_datetime(d["ts"])
    return d.sort_values("ts").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, default=2011)
    ap.add_argument("--end-year", type=int, default=dt.date.today().year)
    ap.add_argument("--pairs", default=",".join(PAIRS))
    args = ap.parse_args()
    parts = ROOT / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    for pair in [p.strip() for p in args.pairs.split(",") if p.strip()]:
        for year in range(args.start_year, args.end_year + 1):
            out = parts / f"{pair}_{year}.parquet"
            if out.exists():
                continue
            d = collect_year(pair, year)
            if len(d):
                d.to_parquet(out, index=False)
                print(f"{pair} {year}: {len(d):,} rows", flush=True)
    (ROOT / "manifest.json").write_text(json.dumps({
        "source": f"{BASE} /v3/instruments/candles M1 price=BA (read-only)",
        "pairs": list(PAIRS),
        "note": "mid OHLC + close bid/ask; volume = tick count; times UTC",
        "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "parts": sorted(p.name for p in parts.glob("*.parquet")),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("done")


if __name__ == "__main__":
    main()
