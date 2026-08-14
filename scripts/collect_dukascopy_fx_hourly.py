#!/usr/bin/env python3
"""Collect Dukascopy hourly bid/ask (monthly files) for selected pairs, resumable.

Hourly monthly files are 30x fewer requests than minute day-files, and the hour
containing the 9:55 JST Tokyo fix (00:00-01:00 UTC) is enough for a first pass at
the gotobi anomaly. Use while the datafeed is throttling us (~10-30s/request).
"""
from __future__ import annotations
import argparse, datetime as dt, json, time
from pathlib import Path
import pandas as pd
from scripts.collect_dukascopy_fx import month_frame, Unavailable

ROOT = Path("data/fx_dukascopy_hour")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="USDJPY")
    ap.add_argument("--start-year", type=int, default=2011)
    ap.add_argument("--end-year", type=int, default=2026)
    ap.add_argument("--sleep", type=float, default=0.3)
    args = ap.parse_args()
    parts = ROOT / "parts"; parts.mkdir(parents=True, exist_ok=True)
    gaps = []
    for pair in [p.strip().upper() for p in args.pairs.split(",") if p.strip()]:
        for year in range(args.start_year, args.end_year + 1):
            out = parts / f"{pair}_{year}.parquet"
            if out.exists():
                continue
            frames = []
            for m0 in range(12):
                if dt.date(year, m0 + 1, 1) >= dt.date.today():
                    break
                try:
                    bid = month_frame(pair, year, m0, "BID")
                    ask = month_frame(pair, year, m0, "ASK")
                except Unavailable as e:
                    gaps.append(str(e)); continue
                if bid.empty or ask.empty:
                    continue
                m = bid.merge(ask, on="ts", suffixes=("_bid", "_ask"))
                m["mid"] = (m["close_bid"] + m["close_ask"]) / 2
                m["spread"] = m["close_ask"] - m["close_bid"]
                frames.append(m); time.sleep(args.sleep)
            if frames:
                pd.concat(frames, ignore_index=True).to_parquet(out, index=False)
                print(f"{pair} {year}: {sum(len(f) for f in frames):,} rows", flush=True)
        (ROOT / "manifest.json").write_text(json.dumps(
            {"pairs": args.pairs, "gaps": gaps,
             "updated_at": dt.datetime.now(dt.timezone.utc).isoformat()},
            ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"done gaps={len(gaps)}")

if __name__ == "__main__":
    main()
