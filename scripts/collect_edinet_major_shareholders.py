#!/usr/bin/env python3
"""Resumable append-only collector for J-Quants /v2/edinet/major-shareholders.

有価証券報告書の「大株主の状況」(上位10名) を提出日単位で収集する。
PIT時刻は SubDate + SubTime (提出日時)。当該日のレスポンスをそのまま保存し、
同じ日を二度書かない (scanned_dates.jsonl で再開可能)。
"""
from __future__ import annotations
import argparse, datetime as dt, hashlib, json, time
from pathlib import Path
import requests
from data.collectors.config import JQUANTS_API_KEY, JQUANTS_BASE

URL = f"{JQUANTS_BASE}/v2/edinet/major-shareholders"
ROOT = Path("data/jp_ownership")
SCANS = ROOT / "scanned_dates.jsonl"
DOCS = ROOT / "filings.jsonl"


def _get(date_str: str, tries: int = 6):
    """Single-day fetch with backoff. Returns (records, raw_bytes)."""
    for i in range(tries):
        try:
            r = requests.get(URL, headers={"x-api-key": JQUANTS_API_KEY},
                             params={"date": date_str}, timeout=60)
            if r.status_code == 200:
                return r.json().get("data", []), r.content
            if r.status_code != 429:
                r.raise_for_status()
        except requests.RequestException:
            if i == tries - 1:
                raise
        time.sleep(min(2 ** i, 30))
    raise RuntimeError(f"J-Quants retry exhausted for {date_str}")


def _scanned() -> set[str]:
    if not SCANS.exists():
        return set()
    return {json.loads(line)["date"] for line in
            SCANS.read_text(encoding="utf-8").splitlines() if line.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    args = ap.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)

    done = _scanned()
    dates, cur, end = [], dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end)
    while cur <= end:
        if cur.weekday() < 5 and cur.isoformat() not in done:
            dates.append(cur)
        cur += dt.timedelta(days=1)
    print(f"dates_pending={len(dates)} already_scanned={len(done)}", flush=True)

    total = 0
    for n, d in enumerate(dates, 1):
        ds = d.isoformat()
        records, raw = _get(d.strftime("%Y%m%d"))
        if records:
            with DOCS.open("a", encoding="utf-8") as fh:
                for row in records:
                    row["_fetch_date"] = ds
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        with SCANS.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "date": ds, "records": len(records),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            }, ensure_ascii=False) + "\n")
        total += len(records)
        if n % 100 == 0 or n == len(dates):
            print(f"  {n}/{len(dates)} {ds} filings_so_far={total}", flush=True)
        time.sleep(0.12)
    print(f"done pending_processed={len(dates)} new_filings={total}", flush=True)


if __name__ == "__main__":
    main()
