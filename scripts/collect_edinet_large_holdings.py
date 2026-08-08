#!/usr/bin/env python3
"""Collect EDINET large-holding (5% rule) filing METADATA -- docTypeCode 350.

EDINET's public retention for large-holding reports is 5 years and ROLLS
DAILY, so history erodes; collect now, append forever (same urgency as the
earnings-calendar snapshots). Metadata only (filer, issuer, description,
submit time) -- enough for the new-filing event study; holding percentages
would need per-document XBRL downloads (not done here).

Resumable: one JSON-lines file per month under data/jp_large_holdings/,
already-collected dates are skipped. ~2 req/sec against the v2 API.
"""
from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

import requests

from data.collectors.config import EDINET_API_KEY

BASE = "https://api.edinet-fsa.go.jp/api/v2"
ROOT = Path("data/jp_large_holdings")
START = dt.date(2021, 9, 1)
FIELDS = ["docID", "docTypeCode", "docDescription", "filerName", "edinetCode",
          "issuerEdinetCode", "secCode", "submitDateTime", "periodStart",
          "periodEnd", "withdrawalStatus"]


def collected_dates() -> set[str]:
    seen = set()
    for f in ROOT.glob("large_holdings_*.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                seen.add(json.loads(line)["_list_date"])
    return seen


def fetch_day(ds: str, tries: int = 6) -> list[dict]:
    for attempt in range(tries):
        try:
            r = requests.get(f"{BASE}/documents.json",
                             params={"date": ds, "type": 2,
                                     "Subscription-Key": EDINET_API_KEY}, timeout=60)
            r.raise_for_status()
            return r.json().get("results", []) or []
        except Exception:  # noqa: BLE001
            if attempt == tries - 1:
                raise
            time.sleep(min(2.0 ** attempt, 60))
    return []


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    seen = collected_dates()
    day = START
    today = dt.date.today()
    n_days = n_rows = 0
    while day <= today:
        ds = day.isoformat()
        if ds not in seen:
            rows = [{k: d.get(k) for k in FIELDS} | {"_list_date": ds}
                    for d in fetch_day(ds) if d.get("docTypeCode") == "350"]
            out = ROOT / f"large_holdings_{ds[:7]}.jsonl"
            with out.open("a", encoding="utf-8") as fh:
                if not rows:      # record the empty day so resume skips it
                    fh.write(json.dumps({"_list_date": ds, "_empty": True}) + "\n")
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_days += 1
            n_rows += len(rows)
            time.sleep(0.35)
        day += dt.timedelta(days=1)
    print(json.dumps({"fetched_days": n_days, "rows": n_rows}))


if __name__ == "__main__":
    main()
