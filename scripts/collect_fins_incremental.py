#!/usr/bin/env python3
"""Incremental J-Quants fins/summary collector (V4 forward-seal data feed).

The static snapshot data/cache/fins_20260424* ends at DiscDate=2026-04-24.
This script fetches /v2/fins/summary by date from the last collected day
onward and merges rows into data/cache/fins_incremental.json.gz using the
same dict[code -> list[row]] layout, so run_value_event_v1.load_fins() picks
everything up unchanged (it dedups on Code/DiscDate/DiscTime/DiscNo).

The cursor is derived from the incremental file itself (max DiscDate, falling
back to the snapshot end), and the cursor day is always re-fetched so a run
interrupted mid-day cannot leave a hole. Run SOLO (J-Quants 429 policy).
"""
from __future__ import annotations

import datetime as dt
import gzip
import json
import time
from pathlib import Path

import requests

from data.collectors.config import JQUANTS_API_KEY, JQUANTS_BASE

OUT = Path("data/cache/fins_incremental.json.gz")
SNAPSHOT_END = dt.date(2026, 4, 24)
HEADERS = {"x-api-key": JQUANTS_API_KEY}


def _load_existing() -> dict[str, list]:
    if not OUT.exists():
        return {}
    with gzip.open(OUT, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def _cursor(payload: dict[str, list]) -> dt.date:
    latest = max((r.get("DiscDate", "") for rows in payload.values() for r in rows),
                 default="")
    if not latest:
        return SNAPSHOT_END
    return dt.date.fromisoformat(latest[:10])


def _fetch_day(day: dt.date, tries: int = 9) -> list[dict]:
    params = {"date": day.strftime("%Y%m%d")}
    rows: list[dict] = []
    for attempt in range(tries):
        try:
            pagination = None
            rows = []
            while True:
                p = dict(params, **({"pagination_key": pagination} if pagination else {}))
                r = requests.get(JQUANTS_BASE + "/v2/fins/summary", headers=HEADERS,
                                 params=p, timeout=60)
                r.raise_for_status()
                j = r.json()
                rows.extend(j.get("data", []))
                pagination = j.get("pagination_key")
                if not pagination:
                    return rows
        except Exception:  # noqa: BLE001 - throttle/5xx: back off and retry
            if attempt == tries - 1:
                raise
            time.sleep(min(3.0 * 2.0 ** attempt, 300))
    return rows


def _save(payload: dict[str, list]) -> None:
    tmp = OUT.with_suffix(".gz.tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    tmp.replace(OUT)


def main() -> None:
    payload = _load_existing()
    start = _cursor(payload)          # re-fetch the cursor day (dedup makes it safe)
    today = dt.date.today()
    day, fetched_days, fetched_rows = start, 0, 0
    seen = {(r.get("Code"), r.get("DiscDate"), r.get("DiscTime"), r.get("DiscNo"))
            for rows_ in payload.values() for r in rows_}
    try:
        while day <= today:
            for r in _fetch_day(day):
                key = (r.get("Code"), r.get("DiscDate"), r.get("DiscTime"), r.get("DiscNo"))
                if key in seen:
                    continue
                seen.add(key)
                payload.setdefault(str(r.get("Code")), []).append(r)
                fetched_rows += 1
            fetched_days += 1
            day += dt.timedelta(days=1)
            if fetched_days % 30 == 0:
                _save(payload)        # checkpoint so a mid-run 429 loses nothing
            time.sleep(0.5)           # stay well inside the shared rate limit
    finally:
        _save(payload)               # partial progress survives; cursor resumes
    print(json.dumps({"cursor_start": start.isoformat(), "days": fetched_days,
                      "new_rows": fetched_rows,
                      "total_rows": sum(len(v) for v in payload.values())}))


if __name__ == "__main__":
    main()
