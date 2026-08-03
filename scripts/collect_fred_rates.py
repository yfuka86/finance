#!/usr/bin/env python3
"""Collect 3-month interbank rates for the eight majors from FRED (no API key).

Why 3-month interbank and not the policy rate: FX swap points are priced off the
short-term money market, not the central bank's target. The interbank series also
happen to be the only ones that stay current for every currency — the policy-rate
series (IRSTCI01*) go stale at 2024-03 for CHF and 2024-12 for NZD, which would
silently break the carry proxy for the most recent two years.

Monthly. Forward-filled to daily downstream; short rates move slowly enough that
this is not the binding approximation — the broker's own spread is.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
from pathlib import Path

import pandas as pd
import requests

SERIES = {
    "USD": "IR3TIB01USM156N", "JPY": "IR3TIB01JPM156N", "EUR": "IR3TIB01EZM156N",
    "GBP": "IR3TIB01GBM156N", "CHF": "IR3TIB01CHM156N", "AUD": "IR3TIB01AUM156N",
    "CAD": "IR3TIB01CAM156N", "NZD": "IR3TIB01NZM156N",
}
URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
ROOT = Path("data/fx_rates")


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    cols, meta = {}, {}
    for ccy, sid in SERIES.items():
        r = requests.get(URL.format(sid=sid), timeout=60)
        r.raise_for_status()
        d = pd.read_csv(io.StringIO(r.text))
        d.columns = ["date", "value"]
        d["date"] = pd.to_datetime(d["date"])
        d["value"] = pd.to_numeric(d["value"], errors="coerce") / 100.0   # % -> 小数
        d = d.dropna().set_index("date")["value"]
        cols[ccy] = d
        meta[ccy] = {"series_id": sid, "rows": int(len(d)),
                     "start": str(d.index.min().date()), "end": str(d.index.max().date()),
                     "sha256": hashlib.sha256(r.content).hexdigest()}
        print(f"{ccy} {sid}: {len(d)} rows {d.index.min().date()}..{d.index.max().date()}")
    out = pd.DataFrame(cols).sort_index()
    out.to_parquet(ROOT / "short_rates_monthly.parquet")
    (ROOT / "manifest.json").write_text(json.dumps({
        "source": "FRED fredgraph.csv (no key)", "frequency": "monthly",
        "definition": "3-month interbank offered rate, decimal (0.05 = 5%)",
        "note": "policy-rate series IRSTCI01* go stale (CHF 2024-03, NZD 2024-12); "
                "interbank stays current and is the better swap proxy anyway",
        "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(), "series": meta,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {out.shape} -> {ROOT/'short_rates_monthly.parquet'}")


if __name__ == "__main__":
    main()
