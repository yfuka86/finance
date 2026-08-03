#!/usr/bin/env python3
"""Collect ECB daily euro foreign-exchange reference rates.

Primary source, free, no key, 1999-01-04 onwards, published every TARGET day.
https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip

★これは**14:15 CET の参照レート（仲値）**であって約定可能な価格ではない。
研究用のプロキシとしては使えるが、実弾判定にはブローカーの実レートが要る
（オプションの清算値と同じ扱い＝合格しても PAPER ONLY）。
"""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
import datetime as dt

import pandas as pd
import requests

URL = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip"
MAJORS = ("USD", "JPY", "GBP", "CHF", "AUD", "CAD", "NZD")   # EUR は基準通貨
ROOT = Path("data/fx_ecb")


def fetch() -> tuple[pd.DataFrame, str]:
    r = requests.get(URL, timeout=120)
    r.raise_for_status()
    digest = hashlib.sha256(r.content).hexdigest()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        name = z.namelist()[0]
        raw = z.read(name)
    d = pd.read_csv(io.BytesIO(raw))
    d.columns = [c.strip() for c in d.columns]
    d["Date"] = pd.to_datetime(d["Date"])
    keep = ["Date", *MAJORS]
    d = d[keep].apply(lambda s: pd.to_numeric(s, errors="coerce") if s.name != "Date" else s)
    return d.dropna().sort_values("Date").reset_index(drop=True), digest


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    d, digest = fetch()
    out = ROOT / "eurofxref_hist.parquet"
    d.to_parquet(out, index=False)
    (ROOT / "manifest.json").write_text(json.dumps({
        "source": URL, "sha256": digest, "rows": len(d),
        "start": str(d["Date"].min().date()), "end": str(d["Date"].max().date()),
        "currencies": ["EUR", *MAJORS],
        "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "caveat": "ECB reference rates are 14:15 CET fixings, not tradable quotes",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"rows={len(d)} {d['Date'].min().date()}..{d['Date'].max().date()} sha256={digest[:12]}")


if __name__ == "__main__":
    main()
