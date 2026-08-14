#!/usr/bin/env python3
"""Collect free 1-minute FX bars from HistData.com (second source, no throttle).

Dukascopy is throttling this IP (11-28s/request), which makes its per-day minute
files (~65k requests) infeasible. HistData serves one ZIP per pair-year via a
tk-token POST — 7 pairs x 16 years ≈ 230 requests total.

Caveats (recorded in the manifest):
  * Timestamps are **EST fixed, no DST** (UTC-5 always) -> converted to UTC here.
  * M1 bars are **bid-quote based**; no ask, so no spread. Costs must come from
    a cost model (GMO 0.2sen etc.) or Dukascopy's measured spreads.
  * Data starts ~2000s depending on pair; volume column is tick count.

Format inside the ZIP: DT;O;H;L;C;V with DT = "YYYYMMDD HHMMSS".
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import re
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

PAIRS = ("USDJPY", "EURUSD", "GBPUSD", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD")
ROOT = Path("data/fx_histdata_min")
PAGE = ("https://www.histdata.com/download-free-forex-historical-data/"
        "?/ascii/1-minute-bar-quotes/{pair}/{ym}")
GET = "https://www.histdata.com/get.php"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                         "AppleWebKit/537.36 Chrome/126.0 Safari/537.36"}


def fetch_zip(session, pair: str, ym: str) -> bytes | None:
    """GET the page for its tk token, then POST get.php. ym = '2015' or '2026/7'.

    ★requests はクエリ文字列 `?/ascii/...` の `/` を再エンコードし、サーバが
    フォーム無しの別ページを返す（実際に全滅した）。curl はそのまま送るので
    subprocess の curl で GET/POST する。
    """
    import subprocess, tempfile, os
    url = PAGE.format(pair=pair.lower(), ym=ym)
    ua = HEADERS["User-Agent"]
    with tempfile.TemporaryDirectory() as td:
        cj = os.path.join(td, "cj.txt")
        g = subprocess.run(["curl", "-sS", "-c", cj, "-A", ua, "--max-time", "60", url],
                           capture_output=True)
        html = g.stdout.decode("utf-8", errors="replace")
        m = re.search(r'name="tk"[^>]*value="([0-9a-f]+)"', html)
        if not m:
            return None                      # データ無し or クールダウン中
        fields = {"tk": m.group(1)}
        for k in ("date", "datemonth", "platform", "timeframe", "fxpair"):
            mm = re.search(rf'name="{k}"[^>]*value="([^"]*)"', html)
            fields[k] = mm.group(1) if mm else ""
        data = "&".join(f"{k}={v}" for k, v in fields.items())
        pz = subprocess.run(["curl", "-sS", "-b", cj, "-A", ua, "-e", url,
                             "--max-time", "180", "--data", data, GET],
                            capture_output=True)
        return pz.stdout if pz.stdout.startswith(b"PK") else None


def parse_zip(raw: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not csvs:
            return pd.DataFrame()
        body = z.read(csvs[0]).decode("ascii", errors="replace")
    rows = []
    for line in body.splitlines():
        p = line.strip().split(";")
        if len(p) < 5:
            continue
        rows.append((p[0], float(p[1]), float(p[2]), float(p[3]), float(p[4])))
    d = pd.DataFrame(rows, columns=["dt", "open", "high", "low", "close"])
    # ★HistData は EST固定（夏時間なし）= UTC-5。UTCへ変換して保存する。
    d["ts"] = pd.to_datetime(d["dt"], format="%Y%m%d %H%M%S") + pd.Timedelta(hours=5)
    return d[["ts", "open", "high", "low", "close"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, default=2011)
    ap.add_argument("--end-year", type=int, default=2026)
    ap.add_argument("--sleep", type=float, default=2.0)
    args = ap.parse_args()
    parts = ROOT / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    ses = requests.Session()
    log = []
    today = dt.date.today()
    for pair in PAIRS:
        for year in range(args.start_year, args.end_year + 1):
            targets = [str(year)] if year < today.year else [
                f"{year}/{m}" for m in range(1, today.month)]
            for ym in targets:
                tag = ym.replace("/", "-")
                out = parts / f"{pair}_{tag}.parquet"
                if out.exists():
                    continue
                try:
                    raw = fetch_zip(ses, pair, ym)
                except requests.RequestException as e:
                    log.append(f"ERR {pair} {ym} {e}")
                    continue
                if raw is None:
                    log.append(f"MISS {pair} {ym}")
                    continue
                d = parse_zip(raw)
                if len(d):
                    d.to_parquet(out, index=False)
                    print(f"{pair} {ym}: {len(d):,} rows", flush=True)
                time.sleep(args.sleep)
    (ROOT / "manifest.json").write_text(json.dumps({
        "source": "histdata.com ASCII M1 (bid-based bars, EST-fixed converted to UTC)",
        "pairs": list(PAIRS), "issues": log,
        "caveats": ["bid-based: no ask/spread — use GMO cost model or Dukascopy spreads",
                    "timestamps converted EST(UTC-5 fixed) -> UTC"],
        "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"done issues={len(log)}")


if __name__ == "__main__":
    main()
