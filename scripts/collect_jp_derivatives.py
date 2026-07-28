"""Collect Premium context: index-futures (先物, incl. night session), all
indices, short-selling ratios. Robust + resumable (per-year cache, 429 backoff).
Run SOLO after the equity collector to avoid rate limits.

Futures 'Date' bundles the day session (M*) and the following night session (E*),
so EC/MC-1 on date D is the overnight (US-spanning) move that hits the next cash
open — a clean market-level overnight factor.
"""
import time
from pathlib import Path

import pandas as pd

import jquantsapi
from data.collectors.config import JQUANTS_API_KEY

import datetime as _dt
START, END = "2018-01-01", _dt.date.today().isoformat()  # ENDは常に当日（不足日のみ取得）
FUT_PRODUCTS = ("NK225F", "NK225MF", "TOPIXF", "JN400F", "DJIAF", "NKVIF", "REITF")
FUT_COLS = ["Code", "ProdCat", "Date", "O", "H", "L", "C",
            "MO", "MC", "EO", "EC", "Vo", "OI", "Settle", "LTD"]
FUT_NUM = ["O", "H", "L", "C", "MO", "MC", "EO", "EC", "Vo", "OI", "Settle"]
OUT = Path("data/jp_derivatives")


def _clean_futures(f: pd.DataFrame) -> pd.DataFrame:
    """Coerce numeric columns; sessions with no night trade come back as ''."""
    f = f.copy()
    for c in FUT_NUM:
        if c in f.columns:
            f[c] = pd.to_numeric(f[c], errors="coerce")
    return f


def _retry(fn, tries=8, base=2.0):
    for i in range(tries):
        try:
            return fn()
        except Exception:  # noqa: BLE001
            if i == tries - 1:
                raise
            time.sleep(min(base ** i, 90))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    client = jquantsapi.ClientV2(api_key=JQUANTS_API_KEY)
    cal = _retry(lambda: client.get_mkt_calendar(from_yyyymmdd=START.replace("-", ""),
                                                 to_yyyymmdd=END.replace("-", "")))
    days = cal[cal["HolDiv"].astype(str).isin(["1", "2"])]["Date"].astype(str).tolist()
    by_year: dict[str, list[str]] = {}
    for d in days:
        by_year.setdefault(d[:4], []).append(d)

    for year, ydays in sorted(by_year.items()):
        fpath = OUT / f"futures_{year}.parquet"
        ipath = OUT / f"indices_{year}.parquet"
        spath = OUT / f"short_ratio_{year}.parquet"
        fut = [pd.read_parquet(fpath)] if fpath.exists() else []
        idx = [pd.read_parquet(ipath)] if ipath.exists() else []
        shr = [pd.read_parquet(spath)] if spath.exists() else []
        have = set()
        if fut:
            have = set(pd.to_datetime(fut[0]["Date"]).dt.strftime("%Y-%m-%d"))
        todo = [d for d in ydays if d not in have]
        if not todo:
            print(f"year {year}: complete, skip", flush=True)
            continue

        def _flush():
            if fut:
                pd.concat(fut, ignore_index=True).drop_duplicates(["Date", "Code"]).to_parquet(fpath, index=False)
            if idx:
                pd.concat(idx, ignore_index=True).drop_duplicates(["Date", "Code"]).to_parquet(ipath, index=False)
            if shr:
                pd.concat(shr, ignore_index=True).to_parquet(spath, index=False)

        for i, d in enumerate(todo, 1):
            ymd = d.replace("-", "")
            try:
                f = _retry(lambda: client.get_drv_bars_daily_fut(date_yyyymmdd=ymd))
                if len(f):
                    f = f[f["ProdCat"].astype(str).isin(FUT_PRODUCTS)]
                    fut.append(_clean_futures(f[[c for c in FUT_COLS if c in f.columns]]))
            except Exception as exc:  # noqa: BLE001
                print(f"  {year} fut MISS {d}: {str(exc)[:50]}", flush=True)
            try:
                x = _retry(lambda: client.get_idx_bars_daily(date_yyyymmdd=ymd))
                if len(x):
                    idx.append(x)
            except Exception:  # noqa: BLE001
                pass
            try:
                s = _retry(lambda: client.get_mkt_short_ratio(date_yyyymmdd=ymd))
                if len(s):
                    shr.append(s)
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.12)
            if i % 50 == 0:
                _flush()
                print(f"  {year} {i}/{len(todo)} fut_rows={sum(map(len,fut))} (flushed)", flush=True)
        _flush()
        print(f"year {year}: fut_rows={sum(map(len,fut))} idx_rows={sum(map(len,idx))}", flush=True)

    # NOTE: 結合ファイルは作らない（年次ファイルが正本。読み手はglobでユニオンする）。
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
