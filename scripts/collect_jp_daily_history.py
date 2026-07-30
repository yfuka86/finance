"""Idempotent canonical daily-bar collector (split-adjusted).

Never downloads a date twice: it scans EVERY on-disk source (screener day
snapshots, the intraday reference window, and previously collected years),
computes which trading days are already present, and fetches ONLY the missing
ones. Re-running after completion issues zero API calls.

Newly fetched days are cached per-year under data/jp_daily_history/. The existing
screener/reference files are left in place (not re-copied), so no date is stored
twice either. trading.jp_intraday.daily_gap.load_existing_daily() unions all
sources at read time. Run SOLO (concurrent collectors trip 429s).

Window default 2021-09-01..2026-07-24; Premium allows up to 20y — widen START to
extend. Training is done locally, so everything lands on this machine.
"""
import datetime as _dt
import glob
import time
from pathlib import Path

import pandas as pd

import jquantsapi
from data.collectors.config import JQUANTS_API_KEY

START, END = "2018-01-01", _dt.date.today().isoformat()  # ENDは常に当日（不足日のみ取得）
OUT = Path("data/jp_daily_history")
# Keep RAW open/close too (O/C) for accurate ¥ unit-lot sizing, alongside adjusted.
KEEP = ["Date", "Code", "O", "C", "AdjO", "AdjH", "AdjL", "AdjC", "AdjVo", "Va"]
MIN_ROWS = 3000  # a real full-market day has ~4000+ names; below this = partial


def _retry(fn, tries=8, base=2.0):
    for i in range(tries):
        try:
            return fn()
        except Exception:  # noqa: BLE001
            if i == tries - 1:
                raise
            time.sleep(min(base ** i, 90))


def _covered_dates() -> set:
    """Dates already present (>=MIN_ROWS names) in the **canonical** stores.

    2026-07-30 修正: data/cache/bars_day_* は**未調整価格**のスナップショット
    （AdjC列に生値が入っている）。これを「収集済み」と見なすと、その期間だけ調整基準が
    ずれた区間がパネルに残り、区間境界で×4〜×200の偽リターンを生む（実害を確認）。
    よってカバレッジ判定は調整済みの正本のみで行い、bars_day_* が埋めている日も
    正本として取り直す（load_existing_daily 側でも正本を優先するよう修正済み）。
    """
    sources = (
        ["data/jp_intraday_reference/daily_20260528_20260724.parquet"]
        + glob.glob("data/jp_daily_history/daily_adj_*.parquet")
    )
    covered = set()
    for p in sources:
        try:
            d = pd.read_parquet(p, columns=["Date"])
        except Exception:  # noqa: BLE001
            continue
        counts = pd.to_datetime(d["Date"]).dt.normalize().value_counts()
        covered |= set(counts[counts >= MIN_ROWS].index)
    return covered


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    client = jquantsapi.ClientV2(api_key=JQUANTS_API_KEY)
    cal = _retry(lambda: client.get_mkt_calendar(from_yyyymmdd=START.replace("-", ""),
                                                 to_yyyymmdd=END.replace("-", "")))
    full = [pd.Timestamp(d) for d in cal[cal["HolDiv"].astype(str).isin(["1", "2"])]["Date"]]
    covered = _covered_dates()
    missing = [d for d in full if d.normalize() not in covered]
    print(f"trading days={len(full)} already_on_disk={len(covered & {d.normalize() for d in full})} "
          f"to_fetch={len(missing)}", flush=True)
    if not missing:
        print("nothing to collect — all dates already on disk.", flush=True)
        return

    by_year: dict[str, list[pd.Timestamp]] = {}
    for d in missing:
        by_year.setdefault(str(d.year), []).append(d)

    for year, ydays in sorted(by_year.items()):
        ypath = OUT / f"daily_adj_{year}.parquet"
        existing = pd.read_parquet(ypath) if ypath.exists() else None
        have = set(pd.to_datetime(existing["Date"]).dt.normalize()) if existing is not None else set()
        todo = [d for d in ydays if d.normalize() not in have]
        if not todo:
            print(f"year {year}: already complete, skip", flush=True)
            continue
        frames = [existing] if existing is not None else []
        fetched, miss = 0, []

        def _flush():
            if frames:
                out = pd.concat(frames, ignore_index=True).drop_duplicates(["Date", "Code"])
                out.to_parquet(ypath, index=False)

        for i, d in enumerate(todo, 1):
            ymd = d.strftime("%Y%m%d")
            try:
                r = _retry(lambda ymd=ymd: client.get_eq_bars_daily(date_yyyymmdd=ymd))
                if len(r) >= MIN_ROWS:
                    frames.append(r[KEEP])
                    fetched += 1
            except Exception as exc:  # noqa: BLE001
                miss.append(ymd)
                print(f"  {year} MISS {ymd}: {str(exc)[:50]}", flush=True)
            time.sleep(0.12)
            if i % 50 == 0:
                _flush()  # incremental save so a kill never loses more than a batch
                print(f"  {year} {i}/{len(todo)} fetched={fetched} (flushed)", flush=True)
        _flush()
        print(f"year {year}: +{fetched} days (missing={len(miss)}) -> {ypath.name}", flush=True)

    inv_path = OUT / "investor_types_2021_2026.parquet"
    if not inv_path.exists():
        inv = _retry(lambda: client.get_eq_investor_types(from_yyyymmdd=START.replace("-", ""),
                                                          to_yyyymmdd=END.replace("-", "")))
        inv.to_parquet(inv_path, index=False)
        print(f"wrote {inv_path.name} rows={len(inv)}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
