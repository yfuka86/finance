"""2年分の1分足(月次filtered)を5分足に変換して1本のparquetへ（冪等）。

セッション安全: 東証セッション外の行を落としてから5分floorで集約するため、
昼休み・日跨ぎのバーは構造上できない（resample_barsと同じ保証をベクトル化で実現）。
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("data/jp_minutes_2y/jp_5m_2024-08-01_2026-07-24.parquet")
AM_OPEN, AM_CLOSE, PM_OPEN, PM_CLOSE = 9 * 60, 11 * 60 + 29, 12 * 60 + 30, 15 * 60 + 29


def resample_file(path: str) -> pd.DataFrame:
    raw = pd.read_parquet(path)
    raw["timestamp"] = pd.to_datetime(
        raw["Date"].astype(str) + " " + raw["Time"].astype(str))
    minute = raw["timestamp"].dt.hour * 60 + raw["timestamp"].dt.minute
    in_session = minute.between(AM_OPEN, AM_CLOSE) | minute.between(PM_OPEN, PM_CLOSE)
    raw = raw[in_session].copy()
    raw["symbol"] = raw["Code"].astype(str).str.removesuffix("0")
    raw["bin"] = raw["timestamp"].dt.floor("5min")
    raw = raw.sort_values(["symbol", "timestamp"])
    for c in ("O", "H", "L", "C", "Vo"):
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    out = raw.groupby(["symbol", "bin"], sort=False).agg(
        open=("O", "first"), high=("H", "max"), low=("L", "min"),
        close=("C", "last"), volume=("Vo", "sum")).reset_index()
    return out.rename(columns={"bin": "timestamp"})


def main() -> None:
    if OUT.exists():
        print("already resampled, skip")
        return
    files = sorted(glob.glob("data/jp_minutes_2y/bulk_filtered/*.parquet"))
    pieces = []
    for i, f in enumerate(files, 1):
        pieces.append(resample_file(f))
        print(f"resample {i}/{len(files)} rows_so_far={sum(map(len, pieces)):,}", flush=True)
    allrows = pd.concat(pieces, ignore_index=True)
    allrows["timestamp"] = allrows["timestamp"].dt.tz_localize("Asia/Tokyo")
    allrows = allrows.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    allrows.to_parquet(OUT, index=False)
    print(f"DONE rows={len(allrows):,} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
