"""
月単位のローソク足データキャッシュ。

saved_data/{symbol}/{interval}/ 配下に YYYY-MM.parquet で保存。
一度取得した月のデータは再取得不要。
"""
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "saved_data"


def _cache_path(symbol: str, interval: str, year: int, month: int) -> Path:
    return CACHE_DIR / symbol / interval / f"{year:04d}-{month:02d}.parquet"


def _ensure_dir(symbol: str, interval: str):
    d = CACHE_DIR / symbol / interval
    d.mkdir(parents=True, exist_ok=True)


def has_month(symbol: str, interval: str, year: int, month: int) -> bool:
    return _cache_path(symbol, interval, year, month).exists()


def load_month(symbol: str, interval: str, year: int, month: int) -> Optional[pd.DataFrame]:
    path = _cache_path(symbol, interval, year, month)
    if path.exists():
        df = pd.read_parquet(path)
        logger.debug(f"Cache hit: {symbol}/{interval}/{year:04d}-{month:02d} ({len(df)} rows)")
        return df
    return None


def save_month(symbol: str, interval: str, year: int, month: int, df: pd.DataFrame):
    _ensure_dir(symbol, interval)
    path = _cache_path(symbol, interval, year, month)
    df.to_parquet(path)
    logger.info(f"Cache saved: {path} ({len(df)} rows)")


def load_months(symbol: str, interval: str,
                start_year: int, start_month: int,
                end_year: int, end_month: int) -> tuple[pd.DataFrame, list[tuple[int, int]]]:
    """
    指定範囲のキャッシュ済みデータを読み込む。

    Returns:
        (cached_df, missing_months) — cached_df は結合済み、missing_months は未取得の (year, month) リスト
    """
    frames = []
    missing = []

    y, m = start_year, start_month
    while (y, m) <= (end_year, end_month):
        df = load_month(symbol, interval, y, m)
        if df is not None:
            frames.append(df)
        else:
            missing.append((y, m))
        # 次の月へ
        m += 1
        if m > 12:
            m = 1
            y += 1

    if frames:
        cached = pd.concat(frames).sort_index()
        cached = cached[~cached.index.duplicated(keep="first")]
    else:
        cached = pd.DataFrame()

    return cached, missing


def list_cached(symbol: str = None) -> list[dict]:
    """キャッシュ済みデータの一覧を返す。"""
    result = []
    if not CACHE_DIR.exists():
        return result
    dirs = [CACHE_DIR / symbol] if symbol else CACHE_DIR.iterdir()
    for sym_dir in dirs:
        if not sym_dir.is_dir():
            continue
        for iv_dir in sym_dir.iterdir():
            if not iv_dir.is_dir():
                continue
            for f in sorted(iv_dir.glob("*.parquet")):
                result.append({
                    "symbol": sym_dir.name,
                    "interval": iv_dir.name,
                    "month": f.stem,
                    "size_kb": round(f.stat().st_size / 1024, 1),
                })
    return result


def delete_cached(symbol: str, interval: str, year: int, month: int) -> bool:
    path = _cache_path(symbol, interval, year, month)
    if path.exists():
        path.unlink()
        return True
    return False


def cache_size_mb() -> float:
    """キャッシュ全体のサイズ(MB)を返す。"""
    if not CACHE_DIR.exists():
        return 0.0
    total = sum(f.stat().st_size for f in CACHE_DIR.rglob("*.parquet"))
    return round(total / (1024 * 1024), 2)
