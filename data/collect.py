"""
Data collection entry point.
Downloads US (Stooq) and JP (J-Quants + Stooq backfill) sector ETF data,
then builds return matrices for backtesting.
"""
import os
import time
import pandas as pd
from data.collectors.config import US_TICKERS, JP_TICKERS, START_DATE, END_DATE, RAW_DATA_DIR
from data.collectors.stooq import download_stooq
from data.collectors.jquants import download_jquants


def _download_jp_etf(ticker: str, start: str, end: str) -> pd.DataFrame:
    """JP ETF: J-Quants (2022~) + Stooq backfill (~2021)."""
    frames = []
    jq_start = max(start, "2022-01-01")
    if jq_start <= end:
        print(f"  {ticker}: J-Quants {jq_start} ~ {end}")
        jq_df = download_jquants(ticker, jq_start, end)
        if not jq_df.empty:
            frames.append(jq_df)
        time.sleep(0.3)
    stooq_end = min(end, "2021-12-31")
    if start <= stooq_end:
        print(f"  {ticker}: Stooq {start} ~ {stooq_end}")
        stooq_df = download_stooq(ticker, start, stooq_end)
        if not stooq_df.empty:
            frames.append(stooq_df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames)
    return combined[~combined.index.duplicated(keep="first")].sort_index()


def build_return_matrices(us_ohlc: dict, jp_ohlc: dict):
    """
    US: Close-to-Close return
    JP: Open-to-Close return (full day) + Open-to-AM-Close approximation
    """
    # US cc returns
    us_close = pd.DataFrame({t: df["Close"] for t, df in us_ohlc.items() if not df.empty and "Close" in df.columns})
    us_cc = us_close.pct_change(fill_method=None).dropna(how="all")

    # JP oc returns (full day)
    jp_oc = pd.DataFrame({
        t: df["Close"] / df["Open"] - 1
        for t, df in jp_ohlc.items()
        if not df.empty and "Open" in df.columns
    }).dropna(how="all")

    # JP AM approximation (50% of daily move): Open → ~11:30
    AM_RATIO = 0.5
    jp_am = pd.DataFrame({
        t: (df["Open"] + AM_RATIO * (df["Close"] - df["Open"])) / df["Open"] - 1
        for t, df in jp_ohlc.items()
        if not df.empty and "Open" in df.columns
    }).dropna(how="all")

    # JP PM return approximation: ~12:30 → Close
    # PM_Open ≈ Open + AM_RATIO * (Close - Open)
    jp_pm = pd.DataFrame({
        t: df["Close"] / (df["Open"] + AM_RATIO * (df["Close"] - df["Open"])) - 1
        for t, df in jp_ohlc.items()
        if not df.empty and "Open" in df.columns
    }).dropna(how="all")

    return us_cc, jp_oc, jp_am, jp_pm


def collect(force: bool = False):
    """Main collection pipeline. Returns (us_cc, jp_oc, jp_am, jp_pm) DataFrames."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    us_path = os.path.join(RAW_DATA_DIR, "us_cc_returns.csv")
    jp_path = os.path.join(RAW_DATA_DIR, "jp_oc_returns.csv")
    jp_am_path = os.path.join(RAW_DATA_DIR, "jp_am_returns.csv")
    jp_pm_path = os.path.join(RAW_DATA_DIR, "jp_pm_returns.csv")

    if not force and all(os.path.exists(p) for p in [us_path, jp_path, jp_am_path, jp_pm_path]):
        print("Loading cached data...")
        return (
            pd.read_csv(us_path, index_col=0, parse_dates=True),
            pd.read_csv(jp_path, index_col=0, parse_dates=True),
            pd.read_csv(jp_am_path, index_col=0, parse_dates=True),
            pd.read_csv(jp_pm_path, index_col=0, parse_dates=True),
        )

    print("Downloading US sector ETFs (Stooq)...")
    us_ohlc = {}
    for t in US_TICKERS:
        print(f"  {t}: Stooq {START_DATE} ~ {END_DATE}")
        us_ohlc[t] = download_stooq(t, START_DATE, END_DATE)

    print("\nDownloading JP sector ETFs (J-Quants + Stooq)...")
    jp_ohlc = {}
    for t in JP_TICKERS:
        jp_ohlc[t] = _download_jp_etf(t, START_DATE, END_DATE)

    print("\nBuilding return matrices...")
    us_cc, jp_oc, jp_am, jp_pm = build_return_matrices(us_ohlc, jp_ohlc)

    us_cc.to_csv(us_path)
    jp_oc.to_csv(jp_path)
    jp_am.to_csv(jp_am_path)
    jp_pm.to_csv(jp_pm_path)

    print(f"US: {us_cc.shape}, JP full: {jp_oc.shape}, JP AM: {jp_am.shape}, JP PM: {jp_pm.shape}")
    print(f"Saved to {RAW_DATA_DIR}/")
    return us_cc, jp_oc, jp_am, jp_pm


if __name__ == "__main__":
    collect()
