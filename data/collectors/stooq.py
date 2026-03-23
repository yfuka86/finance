"""
Stooq data collector.
Free, no API key required. Provides daily OHLCV for US & JP ETFs.
"""
import io
import time
import requests
import pandas as pd


def _to_stooq_ticker(ticker: str) -> str:
    if ticker.endswith(".T"):
        return ticker.replace(".T", ".jp")
    return f"{ticker.lower()}.us"


def download_stooq(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download daily OHLCV from Stooq."""
    stooq = _to_stooq_ticker(ticker)
    d1 = start.replace("-", "")
    d2 = end.replace("-", "")
    url = f"https://stooq.com/q/d/l/?s={stooq}&d1={d1}&d2={d2}&i=d"
    resp = requests.get(url, timeout=30)
    if resp.status_code != 200 or "No data" in resp.text:
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(resp.text), parse_dates=["Date"], index_col="Date")
    df = df.sort_index()
    time.sleep(0.5)
    return df
