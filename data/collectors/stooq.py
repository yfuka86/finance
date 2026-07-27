"""
Stooq data collector.
Free, no API key required. Provides daily OHLCV for US & JP ETFs.
"""
import io
import time
import requests
import pandas as pd


# Stooq blocks requests without a browser User-Agent (returns a 404/HTML page),
# and rate-limits CSV downloads per IP (serves a JS "verify your browser" page).
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}


def _to_stooq_ticker(ticker: str) -> str:
    if ticker.endswith(".T"):
        return ticker.replace(".T", ".jp")
    if ticker.startswith("^"):  # indices, e.g. ^spx, ^ndq, ^nkx
        return ticker.lower()
    return f"{ticker.lower()}.us"


def download_stooq(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download daily OHLCV from Stooq. Returns empty on block/no-data."""
    stooq = _to_stooq_ticker(ticker)
    d1 = start.replace("-", "")
    d2 = end.replace("-", "")
    url = f"https://stooq.com/q/d/l/?s={stooq}&d1={d1}&d2={d2}&i=d"
    resp = requests.get(url, timeout=30, headers=_HEADERS)
    text = resp.text
    # A real CSV starts with the header row; an HTML body means block/limit/no-data.
    if resp.status_code != 200 or not text.lstrip().startswith("Date,"):
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(text), parse_dates=["Date"], index_col="Date")
    df = df.sort_index()
    time.sleep(0.5)
    return df
