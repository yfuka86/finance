"""
J-Quants API V2 data collector.
Authentication: x-api-key header.
Free plan: data available from 2022 onwards.
"""
import time
import requests
import pandas as pd
from data.collectors.config import JQUANTS_API_KEY, JQUANTS_BASE

_RENAME = {
    "O": "Open", "H": "High", "L": "Low", "C": "Close",
    "Vo": "Volume", "Va": "Value",
    "AdjO": "AdjOpen", "AdjH": "AdjHigh",
    "AdjL": "AdjLow", "AdjC": "AdjClose", "AdjVo": "AdjVolume",
}


def download_jquants(code: str, start: str, end: str) -> pd.DataFrame:
    """Download daily OHLCV from J-Quants API V2 with pagination."""
    headers = {"x-api-key": JQUANTS_API_KEY}
    jq_code = code.replace(".T", "")
    from_date = start.replace("-", "")
    to_date = end.replace("-", "")

    all_records = []
    params = {"code": jq_code, "from": from_date, "to": to_date}

    while True:
        resp = requests.get(
            f"{JQUANTS_BASE}/v2/equities/bars/daily",
            params=params, headers=headers, timeout=30,
        )
        if resp.status_code != 200:
            print(f"  J-Quants error for {code}: {resp.status_code}")
            break
        body = resp.json()
        all_records.extend(body.get("data", []))
        pagination_key = body.get("pagination_key")
        if not pagination_key:
            break
        params["pagination_key"] = pagination_key
        time.sleep(0.5)

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index().rename(columns=_RENAME)
    return df


# ── Fundamental / screening endpoints ─────────────────────────

def _paginated_get(path: str, params: dict, key: str) -> pd.DataFrame:
    """Generic paginated GET helper."""
    headers = {"x-api-key": JQUANTS_API_KEY}
    all_records = []
    while True:
        resp = requests.get(
            f"{JQUANTS_BASE}{path}",
            params=params, headers=headers, timeout=30,
        )
        if resp.status_code != 200:
            print(f"  J-Quants {path} error: {resp.status_code}")
            break
        body = resp.json()
        all_records.extend(body.get(key, []))
        pagination_key = body.get("pagination_key")
        if not pagination_key:
            break
        params["pagination_key"] = pagination_key
        time.sleep(0.3)
    return pd.DataFrame(all_records) if all_records else pd.DataFrame()


def fetch_listed_info() -> pd.DataFrame:
    """Fetch all listed stock info."""
    return _paginated_get("/v2/listed/info", {}, "info")


def fetch_daily_quotes(date: str = None, code: str = None,
                       from_date: str = None, to_date: str = None) -> pd.DataFrame:
    """
    Fetch daily quotes (OHLCV + MarketCap etc.).
      date      – single date YYYY-MM-DD → all stocks for that date
      code      – stock code (e.g. "7203" or "7203.T")
      from_date / to_date – date range (with code)
    """
    params = {}
    if date:
        params["date"] = date.replace("-", "")
    if code:
        params["code"] = code.replace(".T", "")
    if from_date:
        params["from"] = from_date.replace("-", "")
    if to_date:
        params["to"] = to_date.replace("-", "")
    df = _paginated_get("/v2/prices/daily_quotes", params, "daily_quotes")
    if not df.empty and "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    return df


def fetch_statements(code: str) -> pd.DataFrame:
    """Fetch financial statements for a stock."""
    params = {"code": code.replace(".T", "")}
    return _paginated_get("/v2/fins/statements", params, "statements")


def fetch_earnings_calendar(from_date: str = None,
                            to_date: str = None) -> pd.DataFrame:
    """Fetch earnings announcement calendar."""
    params = {}
    if from_date:
        params["from"] = from_date.replace("-", "")
    if to_date:
        params["to"] = to_date.replace("-", "")
    return _paginated_get("/v2/fins/announcement", params, "announcement")
