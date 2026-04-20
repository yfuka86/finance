"""
J-Quants API V2 data collector.
Authentication: x-api-key header.

Endpoint reference (2025-12~ V2):
  /v2/equities/bars/daily    OHLCV
  /v2/equities/master        銘柄マスタ (名前・業種・市場)
  /v2/fins/summary           決算サマリー (EPS, BPS, CashEq 等)
  /v2/equities/earnings-calendar  決算発表予定
  /v2/markets/calendar       営業日カレンダー
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


def _paginated_get(path: str, params: dict, key: str = "data") -> pd.DataFrame:
    """Generic paginated GET helper."""
    headers = {"x-api-key": JQUANTS_API_KEY}
    all_records = []
    while True:
        for attempt in range(4):
            resp = requests.get(
                f"{JQUANTS_BASE}{path}",
                params=params, headers=headers, timeout=30,
            )
            if resp.status_code == 429:
                wait = 2 ** attempt + 1
                time.sleep(wait)
                continue
            break
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


def download_jquants(code: str, start: str, end: str) -> pd.DataFrame:
    """Download daily OHLCV from J-Quants API V2 with pagination."""
    jq_code = code.replace(".T", "")
    from_date = start.replace("-", "")
    to_date = end.replace("-", "")

    df = _paginated_get(
        "/v2/equities/bars/daily",
        {"code": jq_code, "from": from_date, "to": to_date},
    )
    if df.empty:
        return df

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index().rename(columns=_RENAME)
    return df


# ── Screening / fundamental endpoints ─────────────────────────

def fetch_equities_master() -> pd.DataFrame:
    """銘柄マスタ (コード / 銘柄名 / 業種 / 市場区分)."""
    return _paginated_get("/v2/equities/master", {})


def fetch_fins_summary(code: str) -> pd.DataFrame:
    """決算サマリー (EPS / BPS / CashEq / 売上 / 利益 etc.)."""
    return _paginated_get(
        "/v2/fins/summary",
        {"code": code.replace(".T", "")},
    )


def fetch_earnings_calendar(from_date: str = None,
                            to_date: str = None) -> pd.DataFrame:
    """決算発表予定カレンダー."""
    params = {}
    if from_date:
        params["from"] = from_date.replace("-", "")
    if to_date:
        params["to"] = to_date.replace("-", "")
    return _paginated_get("/v2/equities/earnings-calendar", params)


def fetch_bars_daily(date: str = None, code: str = None,
                     from_date: str = None, to_date: str = None) -> pd.DataFrame:
    """日足 OHLCV (全銘柄一括 or 個別銘柄)."""
    params = {}
    if date:
        params["date"] = date.replace("-", "")
    if code:
        params["code"] = code.replace(".T", "")
    if from_date:
        params["from"] = from_date.replace("-", "")
    if to_date:
        params["to"] = to_date.replace("-", "")
    return _paginated_get("/v2/equities/bars/daily", params)


# ── Legacy aliases (backward compat) ──────────────────────────

def fetch_listed_info() -> pd.DataFrame:
    """Alias for fetch_equities_master."""
    return fetch_equities_master()


def fetch_statements(code: str) -> pd.DataFrame:
    """Alias for fetch_fins_summary."""
    return fetch_fins_summary(code)


def fetch_daily_quotes(date: str = None, code: str = None,
                       from_date: str = None, to_date: str = None) -> pd.DataFrame:
    """Alias for fetch_bars_daily."""
    return fetch_bars_daily(date=date, code=code,
                            from_date=from_date, to_date=to_date)
