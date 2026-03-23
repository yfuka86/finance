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
