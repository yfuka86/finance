"""Collect US sector-ETF, US-index and Nikkei daily history from Stooq.

RUN THIS ON YOUR OWN MACHINE — Stooq rate-limits/serves a JS "verify your
browser" page from cloud/CI IPs, so it returns nothing from the sandbox. From a
normal residential IP it works. Output feeds trading.jp_intraday.us_context and
the overnight-gap decomposition (US vs idiosyncratic).
"""
from pathlib import Path

import pandas as pd

from data.collectors.stooq import download_stooq

START, END = "2021-09-01", "2026-07-24"
# 11 US sector SPDRs + broad US indices + Nikkei 225 (先物のオーバーナイト代理).
US_SECTORS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "XLC", "XLRE"]
INDICES = ["^spx", "^ndq", "^nkx"]
OUT = Path("data/us_context")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    closes = {}
    for ticker in US_SECTORS + INDICES:
        df = download_stooq(ticker, START, END)
        if df.empty:
            print(f"  MISSING {ticker} (blocked or no data)")
            continue
        closes[ticker] = df["Close"]
        print(f"  {ticker}: {len(df)} rows {df.index.min().date()}..{df.index.max().date()}")
    if not closes:
        raise SystemExit("Stooq returned nothing — run this from a residential IP.")
    prices = pd.DataFrame(closes).sort_index()
    returns = prices.pct_change().dropna(how="all")
    prices.to_parquet(OUT / "us_prices_2021_2026.parquet")
    returns.to_parquet(OUT / "us_returns_2021_2026.parquet")
    print(f"wrote {OUT}/us_returns_2021_2026.parquet  shape={returns.shape}")


if __name__ == "__main__":
    main()
