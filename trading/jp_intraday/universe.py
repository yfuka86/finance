from __future__ import annotations

import pandas as pd


def build_point_in_time_universe(
    daily_bars: pd.DataFrame,
    memberships: pd.DataFrame,
    share_snapshots: pd.DataFrame,
    min_market_cap_yen: float = 100_000_000_000,
) -> pd.DataFrame:
    """Build a survivorship-safe TOPIX/market-cap universe.

    memberships: symbol,effective_from,effective_to (inclusive; blank end is open).
    share_snapshots: symbol,known_at,shares. ``known_at`` must be the date the value
    became public, not the fiscal period end.
    Market cap for date D uses D-1 close and the latest shares known before D.
    """
    prices = daily_bars[["date", "symbol", "close"]].copy()
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["symbol", "date"])
    prices["prior_close"] = prices.groupby("symbol")["close"].shift(1)

    shares = share_snapshots[["symbol", "known_at", "shares"]].copy()
    shares["known_at"] = pd.to_datetime(shares["known_at"])
    shares = shares.sort_values(["known_at", "symbol"])
    chunks = []
    for symbol, group in prices.groupby("symbol", sort=False):
        known = shares[shares["symbol"].eq(symbol)].sort_values("known_at")
        if known.empty:
            continue
        chunks.append(pd.merge_asof(
            group.sort_values("date"), known[["known_at", "shares"]],
            left_on="date", right_on="known_at", direction="backward",
            allow_exact_matches=False,
        ))
    if not chunks:
        raise ValueError("no symbols have point-in-time share snapshots")
    base = pd.concat(chunks, ignore_index=True)
    base["market_cap"] = base["prior_close"] * base["shares"]

    members = memberships[["symbol", "effective_from", "effective_to"]].copy()
    members["effective_from"] = pd.to_datetime(members["effective_from"])
    members["effective_to"] = pd.to_datetime(members["effective_to"]).fillna(pd.Timestamp.max.normalize())
    joined = base.merge(members, on="symbol", how="inner")
    eligible = (
        joined["date"].between(joined["effective_from"], joined["effective_to"])
        & joined["market_cap"].ge(min_market_cap_yen)
    )
    return joined.loc[eligible, ["date", "symbol", "market_cap"]].drop_duplicates(
        ["date", "symbol"]
    ).sort_values(["date", "symbol"]).reset_index(drop=True)


def filter_intraday(bars: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    work = bars.copy()
    work["date"] = work["timestamp"].dt.tz_localize(None).dt.normalize()
    allowed = universe[["date", "symbol"]].copy()
    allowed["date"] = pd.to_datetime(allowed["date"])
    return work.merge(allowed, on=["date", "symbol"], how="inner").drop(columns="date")
