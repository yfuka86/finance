"""US-session / index overnight context aligned to JP trading days.

A US cash session dated X closes ~06:00 JST on the next JP trading day, before
Tokyo opens at 09:00. So the overnight signal for JP day D is the US return dated
on the prior US session. Aligning by "the last available US return strictly
before the JP open" both (a) keeps it point-in-time safe and (b) captures where
the overnight index-futures move originated.
"""
from __future__ import annotations

import pandas as pd


def align_overnight(us_returns: pd.DataFrame, jp_days: pd.DatetimeIndex) -> pd.DataFrame:
    """For each JP trading day, take the most recent US return known before it.

    us_returns: index = US calendar date, columns = instruments (daily returns).
    jp_days:    JP trading days (tz-naive dates).
    Returns a frame indexed by JP day with the same columns (US move usable at open).
    """
    us = us_returns.sort_index()
    us.index = pd.to_datetime(us.index)
    jp = pd.DatetimeIndex(sorted(pd.to_datetime(jp_days).normalize().unique()))
    # merge_asof backward with allow_exact_matches=False: strictly-earlier US date.
    left = pd.DataFrame({"jp_day": jp})
    right = us.reset_index().rename(columns={us.index.name or "index": "us_day"})
    right = right.rename(columns={right.columns[0]: "us_day"})
    merged = pd.merge_asof(left, right, left_on="jp_day", right_on="us_day",
                           direction="backward", allow_exact_matches=False)
    return merged.set_index("jp_day").drop(columns="us_day")


def overnight_market_factor(us_returns: pd.DataFrame, jp_days: pd.DatetimeIndex,
                            columns: list[str] | None = None) -> pd.Series:
    """Broad US overnight move (mean across the given instruments) per JP day."""
    aligned = align_overnight(us_returns, jp_days)
    cols = columns or list(aligned.columns)
    return aligned[cols].mean(axis=1).rename("us_overnight")
