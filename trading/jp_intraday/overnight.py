"""Overnight-gap + order-flow intraday research.

The overnight gap (prior session close -> today's opening auction) is where the
US session and overnight index-futures moves get priced into each Japanese name.
So the *residual* gap (a name's gap minus the market's gap) is a data-local proxy
for "how did US / futures move this name overnight, beyond the whole market".

Strategies here are cross-sectional, dollar-neutral, and leakage-safe: every
signal is known at the day's open, and positions are executed at the next bar's
open by ``backtest.simulate``. Order-flow imbalance (需給) is merged in as an
optional confirming/independent signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import metrics, simulate
from .data import am_pm_session
from .strategy import rank_long_short_weights


STRATEGIES = (
    "gap_fade",            # fade the idiosyncratic overnight move
    "gap_momentum",        # ride it
    "sector_gap_fade",     # fade the gap net of the sector's gap
    "gap_fade_ofi",        # overnight reversal + order-flow, combined ranks
    "ofi_reversal",        # 需給 only, for comparison
)


def prepare_overnight(
    bars: pd.DataFrame,
    sectors: pd.DataFrame,
    ticks: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach overnight-gap and order-flow features to intraday bars."""
    x = bars.sort_values(["symbol", "timestamp"]).reset_index(drop=True).copy()
    x = x.merge(sectors[["symbol", "sector"]].drop_duplicates("symbol"), on="symbol", how="left")
    x["sector"] = x["sector"].fillna("unknown")
    x["date"] = x["timestamp"].dt.tz_localize(None).dt.normalize()
    x["session"] = am_pm_session(x["timestamp"])
    x["session_bar"] = x.groupby(["date", "session"])["timestamp"].rank(method="dense").sub(1)

    # Per symbol/day open (opening auction) and close, then the gap vs prior day.
    daily = x.groupby(["symbol", "date"], as_index=False).agg(
        first_open=("open", "first"), last_close=("close", "last"),
    ).sort_values(["symbol", "date"])
    daily["prev_close"] = daily.groupby("symbol")["last_close"].shift(1)
    daily["gap"] = daily["first_open"].div(daily["prev_close"]).sub(1)
    x = x.merge(daily[["symbol", "date", "gap"]], on=["symbol", "date"], how="left")

    # Market/sector-neutral residual gap (known at the open, constant intraday).
    market_gap = x.groupby("timestamp")["gap"].transform("mean")
    x["residual_gap"] = x["gap"].sub(market_gap)
    x["sector_gap_residual"] = x["residual_gap"].sub(
        x.groupby(["timestamp", "sector"])["residual_gap"].transform("mean")
    )

    if ticks is not None:
        x = x.merge(ticks, on=["timestamp", "symbol"], how="left")
        for column in ("signed_volume", "traded_volume"):
            x[column] = x.get(column, pd.Series(0.0, index=x.index)).fillna(0.0)
        keys = [x["symbol"], x["date"], x["session"]]
        cum_signed = x["signed_volume"].groupby(keys).cumsum()
        cum_volume = x["traded_volume"].groupby(keys).cumsum()
        x["cum_ofi"] = cum_signed.div(cum_volume.replace(0, np.nan))
        x["ofi_residual"] = x["cum_ofi"].sub(
            x.groupby("timestamp")["cum_ofi"].transform("mean")
        )
    else:
        x["cum_ofi"] = np.nan
        x["ofi_residual"] = np.nan
    return x


def _blended_rank_score(frame: pd.DataFrame, columns: list[str], signs: list[int]) -> pd.Series:
    """Average of per-timestamp percentile ranks of several signed signals."""
    ranks = []
    for column, sign in zip(columns, signs):
        r = (sign * frame[column]).groupby(frame["timestamp"]).rank(pct=True)
        ranks.append(r)
    return pd.concat(ranks, axis=1).mean(axis=1)


def overnight_weights(
    frame: pd.DataFrame, strategy: str, quantile: float = 0.2,
    signal_bar: int = 1, holding_bars: int = 12,
) -> pd.Series:
    """Signal early in the morning session and hold for ``holding_bars`` bars."""
    if strategy == "gap_fade":
        score = -frame["residual_gap"]
    elif strategy == "gap_momentum":
        score = frame["residual_gap"]
    elif strategy == "sector_gap_fade":
        score = -frame["sector_gap_residual"]
    elif strategy == "gap_fade_ofi":
        score = _blended_rank_score(frame, ["residual_gap", "ofi_residual"], [-1, -1])
    elif strategy == "ofi_reversal":
        score = -frame["ofi_residual"]
    else:
        raise ValueError(f"unknown strategy: {strategy}")

    # Only trade the morning session, where the overnight gap is freshest.
    tradable = frame["session"].eq("am") & score.notna()
    signal = frame["session_bar"].eq(signal_bar) & tradable
    desired = rank_long_short_weights(frame, score.where(tradable, np.nan), quantile).where(signal)
    held = desired.groupby([frame["symbol"], frame["date"], frame["session"]]).ffill().fillna(0.0)
    window = frame["session_bar"].between(signal_bar, signal_bar + holding_bars - 1)
    return held.where(window & frame["session"].eq("am"), 0.0)


def evaluate_overnight(
    frame: pd.DataFrame, strategy: str, start: str, end: str,
    interval_minutes: int = 5, quantile: float = 0.2, slippage_bps: float = 2.0,
    signal_bar: int = 1, holding_bars: int = 12,
) -> tuple[pd.DataFrame, dict]:
    selected = frame[frame["timestamp"].dt.tz_localize(None).between(start, end + " 23:59:59")].copy()
    weights = overnight_weights(selected, strategy, quantile, signal_bar, holding_bars)
    returns = simulate(selected, weights, interval_minutes, slippage_bps=slippage_bps)
    summary = metrics(returns["net_return"], interval_minutes)
    gross = simulate(selected, weights, interval_minutes,
                     commission_bps=0.0, slippage_bps=0.0, borrow_rate_annual=0.0)
    summary["gross_sharpe"] = metrics(gross["net_return"], interval_minutes)["sharpe"]
    summary["gross_return_sum"] = float(gross["net_return"].sum())
    summary["turnover"] = float(returns["turnover"].sum())
    summary["strategy"] = strategy
    return returns, summary
