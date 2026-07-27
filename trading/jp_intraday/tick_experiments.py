from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import metrics, simulate
from .data import am_pm_session
from .strategy import rank_long_short_weights


TICK_STRATEGIES = (
    "ofi_momentum", "ofi_reversal", "ofi_sector_momentum", "ofi_price_divergence",
)


def prepare_tick_signals(bars: pd.DataFrame, ticks: pd.DataFrame, sectors: pd.DataFrame):
    x = bars.merge(ticks, on=["timestamp", "symbol"], how="left")
    for column in ("signed_volume", "traded_volume", "trade_count"):
        x[column] = x[column].fillna(0.0)
    x = x.merge(sectors[["symbol", "sector"]].drop_duplicates("symbol"), on="symbol", how="left")
    x["sector"] = x["sector"].fillna("unknown")
    date = x["timestamp"].dt.date
    x["session"] = am_pm_session(x["timestamp"])
    keys = [x["symbol"], date, x["session"]]
    x["session_bar"] = x.groupby([date, "session"])["timestamp"].rank(method="dense").sub(1)
    x["cum_signed"] = x["signed_volume"].groupby(keys).cumsum()
    x["cum_volume"] = x["traded_volume"].groupby(keys).cumsum()
    x["cum_ofi"] = x["cum_signed"].div(x["cum_volume"].replace(0, np.nan))
    session_open = x["open"].groupby(keys).transform("first")
    session_return = x["close"].div(session_open).sub(1)
    x["price_residual"] = session_return - session_return.groupby(x["timestamp"]).transform("mean")
    x["ofi_residual"] = x["cum_ofi"] - x["cum_ofi"].groupby(x["timestamp"]).transform("mean")
    x["sector_ofi"] = x["ofi_residual"] - x.groupby(
        ["timestamp", "sector"]
    )["ofi_residual"].transform("mean")
    return x


def tick_strategy_weights(frame: pd.DataFrame, strategy: str, quantile: float = .1):
    if strategy == "ofi_momentum":
        score = frame["ofi_residual"]
    elif strategy == "ofi_reversal":
        score = -frame["ofi_residual"]
    elif strategy == "ofi_sector_momentum":
        score = frame["sector_ofi"]
    elif strategy == "ofi_price_divergence":
        score = frame["ofi_residual"].groupby(frame["timestamp"]).rank(pct=True) - frame[
            "price_residual"
        ].groupby(frame["timestamp"]).rank(pct=True)
    else:
        raise ValueError(strategy)
    signal_bar, holding_bars = 6, 6
    signal = frame["session_bar"].eq(signal_bar)
    desired = rank_long_short_weights(frame, score, quantile).where(signal)
    date = frame["timestamp"].dt.date
    held = desired.groupby([frame["symbol"], date, frame["session"]]).ffill().fillna(0.0)
    return held.where(frame["session_bar"].between(signal_bar, signal_bar + holding_bars - 1), 0.0)


def evaluate_tick_strategy(frame: pd.DataFrame, strategy: str, start: str, end: str,
                           interval_minutes: int = 5, slippage_bps: float = 2.0):
    selected = frame[frame["timestamp"].dt.tz_localize(None).between(start, end + " 23:59:59")].copy()
    weights = tick_strategy_weights(selected, strategy)
    returns = simulate(selected, weights, interval_minutes, slippage_bps=slippage_bps)
    summary = metrics(returns["net_return"], interval_minutes)
    gross = simulate(selected, weights, interval_minutes,
                     commission_bps=0.0, slippage_bps=0.0, borrow_rate_annual=0.0)
    summary.update(strategy=strategy, turnover=float(returns["turnover"].sum()),
                   gross_return_sum=float(gross["net_return"].sum()))
    return returns, summary
