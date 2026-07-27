from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import metrics, simulate
from .data import am_pm_session
from .features import session_vwap_deviation
from .strategy import rank_long_short_weights


STRATEGIES = (
    "reversal_30m", "sector_reversal_30m", "vwap_reversal",
    "volume_reversal_30m", "momentum_30m",
)


def prepare_signals(bars: pd.DataFrame, sectors: pd.DataFrame, interval_minutes: int = 5):
    x = bars.sort_values(["symbol", "timestamp"]).reset_index(drop=True).copy()
    x = x.merge(sectors[["symbol", "sector"]].drop_duplicates("symbol"), on="symbol", how="left")
    x["sector"] = x["sector"].fillna("unknown")
    date = x["timestamp"].dt.date
    x["session"] = am_pm_session(x["timestamp"])
    keys = [x["symbol"], date, x["session"]]
    session_open = x["open"].groupby(keys).transform("first")
    x["session_return"] = x["close"].div(session_open).sub(1)
    x["vwap_deviation"] = session_vwap_deviation(x, keys)
    median_volume = x["volume"].groupby(keys).transform(
        lambda s: s.expanding(min_periods=3).median()
    )
    x["volume_shock"] = np.log1p(x["volume"]).sub(np.log1p(median_volume))
    x["market_return"] = x.groupby("timestamp")["session_return"].transform("mean")
    x["residual_return"] = x["session_return"] - x["market_return"]
    x["sector_residual"] = x["residual_return"] - x.groupby(
        ["timestamp", "sector"]
    )["residual_return"].transform("mean")
    x["session_bar"] = x.groupby([date, "session"])["timestamp"].rank(method="dense").sub(1)
    signal_bar = 30 // interval_minutes
    x["is_signal"] = x["session_bar"].eq(signal_bar)
    return x


def strategy_weights(frame: pd.DataFrame, strategy: str, quantile: float = 0.1) -> pd.Series:
    if strategy == "reversal_30m":
        score = -frame["residual_return"]
    elif strategy == "sector_reversal_30m":
        score = -frame["sector_residual"]
    elif strategy == "vwap_reversal":
        score = -frame["vwap_deviation"]
    elif strategy == "volume_reversal_30m":
        volume_rank = frame["volume_shock"].groupby(frame["timestamp"]).rank(pct=True)
        score = -frame["residual_return"] * volume_rank
    elif strategy == "momentum_30m":
        score = frame["residual_return"]
    else:
        raise ValueError(f"unknown strategy: {strategy}")
    desired = rank_long_short_weights(frame, score, quantile).where(frame["is_signal"])
    date = frame["timestamp"].dt.date
    return desired.groupby([frame["symbol"], date, frame["session"]]).ffill().fillna(0.0)


def evaluate_strategy(
    frame: pd.DataFrame, strategy: str, start: str, end: str,
    interval_minutes: int = 5, quantile: float = 0.1,
    slippage_bps: float = 2.0,
):
    selected = frame[frame["timestamp"].dt.tz_localize(None).between(start, end + " 23:59:59")].copy()
    weights = strategy_weights(selected, strategy, quantile)
    returns = simulate(selected, weights, interval_minutes, slippage_bps=slippage_bps)
    summary = metrics(returns["net_return"], interval_minutes)
    summary["turnover"] = float(returns["turnover"].sum())
    # True gross = returns with every trading cost (slippage, commission, borrow) off,
    # so a short-heavy strategy's borrow charge is not silently left in "gross".
    gross = simulate(selected, weights, interval_minutes,
                     commission_bps=0.0, slippage_bps=0.0, borrow_rate_annual=0.0)
    summary["gross_return_sum"] = float(gross["net_return"].sum())
    return returns, summary
