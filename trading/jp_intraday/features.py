from __future__ import annotations

import numpy as np
import pandas as pd

from .data import (
    AM_OPEN_MINUTE, NOON_MINUTE, PM_OPEN_MINUTE, SESSION_MINUTES,
    am_pm_session, minute_of_day,
)


FEATURES = (
    "reversal_1", "momentum_5", "momentum_15", "volatility_15",
    "vwap_deviation", "volume_shock", "range_position", "market_return_1",
    "minute_sin", "minute_cos",
)


def session_vwap_deviation(frame: pd.DataFrame, keys: list) -> pd.Series:
    """Close vs cumulative session VWAP; ``keys`` groups per symbol/day/session."""
    typical = (frame["high"] + frame["low"] + frame["close"]) / 3
    cum_value = (typical * frame["volume"]).groupby(keys).cumsum()
    cum_volume = frame["volume"].groupby(keys).cumsum().replace(0, np.nan)
    return frame["close"].div(cum_value.div(cum_volume)).sub(1)


def make_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Create features available at each timestamp's close only."""
    x = bars.sort_values(["symbol", "timestamp"]).copy()
    session_date = x["timestamp"].dt.date.astype(str)
    minutes = minute_of_day(x["timestamp"])
    session = am_pm_session(x["timestamp"])
    session_key = [x["symbol"], session_date, session]
    group = x.groupby(session_key, sort=False)
    ret1 = group["close"].pct_change(fill_method=None)
    x["reversal_1"] = -ret1
    x["momentum_5"] = group["close"].pct_change(5, fill_method=None)
    x["momentum_15"] = group["close"].pct_change(15, fill_method=None)
    x["volatility_15"] = ret1.groupby(session_key).transform(
        lambda s: s.rolling(15, min_periods=10).std()
    )
    x["vwap_deviation"] = session_vwap_deviation(x, session_key)
    median_volume = x["volume"].groupby(session_key).transform(
        lambda s: s.rolling(30, min_periods=10).median()
    )
    x["volume_shock"] = np.log1p(x["volume"]).sub(np.log1p(median_volume))
    spread = x["high"].sub(x["low"]).replace(0, np.nan)
    x["range_position"] = x["close"].sub(x["low"]).div(spread).sub(0.5)
    x["market_return_1"] = ret1.groupby(x["timestamp"]).transform("mean")
    trading_minute = np.where(
        minutes < NOON_MINUTE, minutes - AM_OPEN_MINUTE, minutes - PM_OPEN_MINUTE + SESSION_MINUTES
    )
    phase = 2 * np.pi * trading_minute / (2 * SESSION_MINUTES)
    x["minute_sin"], x["minute_cos"] = np.sin(phase), np.cos(phase)
    return x


def add_forward_target(feature_frame: pd.DataFrame, interval_minutes: int) -> pd.DataFrame:
    """Target features at t with the tradable open(t+1)->open(t+2) return."""
    x = feature_frame.sort_values(["symbol", "timestamp"]).copy()
    group = x.groupby("symbol", sort=False)
    entry = group["open"].shift(-1)
    exit_ = group["open"].shift(-2)
    t1 = group["timestamp"].shift(-1)
    t2 = group["timestamp"].shift(-2)
    expected = pd.Timedelta(minutes=interval_minutes)
    valid = t1.sub(x["timestamp"]).eq(expected) & t2.sub(t1).eq(expected)
    x["target"] = np.where(valid, exit_.div(entry).sub(1), np.nan)
    x["target"] = x["target"].sub(x.groupby("timestamp")["target"].transform("mean"))
    return x
