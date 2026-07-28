from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StrategyParams:
    lookback: int
    direction: int = 1  # 1 momentum, -1 mean reversion
    entry_quantile: float = 0.7


def scores(bars: pd.DataFrame, params: StrategyParams) -> pd.Series:
    """Price score known at each bar close; rolling windows never cross symbols."""
    ordered = bars.sort_values(["symbol", "timestamp"])
    by_symbol = ordered.groupby("symbol", sort=False)["close"]
    ret = by_symbol.pct_change(params.lookback, fill_method=None)
    one_bar = by_symbol.pct_change(fill_method=None)
    vol = one_bar.groupby(ordered["symbol"], sort=False).transform(
        lambda x: x.rolling(params.lookback, min_periods=params.lookback).std()
    )
    result = params.direction * ret / (vol * np.sqrt(params.lookback)).replace(0, np.nan)
    return result.reindex(bars.index)


def fit_threshold(train_scores: pd.Series, quantile: float) -> float:
    clean = train_scores.replace([np.inf, -np.inf], np.nan).dropna().abs()
    if clean.empty:
        raise ValueError("training data is too short to fit a threshold")
    return float(clean.quantile(quantile))


def dollar_neutral_weights(
    long_mask: pd.Series, short_mask: pd.Series, timestamp: pd.Series,
) -> pd.Series:
    """Equal-weight each side to +/-0.5, per timestamp, only when both sides exist."""
    long_n = long_mask.groupby(timestamp).transform("sum")
    short_n = short_mask.groupby(timestamp).transform("sum")
    both = long_n.gt(0) & short_n.gt(0)
    weights = pd.Series(0.0, index=long_mask.index)
    weights.loc[both & long_mask] = 0.5 / long_n.loc[both & long_mask]
    weights.loc[both & short_mask] = -0.5 / short_n.loc[both & short_mask]
    return weights


def rank_long_short_weights(
    frame: pd.DataFrame, score: pd.Series, quantile: float,
) -> pd.Series:
    """Cross-sectional quantile long/short weights, dollar-neutral per timestamp."""
    pct = score.groupby(frame["timestamp"]).rank(pct=True)
    long = pct.ge(1 - quantile)
    short = pct.le(quantile)
    return dollar_neutral_weights(long, short, frame["timestamp"])


def market_neutral_weights(
    bars: pd.DataFrame, score: pd.Series, threshold: float,
) -> pd.Series:
    raw = pd.Series(np.where(score > threshold, 1.0,
                    np.where(score < -threshold, -1.0, 0.0)), index=bars.index)
    return dollar_neutral_weights(raw.gt(0), raw.lt(0), bars["timestamp"])
