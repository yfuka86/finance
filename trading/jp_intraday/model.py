from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import metrics, simulate
from .features import FEATURES, add_forward_target, make_features
from .strategy import rank_long_short_weights


@dataclass(frozen=True)
class ModelConfig:
    train_days: int = 60
    test_days: int = 10
    step_days: int = 10
    interval_minutes: int = 1
    long_short_quantile: float = 0.15
    rebalance_bars: int = 1
    rebalance_offset: int = 0
    alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
    commission_bps: float = 0.0
    slippage_bps: float = 2.0
    borrow_rate_annual: float = 0.02


def _standardize_fit(frame: pd.DataFrame):
    mean = frame.mean()
    scale = frame.std().replace(0, 1).fillna(1)
    return mean, scale


def _ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    penalty = np.eye(x.shape[1]) * alpha
    return np.linalg.solve(x.T @ x + penalty, x.T @ y)


def _hold_between_rebalances(
    frame: pd.DataFrame, desired: pd.Series, bars: int, offset: int = 0,
) -> pd.Series:
    if bars <= 1:
        return desired
    date = frame["timestamp"].dt.date
    timestamp_number = frame["timestamp"].groupby(date).rank(method="dense").sub(1)
    rebalance = timestamp_number.ge(offset) & timestamp_number.sub(offset).mod(bars).eq(0)
    held = desired.where(rebalance)
    held = held.groupby([frame["symbol"], date]).ffill().fillna(0.0)
    return held


def _fit(frame: pd.DataFrame, alpha: float):
    clean = frame.dropna(subset=[*FEATURES, "target"])
    mean, scale = _standardize_fit(clean[list(FEATURES)])
    x = clean[list(FEATURES)].sub(mean).div(scale).to_numpy()
    beta = _ridge_fit(x, clean["target"].to_numpy(), alpha)
    return mean, scale, beta


def _predict(frame: pd.DataFrame, fitted) -> pd.Series:
    mean, scale, beta = fitted
    x = frame[list(FEATURES)].sub(mean).div(scale)
    result = pd.Series(np.nan, index=frame.index)
    valid = x.notna().all(axis=1)
    result.loc[valid] = x.loc[valid].to_numpy() @ beta
    return result


def run_model_walk_forward(bars: pd.DataFrame, cfg: ModelConfig):
    """Nested, chronological model selection followed by untouched OOS folds."""
    if cfg.step_days < cfg.test_days:
        raise ValueError("step_days must be >= test_days")
    featured = add_forward_target(make_features(bars), cfg.interval_minutes)
    featured = featured.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    dates = pd.Index(sorted(pd.unique(featured["timestamp"].dt.date)))
    cursor, fold_id = cfg.train_days, 0
    all_returns, folds, coefficients = [], [], []
    while cursor + cfg.test_days <= len(dates):
        train_dates = dates[cursor-cfg.train_days:cursor]
        test_dates = dates[cursor:cursor+cfg.test_days]
        split = max(int(len(train_dates) * 0.8), 1)
        fit_dates, validation_dates = train_dates[:split], train_dates[split:]
        if len(validation_dates) == 0:
            raise ValueError("training window is too short for nested validation")
        fit_frame = featured[featured["timestamp"].dt.date.isin(fit_dates)]
        validation = featured[featured["timestamp"].dt.date.isin(validation_dates)]
        candidates = []
        for alpha in cfg.alphas:
            fitted = _fit(fit_frame, alpha)
            pred = _predict(validation, fitted)
            weight = rank_long_short_weights(validation, pred, cfg.long_short_quantile)
            weight = _hold_between_rebalances(
                validation, weight, cfg.rebalance_bars, cfg.rebalance_offset
            )
            result = simulate(validation, weight, cfg.interval_minutes, cfg.commission_bps,
                              cfg.slippage_bps, cfg.borrow_rate_annual)
            candidates.append((metrics(result["net_return"], cfg.interval_minutes)["sharpe"], alpha))
        validation_sharpe, alpha = max(candidates, key=lambda item: item[0])
        train = featured[featured["timestamp"].dt.date.isin(train_dates)]
        fitted = _fit(train, alpha)
        test = featured[featured["timestamp"].dt.date.isin(test_dates)].copy()
        pred = _predict(test, fitted)
        weights = rank_long_short_weights(test, pred, cfg.long_short_quantile)
        weights = _hold_between_rebalances(
            test, weights, cfg.rebalance_bars, cfg.rebalance_offset
        )
        result = simulate(test, weights, cfg.interval_minutes, cfg.commission_bps,
                          cfg.slippage_bps, cfg.borrow_rate_annual)
        result["fold"] = fold_id
        all_returns.append(result)
        mean, scale, beta = fitted
        coefficients.extend({"fold": fold_id, "feature": name, "coefficient": float(value / scale[name])}
                            for name, value in zip(FEATURES, beta))
        folds.append({
            "fold": fold_id, "train_start": str(train_dates[0]), "train_end": str(train_dates[-1]),
            "validation_start": str(validation_dates[0]), "validation_end": str(validation_dates[-1]),
            "test_start": str(test_dates[0]), "test_end": str(test_dates[-1]),
            "alpha": alpha, "validation_net_sharpe": validation_sharpe,
            "train_rows": len(train), "test_rows": len(test),
        })
        fold_id += 1
        cursor += cfg.step_days
    if not all_returns:
        raise ValueError("not enough days for a model walk-forward fold")
    returns = pd.concat(all_returns, ignore_index=True)
    return returns, pd.DataFrame(folds), pd.DataFrame(coefficients), metrics(
        returns["net_return"], cfg.interval_minutes
    )
