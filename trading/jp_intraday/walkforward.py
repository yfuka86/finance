from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product

import pandas as pd

from .backtest import metrics, simulate
from .strategy import StrategyParams, fit_threshold, market_neutral_weights, scores


@dataclass(frozen=True)
class WalkForwardConfig:
    train_days: int = 40
    test_days: int = 10
    step_days: int = 10
    interval_minutes: int = 1
    commission_bps: float = 0.0
    slippage_bps: float = 2.0
    borrow_rate_annual: float = 0.02
    lookbacks: tuple[int, ...] = (5, 15, 30)
    directions: tuple[int, ...] = (1, -1)
    entry_quantiles: tuple[float, ...] = (0.6, 0.75)


def _evaluate(bars: pd.DataFrame, params: StrategyParams, threshold: float, cfg: WalkForwardConfig):
    score = scores(bars, params)
    weights = market_neutral_weights(bars, score, threshold)
    return simulate(bars, weights, cfg.interval_minutes, cfg.commission_bps,
                    cfg.slippage_bps, cfg.borrow_rate_annual)


def run_walk_forward(bars: pd.DataFrame, cfg: WalkForwardConfig):
    """Tune on strictly earlier dates and evaluate each later test fold once."""
    if cfg.step_days < cfg.test_days:
        raise ValueError("step_days must be >= test_days so test folds cannot overlap")
    bars = bars.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    dates = pd.Index(sorted(pd.unique(bars["timestamp"].dt.date)))
    candidates = [StrategyParams(*x) for x in product(
        cfg.lookbacks, cfg.directions, cfg.entry_quantiles
    )]
    folds, test_returns = [], []
    cursor = cfg.train_days
    fold_id = 0
    while cursor + cfg.test_days <= len(dates):
        train_dates = dates[cursor - cfg.train_days:cursor]
        test_dates = dates[cursor:cursor + cfg.test_days]
        if not train_dates[-1] < test_dates[0]:
            raise AssertionError("training data must end before the test period")
        train = bars[bars["timestamp"].dt.date.isin(train_dates)].copy()
        test = bars[bars["timestamp"].dt.date.isin(test_dates)].copy()

        ranked = []
        for params in candidates:
            train_score = scores(train, params)
            threshold = fit_threshold(train_score, params.entry_quantile)
            result = _evaluate(train, params, threshold, cfg)
            ranked.append((metrics(result["net_return"], cfg.interval_minutes)["sharpe"], params, threshold))
        train_sharpe, best, threshold = max(ranked, key=lambda x: x[0])

        # Pre-test history supplies rolling warm-up only. Threshold/parameters stay frozen.
        history = bars[bars["timestamp"] <= test["timestamp"].max()].copy()
        score = scores(history, best)
        weights = market_neutral_weights(history, score, threshold)
        result = simulate(history, weights, cfg.interval_minutes, cfg.commission_bps,
                          cfg.slippage_bps, cfg.borrow_rate_annual)
        start_ts, end_ts = test["timestamp"].min(), test["timestamp"].max()
        result = result[result["timestamp"].between(start_ts, end_ts)].copy()
        result["fold"] = fold_id
        test_returns.append(result)
        folds.append({
            "fold": fold_id, "train_start": str(train_dates[0]), "train_end": str(train_dates[-1]),
            "test_start": str(test_dates[0]), "test_end": str(test_dates[-1]),
            "train_sharpe": train_sharpe, "threshold": threshold, **asdict(best),
        })
        fold_id += 1
        cursor += cfg.step_days
    if not folds:
        raise ValueError("not enough trading days for one walk-forward fold")
    returns = pd.concat(test_returns, ignore_index=True).drop_duplicates("timestamp", keep="first")
    return returns, pd.DataFrame(folds), metrics(returns["net_return"], cfg.interval_minutes)
