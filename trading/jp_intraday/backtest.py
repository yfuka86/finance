from __future__ import annotations

import numpy as np
import pandas as pd

from .data import NOON_MINUTE, minute_of_day

TRADING_DAYS_PER_YEAR = 245
TRADING_MINUTES_PER_DAY = 300  # two 150-minute sessions


def _bars_per_year(interval_minutes: int) -> float:
    return TRADING_DAYS_PER_YEAR * (TRADING_MINUTES_PER_DAY / interval_minutes)


def simulate(
    bars: pd.DataFrame,
    weights: pd.Series,
    interval_minutes: int,
    commission_bps: float = 0.0,
    slippage_bps: float = 2.0,
    borrow_rate_annual: float = 0.02,
) -> pd.DataFrame:
    """Evaluate close-t signal at next bar's open, with one-way trading costs."""
    work = bars[["timestamp", "symbol", "open"]].copy()
    work["weight"] = weights.reindex(work.index).fillna(0.0)
    work = work.sort_values(["symbol", "timestamp"])
    by_symbol = work.groupby("symbol", sort=False)
    work["entry_open"] = by_symbol["open"].shift(-1)
    work["exit_open"] = by_symbol["open"].shift(-2)
    work["entry_timestamp"] = by_symbol["timestamp"].shift(-1)
    work["exit_timestamp"] = by_symbol["timestamp"].shift(-2)
    # Never hold through lunch, overnight, or a missing bar.
    def session_id(ts: pd.Series) -> pd.Series:
        local = ts.dt.tz_localize(None)
        return local.dt.normalize().astype(str) + "_" + minute_of_day(local).lt(NOON_MINUTE).astype(str)

    current_session = session_id(work["timestamp"])
    entry_session = session_id(work["entry_timestamp"])
    exit_session = session_id(work["exit_timestamp"])
    contiguous = current_session.eq(entry_session) & entry_session.eq(exit_session)
    work["forward_return"] = np.where(
        contiguous, work["exit_open"].div(work["entry_open"]).sub(1.0), 0.0
    )
    # A signal without both a future entry and exit bar cannot create a position.
    work.loc[~contiguous, "weight"] = 0.0
    by_symbol = work.groupby("symbol", sort=False)
    prior = by_symbol["weight"].shift(1).fillna(0.0)
    work["turnover"] = work["weight"].sub(prior).abs()
    bars_per_year = _bars_per_year(interval_minutes)
    cost_rate = (commission_bps + slippage_bps) / 10_000
    work["pnl"] = (
        work["weight"] * work["forward_return"]
        - work["turnover"] * cost_rate
        - work["weight"].clip(upper=0).abs() * borrow_rate_annual / bars_per_year
    )
    return work.groupby("timestamp", as_index=False).agg(
        net_return=("pnl", "sum"), turnover=("turnover", "sum"),
        gross_exposure=("weight", lambda x: x.abs().sum()),
    )


def metrics(returns: pd.Series, interval_minutes: int) -> dict[str, float]:
    returns = returns.fillna(0.0)
    equity = (1.0 + returns).cumprod()
    bars_per_year = _bars_per_year(interval_minutes)
    vol = returns.std(ddof=1)
    sharpe = returns.mean() / vol * np.sqrt(bars_per_year) if vol > 0 else 0.0
    drawdown = equity.div(equity.cummax()).sub(1.0)
    years = max(len(returns) / bars_per_year, 1 / bars_per_year)
    return {
        "total_return": float(equity.iloc[-1] - 1) if len(equity) else 0.0,
        "cagr": float(equity.iloc[-1] ** (1 / years) - 1) if len(equity) else 0.0,
        "sharpe": float(sharpe),
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
    }
