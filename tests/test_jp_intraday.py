import unittest

import numpy as np
import pandas as pd

from trading.jp_intraday.data import resample_bars
from trading.jp_intraday.backtest import simulate
from trading.jp_intraday.model import ModelConfig, run_model_walk_forward
from trading.jp_intraday.strategy import StrategyParams, scores
from trading.jp_intraday.universe import build_point_in_time_universe
from trading.jp_intraday.walkforward import WalkForwardConfig, run_walk_forward


def sample(days=12, symbols=4):
    rng = np.random.default_rng(7)
    rows = []
    for day in pd.bdate_range("2025-01-06", periods=days, tz="Asia/Tokyo"):
        for minute in range(20):
            ts = day + pd.Timedelta(hours=9, minutes=minute)
            for symbol in range(symbols):
                price = 1000 + symbol * 50 + day.day * (symbol - 1.5) + minute * (symbol - 1.5)
                price += rng.normal(0, .05)
                rows.append((ts, str(1000 + symbol), price, price+.2, price-.2, price+.05, 1000))
    return pd.DataFrame(rows, columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"])


class IntradayTest(unittest.TestCase):
    def test_signal_executes_after_signal_close(self):
        bars = sample(1, 2)
        weights = pd.Series(0.0, index=bars.index)
        symbol_rows = bars.index[bars.symbol.eq("1000")]
        weights.loc[symbol_rows[0]] = 1.0
        result = simulate(bars, weights, 1, slippage_bps=0, borrow_rate_annual=0)
        expected = (bars.loc[symbol_rows[2], "open"] / bars.loc[symbol_rows[1], "open"]) - 1
        self.assertAlmostEqual(result.iloc[0].net_return, expected)

    def test_untradeable_end_of_session_signal_has_no_cost(self):
        bars = sample(1, 2)
        weights = pd.Series(0.0, index=bars.index)
        weights.iloc[-1] = 1.0
        result = simulate(bars, weights, 1, slippage_bps=2, borrow_rate_annual=0)
        self.assertEqual(result.net_return.sum(), 0.0)

    def test_score_has_no_future_dependency(self):
        bars = sample(2)
        before = scores(bars, StrategyParams(3))
        changed = bars.copy()
        changed.loc[changed.index[-4:], "close"] *= 10
        after = scores(changed, StrategyParams(3))
        pd.testing.assert_series_equal(before.iloc[:-4], after.iloc[:-4])

    def test_resample_does_not_bridge_lunch(self):
        bars = sample(1)
        extra = bars.iloc[:8].copy()
        extra["timestamp"] = extra["timestamp"] + pd.Timedelta(hours=3, minutes=30)
        out = resample_bars(pd.concat([bars, extra], ignore_index=True), 5)
        self.assertFalse(((out.timestamp.dt.hour == 11) & (out.timestamp.dt.minute >= 30)).any())

    def test_walk_forward_is_strictly_chronological(self):
        returns, folds, summary = run_walk_forward(sample(), WalkForwardConfig(
            train_days=6, test_days=3, step_days=3, lookbacks=(3,),
            directions=(1,), entry_quantiles=(.5,),
        ))
        self.assertTrue((pd.to_datetime(folds.train_end) < pd.to_datetime(folds.test_start)).all())
        self.assertGreater(len(returns), 0)
        self.assertIn("sharpe", summary)

    def test_point_in_time_market_cap_uses_prior_close(self):
        daily = pd.DataFrame({
            "date": pd.to_datetime(["2025-01-01", "2025-01-02"] * 1),
            "symbol": ["1000", "1000"], "close": [1000.0, 2000.0],
        })
        membership = pd.DataFrame({"symbol": ["1000"], "effective_from": ["2024-01-01"],
                                   "effective_to": [None]})
        shares = pd.DataFrame({"symbol": ["1000"], "known_at": ["2024-01-01"],
                               "shares": [100_000_000]})
        universe = build_point_in_time_universe(daily, membership, shares)
        self.assertEqual(universe.date.iloc[0], pd.Timestamp("2025-01-02"))
        self.assertEqual(universe.market_cap.iloc[0], 100_000_000_000)

    def test_model_walk_forward_is_out_of_sample(self):
        returns, folds, coefficients, summary = run_model_walk_forward(sample(14, 6), ModelConfig(
            train_days=8, test_days=3, step_days=3, alphas=(1.0,),
        ))
        self.assertTrue((pd.to_datetime(folds.train_end) < pd.to_datetime(folds.test_start)).all())
        self.assertGreater(len(coefficients), 0)
        self.assertIn("max_drawdown", summary)


if __name__ == "__main__":
    unittest.main()
