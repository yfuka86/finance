"""Guards for the shared helpers extracted during refactor + the gross-cost fix."""
import unittest

import numpy as np
import pandas as pd

from trading.jp_intraday.backtest import _bars_per_year
from trading.jp_intraday.collector import _strip_jquants_suffix, _to_jquants_codes
from trading.jp_intraday.data import am_pm_session, minute_of_day
from trading.jp_intraday.experiments import evaluate_strategy, prepare_signals
from trading.jp_intraday.strategy import (
    market_neutral_weights, rank_long_short_weights,
)


def _sample(days=16, symbols=8):
    rng = np.random.default_rng(3)
    rows = []
    for day in pd.bdate_range("2025-02-03", periods=days, tz="Asia/Tokyo"):
        slots = [(9, m) for m in range(120)] + [(12, 30 + m) for m in range(90)]
        for hour, minute in slots:
            ts = day + pd.Timedelta(hours=hour, minutes=minute)
            base = hour * 60 + minute
            for s in range(symbols):
                px = 1000 + s * 40 + base * (s - 3.5) * 0.02 + rng.normal(0, 0.4)
                rows.append((ts, str(1000 + s), px, px + 0.5, px - 0.5, px, 1500 + s))
    return pd.DataFrame(rows, columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"])


class HelperTest(unittest.TestCase):
    def test_jquants_code_roundtrip(self):
        self.assertEqual(_to_jquants_codes(["7203", "72030", 6758]), {"72030", "67580"})
        stripped = _strip_jquants_suffix(pd.Series(["72030", "67580"]))
        self.assertEqual(list(stripped), ["7203", "6758"])

    def test_session_helpers(self):
        ts = pd.to_datetime(pd.Series(["2025-02-03 09:30", "2025-02-03 13:00"])).dt.tz_localize("Asia/Tokyo")
        self.assertEqual(list(minute_of_day(ts)), [9 * 60 + 30, 13 * 60])
        self.assertEqual(list(am_pm_session(ts)), ["am", "pm"])

    def test_bars_per_year(self):
        self.assertEqual(_bars_per_year(5), 245 * 60)
        self.assertEqual(_bars_per_year(1), 245 * 300)

    def test_rank_and_threshold_weights_are_dollar_neutral(self):
        frame = pd.DataFrame({
            "timestamp": pd.to_datetime(["2025-02-03 09:00"] * 6),
            "symbol": list("abcdef"),
        })
        score = pd.Series([5.0, 4.0, 3.0, -3.0, -4.0, -5.0], index=frame.index)
        w = rank_long_short_weights(frame, score, 0.34)
        self.assertAlmostEqual(w[w > 0].sum(), 0.5)
        self.assertAlmostEqual(w[w < 0].sum(), -0.5)
        self.assertAlmostEqual(w.sum(), 0.0)
        # threshold builder shares the same normalization tail
        tw = market_neutral_weights(frame, score, 3.5)
        self.assertAlmostEqual(tw.sum(), 0.0)
        self.assertAlmostEqual(tw[tw > 0].sum(), 0.5)


class GrossCostFixTest(unittest.TestCase):
    def test_gross_excludes_borrow_and_slippage(self):
        prepared = prepare_signals(
            _sample().pipe(lambda b: b),  # 1m bars are fine for prepare_signals at interval 5
            pd.DataFrame({"symbol": [str(1000 + s) for s in range(8)], "sector": ["A"] * 4 + ["B"] * 4}),
            5,
        )
        _, summary = evaluate_strategy(prepared, "reversal_30m", "2025-02-03", "2025-02-24", 5, 0.25)
        net = summary["total_return"]
        # A short-holding strategy pays borrow + slippage, so true gross must sit
        # strictly above the net total return.
        self.assertGreater(summary["gross_return_sum"], net)


if __name__ == "__main__":
    unittest.main()
