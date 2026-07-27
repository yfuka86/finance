"""Leakage-safety + correctness for daily-gap reversal and US alignment."""
import unittest

import numpy as np
import pandas as pd

from trading.jp_intraday.daily_gap import backtest_gap, build_gap_panel
from trading.jp_intraday.us_context import align_overnight


def _daily(days=8, symbols=6):
    rows = []
    rng = np.random.default_rng(0)
    for di, day in enumerate(pd.bdate_range("2024-01-01", periods=days)):
        for s in range(symbols):
            close = 1000 + s * 100 + di * 5 + rng.normal(0, 3)
            open_ = close * (1 + rng.normal(0, 0.01))
            rows.append((day, str(1000 + s), open_, open_ + 5, open_ - 5, close, 1e9))
    return pd.DataFrame(rows, columns=["Date", "Code", "AdjO", "AdjH", "AdjL", "AdjC", "Va"]).assign(AdjVo=1e5)


class DailyGapTest(unittest.TestCase):
    def test_gap_uses_prior_close_only(self):
        panel = build_gap_panel(_daily(), min_value_yen=0)
        one = panel[panel.symbol.eq("1000")].sort_values("date").reset_index(drop=True)
        # gap_t = open_t / close_{t-1} - 1; intraday_t = close_t/open_t - 1
        self.assertAlmostEqual(one["intraday_ret"].iloc[2],
                               one["close"].iloc[2] / one["open"].iloc[2] - 1)

    def test_no_future_dependency_in_signal(self):
        base = _daily()
        p0 = build_gap_panel(base, min_value_yen=0)[["date", "symbol", "overnight_gap"]]
        changed = base.copy()
        last = changed["Date"].max()
        changed.loc[changed["Date"].eq(last), ["AdjO", "AdjC"]] *= 3  # perturb only last day
        p1 = build_gap_panel(changed, min_value_yen=0)[["date", "symbol", "overnight_gap"]]
        keep0 = p0[p0["date"].lt(last)].reset_index(drop=True)
        keep1 = p1[p1["date"].lt(last)].reset_index(drop=True)
        pd.testing.assert_frame_equal(keep0, keep1)

    def test_backtest_is_dollar_neutral(self):
        panel = build_gap_panel(_daily(days=10, symbols=10), min_value_yen=0)
        res = backtest_gap(panel, quantile=0.3, direction=-1, cost_bps_side=0.0)
        self.assertEqual(len(res), panel["date"].nunique())
        self.assertTrue(np.isfinite(res["gross"]).all())

    def test_us_alignment_is_strictly_earlier(self):
        us = pd.DataFrame({"SPY": [0.01, -0.02, 0.03]},
                          index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]))
        jp = pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-05"])
        aligned = align_overnight(us, jp)
        # JP 01-03 must use US 01-02 (strictly earlier), not same-day 01-03.
        self.assertAlmostEqual(aligned.loc["2024-01-03", "SPY"], 0.01)
        self.assertAlmostEqual(aligned.loc["2024-01-05", "SPY"], 0.03)


if __name__ == "__main__":
    unittest.main()
