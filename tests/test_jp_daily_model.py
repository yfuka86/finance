"""Walk-forward is strictly out-of-sample; features are point-in-time."""
import unittest

import numpy as np
import pandas as pd

from trading.jp_intraday.daily_model import (
    BASE_FEATURES, build_daily_features, walk_forward, walk_forward_predictions,
)


def _daily(years=4, symbols=40):
    rng = np.random.default_rng(1)
    rows = []
    start = pd.Timestamp("2021-01-04")
    for di, day in enumerate(pd.bdate_range(start, periods=years * 252)):
        for s in range(symbols):
            close = 1000 + s * 20 + rng.normal(0, 5)
            # engineer weak overnight reversal: open gaps, close reverts part of it
            gap = rng.normal(0, 0.01)
            open_ = close * (1 + gap)
            close_next = open_ * (1 - 0.3 * gap + rng.normal(0, 0.008))
            rows.append((day, str(1000 + s), open_, open_ * 1.01, open_ * 0.99, close_next, 2e9))
    return pd.DataFrame(rows, columns=["Date", "Code", "AdjO", "AdjH", "AdjL", "AdjC", "Va"]).assign(AdjVo=1e5)


class DailyModelTest(unittest.TestCase):
    def test_walkforward_is_out_of_sample_and_grows(self):
        panel = build_daily_features(_daily(), min_value_yen=0)
        wf = walk_forward(panel, BASE_FEATURES, quantile=0.2, cost_bps_side=3.0, min_train_years=2)
        rows = wf[wf["test_year"] != "MEAN"]
        self.assertGreater(len(rows), 0)
        # training set strictly grows as the test year advances (expanding window)
        self.assertTrue(rows["train_rows"].is_monotonic_increasing)
        self.assertIn("net_sharpe", wf.columns)

    def test_features_present_and_finite_target(self):
        panel = build_daily_features(_daily(years=3), min_value_yen=0)
        for f in BASE_FEATURES:
            self.assertIn(f, panel.columns)
        self.assertTrue(np.isfinite(panel["target"].dropna()).all())

    def test_features_are_point_in_time(self):
        # vol20 includes today's (not-yet-known) close -> must NOT be a feature.
        self.assertNotIn("vol20", BASE_FEATURES)
        self.assertIn("ivol", BASE_FEATURES)

    def test_ml_trains_on_past_only(self):
        # Perturbing the LAST year's data must not change any EARLIER year's OOS
        # predictions — proving each fold trains only on strictly-past data.
        panel = build_daily_features(_daily(years=5), min_value_yen=0)
        base = walk_forward_predictions(panel, BASE_FEATURES)
        last_year = panel["date"].dt.year.max()
        changed = panel.copy()
        mask = changed["date"].dt.year.eq(last_year)
        for col in ["residual_gap", "gap_abs", "ivol", "target"]:
            changed.loc[mask, col] = changed.loc[mask, col] * 7.0 + 1.0
        after = walk_forward_predictions(changed, BASE_FEATURES)
        b = base[base["date"].dt.year < last_year].reset_index(drop=True)
        a = after[after["date"].dt.year < last_year].reset_index(drop=True)
        self.assertGreater(len(b), 0)
        pd.testing.assert_series_equal(b["pred"], a["pred"])


if __name__ == "__main__":
    unittest.main()
