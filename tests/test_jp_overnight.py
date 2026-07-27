"""Leakage-safety + correctness for the overnight-gap features."""
import unittest

import numpy as np
import pandas as pd

from trading.jp_intraday.overnight import prepare_overnight


def _bars(days=3, symbols=2):
    rows = []
    for di, day in enumerate(pd.bdate_range("2025-03-03", periods=days, tz="Asia/Tokyo")):
        slots = [(9, m) for m in range(6)] + [(12, 30 + m) for m in range(6)]
        for hour, minute in slots:
            ts = day + pd.Timedelta(hours=hour, minutes=minute)
            for s in range(symbols):
                px = 100 + s * 10 + di + (0.5 if (hour, minute) == (9, 0) else 0.0)
                rows.append((ts, str(s), px, px + 0.2, px - 0.2, px + 0.05, 1000))
    return pd.DataFrame(rows, columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"])


SECTORS = pd.DataFrame({"symbol": ["0", "1"], "sector": ["A", "A"]})


class OvernightTest(unittest.TestCase):
    def test_gap_uses_prior_close_and_todays_open(self):
        f = prepare_overnight(_bars(), SECTORS)
        s0 = f[f.symbol.eq("0")].sort_values("timestamp")
        opens = s0.groupby(s0.timestamp.dt.date)["open"].first()
        closes = s0.groupby(s0.timestamp.dt.date)["close"].last()
        dates = list(opens.index)
        expected = opens.iloc[1] / closes.iloc[0] - 1
        got = s0[s0.timestamp.dt.normalize().dt.date.eq(dates[1])]["gap"].iloc[0]
        self.assertAlmostEqual(got, expected)
        # first observed day has no prior close -> gap is NaN
        self.assertTrue(np.isnan(s0[s0.timestamp.dt.normalize().dt.date.eq(dates[0])]["gap"].iloc[0]))

    def test_gap_has_no_future_dependency(self):
        base = _bars()
        before = prepare_overnight(base, SECTORS)[["timestamp", "symbol", "gap", "residual_gap"]]
        # Perturb every bar strictly after the last day's opening bar.
        changed = base.copy()
        last_day = changed.timestamp.dt.normalize().max()
        first_open_ts = changed[changed.timestamp.dt.normalize().eq(last_day)].timestamp.min()
        mask = changed.timestamp.gt(first_open_ts)
        changed.loc[mask, ["open", "high", "low", "close"]] *= 5.0
        after = prepare_overnight(changed, SECTORS)[["timestamp", "symbol", "gap", "residual_gap"]]
        # Gaps for all but the perturbed future bars must be identical.
        keep = before["timestamp"].le(first_open_ts)
        pd.testing.assert_frame_equal(
            before[keep].reset_index(drop=True), after[keep].reset_index(drop=True)
        )


if __name__ == "__main__":
    unittest.main()
