import unittest

import pandas as pd

from trading.jp_intraday.strategies import unit_lot_backtest


class UnitLotEmptyFrameTest(unittest.TestCase):
    def test_typed_empty_frame_does_not_raise(self):
        # 強いユニバース制約でウォークフォワードが空になると、score_frame は
        # 価格列（open/raw_open）を持たない型付き空フレームを返す。
        # unit_lot_backtest は KeyError を出さず空の日次/明細を返すこと（回帰: #8）。
        empty = pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"),
                              "symbol": pd.Series(dtype=str),
                              "pred": pd.Series(dtype=float),
                              "_s": pd.Series(dtype=float)})
        daily, blotter = unit_lot_backtest(empty)
        self.assertTrue(daily.empty)
        self.assertTrue(blotter.empty)
        self.assertIn("net", daily.columns)
        self.assertIn("date", daily.columns)

    def test_completely_empty_frame_does_not_raise(self):
        daily, blotter = unit_lot_backtest(pd.DataFrame())
        self.assertTrue(daily.empty)
        self.assertTrue(blotter.empty)


if __name__ == "__main__":
    unittest.main()
