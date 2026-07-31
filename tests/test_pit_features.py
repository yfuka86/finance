"""特徴量の PIT（先読み防止）ガード.

2026-07-31 に「気配不要ML」に `vol20`（shiftなしの20日ローリング標準偏差＝当日の
終値リターンを含む）を足してしまい、選択窓 Sharpe が 1.30→2.75 に跳ねた。
本番の BASE_FEATURES は無傷だったが、同じ踏み方を繰り返さないよう固定する。
"""
import unittest

import numpy as np
import pandas as pd

from trading.jp_intraday.daily_model import BASE_FEATURES

# 当日の終値/日中リターンを含むため、寄付き時点では未知＝予測特徴量に使えない列
NON_PIT_COLS = {"vol20", "vol20_floor", "intraday_ret", "target", "ret", "close", "value"}


class TestFeaturePIT(unittest.TestCase):
    def test_base_features_contain_no_lookahead_columns(self):
        leaked = sorted(set(BASE_FEATURES) & NON_PIT_COLS)
        self.assertEqual(leaked, [], f"BASE_FEATURES に先読み列が混入: {leaked}")

    def test_ivol_is_the_pit_safe_volatility(self):
        """予測に使ってよいボラは ivol（shift=1）であって vol20 ではない."""
        self.assertIn("ivol", BASE_FEATURES)
        self.assertNotIn("vol20", BASE_FEATURES)

    def test_vol20_is_documented_as_non_pit(self):
        """vol20 の定義箇所に PIT 警告コメントがあること（次に触る人が踏まないように）."""
        src = open("trading/jp_intraday/daily_model.py", encoding="utf-8").read()
        i = src.index('p["vol20"] = _groll')
        self.assertIn("PIT注意", src[max(0, i - 400):i])


class TestGrollShiftSemantics(unittest.TestCase):
    def test_shift1_rolling_excludes_current_row(self):
        """shift=1 のローリングが「当日を含まない」ことを直接確認する."""
        n = 30
        p = pd.DataFrame({"symbol": ["A"] * n, "x": [1.0] * (n - 1) + [100.0]})

        def groll(series, window, minp, fn, shift=0):
            r = getattr(series.groupby(p["symbol"]).rolling(window, min_periods=minp), fn)()
            r = r.reset_index(level=0, drop=True)
            return r.groupby(p["symbol"]).shift(shift) if shift else r

        no_shift = groll(p["x"], 20, 10, "std")
        shifted = groll(p["x"], 20, 10, "std", shift=1)
        # 最終行: shiftなしは急変(100)を取り込んで非ゼロ、shift=1 は取り込まない
        self.assertGreater(no_shift.iloc[-1], 1.0)
        self.assertTrue(np.isclose(shifted.iloc[-1], 0.0))


if __name__ == "__main__":
    unittest.main()


class TestShortablePIT(unittest.TestCase):
    """shortable が「当時の」貸借区分であること（2026-07-31 発見のルックアヘッド）.

    master.parquet は Date が1枚だけのスナップショット。そこから作った貸借フラグを
    日付キーなしで全期間にマージすると、後から貸借になった銘柄が過去に遡って
    ショート可能になる。週次 margin_interest の IssType で PIT 復元する。
    """

    def test_master_is_a_single_snapshot(self):
        """前提の明示: master は時系列ではない（この前提が崩れたら実装を見直す）."""
        import os

        import pandas as pd
        path = "data/jp_daily_history/master.parquet"
        if not os.path.exists(path):
            self.skipTest("master 未取得の環境")
        self.assertEqual(pd.read_parquet(path, columns=["Date"])["Date"].nunique(), 1)

    def test_pit_lendable_loader_returns_time_series(self):
        import glob as g

        from trading.jp_intraday.daily_model import _load_pit_lendable
        if not g.glob("data/jp_flows/margin_interest_*.parquet"):
            self.skipTest("週次信用残が未取得の環境")
        d = _load_pit_lendable()
        self.assertIsNotNone(d)
        self.assertGreater(d["date"].nunique(), 100)      # 1枚スナップショットではない
        self.assertEqual(d["pit_lendable"].dtype, bool)

    def test_panel_shortable_matches_pit_rate_not_master_rate(self):
        """パネルの shortable 率が master 由来(約96%)でなく PIT(約92.6%)に寄ること."""
        import glob as g

        import pandas as pd
        if not g.glob("data/jp_flows/margin_interest_*.parquet"):
            self.skipTest("週次信用残が未取得の環境")
        from trading.jp_intraday.daily_model import load_panel_cached
        p = load_panel_cached(min_value_yen=1e9)
        rate = p[p["date"] <= pd.Timestamp("2024-12-31")]["shortable"].mean()
        self.assertLess(rate, 0.945, "master スナップショット由来のままの可能性")
        self.assertGreater(rate, 0.90)
