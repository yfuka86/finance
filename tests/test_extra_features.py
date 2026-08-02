"""未使用需給データ4系統の特徴量の PIT ガード."""
import glob
import unittest

import numpy as np
import pandas as pd

from trading.jp_intraday.extra_features import EXTRA_FEATURES, _shift_sessions, _z


class TestSessionShift(unittest.TestCase):
    def setUp(self):
        # 連休を含む営業日列（1/4,1/5 のあと 1/9 に飛ぶ＝3連休）
        self.sessions = pd.Index(pd.to_datetime(
            ["2024-01-04", "2024-01-05", "2024-01-09", "2024-01-10", "2024-01-11"]))

    def test_shift_is_by_sessions_not_calendar_days(self):
        """連休跨ぎで暦日ずらしをしないこと（実際に踏んだ schema v7 の罠と同型）."""
        df = pd.DataFrame({"src": pd.to_datetime(["2024-01-05"])})
        out = _shift_sessions(df, "src", 1, self.sessions)
        # 1営業日後は 1/6(土) ではなく 1/9
        self.assertEqual(out["date"].iloc[0], pd.Timestamp("2024-01-09"))

    def test_non_session_source_date_rounds_forward(self):
        """記録日が休日（週次信用残の基準日など）なら次の営業日に丸める."""
        df = pd.DataFrame({"src": pd.to_datetime(["2024-01-06"])})   # 土曜
        out = _shift_sessions(df, "src", 0, self.sessions)
        self.assertEqual(out["date"].iloc[0], pd.Timestamp("2024-01-09"))

    def test_rows_beyond_last_session_are_dropped(self):
        """未来にはみ出す行は落とす（存在しない日付を捏造しない）."""
        df = pd.DataFrame({"src": pd.to_datetime(["2024-01-11"])})
        self.assertTrue(_shift_sessions(df, "src", 2, self.sessions).empty)

    def test_lag_is_monotone(self):
        df = pd.DataFrame({"src": pd.to_datetime(["2024-01-04"])})
        a = _shift_sessions(df, "src", 1, self.sessions)["date"].iloc[0]
        b = _shift_sessions(df, "src", 2, self.sessions)["date"].iloc[0]
        self.assertLess(a, b)


class TestZ(unittest.TestCase):
    def test_z_is_causal(self):
        """zスコアが過去のみ参照（未来の平均/分散を使わない）."""
        s = pd.Series([1.0] * 20 + [50.0])
        by = pd.Series(["A"] * 21)
        z = _z(s, by, win=10)
        self.assertTrue(np.isnan(z.iloc[0]) or abs(z.iloc[0]) < 5.001)
        self.assertGreater(z.iloc[-1], 1.0)   # 最終行の急変は検知される


class TestBuild(unittest.TestCase):
    def test_features_present_and_non_degenerate(self):
        if not glob.glob("data/jp_flows/margin_interest_*.parquet"):
            self.skipTest("需給データが未取得の環境")
        from trading.jp_intraday.extra_features import build_extra_features
        sessions = pd.Index(pd.bdate_range("2018-01-01", "2026-07-30"))
        e = build_extra_features(sessions)
        for c in ["xt_margin_ratio_z", "xt_ssr_to_so", "xt_alert_flag"]:
            self.assertIn(c, e.columns)
        self.assertGreater(len(e), 10000)


if __name__ == "__main__":
    unittest.main()
