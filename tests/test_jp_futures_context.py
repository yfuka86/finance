"""Correctness + no-lookahead for the index-futures overnight factor."""
import unittest

import pandas as pd

from trading.jp_intraday.futures_context import front_month, overnight_factor


def _futures():
    # Two dates, two NK225F contracts (front + next). Front = nearest LTD.
    # 実データのセッション順: ナイト(EO->EC, 前日夕〜当日朝06:00) が先、その後 日中(->C)。
    rows = [
        # Date, Code, ProdCat, O, C, EC, OI, LTD, Settle
        ("2024-01-04", "F1", "NK225F", 100.0, 110.0, 121.0, 5000, "2024-03-08", 110.0),
        ("2024-01-04", "F2", "NK225F", 100.0, 110.0, 121.0, 100,  "2024-06-14", 110.0),
        ("2024-01-05", "F1", "NK225F", 121.0, 130.0, 140.0, 5000, "2024-03-08", 130.0),
    ]
    df = pd.DataFrame(rows, columns=["Date", "Code", "ProdCat", "O", "C", "EC", "OI", "LTD", "Settle"])
    df["H"] = df["EC"]; df["L"] = df["O"]; df["MO"] = df["O"]; df["MC"] = df["C"]
    df["EO"] = df["O"]; df["Vo"] = 1000   # 実データではナイトが先＝日通し始値 O == EO (99.9%)
    return df


class FuturesContextTest(unittest.TestCase):
    def test_front_month_picks_nearest_ltd(self):
        fm = front_month(_futures(), "NK225F")
        self.assertEqual(list(fm[fm.Date.eq("2024-01-04")]["Code"]), ["F1"])

    def test_overnight_factor_and_alignment(self):
        f = overnight_factor(_futures(), "NK225F")
        # 真のオーバーナイト: 当日朝06:00のナイト引け EC_D ÷ 前日の日中引け C_{D-1}
        # 2024-01-05: EC=140, 前日 C=110 → 140/110-1（当日の寄付き前に確定）
        self.assertIn(pd.Timestamp("2024-01-05"), f.index)
        self.assertAlmostEqual(f.loc["2024-01-05", "night_ret"], 140.0 / 110.0 - 1)
        # 日中セッション: C_D / EO_D - 1（EO=前日Cを模したフィクスチャでは 130/121-1）
        self.assertAlmostEqual(f.loc["2024-01-05", "day_ret"], 130.0 / 121.0 - 1)
        # 初日はナイトの起点(前日引け)が無いので NaN
        self.assertTrue(pd.isna(f.loc["2024-01-04", "night_ret"]))


if __name__ == "__main__":
    unittest.main()
