"""Correctness + no-lookahead for the index-futures overnight factor."""
import unittest

import pandas as pd

from trading.jp_intraday.futures_context import front_month, overnight_factor


def _futures():
    # Two dates, two NK225F contracts (front + next). Front = nearest LTD.
    # Session sequence per date: O (open) -> C (day close) -> EC (night close).
    rows = [
        # Date, Code, ProdCat, O, C, EC, OI, LTD, Settle
        ("2024-01-04", "F1", "NK225F", 100.0, 110.0, 121.0, 5000, "2024-03-08", 110.0),
        ("2024-01-04", "F2", "NK225F", 100.0, 110.0, 121.0, 100,  "2024-06-14", 110.0),
        ("2024-01-05", "F1", "NK225F", 121.0, 130.0, 140.0, 5000, "2024-03-08", 130.0),
    ]
    df = pd.DataFrame(rows, columns=["Date", "Code", "ProdCat", "O", "C", "EC", "OI", "LTD", "Settle"])
    df["H"] = df["EC"]; df["L"] = df["O"]; df["MO"] = df["O"]; df["MC"] = df["C"]
    df["EO"] = df["C"]; df["Vo"] = 1000
    return df


class FuturesContextTest(unittest.TestCase):
    def test_front_month_picks_nearest_ltd(self):
        fm = front_month(_futures(), "NK225F")
        self.assertEqual(list(fm[fm.Date.eq("2024-01-04")]["Code"]), ["F1"])

    def test_overnight_factor_and_alignment(self):
        f = overnight_factor(_futures(), "NK225F")
        # night_ret on 2024-01-04 = EC/C - 1 = 121/110 - 1, aligned to next cash day.
        self.assertIn(pd.Timestamp("2024-01-05"), f.index)
        self.assertAlmostEqual(f.loc["2024-01-05", "night_ret"], 121.0 / 110.0 - 1)
        # day_ret on 2024-01-04 = C/O - 1 = 110/100 - 1 = 0.10
        self.assertAlmostEqual(f.loc["2024-01-05", "day_ret"], 110.0 / 100.0 - 1)


if __name__ == "__main__":
    unittest.main()
