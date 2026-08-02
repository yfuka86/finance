"""気配歪みキャリブレーション（実弾GO/NO-GOの対応表）の部品テスト."""
import glob
import os
import unittest

import numpy as np
import pandas as pd

from scripts.quote_distortion_calibration import book_overlap, distort, recompute_gap_features


def _panel(n_days=3, n_sym=20):
    rng = np.random.default_rng(0)
    rows = []
    for d in pd.date_range("2026-01-05", periods=n_days, freq="B"):
        for i in range(n_sym):
            rows.append({"date": d, "symbol": f"{1000+i}0", "sector": f"S{i % 4}",
                         "overnight_gap": rng.normal(0, 0.02),
                         "sector_index_gap": rng.normal(0, 0.005),
                         "gap_vol60": 0.02})
    return pd.DataFrame(rows)


class TestRecompute(unittest.TestCase):
    def test_residual_gap_is_demeaned_per_date(self):
        p = recompute_gap_features(_panel())
        self.assertTrue(np.allclose(p.groupby("date")["residual_gap"].mean(), 0, atol=1e-12))

    def test_sector_resid_gap_is_demeaned_within_date_sector(self):
        p = recompute_gap_features(_panel())
        m = p.groupby(["date", "sector"])["sector_resid_gap"].mean()
        self.assertTrue(np.allclose(m, 0, atol=1e-12))

    def test_gap_vol60_is_not_distorted(self):
        """gap_vol60 は過去実績なので歪ませてはいけない（当日気配とは無関係）."""
        base = _panel()
        out = distort(base, sigma_bps=500, seed=1)
        pd.testing.assert_series_equal(base["gap_vol60"], out["gap_vol60"])

    def test_compression_scales_gap_but_keeps_ranking(self):
        base = recompute_gap_features(_panel())
        out = distort(_panel(), lam=0.3)
        self.assertAlmostEqual(
            float(out["residual_gap"].std() / base["residual_gap"].std()), 0.3, places=6)
        for d, g in out.groupby("date"):
            b = base[base["date"].eq(d)].set_index("symbol")["residual_gap"]
            self.assertGreater(g.set_index("symbol")["residual_gap"].corr(b, method="spearman"),
                               0.999)

    def test_clip_bounds_the_gap(self):
        out = distort(_panel(), clip_pct=1.0)
        self.assertLessEqual(out["overnight_gap"].abs().max(), 0.01 + 1e-12)


class TestBookOverlap(unittest.TestCase):
    def _blot(self, syms, side="LONG", day="2026-01-05"):
        return pd.DataFrame({"date": pd.to_datetime([day] * len(syms)), "symbol": syms,
                             "side_label": [side] * len(syms),
                             "position_yen": [1e6] * len(syms)})

    def test_identical_books_are_fully_overlapping(self):
        b = self._blot(["1", "2", "3", "4"])
        self.assertEqual(book_overlap(b, b), (1.0, 1.0))

    def test_disjoint_books_have_zero_overlap(self):
        n, y = book_overlap(self._blot(["1", "2"]), self._blot(["3", "4"]))
        self.assertEqual((n, y), (0.0, 0.0))

    def test_same_symbol_opposite_side_is_not_a_match(self):
        """符号が逆なら別建玉。銘柄名だけで一致とみなしてはいけない."""
        n, _ = book_overlap(self._blot(["1", "2"], "LONG"), self._blot(["1", "2"], "SHORT"))
        self.assertEqual(n, 0.0)

    def test_partial_overlap(self):
        n, _ = book_overlap(self._blot(["1", "2", "3", "4"]), self._blot(["1", "2", "9", "8"]))
        self.assertAlmostEqual(n, 0.5)


class TestOfficialPricesCoverage(unittest.TestCase):
    def test_official_prices_are_not_a_3pct_subsample(self):
        """daily_adj の生値列 O/C は年により欠落（2022-24は列自体が無い・2025は3.2%）。

        決め打ちで読むと3%のデータだけで実効コストを判定してしまうので、
        調整値フォールバックが効いていることを確認する。
        """
        if not glob.glob("data/jp_daily_history/daily_adj_202[4-9].parquet"):
            self.skipTest("日次データが未取得の環境")
        if os.environ.get("SKIP_DATA_TESTS"):
            self.skipTest("SKIP_DATA_TESTS")
        from scripts.analyze_effective_cost import _official_prices
        d = _official_prices()
        self.assertGreater(d["open"].notna().mean(), 0.5)


if __name__ == "__main__":
    unittest.main()
