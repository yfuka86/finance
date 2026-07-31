"""売買内訳フロー特徴量の健全性テスト（PIT・恒等式・標準化）."""
import glob
import unittest

import numpy as np
import pandas as pd

from trading.jp_intraday.flow_features import (
    _BUY, _SELL, FLOW_FEATURES, _zscore, build_flow_features,
)


class TestAccountingIdentity(unittest.TestCase):
    def test_buy_equals_sell_so_net_is_not_a_feature(self):
        """★売買内訳は会計恒等式で 買い代金≡売り代金。

        (buy-sell)/total を特徴量にすると**定義上つねに0**になる（実際に踏んだ罠）。
        情報はネットでなく構成比にあることを、データ側から固定しておく。
        """
        fs = sorted(glob.glob("data/jp_flows/breakdown_*.parquet"))
        if not fs:
            self.skipTest("売買内訳データが未取得の環境")
        d = pd.read_parquet(fs[-1], columns=_SELL + _BUY)
        self.assertGreater(np.isclose(d[_BUY].sum(axis=1), d[_SELL].sum(axis=1)).mean(), 0.999)

    def test_shipped_features_are_non_degenerate(self):
        """出荷している特徴量が定数（std=0）でないこと＝上の罠の再発防止."""
        fs = sorted(glob.glob("data/jp_flows/breakdown_*.parquet"))
        if not fs:
            self.skipTest("売買内訳データが未取得の環境")
        f = build_flow_features(lag=2)
        for c in FLOW_FEATURES:
            self.assertGreater(float(f[c].std(skipna=True)), 0.1, f"{c} が退化している")


class TestZScore(unittest.TestCase):
    def test_zscore_is_per_symbol(self):
        """銘柄ごとに標準化されること（水準差の異なる銘柄が混ざらない）."""
        n = 60
        s = pd.Series(list(np.arange(n) * 1.0) + list(np.arange(n) * 1.0 + 1000))
        by = pd.Series(["A"] * n + ["B"] * n)
        z = _zscore(s, by, win=20)
        a, b = z[:n].dropna(), z[n:].dropna()
        # 水準が1000違っても、銘柄内標準化なので同じ形になる
        np.testing.assert_allclose(a.to_numpy()[-10:], b.to_numpy()[-10:], rtol=1e-9)

    def test_zscore_uses_only_past(self):
        """rolling は過去のみ参照（先読みしない）."""
        s = pd.Series([1.0] * 30 + [100.0] + [1.0] * 30)
        by = pd.Series(["A"] * 61)
        z = _zscore(s, by, win=20)
        # 急変の**前**の値は急変の影響を受けない
        self.assertTrue(np.isclose(z.iloc[29], 0.0) or np.isnan(z.iloc[29]))


class TestPIT(unittest.TestCase):
    def test_lag_shifts_by_business_sessions_not_calendar_days(self):
        """営業日インデックスでずらすこと（連休跨ぎで新しいデータを掴まない）.

        lag を1増やすと、各銘柄の系列がちょうど1営業日ぶん後ろにずれるはず。
        """
        if not glob.glob("data/jp_flows/breakdown_*.parquet"):
            self.skipTest("売買内訳データが未取得の環境")
        f1 = build_flow_features(lag=1)
        f2 = build_flow_features(lag=2)
        sym = f1["symbol"].iloc[0]
        a = f1[f1["symbol"].eq(sym)].set_index("date")["flow_close_z"].dropna()
        b = f2[f2["symbol"].eq(sym)].set_index("date")["flow_close_z"].dropna()
        sessions = a.index.union(b.index).sort_values()
        shifted = a.reindex(sessions).shift(1)
        common = shifted.dropna().index.intersection(b.index)
        self.assertGreater(len(common), 100)
        np.testing.assert_allclose(shifted[common].to_numpy(), b[common].to_numpy(), rtol=1e-9)

    def test_lag_is_strictly_positive(self):
        """lag=0（＝当日のフローを当日使う）はリークなので既定にしない."""
        import inspect
        sig = inspect.signature(build_flow_features)
        self.assertGreaterEqual(sig.parameters["lag"].default, 2)


if __name__ == "__main__":
    unittest.main()
