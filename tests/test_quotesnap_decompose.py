"""quotesnap 誤差分解（実弾GO/NO-GO の計測器）の妥当性テスト.

この計測器が誤ると「気配が使えない」状態を「使える」と誤判定して実弾に進む。
旧版は実ギャップを気配に回帰しており、誤差変数バイアスで**純ランダムノイズを
圧縮（＝無害・補正可能）として報告**していた。その退行を防ぐ。
"""
import unittest

import numpy as np

from scripts.analyze_quotesnap import decompose


def _truth(n=3000, seed=0):
    return np.random.default_rng(seed).standard_t(3, n) * 1.2


class TestDecompose(unittest.TestCase):
    def test_uniform_compression_is_recovered_and_flagged_harmless(self):
        """決定論的な一様圧縮: λを真値通り回収し、R²≈1・σ≈0・選択一致100%."""
        a = _truth()
        r = decompose(a * 0.3, a)
        self.assertAlmostEqual(r["lam"], 0.30, places=2)   # 旧版は 3.33（逆数）を返していた
        self.assertGreater(r["r2"], 0.999)
        self.assertLess(r["sigma_bps"], 1.0)
        self.assertEqual(r["overlap"], 1.0)

    def test_random_noise_is_not_mistaken_for_compression(self):
        """★本丸の退行テスト: ランダムノイズをλ<1（=圧縮・無害）と報告してはならない."""
        a = _truth()
        q = a + np.random.default_rng(1).normal(0, 2.5, len(a))   # σ=250bps
        r = decompose(q, a)
        self.assertGreater(r["lam"], 0.9)          # 圧縮ではない＝λは1近傍
        self.assertLess(r["r2"], 0.6)              # R²が判別子として低く出る
        self.assertAlmostEqual(r["sigma_bps"], 250, delta=30)     # σの真値を回収

    def test_sigma_scales_with_injected_noise(self):
        a = _truth()
        rng = np.random.default_rng(2)
        s250 = decompose(a + rng.normal(0, 2.5, len(a)), a)["sigma_bps"]
        s500 = decompose(a + rng.normal(0, 5.0, len(a)), a)["sigma_bps"]
        self.assertAlmostEqual(s500 / s250, 2.0, delta=0.3)

    def test_mixture_separates_systematic_and_random(self):
        """圧縮とノイズが混ざっていても両方を分離して回収できる."""
        a = _truth()
        q = a * 0.3 + np.random.default_rng(3).normal(0, 2.5, len(a))
        r = decompose(q, a)
        self.assertAlmostEqual(r["lam"], 0.30, delta=0.05)
        self.assertAlmostEqual(r["sigma_bps"], 250, delta=30)

    def test_clip_kills_selection_despite_perfect_rank_correlation(self):
        """クリップは順位相関1.000でも上位k選択が壊れる（svdnスリーブの死因）.

        順位相関を合格判定に使ってはいけないことの根拠。
        """
        a = _truth()
        r = decompose(np.clip(a, -3, 3), a)
        self.assertGreater(r["rho"], 0.99)         # 順位は完全に保たれて見えるが…
        self.assertLess(r["overlap"], 0.3)         # …選ぶ銘柄は総取っ替えになる

    def test_lam_big_detects_clipping(self):
        """大ギャップ帯のλが全体より小さい＝クリップの検知."""
        a = _truth()
        r = decompose(np.clip(a, -3, 3), a)
        self.assertIn("lam_big", r)
        self.assertLess(r["lam_big"], r["lam"])


if __name__ == "__main__":
    unittest.main()
