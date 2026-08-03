"""50銘柄PUSH実験の中核（候補選定・一致率・気配値の取り方）の回帰テスト。"""
import datetime as dt
import unittest

import pandas as pd

from trading.jp_intraday.live.push_experiment import (
    MAX_PUSH, book_from_scores, book_overlap, quote_from_board, select_symbols, wait_until,
)


def frame(n=200):
    return pd.DataFrame({
        "symbol": [f"{1000+i}0" for i in range(n)],
        "_s": [(-1) ** i * (i % 50) / 10 for i in range(n)],
        "residual_gap": [((i % 41) - 20) / 100 for i in range(n)],
        "prev_value": [1e9 + i * 1e7 for i in range(n)],
    })


class SelectTest(unittest.TestCase):
    def test_never_exceeds_push_limit(self):
        for m in ("strategy", "absgap", "liquidity"):
            got = select_symbols(frame(), m, n=80)
            self.assertLessEqual(len(got), MAX_PUSH, m)
            self.assertEqual(len(set(got)), len(got), f"{m}: 重複がある")

    def test_strategy_selection_takes_both_extremes(self):
        f = frame()
        got = set(select_symbols(f, "strategy", n=20))
        self.assertTrue(got & set(f.nlargest(3, "_s")["symbol"]), "上位が入っていない")
        self.assertTrue(got & set(f.nsmallest(3, "_s")["symbol"]), "下位が入っていない")

    def test_absgap_selection_takes_gap_tails(self):
        f = frame()
        got = set(select_symbols(f, "absgap", n=20))
        self.assertTrue(got & set(f.nlargest(3, "residual_gap")["symbol"]))
        self.assertTrue(got & set(f.nsmallest(3, "residual_gap")["symbol"]))

    def test_liquidity_is_the_control_group(self):
        f = frame()
        got = select_symbols(f, "liquidity", n=10)
        self.assertEqual(set(got), set(f.nlargest(10, "prev_value")["symbol"]))

    def test_empty_frame_is_safe(self):
        self.assertEqual(select_symbols(frame(0), "strategy"), [])

    def test_selection_contains_the_book_it_would_trade(self):
        # 2パス方式の前提: 早い1周で選んだ建玉は候補50に必ず含まれること
        f = frame()
        chosen = set(select_symbols(f, "strategy", n=MAX_PUSH))
        book = book_from_scores(f, names_per_side=8)
        self.assertTrue(book <= chosen, "早い1周の建玉が候補に含まれていない")


class OverlapTest(unittest.TestCase):
    def test_identical_books_are_100pct(self):
        self.assertEqual(book_overlap({"a", "b"}, {"a", "b"}), 1.0)

    def test_disjoint_books_are_zero(self):
        self.assertEqual(book_overlap({"a"}, {"b"}), 0.0)

    def test_half_overlap(self):
        self.assertAlmostEqual(book_overlap({"a", "b"}, {"b", "c"}), 1 / 3)


class QuoteTest(unittest.TestCase):
    def test_prefers_bid_ask_mid_over_calcprice(self):
        # 寄前の CalcPrice は前日終値のまま（2026-07-31 実測）。気配は bid/ask にある
        b = {"BidPrice": 1270.5, "AskPrice": 1270.0, "CalcPrice": 1295.5,
             "CurrentPrice": 1295.5}
        self.assertAlmostEqual(quote_from_board(b), 1270.25)

    def test_falls_back_to_single_side(self):
        self.assertEqual(quote_from_board({"BidPrice": 100.0, "CalcPrice": 90.0}), 100.0)

    def test_falls_back_to_current_when_no_quote(self):
        self.assertEqual(quote_from_board({"CurrentPrice": 55.0}), 55.0)

    def test_zero_when_nothing(self):
        self.assertEqual(quote_from_board({}), 0.0)


class WaitTest(unittest.TestCase):
    def test_returns_immediately_when_time_has_passed(self):
        now = dt.datetime(2026, 8, 3, 9, 30)
        slept = []
        wait_until("08:50", now_fn=lambda: now, sleep=slept.append)
        self.assertEqual(slept, [])

    def test_sleeps_until_target(self):
        clock = {"t": dt.datetime(2026, 8, 3, 8, 49, 30)}

        def sleep(sec):
            clock["t"] += dt.timedelta(seconds=sec)

        wait_until("08:50", now_fn=lambda: clock["t"], sleep=sleep)
        self.assertGreaterEqual(clock["t"], dt.datetime(2026, 8, 3, 8, 50))
        self.assertLess(clock["t"], dt.datetime(2026, 8, 3, 8, 50, 6))



class EnsembleBookTest(unittest.TestCase):
    """ensemble_core は「スリーブごとに上位/下位n本→統合」。z合成の1本とは別物。"""

    def _last(self, n=120):
        import numpy as np
        return pd.DataFrame({
            "symbol": [f"{1000+i}0" for i in range(n)],
            "shortable": [True] * n,
            "prev_value": [5e9] * n,
            "short_restricted": [False] * n,
        })

    def test_union_of_sleeves_and_candidates_contain_it(self):
        from unittest import mock
        from trading.jp_intraday.live import push_experiment as pxm

        last = self._last()
        # 2スリーブが別々の銘柄を選ぶ状況を作る
        def fake_score(l, opens, member):
            f = l.copy()
            f["_s"] = range(len(f)) if member == "A" else range(len(f), 0, -1)
            return f

        spec = {"kind": "ensemble", "members": [("A", 0.5), ("B", 0.5)]}
        with mock.patch.dict("trading.jp_intraday.strategies.STRATEGIES",
                             {"ENS": spec, "A": spec, "B": spec}, clear=False), \
             mock.patch("trading.jp_intraday.live.executor._score_today", fake_score):
            book, per = pxm.ensemble_book(last, {}, "ENS", names_per_side=8)
            self.assertEqual(sorted(per), ["A", "B"])
            self.assertEqual(book, per["A"] | per["B"])
            chosen, diag = pxm.select_for_ensemble(last, {}, "ENS", 8, n=50)
            self.assertLessEqual(len(chosen), 50)
            self.assertTrue(book <= set(chosen), "建玉が候補に入っていない")
            self.assertGreater(diag["depth_used"], 8, "次点で枠を埋めていない")

    def test_short_side_respects_eligibility(self):
        from trading.jp_intraday.live import push_experiment as pxm
        f = self._last(40)
        f["_s"] = range(40)
        f.loc[f.index[:20], "shortable"] = False        # 下位20本は売建不可
        book = pxm.book_from_scores(f, names_per_side=5)
        shorts = {s for s in book if s not in set(f.nlargest(5, "_s")["symbol"])}
        self.assertTrue(all(f.set_index("symbol").loc[s, "shortable"] for s in shorts),
                        "売建不可の銘柄がショートに入っている")

if __name__ == "__main__":
    unittest.main()


class OutPathTest(unittest.TestCase):
    def test_dry_run_writes_to_a_separate_file(self):
        from trading.jp_intraday.live.push_experiment import out_path
        real = out_path("2026-08-03", dry=False).name
        dry = out_path("2026-08-03", dry=True).name
        self.assertNotEqual(real, dry, "スモークテストが本番記録を上書きする")
        self.assertTrue(real.startswith("push_experiment_2"))
        self.assertIn("_dry_", dry)
