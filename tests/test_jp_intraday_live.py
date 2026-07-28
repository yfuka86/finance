"""Offline tests for the live-trading module (no kabuステーション / Windows needed)."""
import unittest

import pandas as pd

from trading.jp_intraday.live.config import LiveConfig
from trading.jp_intraday.live.executor import compute_today_signals, enter, exit_all
from trading.jp_intraday.live.kabu_client import (
    KabuAPIError, KabuClient, SIDE_BUY, SIDE_SELL, to_kabu_symbol,
)
from trading.jp_intraday.live.mock_client import MockKabuClient


class SymbolTest(unittest.TestCase):
    def test_jquants_to_kabu(self):
        self.assertEqual(to_kabu_symbol("13010"), "1301")   # 5-digit -> 4-char
        self.assertEqual(to_kabu_symbol("130A0"), "130A")   # alphanumeric new code
        self.assertEqual(to_kabu_symbol("1301"), "1301")    # already 4-char passthrough


class ConfigGateTest(unittest.TestCase):
    def test_margin_ratio_scales_gross_target(self):
        # 本番推奨: ¥20M×信用2.0倍 → グロス目標¥40M（max_gross も自動整合）
        cfg = LiveConfig()
        self.assertEqual(cfg.margin_ratio, 2.0)
        self.assertEqual(cfg.names_per_side, 8)
        self.assertEqual(cfg.max_gross_yen, cfg.capital_yen * cfg.margin_ratio)
        with self.assertRaises(ValueError):
            LiveConfig(margin_ratio=4.0).validate()   # 保証金率30% → 3.3x が上限

    def test_order_gating_locks(self):
        self.assertFalse(LiveConfig(env="test", dry_run=True).will_send_orders)
        self.assertTrue(LiveConfig(env="test", dry_run=False).paper_orders_enabled)
        self.assertFalse(LiveConfig(env="prod", dry_run=False, live_confirmed=False).orders_enabled)
        self.assertTrue(LiveConfig(env="prod", dry_run=False, live_confirmed=True).orders_enabled)
        # mock exercises the flow but is never "real"
        self.assertFalse(LiveConfig(env="mock").orders_enabled)
        self.assertTrue(LiveConfig(env="mock").will_send_orders)


class OrderValidationTest(unittest.TestCase):
    def test_nonzero_result_raises(self):
        c = KabuClient("api", "order", env="test")
        c._token = "t"
        c._request = lambda *a, **k: {"Result": 4, "Message": "建余力不足"}  # 200 w/ business reject
        with self.assertRaises(KabuAPIError):
            c.send_margin_open("13010", SIDE_BUY, 100)

    def test_success_passes(self):
        c = KabuClient("api", "order", env="test")
        c._token = "t"
        c._request = lambda *a, **k: {"Result": 0, "OrderId": "OK1"}
        self.assertEqual(c.send_margin_open("13010", SIDE_BUY, 100)["OrderId"], "OK1")


class ExitIdempotencyTest(unittest.TestCase):
    def test_exit_closes_leaves_minus_hold(self):
        c = MockKabuClient({})
        c._positions = [{"Symbol": "7203", "Side": SIDE_BUY, "LeavesQty": 500,
                         "HoldQty": 300, "ExecutionID": "X1", "ExecutionDay": 0}]
        res = exit_all(c, LiveConfig(env="mock"))
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["qty"], 200)            # 500 - 300, not 500 (no over-close)
        self.assertEqual(res[0]["close_side"], SIDE_SELL)  # opposite of long

    def test_exit_rerun_does_not_overclose(self):
        c = MockKabuClient({"7203": 2000})
        c._positions = [{"Symbol": "7203", "Side": SIDE_SELL, "LeavesQty": 300,
                         "HoldQty": 0, "ExecutionID": "X2", "ExecutionDay": 0}]
        cfg = LiveConfig(env="mock")
        exit_all(c, cfg)                                 # first flatten
        second = exit_all(c, cfg)                        # re-run: nothing left to close
        self.assertEqual(second, [])


class EntryTest(unittest.TestCase):
    def test_entry_skips_already_held(self):
        c = MockKabuClient({"7203": 2000})
        c._positions = [{"Symbol": "7203", "Side": SIDE_BUY, "LeavesQty": 100,
                         "HoldQty": 0, "ExecutionID": "X1", "ExecutionDay": 0}]
        plan = pd.DataFrame([{"symbol": "72030", "kabu_symbol": "7203", "name": "トヨタ",
                              "side": SIDE_BUY, "side_label": "LONG", "lots": 1, "qty": 100,
                              "est_price": 2000.0, "residual_gap": -0.02, "est_yen": 200000.0}])
        res = enter(c, LiveConfig(env="mock"), plan, force=True)
        self.assertEqual(res[0].get("skipped"), "already held/working")


class SignalTest(unittest.TestCase):
    def test_compute_today_signals_is_demeaned_and_scored(self):
        last = pd.DataFrame({
            "symbol": ["1", "2", "3", "4"], "sector": ["A", "A", "B", "B"],
            "close": [100.0, 200, 300, 400], "raw_close": [100.0, 200, 300, 400],
            "gap_vol60": [0.02] * 4, "ivol": [0.02] * 4, "shortable": [True] * 4,
        })
        opens = {"1": 98.0, "2": 206.0, "3": 297.0, "4": 412.0}  # gaps -2%,+3%,-1%,+3%
        scored = compute_today_signals(last, opens, "gap_reversal")
        self.assertAlmostEqual(scored["residual_gap"].mean(), 0.0, places=9)  # cross-sec demeaned
        self.assertIn("_s", scored.columns)
        # gap_reversal score = -residual_gap: the biggest down-gap (sym1, -2%) scores highest (long).
        self.assertEqual(scored.loc[scored["_s"].idxmax(), "symbol"], "1")


if __name__ == "__main__":
    unittest.main()


class ShortableCheckTest(unittest.TestCase):
    def test_verify_shortable_bans_and_backfills(self):
        from trading.jp_intraday.live.executor import verify_shortable
        c = MockKabuClient({}, short_banned={"72030"})
        cache = {}
        banned = verify_shortable(c, ["7203", "6758"], cache)
        self.assertEqual(banned, {"7203"})          # 不可銘柄を検出
        self.assertIn("6758", cache)                 # キャッシュ済み
        # フラグ不明（空dict）は不可扱いにしない
        cache2 = {"9999": {}}
        self.assertEqual(verify_shortable(c, ["9999"], cache2), set())
