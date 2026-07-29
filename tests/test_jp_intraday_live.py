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


class PreflightMarkerTest(unittest.TestCase):
    def test_mock_enter_does_not_block_same_day_rerun(self):
        # preflight(mock)が entry マーカーを書くと、同日の本番 entry が
        # RuntimeError で止まる回帰の防止: mock はマーカーを書かない・見ない。
        client = MockKabuClient({"1301": 1000.0})
        cfg = LiveConfig(env="mock")
        plan = pd.DataFrame([{"symbol": "13010", "kabu_symbol": "1301", "side": SIDE_BUY,
                              "qty": 100, "est_price": 1000.0}])
        enter(client, cfg, plan)
        client2 = MockKabuClient({"1301": 1000.0})
        enter(client2, cfg, plan)   # 2回目も RuntimeError にならないこと


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


class ShadowMonitorTest(unittest.TestCase):
    def test_tp_sl_trigger_once_and_fill_proxy(self):
        from trading.jp_intraday.live.shadow import ShadowMonitor
        m = ShadowMonitor(tp_pct=2.0, sl_pct=2.0)
        # ロング: +2%到達でTP発動（1回のみ）→次サンプルでfill_proxy
        ev = m.process_sample("09:10:00", "1301", "LONG", 1000.0, {"CurrentPrice": 1021.0})
        self.assertEqual([e["type"] for e in ev], ["trigger"])
        self.assertEqual(ev[0]["kind"], "TP")
        ev2 = m.process_sample("09:11:00", "1301", "LONG", 1000.0, {"CurrentPrice": 1018.0})
        kinds = [e["type"] for e in ev2]
        self.assertIn("fill_proxy", kinds)          # F1想定の約定proxy
        self.assertNotIn("trigger", kinds)          # 再発動しない
        self.assertAlmostEqual(ev2[0]["gap_bps"], (1018/1021 - 1) * 1e4, places=6)

    def test_short_sl_direction(self):
        from trading.jp_intraday.live.shadow import ShadowMonitor
        m = ShadowMonitor()
        # ショートの損失=価格上昇。+2%上昇でSL発動、TPは発動しない
        ev = m.process_sample("10:00:00", "9984", "SHORT", 5000.0, {"CurrentPrice": 5101.0})
        self.assertEqual(len(ev), 1)
        self.assertEqual(ev[0]["kind"], "SL")

    def test_run_shadow_loop_writes_summary(self):
        import datetime as dt
        from trading.jp_intraday.live.shadow import run_shadow
        client = MockKabuClient({"1301": 1000.0})
        cfg = LiveConfig(env="mock")
        # 建玉を1つ作る（mockの建玉APIを利用）
        client.send_margin_open("13010", SIDE_BUY, 100, front_order_type=13,
                                margin_type=3, account_type=4)
        client._prices["1301"] = 1030.0   # +3% -> TP発動相当
        times = [dt.datetime(2026, 7, 30, 9, 1), dt.datetime(2026, 7, 30, 9, 2),
                 dt.datetime(2026, 7, 30, 15, 33)]
        summary = run_shadow(client, cfg, until="15:32", interval_s=0,
                             sleep=lambda s: None, now_fn=lambda: times.pop(0))
        self.assertGreaterEqual(summary["samples"], 1)
        self.assertEqual(summary["n_tp"] + summary["n_sl"], len(summary["triggers"]))
