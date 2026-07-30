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


class ShortGuardTest(unittest.TestCase):
    def _frame(self):
        import numpy as np
        n = 40
        return pd.DataFrame({
            "date": pd.Timestamp("2026-07-30"), "symbol": [f"{1000+i}0" for i in range(n)],
            "open": 1000.0, "raw_open": 1000.0, "prev_close": 1000.0,
            "_s": np.linspace(-1, 1, n), "intraday_ret": 0.0,
            "shortable": True, "prev_value": 5e9,
            "short_restricted": [i < 5 for i in range(n)],   # スコア最低=ショート候補側に規制5銘柄
            "mktcap_yen": [5e9 if i < 10 else 5e10 for i in range(n)],
        })

    def test_unit_lot_excludes_restricted_shorts(self):
        from trading.jp_intraday.strategies import unit_lot_backtest
        f = self._frame()
        _, blot = unit_lot_backtest(f, capital_yen=2e7, names_per_side=8)
        shorts = set(blot[blot["side_label"] == "SHORT"]["symbol"])
        restricted = set(f[f["short_restricted"]]["symbol"])
        self.assertFalse(shorts & restricted)    # 規制銘柄はショートに入らない

    def test_unit_lot_short_mktcap_floor(self):
        from trading.jp_intraday.strategies import unit_lot_backtest
        f = self._frame()
        f["short_restricted"] = False
        _, blot = unit_lot_backtest(f, capital_yen=2e7, names_per_side=8,
                                    short_min_mktcap_yen=1e10)
        shorts = blot[blot["side_label"] == "SHORT"]["symbol"]
        small = set(f[f["mktcap_yen"] < 1e10]["symbol"])
        self.assertFalse(set(shorts) & small)    # 時価総額フロア未満はショートに入らない

    def test_verify_short_regulations_defensive_parse(self):
        from trading.jp_intraday.live.executor import verify_short_regulations
        class RegClient:
            def regulations(self, symbol, exchange=1):
                if symbol == "7203":
                    return {"RegulationsInfo": [{"Side": "1", "Product": "2",
                                                 "Reason": "新規売停止"}]}
                if symbol == "9984":
                    return {"RegulationsInfo": [{"Side": "2", "Product": "2"}]}  # 買い規制のみ
                raise RuntimeError("api down")   # 不明=可
        cache = {}
        banned = verify_short_regulations(RegClient(), ["7203", "9984", "6758"], cache)
        self.assertEqual(banned, {"7203"})
        self.assertIn("reg:6758", cache)         # エラーでもキャッシュされ再照会しない


class EffectiveCostTest(unittest.TestCase):
    def test_leg_slippage_sign_convention(self):
        from trading.jp_intraday.live.costs import leg_slippage_bps
        # entry・買い: 寄値1000に対し1001で約定 = 高く買った = +10bpsのコスト
        self.assertAlmostEqual(leg_slippage_bps(1001, 1000, SIDE_BUY, "entry"), 10.0, places=6)
        # entry・売建: 999で約定 = 安く売った = +10bpsのコスト
        self.assertAlmostEqual(leg_slippage_bps(999, 1000, SIDE_SELL, "entry"), 10.0, places=6)
        # exit・売却(買い建玉の決済): 引値1000に対し999 = 安く売った = +10bpsのコスト
        self.assertAlmostEqual(leg_slippage_bps(999, 1000, SIDE_BUY, "exit"), 10.0, places=6)
        # exit・買戻し: 1001 = 高く買い戻した = +10bpsのコスト
        self.assertAlmostEqual(leg_slippage_bps(1001, 1000, SIDE_SELL, "exit"), 10.0, places=6)
        # 板寄せどおり約定すればゼロ
        self.assertAlmostEqual(leg_slippage_bps(1000, 1000, SIDE_BUY, "entry"), 0.0, places=9)

    def test_fills_parse_details_and_fallback(self):
        from trading.jp_intraday.live.costs import _fills_from_orders
        orders = [
            {"Symbol": "1301", "Side": SIDE_BUY, "FrontOrderType": "13",
             "Details": [{"RecType": "1", "Price": 0, "Qty": 100},
                         {"RecType": "8", "Price": 1000.0, "Qty": 60},
                         {"RecType": "8", "Price": 1010.0, "Qty": 40}]},
            {"Symbol": "7203", "Side": SIDE_SELL, "FrontOrderType": "16",
             "Price": 2000.0, "CumQty": 100},          # Details無し→サマリにフォールバック
            {"Symbol": "9984", "Side": SIDE_BUY, "FrontOrderType": "13"},  # 情報なし→除外
        ]
        f = _fills_from_orders(orders)
        self.assertEqual(len(f), 2)
        self.assertAlmostEqual(f[0]["fill_px"], (1000 * 60 + 1010 * 40) / 100)  # 数量加重
        self.assertEqual(f[1]["fill_px"], 2000.0)


class OrderRateLimitTest(unittest.TestCase):
    def test_order_interval_matches_current_api_limit(self):
        # kabu Ver5.44.0.0(2026-07-10)で発注は10件/秒に緩和。余裕を見て8件/秒で運用する。
        # 制限が戻された場合はこのテストが失敗して気付けるようにしておく。
        from trading.jp_intraday.live.executor import ORDER_INTERVAL_S
        self.assertGreaterEqual(ORDER_INTERVAL_S, 0.1)   # 10件/秒を超えない
        self.assertLessEqual(ORDER_INTERVAL_S, 0.25)     # 旧制限より速い

    def test_mock_does_not_throttle(self):
        # mock(preflight)はスロットルしない＝テストが遅くならないこと
        import time
        client = MockKabuClient({"1301": 1000.0})
        plan = pd.DataFrame([{"symbol": "13010", "kabu_symbol": "1301", "side": SIDE_BUY,
                              "qty": 100, "est_price": 1000.0}] * 20)
        t0 = time.monotonic()
        enter(client, LiveConfig(env="mock"), plan)
        self.assertLess(time.monotonic() - t0, 1.0)
