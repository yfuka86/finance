"""Offline tests for Slack notifications (no network)."""
import io
import json
import unittest
from unittest import mock

from trading.jp_intraday.live import notifier, reporter
from trading.jp_intraday.live.config import LiveConfig
from trading.jp_intraday.live.notifier import SlackConfig, format_event, post

CFG = SlackConfig(token="xoxb-test", channel="C0TEST", enabled=True)
STAMP = "2026-07-30T08:44:12"


def fake_urlopen(payload: dict):
    """Patch target for urllib.request.urlopen returning a Slack-shaped response."""
    def _open(req, timeout=None):
        _open.body = json.loads(req.data.decode("utf-8"))
        return io.BytesIO(json.dumps(payload).encode())
    _open.body = None
    return _open


class PostTest(unittest.TestCase):
    def test_not_configured_is_a_no_op(self):
        res = post("hello", SlackConfig(token="", channel=""))
        self.assertFalse(res["sent"])

    def test_disabled_by_flag(self):
        res = post("hello", SlackConfig(token="t", channel="c", enabled=False))
        self.assertFalse(res["sent"])

    def test_ok_response(self):
        opener = fake_urlopen({"ok": True, "ts": "1.2"})
        with mock.patch("urllib.request.urlopen", opener):
            res = post("hello", CFG)
        self.assertTrue(res["sent"])
        self.assertEqual(opener.body["channel"], "C0TEST")
        self.assertEqual(opener.body["text"], "hello")

    def test_slack_level_error_is_not_success(self):
        # Slack は失敗も HTTP 200 + ok:false で返す（not_in_channel など）
        with mock.patch("urllib.request.urlopen", fake_urlopen({"ok": False, "error": "not_in_channel"})):
            res = post("hello", CFG)
        self.assertFalse(res["sent"])
        self.assertEqual(res["error"], "not_in_channel")

    def test_network_failure_never_raises(self):
        # 発注フローが Slack 障害で止まらないことが最重要
        with mock.patch("urllib.request.urlopen", side_effect=OSError("boom")):
            res = post("hello", CFG)
        self.assertFalse(res["sent"])
        self.assertIn("boom", res["error"])


class FormatTest(unittest.TestCase):
    def test_plan(self):
        text = format_event("plan", {"meta": {"data_date": "2026-07-29", "coverage": 0.98,
                                              "n_long": 8, "n_short": 8, "gross_yen": 39_800_000,
                                              "shorts_banned": ["7203", "6758"]}},
                            "prod", STAMP)
        self.assertIn("L 8 / S 8", text)
        self.assertIn("¥39.8M", text)
        self.assertIn("98%", text)
        self.assertIn("売り禁止 2 銘柄", text)
        self.assertIn("🔴本番", text)

    def test_entry_counts_and_failures(self):
        orders = [{"symbol": "1301"}, {"symbol": "1332", "error": "order rejected: 100368"},
                  {"symbol": "1333", "skipped": "already held/working"}]
        text = format_event("entry", {"orders": orders, "meta": {"gross_yen": 39_800_000}},
                            "prod", STAMP)
        self.assertIn("送信 3 件（成功 1 / 失敗 1 / スキップ 1）", text)
        self.assertIn("1332", text)
        self.assertIn("100368", text)

    def test_exit_without_failures_has_no_alert(self):
        text = format_event("exit", {"orders": [{"symbol": "1301"}]}, "prod", STAMP)
        self.assertIn("送信 1 件（成功 1 / 失敗 0 / スキップ 0）", text)
        self.assertNotIn("⚠️", text)

    def test_state_flat_is_quiet(self):
        text = format_event("state", {"positions": [], "margin": {"MarginAccountWallet": 10_700_000}},
                            "prod", STAMP)
        self.assertIn("建玉 0 件", text)
        self.assertNotIn("🚨", text)

    def test_state_alerts_when_positions_remain(self):
        # 場中フラット戦略なので引け後の建玉は異常（返済漏れ）。ここは絶対に鳴らす
        pos = [{"Symbol": "1301", "SymbolName": "極洋", "Side": "2", "LeavesQty": 100,
                "ProfitLoss": -12_000}]
        text = format_event("state", {"positions": pos, "margin": {"MarginAccountWallet": 1e7}},
                            "prod", STAMP)
        self.assertIn("🚨", text)
        self.assertIn("引け後に建玉が残っています", text)
        self.assertIn("−¥12,000", text)

    def test_unknown_event_returns_none(self):
        self.assertIsNone(format_event("train", {}, "prod", STAMP))


class NotifyEventTest(unittest.TestCase):
    def test_mock_env_is_not_notified(self):
        # preflight(mock) は毎朝走るので通知しない
        with mock.patch.object(notifier, "post") as p:
            res = notifier.notify_event("plan", {"meta": {}}, "mock", STAMP)
        p.assert_not_called()
        self.assertFalse(res["sent"])

    def test_prod_event_is_notified(self):
        with mock.patch.object(notifier, "post", return_value={"sent": True}) as p:
            notifier.notify_event("exit", {"orders": []}, "prod", STAMP)
        p.assert_called_once()

    def test_error_alert_wraps_detail_in_code_block(self):
        with mock.patch.object(notifier, "post", return_value={"sent": True}) as p:
            notifier.notify_error("entry 失敗 (exit=1)", "Traceback...", env="prod")
        text = p.call_args[0][0]
        self.assertIn("🚨", text)
        self.assertIn("```", text)
        self.assertIn("Traceback...", text)


class ReporterHookTest(unittest.TestCase):
    def test_report_returns_slack_result_and_survives_slack_failure(self):
        cfg = LiveConfig(env="prod", report_url="")   # ダッシュボード送信なし = ローカル監査のみ
        with mock.patch.object(notifier, "post", side_effect=RuntimeError("should be caught")):
            with mock.patch.object(reporter, "_audit"):
                with mock.patch.object(notifier, "notify_event", return_value={"sent": False}):
                    res = reporter.report(cfg, "exit", {"orders": []}, STAMP)
        self.assertIn("slack", res)


if __name__ == "__main__":
    unittest.main()
