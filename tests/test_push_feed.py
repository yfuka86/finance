"""PUSH配信フィードの回帰テスト（ネットワーク・kabuステーション不要）。

守りたい性質:
  * 登録は50銘柄まで（超えたら即エラー。黙って切り捨てない）
  * 受信したメッセージが銘柄ごとの最新板として保持される
  * snapshot() が各銘柄の取得時刻を持ち、スメア（最古と最新の差）を測れる
  * RESTのシードで初期値が埋まる（PUSHは更新があるまで無音のため）
"""
import json
import time
import unittest
from unittest import mock

from trading.jp_intraday.live.push_feed import MAX_REGISTERED, PushBoardFeed


class FakeClient:
    base = "http://localhost:18080/kabusapi"

    def __init__(self):
        self.registered = None
        self.board_calls = 0
        self.calls: list = []

    def _request(self, method, path, json=None, **kw):
        if path == "/register":
            self.calls.append("register")
            self.registered = [s["Symbol"] for s in json["Symbols"]]
            return {"RegistList": [{"Symbol": s} for s in self.registered]}
        raise AssertionError(f"未知のリクエスト {method} {path}")

    def unregister_all(self):
        self.calls.append("unregister_all")
        self.registered = None
        return {}

    def board(self, symbol, exchange=1):
        # 実クライアントの board() は45件ごとに unregister_all を呼ぶ。
        # シードでこれを使うと自分の登録を消してしまうので、使われたら失敗させる。
        raise AssertionError("シードで board() を使ってはいけない（登録が消える）")

    def _board_or_none(self, symbol, exchange=1):
        self.board_calls += 1
        return {"Symbol": symbol[:-1] if len(symbol) == 5 else symbol, "CurrentPrice": 100.0}


class FakeWSApp:
    """websocket.WebSocketApp の差し替え。run_forever は何もせず待つだけ。"""

    instances: list = []

    def __init__(self, url, on_message=None, on_open=None, on_error=None):
        self.url = url
        self.on_message = on_message
        self.on_open = on_open
        self.closed = False
        FakeWSApp.instances.append(self)

    def run_forever(self):
        if self.on_open:
            self.on_open(self)
        while not self.closed:
            time.sleep(0.01)

    def close(self):
        self.closed = True

    def push(self, board: dict):
        self.on_message(self, json.dumps(board))


class PushFeedTest(unittest.TestCase):
    def setUp(self):
        FakeWSApp.instances = []
        self.patch = mock.patch("trading.jp_intraday.live.push_feed.websocket.WebSocketApp",
                                FakeWSApp)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_rejects_more_than_50_symbols(self):
        with self.assertRaises(ValueError):
            PushBoardFeed(FakeClient(), [f"{1000+i}0" for i in range(MAX_REGISTERED + 1)])

    def test_clears_registry_before_registering(self):
        # スイープ中の自動登録が枠(50)を埋めているので、先に全消去しないと
        # 追加登録が黙って弾かれる（2026-08-03 の実障害）
        c = FakeClient()
        with PushBoardFeed(c, ["13010"], seed_via_rest=False, log=lambda *_: None):
            pass
        self.assertEqual(c.calls[:2], ["unregister_all", "register"],
                         f"登録前に unregister_all していない: {c.calls}")

    def test_registers_in_kabu_4digit_form_and_seeds_via_rest(self):
        c = FakeClient()
        with PushBoardFeed(c, ["13010", "13020"], log=lambda *_: None) as feed:
            self.assertEqual(c.registered, ["1301", "1302"])
            self.assertEqual(c.board_calls, 2, "RESTシードが動いていない")
            self.assertNotIn("unregister_all", c.calls[2:],
                             "シード中に登録が消されている（PUSHが止まる）")
            snap = feed.snapshot()
        self.assertEqual(sorted(snap), ["13010", "13020"])   # キーは入力表記のまま

    def test_push_message_updates_latest_board(self):
        c = FakeClient()
        with PushBoardFeed(c, ["13010"], seed_via_rest=False, log=lambda *_: None) as feed:
            ws = FakeWSApp.instances[-1]
            ws.push({"Symbol": "1301", "CurrentPrice": 555.0, "CalcPrice": 556.0})
            for _ in range(50):
                if feed.snapshot():
                    break
                time.sleep(0.01)
            snap = feed.snapshot()
        self.assertEqual(snap["13010"]["board"]["CurrentPrice"], 555.0)
        self.assertLess(snap["13010"]["age_s"], 5)

    def test_unknown_symbol_is_ignored(self):
        c = FakeClient()
        with PushBoardFeed(c, ["13010"], seed_via_rest=False, log=lambda *_: None) as feed:
            FakeWSApp.instances[-1].push({"Symbol": "9999", "CurrentPrice": 1.0})
            time.sleep(0.05)
            self.assertEqual(feed.snapshot(), {})

    def test_smear_is_measurable(self):
        c = FakeClient()
        with PushBoardFeed(c, ["13010", "13020"], seed_via_rest=False, log=lambda *_: None) as feed:
            ws = FakeWSApp.instances[-1]
            ws.push({"Symbol": "1301", "CurrentPrice": 1.0})
            time.sleep(0.12)
            ws.push({"Symbol": "1302", "CurrentPrice": 2.0})
            time.sleep(0.05)
            smear = feed.smear_seconds()
        self.assertGreater(smear, 0.05)
        self.assertLess(smear, 2.0)


if __name__ == "__main__":
    unittest.main()
