"""板の一括取得（並列＋45件ごと登録）の回帰テスト。ネットワーク不要。

守りたい性質:
  * 登録上限50を超えない（45件ごとに unregister→register し直す）
  * 1銘柄の失敗で全体が落ちない
  * boards() を持たないクライアント（mock）は従来の直列経路にフォールバックする
  * 並列でもレート制限を破らない（送信開始が MIN_INTERVAL 以上あくこと）
"""
import os
import threading
import time
import unittest
from unittest import mock

from trading.jp_intraday.live.executor import open_prices
from trading.jp_intraday.live.kabu_client import KabuClient


class FakeResponse:
    def __init__(self, status_code=200, body=None):
        self.status_code = status_code
        self._body = body if body is not None else {}
        self.text = "{}"

    @property
    def ok(self):
        return self.status_code < 400

    def json(self):
        return self._body


class FakeStation:
    """kabuステーションの板挙動を模したフェイク（登録上限50・未登録は遅い）。

    HTTP層（Session.request）で差し込むので、スロットル・リトライ・401再認証を含む
    KabuClient._request の実コードがそのまま動く。
    """

    def __init__(self, fail_symbols=(), latency=0.0):
        self.registered: set = set()
        self.fail = set(fail_symbols)
        self.latency = latency
        self.max_registered = 0
        self.calls: list = []
        self.starts: list = []
        self._lock = threading.Lock()

    def request(self, method, url, headers=None, params=None, json=None, timeout=None):
        path = url.split("/kabusapi")[1]
        with self._lock:
            self.calls.append((method, path))
            self.starts.append(time.monotonic())
        if path == "/unregister/all":
            self.registered.clear()
            return FakeResponse()
        if path == "/register":
            with self._lock:
                for s in json["Symbols"]:
                    self.registered.add(s["Symbol"])
                self.max_registered = max(self.max_registered, len(self.registered))
            return FakeResponse(body={"RegistList": sorted(self.registered)})
        sym = path.split("/board/")[1].split("@")[0]
        with self._lock:
            self.registered.add(sym)
            self.max_registered = max(self.max_registered, len(self.registered))
            over = len(self.registered) > 50
        if over:
            return FakeResponse(400, {"Code": 4002006})
        if sym in self.fail:
            return FakeResponse(400, {"Code": 4002099})
        if self.latency:
            time.sleep(self.latency)
        return FakeResponse(body={"Symbol": sym, "CalcPrice": 1000.0 + int(sym) % 100})


class FakeSession:
    def __init__(self, station):
        self.request = station.request


def client_with(station: FakeStation, min_interval: float = 0.0) -> KabuClient:
    c = KabuClient("pw", "opw", env="prod")
    c._token = "tok"
    c.MIN_INTERVAL = min_interval
    c._session = FakeSession(station)              # メインスレッド用
    c._session_factory = lambda: FakeSession(station)   # ワーカースレッド用
    return c


class BulkBoardTest(unittest.TestCase):
    def test_stays_under_registration_cap(self):
        st = FakeStation()
        got = client_with(st).boards([f"{1000 + i}0" for i in range(200)])
        self.assertEqual(len(got), 200)
        self.assertLessEqual(st.max_registered, 50, "登録数が上限50を超えている")
        self.assertEqual(st.calls.count(("PUT", "/unregister/all")), 5)  # 200/45 → 5チャンク

    def test_one_bad_symbol_does_not_kill_the_sweep(self):
        st = FakeStation(fail_symbols={"1010"})
        got = client_with(st).boards([f"{1000 + i}0" for i in range(20)])
        self.assertEqual(len(got), 19)
        self.assertNotIn("10100", got)

    def test_keys_are_the_input_symbols(self):
        # 内部では4桁に変換して問い合わせるが、返すキーは入力(J-Quants5桁)のまま
        st = FakeStation()
        got = client_with(st).boards(["13010", "13020"])
        self.assertEqual(sorted(got), ["13010", "13020"])

    def test_rate_limit_is_respected_under_parallelism(self):
        # 並列でも送信開始は直列化される。Windows のタイマ分解能(~15.6ms)があるので
        # 個々の間隔ではなく「平均レートが上限を超えないこと」で判定する。
        interval, n = 0.05, 30
        st = FakeStation(latency=0.01)
        t0 = time.monotonic()
        client_with(st, min_interval=interval).boards([f"{1000 + i}0" for i in range(n)])
        elapsed = time.monotonic() - t0
        sends = len(st.starts)
        self.assertGreaterEqual(elapsed, sends * interval * 0.8,
                                f"平均レートが上限超過 ({sends}送信 / {elapsed:.2f}秒)")

    def test_open_prices_uses_bulk_path_only_when_opted_in(self):
        # 既定は直列（本番実測でカバレッジ100%）。一括経路は KABU_BULK_BOARDS=1 のときだけ。
        st = FakeStation()
        with mock.patch.dict(os.environ, {"KABU_BULK_BOARDS": "1"}):
            opens = open_prices(client_with(st), ["13010", "13020"], progress=False)
        self.assertEqual(len(opens), 2)
        self.assertIn(("PUT", "/register"), st.calls)

        st2 = FakeStation()
        with mock.patch.dict(os.environ, {"KABU_BULK_BOARDS": "0"}):
            opens = open_prices(client_with(st2), ["13010", "13020"], progress=False)
        self.assertEqual(len(opens), 2)
        self.assertNotIn(("PUT", "/register"), st2.calls)


class SerialFallbackTest(unittest.TestCase):
    class OnlySingleBoard:
        """boards() を持たない旧来クライアント（mock 相当）。"""

        def __init__(self):
            self.n = 0

        def board(self, symbol, exchange=1):
            self.n += 1
            if symbol == "9999":
                raise RuntimeError("no such symbol")
            return {"CalcPrice": 500.0}

    def test_falls_back_to_serial_and_skips_failures(self):
        c = self.OnlySingleBoard()
        opens = open_prices(c, ["13010", "9999", "13020"], progress=False)
        self.assertEqual(opens, {"13010": 500.0, "13020": 500.0})
        self.assertEqual(c.n, 3)


if __name__ == "__main__":
    unittest.main()
