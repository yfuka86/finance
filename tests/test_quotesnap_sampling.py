"""quotesnap の件数制限とスナップ時刻の整合（ネットワーク不要）。

背景: 板は実測 ~1秒/銘柄。全626銘柄だと1時点=約11分かかり 08:50/08:55/08:59 の
3時点が互いに重なって「その時刻の気配」でなくなる（寄付き後にずれ込む）。
"""
import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from trading.jp_intraday.live import quotesnap
from trading.jp_intraday.live.config import LiveConfig
from trading.jp_intraday.live.quotesnap import sample_symbols


class SampleTest(unittest.TestCase):
    def test_no_limit_keeps_everything(self):
        syms = [f"{i}" for i in range(100)]
        self.assertEqual(sample_symbols(syms, 0), syms)
        self.assertEqual(sample_symbols(syms, 200), syms)

    def test_evenly_spaced_and_stable(self):
        syms = [f"{i}" for i in range(600)]
        got = sample_symbols(syms, 60)
        self.assertEqual(len(got), 60)
        self.assertEqual(got, sample_symbols(syms, 60))     # 時点間で同じ銘柄になる
        self.assertEqual(got[0], "0")
        self.assertGreater(int(got[-1]), 500, "末尾側も拾えていない=先頭偏り")
        self.assertEqual(len(set(got)), 60, "重複がある")


class FakeClient:
    def __init__(self):
        self.calls = 0

    def board(self, symbol, exchange=1):
        self.calls += 1
        return {"CalcPrice": 1000.0, "CurrentPrice": 1001.0, "BidPrice": 999.0, "AskPrice": 1002.0}


class SnapshotTimingTest(unittest.TestCase):
    def test_each_snapshot_fits_before_its_target_time(self):
        # 60銘柄なら1時点~66秒。08:50/08:55/08:59 に間に合う想定で開始する
        clock = {"t": dt.datetime(2026, 7, 31, 8, 47, 0)}
        slept: list = []

        def now_fn():
            return clock["t"]

        def sleep(sec):
            slept.append(sec)
            clock["t"] += dt.timedelta(seconds=sec)

        client = FakeClient()
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(quotesnap, "_OUT_DIR", Path(tmp)):
                out = quotesnap.run_quotesnap(client, LiveConfig(env="prod"),
                                              [f"{i}" for i in range(600)],
                                              sleep=sleep, now_fn=now_fn, limit=60)
                rows = [json.loads(l) for l in
                        (Path(tmp) / f"quotesnap_{out['day']}.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(out["counts"], {"08:50": 60, "08:55": 60, "08:59": 60})
        self.assertEqual(client.calls, 180)
        # 各スナップの取得開始は目標時刻の直前（66秒前後）に来ていること
        for snap in ("08:50", "08:55", "08:59"):
            first = min(r["time"] for r in rows if r["snap"] == snap)
            target = dt.datetime.strptime(snap + ":00", "%H:%M:%S")
            start = dt.datetime.strptime(first, "%H:%M:%S")
            lead = (target - start).total_seconds()
            self.assertGreaterEqual(lead, 0, f"{snap} の取得が目標時刻を過ぎてから始まっている")
            self.assertLessEqual(lead, 90, f"{snap} の取得開始が早すぎる（気配が古くなる）")


class SlowMorningTest(unittest.TestCase):
    """遅い日でも各スナップが自分の時刻枠を守る（件数より時点ラベルの正しさを優先）。"""

    class SlowClient:
        def __init__(self, clock, sec_per_symbol):
            self.clock = clock
            self.sec = sec_per_symbol

        def board(self, symbol, exchange=1):
            self.clock["t"] += dt.timedelta(seconds=self.sec)
            return {"CalcPrice": 1000.0}

    def test_snapshot_is_cut_off_instead_of_bleeding_into_the_next(self):
        clock = {"t": dt.datetime(2026, 7, 31, 8, 47, 0)}
        client = self.SlowClient(clock, sec_per_symbol=5)   # 1銘柄5秒の最悪日
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(quotesnap, "_OUT_DIR", Path(tmp)):
                out = quotesnap.run_quotesnap(
                    client, LiveConfig(env="prod"), [f"{i}" for i in range(600)],
                    sleep=lambda s: clock.__setitem__("t", clock["t"] + dt.timedelta(seconds=s)),
                    now_fn=lambda: clock["t"], limit=60)
                rows = [json.loads(l) for l in
                        (Path(tmp) / f"quotesnap_{out['day']}.jsonl").read_text(encoding="utf-8").splitlines()]
        # 3時点とも記録は残る（件数は減る）
        self.assertEqual(sorted(out["counts"]), ["08:50", "08:55", "08:59"])
        self.assertTrue(all(v > 0 for v in out["counts"].values()), out["counts"])
        self.assertTrue(any(v < 60 for v in out["counts"].values()), "打ち切りが効いていない")
        # 各行の時刻が自分のスナップ時刻の近傍に収まっている
        for snap, tol in (("08:50", 45), ("08:55", 45), ("08:59", 45)):
            target = dt.datetime.strptime(snap + ":00", "%H:%M:%S")
            for r in (x for x in rows if x["snap"] == snap):
                delta = (dt.datetime.strptime(r["time"], "%H:%M:%S") - target).total_seconds()
                self.assertLessEqual(delta, tol + 10, f"{snap} の記録が {r['time']} まで流れている")


if __name__ == "__main__":
    unittest.main()
