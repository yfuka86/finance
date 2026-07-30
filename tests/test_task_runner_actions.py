"""run_live のアクションが、Windows タスクランナーからすべて起動できることを保証する。

2026-07-31 の実障害: quotesnap タスクを登録したが run_live_task.ps1 の ValidateSet に
"quotesnap" を足し忘れ、パラメータ検証で即失敗（ログすら残らず result=1）。
最重要の検証タスクが1日不発になった。モジュール単体とランナー(-Action state)は
検証していたが「そのアクションをランナー経由で起動する」経路が抜けていた。
"""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_live_task.ps1"
RUN_LIVE = ROOT / "trading" / "jp_intraday" / "live" / "run_live.py"


def runner_actions() -> set:
    text = RUNNER.read_text(encoding="utf-8-sig")
    m = re.search(r"\[ValidateSet\((.*?)\)\]", text, re.S)
    assert m, "run_live_task.ps1 に ValidateSet が見つからない"
    return set(re.findall(r'"([^"]+)"', m.group(1)))


def run_live_actions() -> set:
    text = RUN_LIVE.read_text(encoding="utf-8")
    m = re.search(r'add_argument\("action",\s*choices=\[(.*?)\]', text, re.S)
    assert m, "run_live.py に action の choices が見つからない"
    return set(re.findall(r'"([^"]+)"', m.group(1)))


class RunnerCoverageTest(unittest.TestCase):
    def test_every_action_can_be_launched_from_the_task_runner(self):
        missing = run_live_actions() - runner_actions()
        self.assertEqual(missing, set(),
                         f"ランナーの ValidateSet に無いアクション: {sorted(missing)} "
                         "（タスク登録しても起動時に検証エラーで即死する）")

    def test_runner_does_not_advertise_unknown_actions(self):
        # collect/probe はランナー専用（python側のactionではない）ので除外
        runner_only = {"collect", "probe"}
        unknown = runner_actions() - run_live_actions() - runner_only
        self.assertEqual(unknown, set(), f"run_live に無いアクション: {sorted(unknown)}")


if __name__ == "__main__":
    unittest.main()
