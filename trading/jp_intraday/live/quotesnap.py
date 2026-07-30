"""寄前気配スナップショット記録（戦略成立性の検証・発注なし）.

「08:50の気配は寄値と全く違うのでは」という仮説を実測で検証する:
全ユニバースの板寄せ気配(CalcPrice)を 08:50 / 08:55 / 08:59 の3時点で記録し、
翌日 J-Quants の実寄値と突き合わせて
  - 時点別の乖離分布（収束カーブ）
  - ギャップ大の銘柄（=戦略が選ぶテール）での乖離
  - 系統的バイアス（気配ギャップは縮む方向に収束するか）
を測る。**発注は一切しない**（read-only）。1朝で 銘柄数×3時点 のペアが取れる。

実行（Windows・タスクスケジューラ 08:47 起動）:
  python -m trading.jp_intraday.live.run_live quotesnap
出力: data/live_reports/quotesnap_YYYY-MM-DD.jsonl
分析: PYTHONPATH=. python scripts/analyze_quotesnap.py（実寄値データ到着後）
"""
from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

from .config import LiveConfig
from .kabu_client import KabuClientProtocol, to_kabu_symbol

SNAP_TIMES = ("08:50", "08:55", "08:59")   # 3時点の収束カーブを取る
_OUT_DIR = Path("data/live_reports")


def run_quotesnap(client: KabuClientProtocol, cfg: LiveConfig,
                  symbols: list[str], sleep=time.sleep, now_fn=None) -> dict:
    """SNAP_TIMES それぞれの直前に全銘柄の板を取得して記録する."""
    now_fn = now_fn or (lambda: dt.datetime.now())
    day = now_fn().strftime("%Y-%m-%d")
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = _OUT_DIR / f"quotesnap_{day}.jsonl"
    counts = {}
    with path.open("a", encoding="utf-8") as f:
        for snap in SNAP_TIMES:
            # スナップ時刻まで待機（板取得~66秒を見込み、取得開始=目標時刻-70秒）
            while True:
                now = now_fn()
                hm = now.strftime("%H:%M:%S")
                target = f"{snap}:00"
                lead = (dt.datetime.strptime(target, "%H:%M:%S")
                        - dt.datetime.strptime(hm, "%H:%M:%S")).total_seconds()
                if lead <= 70:
                    break
                sleep(min(lead - 70, 10))
            n = 0
            for s in symbols:
                ksym = to_kabu_symbol(s)
                try:
                    b = client.board(ksym)
                except Exception:  # noqa: BLE001
                    continue
                row = {"snap": snap, "time": now_fn().strftime("%H:%M:%S"),
                       "symbol": ksym,
                       "calc": b.get("CalcPrice"), "current": b.get("CurrentPrice"),
                       "bid": b.get("BidPrice"), "ask": b.get("AskPrice")}
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                n += 1
            counts[snap] = n
            f.flush()
    return {"day": day, "counts": counts, "log": str(path)}
