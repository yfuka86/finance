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
SNAP_SECONDS_PER_SYMBOL = 1.1   # 実測(2026-07-30): 未登録銘柄の板は中央値~900ms・1割は5秒
SNAP_TOLERANCE_SEC = 45         # 目標時刻をこれ以上過ぎたら打ち切る（時点ラベルを守る）


def sample_symbols(symbols: list[str], limit: int) -> list[str]:
    """等間隔サンプリング（先頭に偏らせない・毎回同じ銘柄になるので時点比較が成立する）。

    全銘柄を1時点あたり1秒/銘柄で舐めると626銘柄=約11分かかり、08:50/08:55/08:59 の
    3時点が互いに重なって「その時刻の気配」でなくなる（寄付き後にずれ込む）。
    σの推定には数十銘柄あれば十分なので、時間予算に収まる件数に間引く。
    """
    if not limit or limit <= 0 or len(symbols) <= limit:
        return list(symbols)
    step = len(symbols) / limit
    return [symbols[min(int(i * step), len(symbols) - 1)] for i in range(limit)]


def run_quotesnap(client: KabuClientProtocol, cfg: LiveConfig,
                  symbols: list[str], sleep=time.sleep, now_fn=None,
                  limit: int = 0) -> dict:
    """SNAP_TIMES それぞれの直前に板を取得して記録する（limit>0 なら等間隔サンプル）."""
    symbols = sample_symbols(list(symbols), limit)
    lead_sec = max(70, int(len(symbols) * SNAP_SECONDS_PER_SYMBOL))
    now_fn = now_fn or (lambda: dt.datetime.now())
    day = now_fn().strftime("%Y-%m-%d")
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = _OUT_DIR / f"quotesnap_{day}.jsonl"
    counts = {}
    with path.open("a", encoding="utf-8") as f:
        for snap in SNAP_TIMES:
            # スナップ時刻まで待機（取得所要を見込んで 目標時刻-lead_sec に開始）
            while True:
                now = now_fn()
                hm = now.strftime("%H:%M:%S")
                target = f"{snap}:00"
                lead = (dt.datetime.strptime(target, "%H:%M:%S")
                        - dt.datetime.strptime(hm, "%H:%M:%S")).total_seconds()
                if lead <= lead_sec:
                    break
                sleep(min(lead - lead_sec, 10))
            # 板の所要は日によってブレる（実測1.0〜5.0秒/銘柄）。遅い日に取り切ろうとすると
            # 08:50のスナップが08:55側へ食い込み「その時刻の気配」でなくなる。
            # 時刻で打ち切り、件数が減ることを受け入れる（ラベルの正しさを優先）。
            deadline = (dt.datetime.strptime(f"{snap}:00", "%H:%M:%S")
                        + dt.timedelta(seconds=SNAP_TOLERANCE_SEC)).time()
            n = 0
            for s in symbols:
                if now_fn().time() > deadline:
                    print(f"  quotesnap {snap}: 時間切れで打ち切り（{n}/{len(symbols)}銘柄）",
                          flush=True)
                    break
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
