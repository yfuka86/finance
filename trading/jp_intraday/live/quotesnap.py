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

# 前日確定データだけで候補を絞るための指標（気配は一切使わない）。
# 自身の残差ギャップの60日ボラ＝「今日ギャップが大きく出そうな銘柄」。
SCREEN_COL = "gap_vol60"
DEFAULT_SHORTLIST = 50
FETCH_RATE_PER_SEC = 9.5      # kabu /board の実測スループット（1銘柄1リクエスト）


def shortlist_symbols(panel, k: int | None = DEFAULT_SHORTLIST,
                      screen: str = SCREEN_COL) -> list[str]:
    """直近営業日のパネルから候補を k 銘柄へ絞る（k=None なら全銘柄）.

    ★これは「気配を速く取るため」の絞り込みであって、流動性フロアで削るのとは別物。
    467銘柄だと kabu では49-66秒かかり、最初と最後で気配が1分ずれる。50銘柄なら
    約5秒で撮れる。代償は Sharpe 3.43→1.69（OOS24+・シミュレーション）。
    """
    last = panel[panel["date"].eq(panel["date"].max())]
    if k is None:
        return list(last["symbol"])
    ranked = last.dropna(subset=[screen]).nlargest(k, screen)
    return list(ranked["symbol"])


def snapshot_lead_seconds(n_symbols: int) -> float:
    """取得開始を目標時刻の何秒前にするか（銘柄数に比例させる）.

    ★固定70秒のままだと、50銘柄でも「70秒前の気配」を撮ることになり
    絞り込んだ意味が消える。取得所要 + 余裕5秒。
    """
    return max(10.0, n_symbols / FETCH_RATE_PER_SEC + 5.0)


def run_quotesnap(client: KabuClientProtocol, cfg: LiveConfig,
                  symbols: list[str], sleep=time.sleep, now_fn=None) -> dict:
    """SNAP_TIMES それぞれの直前に全銘柄の板を取得して記録する."""
    now_fn = now_fn or (lambda: dt.datetime.now())
    day = now_fn().strftime("%Y-%m-%d")
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = _OUT_DIR / f"quotesnap_{day}.jsonl"
    lead_s = snapshot_lead_seconds(len(symbols))
    counts = {}
    with path.open("a", encoding="utf-8") as f:
        for snap in SNAP_TIMES:
            # スナップ時刻まで待機（取得所要は銘柄数に比例するのでリードも比例させる）
            while True:
                now = now_fn()
                hm = now.strftime("%H:%M:%S")
                target = f"{snap}:00"
                lead = (dt.datetime.strptime(target, "%H:%M:%S")
                        - dt.datetime.strptime(hm, "%H:%M:%S")).total_seconds()
                if lead <= lead_s:
                    break
                sleep(min(lead - lead_s, 10))
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
    # 取得の実所要（=同時性）は分析側の必須メタ。1朝ぶんの最初/最後の時刻差で測る。
    return {"day": day, "counts": counts, "log": str(path),
            "n_symbols": len(symbols), "lead_seconds": round(lead_s, 1)}
