"""シャドーTP/SL監視（発注なし・実測専用）.

日中利確/損切りオーバーレイ（TP+2%/SL-2%・検証済みだが導入保留、AGENTS.md参照）の
導入判断材料を集める: 建玉の仮想発動を1分間隔で検知し、
  - 検知時点の気配スナップショット（実勢で約定できたであろう価格）
  - 次サンプル時点の価格（リサーチのF1「次バーopen約定」仮定に対応）
を記録する。実測スリッページ = F1想定価格 vs 実勢気配 の差が判断の本体。

**発注・取消は一切行わない**（read-only API のみ）。閾値は事前登録値（TP+2%/SL-2%）で
固定 — シャドー期間中に動かすと実測の意味が消えるため変更禁止。

実行（Windows・タスクスケジューラ 09:01 起動を想定）:
  python -m trading.jp_intraday.live.run_live shadow          # 15:32 まで自動ループ
出力: data/live_reports/shadow_YYYY-MM-DD.jsonl（全サンプル+イベント）
      終了時にサマリを Web 管理画面へ report(event="shadow_summary")。
"""
from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

from .config import LiveConfig
from .kabu_client import KabuClientProtocol, to_kabu_symbol

# 事前登録済みの閾値（リサーチの確認窓で評価された値。tuning禁止）
TP_PCT = 2.0
SL_PCT = 2.0
# 記録する板フィールド（意味の解釈は分析側で行う。kabu APIのBid/Ask命名は要実データ確認）
_BOARD_KEYS = ("CurrentPrice", "CurrentPriceTime", "CalcPrice", "BidPrice", "BidQty",
               "AskPrice", "AskQty", "TradingVolume")

_STATE_DIR = Path("data/live_reports")


def _board_snap(board: dict) -> dict:
    return {k: board.get(k) for k in _BOARD_KEYS if k in board}


class ShadowMonitor:
    """建玉ごとの仮想TP/SL状態機械（純ロジック・テスト可能）."""

    def __init__(self, tp_pct: float = TP_PCT, sl_pct: float = SL_PCT):
        self.tp = tp_pct / 100.0
        self.sl = sl_pct / 100.0
        self.triggered: dict[tuple[str, str], dict] = {}   # (symbol, kind) -> event
        self.pending_fill: list[dict] = []                 # 次サンプルで約定proxyを記録

    def process_sample(self, now: str, symbol: str, side: str, entry_px: float,
                       board: dict) -> list[dict]:
        """1サンプル処理。返り値は記録すべきイベント行（0〜複数）."""
        out = []
        px = board.get("CurrentPrice") or board.get("CalcPrice")
        if not px or not entry_px or float(px) <= 0:
            return out
        px = float(px)
        # ロング: ret=px/建値-1 / ショート: 利益方向を正に揃える
        raw = px / float(entry_px) - 1.0
        ret = raw if side == "LONG" else -raw

        # 直前トリガーの「次サンプル約定proxy」（F1仮定の実測対応物）
        for ev in self.pending_fill:
            if ev["symbol"] == symbol:
                out.append({"type": "fill_proxy", "time": now, "symbol": symbol,
                            "kind": ev["kind"], "px": px, "board": _board_snap(board),
                            "detect_px": ev["px"],
                            "gap_bps": (px / ev["px"] - 1.0) * 1e4})
        self.pending_fill = [e for e in self.pending_fill if e["symbol"] != symbol]

        for kind, hit in (("TP", ret >= self.tp), ("SL", ret <= -self.sl)):
            key = (symbol, kind)
            if hit and key not in self.triggered:
                ev = {"type": "trigger", "time": now, "symbol": symbol, "side": side,
                      "kind": kind, "entry_px": float(entry_px), "px": px,
                      "ret_pct": round(ret * 100, 3), "board": _board_snap(board)}
                self.triggered[key] = ev
                self.pending_fill.append(ev)
                out.append(ev)
        return out


def run_shadow(client: KabuClientProtocol, cfg: LiveConfig,
               until: str = "15:32", interval_s: float = 60.0,
               sleep=time.sleep, now_fn=None) -> dict:
    """場中ループ本体。positions(建値=Price)を監視し JSONL に全サンプルを記録."""
    now_fn = now_fn or (lambda: dt.datetime.now())
    day = now_fn().strftime("%Y-%m-%d")
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = _STATE_DIR / f"shadow_{day}.jsonl"
    mon = ShadowMonitor()
    n_samples = 0
    with path.open("a", encoding="utf-8") as f:
        while True:
            now = now_fn()
            if now.strftime("%H:%M") >= until:
                break
            positions = [p for p in client.positions(product=2)
                         if float(p.get("LeavesQty") or 0) > 0]
            for pos in positions:
                ksym = to_kabu_symbol(pos.get("Symbol"))
                side = "LONG" if str(pos.get("Side")) == "2" else "SHORT"
                entry_px = pos.get("Price")
                try:
                    board = client.board(ksym)
                except Exception:  # noqa: BLE001 - 板取得失敗はスキップ（次周期で再試行）
                    continue
                stamp = now.strftime("%H:%M:%S")
                row = {"type": "sample", "time": stamp, "symbol": ksym, "side": side,
                       "entry_px": entry_px, "px": board.get("CurrentPrice")}
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                n_samples += 1
                for ev in mon.process_sample(stamp, ksym, side, float(entry_px or 0), board):
                    f.write(json.dumps(ev, ensure_ascii=False, default=str) + "\n")
            f.flush()
            sleep(interval_s)
    summary = {"day": day, "samples": n_samples,
               "triggers": [dict(v, board=None) for v in mon.triggered.values()],
               "n_tp": sum(1 for k in mon.triggered if k[1] == "TP"),
               "n_sl": sum(1 for k in mon.triggered if k[1] == "SL"),
               "log": str(path)}
    return summary
