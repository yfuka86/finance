"""50銘柄PUSH実験: 「スメアの無い気配」なら寄値を予測できるのかを実測する.

背景（STATUS §0）:
  実弾NO-GOの根拠になった「気配が寄値を予測しない」という実測は、
  **REST1周が十数分かかることによるスメアと交絡していて分離できていない**。
  登録済み50銘柄なら同時スナップショットが取れる（実測: 未登録900ms → 登録済み1.4ms）ので、
  同じ日・同じ銘柄で「スメアあり気配」と「同時気配」を並べれば交絡を切り離せる。

実験の流れ（発注は一切しない）:
  1. 早い時間に全ユニバースをREST1周（**スメアあり**）→ 各銘柄の取得時刻も記録
  2. その結果から候補50銘柄を選ぶ（選定方式は切替可能＝ここも測定対象）
  3. 50銘柄をPUSH登録し、08:50/08:55/08:59 に**同時スナップショット**
  4. 夕方、実寄値と突き合わせて以下を出す（analyze_push_experiment.py）
     - 候補リコール: 実寄値で組んだ本来の建玉のうち候補50に入っていた割合
       ＝**2パス方式の上限**。ここが低ければ設計として成立しない
     - スメアあり vs 同時 の λ/R²/σ_ε 比較 ＝ **交絡の切り分け**
     - 候補50内での建玉一致率（事前登録の判定表に当てる本体指標）

選定方式（--select）:
  strategy … 本番と同じスコアで上位/下位を取る（実運用に最も近い）
  absgap   … 残差ギャップの絶対値上位（テール＝αの source を厚く見る）
  liquidity… 流動性上位（対照群。気配は安定するはずだがαは薄い）
"""
from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

_OUT_DIR = Path("data/live_reports")
SNAP_TIMES = ("08:50", "08:55", "08:59")
MAX_PUSH = 50


def score_frame(last: pd.DataFrame, opens: dict, strategy: str) -> pd.DataFrame:
    """今日のスコア表を作る。ensemble はスリーブごとにスコアしてz合成する。

    本番の `generate_plan` はスリーブごとに建玉を作って統合するが、ここでの用途は
    **候補選定と一致率の測定**なので、スリーブのzスコア加重平均で1本のランキングにする
    （両スリーブの上位/下位が候補50に入ることが目的で、発注数量は作らない）。
    """
    from trading.jp_intraday.live.executor import _score_today
    from trading.jp_intraday.strategies import STRATEGIES

    spec = STRATEGIES[strategy]
    if spec["kind"] != "ensemble":
        return _score_today(last, opens, strategy)

    merged: pd.DataFrame | None = None
    for member, w in spec["members"]:
        s = _score_today(last, opens, member)[["symbol", "_s"]].copy()
        sd = s["_s"].std()
        s["_z"] = (s["_s"] - s["_s"].mean()) / (sd if sd and np.isfinite(sd) else 1.0)
        s["_z"] *= w
        s = s.drop(columns="_s")
        merged = s if merged is None else merged.merge(s, on="symbol", how="outer",
                                                       suffixes=("", f"_{member}"))
    zcols = [c for c in merged.columns if c.startswith("_z")]
    merged["_s"] = merged[zcols].fillna(0).sum(axis=1)
    base = last[["symbol", "residual_gap", "prev_value"]] if "residual_gap" in last \
        else last[["symbol", "prev_value"]]
    out = merged[["symbol", "_s"]].merge(base, on="symbol", how="left")
    if "residual_gap" not in out:
        # ギャップ列は _score_today 側で作られる（メンバの1つから拝借する）
        g = _score_today(last, opens, spec["members"][0][0])[["symbol", "residual_gap"]]
        out = out.merge(g, on="symbol", how="left")
    return out.dropna(subset=["_s"])


# ── 候補選定（純関数・テスト可能） ──────────────────────────────
def select_symbols(scored: pd.DataFrame, method: str, n: int = MAX_PUSH,
                   names_per_side: int = 8) -> list[str]:
    """早い1周の結果から、PUSH登録する n 銘柄を選ぶ。

    scored: symbol / _s(スコア) / residual_gap / prev_value を持つ DataFrame。
    両端を対称に取る（戦略はロング上位とショート下位の両方を建てるため）。
    """
    if scored.empty:
        return []
    n = min(n, MAX_PUSH, len(scored))
    if method == "liquidity":
        return list(scored.nlargest(n, "prev_value")["symbol"])
    key = "_s" if method == "strategy" else "residual_gap"
    if method == "absgap":
        # 絶対値上位＝ギャップのテール。符号のバランスは崩さない
        half = n // 2
        up = scored.nlargest(half, "residual_gap")["symbol"].tolist()
        dn = scored.nsmallest(n - half, "residual_gap")["symbol"].tolist()
        return list(dict.fromkeys(up + dn))[:n]
    half = n // 2
    top = scored.nlargest(half, key)["symbol"].tolist()
    bot = scored.nsmallest(n - half, key)["symbol"].tolist()
    out = list(dict.fromkeys(top + bot))[:n]
    if len(out) < n:                      # 重複で足りない分を中位から補充
        rest = [s for s in scored["symbol"] if s not in set(out)]
        out += rest[: n - len(out)]
    return out


def book_from_scores(scored: pd.DataFrame, names_per_side: int) -> set:
    """スコアから建てる銘柄集合（ロング上位n＋ショート下位n）。一致率の比較用。"""
    if scored.empty:
        return set()
    longs = set(scored.nlargest(names_per_side, "_s")["symbol"])
    shorts = set(scored.nsmallest(names_per_side, "_s")["symbol"])
    return longs | shorts


def book_overlap(a: set, b: set) -> float:
    """建玉一致率（事前登録の判定表に当てる指標）。"""
    if not a and not b:
        return float("nan")
    return len(a & b) / max(len(a | b), 1)


# ── 記録 ────────────────────────────────────────────────────────
def out_path(day: str | None = None) -> Path:
    day = day or dt.date.today().isoformat()
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    return _OUT_DIR / f"push_experiment_{day}.json"


def save(record: dict, day: str | None = None) -> Path:
    p = out_path(day)
    p.write_text(json.dumps(record, ensure_ascii=False, default=str, indent=1),
                 encoding="utf-8")
    return p


def wait_until(hhmm: str, now_fn=dt.datetime.now, sleep=time.sleep,
               lead_s: float = 0.0) -> None:
    """指定時刻（-lead_s）まで待つ。過ぎていれば即戻る。"""
    target = dt.datetime.strptime(hhmm, "%H:%M").time()
    while True:
        now = now_fn()
        t = dt.datetime.combine(now.date(), target) - dt.timedelta(seconds=lead_s)
        remain = (t - now).total_seconds()
        if remain <= 0:
            return
        sleep(min(remain, 5))


def quote_from_board(board: dict) -> float:
    """気配値。寄前は CalcPrice が前日終値のままなので **bid/ask の仲値を優先**する
    （2026-07-31 実測: CalcPrice==前日終値が16/16銘柄。CalcPriceは寄前の気配ではない）。
    """
    b, a = board.get("BidPrice"), board.get("AskPrice")
    if b and a:
        return (float(b) + float(a)) / 2
    if b or a:
        return float(b or a)
    px = board.get("CurrentPrice") or board.get("CalcPrice") or 0
    return float(px) if px else 0.0
