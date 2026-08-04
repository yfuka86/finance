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
# 「気配をいつまで待てば選択が良くなるか」の曲線を引くための時点。
# PUSHなら取得作業が不要なので**決定を寄付き直前まで遅らせられる**（従来は1周30分が
# 律速で08:20固定だった）。発注は実測8件/秒＝32銘柄で約4秒なので 08:59:30 決定でも
# 板寄せに間に合う。最終10分を細かく刻んで限界点を測る。
SNAP_TIMES = ("08:30", "08:40", "08:50", "08:55", "08:57", "08:58",
              "08:59:00", "08:59:30", "08:59:50")
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
def select_for_ensemble(last: pd.DataFrame, opens: dict, strategy: str,
                        names_per_side: int, n: int = MAX_PUSH) -> tuple[list[str], dict]:
    """**2パス方式の1パス目**: 早い1周の気配で候補 n 銘柄を選ぶ（ensemble対応）。

    ①まず本番と同じ建て方の建玉（両スリーブの統合）を必ず入れる
    ②残り枠を各スリーブの次点（n+1位以降）で埋める
       ← 気配が更新されると建玉は入れ替わるので、**境界の外側に余裕を持たせる**のが要点。
         この余裕がどれだけ効くかが実験のA指標（候補リコール）で測られる。
    戻り値: (候補リスト, 診断情報)
    """
    from trading.jp_intraday.live.executor import _score_today
    from trading.jp_intraday.strategies import STRATEGIES

    book, per = ensemble_book(last, opens, strategy, names_per_side)
    chosen = list(book)[:n]
    spec = STRATEGIES[strategy]
    members = [m for m, _ in (spec.get("members") or [(strategy, 1.0)])]
    depth = names_per_side
    while len(chosen) < n and depth < names_per_side * 6:
        depth += 1
        for member in members:
            s = _score_today(last, opens, member)
            for col in ("shortable", "prev_value", "short_restricted"):
                if col not in s and col in last:
                    s = s.merge(last[["symbol", col]], on="symbol", how="left")
            for sym in book_from_scores(s, depth):
                if sym not in chosen and len(chosen) < n:
                    chosen.append(sym)
    return chosen, {"book": sorted(book),
                    "per_sleeve": {k: sorted(v) for k, v in per.items()},
                    "depth_used": depth}


def select_symbols(scored: pd.DataFrame, method: str, n: int = MAX_PUSH,
                   names_per_side: int = 8) -> list[str]:
    """早い1周の結果から、PUSH登録する n 銘柄を選ぶ（absgap/liquidity 用の簡易版）。

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


def _short_pool(scored: pd.DataFrame) -> pd.DataFrame:
    """executor._sleeve_rows と同じショート適格フィルタ（貸借・流動性・規制）。"""
    p = scored
    if "shortable" in p:
        p = p[p["shortable"].fillna(False)]
    if "prev_value" in p:
        p = p[p["prev_value"].fillna(0) >= 1e9]
    if "short_restricted" in p:
        p = p[~p["short_restricted"].fillna(False)]
    return p


def book_from_scores(scored: pd.DataFrame, names_per_side: int) -> set:
    """1スリーブの建玉（ロング上位n＋ショート下位n）。ショート側は適格銘柄のみ。"""
    if scored.empty:
        return set()
    longs = set(scored.nlargest(names_per_side, "_s")["symbol"])
    shorts = set(_short_pool(scored).nsmallest(names_per_side, "_s")["symbol"]) - longs
    return longs | shorts


def ensemble_book(last: pd.DataFrame, opens: dict, strategy: str,
                  names_per_side: int) -> tuple[set, dict]:
    """**本番と同じ建て方**でensembleの建玉集合を作る。

    ensemble_core は「スリーブごとに上位/下位n本を取り、両スリーブを統合」する
    （executor.generate_plan と同じ）。zスコアを合成した1本のランキングとは
    別物になるので、一致率の測定にはこちらを使う。
    戻り値: (統合後の建玉集合, {スリーブ名: そのスリーブの建玉集合})
    """
    from trading.jp_intraday.live.executor import _score_today
    from trading.jp_intraday.strategies import STRATEGIES

    spec = STRATEGIES[strategy]
    members = spec.get("members", [(strategy, 1.0)]) if spec["kind"] == "ensemble" \
        else [(strategy, 1.0)]
    per: dict = {}
    for member, _w in members:
        s = _score_today(last, opens, member)
        for col in ("shortable", "prev_value", "short_restricted"):
            if col not in s and col in last:
                s = s.merge(last[["symbol", col]], on="symbol", how="left")
        per[member] = book_from_scores(s, names_per_side)
    return set().union(*per.values()) if per else set(), per


def book_overlap(a: set, b: set) -> float:
    """建玉一致率（事前登録の判定表に当てる指標）。"""
    if not a and not b:
        return float("nan")
    return len(a & b) / max(len(a | b), 1)


# ── 記録 ────────────────────────────────────────────────────────
def out_path(day: str | None = None, dry: bool = False) -> Path:
    """本番の記録は push_experiment_YYYY-MM-DD.json。

    **スモークテスト(--dry)は別ファイルに書く**。同じパスに書くと、その日の
    本番記録を上書きして消してしまう（2026-08-03 に実際にやらかした）。
    """
    day = day or dt.date.today().isoformat()
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    return _OUT_DIR / f"push_experiment{'_dry' if dry else ''}_{day}.json"


def save(record: dict, day: str | None = None, dry: bool = False) -> Path:
    p = out_path(day, dry)
    p.write_text(json.dumps(record, ensure_ascii=False, default=str, indent=1),
                 encoding="utf-8")
    return p


def wait_until(hhmm: str, now_fn=dt.datetime.now, sleep=time.sleep,
               lead_s: float = 0.0) -> None:
    """指定時刻（-lead_s）まで待つ。過ぎていれば即戻る。"HH:MM" と "HH:MM:SS" 対応。"""
    fmt = "%H:%M:%S" if hhmm.count(":") == 2 else "%H:%M"
    target = dt.datetime.strptime(hhmm, fmt).time()
    while True:
        now = now_fn()
        t = dt.datetime.combine(now.date(), target) - dt.timedelta(seconds=lead_s)
        remain = (t - now).total_seconds()
        if remain <= 0:
            return
        sleep(min(remain, 5) if remain > 5 else remain)   # 秒精度で刻むため


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
