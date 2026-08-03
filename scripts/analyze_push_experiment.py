"""50銘柄PUSH実験の判定材料を出す（実寄値が入った後・夕方に実行）。

    PYTHONPATH=. python scripts/analyze_push_experiment.py

出す数字（このスクリプトは合否を宣言しない・材料を出すだけ）:
  A. **候補リコール**: 実寄値で組んだ本来の建玉のうち、候補50に入っていた割合
     → 2パス方式（早い1周で絞る→同時気配で決める）の**上限**。低ければ設計が成立しない
  B. **スメアあり vs 同時 の比較**: 同じ銘柄・同じ日で λ/R²/σ_ε がどう変わるか
     → 「気配が予測しない」がスメア由来だったのかを切り分ける（STATUS §0 の未解決点）
  C. **建玉一致率**: 候補50内で、気配で選んだ建玉と実寄値で選んだ建玉の一致率
     → analyze_quotesnap.py の事前登録表に当てて Sharpe 維持率を読む
"""
from __future__ import annotations

import glob
import json

import numpy as np
import pandas as pd

from trading.jp_intraday.daily_model import load_panel_cached
from trading.jp_intraday.live import push_experiment as pxm
from trading.jp_intraday.live.config import LiveConfig


def decompose(gap_quote: np.ndarray, gap_actual: np.ndarray) -> dict:
    """quote = λ·actual + ε（analyze_quotesnap.py と同じ向き）。"""
    x, y = np.asarray(gap_actual, float), np.asarray(gap_quote, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or (x * x).sum() == 0:
        return {"n": len(x)}
    lam = float((x * y).sum() / (x * x).sum())
    resid = (y - lam * x) * 100.0
    ss = ((y - y.mean()) ** 2).sum()
    return {"n": len(x), "lam": lam,
            "r2": float(1 - ((y - lam * x) ** 2).sum() / ss) if ss > 0 else float("nan"),
            "sigma_bps": float(resid.std()),
            "rho": float(pd.Series(y).corr(pd.Series(x), method="spearman"))}


def main() -> int:
    files = sorted(glob.glob("data/live_reports/push_experiment_2*.json")  # _dry_ は除外)
    if not files:
        raise SystemExit("実験ログがありません（scripts/run_push_experiment.py を朝に実行）")
    cfg = LiveConfig.from_env()
    panel = load_panel_cached(min_value_yen=cfg.min_value_yen)
    panel["day"] = pd.to_datetime(panel["date"]).dt.strftime("%Y-%m-%d")

    for f in files:
        rec = json.loads(open(f, encoding="utf-8").read())
        day = rec["day"]
        p = panel[panel["day"] == day]
        if p.empty:
            print(f"\n[{day}] 実寄値がまだパネルに無い（collect_jp_daily_history 後に再実行）")
            continue
        print(f"\n===== {day} （選定方式={rec['select']}・ユニバース{rec['universe_n']}） =====")
        prev_close = (p["raw_close"] / (1 + p["overnight_gap"])).where(
            p["overnight_gap"].notna())
        actual = pd.DataFrame({"symbol": p["symbol"],
                               "gap_actual": p["overnight_gap"] * 100,
                               "open_actual": p["raw_close"].where(False)})
        # 実寄値ベースの本来の建玉（全ユニバース）
        opens_actual = dict(zip(p["symbol"], (1 + p["overnight_gap"]) * prev_close))
        nps = rec.get("names_per_side", cfg.names_per_side)
        true_book, per_sleeve = pxm.ensemble_book(p.copy(), opens_actual, cfg.strategy, nps)
        chosen = set(rec["chosen"])
        recall = len(true_book & chosen) / max(len(true_book), 1)
        print(f"A. 候補リコール: 本来の建玉 {len(true_book)}銘柄中 "
              f"{len(true_book & chosen)}銘柄が候補{len(chosen)}に含まれる → **{recall*100:.0f}%**")
        print(f"   （早い1周で選んだ建玉候補との一致: "
              f"{pxm.book_overlap(set(rec['early_book']), true_book)*100:.0f}%）")

        gap_map = dict(zip(actual["symbol"], actual["gap_actual"]))
        pc_map = dict(zip(p["symbol"], prev_close))

        def gaps(quotes: dict) -> tuple:
            gq, ga = [], []
            for s, q in quotes.items():
                if s in gap_map and pc_map.get(s):
                    gq.append((float(q) / float(pc_map[s]) - 1) * 100)
                    ga.append(float(gap_map[s]))
            return np.array(gq), np.array(ga)

        # B. スメアあり（早い1周）vs 同時（PUSH）
        early_q = {s: v for s, v in rec["sweep"]["quotes"].items() if s in chosen}
        gq, ga = gaps(early_q)
        d = decompose(gq, ga)
        print(f"B. スメアあり（早い1周・スメア{rec['sweep']['smear_s']:.0f}秒・候補50のみ）")
        print(f"   {_fmt(d)}")
        for hhmm, snap in rec["snapshots"].items():
            gq2, ga2 = gaps({s: v["q"] for s, v in snap["quotes"].items()})
            d2 = decompose(gq2, ga2)
            print(f"   同時 {hhmm}（スメア{snap['smear_s']:.1f}秒・PUSH {snap['push_messages']}件）")
            print(f"   {_fmt(d2)}")

        # C. 候補50内での建玉一致率
        sub = p[p["symbol"].isin(chosen)].copy()
        n_side = min(nps, max(len(sub) // 4, 1))
        book_actual, _ = pxm.ensemble_book(
            sub, {s: opens_actual[s] for s in sub["symbol"] if s in opens_actual},
            cfg.strategy, n_side)
        for hhmm, snap in rec["snapshots"].items():
            qmap = {s: v["q"] for s, v in snap["quotes"].items() if s in set(sub["symbol"])}
            if not qmap:
                continue
            book_q, _ = pxm.ensemble_book(sub, qmap, cfg.strategy, n_side)
            print(f"C. 建玉一致率（候補内・{n_side}銘柄/側）{hhmm}: "
                  f"**{pxm.book_overlap(book_q, book_actual)*100:.0f}%**")
        print("   判定表: ≥75%→実弾GO可 / 50-75%→減額で開始 / <50%→NO-GO"
              "（analyze_quotesnap.py の事前登録表）")
    return 0


def _fmt(d: dict) -> str:
    if "lam" not in d:
        return f"   n={d.get('n', 0)} … データ不足"
    return (f"   n={d['n']} λ={d['lam']:.3f} R²={d['r2']:.3f} "
            f"σ_ε={d['sigma_bps']:.0f}bps 順位ρ={d['rho']:.3f}")


if __name__ == "__main__":
    raise SystemExit(main())
