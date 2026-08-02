"""気配歪み → 成績 の対応表を作る（実弾GO/NO-GOの合格基準を**事前登録**するため）.

## なぜ必要か

`analyze_quotesnap.py` は実測の気配誤差を λ / R² / σ_ε / 選択一致率 に分解するが、
**その数字がいくらなら実弾GOなのかの対応表が無い**。実測を見てから基準を決めると
「思ったより悪かったので基準を緩める」が必ず起きる（OOS絶対規律と同じ問題）。

そこで**実測が届く前に**、既知の歪みを本番と同じ経路（ensemble_core・¥20M・8銘柄/側・
信用2倍・¥10億フロア）に注入して「選択一致率 X% のとき Sharpe Y」の対応表を作る。

## 何を歪ませ、何を歪ませないか

歪むのは**当日の寄付き気配だけ**。以下は歪ませない（実際に既知だから）:
  - 執行価格（`open`）と当日リターン（`intraday_ret`）… 板寄せで必ず実寄値に約定する
  - 前日以前の全特徴量（`prev_resid_gap` `prev_intraday` `gap_vol60` 等）… 実現済み
歪ませるのは当日ギャップ由来の列のみ:
  `overnight_gap` → `residual_gap` `sector_resid_gap` `gap_abs` `gap_z` `idio_gap2`
系統歪み（圧縮λ・クリップ）はセクター指数ギャップにも同じ倍率で適用する（指数も同じ板寄せ
機構で決まるため）。ランダム誤差は定義上個別なのでセクター指数には乗せない
（多数銘柄の平均で σ/√N に潰れる）。

## 既知の限界（正直に）

ML スリーブは学習と予測の両方に歪んだ特徴量を渡している。実際は**学習は実寄値（クリーン）・
予測だけ気配（ノイズ）**なので:
  - 圧縮λ: 学習側も圧縮されると ridge が係数を自動的に大きくして完全復元する。
    これは「λは補正可能」の主張と同じ機構なので**この対応表は正しい**
  - ランダム誤差: 学習側もノイズ入りだと誤差変数バイアスで係数が縮み、それが偶然
    保護的に働く。実際の「クリーン学習×ノイズ予測」より**楽観側**に出る
  → したがって**ランダム誤差の行は下限（これ以上は悪化しうる）として読む**。
    svdn スリーブ（ルール系・学習なし）にはこの留保は付かない。

実行: PYTHONPATH=. python scripts/quote_distortion_calibration.py [--quick]
出力: data/live_reports/quote_distortion_calibration.csv
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

OOS_START = "2024-01-01"          # 本番換算OOSの起点（暦年境界）
NAMES_PER_SIDE = 8
CAPITAL, MARGIN, COST_BPS = 2e7, 2.0, 7.0


def distort(panel: pd.DataFrame, lam: float = 1.0, clip_pct: float | None = None,
            sigma_bps: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """当日ギャップだけを歪ませ、派生列を daily_model と同じ定義で作り直す."""
    p = panel.copy()
    g = p["overnight_gap"].to_numpy(copy=True)
    sec = p["sector_index_gap"].to_numpy(copy=True) if "sector_index_gap" in p else None

    if clip_pct is not None:                       # 特別気配の上限クリップ（%）
        g = np.clip(g, -clip_pct / 100, clip_pct / 100)
        if sec is not None:
            sec = np.clip(sec, -clip_pct / 100, clip_pct / 100)
    if lam != 1.0:                                 # 一様圧縮（系統歪み）
        g = g * lam
        if sec is not None:
            sec = sec * lam
    if sigma_bps:                                  # ランダム誤差（銘柄固有）
        rng = np.random.default_rng(seed)
        g = g + rng.normal(0, sigma_bps / 1e4, len(g))

    p["overnight_gap"] = g
    if sec is not None:
        p["sector_index_gap"] = sec
    return recompute_gap_features(p)


def recompute_gap_features(p: pd.DataFrame) -> pd.DataFrame:
    """overnight_gap を書き換えた後、派生列を daily_model.build_panel と同一定義で作り直す.

    gap_vol60 は過去実績（実寄値ベース）なので再計算しない＝歪ませない。
    """
    p["residual_gap"] = p["overnight_gap"].sub(p.groupby("date")["overnight_gap"].transform("mean"))
    p["sector_resid_gap"] = p["residual_gap"].sub(
        p.groupby(["date", "sector"])["residual_gap"].transform("mean"))
    p["gap_abs"] = p["residual_gap"].abs()
    if "gap_vol60" in p:
        p["gap_z"] = p["residual_gap"] / p["gap_vol60"]
    if "sector_index_gap" in p:
        p["idio_gap2"] = p["overnight_gap"] - p["sector_index_gap"]
    return p


def book_overlap(base_blot: pd.DataFrame, alt_blot: pd.DataFrame) -> tuple[float, float]:
    """実際に建てた玉どうしの一致率（銘柄数ベース / ¥加重）を日次平均で返す.

    ★スコアフレーム同士を突き合わせてはいけない: svdn_concentrated は各フレームが
    自分の上位5%テールだけを残すため、inner merge した時点で「両方が選んだ銘柄」に
    偏り、不一致が構造的に消える（実測: 2,046日中16本以上残るのは24日だけ＝
    サンプルの1.2%で測っていた）。**必ず約定ベース（blotter）で測る**。
    """
    def key(b):
        x = b[["date", "symbol", "side_label", "position_yen"]].copy()
        x["k"] = x["symbol"].astype(str) + "|" + x["side_label"].astype(str)
        return x
    a, c = key(base_blot), key(alt_blot)
    n_hits, y_hits = [], []
    for day, ga in a.groupby("date"):
        gc = c[c["date"].eq(day)]
        if ga.empty:
            continue
        sa, sc = set(ga["k"]), set(gc["k"])
        n_hits.append(len(sa & sc) / len(sa))
        w = ga.set_index("k")["position_yen"].abs()
        y_hits.append(w[w.index.isin(sc)].sum() / w.sum() if w.sum() else np.nan)
    return (float(np.nanmean(n_hits)) if n_hits else float("nan"),
            float(np.nanmean(y_hits)) if y_hits else float("nan"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="シナリオを絞って動作確認")
    args = ap.parse_args()

    from trading.jp_intraday.daily_model import annualized_stats, load_panel_cached
    from trading.jp_intraday.strategies import run_unit_lot

    print("パネル構築（本番条件 ¥10億フロア）…")
    panel = load_panel_cached(min_value_yen=1e9)
    print(f"  {panel['date'].min().date()} 〜 {panel['date'].max().date()} / "
          f"{panel['symbol'].nunique()}銘柄 {len(panel):,}行\n")

    scenarios = [
        ("歪みなし（基準）",              dict()),
        ("一様圧縮 λ=0.5",                dict(lam=0.5)),
        ("一様圧縮 λ=0.3",                dict(lam=0.3)),
        ("±3%クリップ",                   dict(clip_pct=3.0)),
        ("ランダム σ=100bps",             dict(sigma_bps=100)),
        ("ランダム σ=250bps",             dict(sigma_bps=250)),
        ("ランダム σ=500bps",             dict(sigma_bps=500)),
    ] if not args.quick else [
        ("歪みなし（基準）", dict()), ("ランダム σ=250bps", dict(sigma_bps=250)),
    ]

    rows, base_blot = [], None
    for name, kw in scenarios:
        p = distort(panel, **kw) if kw else panel
        d, blot = run_unit_lot(p, "ensemble_core", capital_yen=CAPITAL,
                               names_per_side=NAMES_PER_SIDE, margin_ratio=MARGIN,
                               cost_bps_side=COST_BPS)
        if base_blot is None:
            base_blot = blot
        n_ov, y_ov = (1.0, 1.0) if not kw else book_overlap(base_blot, blot)
        d["date"] = pd.to_datetime(d["date"])
        s = annualized_stats(d[d["date"] >= pd.Timestamp(OOS_START)], "net")
        rows.append({"シナリオ": name, "銘柄一致率": n_ov, "¥加重一致率": y_ov,
                     "年率": s["ann_return"], "Sharpe": s["sharpe"],
                     "maxDD": s["max_drawdown"], "日次勝率": s.get("win_rate", float("nan"))})
        print(f"  {name:22s} 銘柄一致 {n_ov*100:5.1f}% / ¥加重 {y_ov*100:5.1f}%  "
              f"年率 {s['ann_return']*100:6.1f}%  Sh {s['sharpe']:5.2f}  DD {s['max_drawdown']*100:6.1f}%")

    out = pd.DataFrame(rows)
    base_sh = out.iloc[0]["Sharpe"]
    out["Sharpe維持率"] = out["Sharpe"] / base_sh
    path = "data/live_reports/quote_distortion_calibration.csv"
    out.to_csv(path, index=False)

    print(f"\n{'='*78}\n【対応表】OOS24+ ・ensemble_core ¥20M/8銘柄per side/信用2倍/{COST_BPS:.0f}bps")
    print(out.assign(**{
        "銘柄一致%": (out["銘柄一致率"] * 100).round(1),
        "¥加重一致%": (out["¥加重一致率"] * 100).round(1),
        "年率%": (out["年率"] * 100).round(1),
        "Sharpe": out["Sharpe"].round(2),
        "maxDD%": (out["maxDD"] * 100).round(1),
        "維持率%": (out["Sharpe維持率"] * 100).round(0),
    })[["シナリオ", "銘柄一致%", "¥加重一致%", "年率%", "Sharpe", "maxDD%",
        "維持率%"]].to_string(index=False))
    print(f"\n保存: {path}")
    print("\n※ML行の留保: 学習側も歪ませているため、ランダム誤差の行は**楽観側**")
    print("　（実際はクリーン学習×ノイズ予測でこれ以上悪化しうる）。下限として読む。")


if __name__ == "__main__":
    main()
