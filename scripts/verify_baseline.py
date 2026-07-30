"""確定ベースライン (README/AGENTS の数値) の再現検証ランナー。

環境再構築（Windows 検証機など）後にデータ・コードが正しく揃ったかを、
本命 ensemble_core の3構成で確認する:

  1. リサーチ @3bps — OOS(2024-08-01+) ネットSharpe ≈ 8.77
  2. リサーチ @7bps — OOS(2024-08-01+) ネットSharpe ≈ 7.14
  3. 本番換算 単元BT ¥20M・8銘柄/側・信用2倍・7bps
       全期間   ≈ 年率53.6% / Sh 2.20 / DD −21%
       OOS24+  ≈ 年率101%  / Sh 3.21 / DD −18%   ※起点 2024-01-01

注意: 「OOS24+（本番換算）」の起点は 2024-01-01（暦年境界＝MLは2023年まで学習）。
リサーチOOSの起点 2024-08-01（IS前60%/OOS後40%の分割点）とは異なる。

データ期間が正本（日次2018-01〜2026-07-24）より延長されている場合、全期間系の
数値は変わり得る。Sharpe は既定 ±10% を許容（--tol で変更、マスタ更新でも±1%程度動く）。

実行:  PYTHONPATH=. python scripts/verify_baseline.py [--tol 0.10]
終了コード: 全PASS=0 / FAILあり=1
"""
import argparse
import sys
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

RESEARCH_OOS = "2024-08-01"   # IS(前60%)/OOS(後40%) 分割点
UNIT_OOS = "2024-01-01"       # 本番換算のOOS24+（暦年境界）

# (名称, 期待値dict)
EXPECTED = [
    ("リサーチ @3bps OOS24-08+", {"sharpe": 8.77}),
    ("リサーチ @7bps OOS24-08+", {"sharpe": 7.14}),
    # 2026-07-30 更新: 規制銘柄ショート除外（発注拒否前提）+ 全面流動性フロア¥10億が本番条件
    ("単元¥20M 8/側 2.0x 7bps 全期間", {"sharpe": 1.70, "ann_return": 0.356, "max_drawdown": -0.187}),
    ("単元¥20M 8/側 2.0x 7bps OOS24+", {"sharpe": 2.98, "ann_return": 0.791, "max_drawdown": -0.187}),
]


def _check(name: str, got: dict, exp: dict, tol: float) -> bool:
    ok = True
    parts = []
    for k, e in exp.items():
        g = got[k]
        rel = abs(g - e) / max(abs(e), 1e-9)
        # リターン/DDはSharpeよりノイジーなので許容2倍
        lim = tol if k == "sharpe" else tol * 2
        good = rel <= lim
        ok &= good
        parts.append(f"{k}={g:.3f} (期待{e:.3f}, 乖離{rel*100:.1f}%{'' if good else ' ✗'})")
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: " + ", ".join(parts))
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=0.10, help="Sharpe相対許容（既定10%%）")
    args = ap.parse_args()

    from trading.jp_intraday.daily_model import annualized_stats, load_panel_cached
    from trading.jp_intraday.strategies import run_strategy, run_unit_lot

    print("パネル構築（初回~15s / ウォーム~1s）…")
    panel = load_panel_cached()                      # リサーチ基準（¥5億）
    panel_prod = load_panel_cached(min_value_yen=1e9)  # 本番条件（¥10億全面フロア）
    print(f"  {panel['date'].min().date()} 〜 {panel['date'].max().date()} / "
          f"{panel['symbol'].nunique()}銘柄 {len(panel):,}行")

    results = []
    for bps, (name, exp) in zip((3.0, 7.0), EXPECTED[:2]):
        d, _ = run_strategy(panel, "ensemble_core", cost_bps_side=bps)
        d["date"] = pd.to_datetime(d["date"])
        s = annualized_stats(d[d["date"] >= pd.Timestamp(RESEARCH_OOS)], "net")
        results.append(_check(name, s, exp, args.tol))

    d, _ = run_unit_lot(panel_prod, "ensemble_core", capital_yen=2e7, names_per_side=8,
                        margin_ratio=2.0, cost_bps_side=7.0)
    d["date"] = pd.to_datetime(d["date"])
    results.append(_check(EXPECTED[2][0], annualized_stats(d, "net"), EXPECTED[2][1], args.tol))
    o = annualized_stats(d[d["date"] >= pd.Timestamp(UNIT_OOS)], "net")
    results.append(_check(EXPECTED[3][0], o, EXPECTED[3][1], args.tol))

    n_ok = sum(results)
    print(f"\n{n_ok}/{len(results)} PASS")
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    main()
