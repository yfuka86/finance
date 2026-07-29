"""売り制約を段階的に外した場合のレバレッジ検証（実データ・実単元・実保証金）。

本番構成 ensemble_core（¥20M・8銘柄/側/スリーブ・7bps）のまま、ショート側の
3制約を段階的に解除して信用倍率を掃引する:

  A 本番制約   : 貸借銘柄のみ + 売買代金≥¥10億 + 価格規制トリガー銘柄50単元キャップ
  B キャップ解除: 50単元キャップのみ解除（貸借・フロアは維持）
  C フロアも解除: + 売買代金フロア解除（貸借のみ維持）
  D 全解除     : + 貸借限定も解除（全銘柄ショート可＝理論上限。一般信用等で
                  借株できた場合の上限であり、制度信用では実行不能な参考値）

価格・ユニバース・単元(100株)・保証金（ストップ高×30%・超過日は縮小）は全て実データ／
実ルールのまま。信用倍率の上限3.3xは法定（保証金率30%）なので掃引もそこまで。

実行:  PYTHONPATH=. python scripts/experiment_short_constraints_leverage.py
"""
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

from trading.jp_intraday.daily_model import annualized_stats, load_panel_cached
from trading.jp_intraday.strategies import (
    STRATEGIES, _combine_sleeves, score_frame, unit_lot_backtest,
)

CAPITAL = 2e7
NPS = 8
COST = 7.0
MARGINS = [1.0, 2.0, 2.5, 3.0, 3.3]
VARIANTS = {
    "A 本番制約":    dict(require_shortable=True,  short_min_value_yen=1e9, apply_short_reg_cap=True),
    "B キャップ解除": dict(require_shortable=True,  short_min_value_yen=1e9, apply_short_reg_cap=False),
    "C フロアも解除": dict(require_shortable=True,  short_min_value_yen=0.0, apply_short_reg_cap=False),
    "D 全解除":      dict(require_shortable=False, short_min_value_yen=0.0, apply_short_reg_cap=False),
}


def run(frames, members, margin, **kw):
    sleeves = []
    for m, w in members:
        con = STRATEGIES[m].get("construction", "dollar_neutral")
        d, b = unit_lot_backtest(frames[m], capital_yen=CAPITAL * w, names_per_side=NPS,
                                 margin_ratio=margin, cost_bps_side=COST,
                                 construction=con, **kw)
        d = d.copy()
        d[["gross", "net"]] = d[["gross", "net"]] * w
        sleeves.append((1.0, d, b))
    daily, _ = _combine_sleeves(sleeves)
    daily["date"] = pd.to_datetime(daily["date"])
    return daily


def main() -> None:
    panel = load_panel_cached()
    members = STRATEGIES["ensemble_core"]["members"]
    frames = {m: score_frame(panel, m) for m, _ in members}
    print(f"panel: {panel['date'].min().date()}〜{panel['date'].max().date()} / "
          f"ensemble_core ¥{CAPITAL/1e6:.0f}M・{NPS}銘柄/側/スリーブ・{COST:.0f}bps")

    for vname, kw in VARIANTS.items():
        print(f"\n== {vname} ==")
        print("  倍率   年率%   Sharpe  勝率%   最大DD%  実効レバ  ショート充足率  | OOS24+ 年率%/Sh")
        for m in MARGINS:
            d = run(frames, members, m, **kw)
            s = annualized_stats(d, "net")
            so = annualized_stats(d[d["date"] >= "2024-01-01"], "net")
            eff = d["deployed_yen"].mean() / CAPITAL
            # ショート側が目標（片側=元本×倍率/2）をどれだけ満たせたか
            short_fill = d["short_yen"].mean() / (CAPITAL * m / 2)
            print(f"  {m:3.1f}x {s['ann_return']*100:7.1f} {s['sharpe']:7.2f} "
                  f"{s['win_rate']*100:6.1f} {s['max_drawdown']*100:8.1f} {eff:8.2f}x "
                  f"{short_fill*100:11.0f}%     | {so['ann_return']*100:6.1f}%/{so['sharpe']:.2f}")


if __name__ == "__main__":
    main()
