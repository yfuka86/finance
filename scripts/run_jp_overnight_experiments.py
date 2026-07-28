"""Overnight-gap + order-flow edge scan on the 100bn TOPIX intraday universe.

Short sample (~39 sessions), so this is SIGNAL DISCOVERY, not a validated
backtest: we ask whether any signal has a positive *gross* Sharpe before costs.
"""
from pathlib import Path
import json

import pandas as pd

from trading.jp_intraday.overnight import STRATEGIES, evaluate_overnight, prepare_overnight


DEV_START, DEV_END = "2026-06-01", "2026-07-17"  # within tick coverage


def main() -> None:
    bars = pd.read_parquet("data/jp_minutes_100bn_bulk/jp_5m_eligible.parquet")
    ticks = pd.read_parquet(
        "data/jp_ticks_100bn/tick_imbalance_5m_2026-06-01_2026-07-17.parquet"
    )[["timestamp", "symbol", "signed_volume", "traded_volume"]]
    topix = pd.read_csv(
        "data/jp_intraday_reference/topixweight_current.csv",
        encoding="cp932", dtype={"コード": str},
    )
    sectors = pd.DataFrame({"symbol": topix["コード"], "sector": topix["業種"]}).dropna()

    frame = prepare_overnight(bars, sectors, ticks)
    print(f"symbols={frame['symbol'].nunique()}  sessions={frame['date'].nunique()}  "
          f"gap coverage={frame['residual_gap'].notna().mean():.1%}")

    output = Path("data/jp_overnight_experiments")
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    for strategy in STRATEGIES:
        for holding in (6, 12, 30):  # 30min, 1h, whole morning
            _, summary = evaluate_overnight(
                frame, strategy, DEV_START, DEV_END,
                quantile=0.2, signal_bar=1, holding_bars=holding,
            )
            summary["holding_bars"] = holding
            rows.append(summary)
    ranking = pd.DataFrame(rows).sort_values("gross_sharpe", ascending=False)
    cols = ["strategy", "holding_bars", "gross_sharpe", "sharpe",
            "gross_return_sum", "total_return", "turnover", "max_drawdown"]
    ranking = ranking[cols]
    ranking.to_csv(output / "overnight_ranking.csv", index=False)
    print(ranking.to_string(index=False))
    best = ranking.iloc[0].to_dict()
    (output / "overnight_best.json").write_text(json.dumps(best, indent=2, default=float))
    print("\nBEST:", json.dumps({k: best[k] for k in ("strategy", "holding_bars", "gross_sharpe", "sharpe")}, default=float))


if __name__ == "__main__":
    main()
