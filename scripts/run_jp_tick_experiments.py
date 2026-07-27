from pathlib import Path
import pandas as pd

from trading.jp_intraday.tick_experiments import (
    TICK_STRATEGIES, evaluate_tick_strategy, prepare_tick_signals,
)


def main():
    bars = pd.read_parquet("data/jp_minutes_100bn_bulk/jp_5m_eligible.parquet")
    ticks = pd.read_parquet("data/jp_ticks_100bn/tick_imbalance_5m_2026-06-01_2026-07-17.parquet")
    topix = pd.read_csv("data/jp_intraday_reference/topixweight_current.csv", encoding="cp932", dtype={"コード": str})
    sectors = pd.DataFrame({"symbol": topix["コード"], "sector": topix["業種"]}).dropna()
    frame = prepare_tick_signals(bars, ticks, sectors)
    output = Path("data/jp_tick_experiments"); output.mkdir(parents=True, exist_ok=True)
    rows = []
    for strategy in TICK_STRATEGIES:
        returns, summary = evaluate_tick_strategy(frame, strategy, "2026-06-01", "2026-07-17")
        rows.append(summary); returns.to_csv(output / f"development_{strategy}.csv", index=False)
    ranking = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    ranking.to_csv(output / "development_ranking.csv", index=False)
    print(ranking.to_string(index=False))


if __name__ == "__main__":
    main()
