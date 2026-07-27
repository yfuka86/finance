from pathlib import Path
import json

import pandas as pd

from trading.jp_intraday.experiments import (
    STRATEGIES, evaluate_strategy, prepare_signals,
)


def main() -> None:
    bars = pd.read_parquet("data/jp_minutes_100bn_bulk/jp_5m_eligible.parquet")
    topix = pd.read_csv(
        "data/jp_intraday_reference/topixweight_current.csv", encoding="cp932",
        dtype={"コード": str},
    )
    sectors = pd.DataFrame({"symbol": topix["コード"], "sector": topix["業種"]}).dropna()
    prepared = prepare_signals(bars, sectors, 5)
    output = Path("data/jp_strategy_experiments")
    output.mkdir(parents=True, exist_ok=True)
    development = []
    for strategy in STRATEGIES:
        returns, summary = evaluate_strategy(
            prepared, strategy, "2026-06-01", "2026-07-17", 5
        )
        summary["strategy"] = strategy
        development.append(summary)
        returns.to_csv(output / f"development_{strategy}.csv", index=False)
    ranking = pd.DataFrame(development).sort_values("sharpe", ascending=False)
    ranking.to_csv(output / "development_ranking.csv", index=False)
    best = str(ranking.iloc[0]["strategy"])
    final_returns, final = evaluate_strategy(
        prepared, best, "2026-07-21", "2026-07-24", 5
    )
    final["strategy"] = best
    final_returns.to_csv(output / "final_holdout_returns.csv", index=False)
    (output / "final_holdout_summary.json").write_text(json.dumps(final, indent=2))
    print(ranking.to_string(index=False))
    print("FINAL", json.dumps(final))


if __name__ == "__main__":
    main()
