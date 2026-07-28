"""Validate overnight-gap reversal over ~5y of daily data (regime-stability test)."""
from pathlib import Path
import json

import pandas as pd

from trading.jp_intraday.daily_gap import (
    backtest_gap, build_gap_panel, load_existing_daily, report,
)


def main() -> None:
    daily = load_existing_daily()  # reuse on-disk data only; no re-download
    out = Path("data/jp_daily_gap_experiments")
    out.mkdir(parents=True, exist_ok=True)

    panel = build_gap_panel(daily, min_value_yen=5e8)
    print(f"panel: rows={len(panel)}  days={panel['date'].nunique()}  "
          f"symbols={panel['symbol'].nunique()}  "
          f"avg_names/day={len(panel)/panel['date'].nunique():.0f}")

    summary = {}
    for label, direction in [("gap_fade", -1), ("gap_momentum", 1)]:
        rep = report(backtest_gap(panel, quantile=0.2, direction=direction, cost_bps_side=5.0))
        print(f"\n=== {label} (quantile=0.2, cost=5bps/side) ===")
        print(rep.to_string(index=False))
        rep.to_csv(out / f"{label}_yearly.csv", index=False)
        summary[label] = rep[rep["period"] == "ALL"].iloc[0].to_dict()

    # quantile sensitivity for the fade (are results knob-robust?)
    print("\n=== gap_fade quantile sensitivity (ALL period) ===")
    grid = []
    for q in (0.1, 0.15, 0.2, 0.3):
        rep = report(backtest_gap(panel, quantile=q, direction=-1, cost_bps_side=5.0))
        row = rep[rep["period"] == "ALL"].iloc[0]
        grid.append({"quantile": q, "gross_sharpe": row["gross_sharpe"], "net_sharpe": row["net_sharpe"]})
    print(pd.DataFrame(grid).to_string(index=False))

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
