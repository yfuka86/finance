from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .data import load_bars, resample_bars
from .model import ModelConfig, run_model_walk_forward
from .universe import build_point_in_time_universe, filter_intraday


def main() -> None:
    p = argparse.ArgumentParser(description="Point-in-time TOPIX intraday model research")
    p.add_argument("bars")
    p.add_argument("memberships", help="symbol,effective_from,effective_to")
    p.add_argument("shares", help="symbol,known_at,shares")
    p.add_argument("--interval", type=int, choices=(1, 5), default=1)
    p.add_argument("--min-market-cap", type=float, default=100_000_000_000)
    p.add_argument("--train-days", type=int, default=60)
    p.add_argument("--test-days", type=int, default=10)
    p.add_argument("--slippage-bps", type=float, default=2.0)
    p.add_argument("--output", default="data/jp_intraday_model_results")
    args = p.parse_args()
    one_minute = load_bars(args.bars)
    daily = one_minute.assign(date=one_minute.timestamp.dt.tz_localize(None).dt.normalize()).groupby(
        ["date", "symbol"], as_index=False
    ).agg(close=("close", "last"))
    membership = pd.read_csv(args.memberships, dtype={"symbol": str})
    shares = pd.read_csv(args.shares, dtype={"symbol": str})
    universe = build_point_in_time_universe(daily, membership, shares, args.min_market_cap)
    bars = resample_bars(filter_intraday(one_minute, universe), args.interval)
    cfg = ModelConfig(train_days=args.train_days, test_days=args.test_days,
                      interval_minutes=args.interval, slippage_bps=args.slippage_bps)
    returns, folds, coefficients, summary = run_model_walk_forward(bars, cfg)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    universe.to_csv(output / "point_in_time_universe.csv", index=False)
    returns.to_csv(output / f"model_returns_{args.interval}m.csv", index=False)
    folds.to_csv(output / f"model_folds_{args.interval}m.csv", index=False)
    coefficients.to_csv(output / f"coefficients_{args.interval}m.csv", index=False)
    (output / f"model_summary_{args.interval}m.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
