from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import load_bars, resample_bars
from .walkforward import WalkForwardConfig, run_walk_forward


def main() -> None:
    parser = argparse.ArgumentParser(description="Leakage-safe JP intraday walk-forward test")
    parser.add_argument("bars", help="CSV/parquet with timestamp,symbol,OHLCV")
    parser.add_argument("--interval", type=int, choices=(1, 5), default=1)
    parser.add_argument("--train-days", type=int, default=40)
    parser.add_argument("--test-days", type=int, default=10)
    parser.add_argument("--step-days", type=int, default=10)
    parser.add_argument("--commission-bps", type=float, default=0.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--borrow-rate", type=float, default=0.02)
    parser.add_argument("--output", default="data/jp_intraday_results")
    args = parser.parse_args()
    bars = resample_bars(load_bars(args.bars), args.interval)
    cfg = WalkForwardConfig(
        train_days=args.train_days, test_days=args.test_days, step_days=args.step_days,
        interval_minutes=args.interval, commission_bps=args.commission_bps,
        slippage_bps=args.slippage_bps, borrow_rate_annual=args.borrow_rate,
    )
    returns, folds, summary = run_walk_forward(bars, cfg)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    returns.to_csv(output / f"returns_{args.interval}m.csv", index=False)
    folds.to_csv(output / f"folds_{args.interval}m.csv", index=False)
    (output / f"summary_{args.interval}m.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
