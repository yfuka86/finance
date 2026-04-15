#!/usr/bin/env python3
"""
全プリセットのバックテストを一括実行し、年利20%以上のもの残す。
2025年1月〜3月（利用可能な直近データ）で検証。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trading.bybit.backtest import run_backtest
from trading.bybit.presets import STRATEGY_PRESETS

SYMBOL = "BTCUSDT"
START = "2025-01-01"
END = "2025-03-31"
INITIAL_EQUITY = 10000.0
THRESHOLD_ANNUAL_RETURN = 20.0


def main():
    results = []
    total = len(STRATEGY_PRESETS)

    for i, (key, preset) in enumerate(STRATEGY_PRESETS.items()):
        interval = preset["recommended_interval"]
        strategy = preset["strategy"]
        params = preset["params"]

        print(f"\n[{i+1}/{total}] {preset['name']} ({key})")
        print(f"  Strategy: {strategy}, Interval: {interval}m")

        try:
            result = run_backtest(
                strategy=strategy,
                symbol=SYMBOL,
                interval=interval,
                start=START,
                end=END,
                initial_equity=INITIAL_EQUITY,
                slippage_bps=1.0,
                **params,
            )
            m = result.metrics
            ann_ret = m.get("annualized_return_pct", 0)
            print(f"  Return: {m.get('total_return_pct', 0):+.2f}% | "
                  f"Annual: {ann_ret:+.2f}% | "
                  f"Sharpe: {m.get('sharpe_ratio', 0):.3f} | "
                  f"MaxDD: {m.get('max_drawdown_pct', 0):.2f}% | "
                  f"Trades: {m.get('n_trades', 0)} | "
                  f"WinRate: {m.get('win_rate_pct', 0):.1f}%")
            results.append((key, preset["name"], ann_ret, m))
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append((key, preset["name"], 0, {}))

    # Summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY (sorted by annual return)")
    print("=" * 80)

    results.sort(key=lambda x: x[2], reverse=True)

    passed = []
    for key, name, ann_ret, m in results:
        marker = "✓" if ann_ret >= THRESHOLD_ANNUAL_RETURN else "✗"
        print(f"  {marker} {name:30s}  Annual: {ann_ret:+8.2f}%  "
              f"Sharpe: {m.get('sharpe_ratio', 0):6.3f}  "
              f"MaxDD: {m.get('max_drawdown_pct', 0):7.2f}%  "
              f"Trades: {m.get('n_trades', 0):5d}")
        if ann_ret >= THRESHOLD_ANNUAL_RETURN:
            passed.append(key)

    print(f"\n年利{THRESHOLD_ANNUAL_RETURN}%以上: {len(passed)}/{len(results)}")
    print(f"合格: {', '.join(passed)}")

    return passed


if __name__ == "__main__":
    main()
