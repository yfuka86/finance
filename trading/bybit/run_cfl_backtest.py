"""Run MTF Confluence backtests across all symbols for 2023-2025."""
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bybit.backtest import run_backtest

SYMBOLS = ["ADAUSDT", "AVAXUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT",
           "DOTUSDT", "ETHUSDT", "LINKUSDT", "SOLUSDT", "XRPUSDT"]

YEARS = [
    ("2023", "2023-01-01", "2023-12-31"),
    ("2024", "2024-01-01", "2024-12-31"),
    ("2025", "2025-01-01", "2025-03-31"),
]

PARAMS = dict(
    cfl_rsi_period=14,
    cfl_rsi_entry=35.0,
    cfl_vol_lookback=60,
    cfl_vol_mult=1.5,
    cfl_ema_15m=20,
    cfl_bars_per_15m=3,
    cfl_ema_fast=20,
    cfl_ema_slow=50,
    cfl_adx_period=14,
    cfl_adx_threshold=25.0,
    cfl_bars_per_1h=12,
    cfl_sl_pct=0.75,
    cfl_tp_pct=1.0,
    cfl_max_hold=72,
    cfl_cooldown=36,
    cfl_order_size_usd=3000.0,
)

results = {}

for sym in SYMBOLS:
    results[sym] = {}
    for label, start, end in YEARS:
        print(f"\n{'='*60}")
        print(f"  {sym} / {label} (5m)")
        print(f"{'='*60}")
        try:
            r = run_backtest(
                strategy="mtf_confluence",
                symbol=sym,
                interval="5",
                start=start,
                end=end,
                initial_equity=10000.0,
                slippage_bps=1.0,
                **PARAMS,
            )
            m = r.metrics
            results[sym][label] = {
                "total_return_pct": m.get("total_return_pct", 0),
                "annualized_return_pct": m.get("annualized_return_pct", 0),
                "sharpe_ratio": m.get("sharpe_ratio", 0),
                "sortino_ratio": m.get("sortino_ratio", 0),
                "max_drawdown_pct": m.get("max_drawdown_pct", 0),
                "n_trades": m.get("n_trades", 0),
                "win_rate_pct": m.get("win_rate_pct", 0),
                "profit_factor": m.get("profit_factor", 0),
            }
            print(f"  Return: {m.get('total_return_pct', 0):.2f}%  "
                  f"Sharpe: {m.get('sharpe_ratio', 0):.3f}  "
                  f"Trades: {m.get('n_trades', 0)}  "
                  f"WR: {m.get('win_rate_pct', 0):.1f}%  "
                  f"MaxDD: {m.get('max_drawdown_pct', 0):.2f}%")
        except Exception as e:
            print(f"  ERROR: {e}")
            results[sym][label] = {
                "total_return_pct": 0, "annualized_return_pct": 0,
                "sharpe_ratio": 0, "sortino_ratio": 0,
                "max_drawdown_pct": 0, "n_trades": 0,
                "win_rate_pct": 0, "profit_factor": 0,
            }

# Save results
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "validated_results_cfl.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}")

# Print summary
print("\n\n" + "="*80)
print("SUMMARY: MTF Confluence Strategy (5m)")
print("="*80)
print(f"{'Symbol':<12} {'2023 SR':>10} {'2024 SR':>10} {'2025 SR':>10} {'2023 Ret%':>10} {'2024 Ret%':>10} {'2025 Ret%':>10}")
print("-"*80)
for sym in SYMBOLS:
    d = results[sym]
    print(f"{sym:<12} "
          f"{d.get('2023',{}).get('sharpe_ratio',0):>10.3f} "
          f"{d.get('2024',{}).get('sharpe_ratio',0):>10.3f} "
          f"{d.get('2025',{}).get('sharpe_ratio',0):>10.3f} "
          f"{d.get('2023',{}).get('total_return_pct',0):>10.2f} "
          f"{d.get('2024',{}).get('total_return_pct',0):>10.2f} "
          f"{d.get('2025',{}).get('total_return_pct',0):>10.2f}")
