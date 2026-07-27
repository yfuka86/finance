"""
Parameter optimization for Momentum SOL and Donchian SOL strategies.
Sweeps one dimension at a time on 2024, then combines best and tests 2023-2025.
"""
import sys
import os
import time
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from trading.bybit.backtest import run_backtest

SYMBOL = "SOLUSDT"
TIMEFRAMES = ["240", "60"]
TF_LABELS = {"240": "4H", "60": "1H"}
YEARS = {"2023": ("2023-01-01", "2023-12-31"),
         "2024": ("2024-01-01", "2024-12-31"),
         "2025": ("2025-01-01", "2025-12-31")}


def fmt_metrics(m):
    return (f"Ret={m.get('total_return_pct',0):+.1f}%  "
            f"Sharpe={m.get('sharpe_ratio',0):.2f}  "
            f"MaxDD={m.get('max_drawdown_pct',0):.1f}%  "
            f"Trades={m.get('n_trades',0)}  "
            f"WR={m.get('win_rate_pct',0):.0f}%  "
            f"PF={m.get('profit_factor',0):.2f}")


def run_bt(strategy, interval, start, end, **params):
    """Run backtest with retry on transient errors."""
    for attempt in range(3):
        try:
            r = run_backtest(strategy=strategy, symbol=SYMBOL, interval=interval,
                             start=start, end=end, initial_equity=10000.0, **params)
            return r.metrics
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
            else:
                print(f"  ERROR: {e}")
                return {"total_return_pct": -999, "sharpe_ratio": -999,
                        "max_drawdown_pct": -99, "n_trades": 0,
                        "win_rate_pct": 0, "profit_factor": 0}


def sweep_one_dim(strategy, interval, start, end, base_params, dim_name, dim_values, param_prefix):
    """Sweep one dimension, return list of (value, metrics)."""
    results = []
    for val in dim_values:
        params = dict(base_params)
        params[f"{param_prefix}_{dim_name}"] = val
        m = run_bt(strategy, interval, start, end, **params)
        results.append((val, m))
        print(f"    {dim_name}={val:<6}  {fmt_metrics(m)}")
    return results


def pick_best(results, key="sharpe_ratio"):
    """Pick value with best sharpe (tie-break on return)."""
    valid = [(v, m) for v, m in results if m.get("sharpe_ratio", -999) > -999]
    if not valid:
        return results[0][0]
    best = max(valid, key=lambda x: (x[1].get(key, -999), x[1].get("total_return_pct", -999)))
    return best[0]


# =====================================================================
# 1. MOMENTUM SOL
# =====================================================================
print("=" * 80)
print("  MOMENTUM SOL OPTIMIZATION")
print("=" * 80)

mom_base = dict(
    mom_fast_period=20, mom_slow_period=50,
    mom_atr_period=14, mom_atr_multiplier=3.0,
    mom_order_size_usd=5000
)

mom_sweeps = {
    "fast_period": [10, 15, 20, 30],
    "slow_period": [40, 50, 75, 100],
    "atr_multiplier": [2.0, 3.0, 4.0, 5.0],
}

mom_best_configs = []  # (tf, params, label)

for tf in TIMEFRAMES:
    print(f"\n{'─'*60}")
    print(f"  Momentum {TF_LABELS[tf]} - Sweep on 2024")
    print(f"{'─'*60}")

    best_vals = {}

    for dim_name, dim_values in mom_sweeps.items():
        print(f"\n  Sweeping {dim_name}: {dim_values}")
        results = sweep_one_dim("momentum", tf, "2024-01-01", "2024-12-31",
                                mom_base, dim_name, dim_values, "mom")
        best_vals[dim_name] = pick_best(results)
        print(f"  >>> Best {dim_name} = {best_vals[dim_name]}")

    # Combine best
    combined = dict(mom_base)
    combined["mom_fast_period"] = best_vals["fast_period"]
    combined["mom_slow_period"] = best_vals["slow_period"]
    combined["mom_atr_multiplier"] = best_vals["atr_multiplier"]

    label = (f"fast={best_vals['fast_period']}, slow={best_vals['slow_period']}, "
             f"atr_mult={best_vals['atr_multiplier']}")
    print(f"\n  Combined best: {label}")

    # Test combined on 2024
    print(f"\n  Combined on 2024:")
    m2024 = run_bt("momentum", tf, "2024-01-01", "2024-12-31", **combined)
    print(f"    2024: {fmt_metrics(m2024)}")

    # Test all 3 years
    print(f"\n  Testing all years:")
    year_metrics = {}
    for yr, (s, e) in YEARS.items():
        m = run_bt("momentum", tf, s, e, **combined)
        year_metrics[yr] = m
        print(f"    {yr}: {fmt_metrics(m)}")

    # Compute 3-year aggregate
    avg_sharpe = sum(year_metrics[y].get("sharpe_ratio", 0) for y in YEARS) / 3
    avg_ret = sum(year_metrics[y].get("total_return_pct", 0) for y in YEARS) / 3
    worst_dd = min(year_metrics[y].get("max_drawdown_pct", 0) for y in YEARS)
    print(f"    3yr avg: Sharpe={avg_sharpe:.2f}  AvgRet={avg_ret:+.1f}%  WorstDD={worst_dd:.1f}%")

    mom_best_configs.append({
        "tf": TF_LABELS[tf],
        "params": combined,
        "label": label,
        "year_metrics": year_metrics,
        "avg_sharpe": avg_sharpe,
        "avg_ret": avg_ret,
        "worst_dd": worst_dd,
    })


# =====================================================================
# 2. DONCHIAN SOL
# =====================================================================
print("\n\n" + "=" * 80)
print("  DONCHIAN BREAKOUT SOL OPTIMIZATION")
print("=" * 80)

don_base = dict(
    don_entry_period=20, don_exit_period=10,
    don_atr_period=20, don_atr_sl_mult=3.0,
    don_order_size_usd=5000
)

don_sweeps = {
    "entry_period": [10, 15, 20, 30, 40],
    "exit_period": [5, 10, 15],
    "atr_sl_mult": [2.0, 3.0, 4.0, 5.0],
}

don_best_configs = []

for tf in TIMEFRAMES:
    print(f"\n{'─'*60}")
    print(f"  Donchian {TF_LABELS[tf]} - Sweep on 2024")
    print(f"{'─'*60}")

    best_vals = {}

    for dim_name, dim_values in don_sweeps.items():
        print(f"\n  Sweeping {dim_name}: {dim_values}")
        results = sweep_one_dim("donchian_breakout", tf, "2024-01-01", "2024-12-31",
                                don_base, dim_name, dim_values, "don")
        best_vals[dim_name] = pick_best(results)
        print(f"  >>> Best {dim_name} = {best_vals[dim_name]}")

    # Combine best
    combined = dict(don_base)
    combined["don_entry_period"] = best_vals["entry_period"]
    combined["don_exit_period"] = best_vals["exit_period"]
    combined["don_atr_sl_mult"] = best_vals["atr_sl_mult"]

    label = (f"entry={best_vals['entry_period']}, exit={best_vals['exit_period']}, "
             f"atr_sl={best_vals['atr_sl_mult']}")
    print(f"\n  Combined best: {label}")

    # Test combined on 2024
    print(f"\n  Combined on 2024:")
    m2024 = run_bt("donchian_breakout", tf, "2024-01-01", "2024-12-31", **combined)
    print(f"    2024: {fmt_metrics(m2024)}")

    # Test all 3 years
    print(f"\n  Testing all years:")
    year_metrics = {}
    for yr, (s, e) in YEARS.items():
        m = run_bt("donchian_breakout", tf, s, e, **combined)
        year_metrics[yr] = m
        print(f"    {yr}: {fmt_metrics(m)}")

    avg_sharpe = sum(year_metrics[y].get("sharpe_ratio", 0) for y in YEARS) / 3
    avg_ret = sum(year_metrics[y].get("total_return_pct", 0) for y in YEARS) / 3
    worst_dd = min(year_metrics[y].get("max_drawdown_pct", 0) for y in YEARS)
    print(f"    3yr avg: Sharpe={avg_sharpe:.2f}  AvgRet={avg_ret:+.1f}%  WorstDD={worst_dd:.1f}%")

    don_best_configs.append({
        "tf": TF_LABELS[tf],
        "params": combined,
        "label": label,
        "year_metrics": year_metrics,
        "avg_sharpe": avg_sharpe,
        "avg_ret": avg_ret,
        "worst_dd": worst_dd,
    })


# =====================================================================
# FINAL SUMMARY
# =====================================================================
print("\n\n" + "=" * 80)
print("  FINAL SUMMARY: TOP CONFIGS (ranked by 3yr avg Sharpe)")
print("=" * 80)

all_configs = []
for c in mom_best_configs:
    all_configs.append(("Momentum", c))
for c in don_best_configs:
    all_configs.append(("Donchian", c))

# Sort by avg sharpe
all_configs.sort(key=lambda x: x[1]["avg_sharpe"], reverse=True)

print(f"\n{'─'*60}")
print("  ALL CONFIGS RANKED")
print(f"{'─'*60}")
for i, (strat, c) in enumerate(all_configs, 1):
    print(f"\n  #{i} {strat} {c['tf']}")
    print(f"     Params: {c['label']}")
    print(f"     3yr avg Sharpe={c['avg_sharpe']:.2f}  AvgRet={c['avg_ret']:+.1f}%  WorstDD={c['worst_dd']:.1f}%")
    for yr in ["2023", "2024", "2025"]:
        m = c["year_metrics"][yr]
        print(f"     {yr}: {fmt_metrics(m)}")

# Top 3 per strategy
for strat_name, configs in [("Momentum", mom_best_configs), ("Donchian", don_best_configs)]:
    print(f"\n{'─'*60}")
    print(f"  TOP {strat_name.upper()} CONFIGS")
    print(f"{'─'*60}")
    ranked = sorted(configs, key=lambda c: c["avg_sharpe"], reverse=True)
    for i, c in enumerate(ranked[:3], 1):
        print(f"\n  #{i} {c['tf']}  {c['label']}")
        print(f"     3yr: Sharpe={c['avg_sharpe']:.2f}  AvgRet={c['avg_ret']:+.1f}%  WorstDD={c['worst_dd']:.1f}%")
        for yr in ["2023", "2024", "2025"]:
            m = c["year_metrics"][yr]
            print(f"     {yr}: {fmt_metrics(m)}")

print("\nDone.")
