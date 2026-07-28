#!/usr/bin/env python3
"""
MTF RSI2 SOL parameter optimization script.

Smart approach:
1. Fix base params, sweep one dimension at a time on 2024 data
2. Combine best values, test on all 3 years (2023, 2024, 2025)
"""
import sys
import os
import logging
import itertools
from copy import deepcopy

# Suppress noisy logs during optimization
logging.basicConfig(level=logging.WARNING)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from trading.bybit.backtest import run_backtest

# ── Base parameters ──────────────────────────────────────────────
BASE_PARAMS = dict(
    mtf_rsi_period=2,
    mtf_rsi_entry_long=5.0,
    mtf_rsi_entry_short=95.0,
    mtf_rsi_exit_long=65.0,
    mtf_rsi_exit_short=35.0,
    mtf_atr_period=14,
    mtf_atr_sl_mult=3.5,
    mtf_atr_tp_mult=7.0,
    mtf_htf_bars=4,
    mtf_htf_ema_fast=20,
    mtf_htf_ema_slow=50,
    mtf_order_size_usd=3000,
    mtf_min_hold=24,
    mtf_cooldown=48,
    mtf_trend_gap_pct=1.0,
)

STRATEGY = "mtf_rsi2"
SYMBOL = "SOLUSDT"

# ── Timeframe configs ────────────────────────────────────────────
TF_CONFIGS = {
    "60m": {
        "interval": "60",
        "htf_bars": 4,
        "cooldown_values": [24, 48, 72],
        "min_hold_values": [12, 24, 36],
    },
    "15m": {
        "interval": "15",
        "htf_bars": 16,
        "cooldown_values": [96, 192, 288],
        "min_hold_values": [48, 96, 144],
    },
}

# ── Sweep ranges ─────────────────────────────────────────────────
SWEEP_PARAMS = {
    "rsi_entry_long": [3, 5, 10],
    "rsi_exit_long": [55, 60, 65, 70],
    "atr_sl_mult": [2.0, 3.0, 3.5, 4.0],
    "atr_tp_mult": [5.0, 7.0, 10.0],
    "trend_gap_pct": [0.5, 1.0, 1.5, 2.0],
}


def run_single(interval, start, end, overrides):
    """Run a single backtest and return key metrics."""
    params = deepcopy(BASE_PARAMS)
    params.update(overrides)
    try:
        result = run_backtest(
            strategy=STRATEGY,
            symbol=SYMBOL,
            interval=interval,
            start=start,
            end=end,
            initial_equity=10000.0,
            slippage_bps=1.0,
            **params,
        )
        m = result.metrics
        return {
            "return_pct": m.get("total_return_pct", 0),
            "sharpe": m.get("sharpe_ratio", 0),
            "max_dd": m.get("max_drawdown_pct", 0),
            "n_trades": m.get("n_trades", 0),
            "win_rate": m.get("win_rate_pct", 0),
            "profit_factor": m.get("profit_factor", 0),
        }
    except Exception as e:
        print(f"  ERROR: {e}")
        return {"return_pct": -999, "sharpe": -999, "max_dd": -999, "n_trades": 0, "win_rate": 0, "profit_factor": 0}


def print_result(label, res):
    """Print a single result row."""
    print(f"  {label:<55s} | ret={res['return_pct']:>8.2f}% | sharpe={res['sharpe']:>6.3f} | maxDD={res['max_dd']:>7.2f}% | trades={res['n_trades']:>4d} | WR={res['win_rate']:>5.1f}% | PF={res['profit_factor']:>5.2f}")


def sweep_dimension(dim_name, values, param_key, tf_name, tf_config):
    """Sweep one parameter dimension, return best value."""
    interval = tf_config["interval"]
    print(f"\n{'='*100}")
    print(f"SWEEP: {dim_name} = {values}  (TF={tf_name}, 2024)")
    print(f"{'='*100}")

    best_val = None
    best_sharpe = -999

    for val in values:
        overrides = {"mtf_htf_bars": tf_config["htf_bars"]}
        overrides[param_key] = val
        # Mirror short side for entry/exit
        if param_key == "mtf_rsi_entry_long":
            overrides["mtf_rsi_entry_short"] = 100.0 - val
        elif param_key == "mtf_rsi_exit_long":
            overrides["mtf_rsi_exit_short"] = 100.0 - val

        label = f"{dim_name}={val}"
        res = run_single(interval, "2024-01-01", "2024-12-31", overrides)
        print_result(label, res)

        if res["sharpe"] > best_sharpe:
            best_sharpe = res["sharpe"]
            best_val = val

    print(f"  >>> Best {dim_name}: {best_val} (sharpe={best_sharpe:.3f})")
    return best_val


def main():
    all_final_results = []

    for tf_name, tf_config in TF_CONFIGS.items():
        interval = tf_config["interval"]
        print(f"\n{'#'*100}")
        print(f"# TIMEFRAME: {tf_name} (interval={interval}, htf_bars={tf_config['htf_bars']})")
        print(f"{'#'*100}")

        # ── Phase 1: Sweep each dimension independently on 2024 ──

        best_entry = sweep_dimension(
            "rsi_entry_long", SWEEP_PARAMS["rsi_entry_long"],
            "mtf_rsi_entry_long", tf_name, tf_config,
        )

        best_exit = sweep_dimension(
            "rsi_exit_long", SWEEP_PARAMS["rsi_exit_long"],
            "mtf_rsi_exit_long", tf_name, tf_config,
        )

        best_sl = sweep_dimension(
            "atr_sl_mult", SWEEP_PARAMS["atr_sl_mult"],
            "mtf_atr_sl_mult", tf_name, tf_config,
        )

        best_tp = sweep_dimension(
            "atr_tp_mult", SWEEP_PARAMS["atr_tp_mult"],
            "mtf_atr_tp_mult", tf_name, tf_config,
        )

        best_cooldown = sweep_dimension(
            "cooldown", tf_config["cooldown_values"],
            "mtf_cooldown", tf_name, tf_config,
        )

        best_minhold = sweep_dimension(
            "min_hold", tf_config["min_hold_values"],
            "mtf_min_hold", tf_name, tf_config,
        )

        best_gap = sweep_dimension(
            "trend_gap_pct", SWEEP_PARAMS["trend_gap_pct"],
            "mtf_trend_gap_pct", tf_name, tf_config,
        )

        # ── Phase 2: Combine best values, test all 3 years ──────

        combined = {
            "mtf_htf_bars": tf_config["htf_bars"],
            "mtf_rsi_entry_long": best_entry,
            "mtf_rsi_entry_short": 100.0 - best_entry,
            "mtf_rsi_exit_long": best_exit,
            "mtf_rsi_exit_short": 100.0 - best_exit,
            "mtf_atr_sl_mult": best_sl,
            "mtf_atr_tp_mult": best_tp,
            "mtf_cooldown": best_cooldown,
            "mtf_min_hold": best_minhold,
            "mtf_trend_gap_pct": best_gap,
        }

        print(f"\n{'='*100}")
        print(f"COMBINED BEST for {tf_name}: {combined}")
        print(f"{'='*100}")

        years = [
            ("2023", "2023-01-01", "2023-12-31"),
            ("2024", "2024-01-01", "2024-12-31"),
            ("2025", "2025-01-01", "2025-12-31"),
        ]

        year_results = {}
        for year_label, start, end in years:
            res = run_single(interval, start, end, combined)
            print_result(f"{tf_name} combined | {year_label}", res)
            year_results[year_label] = res

        sum_ret = sum(year_results[y]["return_pct"] for y in ["2023", "2024", "2025"])
        avg_sharpe = sum(year_results[y]["sharpe"] for y in ["2023", "2024", "2025"]) / 3
        worst_dd = min(year_results[y]["max_dd"] for y in ["2023", "2024", "2025"])

        all_final_results.append({
            "tf": tf_name,
            "params": combined,
            "year_results": year_results,
            "sum_return": sum_ret,
            "avg_sharpe": avg_sharpe,
            "worst_dd": worst_dd,
        })

        # ── Phase 2b: Also test nearby combinations (top-3 grid) ──
        # Test a few alternative combos around the best values
        alt_entries = [v for v in SWEEP_PARAMS["rsi_entry_long"] if v != best_entry][:1]
        alt_exits = [v for v in SWEEP_PARAMS["rsi_exit_long"] if v != best_exit][:1]

        for alt_e in [best_entry] + alt_entries:
            for alt_x in [best_exit] + alt_exits:
                if alt_e == best_entry and alt_x == best_exit:
                    continue  # already tested
                alt_combined = deepcopy(combined)
                alt_combined["mtf_rsi_entry_long"] = alt_e
                alt_combined["mtf_rsi_entry_short"] = 100.0 - alt_e
                alt_combined["mtf_rsi_exit_long"] = alt_x
                alt_combined["mtf_rsi_exit_short"] = 100.0 - alt_x

                alt_year_results = {}
                for year_label, start, end in years:
                    res = run_single(interval, start, end, alt_combined)
                    print_result(f"{tf_name} alt(e={alt_e},x={alt_x}) | {year_label}", res)
                    alt_year_results[year_label] = res

                alt_sum_ret = sum(alt_year_results[y]["return_pct"] for y in ["2023", "2024", "2025"])
                alt_avg_sharpe = sum(alt_year_results[y]["sharpe"] for y in ["2023", "2024", "2025"]) / 3
                alt_worst_dd = min(alt_year_results[y]["max_dd"] for y in ["2023", "2024", "2025"])

                all_final_results.append({
                    "tf": tf_name,
                    "params": alt_combined,
                    "year_results": alt_year_results,
                    "sum_return": alt_sum_ret,
                    "avg_sharpe": alt_avg_sharpe,
                    "worst_dd": alt_worst_dd,
                })

    # ── Final ranking ────────────────────────────────────────────
    print(f"\n\n{'#'*100}")
    print(f"# FINAL RANKING: TOP 5 COMBOS (by avg Sharpe across 2023-2025)")
    print(f"{'#'*100}")

    all_final_results.sort(key=lambda x: x["avg_sharpe"], reverse=True)

    for rank, r in enumerate(all_final_results[:5], 1):
        print(f"\n--- Rank #{rank} ---")
        print(f"  TF: {r['tf']}")
        print(f"  Params: entry={r['params']['mtf_rsi_entry_long']}, exit={r['params']['mtf_rsi_exit_long']}, "
              f"SL={r['params']['mtf_atr_sl_mult']}, TP={r['params']['mtf_atr_tp_mult']}, "
              f"cooldown={r['params']['mtf_cooldown']}, min_hold={r['params']['mtf_min_hold']}, "
              f"gap={r['params']['mtf_trend_gap_pct']}")
        print(f"  Avg Sharpe: {r['avg_sharpe']:.3f} | Sum Return: {r['sum_return']:.2f}% | Worst DD: {r['worst_dd']:.2f}%")
        for y in ["2023", "2024", "2025"]:
            yr = r['year_results'][y]
            print(f"    {y}: ret={yr['return_pct']:>8.2f}% | sharpe={yr['sharpe']:>6.3f} | maxDD={yr['max_dd']:>7.2f}% | trades={yr['n_trades']:>4d} | WR={yr['win_rate']:>5.1f}%")

    # Also rank by sum of returns
    print(f"\n\n{'#'*100}")
    print(f"# FINAL RANKING: TOP 5 COMBOS (by sum of returns across 2023-2025)")
    print(f"{'#'*100}")

    all_final_results.sort(key=lambda x: x["sum_return"], reverse=True)

    for rank, r in enumerate(all_final_results[:5], 1):
        print(f"\n--- Rank #{rank} ---")
        print(f"  TF: {r['tf']}")
        print(f"  Params: entry={r['params']['mtf_rsi_entry_long']}, exit={r['params']['mtf_rsi_exit_long']}, "
              f"SL={r['params']['mtf_atr_sl_mult']}, TP={r['params']['mtf_atr_tp_mult']}, "
              f"cooldown={r['params']['mtf_cooldown']}, min_hold={r['params']['mtf_min_hold']}, "
              f"gap={r['params']['mtf_trend_gap_pct']}")
        print(f"  Sum Return: {r['sum_return']:.2f}% | Avg Sharpe: {r['avg_sharpe']:.3f} | Worst DD: {r['worst_dd']:.2f}%")
        for y in ["2023", "2024", "2025"]:
            yr = r['year_results'][y]
            print(f"    {y}: ret={yr['return_pct']:>8.2f}% | sharpe={yr['sharpe']:>6.3f} | maxDD={yr['max_dd']:>7.2f}% | trades={yr['n_trades']:>4d} | WR={yr['win_rate']:>5.1f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
