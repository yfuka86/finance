#!/usr/bin/env python3
"""
Parameter optimization for short-timeframe strategies:
  1. Volume Spike (5m)
  2. VWAP Reversion (15m / 1H)

One-dimensional sweep on 2024, combine best, then validate on 2023-2025.
"""
import sys
import time
import copy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trading.bybit.backtest import run_backtest

# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def run_single(strategy, symbol, interval, start, end, **params):
    """Run one backtest, return metrics dict (or empty on failure)."""
    try:
        r = run_backtest(
            strategy=strategy, symbol=symbol, interval=interval,
            start=start, end=end, initial_equity=10000.0, slippage_bps=1.0,
            **params,
        )
        return r.metrics
    except Exception as e:
        print(f"    [ERROR] {e}")
        return {}


def run_multi_symbol(strategy, symbols, interval, start, end, **params):
    """Run across symbols, return aggregated metrics."""
    all_ret = []
    all_sharpe = []
    all_dd = []
    all_trades = []
    all_pf = []
    all_wr = []
    for sym in symbols:
        m = run_single(strategy, sym, interval, start, end, **params)
        if m:
            all_ret.append(m.get("total_return_pct", 0))
            all_sharpe.append(m.get("sharpe_ratio", 0))
            all_dd.append(m.get("max_drawdown_pct", 0))
            all_trades.append(m.get("n_trades", 0))
            all_pf.append(m.get("profit_factor", 0))
            all_wr.append(m.get("win_rate_pct", 0))
    if not all_ret:
        return {}
    return {
        "total_return_pct": sum(all_ret) / len(all_ret),
        "sharpe_ratio": sum(all_sharpe) / len(all_sharpe),
        "max_drawdown_pct": min(all_dd),  # worst DD
        "n_trades": sum(all_trades),
        "profit_factor": sum(all_pf) / len(all_pf),
        "win_rate_pct": sum(all_wr) / len(all_wr),
    }


def fmt_metrics(m):
    if not m:
        return "FAILED"
    return (f"Ret: {m.get('total_return_pct',0):+7.2f}% | "
            f"Sharpe: {m.get('sharpe_ratio',0):6.3f} | "
            f"MaxDD: {m.get('max_drawdown_pct',0):7.2f}% | "
            f"Trades: {m.get('n_trades',0):4d} | "
            f"PF: {m.get('profit_factor',0):5.2f} | "
            f"WR: {m.get('win_rate_pct',0):5.1f}%")


def sweep_one_dim(strategy, symbols, interval, start, end,
                  base_params, param_name, values, config_key):
    """Sweep one parameter dimension. Returns best value."""
    print(f"\n  --- Sweeping {param_name}: {values} ---")
    best_val = base_params[config_key]
    best_sharpe = -999
    for v in values:
        params = copy.copy(base_params)
        params[config_key] = v
        m = run_multi_symbol(strategy, symbols, interval, start, end, **params)
        sharpe = m.get("sharpe_ratio", -999) if m else -999
        marker = ""
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_val = v
            marker = " <-- BEST"
        print(f"    {param_name}={v:8} => {fmt_metrics(m)}{marker}")
    print(f"  => Best {param_name} = {best_val} (Sharpe={best_sharpe:.3f})")
    return best_val


def multi_year_test(strategy, symbols, interval, base_params, years, label=""):
    """Test on multiple years, return list of (year, metrics)."""
    print(f"\n{'='*80}")
    print(f"  Multi-year validation: {label}")
    print(f"{'='*80}")
    results = []
    for y in years:
        start = f"{y}-01-01"
        end = f"{y}-12-31"
        print(f"\n  Year {y}:")
        per_sym = {}
        for sym in symbols:
            m = run_single(strategy, sym, interval, start, end, **base_params)
            per_sym[sym] = m
            print(f"    {sym}: {fmt_metrics(m)}")
        # Aggregate
        agg = run_multi_symbol(strategy, symbols, interval, start, end, **base_params)
        print(f"    AVG:       {fmt_metrics(agg)}")
        results.append((y, agg, per_sym))
    return results


# ═══════════════════════════════════════════════════════════════════
# 1. Volume Spike (5m)
# ═══════════════════════════════════════════════════════════════════

def optimize_volume_spike():
    print("\n" + "#"*80)
    print("#  VOLUME SPIKE (5m) OPTIMIZATION")
    print("#"*80)

    strategy = "volume_spike"
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    interval = "5"
    opt_start = "2024-01-01"
    opt_end = "2024-12-31"

    base = dict(
        vs_vol_lookback=200,
        vs_vol_zscore=4.0,
        vs_wick_ratio=0.65,
        vs_max_hold=12,
        vs_cooldown=12,
        vs_stop_buffer_pct=0.1,
        vs_tp_wick_mult=2.0,
        vs_order_size_usd=2000,
        vs_min_range_pct=0.15,
    )

    # Baseline
    print("\n  --- BASELINE (2024) ---")
    m = run_multi_symbol(strategy, symbols, interval, opt_start, opt_end, **base)
    print(f"    Baseline: {fmt_metrics(m)}")

    # Sweep dimensions one at a time
    best_zscore = sweep_one_dim(
        strategy, symbols, interval, opt_start, opt_end,
        base, "vol_zscore", [3.0, 3.5, 4.0, 5.0], "vs_vol_zscore")
    base["vs_vol_zscore"] = best_zscore

    best_wick = sweep_one_dim(
        strategy, symbols, interval, opt_start, opt_end,
        base, "wick_ratio", [0.50, 0.55, 0.60, 0.65, 0.70], "vs_wick_ratio")
    base["vs_wick_ratio"] = best_wick

    best_hold = sweep_one_dim(
        strategy, symbols, interval, opt_start, opt_end,
        base, "max_hold", [6, 12, 24], "vs_max_hold")
    base["vs_max_hold"] = best_hold

    best_cd = sweep_one_dim(
        strategy, symbols, interval, opt_start, opt_end,
        base, "cooldown", [6, 12, 24, 48], "vs_cooldown")
    base["vs_cooldown"] = best_cd

    best_tp = sweep_one_dim(
        strategy, symbols, interval, opt_start, opt_end,
        base, "tp_wick_mult", [1.5, 2.0, 3.0], "vs_tp_wick_mult")
    base["vs_tp_wick_mult"] = best_tp

    best_range = sweep_one_dim(
        strategy, symbols, interval, opt_start, opt_end,
        base, "min_range_pct", [0.10, 0.15, 0.20, 0.30], "vs_min_range_pct")
    base["vs_min_range_pct"] = best_range

    print(f"\n  === BEST COMBINED VOLUME SPIKE CONFIG ===")
    for k, v in sorted(base.items()):
        print(f"    {k} = {v}")

    # Multi-year validation
    years = [2023, 2024, 2025]
    vs_results = multi_year_test(strategy, symbols, interval, base, years,
                                  label="Volume Spike (5m)")
    return base, vs_results


# ═══════════════════════════════════════════════════════════════════
# 2. VWAP Reversion (15m and 1H)
# ═══════════════════════════════════════════════════════════════════

def optimize_vwap_reversion():
    print("\n" + "#"*80)
    print("#  VWAP REVERSION OPTIMIZATION")
    print("#"*80)

    strategy = "vwap_reversion"
    symbols = ["BTCUSDT", "SOLUSDT"]

    all_configs = {}
    all_results = {}

    for interval, reset_bars, cd_base, cd_values, label in [
        ("15", 96, 24, [16, 24, 32, 48], "15m"),
        ("60", 24, 6,  [4, 6, 8, 12],   "1H"),
    ]:
        print(f"\n  {'='*60}")
        print(f"  VWAP Reversion — {label} interval")
        print(f"  {'='*60}")

        opt_start = "2024-01-01"
        opt_end = "2024-12-31"

        base = dict(
            vwap_entry_sd=2.0,
            vwap_stop_sd=2.5,
            vwap_rsi_period=6,
            vwap_rsi_long=20.0,
            vwap_rsi_short=80.0,
            vwap_reset_bars=reset_bars,
            vwap_max_move_pct=3.0,
            vwap_order_size_usd=3000,
            vwap_min_dev_pct=0.5,
            vwap_cooldown=cd_base,
        )

        # Baseline
        print(f"\n  --- BASELINE ({label}, 2024) ---")
        m = run_multi_symbol(strategy, symbols, interval, opt_start, opt_end, **base)
        print(f"    Baseline: {fmt_metrics(m)}")

        # Sweep
        best_entry = sweep_one_dim(
            strategy, symbols, interval, opt_start, opt_end,
            base, "entry_sd", [1.5, 2.0, 2.5, 3.0], "vwap_entry_sd")
        base["vwap_entry_sd"] = best_entry

        best_stop = sweep_one_dim(
            strategy, symbols, interval, opt_start, opt_end,
            base, "stop_sd", [2.0, 2.5, 3.0, 3.5], "vwap_stop_sd")
        base["vwap_stop_sd"] = best_stop

        best_rsi = sweep_one_dim(
            strategy, symbols, interval, opt_start, opt_end,
            base, "rsi_long", [15, 20, 25, 30], "vwap_rsi_long")
        base["vwap_rsi_long"] = best_rsi

        best_dev = sweep_one_dim(
            strategy, symbols, interval, opt_start, opt_end,
            base, "min_dev_pct", [0.2, 0.3, 0.5, 0.8], "vwap_min_dev_pct")
        base["vwap_min_dev_pct"] = best_dev

        best_cd = sweep_one_dim(
            strategy, symbols, interval, opt_start, opt_end,
            base, "cooldown", cd_values, "vwap_cooldown")
        base["vwap_cooldown"] = best_cd

        best_move = sweep_one_dim(
            strategy, symbols, interval, opt_start, opt_end,
            base, "max_move_pct", [2.0, 3.0, 4.0, 5.0], "vwap_max_move_pct")
        base["vwap_max_move_pct"] = best_move

        print(f"\n  === BEST COMBINED VWAP {label} CONFIG ===")
        for k, v in sorted(base.items()):
            print(f"    {k} = {v}")

        # Multi-year
        years = [2023, 2024, 2025]
        vwap_results = multi_year_test(strategy, symbols, interval, base, years,
                                        label=f"VWAP Reversion ({label})")
        all_configs[label] = base
        all_results[label] = vwap_results

    return all_configs, all_results


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    # 1. Volume Spike
    vs_config, vs_results = optimize_volume_spike()

    # 2. VWAP Reversion
    vwap_configs, vwap_results = optimize_vwap_reversion()

    elapsed = time.time() - t0

    # ── Final Summary ──────────────────────────────────────────────
    print("\n" + "="*100)
    print("  FINAL SUMMARY")
    print("="*100)

    # Collect all configs with 3-year avg sharpe
    candidates = []

    # Volume Spike
    avg_sharpe = sum(m.get("sharpe_ratio", 0) for _, m, _ in vs_results) / len(vs_results)
    avg_ret = sum(m.get("total_return_pct", 0) for _, m, _ in vs_results) / len(vs_results)
    worst_dd = min(m.get("max_drawdown_pct", 0) for _, m, _ in vs_results)
    candidates.append(("Volume Spike 5m", vs_config, avg_sharpe, avg_ret, worst_dd, vs_results))

    # VWAP configs
    for label, results in vwap_results.items():
        cfg = vwap_configs[label]
        avg_s = sum(m.get("sharpe_ratio", 0) for _, m, _ in results) / len(results)
        avg_r = sum(m.get("total_return_pct", 0) for _, m, _ in results) / len(results)
        w_dd = min(m.get("max_drawdown_pct", 0) for _, m, _ in results)
        candidates.append((f"VWAP Reversion {label}", cfg, avg_s, avg_r, w_dd, results))

    # Sort by avg sharpe
    candidates.sort(key=lambda x: x[2], reverse=True)

    print(f"\n  TOP CONFIGS (ranked by 3-year avg Sharpe):\n")
    for rank, (name, cfg, avg_s, avg_r, w_dd, results) in enumerate(candidates[:3], 1):
        print(f"  #{rank}  {name}")
        print(f"      3-yr Avg Sharpe: {avg_s:.3f} | Avg Return: {avg_r:+.2f}% | Worst DD: {w_dd:.2f}%")
        print(f"      Config: {cfg}")
        print(f"      Year-by-year:")
        for yr, m, _ in results:
            print(f"        {yr}: {fmt_metrics(m)}")
        print()

    print(f"\n  Total elapsed: {elapsed/60:.1f} minutes")
    print("="*100)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)
    main()
