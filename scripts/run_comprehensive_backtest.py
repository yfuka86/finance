#!/usr/bin/env python3
"""
Comprehensive backtest: MTF RSI2 on ETH/BTC, asymmetric SOL, portfolio simulation.
"""
import sys
import os
import logging

logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from copy import deepcopy
from trading.bybit.backtest import run_backtest

# ── Helpers ──────────────────────────────────────────────────────

def run_single(strategy, symbol, interval, start, end, params):
    """Run one backtest, return metrics dict."""
    try:
        result = run_backtest(
            strategy=strategy, symbol=symbol, interval=interval,
            start=start, end=end, initial_equity=10000.0,
            slippage_bps=1.0, **params,
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
        return {"return_pct": 0, "sharpe": 0, "max_dd": 0, "n_trades": 0, "win_rate": 0, "profit_factor": 0}


def print_header(title):
    print(f"\n{'='*110}")
    print(f"  {title}")
    print(f"{'='*110}")


def print_row(label, res):
    print(f"  {label:<60s} | ret={res['return_pct']:>8.2f}% | sharpe={res['sharpe']:>6.3f} | maxDD={res['max_dd']:>7.2f}% | trades={res['n_trades']:>4d}")


YEARS = [
    ("2023", "2023-01-01", "2023-12-31"),
    ("2024", "2024-01-01", "2024-12-31"),
    ("2025", "2025-01-01", "2025-03-31"),
]

# ── SOL optimized params (baseline) ─────────────────────────────
SOL_PARAMS = dict(
    mtf_rsi_period=2,
    mtf_rsi_entry_long=10.0,
    mtf_rsi_entry_short=90.0,
    mtf_rsi_exit_long=70.0,
    mtf_rsi_exit_short=30.0,
    mtf_atr_period=14,
    mtf_atr_sl_mult=2.0,
    mtf_atr_tp_mult=5.0,
    mtf_htf_bars=16,
    mtf_htf_ema_fast=20,
    mtf_htf_ema_slow=50,
    mtf_order_size_usd=3000,
    mtf_min_hold=96,
    mtf_cooldown=288,
    mtf_trend_gap_pct=1.0,
)

# ══════════════════════════════════════════════════════════════════
# PART 1: MTF RSI2 15m on ETH and BTC (SOL params as baseline)
# ══════════════════════════════════════════════════════════════════

print_header("PART 1a: MTF RSI2 15m - ETH & BTC with SOL-optimized params")

part1_results = {}  # key: (symbol, gap, cooldown) -> {year: metrics}

for symbol in ["ETHUSDT", "BTCUSDT"]:
    print(f"\n  --- {symbol} (baseline SOL params) ---")
    for year_label, start, end in YEARS:
        res = run_single("mtf_rsi2", symbol, "15", start, end, SOL_PARAMS)
        print_row(f"{symbol} {year_label}", res)
        part1_results[(symbol, 1.0, 288)] = part1_results.get((symbol, 1.0, 288), {})
        part1_results[(symbol, 1.0, 288)][year_label] = res


# ── Grid search: trend_gap_pct x cooldown for ETH and BTC ──────

print_header("PART 1b: Grid search trend_gap_pct x cooldown for ETH & BTC")

GAP_VALUES = [0.5, 1.0, 1.5]
COOLDOWN_VALUES = [192, 288, 384]

best_by_symbol = {}  # symbol -> (best_params_key, best_avg_sharpe, year_results)

for symbol in ["ETHUSDT", "BTCUSDT"]:
    print(f"\n  --- {symbol} grid search ---")
    best_sharpe = -999
    best_key = None

    for gap in GAP_VALUES:
        for cd in COOLDOWN_VALUES:
            if gap == 1.0 and cd == 288:
                # Already tested above
                key = (symbol, gap, cd)
                year_res = part1_results[key]
            else:
                params = deepcopy(SOL_PARAMS)
                params["mtf_trend_gap_pct"] = gap
                params["mtf_cooldown"] = cd
                key = (symbol, gap, cd)
                year_res = {}
                for year_label, start, end in YEARS:
                    res = run_single("mtf_rsi2", symbol, "15", start, end, params)
                    year_res[year_label] = res
                part1_results[key] = year_res

            avg_sharpe = sum(year_res[y]["sharpe"] for y in ["2023", "2024", "2025"]) / 3
            sum_ret = sum(year_res[y]["return_pct"] for y in ["2023", "2024", "2025"])
            worst_dd = min(year_res[y]["max_dd"] for y in ["2023", "2024", "2025"])

            label = f"gap={gap}, cd={cd}"
            for y in ["2023", "2024", "2025"]:
                print_row(f"  {symbol} {label} | {y}", year_res[y])
            print(f"    -> avg_sharpe={avg_sharpe:.3f}, sum_ret={sum_ret:.2f}%, worst_dd={worst_dd:.2f}%")
            print()

            if avg_sharpe > best_sharpe:
                best_sharpe = avg_sharpe
                best_key = key

    best_yr = part1_results[best_key]
    best_by_symbol[symbol] = {
        "gap": best_key[1], "cooldown": best_key[2],
        "avg_sharpe": best_sharpe, "year_results": best_yr,
    }
    print(f"  >>> BEST for {symbol}: gap={best_key[1]}, cooldown={best_key[2]}, avg_sharpe={best_sharpe:.3f}")


# ══════════════════════════════════════════════════════════════════
# PART 2: MTF RSI2 SOL 15m - Asymmetric params
# ══════════════════════════════════════════════════════════════════

print_header("PART 2: MTF RSI2 SOL 15m - Asymmetric variations")

sol_variants = {
    "SOL baseline": deepcopy(SOL_PARAMS),
    "SOL long-only (no shorts)": {**deepcopy(SOL_PARAMS), "mtf_rsi_entry_short": 100.0},
    "SOL short-only (no longs)": {**deepcopy(SOL_PARAMS), "mtf_rsi_entry_long": 0.0},
    "SOL tighter short exit (25)": {**deepcopy(SOL_PARAMS), "mtf_rsi_exit_short": 25.0},
}

sol_variant_results = {}

for name, params in sol_variants.items():
    print(f"\n  --- {name} ---")
    sol_variant_results[name] = {}
    for year_label, start, end in YEARS:
        res = run_single("mtf_rsi2", "SOLUSDT", "15", start, end, params)
        print_row(f"{name} | {year_label}", res)
        sol_variant_results[name][year_label] = res

    avg_sharpe = sum(sol_variant_results[name][y]["sharpe"] for y in ["2023", "2024", "2025"]) / 3
    sum_ret = sum(sol_variant_results[name][y]["return_pct"] for y in ["2023", "2024", "2025"])
    print(f"    -> avg_sharpe={avg_sharpe:.3f}, sum_ret={sum_ret:.2f}%")


# ══════════════════════════════════════════════════════════════════
# PART 3: MACD ETH 4H backtest
# ══════════════════════════════════════════════════════════════════

print_header("PART 3a: MACD ADX ETH 4H backtest")

MACD_PARAMS = dict(
    macd_adx_threshold=25.0,
    macd_atr_sl_mult=4.0,
    macd_fast=26,
    macd_slow=52,
    macd_signal=9,
    macd_adx_period=14,
    macd_atr_period=14,
    macd_order_size_usd=3000,
)

macd_eth_results = {}
for year_label, start, end in YEARS:
    res = run_single("macd_adx", "ETHUSDT", "240", start, end, MACD_PARAMS)
    print_row(f"MACD ETH 4H | {year_label}", res)
    macd_eth_results[year_label] = res


# ══════════════════════════════════════════════════════════════════
# PART 3b: Portfolio simulation
# ══════════════════════════════════════════════════════════════════

print_header("PART 3b: Portfolio Simulation ($10,000 per strategy, equal weight)")

# Determine best SOL variant
best_sol_name = max(sol_variant_results.keys(),
                    key=lambda n: sum(sol_variant_results[n][y]["sharpe"] for y in ["2023", "2024", "2025"]))
best_sol = sol_variant_results[best_sol_name]

# Determine if any ETH/BTC MTF RSI2 result is worth including
portfolio_strategies = {
    f"MTF RSI2 SOL 15m ({best_sol_name})": best_sol,
    "MACD ETH 4H": macd_eth_results,
}

# Add best ETH MTF RSI2 if positive avg sharpe
eth_best = best_by_symbol.get("ETHUSDT", {})
if eth_best and eth_best.get("avg_sharpe", 0) > 0:
    portfolio_strategies[f"MTF RSI2 ETH 15m (gap={eth_best['gap']}, cd={eth_best['cooldown']})"] = eth_best["year_results"]

# Add best BTC MTF RSI2 if positive avg sharpe
btc_best = best_by_symbol.get("BTCUSDT", {})
if btc_best and btc_best.get("avg_sharpe", 0) > 0:
    portfolio_strategies[f"MTF RSI2 BTC 15m (gap={btc_best['gap']}, cd={btc_best['cooldown']})"] = btc_best["year_results"]

n_strats = len(portfolio_strategies)
capital_each = 10000.0

print(f"\n  Portfolio components ({n_strats} strategies, ${capital_each:.0f} each):")
for name in portfolio_strategies:
    print(f"    - {name}")

print(f"\n  {'Year':<8s} ", end="")
for name in portfolio_strategies:
    short_name = name[:30]
    print(f"| {short_name:>32s} ", end="")
print(f"| {'PORTFOLIO':>32s}")

print(f"  {'-'*8} ", end="")
for _ in range(n_strats + 1):
    print(f"| {'-'*32} ", end="")
print()

for year_label in ["2023", "2024", "2025"]:
    print(f"  {year_label:<8s} ", end="")
    returns = []
    for name, yr in portfolio_strategies.items():
        r = yr[year_label]["return_pct"]
        returns.append(r)
        short_name = name[:30]
        print(f"| {r:>30.2f}% ", end="")

    # Portfolio return = average of individual returns (equal capital)
    portfolio_ret = sum(returns) / len(returns)
    print(f"| {portfolio_ret:>30.2f}%")

print()

# Summary table with all metrics
print_header("SUMMARY: All strategies by year")
print(f"  {'Strategy':<50s} | {'Year':<6s} | {'Return%':>9s} | {'Sharpe':>7s} | {'MaxDD%':>8s} | {'Trades':>7s}")
print(f"  {'-'*50} | {'-'*6} | {'-'*9} | {'-'*7} | {'-'*8} | {'-'*7}")

all_strats_for_summary = {}

# SOL variants
for name, yr in sol_variant_results.items():
    all_strats_for_summary[name] = yr

# MACD ETH
all_strats_for_summary["MACD ETH 4H"] = macd_eth_results

# Best ETH/BTC MTF RSI2
for symbol in ["ETHUSDT", "BTCUSDT"]:
    best = best_by_symbol.get(symbol, {})
    if best:
        key = f"MTF RSI2 {symbol} 15m (gap={best['gap']}, cd={best['cooldown']})"
        all_strats_for_summary[key] = best["year_results"]

for name, yr in all_strats_for_summary.items():
    for year_label in ["2023", "2024", "2025"]:
        r = yr[year_label]
        print(f"  {name:<50s} | {year_label:<6s} | {r['return_pct']:>8.2f}% | {r['sharpe']:>7.3f} | {r['max_dd']:>7.2f}% | {r['n_trades']:>7d}")

# Portfolio combined annual returns
print(f"\n  {'PORTFOLIO (equal weight)':<50s}", end="")
print()
for year_label in ["2023", "2024", "2025"]:
    returns = [yr[year_label]["return_pct"] for yr in portfolio_strategies.values()]
    sharpes = [yr[year_label]["sharpe"] for yr in portfolio_strategies.values()]
    max_dds = [yr[year_label]["max_dd"] for yr in portfolio_strategies.values()]
    trades = [yr[year_label]["n_trades"] for yr in portfolio_strategies.values()]

    avg_ret = sum(returns) / len(returns)
    avg_sharpe = sum(sharpes) / len(sharpes)
    worst_dd = min(max_dds)
    total_trades = sum(trades)
    print(f"  {'  -> Portfolio':<50s} | {year_label:<6s} | {avg_ret:>8.2f}% | {avg_sharpe:>7.3f} | {worst_dd:>7.2f}% | {total_trades:>7d}")

print("\nDone.")
