"""
Optimization script for choppy/bearish 2025 market strategies.

Tests:
1. Dual Regime (ADX-switched Donchian/BB)
2. Mean Reversion Filtered (BB + RSI + volume)
3. RSI(2) standalone reversion

Sweeps params on 2025, validates best on 2023-2024.
"""
import sys
import os
import itertools
import logging
from datetime import datetime

# Ensure project root on path
sys.path.insert(0, "/Users/yutafukazawa/work/finance")

from trading.bybit.backtest import run_backtest

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def fmt_metrics(m: dict) -> str:
    return (
        f"Ret={m.get('total_return_pct', 0):+.2f}%  "
        f"Sharpe={m.get('sharpe_ratio', 0):.3f}  "
        f"MaxDD={m.get('max_drawdown_pct', 0):.2f}%  "
        f"Trades={m.get('n_trades', 0)}  "
        f"WR={m.get('win_rate_pct', 0):.1f}%  "
        f"PF={m.get('profit_factor', 0):.2f}"
    )


def run_one(strategy, symbol, interval, start, end, **params):
    """Run a single backtest, return metrics dict (or empty on error)."""
    try:
        result = run_backtest(
            strategy=strategy, symbol=symbol, interval=interval,
            start=start, end=end, initial_equity=10000.0,
            slippage_bps=1.0, **params
        )
        return result.metrics
    except Exception as e:
        print(f"  ERROR: {e}")
        return {}


# ====================================================================
#  1. DUAL REGIME OPTIMIZATION
# ====================================================================
def optimize_dual_regime():
    print("=" * 80)
    print("  1. DUAL REGIME OPTIMIZATION")
    print("=" * 80)

    symbols = ["BTCUSDT", "ETHUSDT"]
    intervals = ["60", "240"]

    adx_thresholds = [20, 25, 30]
    atr_sl_mults = [3.0, 4.0, 5.0]
    bb_std_mults = [2.0, 2.5, 3.0]
    don_entry_periods = [20, 30]

    combos = list(itertools.product(adx_thresholds, atr_sl_mults, bb_std_mults, don_entry_periods))
    print(f"\nParam combos: {len(combos)} x {len(symbols)} symbols x {len(intervals)} intervals = {len(combos)*len(symbols)*len(intervals)} runs")
    print(f"Start time: {datetime.now().strftime('%H:%M:%S')}\n")

    results_2025 = []

    for symbol in symbols:
        for interval in intervals:
            print(f"\n--- {symbol} / {interval}m ---")
            for adx_th, atr_sl, bb_std, don_ep in combos:
                params = {
                    "dr_adx_threshold": adx_th,
                    "dr_atr_sl_mult": atr_sl,
                    "dr_bb_std_mult": bb_std,
                    "dr_don_entry_period": don_ep,
                    "dr_order_size_usd": 5000,
                    "dr_don_exit_period": 10,
                    "dr_bb_period": 20,
                    "dr_adx_period": 14,
                    "dr_atr_period": 14,
                    "dr_vol_ratio_limit": 1.5,
                }
                label = f"adx={adx_th} atr_sl={atr_sl} bb_std={bb_std} don_ep={don_ep}"
                m = run_one("dual_regime", symbol, interval, "2025-01-01", "2025-12-31", **params)
                if m:
                    ret = m.get("total_return_pct", -999)
                    print(f"  {label}: {fmt_metrics(m)}")
                    results_2025.append({
                        "symbol": symbol, "interval": interval,
                        "params": params, "label": label, "metrics": m,
                        "ret": ret, "sharpe": m.get("sharpe_ratio", 0),
                    })

    # Sort by return
    results_2025.sort(key=lambda x: x["ret"], reverse=True)

    print("\n" + "=" * 80)
    print("  DUAL REGIME - TOP 10 by 2025 Return")
    print("=" * 80)
    for i, r in enumerate(results_2025[:10]):
        print(f"  #{i+1}: {r['symbol']}/{r['interval']}m  {r['label']}")
        print(f"       {fmt_metrics(r['metrics'])}")

    # Validate top 5 on 2023-2024
    print("\n" + "-" * 80)
    print("  DUAL REGIME - Validation (2023-2024) for top 5")
    print("-" * 80)
    validated = []
    for r in results_2025[:5]:
        if r["ret"] <= 0:
            continue
        m_val = run_one("dual_regime", r["symbol"], r["interval"],
                        "2023-01-01", "2024-12-31", **r["params"])
        if m_val:
            val_ret = m_val.get("total_return_pct", -999)
            print(f"  {r['symbol']}/{r['interval']}m  {r['label']}")
            print(f"    2025: {fmt_metrics(r['metrics'])}")
            print(f"    2023-2024: {fmt_metrics(m_val)}")
            validated.append({
                **r, "val_metrics": m_val, "val_ret": val_ret,
            })

    return results_2025, validated


# ====================================================================
#  2. MEAN REVERSION FILTERED OPTIMIZATION
# ====================================================================
def optimize_mrf():
    print("\n" + "=" * 80)
    print("  2. MEAN REVERSION FILTERED OPTIMIZATION")
    print("=" * 80)

    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    intervals = ["60", "240"]

    bb_mults = [1.5, 2.0, 2.5]
    rsi_oversolds = [20, 25, 30]
    vol_mults = [1.0, 1.5, 2.0]
    atr_sl_mults = [1.5, 2.0, 3.0]

    combos = list(itertools.product(bb_mults, rsi_oversolds, vol_mults, atr_sl_mults))
    print(f"\nParam combos: {len(combos)} x {len(symbols)} symbols x {len(intervals)} intervals = {len(combos)*len(symbols)*len(intervals)} runs")
    print(f"Start time: {datetime.now().strftime('%H:%M:%S')}\n")

    results_2025 = []

    for symbol in symbols:
        for interval in intervals:
            print(f"\n--- {symbol} / {interval}m ---")
            for bb_m, rsi_os, vol_m, atr_sl in combos:
                params = {
                    "mrf_bb_mult": bb_m,
                    "mrf_rsi_oversold": rsi_os,
                    "mrf_rsi_overbought": 100 - rsi_os,  # symmetric
                    "mrf_vol_mult": vol_m,
                    "mrf_atr_sl_mult": atr_sl,
                    "mrf_bb_period": 20,
                    "mrf_rsi_period": 14,
                    "mrf_order_size_usd": 3000,
                }
                label = f"bb={bb_m} rsi_os={rsi_os} vol={vol_m} atr_sl={atr_sl}"
                m = run_one("mean_reversion_filtered", symbol, interval,
                            "2025-01-01", "2025-12-31", **params)
                if m:
                    ret = m.get("total_return_pct", -999)
                    print(f"  {label}: {fmt_metrics(m)}")
                    results_2025.append({
                        "symbol": symbol, "interval": interval,
                        "params": params, "label": label, "metrics": m,
                        "ret": ret, "sharpe": m.get("sharpe_ratio", 0),
                    })

    results_2025.sort(key=lambda x: x["ret"], reverse=True)

    print("\n" + "=" * 80)
    print("  MRF - TOP 10 by 2025 Return")
    print("=" * 80)
    for i, r in enumerate(results_2025[:10]):
        print(f"  #{i+1}: {r['symbol']}/{r['interval']}m  {r['label']}")
        print(f"       {fmt_metrics(r['metrics'])}")

    # Validate top 5 on 2023-2024
    print("\n" + "-" * 80)
    print("  MRF - Validation (2023-2024) for top 5")
    print("-" * 80)
    validated = []
    for r in results_2025[:5]:
        if r["ret"] <= 0:
            continue
        m_val = run_one("mean_reversion_filtered", r["symbol"], r["interval"],
                        "2023-01-01", "2024-12-31", **r["params"])
        if m_val:
            val_ret = m_val.get("total_return_pct", -999)
            print(f"  {r['symbol']}/{r['interval']}m  {r['label']}")
            print(f"    2025: {fmt_metrics(r['metrics'])}")
            print(f"    2023-2024: {fmt_metrics(m_val)}")
            validated.append({
                **r, "val_metrics": m_val, "val_ret": val_ret,
            })

    return results_2025, validated


# ====================================================================
#  3. RSI(2) STANDALONE OPTIMIZATION
# ====================================================================
def optimize_rsi2():
    print("\n" + "=" * 80)
    print("  3. RSI(2) STANDALONE OPTIMIZATION")
    print("=" * 80)

    symbols = ["SOLUSDT", "BTCUSDT"]
    intervals = ["15", "60"]

    rsi_oversolds = [5, 10, 15]
    rsi_overboughts = [85, 90, 95]
    rsi_exit_levels = [50, 55, 60]

    combos = list(itertools.product(rsi_oversolds, rsi_overboughts, rsi_exit_levels))
    print(f"\nParam combos: {len(combos)} x {len(symbols)} symbols x {len(intervals)} intervals = {len(combos)*len(symbols)*len(intervals)} runs")
    print(f"Start time: {datetime.now().strftime('%H:%M:%S')}\n")

    results_2025 = []

    for symbol in symbols:
        for interval in intervals:
            print(f"\n--- {symbol} / {interval}m ---")
            for rsi_os, rsi_ob, rsi_ex in combos:
                params = {
                    "rsi_period": 2,
                    "rsi_oversold": rsi_os,
                    "rsi_overbought": rsi_ob,
                    "rsi_exit_level": rsi_ex,
                    "rsi_order_size_usd": 3000,
                }
                label = f"os={rsi_os} ob={rsi_ob} exit={rsi_ex}"
                m = run_one("rsi_reversion", symbol, interval,
                            "2025-01-01", "2025-12-31", **params)
                if m:
                    ret = m.get("total_return_pct", -999)
                    print(f"  {label}: {fmt_metrics(m)}")
                    results_2025.append({
                        "symbol": symbol, "interval": interval,
                        "params": params, "label": label, "metrics": m,
                        "ret": ret, "sharpe": m.get("sharpe_ratio", 0),
                    })

    results_2025.sort(key=lambda x: x["ret"], reverse=True)

    print("\n" + "=" * 80)
    print("  RSI(2) - TOP 10 by 2025 Return")
    print("=" * 80)
    for i, r in enumerate(results_2025[:10]):
        print(f"  #{i+1}: {r['symbol']}/{r['interval']}m  {r['label']}")
        print(f"       {fmt_metrics(r['metrics'])}")

    # Validate top 5 on 2023-2024
    print("\n" + "-" * 80)
    print("  RSI(2) - Validation (2023-2024) for top 5")
    print("-" * 80)
    validated = []
    for r in results_2025[:5]:
        if r["ret"] <= 0:
            continue
        m_val = run_one("rsi_reversion", r["symbol"], r["interval"],
                        "2023-01-01", "2024-12-31", **r["params"])
        if m_val:
            val_ret = m_val.get("total_return_pct", -999)
            print(f"  {r['symbol']}/{r['interval']}m  {r['label']}")
            print(f"    2025: {fmt_metrics(r['metrics'])}")
            print(f"    2023-2024: {fmt_metrics(m_val)}")
            validated.append({
                **r, "val_metrics": m_val, "val_ret": val_ret,
            })

    return results_2025, validated


# ====================================================================
#  MAIN
# ====================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("  CHOPPY/BEARISH MARKET STRATEGY OPTIMIZATION")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # 1. Dual Regime
    dr_results, dr_validated = optimize_dual_regime()

    # 2. Mean Reversion Filtered
    mrf_results, mrf_validated = optimize_mrf()

    # 3. RSI(2)
    rsi_results, rsi_validated = optimize_rsi2()

    # ================================================================
    #  FINAL SUMMARY
    # ================================================================
    print("\n" + "=" * 80)
    print("  FINAL SUMMARY - Strategies profitable in BOTH 2025 AND 2023-2024")
    print("=" * 80)

    all_validated = []
    for tag, vlist in [("DualRegime", dr_validated), ("MRF", mrf_validated), ("RSI2", rsi_validated)]:
        for v in vlist:
            if v["ret"] > 0 and v.get("val_ret", -999) > 0:
                all_validated.append((tag, v))

    if not all_validated:
        print("\n  *** NO strategy was profitable in both periods. ***")
        print("  Showing best 2025 performers from each category:\n")
        for tag, results in [("DualRegime", dr_results), ("MRF", mrf_results), ("RSI2", rsi_results)]:
            if results:
                best = results[0]
                print(f"  {tag} best 2025: {best['symbol']}/{best['interval']}m  {best['label']}")
                print(f"    {fmt_metrics(best['metrics'])}")
    else:
        all_validated.sort(key=lambda x: x[1]["ret"] + x[1].get("val_ret", 0), reverse=True)
        for i, (tag, v) in enumerate(all_validated):
            print(f"\n  #{i+1} [{tag}] {v['symbol']}/{v['interval']}m")
            print(f"    Params: {v['label']}")
            print(f"    2025:      {fmt_metrics(v['metrics'])}")
            print(f"    2023-2024: {fmt_metrics(v['val_metrics'])}")

    print(f"\n  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
