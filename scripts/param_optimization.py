"""
Parameter optimization for MACD+ADX and Bollinger Band Reversion strategies.
One-dim-at-a-time sweep on 2024, then combine best and validate on 2023/2024/2025.
"""
import sys
import os
import logging
import itertools

# Suppress noisy logs
logging.basicConfig(level=logging.WARNING)
logging.getLogger("trading").setLevel(logging.WARNING)
logging.getLogger("pybit").setLevel(logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading.bybit.backtest import run_backtest

# ── Helpers ──────────────────────────────────────────────────────

def run_bt(strategy, symbol, interval, start, end, **params):
    """Run backtest, return metrics dict augmented with key info."""
    try:
        result = run_backtest(
            strategy=strategy, symbol=symbol, interval=interval,
            start=start, end=end, initial_equity=10000.0,
            slippage_bps=1.0, **params,
        )
        m = result.metrics.copy()
        m["strategy"] = strategy
        m["symbol"] = symbol
        m["interval"] = interval
        m["start"] = start
        m["end"] = end
        m["params"] = params
        return m
    except Exception as e:
        print(f"  ERROR: {strategy} {symbol} {interval} {start}-{end}: {e}")
        return None


def fmt_result(m, label=""):
    if m is None:
        return f"  {label}: ERROR"
    trades = m.get("n_trades", 0)
    fees = m.get("total_fees", 0)
    ret = m.get("total_return_pct", 0)
    sharpe = m.get("sharpe_ratio", 0)
    mdd = m.get("max_drawdown_pct", 0)
    pf = m.get("profit_factor", 0)
    wr = m.get("win_rate_pct", 0)
    return (f"  {label}: ret={ret:+.1f}%  sharpe={sharpe:.2f}  mdd={mdd:.1f}%  "
            f"trades={trades}  fees=${fees:.0f}  PF={pf:.2f}  WR={wr:.0f}%")


YEAR_RANGES = {
    2023: ("2023-01-01", "2023-12-31"),
    2024: ("2024-01-01", "2024-12-31"),
    2025: ("2025-01-01", "2025-12-31"),
}

# ═════════════════════════════════════════════════════════════════
# 1. MACD + ADX OPTIMIZATION
# ═════════════════════════════════════════════════════════════════

print("=" * 80)
print("  MACD + ADX PARAMETER OPTIMIZATION")
print("=" * 80)

MACD_SYMBOLS = ["ETHUSDT", "SOLUSDT", "BTCUSDT"]
MACD_INTERVALS = ["60", "240"]

MACD_BASE = dict(
    macd_fast=12, macd_slow=26, macd_signal=9,
    macd_adx_period=14, macd_adx_threshold=15.0,
    macd_atr_period=14, macd_atr_sl_mult=3.5,
    macd_order_size_usd=3000,
)

# ── Dim 1: ADX threshold sweep (2024) ──
print("\n--- MACD Dim 1: ADX Threshold Sweep (2024) ---")
adx_thresholds = [15, 20, 25, 30, 35]
best_adx = {}

for sym in MACD_SYMBOLS:
    for intv in MACD_INTERVALS:
        print(f"\n{sym} {intv}:")
        best_sharpe = -999
        for adx_t in adx_thresholds:
            params = {**MACD_BASE, "macd_adx_threshold": float(adx_t)}
            m = run_bt("macd_adx", sym, intv, "2024-01-01", "2024-12-31", **params)
            print(fmt_result(m, f"adx_thresh={adx_t:2d}"))
            if m and m.get("sharpe_ratio", -999) > best_sharpe:
                best_sharpe = m["sharpe_ratio"]
                best_adx[(sym, intv)] = adx_t

print("\nBest ADX thresholds:", best_adx)

# ── Dim 2: ATR SL multiplier sweep (2024) ──
print("\n--- MACD Dim 2: ATR SL Multiplier Sweep (2024) ---")
atr_sl_mults = [2.0, 3.0, 3.5, 4.0, 5.0]
best_atr_sl = {}

for sym in MACD_SYMBOLS:
    for intv in MACD_INTERVALS:
        print(f"\n{sym} {intv} (adx_thresh={best_adx.get((sym,intv), 15)}):")
        best_sharpe = -999
        adx_t = best_adx.get((sym, intv), 15)
        for atr_m in atr_sl_mults:
            params = {**MACD_BASE, "macd_adx_threshold": float(adx_t),
                      "macd_atr_sl_mult": atr_m}
            m = run_bt("macd_adx", sym, intv, "2024-01-01", "2024-12-31", **params)
            print(fmt_result(m, f"atr_sl={atr_m:.1f}"))
            if m and m.get("sharpe_ratio", -999) > best_sharpe:
                best_sharpe = m["sharpe_ratio"]
                best_atr_sl[(sym, intv)] = atr_m

print("\nBest ATR SL mults:", best_atr_sl)

# ── Dim 3: MACD period sweep (2024) ──
print("\n--- MACD Dim 3: MACD Period Sweep (2024) ---")
macd_periods = [(12, 26), (16, 36), (20, 50), (26, 52)]
best_macd_period = {}

for sym in MACD_SYMBOLS:
    for intv in MACD_INTERVALS:
        adx_t = best_adx.get((sym, intv), 15)
        atr_m = best_atr_sl.get((sym, intv), 3.5)
        print(f"\n{sym} {intv} (adx={adx_t}, atr_sl={atr_m}):")
        best_sharpe = -999
        for fast, slow in macd_periods:
            params = {**MACD_BASE, "macd_adx_threshold": float(adx_t),
                      "macd_atr_sl_mult": atr_m,
                      "macd_fast": fast, "macd_slow": slow}
            m = run_bt("macd_adx", sym, intv, "2024-01-01", "2024-12-31", **params)
            print(fmt_result(m, f"fast={fast:2d},slow={slow:2d}"))
            if m and m.get("sharpe_ratio", -999) > best_sharpe:
                best_sharpe = m["sharpe_ratio"]
                best_macd_period[(sym, intv)] = (fast, slow)

print("\nBest MACD periods:", best_macd_period)

# ── MACD: Full 3-year validation with best combined params ──
print("\n" + "=" * 80)
print("  MACD+ADX: 3-YEAR VALIDATION (BEST COMBINED PARAMS)")
print("=" * 80)

macd_all_results = []

for sym in MACD_SYMBOLS:
    for intv in MACD_INTERVALS:
        adx_t = best_adx.get((sym, intv), 15)
        atr_m = best_atr_sl.get((sym, intv), 3.5)
        fast, slow = best_macd_period.get((sym, intv), (12, 26))
        params = {**MACD_BASE, "macd_adx_threshold": float(adx_t),
                  "macd_atr_sl_mult": atr_m,
                  "macd_fast": fast, "macd_slow": slow}
        print(f"\n{sym} {intv}  [adx={adx_t}, atr_sl={atr_m}, fast={fast}, slow={slow}]")
        year_results = []
        for year, (s, e) in YEAR_RANGES.items():
            m = run_bt("macd_adx", sym, intv, s, e, **params)
            print(fmt_result(m, f"  {year}"))
            if m:
                m["year"] = year
                m["config_label"] = f"macd_adx|{sym}|{intv}|adx{adx_t}_atr{atr_m}_f{fast}s{slow}"
                year_results.append(m)
                macd_all_results.append(m)

        # Aggregate 3-year
        if len(year_results) == 3:
            avg_ret = sum(r["total_return_pct"] for r in year_results) / 3
            avg_sharpe = sum(r["sharpe_ratio"] for r in year_results) / 3
            worst_dd = min(r["max_drawdown_pct"] for r in year_results)
            total_trades = sum(r["n_trades"] for r in year_results)
            total_fees = sum(r.get("total_fees", 0) for r in year_results)
            print(f"  3Y AVG: ret={avg_ret:+.1f}%  sharpe={avg_sharpe:.2f}  "
                  f"worst_dd={worst_dd:.1f}%  total_trades={total_trades}  total_fees=${total_fees:.0f}")


# ═════════════════════════════════════════════════════════════════
# 2. BOLLINGER BAND REVERSION OPTIMIZATION
# ═════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 80)
print("  BOLLINGER BAND REVERSION PARAMETER OPTIMIZATION")
print("=" * 80)

BB_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
BB_INTERVALS = ["60", "240"]

BB_BASE = dict(
    bb_period=20, bb_std_mult=2.0,
    bb_adx_period=14, bb_adx_threshold=25.0,
    bb_order_size_usd=5000,
)

# ── Dim 1: BB period sweep ──
print("\n--- BB Dim 1: Period Sweep (2024) ---")
bb_periods = [15, 20, 30, 40]
best_bb_period = {}

for sym in BB_SYMBOLS:
    for intv in BB_INTERVALS:
        print(f"\n{sym} {intv}:")
        best_sharpe = -999
        for per in bb_periods:
            params = {**BB_BASE, "bb_period": per}
            m = run_bt("bollinger_reversion", sym, intv, "2024-01-01", "2024-12-31", **params)
            print(fmt_result(m, f"period={per:2d}"))
            if m and m.get("sharpe_ratio", -999) > best_sharpe:
                best_sharpe = m["sharpe_ratio"]
                best_bb_period[(sym, intv)] = per

print("\nBest BB periods:", best_bb_period)

# ── Dim 2: Std mult sweep ──
print("\n--- BB Dim 2: Std Multiplier Sweep (2024) ---")
bb_std_mults = [1.5, 2.0, 2.5, 3.0]
best_bb_std = {}

for sym in BB_SYMBOLS:
    for intv in BB_INTERVALS:
        per = best_bb_period.get((sym, intv), 20)
        print(f"\n{sym} {intv} (period={per}):")
        best_sharpe = -999
        for std_m in bb_std_mults:
            params = {**BB_BASE, "bb_period": per, "bb_std_mult": std_m}
            m = run_bt("bollinger_reversion", sym, intv, "2024-01-01", "2024-12-31", **params)
            print(fmt_result(m, f"std={std_m:.1f}"))
            if m and m.get("sharpe_ratio", -999) > best_sharpe:
                best_sharpe = m["sharpe_ratio"]
                best_bb_std[(sym, intv)] = std_m

print("\nBest BB std mults:", best_bb_std)

# ── Dim 3: ADX threshold sweep ──
print("\n--- BB Dim 3: ADX Threshold Sweep (2024) ---")
bb_adx_thresholds = [20, 25, 30, 35]
best_bb_adx = {}

for sym in BB_SYMBOLS:
    for intv in BB_INTERVALS:
        per = best_bb_period.get((sym, intv), 20)
        std_m = best_bb_std.get((sym, intv), 2.0)
        print(f"\n{sym} {intv} (period={per}, std={std_m}):")
        best_sharpe = -999
        for adx_t in bb_adx_thresholds:
            params = {**BB_BASE, "bb_period": per, "bb_std_mult": std_m,
                      "bb_adx_threshold": float(adx_t)}
            m = run_bt("bollinger_reversion", sym, intv, "2024-01-01", "2024-12-31", **params)
            print(fmt_result(m, f"adx_thresh={adx_t:2d}"))
            if m and m.get("sharpe_ratio", -999) > best_sharpe:
                best_sharpe = m["sharpe_ratio"]
                best_bb_adx[(sym, intv)] = adx_t

print("\nBest BB ADX thresholds:", best_bb_adx)

# ── BB: Full 3-year validation ──
print("\n" + "=" * 80)
print("  BOLLINGER REVERSION: 3-YEAR VALIDATION (BEST COMBINED PARAMS)")
print("=" * 80)

bb_all_results = []

for sym in BB_SYMBOLS:
    for intv in BB_INTERVALS:
        per = best_bb_period.get((sym, intv), 20)
        std_m = best_bb_std.get((sym, intv), 2.0)
        adx_t = best_bb_adx.get((sym, intv), 25)
        params = {**BB_BASE, "bb_period": per, "bb_std_mult": std_m,
                  "bb_adx_threshold": float(adx_t)}
        print(f"\n{sym} {intv}  [period={per}, std={std_m}, adx={adx_t}]")
        year_results = []
        for year, (s, e) in YEAR_RANGES.items():
            m = run_bt("bollinger_reversion", sym, intv, s, e, **params)
            print(fmt_result(m, f"  {year}"))
            if m:
                m["year"] = year
                m["config_label"] = f"bb_rev|{sym}|{intv}|p{per}_s{std_m}_a{adx_t}"
                year_results.append(m)
                bb_all_results.append(m)

        if len(year_results) == 3:
            avg_ret = sum(r["total_return_pct"] for r in year_results) / 3
            avg_sharpe = sum(r["sharpe_ratio"] for r in year_results) / 3
            worst_dd = min(r["max_drawdown_pct"] for r in year_results)
            total_trades = sum(r["n_trades"] for r in year_results)
            total_fees = sum(r.get("total_fees", 0) for r in year_results)
            print(f"  3Y AVG: ret={avg_ret:+.1f}%  sharpe={avg_sharpe:.2f}  "
                  f"worst_dd={worst_dd:.1f}%  total_trades={total_trades}  total_fees=${total_fees:.0f}")


# ═════════════════════════════════════════════════════════════════
# 3. TOP 5 CONFIGS ACROSS ALL STRATEGIES (3-YEAR PERFORMANCE)
# ═════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 80)
print("  TOP 5 CONFIGS: 3-YEAR AGGREGATE PERFORMANCE")
print("=" * 80)

# Group by config_label, compute 3-year aggregate
all_results = macd_all_results + bb_all_results
config_groups = {}
for r in all_results:
    label = r.get("config_label", "?")
    config_groups.setdefault(label, []).append(r)

rankings = []
for label, results in config_groups.items():
    if len(results) < 3:
        continue
    avg_ret = sum(r["total_return_pct"] for r in results) / len(results)
    avg_sharpe = sum(r["sharpe_ratio"] for r in results) / len(results)
    worst_dd = min(r["max_drawdown_pct"] for r in results)
    total_trades = sum(r["n_trades"] for r in results)
    total_fees = sum(r.get("total_fees", 0) for r in results)
    cum_ret = 1.0
    for r in sorted(results, key=lambda x: x["year"]):
        cum_ret *= (1 + r["total_return_pct"] / 100)
    cum_ret_pct = (cum_ret - 1) * 100
    # Consistency: number of profitable years
    profitable_years = sum(1 for r in results if r["total_return_pct"] > 0)

    rankings.append({
        "label": label,
        "avg_ret": avg_ret,
        "cum_ret": cum_ret_pct,
        "avg_sharpe": avg_sharpe,
        "worst_dd": worst_dd,
        "total_trades": total_trades,
        "total_fees": total_fees,
        "profitable_years": profitable_years,
        "years": {r["year"]: r["total_return_pct"] for r in results},
    })

# Sort by average Sharpe (risk-adjusted)
rankings.sort(key=lambda x: x["avg_sharpe"], reverse=True)

for i, r in enumerate(rankings[:5], 1):
    print(f"\n#{i}  {r['label']}")
    print(f"     3Y cumulative return: {r['cum_ret']:+.1f}%  (avg/yr: {r['avg_ret']:+.1f}%)")
    print(f"     Avg Sharpe: {r['avg_sharpe']:.2f}  Worst DD: {r['worst_dd']:.1f}%")
    print(f"     Total trades: {r['total_trades']}  Total fees: ${r['total_fees']:.0f}")
    print(f"     Profitable years: {r['profitable_years']}/3")
    for yr in sorted(r["years"]):
        print(f"       {yr}: {r['years'][yr]:+.1f}%")

if not rankings:
    print("\n  No complete 3-year results available.")

print("\n" + "=" * 80)
print("  OPTIMIZATION COMPLETE")
print("=" * 80)
