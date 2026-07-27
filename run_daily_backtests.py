"""
Run all 9 strategy presets on DAILY (D) timeframe across 10 symbols for 2023-2025.
Save individual results + compiled summary.
"""
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from trading.bybit.backtest import run_backtest
from trading.bybit.presets import STRATEGY_PRESETS, RESULTS_DIR, save_result

SYMBOLS = [
    "ADAUSDT", "AVAXUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT",
    "DOTUSDT", "ETHUSDT", "LINKUSDT", "SOLUSDT", "XRPUSDT",
]

PERIODS = [
    ("2023", "2023-01-01", "2023-12-31"),
    ("2024", "2024-01-01", "2024-12-31"),
    ("2025", "2025-01-01", "2025-04-12"),
]

INTERVAL = "D"
INITIAL_EQUITY = 10000.0
SLIPPAGE_BPS = 1.0


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    preset_keys = list(STRATEGY_PRESETS.keys())
    total_runs = len(preset_keys) * len(SYMBOLS) * len(PERIODS)
    print(f"=== Daily Backtest: {len(preset_keys)} presets x {len(SYMBOLS)} symbols x {len(PERIODS)} years = {total_runs} runs ===\n")

    # Collect all results: {preset_key: {symbol: {year_label: return_pct, ...}}}
    all_results = {}
    errors = []
    run_count = 0

    for pk in preset_keys:
        preset = STRATEGY_PRESETS[pk]
        strategy = preset["strategy"]
        params = preset["params"]
        daily_key = f"{pk}_daily"

        if daily_key not in all_results:
            all_results[daily_key] = {}

        for symbol in SYMBOLS:
            if symbol not in all_results[daily_key]:
                all_results[daily_key][symbol] = {}

            yearly_returns = {}

            for year_label, start, end in PERIODS:
                run_count += 1
                result_key = f"{pk}_daily__{symbol}__D__{start}__{end}"

                # Check if already saved
                result_path = RESULTS_DIR / f"{result_key}.json"
                if result_path.exists():
                    try:
                        saved = json.loads(result_path.read_text())
                        ret = saved["metrics"]["total_return_pct"]
                        yearly_returns[year_label] = ret
                        all_results[daily_key][symbol][year_label] = ret
                        print(f"  [{run_count}/{total_runs}] CACHED {pk} | {symbol} | {year_label}: {ret:+.1f}%")
                        continue
                    except Exception:
                        pass  # re-run if cached file is corrupt

                print(f"  [{run_count}/{total_runs}] {pk} | {symbol} | {start}~{end} ...", end=" ", flush=True)

                try:
                    result = run_backtest(
                        strategy=strategy,
                        symbol=symbol,
                        interval=INTERVAL,
                        start=start,
                        end=end,
                        initial_equity=INITIAL_EQUITY,
                        slippage_bps=SLIPPAGE_BPS,
                        **params,
                    )

                    ret = result.metrics.get("total_return_pct", 0.0)
                    yearly_returns[year_label] = ret

                    # Save individual result
                    metrics_serializable = dict(result.metrics)
                    config_serializable = dict(result.config)
                    save_data = {
                        "_key": result_key,
                        "metrics": metrics_serializable,
                        "config": config_serializable,
                    }
                    result_path.write_text(json.dumps(save_data, indent=2, ensure_ascii=False, default=str))

                    all_results[daily_key][symbol][year_label] = ret
                    print(f"{ret:+.1f}% (trades: {result.metrics.get('n_trades', 0)})")

                    # Small delay to be kind to the API
                    time.sleep(0.3)

                except Exception as e:
                    err_msg = f"{pk} | {symbol} | {year_label}: {e}"
                    errors.append(err_msg)
                    print(f"ERROR: {e}")
                    traceback.print_exc()
                    yearly_returns[year_label] = None
                    all_results[daily_key][symbol][year_label] = None

            # Compute 3yr and Sharpe approximation
            returns = [yearly_returns.get(y) for y in ["2023", "2024", "2025"]]
            valid = [r for r in returns if r is not None]
            if valid:
                total_3yr = sum(valid)
                avg = sum(valid) / len(valid)
                std = (sum((r - avg) ** 2 for r in valid) / len(valid)) ** 0.5 if len(valid) > 1 else 0
                sr = round(avg / std, 2) if std > 0 else 0.0
                all_results[daily_key][symbol]["3yr"] = round(total_3yr, 1)
                all_results[daily_key][symbol]["sr"] = sr
            else:
                all_results[daily_key][symbol]["3yr"] = None
                all_results[daily_key][symbol]["sr"] = None

    # Save validated_results_daily.json
    output_path = Path(__file__).parent / "trading" / "bybit" / "validated_results_daily.json"
    output_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
    print(f"\n=== Saved: {output_path} ===\n")

    # Print summary table
    print("=" * 140)
    print(f"{'Preset':<25} {'Symbol':<12} {'2023':>8} {'2024':>8} {'2025':>8} {'3yr':>8} {'SR':>6}")
    print("-" * 140)
    for pk_daily, symbols_data in all_results.items():
        for symbol, data in symbols_data.items():
            y23 = data.get("2023")
            y24 = data.get("2024")
            y25 = data.get("2025")
            yr3 = data.get("3yr")
            sr = data.get("sr")
            print(f"{pk_daily:<25} {symbol:<12} "
                  f"{(f'{y23:+.1f}%' if y23 is not None else 'ERR'):>8} "
                  f"{(f'{y24:+.1f}%' if y24 is not None else 'ERR'):>8} "
                  f"{(f'{y25:+.1f}%' if y25 is not None else 'ERR'):>8} "
                  f"{(f'{yr3:+.1f}%' if yr3 is not None else 'ERR'):>8} "
                  f"{(f'{sr:.2f}' if sr is not None else 'ERR'):>6}")
        print("-" * 140)

    # Print best combos
    print("\n=== TOP 10 by 3yr return ===")
    flat = []
    for pk_daily, symbols_data in all_results.items():
        for symbol, data in symbols_data.items():
            yr3 = data.get("3yr")
            if yr3 is not None:
                flat.append((pk_daily, symbol, yr3, data.get("sr", 0)))
    flat.sort(key=lambda x: x[2], reverse=True)
    for i, (pk, sym, yr3, sr) in enumerate(flat[:10]):
        print(f"  {i+1}. {pk} / {sym}: 3yr={yr3:+.1f}%, SR={sr:.2f}")

    print(f"\n=== TOP 10 by Sharpe Ratio ===")
    flat.sort(key=lambda x: x[3] if x[3] is not None else -999, reverse=True)
    for i, (pk, sym, yr3, sr) in enumerate(flat[:10]):
        print(f"  {i+1}. {pk} / {sym}: SR={sr:.2f}, 3yr={yr3:+.1f}%")

    if errors:
        print(f"\n=== {len(errors)} ERRORS ===")
        for e in errors:
            print(f"  - {e}")

    print(f"\nDone. {run_count} runs completed.")


if __name__ == "__main__":
    main()
