#!/usr/bin/env python3
"""
全プリセットの月別バックテストを一括実行し saved_results/ に保存。
上位(★)は 2023-01〜2026-03、その他は 2025-01〜2025-12。
"""
import sys, json, calendar
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trading.bybit.backtest import run_backtest
from trading.bybit.presets import STRATEGY_PRESETS, RESULTS_DIR

INITIAL = 10000.0
SLIP = 1.0

TOP_START, TOP_END = (2023, 1), (2026, 3)
OTHER_START, OTHER_END = (2025, 1), (2025, 12)

def mrange(sy, sm, ey, em):
    y, m = sy, sm
    while (y, m) <= (ey, em):
        yield y, m
        m += 1
        if m > 12: m = 1; y += 1

def rpath(pk, sym, iv, label):
    return RESULTS_DIR / f"{pk}__{sym}__{iv}__{label}.json"

def run_month(pk, preset, year, month):
    sym = preset["recommended_symbols"][0]
    iv = preset["recommended_interval"]
    ld = calendar.monthrange(year, month)[1]
    start = f"{year}-{month:02d}-01"
    end = f"{year}-{month:02d}-{ld}"
    label = f"{year}-{month:02d}"
    out = rpath(pk, sym, iv, label)
    if out.exists():
        return
    print(f"  {label} ...", end=" ", flush=True)
    try:
        r = run_backtest(strategy=preset["strategy"], symbol=sym, interval=iv,
                         start=start, end=end, initial_equity=INITIAL,
                         slippage_bps=SLIP, **preset["params"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"metrics": r.metrics, "preset_key": pk,
                                   "symbol": sym, "interval": iv, "label": label,
                                   "start": start, "end": end}, indent=2))
        m = r.metrics
        print(f"Ret={m.get('total_return_pct',0):+.2f}% T={m.get('n_trades',0)}")
    except Exception as e:
        print(f"FAIL: {e}")

def main():
    for i, (pk, p) in enumerate(STRATEGY_PRESETS.items()):
        top = p["name"].startswith("★")
        sy, sm = TOP_START if top else OTHER_START
        ey, em = TOP_END if top else OTHER_END
        months = list(mrange(sy, sm, ey, em))
        print(f"\n[{i+1}/{len(STRATEGY_PRESETS)}] {p['name']} ({len(months)} months)")
        for y, m in months:
            run_month(pk, p, y, m)
    print("\nDone.")

if __name__ == "__main__":
    main()
