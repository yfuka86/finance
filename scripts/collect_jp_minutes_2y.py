"""2年分(Premium add-on枠)の1分足を収集。冪等・既存raw再利用・単独実行。"""
import json
from trading.jp_intraday.collector import collect_jquants_minutes_bulk

SYMS = json.load(open("/private/tmp/claude-504/-Users-yutafukazawa-work-finance/e544c419-377a-4995-910e-05d123350fe2/scratchpad/minute_universe.json"))

if __name__ == "__main__":
    print(f"universe={len(SYMS)} symbols", flush=True)
    collect_jquants_minutes_bulk(SYMS, "2024-08-01", "2026-07-24", "data/jp_minutes_2y")
    print("DONE", flush=True)
