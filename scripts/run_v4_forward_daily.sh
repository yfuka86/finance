#!/bin/zsh
# V4 forward-seal daily feed: bars -> fins -> event ledger (SOLO, sequential).
set -e
cd /Users/yutafukazawa/work/finance
export PYTHONPATH=. PYTHONUTF8=1
LOG=data/value_event_v4_forward/collect.log
{
  echo "=== $(date -u +%FT%TZ) ==="
  python3 scripts/collect_jp_daily_history.py
  python3 scripts/collect_fins_incremental.py
  python3 scripts/collect_v4_forward_events.py
  # EDINET(別API・J-Quants 429と無関係)。大量保有は縦覧5年ローリングのため毎日追記
  python3 scripts/collect_edinet_large_holdings.py
} >> "$LOG" 2>&1
