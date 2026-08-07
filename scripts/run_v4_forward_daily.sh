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
} >> "$LOG" 2>&1
