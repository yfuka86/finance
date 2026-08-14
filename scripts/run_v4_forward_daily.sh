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
  # X11封印フォワード台帳(シグナルのみ・no-peek)
  python3 scripts/collect_x11_forward.py
  # 引けオークション反転: 分足を日次前進収集 → 封印台帳(シグナルのみ) → 明細
  python3 scripts/collect_jp_minutes_daily.py || echo "minute collect skipped"
  python3 scripts/collect_close_auction_forward.py || echo "close-auction ledger skipped"
  python3 scripts/export_close_auction_detail.py || echo "close-auction export skipped"
  python3 scripts/export_oversold_detail.py
  python3 scripts/build_finance_site.py
} >> "$LOG" 2>&1
