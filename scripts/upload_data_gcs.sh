#!/bin/bash
# Mac側: Windows検証用データをGCSへ同期し、7日有効の署名付きURLを発行する。
# 認証: ~/.claude/.secrets/atokyo-data-sync.json（バケット限定SA。無ければ README参照）
set -euo pipefail
cd "$(dirname "$0")/.."

BUCKET=gs://atokyo-trade-data
KEY=~/.claude/.secrets/atokyo-data-sync.json
TAR=/tmp/jp_trading_data.tar.gz

echo "== tarball 作成 =="
tar czf "$TAR" \
  data/jp_daily_history data/jp_derivatives data/jp_intraday_reference \
  data/live_models $(ls data/cache/bars_day_*.parquet)
ls -lh "$TAR"

echo "== アップロード =="
gcloud storage cp "$TAR" "$BUCKET/jp_trading_data.tar.gz" \
  --project=atokyo-trade --impersonate-service-account="" 2>/dev/null || \
GOOGLE_APPLICATION_CREDENTIALS="$KEY" gcloud storage cp "$TAR" "$BUCKET/jp_trading_data.tar.gz" \
  --project=atokyo-trade

echo "== 署名付きURL（7日有効） =="
gcloud storage sign-url "$BUCKET/jp_trading_data.tar.gz" --duration=7d \
  --private-key-file="$KEY" --format="value(signed_url)"
