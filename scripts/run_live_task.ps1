# kabuステーション ライブボット タスクランナー (Windows タスクスケジューラ用)
# 使い方: powershell -ExecutionPolicy Bypass -File scripts\run_live_task.ps1 -Action plan
# 実行モードは .env の KABU_ENV / KABU_DRY_RUN / KABU_LIVE_CONFIRMED に従う
# (このスクリプト自体は何も上書きしない)。ログは data\live_reports\task_logs\ に残す。
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("collect", "plan", "entry", "exit", "state", "train", "preflight", "probe")]
    [string]$Action
)

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONPATH = "."
# タスクスケジューラ実行時は stdout が cp932 になり ¥ や絵文字の print で
# UnicodeEncodeError で落ちる (2026-07-28 の exit で実発生)。UTF-8 を強制する。
$env:PYTHONUTF8 = "1"

$logDir = Join-Path $root "data\live_reports\task_logs"
New-Item -ItemType Directory -Force $logDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$log = Join-Path $logDir "$($Action)_$stamp.log"

$py = Join-Path $root ".venv\Scripts\python.exe"

if ($Action -eq "collect") {
    # 寄付き前の日次データ更新 (冪等・不足日のみ取得)
    & $py scripts\collect_jp_daily_history.py *>> $log
} elseif ($Action -eq "probe") {
    # 発注経路プローブ (約定不能指値→即取消。口座抑止の検知)
    & $py scripts\preflight_order_probe.py *>> $log
} else {
    & $py -m trading.jp_intraday.live.run_live $Action *>> $log
}
$code = $LASTEXITCODE
"exit=$code time=$(Get-Date -Format o)" | Add-Content $log

# 失敗は必ず Slack に出す。python が起動しなかった場合も拾えるようランナー側で行う
# (正常時のイベント通知は python の notifier がイベントごとに送る)。
if ($code -ne 0) {
    $tail = (Get-Content $log -Tail 25 -Encoding UTF8 -ErrorAction SilentlyContinue) -join "`n"
    & (Join-Path $PSScriptRoot "notify_slack.ps1") -Title "$Action 失敗 (exit=$code)" -Detail $tail | Out-Null
}

# ログの肥大化防止: 30日より古いログを削除
Get-ChildItem $logDir -Filter *.log |
    Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-30) } |
    Remove-Item -Force -ErrorAction SilentlyContinue

exit $code
