# kabuステーション ライブボット タスクランナー (Windows タスクスケジューラ用)
# 使い方: powershell -ExecutionPolicy Bypass -File scripts\run_live_task.ps1 -Action plan
# 実行モードは .env の KABU_ENV / KABU_DRY_RUN / KABU_LIVE_CONFIRMED に従う
# (このスクリプト自体は何も上書きしない)。ログは data\live_reports\task_logs\ に残す。
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("collect", "plan", "entry", "exit", "state", "train", "preflight", "probe",
        "quotesnap", "cost", "shadow", "pushexp")]
    [string]$Action
)

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONPATH = "."
# タスクスケジューラ実行時は stdout が cp932 になり ¥ や絵文字の print で
# UnicodeEncodeError で落ちる (2026-07-28 の exit で実発生)。UTF-8 を強制する。
$env:PYTHONUTF8 = "1"
# 出力をブロックバッファさせない。板取得の進捗がリアルタイムでログに出ないと、
# 遅いのか固まったのかを運用中に判別できない (2026-07-30 の実障害で判明)。
$env:PYTHONUNBUFFERED = "1"
# python は PYTHONUTF8=1 で UTF-8 を出すが、PS5.1 の既定デコードは cp932。
# そのままだと**日本語が壊れた状態でログに保存される**（2026-08-03 発覚。
# 「蜿門ｾ・661/686驫俶氛」のような化け方をし、ログの検索・Slack要約が機能しない）。
[Console]::OutputEncoding = [Text.Encoding]::UTF8

$logDir = Join-Path $root "data\live_reports\task_logs"
New-Item -ItemType Directory -Force $logDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$log = Join-Path $logDir "$($Action)_$stamp.log"

$py = Join-Path $root ".venv\Scripts\python.exe"

if ($Action -eq "collect") {
    # 寄付き前の日次データ更新 (冪等・不足日のみ取得)
    & $py scripts\collect_jp_daily_history.py *>> $log
} elseif ($Action -eq "pushexp") {
    # 50銘柄PUSH実験（発注なし。全銘柄1周→候補50を登録→08:50/55/59に同時スナップ）
    & $py scripts\run_push_experiment.py *>> $log
    # 要点だけ Slack に流す（朝の成否がひと目で分かるように）
    $keep = (Get-Content $log -Encoding Unicode -ErrorAction SilentlyContinue |
        Where-Object { $_ -match '取得 |候補 |スリーブ |スメア' }) -join "`n"
    if ($keep) {
        $fence = [string][char]96 * 3
        & (Join-Path $PSScriptRoot "notify_slack.ps1") `
            -Text ("🧪 *50銘柄PUSH実験* (" + (Get-Date -Format "MM/dd HH:mm") +
                   ")`n$fence`n$keep`n$fence") | Out-Null
    }
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
    $tail = (Get-Content $log -Tail 25 -Encoding Unicode -ErrorAction SilentlyContinue) -join "`n"
    & (Join-Path $PSScriptRoot "notify_slack.ps1") -Title "$Action 失敗 (exit=$code)" -Detail $tail | Out-Null
}

# ログの肥大化防止: 30日より古いログを削除
Get-ChildItem $logDir -Filter *.log |
    Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-30) } |
    Remove-Item -Force -ErrorAction SilentlyContinue

exit $code
