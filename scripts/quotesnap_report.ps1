# 当日の quotesnap を実寄値と突き合わせ、結果を Slack に投げる（夕方に実行）
#
#   powershell -ExecutionPolicy Bypass -File scripts\quotesnap_report.ps1
#
# 手順: ①当日分の日次データを取得（J-Quantsの公表は夕方）→ ②analyze_quotesnap.py
#       → ③Slackへ投稿。実寄値がまだ公表されていなければ非0で終了し、
#       タスクスケジューラのリトライ（30分×3回）に任せる。
[CmdletBinding()]
param([switch]$SkipCollect)
$ErrorActionPreference = "Continue"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONPATH = "."
$env:PYTHONUTF8 = "1"
$env:PYTHONUNBUFFERED = "1"
$py = Join-Path $root ".venv\Scripts\python.exe"
$logDir = Join-Path $root "data\live_reports\task_logs"
New-Item -ItemType Directory -Force $logDir | Out-Null
$log = Join-Path $logDir ("quotesnapreport_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))

function Run($exe, $argList, $label) {
    $o = [IO.Path]::GetTempFileName(); $e = [IO.Path]::GetTempFileName()
    $p = Start-Process -FilePath $exe -ArgumentList $argList -WorkingDirectory $root `
        -NoNewWindow -Wait -PassThru -RedirectStandardOutput $o -RedirectStandardError $e
    $out = (Get-Content $o -Raw -Encoding UTF8) + (Get-Content $e -Raw -Encoding UTF8)
    Remove-Item $o, $e -Force -ErrorAction SilentlyContinue
    "=== $label (exit=$($p.ExitCode)) ===`n$out" | Add-Content $log -Encoding utf8
    return @{ code = $p.ExitCode; out = $out }
}

if (-not $SkipCollect) {
    $c = Run $py "scripts\collect_jp_daily_history.py" "collect"
    Write-Host "collect exit=$($c.code)"
}

$a = Run $py "scripts\analyze_quotesnap.py" "analyze"
Write-Host $a.out

if ($a.code -ne 0) {
    # 実寄値未着（夕方の公表前）などはリトライさせる。ログだけ残して非0で抜ける
    Write-Host "分析できませんでした (exit=$($a.code)) → タスクのリトライに任せます"
    exit $a.code
}

# 数値部分だけを抜き出して Slack へ（全文は長いのでヘッダ行と各時点の要約）
$lines = $a.out -split "`r?`n" | Where-Object { $_ -match '^\[|縮小率|判定目安' }
$body = "📐 *気配 vs 実寄値の実測* (" + (Get-Date -Format "MM/dd") + ")`n``````n" +
        (($lines -join "`n").Trim()) + "`n``````"
& (Join-Path $PSScriptRoot "notify_slack.ps1") -Text $body | Out-Null
Write-Host "Slackへ投稿しました"
exit 0
