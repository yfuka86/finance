# PUSH配信(WebSocket)が場中に本当に飛ぶかを確認し、結果をSlackへ投げる
#
#   powershell -ExecutionPolicy Bypass -File scripts\push_check_task.ps1
#
# 発注なし・read-only。寄前(08:00以降)に走らせる想定。
# 判定: PUSH受信>0 かつ スメアが数秒以内 → 50銘柄の同時スナップショットが成立する。
[CmdletBinding()]
param([int]$Seconds = 90, [int]$N = 50)
$ErrorActionPreference = "Continue"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONPATH = "."
$env:PYTHONUTF8 = "1"
$env:PYTHONUNBUFFERED = "1"
$py = Join-Path $root ".venv\Scripts\python.exe"
$logDir = Join-Path $root "data\live_reports\task_logs"
New-Item -ItemType Directory -Force $logDir | Out-Null
$log = Join-Path $logDir ("pushcheck_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))

$o = [IO.Path]::GetTempFileName(); $e = [IO.Path]::GetTempFileName()
$p = Start-Process -FilePath $py -ArgumentList @("scripts\check_push_feed.py",
    "--n", $N, "--seconds", $Seconds) -WorkingDirectory $root `
    -NoNewWindow -Wait -PassThru -RedirectStandardOutput $o -RedirectStandardError $e
$out = (Get-Content $o -Raw -Encoding UTF8) + (Get-Content $e -Raw -Encoding UTF8)
Remove-Item $o, $e -Force -ErrorAction SilentlyContinue
$out | Add-Content $log -Encoding utf8
Write-Host $out

# 要点だけ抜き出して通知（受信件数・スメア・結論行）
$lines = ($out -split "`r?`n") | Where-Object { $_ -match '登録|初期値|結果:|スナップショット|→' }
$fence = [string][char]96 * 3
$body = "🔌 *PUSH配信の実機確認* (" + (Get-Date -Format "MM/dd HH:mm") + ")`n" +
        "$fence`n" + (($lines -join "`n").Trim()) + "`n$fence"
& (Join-Path $PSScriptRoot "notify_slack.ps1") -Text $body | Out-Null
exit $p.ExitCode
