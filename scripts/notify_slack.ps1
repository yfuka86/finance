# Slack へ1通だけ通知する（タスクスケジューラ／PowerShell 側の失敗検知用）
#
#   powershell -File scripts\notify_slack.ps1 -Test
#   powershell -File scripts\notify_slack.ps1 -Title "entry 失敗 (exit=1)" -Detail "<ログ末尾>"
#   powershell -File scripts\notify_slack.ps1 -Text "任意の本文"
#
# トークンは .env の SLACK_BOT_TOKEN / SLACK_CHANNEL（Git管理外）。
# python 側の通知は trading/jp_intraday/live/notifier.py（イベント要約）。こちらは
# 「python が起動すらしなかった」ケースまで拾うための最終防衛線。
# 失敗しても呼び出し元を壊さない（この通知自体の失敗で運用を止めない）。
[CmdletBinding()]
param(
    [string]$Title,
    [string]$Detail = "",
    [string]$Text,
    [switch]$Test
)
$ErrorActionPreference = "Continue"

$root = Split-Path -Parent $PSScriptRoot
$token = ""; $channel = ""
$envFile = Join-Path $root ".env"
if (Test-Path $envFile) {
    foreach ($l in Get-Content $envFile -Encoding UTF8) {
        if ($l -match '^\s*SLACK_BOT_TOKEN\s*=\s*(.+?)\s*$') { $token = $Matches[1].Trim('"').Trim("'") }
        if ($l -match '^\s*SLACK_CHANNEL\s*=\s*(.+?)\s*$') { $channel = $Matches[1].Trim('"').Trim("'") }
    }
}
if (-not $token -or -not $channel) {
    Write-Host "SLACK_BOT_TOKEN / SLACK_CHANNEL が .env にありません（通知スキップ）"
    exit 1
}

if ($Test) {
    $body = "✅ 疎通確認: PowerShell から Slack へ送信できています ($env:COMPUTERNAME)"
} elseif ($Text) {
    $body = $Text
} elseif ($Title) {
    $body = "🚨 *$Title*"
    if ($Detail) {
        $d = $Detail
        if ($d.Length -gt 2500) { $d = $d.Substring($d.Length - 2500) }  # 末尾を残す
        $fence = [string][char]96 * 3   # バッククォート3つ（PSの文字列内で書くと壊れるため文字コードで作る）
        $body += "`n$fence`n$d`n$fence"
    }
} else {
    Write-Host "-Title / -Text / -Test のいずれかを指定してください"
    exit 2
}

try {
    $json = @{ channel = $channel; text = $body; unfurl_links = $false } | ConvertTo-Json -Compress
    $bytes = [Text.Encoding]::UTF8.GetBytes($json)   # PS5.1 は既定で UTF-8 にせず日本語が化ける
    $res = Invoke-RestMethod -Uri "https://slack.com/api/chat.postMessage" -Method Post `
        -Headers @{ Authorization = "Bearer $token" } `
        -ContentType "application/json; charset=utf-8" -Body $bytes -TimeoutSec 10
    if ($res.ok) { Write-Host "Slack: 送信しました"; exit 0 }
    Write-Host "Slack: 送信に失敗しました ($($res.error))"   # 例: not_in_channel = bot 未招待
    exit 1
} catch {
    Write-Host "Slack: 送信に失敗しました ($($_.Exception.Message))"
    exit 1
}
