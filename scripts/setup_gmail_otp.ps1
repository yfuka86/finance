# ワンタイム認証コードを読むための Gmail 資格情報を DPAPI 暗号化して保存する（初回1回だけ）
#
#   対話:
#     powershell -ExecutionPolicy Bypass -File scripts\setup_gmail_otp.ps1
#   非対話（TTYの無い環境。パスワードがシェル履歴に残る点に注意）:
#     powershell -ExecutionPolicy Bypass -File scripts\setup_gmail_otp.ps1 `
#       -Address "yfuka86@gmail.com" -AppPassword "abcd efgh ijkl mnop"
#
# 使うのは Google の **アプリパスワード**（16桁）であって、通常のGoogleパスワードではない。
#   1. https://myaccount.google.com/signinoptions/two-step-verification で2段階認証をON
#   2. https://myaccount.google.com/apppasswords で「kabu-otp」等の名前で発行
#   3. 表示された16桁をこのスクリプトに渡す（空白は入れても除去される）
#
# 保存先: data\live_reports\.gmail_otp.xml (Git管理外・DPAPIでこのユーザー/このマシン限定)
# 平文の .env に置かないのは、メール読み取り権限が kabu の API パスワードより広いため。
[CmdletBinding()]
param(
    [string]$Address,
    [string]$AppPassword
)
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$out = Join-Path $root "data\live_reports\.gmail_otp.xml"
New-Item -ItemType Directory -Force (Split-Path $out) | Out-Null

if (-not $Address -or -not $AppPassword) {
    Write-Host "認証コードが届く Gmail の情報を入力してください" -ForegroundColor Cyan
    Write-Host "※ 通常のGoogleパスワードではなく、2段階認証で発行する「アプリパスワード」です" -ForegroundColor Yellow
}
$addr = if ($Address) { $Address } else { Read-Host "Gmailアドレス" }
$pw = if ($AppPassword) {
    ConvertTo-SecureString ($AppPassword -replace '\s', '') -AsPlainText -Force
} else {
    Read-Host "アプリパスワード(16桁)" -AsSecureString
}
if ([string]::IsNullOrWhiteSpace($addr)) { throw "Gmailアドレスが空です" }

[pscustomobject]@{
    Address     = $addr
    AppPassword = $pw        # SecureString は Export-Clixml で DPAPI 暗号化される
    SavedAt     = (Get-Date).ToString("o")
} | Export-Clixml -Path $out

$acl = Get-Acl $out
$acl.SetAccessRuleProtection($true, $false)
$acl.SetAccessRule((New-Object System.Security.AccessControl.FileSystemAccessRule(
            "$env:USERDOMAIN\$env:USERNAME", "FullControl", "Allow")))
Set-Acl -Path $out -AclObject $acl

Write-Host "`n保存しました: $out" -ForegroundColor Green
Write-Host "確認: powershell -ExecutionPolicy Bypass -File scripts\fetch_otp.ps1 -Probe"
Write-Host "  → 直近7日の該当メール（差出人・件名・コード検出可否）が出れば設定完了です。"
