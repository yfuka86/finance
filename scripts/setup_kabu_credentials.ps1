# kabuステーションのログイン資格情報を DPAPI で暗号化して保存する（初回1回だけ実行）
#
#   powershell -ExecutionPolicy Bypass -File scripts\setup_kabu_credentials.ps1
#
# 保存先: data\live_reports\.kabu_creds.xml (Git管理外)
# DPAPI により「このWindowsユーザー・このマシン」でしか復号できない形式で保存する。
# 平文の .env に置かないのはこのため。別ユーザー/別マシンでは復号できない。
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$out = Join-Path $root "data\live_reports\.kabu_creds.xml"
New-Item -ItemType Directory -Force (Split-Path $out) | Out-Null

Write-Host "kabuステーションのログイン情報を入力してください（画面には表示されません）" -ForegroundColor Cyan
Write-Host "※ APIパスワード・注文パスワードではなく、アプリにログインするIDとパスワードです" -ForegroundColor Yellow

$id = Read-Host "ログインID（口座番号など）"
$pw = Read-Host "ログインパスワード" -AsSecureString

if ([string]::IsNullOrWhiteSpace($id)) { throw "ログインIDが空です" }

[pscustomobject]@{
    LoginId  = $id
    Password = $pw          # SecureString は Export-Clixml で DPAPI 暗号化される
    SavedAt  = (Get-Date).ToString("o")
} | Export-Clixml -Path $out

# 念のためファイルのACLを本人のみに絞る
$acl = Get-Acl $out
$acl.SetAccessRuleProtection($true, $false)
$rule = New-Object System.Security.AccessControl.FileSystemAccessRule(
    "$env:USERDOMAIN\$env:USERNAME", "FullControl", "Allow")
$acl.SetAccessRule($rule)
Set-Acl -Path $out -AclObject $acl

Write-Host "`n保存しました: $out" -ForegroundColor Green
Write-Host "この資格情報は $env:USERNAME@$env:COMPUTERNAME でのみ復号できます。"
Write-Host "次: powershell -ExecutionPolicy Bypass -File scripts\ensure_kabu_login.ps1 -Force"
