# Gmail からワンタイム認証コードを1つ取り出して stdout に出す
#
#   powershell -ExecutionPolicy Bypass -File scripts\fetch_otp.ps1 -Probe
#   powershell -ExecutionPolicy Bypass -File scripts\fetch_otp.ps1 -SinceEpoch 1753771000 -TimeoutSec 180
#
# DPAPI で保存した資格情報（scripts\setup_gmail_otp.ps1）を復号し、環境変数として
# 子プロセス（python）にだけ渡す。平文はディスクに書かない。
# stdout はコード1行のみ（呼び出し側がそのまま使う）。診断は stderr / -LogPath へ。
[CmdletBinding()]
param(
    [double]$SinceEpoch = 0,
    [int]$TimeoutSec = 180,
    [int]$PollSec = 10,
    [switch]$Probe,
    [string]$LogPath
)
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$credPath = Join-Path $root "data\live_reports\.gmail_otp.xml"
$py = Join-Path $root ".venv\Scripts\python.exe"

function Note($msg) {
    Write-Host $msg
    if ($LogPath) { Add-Content -Path $LogPath -Value ("{0} {1}" -f (Get-Date -Format "HH:mm:ss"), $msg) -Encoding utf8 }
}

if (-not (Test-Path $credPath)) {
    Note "ERROR: $credPath がありません → scripts\setup_gmail_otp.ps1 を実行してください"
    exit 2
}
if (-not (Test-Path $py)) { Note "ERROR: $py がありません"; exit 2 }

$c = Import-Clixml $credPath
$env:OTP_IMAP_USER = $c.Address
$env:OTP_IMAP_APP_PASSWORD = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
    [Runtime.InteropServices.Marshal]::SecureStringToBSTR($c.AppPassword))
$env:PYTHONPATH = $root
$env:PYTHONUTF8 = "1"   # タスクスケジューラ実行は cp932 になるため

$pyArgs = @("-m", "trading.jp_intraday.live.otp_mail")
if ($Probe) {
    $pyArgs += "--probe"
} else {
    if ($SinceEpoch -gt 0) { $pyArgs += @("--since-epoch", $SinceEpoch.ToString("F0")) }
    $pyArgs += @("--timeout", $TimeoutSec, "--poll", $PollSec)
}

# ネイティブ実行の stderr を PS 5.1 の `2>` で受けると NativeCommandError になるため
# Start-Process でファイルに分離する。
$outFile = [IO.Path]::GetTempFileName()
$errFile = [IO.Path]::GetTempFileName()
try {
    $p = Start-Process -FilePath $py -ArgumentList $pyArgs -WorkingDirectory $root `
        -NoNewWindow -Wait -PassThru -RedirectStandardOutput $outFile -RedirectStandardError $errFile
    # python は PYTHONUTF8=1 で UTF-8 を吐くので明示（既定の cp932 読みだと日本語が化ける）
    foreach ($l in (Get-Content $errFile -Encoding UTF8 -ErrorAction SilentlyContinue)) {
        if ($l) { Note "  otp: $l" }
    }
    $code = (Get-Content $outFile -Raw -ErrorAction SilentlyContinue)
    if ($code) { Write-Output $code.Trim() }
    exit $p.ExitCode
} finally {
    $env:OTP_IMAP_APP_PASSWORD = $null
    Remove-Item $outFile, $errFile -Force -ErrorAction SilentlyContinue
}
