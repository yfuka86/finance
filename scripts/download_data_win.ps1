# Windows用: 検証データ一式をダウンロードして展開する（リポジトリ直下で実行）
# 使い方:  powershell -ExecutionPolicy Bypass -File scripts\download_data_win.ps1 -Url "<署名付きURL>"
param(
    [Parameter(Mandatory = $true)][string]$Url
)
$ErrorActionPreference = "Stop"
$tar = Join-Path $env:TEMP "jp_trading_data.tar.gz"

Write-Host "downloading data tarball..." -ForegroundColor Cyan
Invoke-WebRequest -Uri $Url -OutFile $tar
Write-Host ("downloaded: {0:N0} MB" -f ((Get-Item $tar).Length / 1MB))

Write-Host "extracting into repo root..." -ForegroundColor Cyan
# Windows 10+ は bsdtar 同梱: .tar.gz をそのまま展開できる
tar -xzf $tar -C .
Remove-Item $tar

Write-Host "done. 検証:" -ForegroundColor Green
Write-Host "  python -m trading.jp_intraday.live.run_live preflight"
