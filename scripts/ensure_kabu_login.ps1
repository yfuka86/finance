# kabuステーションが API を受け付ける状態であることを保証する（毎朝・冪等）
#
#   powershell -ExecutionPolicy Bypass -File scripts\ensure_kabu_login.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\ensure_kabu_login.ps1 -Force
#
# 動作:
#   1. POST /token で疎通確認 → 200 なら何もせず終了（冪等・毎朝流して安全）
#   2. 401/接続不可なら KabuS.exe を終了 → 起動 → ログイン画面を段階的に進める
#   3. API が 200 を返すまでポーリング。成否を終了コードで返す
#
# ログイン画面は WebView2（Chromium）で描画された多段フォーム:
#   [1] 口座番号(8桁) を AutoId=username に入力 → 「次へ」
#   [2] パスワード入力 → 「ログイン」
#   [3] メールのワンタイム認証コード（毎回要求される。端末を信頼する設定は無い）
#       → scripts\fetch_otp.ps1 が Gmail(IMAP) から取得して自動入力する
# 要素は UIAutomation で AutomationId / Name を手がかりに毎回探索する（DOM変更に弱いため
# 見つからなければ状態をログに落として終了し、手動ログインに委ねる）。
#
# 注意: UI自動化は「対話セッションがロックされていない」ことが必要。RDPを閉じるときは
#       ログオフせず切断すること。
[CmdletBinding()]
param(
    [switch]$Force,
    [int]$LoginTimeoutSec = 420,   # 2FAメールの到着待ちを含むので長め
    [int]$OtpTimeoutSec = 180,     # 1回あたりのメール待ち時間
    [int]$MaxCodeTries = 2,        # 認証コードの入力試行上限（誤コードでの締め出し防止）
    [string]$ApiPassword
)
$ErrorActionPreference = "Stop"
foreach ($a in @("UIAutomationClient", "UIAutomationTypes", "System.Windows.Forms")) {
    try { Add-Type -AssemblyName $a } catch {}
}
if (-not ("System.Windows.Automation.AutomationElement" -as [type])) {
    [void][System.Reflection.Assembly]::LoadWithPartialName("UIAutomationClient")
    [void][System.Reflection.Assembly]::LoadWithPartialName("UIAutomationTypes")
}

$root = Split-Path -Parent $PSScriptRoot
$exe = "$env:LOCALAPPDATA\kabuStation\KabuS.exe"
$credPath = Join-Path $root "data\live_reports\.kabu_creds.xml"
$logDir = Join-Path $root "data\live_reports\task_logs"
New-Item -ItemType Directory -Force $logDir | Out-Null

$logFile = Join-Path $logDir ("kabulogin_{0}.log" -f (Get-Date -Format "yyyyMMdd"))

function Log($msg) {
    $line = "{0} {1}" -f (Get-Date -Format "HH:mm:ss"), $msg
    Write-Host $line
    Add-Content -Path $logFile -Value $line -Encoding utf8
}

# 失敗は Slack に出して終わる。朝ここで転ぶと entry が不発になるので気付けることが最重要。
function Fail([int]$code, [string]$msg) {
    Log "ERROR: $msg"
    try {
        $tail = (Get-Content $logFile -Tail 20 -Encoding UTF8 -ErrorAction SilentlyContinue) -join "`n"
        & (Join-Path $PSScriptRoot "notify_slack.ps1") `
            -Title "kabuステーション 自動ログイン失敗 (exit=$code)" -Detail $tail | Out-Null
    } catch {}
    exit $code
}

# ── .env から APIパスワード ────────────────────────────────────
if (-not $ApiPassword) {
    $envFile = Join-Path $root ".env"
    if (Test-Path $envFile) {
        foreach ($l in Get-Content $envFile -Encoding UTF8) {
            if ($l -match '^\s*KABU_API_PASSWORD\s*=\s*(.+?)\s*$') { $ApiPassword = $Matches[1].Trim('"').Trim("'") }
        }
    }
}
if (-not $ApiPassword) { Fail 2 "KABU_API_PASSWORD が取得できません" }

function Test-KabuApi {
    try {
        $body = @{ APIPassword = $ApiPassword } | ConvertTo-Json -Compress
        $r = Invoke-WebRequest -Uri "http://localhost:18080/kabusapi/token" -Method POST `
            -Body $body -ContentType "application/json" -UseBasicParsing -TimeoutSec 8
        return $r.StatusCode -eq 200
    } catch { return $false }
}

if ((Test-KabuApi) -and -not $Force) { Log "OK: 既にログイン済み（何もしません）"; exit 0 }
Log $(if ($Force) { "Force指定: 再起動します" } else { "APIが応答しません（未ログイン）→ 再起動します" })

# ── UIAutomation ヘルパ ────────────────────────────────────────
function Get-Elements($procId) {
    $r = [System.Windows.Automation.AutomationElement]::RootElement
    $c = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::ProcessIdProperty, $procId)
    $out = @()
    foreach ($w in $r.FindAll([System.Windows.Automation.TreeScope]::Children, $c)) {
        $out += $w.FindAll([System.Windows.Automation.TreeScope]::Descendants,
            [System.Windows.Automation.Condition]::TrueCondition)
    }
    return $out
}
function Find-One($els, [string]$type, [string]$autoId, [string]$namePattern) {
    foreach ($e in $els) {
        $t = $e.Current.ControlType.ProgrammaticName -replace 'ControlType\.', ''
        if ($type -and $t -ne $type) { continue }
        if (-not $e.Current.IsEnabled) { continue }
        if ($autoId -and $e.Current.AutomationId -eq $autoId) { return $e }
        if ($namePattern -and $e.Current.Name -match $namePattern) { return $e }
    }
    return $null
}
function Set-Text($el, [string]$text) {
    try {
        $vp = $el.GetCurrentPattern([System.Windows.Automation.ValuePattern]::Pattern)
        $vp.SetValue($text)
        return $true
    } catch {
        try {
            $el.SetFocus(); Start-Sleep -Milliseconds 300
            [System.Windows.Forms.SendKeys]::SendWait("^a")
            [System.Windows.Forms.SendKeys]::SendWait($text)
            return $true
        } catch { return $false }
    }
}
function Invoke-El($el) {
    try {
        $ip = $el.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern)
        $ip.Invoke(); return $true
    } catch {
        try { $el.SetFocus(); [System.Windows.Forms.SendKeys]::SendWait("{ENTER}"); return $true }
        catch { return $false }
    }
}

# ── 既存プロセス終了 → 起動 ───────────────────────────────────
$procs = Get-Process KabuS -ErrorAction SilentlyContinue
if ($procs) {
    Log "既存プロセスを終了: PID $($procs.Id -join ',')"
    $procs | Stop-Process -Force -ErrorAction SilentlyContinue
    for ($i = 0; $i -lt 20 -and (Get-Process KabuS -ErrorAction SilentlyContinue); $i++) { Start-Sleep -Milliseconds 500 }
}
if (-not (Test-Path $exe)) { Fail 2 "$exe がありません" }
$p = Start-Process $exe -PassThru
Log "起動: PID $($p.Id)"

if (-not (Test-Path $credPath)) {
    Log "資格情報なし → アプリ側の保存ログインに期待して待機します"
    Log "  （必要なら scripts\setup_kabu_credentials.ps1 を実行）"
}
$creds = if (Test-Path $credPath) { Import-Clixml $credPath } else { $null }

# ── 段階的にログインを進める ──────────────────────────────────
$stepAccount = $false; $stepPassword = $false
$codeTries = 0
$pwSubmittedEpoch = 0
$deadline = (Get-Date).AddSeconds($LoginTimeoutSec)
$today = Join-Path $logDir ("kabulogin_{0}.log" -f (Get-Date -Format "yyyyMMdd"))

while ((Get-Date) -lt $deadline) {
    if (Test-KabuApi) { Log "ログイン完了（API応答OK）"; exit 0 }
    if (-not $creds) { Start-Sleep -Seconds 3; continue }

    $els = Get-Elements $p.Id

    # [3] ワンタイム認証コード（メール）— 入力欄が出たら Gmail から取りに行く。
    #     パスワード投入時刻より後に届いたメールだけを見るので古いコードは使わない。
    $codeEl = Find-One $els "Edit" "code" '認証コード|ワンタイム|確認コード'
    if ($codeEl) {
        if ($codeTries -ge $MaxCodeTries) {
            Fail 3 "認証コードを $codeTries 回試しましたが通りませんでした（手動確認が必要）"
        }
        $codeTries++
        $since = if ($pwSubmittedEpoch -gt 0) { $pwSubmittedEpoch - 90 }
        else { [DateTimeOffset]::UtcNow.ToUnixTimeSeconds() - 180 }
        Log "[3] 認証コードを要求されました → Gmail から取得します（$codeTries/$MaxCodeTries 回目）"
        $remain = [int]([Math]::Max(30, ($deadline - (Get-Date)).TotalSeconds - 30))
        $otp = & (Join-Path $PSScriptRoot "fetch_otp.ps1") -SinceEpoch $since `
            -TimeoutSec ([Math]::Min($OtpTimeoutSec, $remain)) -LogPath $today
        $otp = ($otp | Select-Object -Last 1)
        if ($LASTEXITCODE -ne 0 -or -not ($otp -match '^\d{4,8}$')) {
            Log "[3] コードを取得できませんでした（fetch_otp exit=$LASTEXITCODE）"
            Start-Sleep -Seconds 5
            continue
        }
        $masked = $otp.Substring(0, 1) + ('*' * ($otp.Length - 2)) + $otp.Substring($otp.Length - 1)
        if (Set-Text $codeEl $otp) { Log "[3] 認証コードを入力: $masked" }
        $otp = $null
        Start-Sleep -Milliseconds 400
        $go = Find-One (Get-Elements $p.Id) "Button" $null '続ける|認証|送信|ログイン|^次へ$'
        if ($go -and (Invoke-El $go)) { Log "[3] 「$($go.Current.Name)」を押下"; Start-Sleep -Seconds 5 }
        else { Log "[3] 送信ボタンが見つかりません" }
        continue
    }

    # [1] 口座番号
    if (-not $stepAccount) {
        $acc = Find-One $els "Edit" "username" '口座番号'
        if ($acc) {
            if (Set-Text $acc $creds.LoginId) { Log "[1] 口座番号を入力: $($creds.LoginId)" }
            Start-Sleep -Milliseconds 400
            $next = Find-One (Get-Elements $p.Id) "Button" $null '^次へ$'
            if ($next -and (Invoke-El $next)) { Log "[1] 「次へ」を押下"; $stepAccount = $true; Start-Sleep -Seconds 3 }
            else { Log "[1] 「次へ」ボタンが見つかりません" }
            continue
        }
    }

    # [2] パスワード（IsPassword の Edit、無ければ password 系 AutomationId）
    if ($stepAccount -and -not $stepPassword) {
        $pwEl = $null
        foreach ($e in $els) {
            if (($e.Current.ControlType.ProgrammaticName -replace 'ControlType\.', '') -ne 'Edit') { continue }
            if (-not $e.Current.IsEnabled) { continue }
            $isPw = $false
            try { $isPw = $e.Current.IsPassword } catch {}
            if ($isPw -or $e.Current.AutomationId -match 'password|passwd' -or $e.Current.Name -match 'パスワード') {
                $pwEl = $e; break
            }
        }
        if ($pwEl) {
            $plain = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
                [Runtime.InteropServices.Marshal]::SecureStringToBSTR($creds.Password))
            $ok = Set-Text $pwEl $plain
            $plain = $null
            Log "[2] パスワードを入力: $ok"
            Start-Sleep -Milliseconds 400
            $btn = Find-One (Get-Elements $p.Id) "Button" $null 'ログイン|^次へ$|送信'
            if ($btn -and (Invoke-El $btn)) {
                # 認証コードメールはこの押下の後に届く。以降の到着分だけを見るため時刻を控える。
                # PS5.1 の Get-Date -UFormat %s はローカル時刻基準でズレるため使わない
                $pwSubmittedEpoch = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
                Log "[2] 「$($btn.Current.Name)」を押下"; $stepPassword = $true; Start-Sleep -Seconds 3
            }
            else { Log "[2] ログインボタンが見つかりません" }
            continue
        }
    }
    Start-Sleep -Seconds 3
}

# タイムアウト時は画面に何が出ているかを残す（次回の調整用）
Log "$LoginTimeoutSec 秒以内にログインできませんでした"
try {
    foreach ($e in (Get-Elements $p.Id)) {
        $t = $e.Current.ControlType.ProgrammaticName -replace 'ControlType\.', ''
        if ($t -in @('Edit', 'Button') -or ($t -eq 'Text' -and $e.Current.Name)) {
            Log ("  画面: [{0}] Name=[{1}] AutoId=[{2}]" -f $t, $e.Current.Name, $e.Current.AutomationId)
        }
    }
} catch {}
Fail 1 "ログインできませんでした（上の画面ダンプを確認して手動ログインしてください）"
