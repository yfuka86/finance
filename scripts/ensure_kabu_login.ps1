# kabuステーションが API を受け付ける状態であることを保証する（毎朝・冪等）
#
#   powershell -ExecutionPolicy Bypass -File scripts\ensure_kabu_login.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\ensure_kabu_login.ps1 -Force   # 健全でも再起動
#
# 動作:
#   1. POST /token で疎通確認 → 200 なら何もせず終了（冪等・毎朝流して安全）
#   2. 401/接続不可なら KabuS.exe を終了 → 起動 → ログイン画面に資格情報を投入
#   3. API が 200 を返すまでポーリング。成否を終了コードで返す
#
# 前提: 事前に scripts\setup_kabu_credentials.ps1 を1回実行しておくこと。
#       アプリ側でID/パスワード保存が効いていれば資格情報なしでも通る（その場合は
#       入力をスキップしてログイン完了を待つだけ）。
#
# 注意: UI自動化は「対話セッションがロックされていない」ことが必要。RDPを閉じるときは
#       ログオフせず切断すること。切断でUIが描画されなくなる環境では、タスクの設定を
#       「ユーザーがログオンしているときのみ実行」にし、コンソールセッションに
#       tscon で戻しておく（README参照）。
[CmdletBinding()]
param(
    [switch]$Force,                     # 疎通OKでも強制的に再起動する
    [int]$LoginTimeoutSec = 180,        # ログイン完了(API 200)までの待ち時間
    [string]$ApiPassword                # 省略時は .env の KABU_API_PASSWORD を読む
)
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName UIAutomationClient, UIAutomationTypes, System.Windows.Forms

$root = Split-Path -Parent $PSScriptRoot
$exe = "$env:LOCALAPPDATA\kabuStation\KabuS.exe"
$credPath = Join-Path $root "data\live_reports\.kabu_creds.xml"
$logDir = Join-Path $root "data\live_reports\task_logs"
New-Item -ItemType Directory -Force $logDir | Out-Null

function Log($msg) {
    $line = "{0} {1}" -f (Get-Date -Format "HH:mm:ss"), $msg
    Write-Host $line
    Add-Content -Path (Join-Path $logDir ("kabulogin_{0}.log" -f (Get-Date -Format "yyyyMMdd"))) `
        -Value $line -Encoding utf8
}

# ── .env から APIパスワードを読む（引数優先） ───────────────────────
if (-not $ApiPassword) {
    $envFile = Join-Path $root ".env"
    if (Test-Path $envFile) {
        foreach ($l in Get-Content $envFile -Encoding UTF8) {
            if ($l -match '^\s*KABU_API_PASSWORD\s*=\s*(.+?)\s*$') {
                $ApiPassword = $Matches[1].Trim('"').Trim("'")
            }
        }
    }
}
if (-not $ApiPassword) { Log "ERROR: KABU_API_PASSWORD が取得できません"; exit 2 }

# ── API 疎通確認 ────────────────────────────────────────────────
function Test-KabuApi {
    try {
        $body = @{ APIPassword = $ApiPassword } | ConvertTo-Json -Compress
        $r = Invoke-WebRequest -Uri "http://localhost:18080/kabusapi/token" -Method POST `
            -Body $body -ContentType "application/json" -UseBasicParsing -TimeoutSec 8
        return $r.StatusCode -eq 200
    } catch { return $false }
}

if ((Test-KabuApi) -and -not $Force) {
    Log "OK: kabuステーションは既にログイン済み（何もしません）"
    exit 0
}
Log $(if ($Force) { "Force指定: 再起動します" } else { "APIが応答しません（未ログイン）→ 再起動します" })

# ── 既存プロセスを終了 ──────────────────────────────────────────
$procs = Get-Process KabuS -ErrorAction SilentlyContinue
if ($procs) {
    Log "既存プロセスを終了: PID $($procs.Id -join ',')"
    $procs | Stop-Process -Force -ErrorAction SilentlyContinue
    for ($i = 0; $i -lt 20 -and (Get-Process KabuS -ErrorAction SilentlyContinue); $i++) {
        Start-Sleep -Milliseconds 500
    }
}

# ── 起動 ───────────────────────────────────────────────────────
if (-not (Test-Path $exe)) { Log "ERROR: 実行ファイルがありません: $exe"; exit 2 }
$p = Start-Process $exe -PassThru
Log "起動: PID $($p.Id)"

# ── ログイン画面を探して資格情報を投入 ──────────────────────────
function Get-ProcWindows($procId) {
    $root = [System.Windows.Automation.AutomationElement]::RootElement
    $cond = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::ProcessIdProperty, $procId)
    return $root.FindAll([System.Windows.Automation.TreeScope]::Children, $cond)
}
function Find-Descendants($el, $typeName) {
    $c = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
        [System.Windows.Automation.ControlType]::$typeName)
    return $el.FindAll([System.Windows.Automation.TreeScope]::Descendants, $c)
}
function Set-EditValue($el, $text) {
    # ValuePattern が使えればそれで、駄目ならフォーカス＋SendKeys
    try {
        $vp = $el.GetCurrentPattern([System.Windows.Automation.ValuePattern]::Pattern)
        $vp.SetValue($text); return $true
    } catch {
        try {
            $el.SetFocus(); Start-Sleep -Milliseconds 200
            [System.Windows.Forms.SendKeys]::SendWait($text); return $true
        } catch { return $false }
    }
}

$creds = $null
if (Test-Path $credPath) {
    $creds = Import-Clixml $credPath
    Log "資格情報を読み込みました（ID: $($creds.LoginId)）"
} else {
    Log "資格情報ファイルなし → アプリ側の保存ログインに期待して待機します"
    Log "  （必要なら scripts\setup_kabu_credentials.ps1 を実行してください）"
}

$filled = $false
$deadline = (Get-Date).AddSeconds($LoginTimeoutSec)
while ((Get-Date) -lt $deadline) {
    if (Test-KabuApi) { Log "ログイン完了（API応答OK）"; exit 0 }

    if ($creds -and -not $filled) {
        foreach ($w in (Get-ProcWindows $p.Id)) {
            $edits = Find-Descendants $w "Edit"
            $buttons = Find-Descendants $w "Button"
            if ($edits.Count -lt 2) { continue }   # ID+パスワードが揃う画面のみ対象

            Log ("ログイン画面を検出: Name=[{0}] Edit={1} Button={2}" -f `
                    $w.Current.Name, $edits.Count, $buttons.Count)
            $plain = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
                [Runtime.InteropServices.Marshal]::SecureStringToBSTR($creds.Password))
            $okId = Set-EditValue $edits[0] $creds.LoginId
            $okPw = Set-EditValue $edits[1] $plain
            $plain = $null
            Log "  入力: ID=$okId パスワード=$okPw"

            # ログインボタンらしきものを押す（名前一致 → 無ければ最初の有効なボタン）
            $target = $null
            foreach ($b in $buttons) {
                if ($b.Current.Name -match 'ログイン|ﾛｸﾞｲﾝ|OK|接続') { $target = $b; break }
            }
            if (-not $target -and $buttons.Count -gt 0) { $target = $buttons[0] }
            if ($target) {
                Log ("  ボタン押下: [{0}]" -f $target.Current.Name)
                try {
                    $ip = $target.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern)
                    $ip.Invoke()
                } catch { $target.SetFocus(); [System.Windows.Forms.SendKeys]::SendWait("{ENTER}") }
            }
            $filled = $true
            break
        }
    }
    Start-Sleep -Seconds 3
}

Log "ERROR: $LoginTimeoutSec 秒以内にログインできませんでした（手動ログインが必要）"
exit 1
