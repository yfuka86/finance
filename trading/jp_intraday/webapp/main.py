"""Web management dashboard receiver for a-tokyo.jp (FastAPI, Cloud Run-ready).

Receives daily plan/entry/exit/state reports from the Windows kabuステーション
client (POST /api/report, bearer auth) and serves a simple dashboard at /.

Storage: Firestore if GOOGLE_CLOUD_PROJECT + google-cloud-firestore are available
(recommended for Cloud Run — it is stateless); otherwise a local JSONL file
(fine for a single VM / local run).

Env:
  REPORT_TOKEN   shared bearer token that the client must present
  DATA_DIR       local fallback store dir (default /tmp/live_reports)
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI(title="A-Tokyo 場中トレード 管理画面")
TOKEN = os.environ.get("REPORT_TOKEN", "")
DATA_DIR = Path(os.environ.get("DATA_DIR", "/tmp/live_reports"))

try:  # optional Firestore backend (recommended on Cloud Run)
    from google.cloud import firestore  # type: ignore
    _fs = firestore.Client() if os.environ.get("GOOGLE_CLOUD_PROJECT") else None
except Exception:
    _fs = None


def _store_local(payload: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with (DATA_DIR / "reports.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def _store(payload: dict) -> None:
    if _fs is not None:
        try:
            _fs.collection("live_reports").add(payload)
            return
        except Exception:  # Firestore unavailable (e.g. DB not created) -> degrade
            pass
    _store_local(payload)


def _recent(limit: int = 50) -> list[dict]:
    if _fs is not None:
        try:
            q = _fs.collection("live_reports").order_by(
                "received", direction=firestore.Query.DESCENDING).limit(limit)
            return [d.to_dict() for d in q.stream()]
        except Exception:
            pass
    path = DATA_DIR / "reports.jsonl"
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()[-limit:]
    return [json.loads(x) for x in reversed(lines)]


@app.post("/api/report")
async def api_report(request: Request, authorization: str = Header(default="")):
    if TOKEN and authorization != f"Bearer {TOKEN}":
        raise HTTPException(status_code=401, detail="unauthorized")
    payload = await request.json()
    payload["received"] = datetime.now(timezone.utc).isoformat()
    _store(payload)
    return {"ok": True}


@app.get("/api/reports")
async def api_reports(limit: int = 50):
    return JSONResponse(_recent(limit))


@app.get("/api/health")  # NOTE: /healthz is a GFE-reserved path on Cloud Run — never reaches the app
async def health():
    return {"ok": True}


# 本番モデルの解説（静的ページ）。数値は OOS のみ（AGENTS.md「OOS絶対規律」）。
_MODEL_PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>本番モデル解説 — A-Tokyo 場中トレード</title>
<style>body{font-family:sans-serif;max-width:960px;margin:2rem auto;color:#111;padding:0 12px;line-height:1.65}
table{border-collapse:collapse;width:100%;margin:1rem 0}
td,th{border:1px solid #ddd;padding:6px 10px;text-align:left;font-size:14px}
th{background:#f4f5f7} h1{color:#4f46e5} h2{color:#4f46e5;border-bottom:2px solid #eef;padding-bottom:4px;margin-top:2.2rem}
pre{background:#0f172a;color:#d1e7ff;padding:14px;border-radius:8px;overflow-x:auto;font-size:12.5px;line-height:1.5}
code{background:#eef;padding:1px 5px;border-radius:4px;font-size:13px}
.chip{display:inline-block;font-size:12px;background:#eef;color:#4f46e5;border-radius:4px;padding:2px 8px;margin-right:6px}
.warn{background:#fefce8;border-left:4px solid #eab308;padding:10px 14px;border-radius:4px;font-size:14px}
.note{color:#555;font-size:13px}</style>
<p><a href="/">← 管理画面トップ</a></p>
<h1>🧠 本番モデル解説</h1>
<p><span class=chip>ensemble_core</span><span class=chip>¥20M・信用2.0倍</span><span class=chip>場中フラット</span>
<span class=chip>個別株のみ</span><span class=chip>一日信用</span></p>

<h2>1. 何が動いているか（一言で）</h2>
<p>「<b>夜間に個別要因で飛んだ寄付きギャップは日中に戻りやすい</b>」という平均回帰アノマリーを、
毎朝寄付きでロング/ショート同額建て（ドルニュートラル）、大引けで全部手仕舞う戦略です。
オーバーナイト保有ゼロ・個別株のみ・単元(100株)の整数ロット。
2つの独立なシグナル（機械学習＋ルール）に資本を50/50で分けたアンサンブル
<code>ensemble_core</code> が本番モデルです。</p>

<h2>2. アーキテクチャ全体図</h2>
<pre>
┌─ データ（J-Quants Premium・毎朝更新）──────────────────────────┐
│ 日足OHLCV(2018+) / 33業種指数 / 先物ナイト / 銘柄マスタ(貸借区分) │
└──────────────┬────────────────────────────────────┘
               ▼ 特徴量パネル（全てPIT: 寄付き時点で既知の情報のみ・当日終値/出来高は不使用）
┌─ スリーブ1: ml_mag_adaptive（資本50%）─────┐ ┌─ スリーブ2: svdn_concentrated（資本50%）──┐
│ ridge回帰・9特徴量（下表）               │ │ ルール系: 33業種内の相対ギャップを      │
│ 目的変数: 当日寄→引リターンの日次平均差  │ │ 銘柄ボラで基準化した逆張りスコア        │
│ 学習: 過去年→翌年を予測（年次WF・リーク  │ │ （セクター要因×ボラ二重中立・集中版）   │
│ なしはテストで機械的に保証）             │ │ 学習パラメータなし＝過学習リスク小      │
│ ウェイト: |予測|比例（上限3×等金額）     │ │ 相関 0.31 ＝ 分散効果でDD半減           │
└──────────────┬─────────────┘ └──────────────┬────────────┘
               ▼ 銘柄選定: 各スリーブ スコア上位8銘柄ロング・下位8銘柄ショート（統合後 実質13〜16/側）
               ▼ 制約: ショートは貸借銘柄のみ・売買代金≥¥10億・価格規制トリガー銘柄は50単元まで
               ▼ サイズ: グロス目標 = 保証金¥20M × 信用倍率2.0 = ¥40M（拘束保証金=ストップ高×30%で日次縮小）
┌─ 執行（Windows VPS・kabuステーションAPI）────────────────────────┐
│ 08:56 プラン確定（板の予想始値で最終スコア）→ 寄成(SOR)で新規建て       │
│ 14:55 引成で全量返済 → 15:30 結果をこの管理画面へ自動報告               │
└────────────────────────────────────────────────┘
</pre>

<h2>3. スリーブ1: ml_mag_adaptive（機械学習）</h2>
<p>モデルは<b>ridge回帰</b>（線形・L2正則化）。派手さはないですが、GBM・交互作用等の複雑化は
検証で全て棄却済みで、この規模のデータでは線形が最強でした。モデルファイルは
<code>ridge_2026.json</code>（2018〜2025年末で学習・2026年中は固定・再学習は年1回で十分と検証済み）。</p>
<table><tr><th>特徴量</th><th>意味（全て寄付き時点で既知）</th></tr>
<tr><td>residual_gap</td><td>寄付きギャップの市場平均差（＝個別要因分。主役）</td></tr>
<tr><td>gap_abs</td><td>ギャップの絶対値（非線形の大きさ効果）</td></tr>
<tr><td>prev_intraday</td><td>前日の寄→引リターン（前日の日中モメンタム）</td></tr>
<tr><td>prev_resid_gap</td><td>前日の残差ギャップ（連日ギャップの文脈）</td></tr>
<tr><td>ivol</td><td>ラグ付き20日ボラ（当日終値を含むvol20は使用禁止＝リーク）</td></tr>
<tr><td>liq_rank</td><td>流動性ランク</td></tr>
<tr><td>amihud20</td><td>Amihud非流動性（出来高由来・前日まで）</td></tr>
<tr><td>idio_gap2</td><td>ギャップの33業種指数ギャップ差（＝真の個別分を精緻化）</td></tr>
<tr><td>sector_index_gap</td><td>33業種指数自体の寄付きギャップ</td></tr></table>
<p>目的変数はその日の<b>クロスセクション平均を引いた寄→引リターン</b>（demeaned）。
「市場が上がるか」ではなく「どの銘柄が相対的に戻るか」だけを学習します。
ウェイトは予測の強さに比例（|予測|比例・1銘柄上限3×等金額）— 自信のある日に厚く張る設計で、
等ウェイト比 +1.3〜2.3 Sharpe の寄与が検証済みです。</p>

<h2>4. スリーブ2: svdn_concentrated（ルール）</h2>
<p>業種全体の動き（継続しやすい）を除いた<b>業種内の相対ギャップ</b>を、銘柄ボラで割って
基準化した逆張りスコア。シグナル上位だけに絞った集中版です。学習パラメータを持たないため
過学習リスクが小さく、ML系と相関0.31しかないのがアンサンブルの肝
（OOSでSharpe押し上げ＋最大DDをほぼ半減）。</p>

<h2>5. 成績（OOSのみ表示）</h2>
<div class=warn><b>OOS絶対規律</b>: このページの数値は全て、モデル・構成の選択に使っていない期間
（Out-of-Sample）のシミュレーションです。選択に使った期間（In-Sample）の成績は原則として
楽観的に歪むため、表示も判断利用も禁止しています。</div>
<table><tr><th>期間（¥20M・8銘柄/側・信用2.0倍・片道7bps想定）</th><th>年率</th><th>Sharpe</th><th>日次勝率</th><th>最大DD</th></tr>
<tr><td>OOS 2024-01〜2026-07</td><td>+101%</td><td>3.21</td><td>58.9%</td><td>−17.9%</td></tr>
<tr><td>うち2025年</td><td>+86.5%</td><td>2.74</td><td>59.1%</td><td>−17.9%</td></tr>
<tr><td>うち2026年(〜7/24)</td><td>+106.4%</td><td>3.18</td><td>55.2%</td><td>−12.7%</td></tr></table>
<p class=note>日次損益の肌感（直近1年・¥20M）: 平均+5.5万円/日・中央値+2.3万円・悪い日5%タイル−58万円・
過去最悪日−115万円（元本の−5.8%）。コスト感応度(OOS): 実効10bpsでも年率75%/Sh2.4、損益分岐は保守側で~14bps。</p>

<h2>6. リスク管理・停止基準（過去2,067日の実分布から逆算）</h2>
<table><tr><th>トリガー</th><th>水準（¥20M基準）</th><th>アクション</th></tr>
<tr><td>単日損失</td><td>〜−120万円</td><td>停止しない（過去最悪−115万円＝経験内）</td></tr>
<tr><td>連敗</td><td>3連敗</td><td>コスト実測の点検のみ（ほぼ毎月起きる正常事象）</td></tr>
<tr><td>連敗</td><td>6連敗 / 8連敗</td><td>サイズ半減 / 停止・点検</td></tr>
<tr><td>週次損失</td><td>−100万 / −160万 / −240万</td><td>注意 / 半減 / 停止</td></tr>
<tr><td>累積DD</td><td>−300万(−15%) / −400万(−20%)</td><td>半減 / 停止・全面見直し</td></tr></table>

<h2>7. 検証の履歴（なぜこの構成を信じるか）</h2>
<ul>
<li><b>リーク防止の機械的保証</b>: 「最終年のデータを改変しても過去年の予測が1bitも変わらない」テストが常時CI相当で通っている（学習は必ず過去→予測は未来）。</li>
<li><b>敵対的検証</b>: 主要な数値は独立再実装（コードを見ずに仕様から書き直す）で±数%一致を確認済み。</li>
<li><b>実取引可能性の監査</b>: 直近1年のショートの98.4%が現行貸借銘柄・参加率p95は売買代金の0.11%・
価格規制キャップの実効1.15%。執行はSOR経由の寄成/引成のみ（実接続で検証済み）。</li>
<li><b>棄却済みの代表例</b>（IS好成績→OOS/現実性で崩壊）: 日中±バリア・遅延エントリー・GBM・
ボラターゲティング・決算日除外・四半期再学習・demeaned×rankブレンド・売り制約解除の高Sharpe(借株不能)。</li>
</ul>
<p class=note>正本ドキュメント: リポジトリの AGENTS.md（確定知見・棄却リスト）と
trading/jp_intraday/live/README.md（運用手順・停止基準）。本ページは 2026-07-29 時点の要約。</p>
"""


@app.get("/model", response_class=HTMLResponse)
async def model_page():
    return _MODEL_PAGE


@app.get("/", response_class=HTMLResponse)
async def home():
    reports = _recent(30)
    latest_plan = next((r for r in reports if r.get("event") in ("plan", "entry")), None)
    rows = ""
    for r in reports:
        rows += (f"<tr><td>{r.get('received','')[:19]}</td><td>{r.get('event')}</td>"
                 f"<td>{r.get('env')}</td><td>{r.get('strategy')}</td>"
                 f"<td>{'実発注' if r.get('orders_enabled') else 'dry'}</td></tr>")
    plan_html = "<p>本日のプランはまだありません。</p>"
    if latest_plan:
        pr = (latest_plan.get("data", {}) or {}).get("plan", [])
        pr = sorted(pr, key=lambda x: -abs(float(x.get("est_yen", 0))))  # 建玉量順
        cells = "".join(
            f"<tr><td>{x.get('symbol')}</td><td>{x.get('name','')}</td>"
            f"<td>{x.get('side_label')}</td><td>{x.get('qty')}</td>"
            f"<td>¥{float(x.get('est_yen',0)):,.0f}</td></tr>" for x in pr)
        plan_html = (f"<h2>最新プラン（{latest_plan.get('time','')[:19]}）</h2>"
                     f"<table><tr><th>コード</th><th>銘柄</th><th>売買</th><th>株数</th><th>建玉¥</th></tr>{cells}</table>")
    return f"""<!doctype html><meta charset=utf-8><title>A-Tokyo 場中トレード</title>
<style>body{{font-family:sans-serif;max-width:960px;margin:2rem auto;color:#111}}
table{{border-collapse:collapse;width:100%;margin:1rem 0}}
td,th{{border:1px solid #ddd;padding:6px 10px;text-align:left;font-size:14px}}
th{{background:#f4f5f7}} h1{{color:#4f46e5}}</style>
<h1>📈 場中フラット・トレード 管理画面</h1>
<p>auカブコム kabuステーションAPI ／ 個別株 L/S ／ オーバーナイトなし
　<a href="/model">🧠 本番モデル解説（何がどう動いているか）</a></p>
{plan_html}
<h2>最近のイベント</h2>
<table><tr><th>受信</th><th>event</th><th>env</th><th>戦略</th><th>モード</th></tr>{rows}</table>
"""
