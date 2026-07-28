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
<p>auカブコム kabuステーションAPI ／ 個別株 L/S ／ オーバーナイトなし</p>
{plan_html}
<h2>最近のイベント</h2>
<table><tr><th>受信</th><th>event</th><th>env</th><th>戦略</th><th>モード</th></tr>{rows}</table>
"""
