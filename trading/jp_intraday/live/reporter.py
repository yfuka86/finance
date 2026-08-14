"""Send the day's plan / fills / positions / P&L to the web dashboard (a-tokyo.jp).

POSTs a JSON payload to cfg.report_url with a bearer token. Also writes a local
JSONL audit log so nothing is lost if the network/endpoint is down, and posts a
short human-readable summary to Slack (notifier — fail-soft).
"""
from __future__ import annotations

import json
from pathlib import Path

import requests

from . import notifier
from .config import PROJECT_ROOT, LiveConfig

_LOG_DIR = PROJECT_ROOT / "data" / "live_reports"


def _audit(payload: dict) -> None:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    with (_LOG_DIR / "reports.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def report(cfg: LiveConfig, event: str, data: dict, stamp: str) -> dict:
    """event: 'plan' | 'entry' | 'exit' | 'state'. ``stamp`` is caller-supplied ISO time."""
    payload = {"event": event, "time": stamp, "env": cfg.env, "strategy": cfg.strategy,
               "capital_yen": cfg.capital_yen, "orders_enabled": cfg.orders_enabled, "data": data}
    _audit(payload)
    # Slack は best-effort（notify_event は例外を投げない）。ダッシュボード送信の成否とは独立。
    slack = notifier.notify_event(event, data, cfg.env, stamp)
    if not cfg.report_url:
        return {"sent": False, "reason": "REPORT_URL empty (audit-logged only)", "slack": slack}
    try:
        headers = {"Content-Type": "application/json"}
        if cfg.report_token:
            headers["Authorization"] = f"Bearer {cfg.report_token}"
        r = requests.post(cfg.report_url, json=payload, headers=headers, timeout=15)
        return {"sent": r.ok, "status": r.status_code, "slack": slack}
    except Exception as exc:  # noqa: BLE001
        return {"sent": False, "reason": str(exc), "slack": slack}
