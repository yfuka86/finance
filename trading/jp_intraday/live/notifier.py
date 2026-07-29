"""Slack notifications for the live trader (chat.postMessage).

送るもの: plan / entry / exit / state の要約と、失敗・異常のアラート。
**fail-soft が絶対条件** — Slack が落ちていても発注フローは止めない。post() は例外を
投げず、結果を dict で返すだけにしてある（呼び出し側は無視してよい）。

設定は .env（Git管理外）:
    SLACK_BOT_TOKEN=xoxb-...     # 必要スコープは chat:write のみ
    SLACK_CHANNEL=C0BLLFT2Y0H    # チャンネルID（bot を /invite しておくこと）
    SLACK_NOTIFY=1               # 0 で全通知オフ（切り分け用）

モック環境（preflight）のイベントは送らない。毎朝の preflight で通知が二重に流れるため。
エラー通知はどの環境でも送る（実害の検知が目的なので取りこぼさない）。
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

from data.collectors.config import _load_local_env

API_URL = "https://slack.com/api/chat.postMessage"
TIMEOUT = 8.0

ENV_MARK = {"prod": "🔴本番", "test": "🟠検証", "mock": "⚪モック"}


@dataclass(frozen=True)
class SlackConfig:
    token: str = ""
    channel: str = ""
    enabled: bool = True

    @classmethod
    def from_env(cls) -> "SlackConfig":
        _load_local_env()
        return cls(
            token=os.environ.get("SLACK_BOT_TOKEN", "").strip(),
            channel=os.environ.get("SLACK_CHANNEL", "").strip(),
            enabled=os.environ.get("SLACK_NOTIFY", "1").strip().lower() not in ("0", "false", "no"),
        )

    @property
    def usable(self) -> bool:
        return bool(self.enabled and self.token and self.channel)


def post(text: str, cfg: SlackConfig | None = None) -> dict:
    """Post to Slack. Never raises — returns {'sent': bool, ...}."""
    cfg = cfg or SlackConfig.from_env()
    if not cfg.usable:
        return {"sent": False, "reason": "slack not configured (SLACK_BOT_TOKEN/SLACK_CHANNEL)"}
    body = json.dumps({"channel": cfg.channel, "text": text,
                       "unfurl_links": False, "unfurl_media": False}).encode()
    req = urllib.request.Request(
        API_URL, data=body, method="POST",
        headers={"Content-Type": "application/json; charset=utf-8",
                 "Authorization": f"Bearer {cfg.token}"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        # Slack は失敗も HTTP 200 + {"ok": false, "error": ...} で返す
        return {"sent": bool(data.get("ok")), "error": data.get("error"), "ts": data.get("ts")}
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return {"sent": False, "error": f"{type(exc).__name__}: {exc}"}


# ── formatting ──────────────────────────────────────────────────────
def _yen(v: float | None) -> str:
    if v is None:
        return "―"
    v = float(v)
    if abs(v) >= 1e8:
        return f"¥{v / 1e8:.2f}億"
    if abs(v) >= 1e6:
        return f"¥{v / 1e6:.1f}M"
    return f"¥{v:,.0f}"


def _signed_yen(v: float | None) -> str:
    if v is None:
        return "―"
    return ("+" if float(v) >= 0 else "−") + _yen(abs(float(v)))


def _order_counts(orders: list[dict]) -> tuple[int, int, int]:
    """(成功, 失敗, スキップ) — dry-run は成功に数える（送信された扱い）。"""
    ok = fail = skip = 0
    for o in orders:
        if o.get("skipped"):
            skip += 1
        elif o.get("error"):
            fail += 1
        else:
            ok += 1
    return ok, fail, skip


def _fail_lines(orders: list[dict], limit: int = 5) -> list[str]:
    bad = [o for o in orders if o.get("error")]
    lines = [f"  • {o.get('symbol')}: {str(o.get('error'))[:160]}" for o in bad[:limit]]
    if len(bad) > limit:
        lines.append(f"  • …ほか {len(bad) - limit} 件")
    return lines


def format_event(event: str, data: dict, env: str, stamp: str) -> str | None:
    """Build the Slack text for one event. Returns None if nothing worth sending."""
    head = f"[{ENV_MARK.get(env, env)}] {stamp[:16].replace('T', ' ')}"

    if event == "plan":
        meta = data.get("meta") or {}
        banned = meta.get("shorts_banned")
        n_banned = len(banned) if isinstance(banned, (list, set, tuple)) else (banned or 0)
        return (f"🗒 *plan* {head}\n"
                f"L {meta.get('n_long', 0)} / S {meta.get('n_short', 0)} ・ "
                f"グロス {_yen(meta.get('gross_yen'))} ・ "
                f"板カバレッジ {float(meta.get('coverage', 0)) * 100:.0f}% ・ "
                f"データ日 {meta.get('data_date', '―')}"
                + (f"\n売り禁止 {n_banned} 銘柄" if n_banned else ""))

    if event in ("entry", "exit"):
        orders = data.get("orders") or []
        ok, fail, skip = _order_counts(orders)
        icon, label = ("🟢", "entry 新規建て") if event == "entry" else ("🔻", "exit 返済")
        lines = [f"{icon} *{label}* {head}",
                 f"送信 {len(orders)} 件（成功 {ok} / 失敗 {fail} / スキップ {skip}）"]
        if event == "entry":
            meta = data.get("meta") or {}
            if meta.get("gross_yen"):
                lines[-1] += f" 想定グロス {_yen(meta['gross_yen'])}"
        if fail:
            lines.append("⚠️ 失敗:")
            lines += _fail_lines(orders)
        return "\n".join(lines)

    if event == "state":
        pos = data.get("positions") or []
        margin = data.get("margin") or {}
        pnl = sum(float(p.get("ProfitLoss") or 0) for p in pos) if pos else 0.0
        lines = [f"📊 *state* {head}",
                 f"建玉 {len(pos)} 件 ・ 保証金 {_yen(margin.get('MarginAccountWallet'))}"
                 + (f" ・ 評価損益 {_signed_yen(pnl)}" if pos else "")]
        if pos:
            # 場中フラット戦略なので大引け後に建玉が残るのは異常（翌朝の建余力にも響く）
            lines.append("🚨 *引け後に建玉が残っています* — 返済漏れの可能性。要確認")
            for p in pos[:8]:
                lines.append(f"  • {p.get('Symbol')} {p.get('SymbolName', '')} "
                             f"side={p.get('Side')} qty={p.get('LeavesQty')}")
        return "\n".join(lines)

    return None


# ── entry points ────────────────────────────────────────────────────
def notify_event(event: str, data: dict, env: str, stamp: str) -> dict:
    """Called from reporter.report for every live event. Mock は送らない。"""
    if env == "mock":
        return {"sent": False, "reason": "mock env (not notified)"}
    text = format_event(event, data, env, stamp)
    if not text:
        return {"sent": False, "reason": f"no formatter for event={event}"}
    return post(text)


def notify_error(title: str, detail: str = "", env: str = "") -> dict:
    """Alert on failures. 環境に関係なく送る（取りこぼさないため）。"""
    head = f"[{ENV_MARK.get(env, env)}] " if env else ""
    text = f"🚨 *{title}* {head}".rstrip()
    if detail:
        text += "\n```\n" + detail.strip()[:2500] + "\n```"
    return post(text)


def main(argv: list[str] | None = None) -> int:
    """CLI: 疎通確認や PowerShell からの単発通知に使う。

        python -m trading.jp_intraday.live.notifier --test
        python -m trading.jp_intraday.live.notifier --error "タスク失敗" --detail "..."
    """
    import argparse
    ap = argparse.ArgumentParser(description="Slack へ通知を送る")
    ap.add_argument("--text", help="そのまま送る本文")
    ap.add_argument("--error", help="アラートのタイトル")
    ap.add_argument("--detail", default="", help="アラートの詳細（コードブロックで送る）")
    ap.add_argument("--test", action="store_true", help="疎通確認メッセージを送る")
    args = ap.parse_args(argv)

    if args.test:
        res = post("✅ 疎通確認: トレードボットから Slack へ送信できています")
    elif args.error:
        res = notify_error(args.error, args.detail)
    elif args.text:
        res = post(args.text)
    else:
        ap.error("--text / --error / --test のいずれかを指定してください")
    print(json.dumps(res, ensure_ascii=False))
    return 0 if res.get("sent") else 1


if __name__ == "__main__":
    raise SystemExit(main())
