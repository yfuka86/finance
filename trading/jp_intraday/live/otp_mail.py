"""Fetch the kabuステーション one-time login code from Gmail over IMAP.

kabuステーションのログインは毎回メールのワンタイム認証コードを要求する
（「この端末を信頼する」相当のオプションは無い）。無人VPSで朝の自動ログインを
完結させるため、届いたメールをIMAPで読んでコードだけ取り出す。

認証は **Gmailのアプリパスワード**（2段階認証を有効にすると発行できる16桁）。
OAuthのブラウザ同意もトークン更新も不要で、無人環境で最も壊れにくい。
資格情報は DPAPI 暗号化して `data/live_reports/.gmail_otp.xml` に置き、
`scripts/fetch_otp.ps1` が復号して環境変数で子プロセスへ渡す（平文をディスクに置かない）。
.env の `OTP_IMAP_USER` / `OTP_IMAP_APP_PASSWORD` でも動く（後方互換・非推奨）。

安全側の設計:
  * メールボックスは **readonly** で開く（既読化も削除もしない）
  * `--since-epoch` より前に届いたメールは無視する ＝ **古いコードを使い回さない**
  * コードは stdout に1行だけ出す（ログ・stderr にはマスクした形しか出さない）

使い方:
    # 直近のコードを最大180秒待って取得（PowerShell から呼ぶ想定）
    python -m trading.jp_intraday.live.otp_mail --since-epoch 1753771000 --timeout 180
    # 受信メールの形（差出人・件名・コード検出可否）を確認するだけ
    python -m trading.jp_intraday.live.otp_mail --probe
"""
from __future__ import annotations

import argparse
import email
import imaplib
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.header import decode_header, make_header
from email.message import Message
from typing import Iterable

from data.collectors.config import _load_local_env

DEFAULT_HOST = "imap.gmail.com"
DEFAULT_MAILBOX = "INBOX"
# 差出人の既定フィルタ（au カブコム証券）。実際のアドレスは --probe で確認して
# OTP_MAIL_FROM で上書きできる。緩めにしておき、最終的な安全弁は受信時刻とする。
DEFAULT_FROM_PATTERN = r"kabu|カブコム|au\s*kabucom"
# 「認証コード」等の近傍にある数字列だけを拾う（口座番号や日付を誤検出しないため）
KEYWORD_PATTERN = r"ワンタイム|認証コード|確認コード|セキュリティコード|verification code|one[- ]?time"
CODE_PATTERN = r"(?<![0-9])([0-9]{4,8})(?![0-9])"
KEYWORD_WINDOW = 120  # キーワード前後この文字数だけを探索する


def mask(code: str) -> str:
    """Log-safe form of a code: 123456 -> 1****6."""
    if len(code) <= 2:
        return "*" * len(code)
    return f"{code[0]}{'*' * (len(code) - 2)}{code[-1]}"


@dataclass(frozen=True)
class MailConfig:
    user: str = ""
    app_password: str = ""
    host: str = DEFAULT_HOST
    mailbox: str = DEFAULT_MAILBOX
    from_pattern: str = DEFAULT_FROM_PATTERN

    @classmethod
    def from_env(cls) -> "MailConfig":
        _load_local_env()
        return cls(
            user=os.environ.get("OTP_IMAP_USER", "").strip(),
            # アプリパスワードは表示が4桁区切りなので空白を落として受ける
            app_password=os.environ.get("OTP_IMAP_APP_PASSWORD", "").replace(" ", ""),
            host=os.environ.get("OTP_IMAP_HOST", DEFAULT_HOST).strip() or DEFAULT_HOST,
            mailbox=os.environ.get("OTP_IMAP_MAILBOX", DEFAULT_MAILBOX).strip() or DEFAULT_MAILBOX,
            from_pattern=os.environ.get("OTP_MAIL_FROM", DEFAULT_FROM_PATTERN),
        )

    def validate(self) -> None:
        if not self.user or not self.app_password:
            raise ValueError(
                "OTP_IMAP_USER / OTP_IMAP_APP_PASSWORD が未設定です。"
                "scripts\\setup_gmail_otp.ps1 で登録してください。"
            )


def _decode(value: str | None) -> str:
    """Decode an RFC2047 header (日本語メールは ISO-2022-JP が多い)."""
    if not value:
        return ""
    try:
        return str(make_header(decode_header(value)))
    except Exception:
        return value


def _part_text(part: Message) -> str:
    try:
        payload = part.get_payload(decode=True)
    except Exception:
        return ""
    if not payload:
        return ""
    charset = part.get_content_charset() or "utf-8"
    for enc in (charset, "utf-8", "iso-2022-jp", "cp932"):
        try:
            return payload.decode(enc, errors="strict")
        except (LookupError, UnicodeDecodeError):
            continue
    return payload.decode("utf-8", errors="ignore")


def message_text(msg: Message) -> str:
    """Subject + body as plain text (HTML tags stripped)."""
    chunks = [_decode(msg.get("Subject"))]
    plain, html = [], []
    for part in msg.walk() if msg.is_multipart() else [msg]:
        if part.get_content_maintype() == "multipart":
            continue
        ctype = part.get_content_type()
        if ctype == "text/plain":
            plain.append(_part_text(part))
        elif ctype == "text/html":
            html.append(_part_text(part))
    body = "\n".join(plain) if any(p.strip() for p in plain) else "\n".join(html)
    if not plain or not any(p.strip() for p in plain):
        body = re.sub(r"<[^>]+>", " ", body)
    chunks.append(body)
    return "\n".join(chunks)


def extract_code(text: str) -> str | None:
    """Pull the one-time code out of a mail body.

    「認証コード」等のキーワードの**後ろ**を最優先で探す（日本語メールは
    「認証コードは 123456 です」の語順）。口座番号(8桁)や日付を拾わないための制約で、
    後ろに無ければ直前、それも無ければ「行内で独立した6桁」をフォールバックに使う。
    """
    keywords = list(re.finditer(KEYWORD_PATTERN, text, re.IGNORECASE))
    for kw in keywords:
        m = re.search(CODE_PATTERN, text[kw.end(): kw.end() + KEYWORD_WINDOW])
        if m:
            return m.group(1)
    for kw in keywords:
        found = re.findall(CODE_PATTERN, text[max(0, kw.start() - KEYWORD_WINDOW): kw.start()])
        if found:
            return found[-1]   # キーワードに最も近いもの
    for line in text.splitlines():
        s = line.strip()
        if re.fullmatch(r"[0-9]{6}", s):
            return s
    return None


def _search_since(imap: imaplib.IMAP4, since: datetime) -> list[bytes]:
    """UIDs of mails newer than `since` (IMAP SINCE is date-granular → 1日広く取る)."""
    date_str = (since - timedelta(days=1)).strftime("%d-%b-%Y")
    typ, data = imap.uid("SEARCH", None, "SINCE", date_str)
    if typ != "OK" or not data or not data[0]:
        return []
    return data[0].split()


def _internal_date(imap: imaplib.IMAP4, uid: bytes) -> datetime | None:
    typ, data = imap.uid("FETCH", uid, "(INTERNALDATE)")
    if typ != "OK" or not data or not data[0]:
        return None
    raw = data[0] if isinstance(data[0], bytes) else str(data[0]).encode()
    m = re.search(rb'INTERNALDATE "([^"]+)"', raw)
    if not m:
        return None
    ts = imaplib.Internaldate2tuple(b'INTERNALDATE "' + m.group(1) + b'"')
    return datetime.fromtimestamp(time.mktime(ts), tz=timezone.utc) if ts else None


@dataclass
class Candidate:
    uid: str
    received: datetime | None
    sender: str
    subject: str
    code: str | None


def scan(cfg: MailConfig, since: datetime, limit: int = 25) -> list[Candidate]:
    """Newest-first scan of mails at/after `since` that look like OTP mails."""
    cfg.validate()
    imap = imaplib.IMAP4_SSL(cfg.host)
    try:
        imap.login(cfg.user, cfg.app_password)
        imap.select(cfg.mailbox, readonly=True)  # 既読化しない
        uids = _search_since(imap, since)
        out: list[Candidate] = []
        for uid in reversed(uids[-max(limit, 1) * 4:]):  # 新しい方から
            received = _internal_date(imap, uid)
            if received is not None and received < since:
                continue
            typ, data = imap.uid("FETCH", uid, "(BODY.PEEK[])")
            if typ != "OK" or not data or not isinstance(data[0], tuple):
                continue
            msg = email.message_from_bytes(data[0][1])
            sender = _decode(msg.get("From"))
            if cfg.from_pattern and not re.search(cfg.from_pattern, sender, re.IGNORECASE):
                continue
            text = message_text(msg)
            out.append(Candidate(uid.decode(), received, sender,
                                 _decode(msg.get("Subject")), extract_code(text)))
            if len(out) >= limit:
                break
        return out
    finally:
        try:
            imap.logout()
        except Exception:
            pass


def wait_for_code(cfg: MailConfig, since: datetime, timeout_sec: int = 180,
                  poll_sec: int = 10, log=lambda m: None) -> str | None:
    """Poll the mailbox until an OTP mail newer than `since` shows up."""
    deadline = time.monotonic() + timeout_sec
    attempt = 0
    while True:
        attempt += 1
        try:
            for cand in scan(cfg, since, limit=5):
                if cand.code:
                    log(f"コード検出 ({attempt}回目): from=[{cand.sender}] "
                        f"subject=[{cand.subject}] code={mask(cand.code)}")
                    return cand.code
            log(f"{attempt}回目: 該当メールなし（{since:%H:%M:%S} 以降を検索）")
        except Exception as exc:  # 一時的なIMAPエラーで諦めない
            log(f"{attempt}回目: IMAPエラー: {type(exc).__name__}: {exc}")
        if time.monotonic() + poll_sec >= deadline:
            return None
        time.sleep(poll_sec)


def _log_stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Gmail からワンタイム認証コードを取得する")
    ap.add_argument("--since-epoch", type=float, default=None,
                    help="この時刻(UNIX秒)以降に届いたメールだけを見る（既定: 5分前）")
    ap.add_argument("--timeout", type=int, default=180, help="待機秒数")
    ap.add_argument("--poll", type=int, default=10, help="ポーリング間隔秒")
    ap.add_argument("--probe", action="store_true",
                    help="コードを出力せず、直近の該当メール一覧だけ表示（設定確認用）")
    args = ap.parse_args(list(argv) if argv is not None else None)

    cfg = MailConfig.from_env()
    since = (datetime.fromtimestamp(args.since_epoch, tz=timezone.utc)
             if args.since_epoch else datetime.now(timezone.utc) - timedelta(minutes=5))

    if args.probe:
        since = datetime.now(timezone.utc) - timedelta(days=7)
        try:
            cands = scan(cfg, since, limit=10)
        except Exception as exc:
            _log_stderr(f"ERROR: {type(exc).__name__}: {exc}")
            return 2
        if not cands:
            _log_stderr("直近7日に該当メールがありません（OTP_MAIL_FROM を見直してください）")
            return 4
        for c in cands:
            when = c.received.astimezone().strftime("%m-%d %H:%M") if c.received else "?"
            _log_stderr(f"{when} from=[{c.sender}] subject=[{c.subject}] "
                        f"code={mask(c.code) if c.code else '検出できず'}")
        return 0

    try:
        cfg.validate()
    except ValueError as exc:
        _log_stderr(f"ERROR: {exc}")
        return 2
    code = wait_for_code(cfg, since, args.timeout, args.poll, log=_log_stderr)
    if not code:
        _log_stderr(f"ERROR: {args.timeout}秒以内に認証コードのメールが届きませんでした")
        return 4
    print(code)  # stdout はコード1行だけ（呼び出し側がそのまま受け取る）
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
