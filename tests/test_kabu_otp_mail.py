"""Offline tests for the Gmail one-time-code reader (no network / no Gmail needed)."""
import email
import time
import unittest
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from trading.jp_intraday.live import otp_mail
from trading.jp_intraday.live.otp_mail import MailConfig, extract_code, mask, message_text

BODY = """auカブコム証券です。
ワンタイム認証コードは 483920 です。
有効期限は5分です。口座番号 01901974 のお客様。
"""


def build_mail(sender="au カブコム証券 <no-reply@kabu.co.jp>", subject="ワンタイム認証コードのお知らせ",
               body=BODY, charset="iso-2022-jp", html=False):
    if html:
        msg = MIMEMultipart("alternative")
        msg.attach(MIMEText("", "plain", "utf-8"))
        msg.attach(MIMEText(f"<html><body><p>{body}</p></body></html>", "html", "utf-8"))
    else:
        msg = MIMEText(body, "plain", charset)
    msg["From"] = sender
    msg["Subject"] = subject
    return msg.as_bytes()


class ExtractCodeTest(unittest.TestCase):
    def test_picks_code_next_to_keyword(self):
        self.assertEqual(extract_code(BODY), "483920")

    def test_account_number_is_not_mistaken_for_a_code(self):
        # 口座番号(8桁)や日付が本文にあってもキーワード近傍を優先する
        text = "口座番号 01901974\n2026年07月29日\n認証コードは 771203 です"
        self.assertEqual(extract_code(text), "771203")

    def test_standalone_six_digits_fallback(self):
        self.assertEqual(extract_code("コードをご入力ください\n\n914725\n\n有効期限5分"), "914725")

    def test_none_when_no_code(self):
        self.assertIsNone(extract_code("メンテナンスのお知らせ。詳細はサイトをご覧ください。"))

    def test_mask_never_leaks_the_middle(self):
        self.assertEqual(mask("483920"), "4****0")


class MessageTextTest(unittest.TestCase):
    def test_iso2022jp_body_decodes(self):
        msg = email.message_from_bytes(build_mail())
        self.assertEqual(extract_code(message_text(msg)), "483920")

    def test_html_only_body_is_stripped(self):
        msg = email.message_from_bytes(build_mail(html=True))
        self.assertEqual(extract_code(message_text(msg)), "483920")


class FakeIMAP:
    """Minimal in-memory IMAP4_SSL stand-in (uid SEARCH/FETCH only)."""

    inbox: list = []       # [(internaldate(datetime), raw_bytes)]
    last_readonly = None

    def __init__(self, host, *a, **kw):
        self.host = host

    def login(self, user, password):
        if password != "app-password":
            raise Exception("AUTHENTICATIONFAILED")
        return "OK", [b""]

    def select(self, mailbox, readonly=False):
        FakeIMAP.last_readonly = readonly
        return "OK", [b"1"]

    def uid(self, cmd, *args):
        if cmd == "SEARCH":
            return "OK", [b" ".join(str(i + 1).encode() for i in range(len(self.inbox)))]
        uid = int(args[0])
        when, raw = self.inbox[uid - 1]
        if "INTERNALDATE" in args[1]:
            stamp = time.strftime('%d-%b-%Y %H:%M:%S +0000', when.utctimetuple())
            return "OK", [b'1 (INTERNALDATE "' + stamp.encode() + b'")']
        return "OK", [((b"1 (BODY[] {%d}" % len(raw)), raw), b")"]

    def logout(self):
        return "OK", [b""]


class ScanTest(unittest.TestCase):
    def setUp(self):
        self._real = otp_mail.imaplib.IMAP4_SSL
        otp_mail.imaplib.IMAP4_SSL = FakeIMAP
        self.cfg = MailConfig(user="yfuka86@gmail.com", app_password="app-password")
        self.now = datetime.now(timezone.utc)

    def tearDown(self):
        otp_mail.imaplib.IMAP4_SSL = self._real

    def test_finds_recent_code(self):
        FakeIMAP.inbox = [(self.now - timedelta(seconds=30), build_mail())]
        cands = otp_mail.scan(self.cfg, self.now - timedelta(minutes=2))
        self.assertEqual([c.code for c in cands], ["483920"])
        self.assertTrue(FakeIMAP.last_readonly, "既読化しないよう readonly で開くこと")

    def test_old_code_is_ignored(self):
        # 前回ログイン時の古いコードを使い回さないことが、この仕組みの安全弁
        FakeIMAP.inbox = [(self.now - timedelta(hours=6), build_mail())]
        self.assertEqual(otp_mail.scan(self.cfg, self.now - timedelta(minutes=2)), [])

    def test_other_senders_are_ignored(self):
        FakeIMAP.inbox = [(self.now, build_mail(sender="phish@example.com",
                                                body="認証コードは 111111 です"))]
        self.assertEqual(otp_mail.scan(self.cfg, self.now - timedelta(minutes=2)), [])

    def test_wait_for_code_returns_none_on_timeout(self):
        FakeIMAP.inbox = []
        self.assertIsNone(otp_mail.wait_for_code(self.cfg, self.now, timeout_sec=1, poll_sec=1))

    def test_missing_credentials_raise(self):
        with self.assertRaises(ValueError):
            otp_mail.scan(MailConfig(user="", app_password=""), self.now)


if __name__ == "__main__":
    unittest.main()
