"""
eスマート証券 kabuステーションAPI クライアント

レート制限対策:
  kabuステーションAPI は短時間に連続リクエストすると
  {"Code":4001009,"Message":"API実行回数エラー"} を返す。
  全リクエストを最小間隔でスロットリングし、実行回数エラーは自動リトライする。
"""
import json
import threading
import time

import requests

from trading.config import (
    KABU_API_BASE,
    KABU_API_PASSWORD,
    KABU_MIN_INTERVAL,
    KABU_MAX_RETRIES,
)

# --- kabuステーションAPI 定数 ---
# Side
SIDE_SELL = "1"
SIDE_BUY = "2"

# CashMargin
CASH = 1          # 現物
MARGIN_OPEN = 2   # 新規信用
MARGIN_CLOSE = 3  # 返済信用

# MarginTradeType (CashMargin=2/3 のとき必須)
MARGIN_SYSTEM = 1      # 制度信用
MARGIN_GENERAL_LONG = 2   # 一般信用(長期)
MARGIN_GENERAL_DAY = 3    # 一般信用(デイトレード)

# DelivType (受渡区分)
DELIV_UNSPECIFIED = 0  # 指定なし(信用新規はこれ)
DELIV_CASH = 2         # お預り金

# AccountType
ACCOUNT_SPECIFIC = 2  # 特定

# FrontOrderType
ORDER_MARKET = 10  # 成行

# FundType (資金区分)
# 検証環境(18081)で実測した結果:
#   現物買い : "02"(保護) / "AA"(信用代用) のみ受理。"  " は 1010004 預り区分未設定。
#   現物売り : "  " のみ受理。"02"/"AA" は 4001005 変換エラー。
#   信用売買 : 省略可 (指定しても受理される)。
# 省略すると現物は 4001005 パラメータ変換エラーになるため必須。
FUND_PROTECTED = "02"      # 保護預り (現物買い)
FUND_MARGIN_SUBSTITUTE = "AA"  # 信用代用 (現物買い)
FUND_CASH_SELL = "  "      # 現物売り・信用取引

# 一般信用の売建可否を表す /symbol のフィールド
MARGIN_SELL_FIELDS = ("MarginSell", "KCMarginSell")


class KabuApiError(RuntimeError):
    """kabuステーションAPI がエラーコードを返した場合の例外"""

    def __init__(self, status: int, code, message: str, url: str):
        self.status = status
        self.code = code
        self.message = message
        self.url = url
        super().__init__(f"[{status}] Code={code} {message} ({url})")


class RateLimitError(KabuApiError):
    """API実行回数エラー (リトライ対象)"""


class KabuStationClient:
    """kabuステーション REST API クライアント"""

    def __init__(self, base: str = None, min_interval: float = None,
                 max_retries: int = None):
        self.base = base or KABU_API_BASE
        self.token = None
        self.min_interval = KABU_MIN_INTERVAL if min_interval is None else min_interval
        self.max_retries = KABU_MAX_RETRIES if max_retries is None else max_retries
        self._lock = threading.Lock()
        self._last_call = 0.0

    # --- 低レベル ---

    def _throttle(self):
        with self._lock:
            wait = self.min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()

    def _request(self, method: str, path: str, *, headers=None, body=None):
        """スロットリングと実行回数エラーのリトライを挟んだリクエスト"""
        url = f"{self.base}{path}"
        backoff = max(self.min_interval, 0.5)
        last_exc = None
        for attempt in range(self.max_retries + 1):
            self._throttle()
            resp = requests.request(
                method, url,
                headers=headers,
                data=None if body is None else json.dumps(body),
                timeout=30,
            )
            if resp.status_code == 200:
                return resp.json()

            try:
                payload = resp.json()
            except ValueError:
                payload = {}
            code = payload.get("Code")
            message = payload.get("Message", resp.text[:200])

            if "実行回数" in str(message):
                last_exc = RateLimitError(resp.status_code, code, message, url)
                if attempt < self.max_retries:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                raise last_exc
            raise KabuApiError(resp.status_code, code, message, url)
        raise last_exc

    def _headers(self):
        if not self.token:
            raise RuntimeError("未認証です。auth() を先に呼んでください。")
        return {"X-API-KEY": self.token, "content-type": "application/json"}

    # --- 認証 ---

    def auth(self, password: str = None):
        """
        トークン取得。

        password を明示指定した場合は環境変数にフォールバックしない
        (空文字を渡したときに意図しない認証情報が使われるのを防ぐ)。
        """
        pw = KABU_API_PASSWORD if password is None else password
        if not pw:
            raise ValueError(
                "APIパスワードが設定されていません。環境変数 KABU_API_PASSWORD を設定してください。"
            )
        result = self._request(
            "POST", "/token",
            headers={"content-type": "application/json"},
            body={"APIPassword": pw},
        )
        self.token = result["Token"]
        return self.token

    # --- 照会系 ---

    def wallet(self):
        """買付余力"""
        return self._request("GET", "/wallet/cash", headers=self._headers())

    def margin_wallet(self):
        """信用新規建可能額"""
        return self._request("GET", "/wallet/margin", headers=self._headers())

    def positions(self, product: int = None):
        """
        保有銘柄一覧。
        product: 0=すべて, 1=現物, 2=信用, 3=先物, 4=OP
        """
        path = "/positions" if product is None else f"/positions?product={product}"
        return self._request("GET", path, headers=self._headers())

    def orders(self, product: int = None):
        """注文一覧"""
        path = "/orders" if product is None else f"/orders?product={product}"
        return self._request("GET", path, headers=self._headers())

    def board(self, symbol: str, exchange: int = 1):
        """板情報 (exchange: 1=東証)"""
        return self._request(
            "GET", f"/board/{symbol}@{exchange}", headers=self._headers()
        )

    def symbol(self, symbol: str, exchange: int = 1):
        """銘柄情報 (売買単位・信用売建可否など)"""
        return self._request(
            "GET", f"/symbol/{symbol}@{exchange}", headers=self._headers()
        )

    def register(self, symbols: list, exchange: int = 1):
        """銘柄登録 (板情報の配信対象にする)"""
        return self._request(
            "PUT", "/register",
            headers=self._headers(),
            body={"Symbols": [{"Symbol": s, "Exchange": exchange} for s in symbols]},
        )

    # --- 発注系 ---

    def send_order(self, payload: dict):
        """注文発注 (汎用)"""
        return self._request(
            "POST", "/sendorder", headers=self._headers(), body=payload
        )

    def cancel_order(self, order_id: str):
        """注文取消"""
        return self._request(
            "PUT", "/cancelorder",
            headers=self._headers(),
            body={"OrderId": order_id},
        )

    # --- 発注ヘルパー ---

    def _market_order(self, symbol, qty, side, cash_margin, exchange,
                      margin_trade_type=None, deliv_type=DELIV_CASH,
                      fund_type=None, close_positions=None):
        payload = {
            "Symbol": symbol,
            "Exchange": exchange,
            "SecurityType": 1,
            "Side": side,
            "CashMargin": cash_margin,
            "DelivType": deliv_type,
            "AccountType": ACCOUNT_SPECIFIC,
            "Qty": qty,
            "FrontOrderType": ORDER_MARKET,
            "Price": 0,
            "ExpireDay": 0,
        }
        if fund_type is not None:
            payload["FundType"] = fund_type
        if margin_trade_type is not None:
            payload["MarginTradeType"] = margin_trade_type
        if close_positions is not None:
            payload["ClosePositions"] = close_positions
        return payload

    def buy_market(self, symbol: str, qty: int, exchange: int = 1,
                   fund_type: str = FUND_PROTECTED):
        """成行で現物買い"""
        return self.send_order(
            self._market_order(symbol, qty, SIDE_BUY, CASH, exchange,
                               fund_type=fund_type)
        )

    def sell_market(self, symbol: str, qty: int, exchange: int = 1):
        """成行で現物売り (保有株の売却)"""
        return self.send_order(
            self._market_order(symbol, qty, SIDE_SELL, CASH, exchange,
                               fund_type=FUND_CASH_SELL)
        )

    def margin_sell_open(self, symbol: str, qty: int, exchange: int = 1,
                         margin_trade_type: int = MARGIN_SYSTEM):
        """成行で信用新規売り (ショート建て)"""
        return self.send_order(
            self._market_order(
                symbol, qty, SIDE_SELL, MARGIN_OPEN, exchange,
                margin_trade_type=margin_trade_type,
                deliv_type=DELIV_UNSPECIFIED,
            )
        )

    def margin_buy_open(self, symbol: str, qty: int, exchange: int = 1,
                        margin_trade_type: int = MARGIN_SYSTEM):
        """成行で信用新規買い"""
        return self.send_order(
            self._market_order(
                symbol, qty, SIDE_BUY, MARGIN_OPEN, exchange,
                margin_trade_type=margin_trade_type,
                deliv_type=DELIV_UNSPECIFIED,
            )
        )

    def margin_close(self, symbol: str, qty: int, side: str, exchange: int = 1,
                     margin_trade_type: int = MARGIN_SYSTEM,
                     close_positions: list = None):
        """
        成行で信用返済。
        side: 建玉と反対の売買区分 (売建の返済買い = SIDE_BUY)
        close_positions: [{"HoldID": ..., "Qty": ...}] 建玉指定

        検証環境で実測したところ ClosePositionOrder による建玉自動選択は
        1009001「建玉が選択されていません」で拒否されるため、
        ClosePositions での明示指定を必須にしている。
        """
        if not close_positions:
            raise ValueError(
                "close_positions が必要です。/positions の HoldID を指定してください。"
            )
        return self.send_order(
            self._market_order(
                symbol, qty, side, MARGIN_CLOSE, exchange,
                margin_trade_type=margin_trade_type,
                deliv_type=DELIV_CASH,
                close_positions=close_positions,
            )
        )

    def can_margin_sell(self, symbol: str, exchange: int = 1):
        """
        信用売建が可能か。
        戻り値: (可否, 詳細dict)
        検証環境は全フィールドが空で返るため、判定は本番環境でのみ有効。
        """
        info = self.symbol(symbol, exchange)
        flags = {f: info.get(f) for f in MARGIN_SELL_FIELDS}
        return any(bool(v) for v in flags.values()), flags
