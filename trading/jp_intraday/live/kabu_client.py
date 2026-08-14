"""auカブコム証券 kabuステーションAPI client (Windows: kabuステーション必須).

The kabuステーション desktop app must be running and logged in on the SAME
Windows machine; it exposes a LOCAL REST/WebSocket API on:
  - 本番 (production):  http://localhost:18080/kabusapi
  - 検証 (test/paper):  http://localhost:18081/kabusapi   ← default here

Auth: POST /token {"APIPassword": ...} -> {"Token": ...}; the token is sent as the
``X-API-KEY`` header on every subsequent call and is invalidated when kabuステーション
restarts, so we transparently re-authenticate on 401.

Order codes (verified against the API reference):
  Side          "1"=売, "2"=買
  CashMargin    1=現物, 2=信用新規, 3=信用返済
  MarginTradeType 1=制度, 2=一般長期, 3=一般デイトレ(一日信用)
  FrontOrderType 10=成行, 13=寄成(前場寄付), 16=引成(大引け), 20=指値
  Exchange 1=東証   SecurityType 1=株式   AccountType 2=一般,4=特定

This module places orders ONLY through explicit methods; the executor keeps a
dry-run guard so nothing is sent unless live trading is deliberately enabled.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol, runtime_checkable

import requests

SIDE_BUY, SIDE_SELL = "2", "1"
FRONT_MARKET, FRONT_OPEN, FRONT_CLOSE, FRONT_LIMIT = 10, 13, 16, 20


class KabuAPIError(RuntimeError):
    pass


def to_kabu_symbol(code: str | int) -> str:
    """J-Quants 5-digit code -> kabu 4-char ticker (drop the trailing padding digit).

    Our panels use J-Quants codes ("13010", "130A0"); kabuステーションAPI expects the
    4-char ticker ("1301", "130A"). Anything not 5 chars is passed through unchanged.
    """
    s = str(code)
    return s[:-1] if len(s) == 5 else s


@runtime_checkable
class KabuClientProtocol(Protocol):
    """Surface the executor depends on — real KabuClient and MockKabuClient share it."""

    def authenticate(self) -> str: ...
    def board(self, symbol: str, exchange: int = 1) -> dict: ...
    def positions(self, product: int = 2) -> list: ...
    def orders(self, product: int = 0) -> list: ...
    def wallet_margin(self) -> dict: ...
    def send_margin_open(self, symbol: str, side: str, qty: int, **kw) -> dict: ...
    def send_margin_close(self, symbol: str, side: str, qty: int, hold_id: str, **kw) -> dict: ...


class KabuClient:
    # kabuステーションAPIの参照系レート制限は10req/s。それを僅かに下回るペース。
    # Session再利用が無いと1リクエスト~0.9秒(毎回TCP+ハンドシェイク)に落ちる実測。
    MIN_INTERVAL = 0.105

    def __init__(self, api_password: str, order_password: str, env: str = "test",
                 host: str = "localhost", timeout: float = 10.0):
        if env not in ("test", "prod"):
            raise ValueError("env must be 'test' or 'prod'")
        port = 18081 if env == "test" else 18080
        self.env = env
        self.base = f"http://{host}:{port}/kabusapi"
        self._api_password = api_password
        self._order_password = order_password
        self.timeout = timeout
        self._token: str | None = None
        self._session = requests.Session()
        self._last_call = 0.0
        self._board_calls = 0
        # boards() で並列に叩くため、スロットルとセッションをスレッド安全にする。
        # Session はスレッドごとに分ける（1本を共有するとコネクションプールで詰まる）。
        self._throttle_lock = threading.Lock()
        self._auth_lock = threading.Lock()
        self._local = threading.local()
        self._session_factory = requests.Session   # テストで差し替え可能にしておく

    @property
    def _sess(self) -> requests.Session:
        s = getattr(self._local, "session", None)
        if s is None:
            s = self._session if threading.current_thread() is threading.main_thread() \
                else self._session_factory()
            self._local.session = s
        return s

    # ── auth ────────────────────────────────────────────────────────
    def authenticate(self) -> str:
        with self._auth_lock:      # 並列取得中に複数スレッドが同時発行しないように
            r = self._sess.post(f"{self.base}/token", json={"APIPassword": self._api_password},
                                timeout=self.timeout)
            r.raise_for_status()
            body = r.json()
            token = body.get("Token")
            if not token:
                raise KabuAPIError(f"authentication failed: {body}")
            self._token = token
            return token

    def _headers(self) -> dict:
        if not self._token:
            self.authenticate()
        return {"Content-Type": "application/json", "X-API-KEY": self._token}

    def _throttle(self) -> None:
        """送信開始のタイミングだけを直列化する（レート制限はプロセス全体で10req/s）。"""
        import time
        with self._throttle_lock:
            wait = self.MIN_INTERVAL - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()

    def _request(self, method: str, path: str, *, params=None, json=None):
        url = f"{self.base}{path}"
        for attempt in range(3):  # retry after re-auth on 401 / brief backoff on 429
            self._throttle()
            r = self._sess.request(method, url, headers=self._headers(),
                                   params=params, json=json, timeout=self.timeout)
            if r.status_code == 401 and attempt == 0:
                self._token = None
                continue
            if r.status_code == 429 and attempt < 2:
                import time
                time.sleep(0.5 * (attempt + 1))
                continue
            if not r.ok:
                raise KabuAPIError(f"{method} {path} -> {r.status_code}: {r.text}")
            return r.json() if r.text else {}
        raise KabuAPIError(f"{method} {path} failed after retries")

    # ── market data ─────────────────────────────────────────────────
    def unregister_all(self) -> dict:
        """登録銘柄リストを全消去 (PUSH配信登録も消えるので注意)."""
        return self._request("PUT", "/unregister/all")

    def board(self, symbol: str, exchange: int = 1) -> dict:
        """Quote/board incl. pre-open indicative (CurrentPrice / CalcPrice / 特別気配).

        GET /board は照会銘柄を自動で銘柄登録し、登録数上限50を超えると
        4002006 で失敗する(実測: 51件目から全滅)。全ユニバースの板を舐める
        plan 用に、45件ごとに登録リストを全消去して回避する。
        """
        if self._board_calls and self._board_calls % 45 == 0:
            try:
                self.unregister_all()
            except KabuAPIError:
                pass  # 消去失敗は次の board エラーで顕在化する
        self._board_calls += 1
        return self._request("GET", f"/board/{to_kabu_symbol(symbol)}@{exchange}")

    # 板の一括取得（全ユニバースを寄付き前に舐めるための唯一の実用経路）
    #
    # 実測 2026-07-30（本番・863銘柄・成功率100%）:
    #   1件ずつ直列     0.72件/秒 → 20分  ← 寄付きに間に合わない（当日の実障害）
    #   並列8+一括登録  1.45件/秒 → 10分
    # 未登録銘柄の GET /board はステーションが取引所へ取りに行くため中央値~900ms
    # (1割はちょうど5秒)。登録済みなら3ms。並列化してもステーション側で直列化される
    # ため2倍が上限だが、この2倍が「寄付きに間に合うか否か」を分ける。
    BOARD_CHUNK = 45          # 登録上限50に対する安全域
    BOARD_WORKERS = 8         # これ以上増やしてもステーション側で詰まるだけ

    def boards(self, symbols, workers: int | None = None, on_progress=None) -> dict:
        """Fetch boards for many symbols. Returns {symbol: board}（失敗銘柄は欠落）.

        45件ごとに登録リストを作り直しつつ、チャンク内は並列に取得する。
        キーは入力の symbol をそのまま使う（J-Quants 5桁のまま返る）。
        """
        syms = list(symbols)
        out: dict = {}
        self._headers()                      # 並列に入る前に認証を済ませておく
        workers = workers or self.BOARD_WORKERS
        for i in range(0, len(syms), self.BOARD_CHUNK):
            part = syms[i:i + self.BOARD_CHUNK]
            # 登録の作り直しは高速化のための補助でしかない。ここでの失敗
            # (タイムアウト含む) で全体を落とさない — 落とすと寄付きに間に合わない。
            # ただし登録を消せていないと上限50に達して取得自体が失敗するため、
            # 消去だけは一度リトライする。
            for attempt in range(2):
                try:
                    self.unregister_all()
                    self._board_calls = 0    # 単発 board() 側のカウンタと整合させる
                    break
                except Exception:  # noqa: BLE001
                    if attempt:
                        pass
            try:
                self._request("PUT", "/register", json={
                    "Symbols": [{"Symbol": to_kabu_symbol(s), "Exchange": 1} for s in part]})
            except Exception:  # noqa: BLE001
                pass                         # 登録失敗でも取得自体は可能（遅くなるだけ）
            with ThreadPoolExecutor(max_workers=workers) as ex:
                for sym, board in zip(part, ex.map(self._board_or_none, part)):
                    if board is not None:
                        out[sym] = board
            if on_progress:
                on_progress(min(i + self.BOARD_CHUNK, len(syms)), len(syms))
        return out

    def _board_or_none(self, symbol: str, exchange: int = 1):
        """board() の登録管理を通さない素の取得（登録は boards() がチャンク単位で管理）。"""
        try:
            return self._request("GET", f"/board/{to_kabu_symbol(symbol)}@{exchange}")
        except Exception:  # noqa: BLE001  1銘柄の失敗で全体を落とさない
            return None

    def symbol_info(self, symbol: str, exchange: int = 1) -> dict:
        return self._request("GET", f"/symbol/{to_kabu_symbol(symbol)}@{exchange}")

    def regulations(self, symbol: str, exchange: int = 1) -> dict:
        """規制情報（新規売停止・注文制限等）。売建候補の発注前チェック用."""
        return self._request("GET", f"/regulations/{to_kabu_symbol(symbol)}@{exchange}")

    # ── account state ───────────────────────────────────────────────
    def positions(self, product: int = 2) -> list:
        """product: 0=all,1=現物,2=信用,3=先物,4=OP."""
        return self._request("GET", "/positions", params={"product": product})

    def orders(self, product: int = 0) -> list:
        return self._request("GET", "/orders", params={"product": product})

    def wallet_margin(self) -> dict:
        return self._request("GET", "/wallet/margin")

    def wallet_cash(self) -> dict:
        return self._request("GET", "/wallet/cash")

    # ── orders ──────────────────────────────────────────────────────
    def _send(self, body: dict) -> dict:
        """POST /sendorder and treat a non-zero Result (business rejection in a 200) as an error."""
        resp = self._request("POST", "/sendorder", json=body)
        if int(resp.get("Result", -1)) != 0:
            raise KabuAPIError(f"order rejected: {resp} (sent {body.get('Symbol')})")
        return resp

    # 信用注文の市場コード。本番実測 (2026-07-29):
    #   Exchange=1(東証) の信用新規は 100368「信用新規注文は抑止されております」で全拒否。
    #   Exchange=9(SOR) なら受理される (指値・寄指の板寄せ系とも確認済み)。
    # SOR対象外銘柄等に備え、9で拒否されたら1で一度だけ再試行する。
    EXCHANGE_SOR = 9
    EXCHANGE_TSE = 1

    def _send_with_exchange_fallback(self, body: dict) -> dict:
        body = dict(body, Exchange=self.EXCHANGE_SOR)
        try:
            return self._send(body)
        except KabuAPIError as first:
            try:
                return self._send(dict(body, Exchange=self.EXCHANGE_TSE))
            except KabuAPIError:
                raise first  # 元エラー (SOR側) のほうが原因を語ることが多い

    def send_margin_open(self, symbol: str, side: str, qty: int, *,
                         front_order_type: int = FRONT_OPEN, margin_type: int = 3,
                         exchange: int = None, account_type: int = 4) -> dict:
        """信用新規建て (default 一日信用=3, 寄成=13, SOR). qty in shares (multiple of 100)."""
        body = {
            "Password": self._order_password, "Symbol": to_kabu_symbol(symbol),
            "SecurityType": 1, "Side": str(side), "CashMargin": 2,
            "MarginTradeType": margin_type, "DelivType": 0, "AccountType": account_type,
            "Qty": int(qty), "FrontOrderType": int(front_order_type), "Price": 0, "ExpireDay": 0,
        }
        if exchange is not None:  # 明示指定時はフォールバックせずそのまま
            return self._send(dict(body, Exchange=exchange))
        return self._send_with_exchange_fallback(body)

    def send_margin_close(self, symbol: str, side: str, qty: int, hold_id: str, *,
                          front_order_type: int = FRONT_CLOSE, margin_type: int = 3,
                          exchange: int = None, account_type: int = 4) -> dict:
        """信用返済 (default 引成=16, SOR→東証フォールバック). ``side`` is the CLOSING side
        (opposite of the position). ``hold_id`` is the position's ExecutionID from ``positions()``."""
        body = {
            "Password": self._order_password, "Symbol": to_kabu_symbol(symbol),
            "SecurityType": 1, "Side": str(side), "CashMargin": 3,
            "MarginTradeType": margin_type, "DelivType": 2, "FundType": "11",
            "AccountType": account_type, "Qty": int(qty), "FrontOrderType": int(front_order_type),
            "Price": 0, "ExpireDay": 0, "ClosePositions": [{"HoldID": hold_id, "Qty": int(qty)}],
        }
        if exchange is not None:
            return self._send(dict(body, Exchange=exchange))
        return self._send_with_exchange_fallback(body)

    def cancel_order(self, order_id: str) -> dict:
        return self._request("PUT", "/cancelorder",
                             json={"OrderId": order_id, "Password": self._order_password})


class HybridKabuClient:
    """板・銘柄情報は本番(18080・参照のみ)、口座・発注は検証(18081・ペーパー)。

    検証環境は板情報を配信しない(全フィールドnull)ため、test環境での通し
    リハーサルには本番の実データが必要になる。本クラスは読み取り系だけを
    本番クライアントへ委譲し、発注系・口座系は検証クライアントに閉じる。
    本番側には /board /symbol のGETしか発行しない。
    """

    def __init__(self, data_client: "KabuClient", trade_client: "KabuClient"):
        assert trade_client.env != "prod", "HybridKabuClient の発注側は test 限定"
        self._data = data_client
        self._trade = trade_client

    def authenticate(self) -> str:
        self._data.authenticate()
        return self._trade.authenticate()

    # 参照系 → 本番 (読み取りのみ)
    def board(self, symbol: str, exchange: int = 1) -> dict:
        return self._data.board(symbol, exchange)

    def boards(self, symbols, workers: int | None = None, on_progress=None) -> dict:
        return self._data.boards(symbols, workers=workers, on_progress=on_progress)

    def symbol_info(self, symbol: str, exchange: int = 1) -> dict:
        return self._data.symbol_info(symbol, exchange)

    def regulations(self, symbol: str, exchange: int = 1) -> dict:
        return self._data.regulations(symbol, exchange) if hasattr(self._data, "regulations") else {}

    # 口座・発注系 → 検証
    def positions(self, product: int = 2) -> list:
        return self._trade.positions(product=product)

    def orders(self, product: int = 0) -> list:
        return self._trade.orders(product=product)

    def wallet_margin(self) -> dict:
        return self._trade.wallet_margin()

    def wallet_cash(self) -> dict:
        return self._trade.wallet_cash()

    def send_margin_open(self, symbol: str, side: str, qty: int, **kw) -> dict:
        return self._trade.send_margin_open(symbol, side, qty, **kw)

    def send_margin_close(self, symbol: str, side: str, qty: int, hold_id: str, **kw) -> dict:
        return self._trade.send_margin_close(symbol, side, qty, hold_id, **kw)

    def cancel_order(self, order_id: str) -> dict:
        return self._trade.cancel_order(order_id)
