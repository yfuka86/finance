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

    # ── auth ────────────────────────────────────────────────────────
    def authenticate(self) -> str:
        r = requests.post(f"{self.base}/token", json={"APIPassword": self._api_password},
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

    def _request(self, method: str, path: str, *, params=None, json=None):
        url = f"{self.base}{path}"
        for attempt in range(2):  # one retry after re-auth on 401
            r = requests.request(method, url, headers=self._headers(),
                                 params=params, json=json, timeout=self.timeout)
            if r.status_code == 401 and attempt == 0:
                self._token = None
                continue
            if not r.ok:
                raise KabuAPIError(f"{method} {path} -> {r.status_code}: {r.text}")
            return r.json() if r.text else {}
        raise KabuAPIError(f"{method} {path} failed after re-auth")

    # ── market data ─────────────────────────────────────────────────
    def board(self, symbol: str, exchange: int = 1) -> dict:
        """Quote/board incl. pre-open indicative (CurrentPrice / CalcPrice / 特別気配)."""
        return self._request("GET", f"/board/{to_kabu_symbol(symbol)}@{exchange}")

    def symbol_info(self, symbol: str, exchange: int = 1) -> dict:
        return self._request("GET", f"/symbol/{to_kabu_symbol(symbol)}@{exchange}")

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

    def send_margin_open(self, symbol: str, side: str, qty: int, *,
                         front_order_type: int = FRONT_OPEN, margin_type: int = 3,
                         exchange: int = 1, account_type: int = 4) -> dict:
        """信用新規建て (default 一日信用=3, 寄成=13). qty in shares (multiple of 100)."""
        return self._send({
            "Password": self._order_password, "Symbol": to_kabu_symbol(symbol), "Exchange": exchange,
            "SecurityType": 1, "Side": str(side), "CashMargin": 2,
            "MarginTradeType": margin_type, "DelivType": 0, "AccountType": account_type,
            "Qty": int(qty), "FrontOrderType": int(front_order_type), "Price": 0, "ExpireDay": 0,
        })

    def send_margin_close(self, symbol: str, side: str, qty: int, hold_id: str, *,
                          front_order_type: int = FRONT_CLOSE, margin_type: int = 3,
                          exchange: int = 1, account_type: int = 4) -> dict:
        """信用返済 (default 引成=16). ``side`` is the CLOSING side (opposite of the position).
        ``hold_id`` is the position's ExecutionID from ``positions()``."""
        return self._send({
            "Password": self._order_password, "Symbol": to_kabu_symbol(symbol), "Exchange": exchange,
            "SecurityType": 1, "Side": str(side), "CashMargin": 3,
            "MarginTradeType": margin_type, "DelivType": 2, "FundType": "11",
            "AccountType": account_type, "Qty": int(qty), "FrontOrderType": int(front_order_type),
            "Price": 0, "ExpireDay": 0, "ClosePositions": [{"HoldID": hold_id, "Qty": int(qty)}],
        })

    def cancel_order(self, order_id: str) -> dict:
        return self._request("PUT", "/cancelorder",
                             json={"OrderId": order_id, "Password": self._order_password})
