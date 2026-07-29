"""In-memory MockKabuClient — runs the full plan→entry→exit→report flow with NO
kabuステーション / Windows dependency (for off-Windows development, preflight, tests).

It mimics the kabuステーションAPI shapes the executor relies on:
  - board(): pre-open indicative from a supplied price map
  - send_margin_open(): opens an in-memory 建玉 (ExecutionID, Side, LeavesQty, HoldQty=0)
  - send_margin_close(): locks HoldQty (pending 返済) then reduces LeavesQty
  - positions()/orders()/wallet_margin(): reflect that state
Returned symbols use the kabu 4-char form, exactly like the real API.
"""
from __future__ import annotations

from .kabu_client import SIDE_BUY, SIDE_SELL, to_kabu_symbol


class MockKabuClient:
    def __init__(self, prices: dict, capital_yen: float = 20_000_000, prev_close: dict | None = None,
                 short_banned: set | None = None):
        # prices/prev_close keyed by the symbol the caller passes (J-Quants 5-digit ok).
        self._prices = {str(k): float(v) for k, v in prices.items()}
        self._prev = {str(k): float(v) for k, v in (prev_close or {}).items()}
        self._capital = capital_yen
        self._positions: list[dict] = []
        self._orders: list[dict] = []
        self._seq = 0
        self._short_banned = {to_kabu_symbol(s) for s in (short_banned or set())}

    def symbol_info(self, symbol: str, exchange: int = 1) -> dict:
        k = to_kabu_symbol(symbol)
        ok = k not in self._short_banned
        return {"Symbol": k, "MarginSell": ok, "KCMarginSell": ok, "MarginBuy": True}

    def authenticate(self) -> str:
        return "mock-token"

    def _price(self, symbol: str) -> float:
        s = str(symbol)
        return self._prices.get(s) or self._prices.get(to_kabu_symbol(s)) or 0.0

    def board(self, symbol: str, exchange: int = 1) -> dict:
        px = self._price(symbol)
        pc = self._prev.get(str(symbol)) or self._prev.get(to_kabu_symbol(str(symbol))) or px
        return {"Symbol": to_kabu_symbol(symbol), "CurrentPrice": px, "CalcPrice": px, "PrevClose": pc}

    def positions(self, product: int = 2) -> list:
        return [dict(p) for p in self._positions if p["LeavesQty"] - p["HoldQty"] > 0]

    def orders(self, product: int = 0) -> list:
        return [dict(o) for o in self._orders]

    def wallet_margin(self) -> dict:
        return {"MarginAccountWallet": self._capital, "DepositkeepRate": None}

    def send_margin_open(self, symbol: str, side: str, qty: int, **kw) -> dict:
        self._seq += 1
        oid = f"MOCK{self._seq:05d}"
        self._positions.append({
            "Symbol": to_kabu_symbol(symbol), "Side": str(side), "LeavesQty": int(qty),
            "HoldQty": 0, "ExecutionID": oid, "ExecutionDay": kw.get("execution_day", 0),
            "Price": self._price(symbol),   # 建値（実APIと同様・シャドー監視が参照）
        })
        self._orders.append({"ID": oid, "Symbol": to_kabu_symbol(symbol), "Side": str(side),
                             "OrderQty": int(qty), "State": 3, "CashMargin": 2})
        return {"Result": 0, "OrderId": oid}

    def send_margin_close(self, symbol: str, side: str, qty: int, hold_id: str, **kw) -> dict:
        self._seq += 1
        oid = f"MOCK{self._seq:05d}"
        for p in self._positions:
            if p["ExecutionID"] == hold_id:
                # lock as pending, then simulate the closing-auction execution
                p["HoldQty"] = min(p["LeavesQty"], p["HoldQty"] + int(qty))
                p["LeavesQty"] = max(0, p["LeavesQty"] - int(qty))
                p["HoldQty"] = max(0, p["HoldQty"] - int(qty))
                break
        self._orders.append({"ID": oid, "Symbol": to_kabu_symbol(symbol), "Side": str(side),
                             "OrderQty": int(qty), "State": 3, "CashMargin": 3})
        return {"Result": 0, "OrderId": oid}

    def cancel_order(self, order_id: str) -> dict:
        return {"Result": 0, "OrderId": order_id}
