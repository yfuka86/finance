"""Bybit API client wrapper using pybit."""
import logging
import time
from typing import Optional

from pybit.unified_trading import HTTP, WebSocket

from .config import BybitConfig

logger = logging.getLogger(__name__)


class BybitClient:
    """Thin wrapper around pybit with reconnection and error handling."""

    def __init__(self, config: BybitConfig):
        self.config = config
        self._http: Optional[HTTP] = None
        self._ws_public: Optional[WebSocket] = None
        self._ws_private: Optional[WebSocket] = None
        self._ws_callbacks: dict = {}

    def connect(self):
        """Initialize HTTP and WebSocket connections."""
        testnet = self.config.network == "testnet"
        demo = self.config.network == "demo"

        self._http = HTTP(
            testnet=testnet,
            demo=demo,
            api_key=self.config.api_key,
            api_secret=self.config.api_secret,
        )
        logger.info(f"HTTP client connected to {self.config.network}")

    def connect_ws(self, on_orderbook=None, on_trade=None, on_kline=None,
                   on_order=None, on_position=None, on_wallet=None):
        """Start WebSocket streams."""
        testnet = self.config.network == "testnet"

        # Public WebSocket (market data)
        self._ws_public = WebSocket(
            testnet=testnet,
            channel_type="linear",
        )
        symbol = self.config.symbol

        if on_orderbook:
            self._ws_public.orderbook_stream(depth=50, symbol=symbol, callback=on_orderbook)
            logger.info(f"Subscribed to orderbook: {symbol}")

        if on_trade:
            self._ws_public.trade_stream(symbol=symbol, callback=on_trade)
            logger.info(f"Subscribed to trades: {symbol}")

        if on_kline:
            self._ws_public.kline_stream(interval=1, symbol=symbol, callback=on_kline)
            logger.info(f"Subscribed to 1m klines: {symbol}")

        # Private WebSocket (account updates)
        if any([on_order, on_position, on_wallet]):
            self._ws_private = WebSocket(
                testnet=testnet,
                channel_type="private",
                api_key=self.config.api_key,
                api_secret=self.config.api_secret,
            )
            if on_order:
                self._ws_private.order_stream(callback=on_order)
            if on_position:
                self._ws_private.position_stream(callback=on_position)
            if on_wallet:
                self._ws_private.wallet_stream(callback=on_wallet)
            logger.info("Private WebSocket streams connected")

    # ── Market Data ──────────────────────────────────────────────

    def get_ticker(self, symbol: Optional[str] = None) -> dict:
        symbol = symbol or self.config.symbol
        resp = self._http.get_tickers(category=self.config.category, symbol=symbol)
        return resp["result"]["list"][0]

    def get_orderbook(self, symbol: Optional[str] = None, limit: int = 25) -> dict:
        symbol = symbol or self.config.symbol
        resp = self._http.get_orderbook(
            category=self.config.category, symbol=symbol, limit=limit
        )
        return resp["result"]

    def get_klines(self, interval: str = "1", limit: int = 200,
                   symbol: Optional[str] = None) -> list:
        symbol = symbol or self.config.symbol
        resp = self._http.get_kline(
            category=self.config.category, symbol=symbol,
            interval=interval, limit=limit,
        )
        return resp["result"]["list"]

    def get_instrument_info(self, symbol: Optional[str] = None) -> dict:
        symbol = symbol or self.config.symbol
        resp = self._http.get_instruments_info(
            category=self.config.category, symbol=symbol
        )
        return resp["result"]["list"][0]

    # ── Account ──────────────────────────────────────────────────

    def get_balance(self, coin: str = "USDT") -> dict:
        resp = self._http.get_wallet_balance(accountType="UNIFIED")
        for acct in resp["result"]["list"]:
            for c in acct.get("coin", []):
                if c["coin"] == coin:
                    return {
                        "equity": float(c.get("equity") or 0),
                        "available": float(c.get("availableToWithdraw") or 0),
                        "unrealized_pnl": float(c.get("unrealisedPnl") or 0),
                    }
        return {"equity": 0, "available": 0, "unrealized_pnl": 0}

    def get_all_balances(self) -> dict:
        """Get balances for all coins."""
        resp = self._http.get_wallet_balance(accountType="UNIFIED")
        balances = {}
        total_equity = 0.0
        for acct in resp["result"]["list"]:
            total_equity = float(acct.get("totalEquity") or 0)
            for c in acct.get("coin", []):
                eq = float(c.get("equity") or 0)
                if eq > 0:
                    balances[c["coin"]] = {
                        "equity": eq,
                        "usd_value": float(c.get("usdValue") or 0),
                        "available": float(c.get("availableToWithdraw") or 0),
                        "unrealized_pnl": float(c.get("unrealisedPnl") or 0),
                    }
        return {"total_equity_usd": total_equity, "coins": balances}

    def get_positions(self, symbol: Optional[str] = None) -> list:
        symbol = symbol or self.config.symbol
        resp = self._http.get_positions(
            category=self.config.category, symbol=symbol
        )
        return resp["result"]["list"]

    # ── Orders ───────────────────────────────────────────────────

    def place_order(self, side: str, qty: str, order_type: str = "Limit",
                    price: Optional[str] = None, reduce_only: bool = False,
                    symbol: Optional[str] = None, **kwargs) -> dict:
        symbol = symbol or self.config.symbol
        params = dict(
            category=self.config.category,
            symbol=symbol,
            side=side,
            orderType=order_type,
            qty=qty,
            reduceOnly=reduce_only,
        )
        if price and order_type == "Limit":
            params["price"] = price
        if kwargs.get("time_in_force"):
            params["timeInForce"] = kwargs["time_in_force"]
        if kwargs.get("stop_loss"):
            params["stopLoss"] = kwargs["stop_loss"]
        if kwargs.get("take_profit"):
            params["takeProfit"] = kwargs["take_profit"]

        resp = self._http.place_order(**params)
        logger.info(f"Order placed: {side} {qty} {symbol} @ {price or 'Market'} -> {resp['result']['orderId']}")
        return resp["result"]

    def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> dict:
        symbol = symbol or self.config.symbol
        resp = self._http.cancel_order(
            category=self.config.category, symbol=symbol, orderId=order_id
        )
        return resp["result"]

    def cancel_all_orders(self, symbol: Optional[str] = None) -> dict:
        symbol = symbol or self.config.symbol
        resp = self._http.cancel_all_orders(
            category=self.config.category, symbol=symbol
        )
        logger.info(f"All orders cancelled for {symbol}")
        return resp["result"]

    def get_open_orders(self, symbol: Optional[str] = None) -> list:
        symbol = symbol or self.config.symbol
        resp = self._http.get_open_orders(
            category=self.config.category, symbol=symbol
        )
        return resp["result"]["list"]

    # ── Leverage ─────────────────────────────────────────────────

    def set_leverage(self, leverage: int, symbol: Optional[str] = None):
        symbol = symbol or self.config.symbol
        try:
            self._http.set_leverage(
                category=self.config.category, symbol=symbol,
                buyLeverage=str(leverage), sellLeverage=str(leverage),
            )
            logger.info(f"Leverage set to {leverage}x for {symbol}")
        except Exception as e:
            if "leverage not modified" in str(e).lower():
                pass  # Already set
            else:
                raise

    # ── Cleanup ──────────────────────────────────────────────────

    def disconnect(self):
        if self._ws_public:
            self._ws_public.exit()
        if self._ws_private:
            self._ws_private.exit()
        logger.info("WebSocket connections closed")
