"""Main trading engine - orchestrates strategy, risk, and execution."""
import logging
import time
import threading
from typing import Optional

from .client import BybitClient
from .config import BybitConfig
from .risk import RiskManager
from .strategy import STRATEGIES
from .strategy.base import BaseStrategy, Signal
from .strategy.grid import GridStrategy

logger = logging.getLogger(__name__)


class TradingEngine:
    """
    Event-driven trading engine.

    Flow:
    1. WebSocket receives market data
    2. Strategy generates signals
    3. Risk manager validates signals
    4. Engine executes orders via client
    """

    def __init__(self, config: BybitConfig):
        self.config = config
        self.client = BybitClient(config)
        self.risk = RiskManager(config)

        strategy_cls = STRATEGIES.get(config.strategy)
        if not strategy_cls:
            raise ValueError(f"Unknown strategy: {config.strategy}. Available: {list(STRATEGIES.keys())}")
        self.strategy: BaseStrategy = strategy_cls(config)

        self._running = False
        self._instrument_info: Optional[dict] = None
        self._min_qty: float = 0.001
        self._qty_step: float = 0.001
        self._tick_size: float = 0.01

    def start(self):
        """Initialize connections and start trading loop."""
        logger.info(f"Starting engine: {self.config.strategy} on {self.config.symbol} ({self.config.network})")

        # Connect HTTP
        self.client.connect()

        # Get instrument info for quantity/price rounding
        self._load_instrument_info()

        # Set leverage
        self.client.set_leverage(self.config.max_leverage)

        # Initialize risk manager with current equity
        balance = self.client.get_balance()
        self.risk.initialize(balance["equity"])
        logger.info(f"Account equity: {balance['equity']:.2f} USDT")

        # Sync existing positions
        self._sync_positions()

        # Warm up strategy with historical klines
        self._warmup()

        # Connect WebSocket with callbacks
        self.client.connect_ws(
            on_orderbook=self._on_orderbook,
            on_trade=self._on_trade,
            on_kline=self._on_kline,
            on_order=self._on_order,
            on_position=self._on_position,
            on_wallet=self._on_wallet,
        )

        # If grid strategy, place initial orders
        if isinstance(self.strategy, GridStrategy) and self.strategy.grid_initialized:
            ticker = self.client.get_ticker()
            current_price = float(ticker["lastPrice"])
            initial_orders = self.strategy.get_initial_orders(current_price)
            for sig in initial_orders:
                self._execute_signal(sig)

        self._running = True
        logger.info("Engine started. Listening for market data...")

        # Heartbeat loop
        self._heartbeat_loop()

    def stop(self):
        """Graceful shutdown."""
        logger.info("Stopping engine...")
        self._running = False

        # Cancel all open orders
        try:
            self.client.cancel_all_orders()
        except Exception as e:
            logger.error(f"Error cancelling orders: {e}")

        self.client.disconnect()
        logger.info("Engine stopped")

    def _load_instrument_info(self):
        info = self.client.get_instrument_info()
        lot_filter = info.get("lotSizeFilter", {})
        price_filter = info.get("priceFilter", {})
        self._min_qty = float(lot_filter.get("minOrderQty", "0.001"))
        self._qty_step = float(lot_filter.get("qtyStep", "0.001"))
        self._tick_size = float(price_filter.get("tickSize", "0.01"))
        logger.info(
            f"Instrument: min_qty={self._min_qty}, qty_step={self._qty_step}, "
            f"tick_size={self._tick_size}"
        )

    def _sync_positions(self):
        """Sync strategy state with existing positions."""
        positions = self.client.get_positions()
        for pos in positions:
            if float(pos.get("size", 0)) > 0:
                self.strategy.on_position_update(pos)
                logger.info(f"Existing position: {pos['side']} {pos['size']} @ {pos['avgPrice']}")

    def _warmup(self):
        """Feed historical klines to strategy for indicator initialization."""
        klines = self.client.get_klines(interval="1", limit=200)
        # Klines are returned newest first, reverse for chronological order
        for kline in reversed(klines):
            self.strategy.on_kline(kline)
        logger.info(f"Strategy warmed up with {len(klines)} historical klines")

    # ── WebSocket Callbacks ──────────────────────────────────────

    def _on_kline(self, message: dict):
        try:
            data_list = message.get("data", [])
            for kline in data_list:
                if not kline.get("confirm", False):
                    continue  # Only process closed candles
                signal = self.strategy.on_kline(kline)
                if signal:
                    self._process_signal(signal)
        except Exception as e:
            logger.error(f"Error in kline handler: {e}", exc_info=True)

    def _on_orderbook(self, message: dict):
        try:
            signals = self.strategy.on_orderbook(message)
            if signals:
                # Cancel existing orders first for market maker
                self.client.cancel_all_orders()
                for signal in signals:
                    self._process_signal(signal)
        except Exception as e:
            logger.error(f"Error in orderbook handler: {e}", exc_info=True)

    def _on_trade(self, message: dict):
        try:
            data_list = message.get("data", [])
            for trade in data_list:
                signal = self.strategy.on_trade(trade)
                if signal:
                    self._process_signal(signal)
        except Exception as e:
            logger.error(f"Error in trade handler: {e}", exc_info=True)

    def _on_order(self, message: dict):
        try:
            data_list = message.get("data", [])
            for order in data_list:
                self.strategy.on_order_update(order)

                # Handle grid fill response
                if isinstance(self.strategy, GridStrategy):
                    if order.get("orderStatus") == "Filled":
                        response = self.strategy.get_fill_response(order)
                        if response:
                            self._process_signal(response)
        except Exception as e:
            logger.error(f"Error in order handler: {e}", exc_info=True)

    def _on_position(self, message: dict):
        try:
            data_list = message.get("data", [])
            for pos in data_list:
                self.strategy.on_position_update(pos)
        except Exception as e:
            logger.error(f"Error in position handler: {e}", exc_info=True)

    def _on_wallet(self, message: dict):
        try:
            data_list = message.get("data", [])
            for wallet in data_list:
                for coin in wallet.get("coin", []):
                    if coin["coin"] == "USDT":
                        equity = float(coin.get("equity", 0))
                        upnl = float(coin.get("unrealisedPnl", 0))
                        self.risk.check(equity, upnl, self.risk.state.position_usd)
        except Exception as e:
            logger.error(f"Error in wallet handler: {e}", exc_info=True)

    # ── Signal Processing ────────────────────────────────────────

    def _process_signal(self, signal: Signal):
        """Validate signal through risk manager and execute."""
        if not self.risk.state.trading_enabled:
            logger.warning(f"Signal rejected (trading disabled): {signal.reason}")
            return

        if not self.risk.can_open_position(signal.side, signal.qty_usd):
            logger.warning(f"Signal rejected (risk limit): {signal.reason}")
            return

        self._execute_signal(signal)

    def _execute_signal(self, signal: Signal):
        """Convert signal to order and send to exchange."""
        try:
            # Convert USD qty to coin qty
            ticker = self.client.get_ticker()
            price = float(ticker["lastPrice"])
            qty = signal.qty_usd / price
            qty = self._round_qty(qty)

            if qty < self._min_qty:
                logger.warning(f"Order too small: {qty} < {self._min_qty}")
                return

            order_type = "Limit" if signal.price else "Market"
            order_price = None
            if signal.price:
                order_price = str(self._round_price(signal.price))

            kwargs = {}
            if signal.stop_loss:
                kwargs["stop_loss"] = str(self._round_price(signal.stop_loss))
            if signal.take_profit:
                kwargs["take_profit"] = str(self._round_price(signal.take_profit))

            result = self.client.place_order(
                side=signal.side,
                qty=str(qty),
                order_type=order_type,
                price=order_price,
                **kwargs,
            )

            self.risk.state.num_trades_today += 1
            logger.info(f"Order executed: {signal.reason} -> {result.get('orderId', 'unknown')}")

        except Exception as e:
            logger.error(f"Order execution failed: {e}", exc_info=True)

    def _round_qty(self, qty: float) -> float:
        """Round quantity to valid step size."""
        if self._qty_step > 0:
            qty = round(qty / self._qty_step) * self._qty_step
        # Round to avoid floating point issues
        decimals = len(str(self._qty_step).rstrip('0').split('.')[-1]) if '.' in str(self._qty_step) else 0
        return round(qty, decimals)

    def _round_price(self, price: float) -> float:
        """Round price to valid tick size."""
        if self._tick_size > 0:
            price = round(price / self._tick_size) * self._tick_size
        decimals = len(str(self._tick_size).rstrip('0').split('.')[-1]) if '.' in str(self._tick_size) else 0
        return round(price, decimals)

    # ── Heartbeat ────────────────────────────────────────────────

    def _heartbeat_loop(self):
        """Periodic health check and status logging."""
        while self._running:
            try:
                time.sleep(self.config.heartbeat_interval)

                # Refresh risk state
                balance = self.client.get_balance()
                positions = self.client.get_positions()

                position_usd = 0
                for pos in positions:
                    size = float(pos.get("size", 0))
                    avg_price = float(pos.get("avgPrice", 0))
                    position_usd += size * avg_price

                trading_ok = self.risk.check(
                    balance["equity"],
                    balance["unrealized_pnl"],
                    position_usd,
                )

                status = self.risk.status()
                logger.info(
                    f"[Heartbeat] equity={status['equity']:.2f} "
                    f"pnl={status['daily_pnl']:.2f} "
                    f"dd={status['drawdown_pct']:.2f}% "
                    f"pos={status['position_usd']:.2f} "
                    f"ok={trading_ok}"
                )

                if not trading_ok:
                    logger.critical(f"Trading disabled: {status['kill_reason']}")
                    self.stop()
                    break

            except KeyboardInterrupt:
                self.stop()
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}", exc_info=True)
