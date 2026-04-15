"""Market making strategy with inventory-aware spread adjustment.

References:
- Hummingbot's pure market making strategy
- Avellaneda-Stoikov market making model
"""
import logging
import time
from typing import Optional

from .base import BaseStrategy, Signal

logger = logging.getLogger(__name__)


class MarketMakerStrategy(BaseStrategy):
    """
    Place limit orders on both sides of the spread to capture bid-ask spread.

    Key features:
    - Inventory-aware: Widen spread on the side with excess inventory
    - Multi-level: Place orders at multiple price levels
    - Auto-refresh: Cancel and replace stale orders
    """

    def __init__(self, config):
        super().__init__(config)
        self.spread_bps = config.mm_spread_bps
        self.order_size_usd = config.mm_order_size_usd
        self.num_levels = config.mm_num_levels
        self.level_spacing_bps = config.mm_level_spacing_bps

        # State
        self.mid_price = None
        self.best_bid = None
        self.best_ask = None
        self.current_inventory = 0.0  # Positive = long, negative = short
        self.active_orders: dict[str, dict] = {}  # orderId -> order info
        self.last_refresh_time = 0
        self.max_inventory_usd = config.max_position_usd

    def on_orderbook(self, orderbook: dict) -> Optional[list[Signal]]:
        """Update quotes based on orderbook changes."""
        data = orderbook.get("data", orderbook)
        bids = data.get("b", [])
        asks = data.get("a", [])

        if not bids or not asks:
            return None

        self.best_bid = float(bids[0][0])
        self.best_ask = float(asks[0][0])
        self.mid_price = (self.best_bid + self.best_ask) / 2

        now = time.time()
        if now - self.last_refresh_time < self.config.order_refresh_interval:
            return None

        self.last_refresh_time = now
        return self._generate_quotes()

    def _generate_quotes(self) -> list[Signal]:
        """Generate bid and ask orders with inventory skew."""
        if self.mid_price is None:
            return []

        # Inventory skew: shift mid price to reduce inventory
        inventory_ratio = self.current_inventory / self.max_inventory_usd if self.max_inventory_usd else 0
        inventory_ratio = max(-1.0, min(1.0, inventory_ratio))

        # Skew: if long, shift quotes down (more aggressive sells, passive buys)
        skew_bps = inventory_ratio * self.spread_bps * 0.5
        adjusted_mid = self.mid_price * (1 - skew_bps / 10000)

        signals = []
        half_spread = self.spread_bps / 2

        for level in range(self.num_levels):
            offset_bps = half_spread + level * self.level_spacing_bps

            # Bid (buy) side - skip if max long inventory
            if inventory_ratio < 0.9:
                bid_price = adjusted_mid * (1 - offset_bps / 10000)
                signals.append(Signal(
                    side="Buy",
                    qty_usd=self.order_size_usd,
                    price=bid_price,
                    reason=f"mm_bid_L{level}",
                ))

            # Ask (sell) side - skip if max short inventory
            if inventory_ratio > -0.9:
                ask_price = adjusted_mid * (1 + offset_bps / 10000)
                signals.append(Signal(
                    side="Sell",
                    qty_usd=self.order_size_usd,
                    price=ask_price,
                    reason=f"mm_ask_L{level}",
                ))

        logger.info(
            f"MM quotes: mid={self.mid_price:.2f} skew={skew_bps:.1f}bps "
            f"inventory={self.current_inventory:.2f} orders={len(signals)}"
        )
        return signals

    def on_kline(self, kline: dict) -> Optional[Signal]:
        return None  # Market maker doesn't use klines for signals

    def on_position_update(self, position: dict):
        size = float(position.get("size", 0))
        side = position.get("side", "")
        avg_price = float(position.get("avgPrice", 0)) or 1.0
        if side == "Sell":
            self.current_inventory = -size * avg_price
        else:
            self.current_inventory = size * avg_price

    def on_order_update(self, order: dict):
        order_id = order.get("orderId", "")
        status = order.get("orderStatus", "")
        if status in ("Filled", "Cancelled", "Rejected"):
            self.active_orders.pop(order_id, None)
        else:
            self.active_orders[order_id] = order
