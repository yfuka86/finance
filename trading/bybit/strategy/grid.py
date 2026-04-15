"""Grid trading strategy for range-bound markets.

Places buy and sell limit orders at fixed intervals. When a buy fills,
a sell is placed one grid level above (and vice versa), capturing profit
from price oscillations.
"""
import logging
from typing import Optional

import numpy as np

from .base import BaseStrategy, Signal

logger = logging.getLogger(__name__)


class GridStrategy(BaseStrategy):
    """
    Grid trading: profit from price oscillation in a defined range.

    Setup: Define upper/lower bounds and number of grid levels.
    Each filled buy creates a sell one level up, and vice versa.
    """

    def __init__(self, config):
        super().__init__(config)
        self.upper_pct = config.grid_upper_pct
        self.lower_pct = config.grid_lower_pct
        self.num_grids = config.grid_num_grids
        self.order_size_usd = config.grid_order_size_usd

        # Grid state
        self.grid_prices: list[float] = []
        self.grid_initialized = False
        self.filled_levels: set[int] = set()  # Levels where buy was filled
        self.active_orders: dict[str, int] = {}  # orderId -> grid level

    def initialize_grid(self, mid_price: float):
        """Set up grid levels around current price."""
        upper = mid_price * (1 + self.upper_pct / 100)
        lower = mid_price * (1 - self.lower_pct / 100)
        self.grid_prices = list(np.linspace(lower, upper, self.num_grids + 1))
        self.grid_initialized = True
        self.filled_levels.clear()
        self.active_orders.clear()
        logger.info(
            f"Grid initialized: {lower:.2f} - {upper:.2f}, "
            f"{self.num_grids} levels, spacing={self.grid_prices[1]-self.grid_prices[0]:.2f}"
        )

    def on_kline(self, kline: dict) -> Optional[Signal]:
        """Initialize grid on first kline if not done."""
        if isinstance(kline, dict):
            close = float(kline.get("close", kline.get("c", 0)))
        else:
            close = float(kline[4])

        if not self.grid_initialized and close > 0:
            self.initialize_grid(close)
            return self._place_initial_orders(close)
        return None

    def _place_initial_orders(self, current_price: float) -> Optional[Signal]:
        """Return signals for initial grid orders (handled by engine as batch)."""
        # We return None here - the engine will call get_initial_orders()
        return None

    def get_initial_orders(self, current_price: float) -> list[Signal]:
        """Generate all initial grid orders."""
        if not self.grid_initialized:
            return []

        signals = []
        for i, price in enumerate(self.grid_prices):
            if price < current_price:
                signals.append(Signal(
                    side="Buy",
                    qty_usd=self.order_size_usd,
                    price=price,
                    reason=f"grid_buy_L{i}",
                ))
            elif price > current_price:
                signals.append(Signal(
                    side="Sell",
                    qty_usd=self.order_size_usd,
                    price=price,
                    reason=f"grid_sell_L{i}",
                ))
        logger.info(f"Initial grid orders: {len(signals)}")
        return signals

    def on_order_update(self, order: dict):
        """When a grid order fills, place the reverse order one level away."""
        order_id = order.get("orderId", "")
        status = order.get("orderStatus", "")

        if status != "Filled":
            return

        level = self.active_orders.pop(order_id, None)
        if level is None:
            return

        side = order.get("side", "")
        if side == "Buy":
            self.filled_levels.add(level)
            # Place sell one level up
            if level + 1 < len(self.grid_prices):
                logger.info(f"Grid buy filled at L{level}, placing sell at L{level+1}")
        elif side == "Sell":
            self.filled_levels.discard(level)
            # Place buy one level down
            if level - 1 >= 0:
                logger.info(f"Grid sell filled at L{level}, placing buy at L{level-1}")

    def get_fill_response(self, filled_order: dict) -> Optional[Signal]:
        """Generate response order when a grid level fills."""
        order_id = filled_order.get("orderId", "")
        side = filled_order.get("side", "")
        price = float(filled_order.get("price", 0))

        # Find closest grid level
        if not self.grid_prices:
            return None

        closest_level = min(range(len(self.grid_prices)),
                           key=lambda i: abs(self.grid_prices[i] - price))

        if side == "Buy" and closest_level + 1 < len(self.grid_prices):
            return Signal(
                side="Sell",
                qty_usd=self.order_size_usd,
                price=self.grid_prices[closest_level + 1],
                reason=f"grid_sell_response_L{closest_level+1}",
            )
        elif side == "Sell" and closest_level - 1 >= 0:
            return Signal(
                side="Buy",
                qty_usd=self.order_size_usd,
                price=self.grid_prices[closest_level - 1],
                reason=f"grid_buy_response_L{closest_level-1}",
            )
        return None

    def on_position_update(self, position: dict):
        pass
