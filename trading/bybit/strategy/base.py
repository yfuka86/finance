"""Base strategy interface."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class Signal:
    """Trading signal produced by a strategy."""
    side: Optional[str] = None   # "Buy" or "Sell" or None
    qty_usd: float = 0.0
    price: Optional[float] = None  # None = market order
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reason: str = ""


class BaseStrategy(ABC):
    """All strategies must implement this interface."""

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def on_kline(self, kline: dict) -> Optional[Signal]:
        """Called on each new kline (candle close). Return Signal or None."""

    def on_orderbook(self, orderbook: dict) -> Optional[list[Signal]]:
        """Called on orderbook update. Return list of Signals or None."""
        return None

    def on_trade(self, trade: dict) -> Optional[Signal]:
        """Called on each public trade. Return Signal or None."""
        return None

    @abstractmethod
    def on_position_update(self, position: dict):
        """Called when position changes."""

    def on_order_update(self, order: dict):
        """Called when order status changes."""
        pass
