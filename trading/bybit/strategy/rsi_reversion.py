"""RSI Mean Reversion strategy (Connors-style short-period RSI).

Uses 2-period RSI for aggressive mean reversion entries.
Exits at RSI midline or opposite extreme.
"""
import logging
from collections import deque
from typing import Optional

import numpy as np

from .base import BaseStrategy, Signal

logger = logging.getLogger(__name__)


class RSIReversionStrategy(BaseStrategy):

    def __init__(self, config):
        super().__init__(config)
        self.rsi_period = config.rsi_period
        self.oversold = config.rsi_oversold
        self.overbought = config.rsi_overbought
        self.exit_level = config.rsi_exit_level
        self.order_size_usd = config.rsi_order_size_usd

        self.closes = deque(maxlen=self.rsi_period + 50)
        self.current_position_side = None
        self.entry_price = 0.0

    def _rsi(self) -> Optional[float]:
        if len(self.closes) < self.rsi_period + 1:
            return None
        closes = list(self.closes)
        deltas = [closes[i] - closes[i - 1] for i in range(-self.rsi_period, 0)]
        gains = [d for d in deltas if d > 0]
        losses = [-d for d in deltas if d < 0]
        avg_gain = np.mean(gains) if gains else 0.0
        avg_loss = np.mean(losses) if losses else 0.0001
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def on_kline(self, kline: dict) -> Optional[Signal]:
        close = float(kline.get("close", kline.get("c", 0)))
        if not kline.get("confirm", True):
            return None

        self.closes.append(close)
        rsi = self._rsi()
        if rsi is None:
            return None

        # Exit
        if self.current_position_side == "Buy" and rsi >= self.exit_level:
            return Signal(side="Sell", qty_usd=self.order_size_usd,
                          reason=f"rsi_exit_long_{rsi:.0f}")
        elif self.current_position_side == "Sell" and rsi <= (100 - self.exit_level):
            return Signal(side="Buy", qty_usd=self.order_size_usd,
                          reason=f"rsi_exit_short_{rsi:.0f}")

        # Entry
        if rsi < self.oversold and self.current_position_side is None:
            return Signal(side="Buy", qty_usd=self.order_size_usd,
                          reason=f"rsi_oversold_{rsi:.0f}")
        elif rsi > self.overbought and self.current_position_side is None:
            return Signal(side="Sell", qty_usd=self.order_size_usd,
                          reason=f"rsi_overbought_{rsi:.0f}")

        return None

    def on_position_update(self, position: dict):
        size = float(position.get("size", 0))
        if size == 0:
            self.current_position_side = None
            self.entry_price = 0.0
        else:
            self.current_position_side = position.get("side", "")
            self.entry_price = float(position.get("avgPrice", 0))
