"""Donchian Channel Breakout (Turtle-style) strategy.

Enter on N-period high/low breakout, exit on shorter-period opposite breakout.
ATR-based position sizing and trailing stop.
"""
import logging
from collections import deque
from typing import Optional

import numpy as np

from .base import BaseStrategy, Signal

logger = logging.getLogger(__name__)


class DonchianBreakoutStrategy(BaseStrategy):

    def __init__(self, config):
        super().__init__(config)
        self.entry_period = config.don_entry_period
        self.exit_period = config.don_exit_period
        self.atr_period = config.don_atr_period
        self.atr_sl_mult = config.don_atr_sl_mult
        self.order_size_usd = config.don_order_size_usd

        max_len = max(self.entry_period, self.exit_period, self.atr_period) + 10
        self.closes = deque(maxlen=max_len)
        self.highs = deque(maxlen=max_len)
        self.lows = deque(maxlen=max_len)

        self.current_position_side = None
        self.entry_price = 0.0
        self.trailing_stop = None

    def _atr(self) -> Optional[float]:
        if len(self.closes) < self.atr_period + 1:
            return None
        closes = list(self.closes)
        highs = list(self.highs)
        lows = list(self.lows)
        trs = []
        for i in range(-self.atr_period, 0):
            tr = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i - 1]),
                     abs(lows[i] - closes[i - 1]))
            trs.append(tr)
        return np.mean(trs)

    def on_kline(self, kline: dict) -> Optional[Signal]:
        close = float(kline.get("close", kline.get("c", 0)))
        high = float(kline.get("high", kline.get("h", 0)))
        low = float(kline.get("low", kline.get("l", 0)))
        if not kline.get("confirm", True):
            return None

        self.closes.append(close)
        self.highs.append(high)
        self.lows.append(low)

        if len(self.highs) < self.entry_period + 1:
            return None

        atr = self._atr()
        if atr is None or atr == 0:
            return None

        highs = list(self.highs)
        lows = list(self.lows)

        # Entry channels (exclude current bar)
        entry_high = max(highs[-(self.entry_period + 1):-1])
        entry_low = min(lows[-(self.entry_period + 1):-1])

        # Exit channels
        exit_high = max(highs[-(self.exit_period + 1):-1]) if len(highs) > self.exit_period else entry_high
        exit_low = min(lows[-(self.exit_period + 1):-1]) if len(lows) > self.exit_period else entry_low

        # Trailing stop check
        if self.current_position_side == "Buy" and self.trailing_stop:
            new_stop = close - atr * self.atr_sl_mult
            self.trailing_stop = max(self.trailing_stop, new_stop)
            if close <= self.trailing_stop:
                return Signal(side="Sell", qty_usd=self.order_size_usd,
                              reason="don_trailing_stop_long")
        elif self.current_position_side == "Sell" and self.trailing_stop:
            new_stop = close + atr * self.atr_sl_mult
            self.trailing_stop = min(self.trailing_stop, new_stop)
            if close >= self.trailing_stop:
                return Signal(side="Buy", qty_usd=self.order_size_usd,
                              reason="don_trailing_stop_short")

        # Exit on opposite channel
        if self.current_position_side == "Buy" and close <= exit_low:
            return Signal(side="Sell", qty_usd=self.order_size_usd,
                          reason="don_exit_long")
        elif self.current_position_side == "Sell" and close >= exit_high:
            return Signal(side="Buy", qty_usd=self.order_size_usd,
                          reason="don_exit_short")

        # Entry on breakout
        if close > entry_high and self.current_position_side is None:
            self.trailing_stop = close - atr * self.atr_sl_mult
            return Signal(side="Buy", qty_usd=self.order_size_usd,
                          reason="don_breakout_long")
        elif close < entry_low and self.current_position_side is None:
            self.trailing_stop = close + atr * self.atr_sl_mult
            return Signal(side="Sell", qty_usd=self.order_size_usd,
                          reason="don_breakout_short")

        return None

    def on_position_update(self, position: dict):
        size = float(position.get("size", 0))
        if size == 0:
            self.current_position_side = None
            self.entry_price = 0.0
            self.trailing_stop = None
        else:
            self.current_position_side = position.get("side", "")
            self.entry_price = float(position.get("avgPrice", 0))
