"""Ichimoku Cloud strategy.

Full TK cross + cloud filter + Chikou confirmation.
Trend-following system with cloud-based trailing stop.
"""
import logging
from collections import deque
from typing import Optional

import numpy as np

from .base import BaseStrategy, Signal

logger = logging.getLogger(__name__)


class IchimokuStrategy(BaseStrategy):

    def __init__(self, config):
        super().__init__(config)
        self.tenkan_period = config.ichi_tenkan
        self.kijun_period = config.ichi_kijun
        self.senkou_b_period = config.ichi_senkou_b
        self.order_size_usd = config.ichi_order_size_usd

        max_len = max(self.senkou_b_period, self.kijun_period * 2) + 30
        self.closes = deque(maxlen=max_len)
        self.highs = deque(maxlen=max_len)
        self.lows = deque(maxlen=max_len)

        self.current_position_side = None
        self.entry_price = 0.0
        self.prev_tenkan = None
        self.prev_kijun = None

    def _midpoint(self, data, period) -> float:
        d = list(data)[-period:]
        return (max(d) + min(d)) / 2

    def on_kline(self, kline: dict) -> Optional[Signal]:
        close = float(kline.get("close", kline.get("c", 0)))
        high = float(kline.get("high", kline.get("h", 0)))
        low = float(kline.get("low", kline.get("l", 0)))
        if not kline.get("confirm", True):
            return None

        self.closes.append(close)
        self.highs.append(high)
        self.lows.append(low)

        if len(self.highs) < self.senkou_b_period + 2:
            return None

        # Ichimoku components
        tenkan = self._midpoint(self.highs, self.tenkan_period)
        kijun = self._midpoint(self.highs, self.kijun_period)
        # Use highs for tenkan/kijun is wrong — need to use both highs and lows
        tenkan = (max(list(self.highs)[-self.tenkan_period:]) +
                  min(list(self.lows)[-self.tenkan_period:])) / 2
        kijun = (max(list(self.highs)[-self.kijun_period:]) +
                 min(list(self.lows)[-self.kijun_period:])) / 2

        senkou_a = (tenkan + kijun) / 2
        senkou_b = (max(list(self.highs)[-self.senkou_b_period:]) +
                    min(list(self.lows)[-self.senkou_b_period:])) / 2

        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)

        # Chikou: current close vs close kijun_period ago
        chikou_bullish = close > list(self.closes)[-(self.kijun_period + 1)] if len(self.closes) > self.kijun_period else False
        chikou_bearish = close < list(self.closes)[-(self.kijun_period + 1)] if len(self.closes) > self.kijun_period else False

        # TK cross detection
        tk_cross_up = (self.prev_tenkan is not None and
                       self.prev_tenkan <= self.prev_kijun and
                       tenkan > kijun)
        tk_cross_down = (self.prev_tenkan is not None and
                         self.prev_tenkan >= self.prev_kijun and
                         tenkan < kijun)

        self.prev_tenkan = tenkan
        self.prev_kijun = kijun

        # Exit: price enters cloud or opposite TK cross
        if self.current_position_side == "Buy":
            if close < cloud_bottom or tk_cross_down:
                return Signal(side="Sell", qty_usd=self.order_size_usd,
                              reason="ichi_exit_long")
        elif self.current_position_side == "Sell":
            if close > cloud_top or tk_cross_up:
                return Signal(side="Buy", qty_usd=self.order_size_usd,
                              reason="ichi_exit_short")

        # Entry: TK cross + price above/below cloud + chikou confirmation
        if tk_cross_up and close > cloud_top and chikou_bullish:
            if self.current_position_side is None:
                return Signal(side="Buy", qty_usd=self.order_size_usd,
                              reason="ichi_tk_cross_long")
            elif self.current_position_side == "Sell":
                return Signal(side="Buy", qty_usd=self.order_size_usd * 2,
                              reason="ichi_reverse_long")

        if tk_cross_down and close < cloud_bottom and chikou_bearish:
            if self.current_position_side is None:
                return Signal(side="Sell", qty_usd=self.order_size_usd,
                              reason="ichi_tk_cross_short")
            elif self.current_position_side == "Buy":
                return Signal(side="Sell", qty_usd=self.order_size_usd * 2,
                              reason="ichi_reverse_short")

        return None

    def on_position_update(self, position: dict):
        size = float(position.get("size", 0))
        if size == 0:
            self.current_position_side = None
            self.entry_price = 0.0
        else:
            self.current_position_side = position.get("side", "")
            self.entry_price = float(position.get("avgPrice", 0))
