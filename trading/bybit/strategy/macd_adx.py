"""MACD + ADX Trend Following strategy.

MACD crossover for entry/exit, ADX filter to confirm trend strength.
Only trades when ADX > threshold (trending market).
"""
import logging
from collections import deque
from typing import Optional

import numpy as np

from .base import BaseStrategy, Signal

logger = logging.getLogger(__name__)


class MACDADXStrategy(BaseStrategy):

    def __init__(self, config):
        super().__init__(config)
        self.fast = config.macd_fast
        self.slow = config.macd_slow
        self.signal_period = config.macd_signal
        self.adx_period = config.macd_adx_period
        self.adx_threshold = config.macd_adx_threshold
        self.atr_period = config.macd_atr_period
        self.atr_sl_mult = config.macd_atr_sl_mult
        self.order_size_usd = config.macd_order_size_usd

        max_len = max(self.slow, self.adx_period * 2, self.atr_period) + 20
        self.closes = deque(maxlen=max_len)
        self.highs = deque(maxlen=max_len)
        self.lows = deque(maxlen=max_len)

        self.fast_ema = None
        self.slow_ema = None
        self.signal_ema = None
        self.prev_histogram = None

        self.current_position_side = None
        self.entry_price = 0.0
        self.trailing_stop = None

    def _ema(self, value: float, prev: Optional[float], period: int) -> float:
        if prev is None:
            return value
        k = 2.0 / (period + 1)
        return value * k + prev * (1 - k)

    def _adx(self) -> float:
        n = self.adx_period
        closes = list(self.closes)
        highs = list(self.highs)
        lows = list(self.lows)
        if len(closes) < n * 2 + 1:
            return 0.0

        plus_dm, minus_dm, tr_list = [], [], []
        for i in range(-n * 2, 0):
            h_diff = highs[i] - highs[i - 1]
            l_diff = lows[i - 1] - lows[i]
            plus_dm.append(max(h_diff, 0) if h_diff > l_diff else 0)
            minus_dm.append(max(l_diff, 0) if l_diff > h_diff else 0)
            tr = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i - 1]),
                     abs(lows[i] - closes[i - 1]))
            tr_list.append(tr)

        atr = np.mean(tr_list[-n:])
        if atr == 0:
            return 0.0
        plus_di = np.mean(plus_dm[-n:]) / atr * 100
        minus_di = np.mean(minus_dm[-n:]) / atr * 100
        di_sum = plus_di + minus_di
        if di_sum == 0:
            return 0.0
        return abs(plus_di - minus_di) / di_sum * 100

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

        if len(self.closes) < self.slow + 5:
            return None

        # MACD
        self.fast_ema = self._ema(close, self.fast_ema, self.fast)
        self.slow_ema = self._ema(close, self.slow_ema, self.slow)
        macd_line = self.fast_ema - self.slow_ema
        self.signal_ema = self._ema(macd_line, self.signal_ema, self.signal_period)
        histogram = macd_line - self.signal_ema

        adx = self._adx()
        atr = self._atr()

        if self.prev_histogram is None:
            self.prev_histogram = histogram
            return None

        # Trailing stop
        if atr and self.current_position_side == "Buy" and self.trailing_stop:
            new_stop = close - atr * self.atr_sl_mult
            self.trailing_stop = max(self.trailing_stop, new_stop)
            if close <= self.trailing_stop:
                self.prev_histogram = histogram
                return Signal(side="Sell", qty_usd=self.order_size_usd,
                              reason="macd_trailing_stop_long")
        elif atr and self.current_position_side == "Sell" and self.trailing_stop:
            new_stop = close + atr * self.atr_sl_mult
            self.trailing_stop = min(self.trailing_stop, new_stop)
            if close >= self.trailing_stop:
                self.prev_histogram = histogram
                return Signal(side="Buy", qty_usd=self.order_size_usd,
                              reason="macd_trailing_stop_short")

        # MACD crossover detection
        cross_up = self.prev_histogram <= 0 and histogram > 0
        cross_down = self.prev_histogram >= 0 and histogram < 0
        self.prev_histogram = histogram

        # Only trade in trending markets
        if adx < self.adx_threshold:
            return None

        # Exit on opposite cross
        if self.current_position_side == "Buy" and cross_down:
            return Signal(side="Sell", qty_usd=self.order_size_usd,
                          reason="macd_exit_long")
        elif self.current_position_side == "Sell" and cross_up:
            return Signal(side="Buy", qty_usd=self.order_size_usd,
                          reason="macd_exit_short")

        # Entry
        if cross_up and macd_line > 0 and self.current_position_side is None:
            if atr:
                self.trailing_stop = close - atr * self.atr_sl_mult
            return Signal(side="Buy", qty_usd=self.order_size_usd,
                          reason="macd_cross_long")
        elif cross_down and macd_line < 0 and self.current_position_side is None:
            if atr:
                self.trailing_stop = close + atr * self.atr_sl_mult
            return Signal(side="Sell", qty_usd=self.order_size_usd,
                          reason="macd_cross_short")

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
