"""Bollinger Band Mean Reversion strategy.

Buy when price closes below lower band, sell when above upper band.
Exit at middle band (SMA). ADX filter to avoid trending markets.
"""
import logging
from collections import deque
from typing import Optional

import numpy as np

from .base import BaseStrategy, Signal

logger = logging.getLogger(__name__)


class BollingerReversionStrategy(BaseStrategy):

    def __init__(self, config):
        super().__init__(config)
        self.period = config.bb_period
        self.std_mult = config.bb_std_mult
        self.adx_period = config.bb_adx_period
        self.adx_thresh = config.bb_adx_threshold
        self.order_size_usd = config.bb_order_size_usd

        max_len = max(self.period, self.adx_period * 2) + 10
        self.closes = deque(maxlen=max_len)
        self.highs = deque(maxlen=max_len)
        self.lows = deque(maxlen=max_len)

        self.current_position_side = None
        self.entry_price = 0.0

    def _sma(self, n: int) -> float:
        return np.mean(list(self.closes)[-n:])

    def _std(self, n: int) -> float:
        return np.std(list(self.closes)[-n:], ddof=1)

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
        dx = abs(plus_di - minus_di) / di_sum * 100
        return dx

    def on_kline(self, kline: dict) -> Optional[Signal]:
        close = float(kline.get("close", kline.get("c", 0)))
        high = float(kline.get("high", kline.get("h", 0)))
        low = float(kline.get("low", kline.get("l", 0)))
        if not kline.get("confirm", True):
            return None

        self.closes.append(close)
        self.highs.append(high)
        self.lows.append(low)

        if len(self.closes) < max(self.period, self.adx_period * 2) + 2:
            return None

        sma = self._sma(self.period)
        std = self._std(self.period)
        upper = sma + self.std_mult * std
        lower = sma - self.std_mult * std
        adx = self._adx()

        # Only trade in ranging markets
        if adx > self.adx_thresh:
            # If in position, check if we should exit at SMA
            if self.current_position_side == "Buy" and close >= sma:
                return Signal(side="Sell", qty_usd=self.order_size_usd,
                              reason="bb_exit_long_sma")
            elif self.current_position_side == "Sell" and close <= sma:
                return Signal(side="Buy", qty_usd=self.order_size_usd,
                              reason="bb_exit_short_sma")
            return None

        # Exit at SMA
        if self.current_position_side == "Buy" and close >= sma:
            return Signal(side="Sell", qty_usd=self.order_size_usd,
                          reason="bb_exit_long_sma")
        elif self.current_position_side == "Sell" and close <= sma:
            return Signal(side="Buy", qty_usd=self.order_size_usd,
                          reason="bb_exit_short_sma")

        # Entry signals
        if close < lower and self.current_position_side is None:
            return Signal(side="Buy", qty_usd=self.order_size_usd,
                          reason="bb_lower_touch")
        elif close > upper and self.current_position_side is None:
            return Signal(side="Sell", qty_usd=self.order_size_usd,
                          reason="bb_upper_touch")

        return None

    def on_position_update(self, position: dict):
        size = float(position.get("size", 0))
        if size == 0:
            self.current_position_side = None
            self.entry_price = 0.0
        else:
            self.current_position_side = position.get("side", "")
            self.entry_price = float(position.get("avgPrice", 0))
