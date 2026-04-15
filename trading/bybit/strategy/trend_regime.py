"""Trend Regime strategy.

Multi-indicator trend confirmation with strict regime filter.
Only enters in strong trends (ADX + MA slope + momentum confirmation).
Wide ATR-based trailing stop for large moves.
Designed for 4H/Daily timeframes with few trades per month.
"""
import logging
from collections import deque
from typing import Optional

import numpy as np

from .base import BaseStrategy, Signal

logger = logging.getLogger(__name__)


class TrendRegimeStrategy(BaseStrategy):
    """
    Strong trend-only strategy:
    - Entry: 50-period MA slope confirms trend + ADX > threshold + price above/below MA
    - Exit: ATR trailing stop (wide) or MA reversal
    - Very few trades, aims for big winners
    """

    def __init__(self, config):
        super().__init__(config)
        self.ma_period = config.tr_ma_period
        self.adx_period = config.tr_adx_period
        self.adx_threshold = config.tr_adx_threshold
        self.atr_period = config.tr_atr_period
        self.atr_sl_mult = config.tr_atr_sl_mult
        self.slope_lookback = config.tr_slope_lookback
        self.order_size_usd = config.tr_order_size_usd

        max_len = max(self.ma_period, self.adx_period * 2, self.atr_period) + 20
        self.closes = deque(maxlen=max_len)
        self.highs = deque(maxlen=max_len)
        self.lows = deque(maxlen=max_len)

        self.current_position_side = None
        self.entry_price = 0.0
        self.trailing_stop = None
        self.highest_since_entry = 0.0
        self.lowest_since_entry = float('inf')

    def _sma(self, n: int) -> float:
        return np.mean(list(self.closes)[-n:])

    def _ma_slope(self) -> float:
        """MA slope as percentage change over slope_lookback bars."""
        closes = list(self.closes)
        if len(closes) < self.ma_period + self.slope_lookback:
            return 0.0
        current_ma = np.mean(closes[-self.ma_period:])
        past_ma = np.mean(closes[-(self.ma_period + self.slope_lookback):-self.slope_lookback])
        if past_ma == 0:
            return 0.0
        return (current_ma - past_ma) / past_ma * 100

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

        atr_val = np.mean(tr_list[-n:])
        if atr_val == 0:
            return 0.0
        plus_di = np.mean(plus_dm[-n:]) / atr_val * 100
        minus_di = np.mean(minus_dm[-n:]) / atr_val * 100
        di_sum = plus_di + minus_di
        if di_sum == 0:
            return 0.0, 0.0, 0.0
        dx = abs(plus_di - minus_di) / di_sum * 100
        return dx, plus_di, minus_di

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

        if len(self.closes) < self.ma_period + self.slope_lookback + 2:
            return None

        ma = self._sma(self.ma_period)
        slope = self._ma_slope()
        adx_result = self._adx()
        atr = self._atr()

        if atr is None or atr == 0:
            return None

        if isinstance(adx_result, tuple):
            adx, plus_di, minus_di = adx_result
        else:
            adx, plus_di, minus_di = adx_result, 0, 0

        # Track extremes for trailing stop
        if self.current_position_side == "Buy":
            self.highest_since_entry = max(self.highest_since_entry, high)
        elif self.current_position_side == "Sell":
            self.lowest_since_entry = min(self.lowest_since_entry, low)

        # Trailing stop management (chandelier exit)
        if self.current_position_side == "Buy" and self.trailing_stop:
            new_stop = self.highest_since_entry - atr * self.atr_sl_mult
            self.trailing_stop = max(self.trailing_stop, new_stop)
            if close <= self.trailing_stop:
                return Signal(side="Sell", qty_usd=self.order_size_usd,
                              reason="tr_trailing_stop_long")
        elif self.current_position_side == "Sell" and self.trailing_stop:
            new_stop = self.lowest_since_entry + atr * self.atr_sl_mult
            self.trailing_stop = min(self.trailing_stop, new_stop)
            if close >= self.trailing_stop:
                return Signal(side="Buy", qty_usd=self.order_size_usd,
                              reason="tr_trailing_stop_short")

        # Exit if MA reverses
        if self.current_position_side == "Buy" and slope < -0.5:
            return Signal(side="Sell", qty_usd=self.order_size_usd,
                          reason="tr_ma_reversal_long")
        elif self.current_position_side == "Sell" and slope > 0.5:
            return Signal(side="Buy", qty_usd=self.order_size_usd,
                          reason="tr_ma_reversal_short")

        # Entry: strong trend confirmation
        if adx < self.adx_threshold:
            return None  # No trend

        # Long: price above MA + positive slope + +DI > -DI
        if (close > ma and slope > 1.0 and plus_di > minus_di and
                self.current_position_side is None):
            self.trailing_stop = close - atr * self.atr_sl_mult
            self.highest_since_entry = high
            return Signal(side="Buy", qty_usd=self.order_size_usd,
                          reason="tr_trend_long")

        # Short: price below MA + negative slope + -DI > +DI
        if (close < ma and slope < -1.0 and minus_di > plus_di and
                self.current_position_side is None):
            self.trailing_stop = close + atr * self.atr_sl_mult
            self.lowest_since_entry = low
            return Signal(side="Sell", qty_usd=self.order_size_usd,
                          reason="tr_trend_short")

        return None

    def on_position_update(self, position: dict):
        size = float(position.get("size", 0))
        if size == 0:
            self.current_position_side = None
            self.entry_price = 0.0
            self.trailing_stop = None
            self.highest_since_entry = 0.0
            self.lowest_since_entry = float('inf')
        else:
            self.current_position_side = position.get("side", "")
            self.entry_price = float(position.get("avgPrice", 0))
