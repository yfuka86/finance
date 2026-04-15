"""Volatility Breakout strategy (Larry Williams / Dual Thrust style).

Enter when price breaks Open + k * Range.
Range = max(HH - LC, HC - LL) over N days for Dual Thrust robustness.
Exit at session end or trailing stop.
"""
import logging
from collections import deque
from typing import Optional

import numpy as np

from .base import BaseStrategy, Signal

logger = logging.getLogger(__name__)


class VolatilityBreakoutStrategy(BaseStrategy):

    def __init__(self, config):
        super().__init__(config)
        self.lookback = config.vb_lookback
        self.k_long = config.vb_k_long
        self.k_short = config.vb_k_short
        self.atr_period = config.vb_atr_period
        self.atr_sl_mult = config.vb_atr_sl_mult
        self.order_size_usd = config.vb_order_size_usd

        max_len = max(self.lookback, self.atr_period) + 10
        self.opens = deque(maxlen=max_len)
        self.closes = deque(maxlen=max_len)
        self.highs = deque(maxlen=max_len)
        self.lows = deque(maxlen=max_len)

        self.current_position_side = None
        self.entry_price = 0.0
        self.trailing_stop = None
        self.bars_in_position = 0
        self.session_bars = 0  # count bars per "session"
        self.session_length = config.vb_session_bars

    def _dual_thrust_range(self) -> Optional[float]:
        if len(self.highs) < self.lookback + 1:
            return None
        highs = list(self.highs)[-(self.lookback + 1):-1]
        lows = list(self.lows)[-(self.lookback + 1):-1]
        closes = list(self.closes)[-(self.lookback + 1):-1]
        hh = max(highs)
        hc = max(closes)
        lc = min(closes)
        ll = min(lows)
        return max(hh - lc, hc - ll)

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
        open_ = float(kline.get("open", kline.get("o", 0)))
        if not kline.get("confirm", True):
            return None

        self.opens.append(open_)
        self.closes.append(close)
        self.highs.append(high)
        self.lows.append(low)
        self.session_bars += 1

        if self.current_position_side:
            self.bars_in_position += 1

        rng = self._dual_thrust_range()
        atr = self._atr()
        if rng is None or atr is None or atr == 0:
            return None

        # Session-based exit
        if self.current_position_side and self.bars_in_position >= self.session_length:
            side = "Sell" if self.current_position_side == "Buy" else "Buy"
            return Signal(side=side, qty_usd=self.order_size_usd,
                          reason="vb_session_exit")

        # Trailing stop
        if self.current_position_side == "Buy" and self.trailing_stop:
            new_stop = close - atr * self.atr_sl_mult
            self.trailing_stop = max(self.trailing_stop, new_stop)
            if close <= self.trailing_stop:
                return Signal(side="Sell", qty_usd=self.order_size_usd,
                              reason="vb_trailing_stop_long")
        elif self.current_position_side == "Sell" and self.trailing_stop:
            new_stop = close + atr * self.atr_sl_mult
            self.trailing_stop = min(self.trailing_stop, new_stop)
            if close >= self.trailing_stop:
                return Signal(side="Buy", qty_usd=self.order_size_usd,
                              reason="vb_trailing_stop_short")

        # Entry: use the open of the current session (approximated by lookback period ago)
        ref_open = open_
        upper_trigger = ref_open + self.k_long * rng
        lower_trigger = ref_open - self.k_short * rng

        if close > upper_trigger and self.current_position_side is None:
            self.trailing_stop = close - atr * self.atr_sl_mult
            self.bars_in_position = 0
            return Signal(side="Buy", qty_usd=self.order_size_usd,
                          reason="vb_breakout_long")
        elif close < lower_trigger and self.current_position_side is None:
            self.trailing_stop = close + atr * self.atr_sl_mult
            self.bars_in_position = 0
            return Signal(side="Sell", qty_usd=self.order_size_usd,
                          reason="vb_breakout_short")

        # Reversal: if in position, opposite signal triggers close + reverse
        if close > upper_trigger and self.current_position_side == "Sell":
            self.trailing_stop = close - atr * self.atr_sl_mult
            self.bars_in_position = 0
            return Signal(side="Buy", qty_usd=self.order_size_usd * 2,
                          reason="vb_reverse_long")
        elif close < lower_trigger and self.current_position_side == "Buy":
            self.trailing_stop = close + atr * self.atr_sl_mult
            self.bars_in_position = 0
            return Signal(side="Sell", qty_usd=self.order_size_usd * 2,
                          reason="vb_reverse_short")

        return None

    def on_position_update(self, position: dict):
        size = float(position.get("size", 0))
        if size == 0:
            self.current_position_side = None
            self.entry_price = 0.0
            self.trailing_stop = None
            self.bars_in_position = 0
        else:
            self.current_position_side = position.get("side", "")
            self.entry_price = float(position.get("avgPrice", 0))
