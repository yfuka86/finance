"""R-Breaker strategy.

Intraday strategy combining breakout and reversal modes.
Calculates 6 pivot-based levels from previous session's OHLC.
"""
import logging
from collections import deque
from typing import Optional

from .base import BaseStrategy, Signal

logger = logging.getLogger(__name__)


class RBreakerStrategy(BaseStrategy):

    def __init__(self, config):
        super().__init__(config)
        self.f1 = config.rb_f1
        self.f2 = config.rb_f2
        self.f3 = config.rb_f3
        self.session_bars = config.rb_session_bars
        self.order_size_usd = config.rb_order_size_usd

        self.prev_high = None
        self.prev_low = None
        self.prev_close = None
        self.session_high = None
        self.session_low = None

        self.bar_count = 0
        self.current_position_side = None
        self.entry_price = 0.0

        # Session tracking
        self.session_highs = deque(maxlen=500)
        self.session_lows = deque(maxlen=500)
        self.session_closes = deque(maxlen=500)

    def _calc_levels(self):
        h, l, c = self.prev_high, self.prev_low, self.prev_close
        pivot = (h + l + c) / 3
        rng = h - l

        sell_setup = pivot + self.f1 * rng
        buy_setup = pivot - self.f1 * rng
        sell_enter = pivot + self.f2 * rng
        buy_enter = pivot - self.f2 * rng
        buy_break = sell_setup + self.f3 * (sell_setup - buy_setup)
        sell_break = buy_setup - self.f3 * (sell_setup - buy_setup)

        return {
            "buy_break": buy_break,
            "sell_setup": sell_setup,
            "sell_enter": sell_enter,
            "pivot": pivot,
            "buy_enter": buy_enter,
            "buy_setup": buy_setup,
            "sell_break": sell_break,
        }

    def on_kline(self, kline: dict) -> Optional[Signal]:
        close = float(kline.get("close", kline.get("c", 0)))
        high = float(kline.get("high", kline.get("h", 0)))
        low = float(kline.get("low", kline.get("l", 0)))
        if not kline.get("confirm", True):
            return None

        self.session_highs.append(high)
        self.session_lows.append(low)
        self.session_closes.append(close)
        self.bar_count += 1

        # Track session high/low
        if self.session_high is None:
            self.session_high = high
            self.session_low = low
        else:
            self.session_high = max(self.session_high, high)
            self.session_low = min(self.session_low, low)

        # Session end: rotate levels
        if self.bar_count >= self.session_bars:
            self.prev_high = self.session_high
            self.prev_low = self.session_low
            self.prev_close = close
            self.session_high = None
            self.session_low = None
            self.bar_count = 0

            # Close any position at session end
            if self.current_position_side:
                side = "Sell" if self.current_position_side == "Buy" else "Buy"
                return Signal(side=side, qty_usd=self.order_size_usd,
                              reason="rb_session_end")
            return None

        if self.prev_high is None:
            return None

        levels = self._calc_levels()

        # Breakout mode
        if close > levels["buy_break"] and self.current_position_side is None:
            return Signal(side="Buy", qty_usd=self.order_size_usd,
                          reason="rb_breakout_long")
        elif close < levels["sell_break"] and self.current_position_side is None:
            return Signal(side="Sell", qty_usd=self.order_size_usd,
                          reason="rb_breakout_short")

        # Reversal mode
        if (self.session_high is not None and
                self.session_high >= levels["sell_setup"] and
                close < levels["sell_enter"] and
                self.current_position_side != "Sell"):
            qty = self.order_size_usd * (2 if self.current_position_side == "Buy" else 1)
            return Signal(side="Sell", qty_usd=qty,
                          reason="rb_reversal_short")
        elif (self.session_low is not None and
              self.session_low <= levels["buy_setup"] and
              close > levels["buy_enter"] and
              self.current_position_side != "Buy"):
            qty = self.order_size_usd * (2 if self.current_position_side == "Sell" else 1)
            return Signal(side="Buy", qty_usd=qty,
                          reason="rb_reversal_long")

        return None

    def on_position_update(self, position: dict):
        size = float(position.get("size", 0))
        if size == 0:
            self.current_position_side = None
            self.entry_price = 0.0
        else:
            self.current_position_side = position.get("side", "")
            self.entry_price = float(position.get("avgPrice", 0))
