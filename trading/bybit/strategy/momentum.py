"""EMA crossover + ATR-based momentum strategy.

References:
- Freqtrade's default strategy patterns
- Classic MACD-style signal generation with ATR-based risk management
"""
import logging
from collections import deque
from typing import Optional

import numpy as np

from .base import BaseStrategy, Signal

logger = logging.getLogger(__name__)


class MomentumStrategy(BaseStrategy):
    """
    Trend-following strategy using dual EMA crossover.

    Entry: Fast EMA crosses above/below Slow EMA.
    Exit: Reverse crossover or ATR-based trailing stop.
    Position sizing: Fixed USD amount per trade.
    """

    def __init__(self, config):
        super().__init__(config)
        self.fast_period = config.mom_fast_period
        self.slow_period = config.mom_slow_period
        self.atr_period = config.mom_atr_period
        self.atr_mult = config.mom_atr_multiplier
        self.order_size_usd = config.mom_order_size_usd

        # Price history for EMA calculation
        max_len = max(self.slow_period, self.atr_period) + 50
        self.closes = deque(maxlen=max_len)
        self.highs = deque(maxlen=max_len)
        self.lows = deque(maxlen=max_len)

        # State
        self.fast_ema = None
        self.slow_ema = None
        self.atr = None
        self.prev_signal_side = None  # Track last signal to avoid duplicates
        self.current_position_side = None  # "Buy" or "Sell" or None
        self.entry_price = 0.0
        self.trailing_stop = None

    def _update_ema(self, value: float, prev_ema: Optional[float], period: int) -> float:
        if prev_ema is None:
            return value
        k = 2.0 / (period + 1)
        return value * k + prev_ema * (1 - k)

    def _compute_atr(self) -> Optional[float]:
        if len(self.closes) < self.atr_period + 1:
            return None
        closes = list(self.closes)
        highs = list(self.highs)
        lows = list(self.lows)
        trs = []
        for i in range(-self.atr_period, 0):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
            trs.append(tr)
        return np.mean(trs)

    def on_kline(self, kline: dict) -> Optional[Signal]:
        """Process closed kline and generate signal."""
        # kline format from pybit: [startTime, open, high, low, close, volume, turnover]
        if isinstance(kline, dict):
            close = float(kline.get("close", kline.get("c", 0)))
            high = float(kline.get("high", kline.get("h", 0)))
            low = float(kline.get("low", kline.get("l", 0)))
            confirm = kline.get("confirm", True)
        else:
            # List format from REST API
            close = float(kline[4])
            high = float(kline[2])
            low = float(kline[3])
            confirm = True

        if not confirm:
            return None  # Skip unconfirmed candles

        self.closes.append(close)
        self.highs.append(high)
        self.lows.append(low)

        # Need enough data
        if len(self.closes) < self.slow_period + 2:
            return None

        # Update indicators
        self.fast_ema = self._update_ema(close, self.fast_ema, self.fast_period)
        self.slow_ema = self._update_ema(close, self.slow_ema, self.slow_period)
        self.atr = self._compute_atr()

        if self.atr is None or self.atr == 0:
            return None

        # Check trailing stop
        if self.current_position_side and self.trailing_stop:
            if self.current_position_side == "Buy" and close <= self.trailing_stop:
                logger.info(f"Trailing stop hit (long): {close:.2f} <= {self.trailing_stop:.2f}")
                self.current_position_side = None
                self.prev_signal_side = None
                return Signal(side="Sell", qty_usd=self.order_size_usd,
                              reason="trailing_stop_long")
            elif self.current_position_side == "Sell" and close >= self.trailing_stop:
                logger.info(f"Trailing stop hit (short): {close:.2f} >= {self.trailing_stop:.2f}")
                self.current_position_side = None
                self.prev_signal_side = None
                return Signal(side="Buy", qty_usd=self.order_size_usd,
                              reason="trailing_stop_short")

        # Update trailing stop
        if self.current_position_side == "Buy":
            new_stop = close - self.atr * self.atr_mult
            if self.trailing_stop is None or new_stop > self.trailing_stop:
                self.trailing_stop = new_stop
        elif self.current_position_side == "Sell":
            new_stop = close + self.atr * self.atr_mult
            if self.trailing_stop is None or new_stop < self.trailing_stop:
                self.trailing_stop = new_stop

        # Generate crossover signal
        signal_side = "Buy" if self.fast_ema > self.slow_ema else "Sell"

        if signal_side == self.prev_signal_side:
            return None  # No crossover

        self.prev_signal_side = signal_side

        # If we have a position in the opposite direction, close + reverse
        if self.current_position_side and self.current_position_side != signal_side:
            qty = self.order_size_usd * 2  # Close + open reverse
        else:
            qty = self.order_size_usd

        stop = close - self.atr * self.atr_mult if signal_side == "Buy" else close + self.atr * self.atr_mult
        tp = close + self.atr * self.atr_mult * 2 if signal_side == "Buy" else close - self.atr * self.atr_mult * 2

        logger.info(
            f"Momentum signal: {signal_side} | fast_ema={self.fast_ema:.2f} "
            f"slow_ema={self.slow_ema:.2f} ATR={self.atr:.2f} SL={stop:.2f} TP={tp:.2f}"
        )

        return Signal(
            side=signal_side,
            qty_usd=qty,
            price=None,  # Market order
            stop_loss=round(stop, 2),
            take_profit=round(tp, 2),
            reason=f"ema_crossover_{signal_side.lower()}",
        )

    def on_position_update(self, position: dict):
        size = float(position.get("size", 0))
        side = position.get("side", "")
        if size == 0:
            self.current_position_side = None
            self.trailing_stop = None
            self.entry_price = 0.0
        else:
            self.current_position_side = side
            self.entry_price = float(position.get("avgPrice", 0))
