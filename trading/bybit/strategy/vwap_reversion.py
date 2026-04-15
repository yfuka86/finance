"""VWAP Reversion Strategy — intraday VWAP deviation mean-reversion.

Logic:
  - Calculate rolling VWAP with daily reset (every reset_bars candles)
  - Compute volume-weighted standard deviation bands
  - Long: price < VWAP - entry_sd*SD AND RSI(6) < rsi_long AND deviation > min_dev_pct
  - Short: price > VWAP + entry_sd*SD AND RSI(6) > rsi_short AND deviation > min_dev_pct
  - Exit: price crosses VWAP ± 0.3*SD (near mean reversion)
  - Stop: beyond stop_sd * SD
  - Regime: skip if intraday move > max_move_pct
  - Cooldown: min cooldown_bars between trades

Timeframe: 15m recommended (96 bars per day)
"""
import logging
import math
from collections import deque
from typing import Optional

import numpy as np

from .base import BaseStrategy, Signal

logger = logging.getLogger(__name__)


class VWAPReversionStrategy(BaseStrategy):

    def __init__(self, config):
        super().__init__(config)
        self.entry_sd = config.vwap_entry_sd             # 2.0
        self.stop_sd = config.vwap_stop_sd               # 3.0
        self.rsi_period = config.vwap_rsi_period          # 6
        self.rsi_long_thresh = config.vwap_rsi_long       # 25
        self.rsi_short_thresh = config.vwap_rsi_short     # 75
        self.reset_bars = config.vwap_reset_bars          # 96 (15m × 96 = 24h)
        self.max_intraday_pct = config.vwap_max_move_pct  # 4.0
        self.order_size_usd = config.vwap_order_size_usd  # 3000
        self.min_dev_pct = getattr(config, 'vwap_min_dev_pct', 0.3)   # min 0.3% from VWAP
        self.cooldown = getattr(config, 'vwap_cooldown', 8)           # 8 bars = 2h

        # VWAP state
        self._cum_vp = 0.0
        self._cum_vol = 0.0
        self._bar_in_session = 0
        self._session_open = None
        self._session_tp = []
        self._session_vol = []

        # RSI
        max_len = max(self.rsi_period + 2, 60)
        self.closes = deque(maxlen=max_len)

        # State
        self.current_position_side = None
        self.entry_price = 0.0
        self._vwap = None
        self._vwap_sd = None
        self._cooldown_remaining = 0

    def _reset_session(self, open_price):
        self._cum_vp = 0.0
        self._cum_vol = 0.0
        self._bar_in_session = 0
        self._session_open = open_price
        self._session_tp = []
        self._session_vol = []

    def _update_vwap(self, typical_price, volume):
        self._cum_vp += typical_price * volume
        self._cum_vol += volume
        self._bar_in_session += 1
        self._session_tp.append(typical_price)
        self._session_vol.append(volume)

        if self._cum_vol > 0:
            self._vwap = self._cum_vp / self._cum_vol
        else:
            self._vwap = typical_price

        if len(self._session_tp) >= 24:  # need at least 6h of data
            tp_arr = np.array(self._session_tp)
            vol_arr = np.array(self._session_vol)
            total_vol = vol_arr.sum()
            if total_vol > 0:
                variance = np.sum(vol_arr * (tp_arr - self._vwap) ** 2) / total_vol
                self._vwap_sd = math.sqrt(variance) if variance > 0 else None
            else:
                self._vwap_sd = None
        else:
            self._vwap_sd = None

    def _rsi(self) -> Optional[float]:
        n = self.rsi_period
        if len(self.closes) < n + 1:
            return None
        c = list(self.closes)
        gains, losses = 0.0, 0.0
        for i in range(-n, 0):
            diff = c[i] - c[i - 1]
            if diff > 0:
                gains += diff
            else:
                losses -= diff
        avg_gain = gains / n
        avg_loss = losses / n
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - 100 / (1 + rs)

    def on_kline(self, kline: dict) -> Optional[Signal]:
        if isinstance(kline, dict):
            o = float(kline.get("open", kline.get("o", 0)))
            close = float(kline.get("close", kline.get("c", 0)))
            high = float(kline.get("high", kline.get("h", 0)))
            low = float(kline.get("low", kline.get("l", 0)))
            volume = float(kline.get("volume", kline.get("v", 0)))
            confirm = kline.get("confirm", True)
        else:
            o = float(kline[1])
            close = float(kline[4])
            high = float(kline[2])
            low = float(kline[3])
            volume = float(kline[5])
            confirm = True

        if not confirm:
            return None

        self.closes.append(close)

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

        # Session reset
        if self._bar_in_session >= self.reset_bars or self._session_open is None:
            self._reset_session(o)

        typical = (high + low + close) / 3
        self._update_vwap(typical, max(volume, 1e-10))

        if self._vwap is None or self._vwap_sd is None:
            return None

        rsi = self._rsi()
        if rsi is None:
            return None

        vwap = self._vwap
        sd = self._vwap_sd

        # Regime filter
        if self._session_open and self._session_open > 0:
            intraday_move = abs(close - self._session_open) / self._session_open * 100
            if intraday_move > self.max_intraday_pct:
                if self.current_position_side:
                    side = "Sell" if self.current_position_side == "Buy" else "Buy"
                    self.current_position_side = None
                    self._cooldown_remaining = self.cooldown
                    return Signal(side=side, qty_usd=self.order_size_usd,
                                  reason="trailing_stop_vwap_regime")
                return None

        # Manage position
        if self.current_position_side == "Buy":
            # Exit near VWAP (TP) or at stop
            if close >= vwap - 0.3 * sd:
                self.current_position_side = None
                self._cooldown_remaining = self.cooldown
                return Signal(side="Sell", qty_usd=self.order_size_usd,
                              reason="trailing_stop_vwap_tp_long")
            if close <= vwap - self.stop_sd * sd:
                self.current_position_side = None
                self._cooldown_remaining = self.cooldown
                return Signal(side="Sell", qty_usd=self.order_size_usd,
                              reason="trailing_stop_vwap_sl_long")
            return None

        if self.current_position_side == "Sell":
            if close <= vwap + 0.3 * sd:
                self.current_position_side = None
                self._cooldown_remaining = self.cooldown
                return Signal(side="Buy", qty_usd=self.order_size_usd,
                              reason="trailing_stop_vwap_tp_short")
            if close >= vwap + self.stop_sd * sd:
                self.current_position_side = None
                self._cooldown_remaining = self.cooldown
                return Signal(side="Buy", qty_usd=self.order_size_usd,
                              reason="trailing_stop_vwap_sl_short")
            return None

        # No position — entries
        if self._cooldown_remaining > 0:
            return None

        dev_pct = abs(close - vwap) / vwap * 100
        if dev_pct < self.min_dev_pct:
            return None

        if close <= vwap - self.entry_sd * sd and rsi < self.rsi_long_thresh:
            return Signal(side="Buy", qty_usd=self.order_size_usd,
                          reason="vwap_long")

        if close >= vwap + self.entry_sd * sd and rsi > self.rsi_short_thresh:
            return Signal(side="Sell", qty_usd=self.order_size_usd,
                          reason="vwap_short")

        return None

    def on_position_update(self, position: dict):
        size = float(position.get("size", 0))
        if size == 0:
            self.current_position_side = None
            self.entry_price = 0.0
        else:
            self.current_position_side = position.get("side", "Buy")
            self.entry_price = float(position.get("avgPrice", 0))
