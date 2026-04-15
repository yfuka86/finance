"""Filtered Mean Reversion strategy.

Combines Bollinger Band + RSI + volume for high-probability mean reversion.
Only trades extreme oversold/overbought with volume confirmation.
Wide target (SMA), tight stop (beyond band).
Designed for 1H-4H with few trades.
"""
import logging
from collections import deque
from typing import Optional

import numpy as np

from .base import BaseStrategy, Signal

logger = logging.getLogger(__name__)


class MeanReversionFilteredStrategy(BaseStrategy):

    def __init__(self, config):
        super().__init__(config)
        self.bb_period = config.mrf_bb_period
        self.bb_mult = config.mrf_bb_mult
        self.rsi_period = config.mrf_rsi_period
        self.rsi_oversold = config.mrf_rsi_oversold
        self.rsi_overbought = config.mrf_rsi_overbought
        self.vol_mult = config.mrf_vol_mult
        self.atr_sl_mult = config.mrf_atr_sl_mult
        self.order_size_usd = config.mrf_order_size_usd

        max_len = max(self.bb_period, self.rsi_period) + 50
        self.closes = deque(maxlen=max_len)
        self.highs = deque(maxlen=max_len)
        self.lows = deque(maxlen=max_len)
        self.volumes = deque(maxlen=max_len)

        self.current_position_side = None
        self.entry_price = 0.0
        self.stop_loss = None
        self.target = None

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

    def _atr(self, period=14) -> Optional[float]:
        if len(self.closes) < period + 1:
            return None
        closes = list(self.closes)
        highs = list(self.highs)
        lows = list(self.lows)
        trs = []
        for i in range(-period, 0):
            tr = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i - 1]),
                     abs(lows[i] - closes[i - 1]))
            trs.append(tr)
        return np.mean(trs)

    def on_kline(self, kline: dict) -> Optional[Signal]:
        close = float(kline.get("close", kline.get("c", 0)))
        high = float(kline.get("high", kline.get("h", 0)))
        low = float(kline.get("low", kline.get("l", 0)))
        volume = float(kline.get("volume", kline.get("v", 0)))
        if not kline.get("confirm", True):
            return None

        self.closes.append(close)
        self.highs.append(high)
        self.lows.append(low)
        self.volumes.append(volume)

        if len(self.closes) < self.bb_period + 5:
            return None

        closes = list(self.closes)
        sma = np.mean(closes[-self.bb_period:])
        std = np.std(closes[-self.bb_period:], ddof=1)
        upper = sma + self.bb_mult * std
        lower = sma - self.bb_mult * std

        rsi = self._rsi()
        atr = self._atr()

        if rsi is None or atr is None or atr == 0:
            return None

        # Volume confirmation
        vol_list = list(self.volumes)
        avg_vol = np.mean(vol_list[-self.bb_period:])
        current_vol = volume
        vol_confirmed = current_vol > avg_vol * self.vol_mult

        # Exit: target hit or stop loss
        if self.current_position_side == "Buy" and self.target is not None:
            if close >= self.target:
                return Signal(side="Sell", qty_usd=self.order_size_usd,
                              reason="mrf_target_long")
            if self.stop_loss is not None and close <= self.stop_loss:
                return Signal(side="Sell", qty_usd=self.order_size_usd,
                              reason="mrf_stop_long")
        elif self.current_position_side == "Sell" and self.target is not None:
            if close <= self.target:
                return Signal(side="Buy", qty_usd=self.order_size_usd,
                              reason="mrf_target_short")
            if self.stop_loss is not None and close >= self.stop_loss:
                return Signal(side="Buy", qty_usd=self.order_size_usd,
                              reason="mrf_stop_short")

        if self.current_position_side is not None:
            return None

        # Entry: BB extreme + RSI extreme + volume spike
        if close < lower and rsi < self.rsi_oversold and vol_confirmed:
            self.target = sma  # Target: mean reversion to SMA
            self.stop_loss = close - atr * self.atr_sl_mult
            return Signal(side="Buy", qty_usd=self.order_size_usd,
                          reason=f"mrf_buy_rsi{rsi:.0f}")

        if close > upper and rsi > self.rsi_overbought and vol_confirmed:
            self.target = sma
            self.stop_loss = close + atr * self.atr_sl_mult
            return Signal(side="Sell", qty_usd=self.order_size_usd,
                          reason=f"mrf_sell_rsi{rsi:.0f}")

        return None

    def on_position_update(self, position: dict):
        size = float(position.get("size", 0))
        if size == 0:
            self.current_position_side = None
            self.entry_price = 0.0
            self.stop_loss = None
            self.target = None
        else:
            self.current_position_side = position.get("side", "")
            self.entry_price = float(position.get("avgPrice", 0))
