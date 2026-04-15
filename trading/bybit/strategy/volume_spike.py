"""Volume Spike Reversal Strategy — 5m volume anomaly + pin bar reversal.

Logic:
  - Detect abnormal volume spikes (z-score > threshold over lookback)
  - Confirm with pin bar pattern (wick > wick_ratio of total range)
  - Additional: candle range must be > min_range_pct of close
  - Long: hammer (long lower wick) after volume spike
  - Short: shooting star (long upper wick) after volume spike
  - Stop: beyond wick extreme + buffer
  - Exit: 2× wick length as TP, or time-based exit (max_hold candles)
  - Cooldown: no re-entry for cooldown_bars after exit

Timeframe: 5m recommended
"""
import logging
from collections import deque
from typing import Optional

import numpy as np

from .base import BaseStrategy, Signal

logger = logging.getLogger(__name__)


class VolumeSpikeStrategy(BaseStrategy):

    def __init__(self, config):
        super().__init__(config)
        self.vol_lookback = config.vs_vol_lookback         # 200
        self.vol_zscore_thresh = config.vs_vol_zscore      # 4.0
        self.wick_ratio = config.vs_wick_ratio             # 0.65
        self.min_range_pct = getattr(config, 'vs_min_range_pct', 0.15)  # 0.15%
        self.max_hold = config.vs_max_hold                 # 12 bars (60min on 5m)
        self.cooldown_bars = config.vs_cooldown            # 12
        self.stop_buffer_pct = config.vs_stop_buffer_pct   # 0.1
        self.tp_wick_mult = config.vs_tp_wick_mult         # 2.0
        self.order_size_usd = config.vs_order_size_usd     # 2000

        # Data
        self.volumes = deque(maxlen=self.vol_lookback + 10)

        # State
        self.current_position_side = None
        self.entry_price = 0.0
        self.stop_loss = None
        self.take_profit = None
        self._bars_in_trade = 0
        self._cooldown_remaining = 0

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

        self.volumes.append(volume)

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

        # Manage existing position
        if self.current_position_side:
            self._bars_in_trade += 1

            if self.current_position_side == "Buy":
                if self.stop_loss and close <= self.stop_loss:
                    self._exit()
                    return Signal(side="Sell", qty_usd=self.order_size_usd,
                                  reason="trailing_stop_vs_sl_long")
                if self.take_profit and close >= self.take_profit:
                    self._exit()
                    return Signal(side="Sell", qty_usd=self.order_size_usd,
                                  reason="trailing_stop_vs_tp_long")
                if self._bars_in_trade >= self.max_hold:
                    self._exit()
                    return Signal(side="Sell", qty_usd=self.order_size_usd,
                                  reason="trailing_stop_vs_timeout_long")

            elif self.current_position_side == "Sell":
                if self.stop_loss and close >= self.stop_loss:
                    self._exit()
                    return Signal(side="Buy", qty_usd=self.order_size_usd,
                                  reason="trailing_stop_vs_sl_short")
                if self.take_profit and close <= self.take_profit:
                    self._exit()
                    return Signal(side="Buy", qty_usd=self.order_size_usd,
                                  reason="trailing_stop_vs_tp_short")
                if self._bars_in_trade >= self.max_hold:
                    self._exit()
                    return Signal(side="Buy", qty_usd=self.order_size_usd,
                                  reason="trailing_stop_vs_timeout_short")
            return None

        # No position
        if self._cooldown_remaining > 0:
            return None
        if len(self.volumes) < self.vol_lookback:
            return None

        # Volume z-score (exclude current bar)
        vols = list(self.volumes)
        past_vols = np.array(vols[-(self.vol_lookback + 1):-1])
        vol_mean = past_vols.mean()
        vol_std = past_vols.std()
        if vol_std == 0 or vol_mean == 0:
            return None
        zscore = (volume - vol_mean) / vol_std
        if zscore < self.vol_zscore_thresh:
            return None

        # Candle range filter
        candle_range = high - low
        if candle_range == 0 or close == 0:
            return None
        range_pct = candle_range / close * 100
        if range_pct < self.min_range_pct:
            return None

        body_top = max(o, close)
        body_bot = min(o, close)
        upper_wick = high - body_top
        lower_wick = body_bot - low

        # Hammer → long
        if lower_wick / candle_range >= self.wick_ratio:
            stop = low * (1 - self.stop_buffer_pct / 100)
            tp = close + lower_wick * self.tp_wick_mult
            self._bars_in_trade = 0
            return Signal(side="Buy", qty_usd=self.order_size_usd,
                          stop_loss=stop, take_profit=tp,
                          reason="vs_hammer_long")

        # Shooting star → short
        if upper_wick / candle_range >= self.wick_ratio:
            stop = high * (1 + self.stop_buffer_pct / 100)
            tp = close - upper_wick * self.tp_wick_mult
            self._bars_in_trade = 0
            return Signal(side="Sell", qty_usd=self.order_size_usd,
                          stop_loss=stop, take_profit=tp,
                          reason="vs_star_short")

        return None

    def _exit(self):
        self.current_position_side = None
        self.stop_loss = None
        self.take_profit = None
        self._cooldown_remaining = self.cooldown_bars

    def on_position_update(self, position: dict):
        size = float(position.get("size", 0))
        if size == 0:
            self.current_position_side = None
            self.entry_price = 0.0
            self.stop_loss = None
            self.take_profit = None
        else:
            self.current_position_side = position.get("side", "Buy")
            self.entry_price = float(position.get("avgPrice", 0))
