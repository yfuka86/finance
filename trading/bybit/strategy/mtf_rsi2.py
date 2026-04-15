"""Multi-Timeframe RSI(2) Strategy — 4H trend + 15m mean-reversion entry.

Logic:
  - Aggregate 15m candles → 4H bars internally (16 bars per 4H)
  - 4H Trend: EMA(20) > EMA(50) → bullish, EMA(20) < EMA(50) → bearish
  - 15m Entry (bullish): RSI(2) < rsi_entry_long (default 5)
  - 15m Entry (bearish): RSI(2) > rsi_entry_short (default 95)
  - Exit: RSI(2) crosses rsi_exit (default 60 for longs, 40 for shorts)
  - Stop: 15m ATR(14) × atr_sl_mult below/above entry
  - Cooldown: min cooldown_bars after each exit before next entry
  - Min hold: don't exit RSI signal for min_hold bars

Reference: Connors RSI(2) adapted for crypto with multi-TF trend filter.
"""
import logging
from collections import deque
from typing import Optional

import numpy as np

from .base import BaseStrategy, Signal

logger = logging.getLogger(__name__)


class MTFRsi2Strategy(BaseStrategy):

    def __init__(self, config):
        super().__init__(config)
        self.rsi_period = config.mtf_rsi_period            # 2
        self.rsi_entry_long = config.mtf_rsi_entry_long     # 5
        self.rsi_entry_short = config.mtf_rsi_entry_short   # 95
        self.rsi_exit_long = config.mtf_rsi_exit_long       # 60
        self.rsi_exit_short = config.mtf_rsi_exit_short     # 40
        self.atr_period = config.mtf_atr_period             # 14
        self.atr_sl_mult = config.mtf_atr_sl_mult           # 2.5
        self.atr_tp_mult = config.mtf_atr_tp_mult           # 4.0
        self.order_size_usd = config.mtf_order_size_usd     # 3000
        self.min_hold = getattr(config, 'mtf_min_hold', 8)  # min 8 bars before RSI exit
        self.cooldown = getattr(config, 'mtf_cooldown', 12) # 12 bars cooldown after exit
        self.trend_gap_pct = getattr(config, 'mtf_trend_gap_pct', 0.0)  # min EMA gap %
        self.dd_limit_pct = getattr(config, 'mtf_dd_limit_pct', 0.0)    # max DD % (0=disabled)
        self.vol_filter_mult = getattr(config, 'mtf_vol_filter', 0.0)   # vol expansion filter

        # 4H aggregation params
        self.htf_bars = config.mtf_htf_bars                 # 16 (15m×16=4H)
        self.htf_ema_fast = config.mtf_htf_ema_fast         # 20
        self.htf_ema_slow = config.mtf_htf_ema_slow         # 50

        # Data storage
        max_ltf = max(self.rsi_period + 2, self.atr_period + 2, 60)
        self.closes = deque(maxlen=max_ltf)
        self.highs = deque(maxlen=max_ltf)
        self.lows = deque(maxlen=max_ltf)

        # 4H aggregation
        self._htf_bar_count = 0
        max_htf = self.htf_ema_slow + 20
        self.htf_closes = deque(maxlen=max_htf)
        self._ema_fast = None
        self._ema_slow = None

        # State
        self.current_position_side = None
        self.entry_price = 0.0
        self.stop_loss = None
        self.take_profit = None
        self.trend = None
        self._bars_in_trade = 0
        self._cooldown_remaining = 0
        self._peak_equity = 0.0
        self._running_pnl = 0.0
        self._dd_paused = False
        # Vol filter: track ATR expansion
        self._recent_atrs = deque(maxlen=30)

    def _update_htf(self, close):
        self._htf_bar_count += 1
        if self._htf_bar_count >= self.htf_bars:
            self.htf_closes.append(close)
            self._htf_bar_count = 0

            k_fast = 2 / (self.htf_ema_fast + 1)
            k_slow = 2 / (self.htf_ema_slow + 1)
            n_htf = len(self.htf_closes)

            if self._ema_fast is None and n_htf >= self.htf_ema_fast:
                self._ema_fast = np.mean(list(self.htf_closes)[-self.htf_ema_fast:])
            elif self._ema_fast is not None:
                self._ema_fast = self._ema_fast * (1 - k_fast) + close * k_fast

            if self._ema_slow is None and n_htf >= self.htf_ema_slow:
                self._ema_slow = np.mean(list(self.htf_closes)[-self.htf_ema_slow:])
            elif self._ema_slow is not None:
                self._ema_slow = self._ema_slow * (1 - k_slow) + close * k_slow

            if self._ema_fast is not None and self._ema_slow is not None:
                if self._ema_fast > self._ema_slow:
                    self.trend = "bull"
                elif self._ema_fast < self._ema_slow:
                    self.trend = "bear"
                else:
                    self.trend = None

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

    def _atr(self) -> Optional[float]:
        if len(self.closes) < self.atr_period + 1:
            return None
        c, h, l_ = list(self.closes), list(self.highs), list(self.lows)
        trs = []
        for i in range(-self.atr_period, 0):
            tr = max(h[i] - l_[i], abs(h[i] - c[i - 1]), abs(l_[i] - c[i - 1]))
            trs.append(tr)
        return np.mean(trs)

    def on_kline(self, kline: dict) -> Optional[Signal]:
        if isinstance(kline, dict):
            close = float(kline.get("close", kline.get("c", 0)))
            high = float(kline.get("high", kline.get("h", 0)))
            low = float(kline.get("low", kline.get("l", 0)))
            confirm = kline.get("confirm", True)
        else:
            close = float(kline[4])
            high = float(kline[2])
            low = float(kline[3])
            confirm = True

        if not confirm:
            return None

        self.closes.append(close)
        self.highs.append(high)
        self.lows.append(low)
        self._update_htf(close)

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

        rsi = self._rsi()
        atr = self._atr()
        if rsi is None or atr is None or atr == 0:
            return None

        # Manage existing position
        if self.current_position_side:
            self._bars_in_trade += 1

            # Hard stop/TP always active
            if self.current_position_side == "Buy":
                if self.stop_loss and close <= self.stop_loss:
                    self._exit()
                    return Signal(side="Sell", qty_usd=self.order_size_usd,
                                  reason="trailing_stop_mtf_sl_long")
                if self.take_profit and close >= self.take_profit:
                    self._exit()
                    return Signal(side="Sell", qty_usd=self.order_size_usd,
                                  reason="trailing_stop_mtf_tp_long")
                # RSI exit only after min hold
                if self._bars_in_trade >= self.min_hold and rsi >= self.rsi_exit_long:
                    self._exit()
                    return Signal(side="Sell", qty_usd=self.order_size_usd,
                                  reason="trailing_stop_mtf_rsi_exit_long")

            elif self.current_position_side == "Sell":
                if self.stop_loss and close >= self.stop_loss:
                    self._exit()
                    return Signal(side="Buy", qty_usd=self.order_size_usd,
                                  reason="trailing_stop_mtf_sl_short")
                if self.take_profit and close <= self.take_profit:
                    self._exit()
                    return Signal(side="Buy", qty_usd=self.order_size_usd,
                                  reason="trailing_stop_mtf_tp_short")
                if self._bars_in_trade >= self.min_hold and rsi <= self.rsi_exit_short:
                    self._exit()
                    return Signal(side="Buy", qty_usd=self.order_size_usd,
                                  reason="trailing_stop_mtf_rsi_exit_short")
            return None

        # Track ATR for vol filter
        self._recent_atrs.append(atr)

        # No position — check cooldown
        if self._cooldown_remaining > 0:
            return None
        if self.trend is None:
            return None

        # Trend strength filter: skip if EMAs too close
        if self.trend_gap_pct > 0 and self._ema_fast and self._ema_slow:
            gap = abs(self._ema_fast - self._ema_slow) / self._ema_slow * 100
            if gap < self.trend_gap_pct:
                return None

        # Vol expansion filter: skip if current ATR >> median ATR
        if self.vol_filter_mult > 0 and len(self._recent_atrs) >= 20:
            median_atr = float(np.median(list(self._recent_atrs)[:-1]))
            if median_atr > 0 and atr / median_atr > self.vol_filter_mult:
                return None

        # DD protection: pause trading if cumulative DD exceeds limit
        if self.dd_limit_pct > 0:
            if self._running_pnl < -self.dd_limit_pct:
                self._dd_paused = True
            if self._dd_paused and self._running_pnl > -self.dd_limit_pct * 0.5:
                self._dd_paused = False  # resume after DD recovery
            if self._dd_paused:
                return None

        if self.trend == "bull" and rsi < self.rsi_entry_long:
            sl = close - atr * self.atr_sl_mult
            tp = close + atr * self.atr_tp_mult
            self._bars_in_trade = 0
            return Signal(side="Buy", qty_usd=self.order_size_usd,
                          stop_loss=sl, take_profit=tp,
                          reason="mtf_rsi2_long")

        if self.trend == "bear" and rsi > self.rsi_entry_short:
            sl = close + atr * self.atr_sl_mult
            tp = close - atr * self.atr_tp_mult
            self._bars_in_trade = 0
            return Signal(side="Sell", qty_usd=self.order_size_usd,
                          stop_loss=sl, take_profit=tp,
                          reason="mtf_rsi2_short")

        return None

    def _exit(self):
        self.current_position_side = None
        self.stop_loss = None
        self.take_profit = None
        self._cooldown_remaining = self.cooldown

    def on_position_update(self, position: dict):
        size = float(position.get("size", 0))
        if size == 0:
            # Track PnL for DD protection
            if self.entry_price > 0 and self.current_position_side:
                last_close = list(self.closes)[-1] if self.closes else self.entry_price
                if self.current_position_side == "Buy":
                    pnl_pct = (last_close - self.entry_price) / self.entry_price * 100
                else:
                    pnl_pct = (self.entry_price - last_close) / self.entry_price * 100
                self._running_pnl += pnl_pct
            self.current_position_side = None
            self.entry_price = 0.0
            self.stop_loss = None
            self.take_profit = None
        else:
            self.current_position_side = position.get("side", "Buy")
            self.entry_price = float(position.get("avgPrice", 0))
