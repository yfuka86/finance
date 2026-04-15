"""Dual Regime Strategy — switches between trend-following and mean-reversion.

Logic:
  - ADX(14) on current timeframe determines regime (past data only)
  - ADX > threshold → Donchian breakout (trend-following)
  - ADX <= threshold → Bollinger Band reversion (mean-reversion)
  - Additional filter: vol expansion (RV5/RV30 > 1.5) → stay flat

This addresses the critical weakness found in 3-year backtests:
single-strategy approaches fail when regime changes (e.g., 2024 Jan -50~-73%).

No training data / look-ahead: ADX and vol ratio are computed
from past bars only, updated each bar.
"""
import logging
import math
from collections import deque
from typing import Optional

import numpy as np

from .base import BaseStrategy, Signal

logger = logging.getLogger(__name__)


class DualRegimeStrategy(BaseStrategy):
    """
    Regime-adaptive strategy.

    Trending (ADX > threshold): Donchian breakout entries
    Ranging  (ADX <= threshold): Bollinger Band mean-reversion entries
    Vol expansion: Stay flat to avoid whipsaws
    """

    def __init__(self, config):
        super().__init__(config)
        # Regime detection
        self.adx_period = config.dr_adx_period          # 14
        self.adx_threshold = config.dr_adx_threshold    # 25
        self.vol_ratio_limit = config.dr_vol_ratio_limit  # 1.5

        # Donchian (trend mode)
        self.don_entry = config.dr_don_entry_period      # 20
        self.don_exit = config.dr_don_exit_period        # 10

        # Bollinger (reversion mode)
        self.bb_period = config.dr_bb_period             # 20
        self.bb_std = config.dr_bb_std_mult              # 2.0

        # Shared
        self.atr_period = config.dr_atr_period           # 14
        self.atr_sl_mult = config.dr_atr_sl_mult         # 3.0
        self.order_size_usd = config.dr_order_size_usd   # 5000

        max_len = max(self.don_entry, self.bb_period, self.adx_period * 3,
                      self.atr_period, 30) + 10
        self.closes = deque(maxlen=max_len)
        self.highs = deque(maxlen=max_len)
        self.lows = deque(maxlen=max_len)

        # State
        self.current_position_side = None
        self.entry_price = 0.0
        self.trailing_stop = None
        self.regime = None  # "trend" or "range" or "flat"

    # ── Indicators (all use past data only) ──────────────────────

    def _adx(self) -> Optional[float]:
        """ADX(period) from past closes/highs/lows. No look-ahead."""
        n = self.adx_period
        if len(self.highs) < n * 2 + 1:
            return None
        h = list(self.highs)
        l_ = list(self.lows)
        c = list(self.closes)

        # +DM, -DM, TR
        plus_dm, minus_dm, tr_list = [], [], []
        for i in range(-n * 2, 0):
            dm_plus = h[i] - h[i - 1]
            dm_minus = l_[i - 1] - l_[i]
            plus_dm.append(max(dm_plus, 0) if dm_plus > dm_minus else 0)
            minus_dm.append(max(dm_minus, 0) if dm_minus > dm_plus else 0)
            tr = max(h[i] - l_[i], abs(h[i] - c[i - 1]), abs(l_[i] - c[i - 1]))
            tr_list.append(tr)

        # Smoothed averages (Wilder's method)
        atr = sum(tr_list[:n]) / n
        plus_di_s = sum(plus_dm[:n]) / n
        minus_di_s = sum(minus_dm[:n]) / n

        for i in range(n, len(tr_list)):
            atr = (atr * (n - 1) + tr_list[i]) / n
            plus_di_s = (plus_di_s * (n - 1) + plus_dm[i]) / n
            minus_di_s = (minus_di_s * (n - 1) + minus_dm[i]) / n

        if atr == 0:
            return None
        plus_di = 100 * plus_di_s / atr
        minus_di = 100 * minus_di_s / atr
        dx_denom = plus_di + minus_di
        if dx_denom == 0:
            return 0.0
        dx = 100 * abs(plus_di - minus_di) / dx_denom
        return dx

    def _vol_ratio(self) -> float:
        """RV(5) / RV(30). Both use only past data."""
        c = list(self.closes)
        if len(c) < 31:
            return 1.0
        rets = [math.log(c[i] / c[i - 1]) for i in range(-30, 0)]
        rv30 = np.std(rets) if len(rets) >= 30 else 1.0
        rv5 = np.std(rets[-5:]) if len(rets) >= 5 else rv30
        return rv5 / rv30 if rv30 > 0 else 1.0

    def _atr(self) -> Optional[float]:
        if len(self.closes) < self.atr_period + 1:
            return None
        c, h, l_ = list(self.closes), list(self.highs), list(self.lows)
        trs = []
        for i in range(-self.atr_period, 0):
            tr = max(h[i] - l_[i], abs(h[i] - c[i - 1]), abs(l_[i] - c[i - 1]))
            trs.append(tr)
        return np.mean(trs)

    def _donchian_high(self) -> float:
        h = list(self.highs)
        return max(h[-self.don_entry:])

    def _donchian_low(self) -> float:
        l_ = list(self.lows)
        return min(l_[-self.don_entry:])

    def _donchian_exit_high(self) -> float:
        h = list(self.highs)
        return max(h[-self.don_exit:])

    def _donchian_exit_low(self) -> float:
        l_ = list(self.lows)
        return min(l_[-self.don_exit:])

    def _bb(self):
        """Returns (upper, mid, lower) Bollinger Bands."""
        c = list(self.closes)
        if len(c) < self.bb_period:
            return None, None, None
        window = c[-self.bb_period:]
        mid = np.mean(window)
        std = np.std(window)
        return mid + self.bb_std * std, mid, mid - self.bb_std * std

    # ── Main logic ───────────────────────────────────────────────

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

        if len(self.closes) < max(self.don_entry, self.bb_period,
                                   self.adx_period * 2 + 1, 31):
            return None

        adx = self._adx()
        if adx is None:
            return None

        vr = self._vol_ratio()
        atr = self._atr()
        if atr is None or atr == 0:
            return None

        # Determine regime
        if vr > self.vol_ratio_limit:
            self.regime = "flat"
        elif adx > self.adx_threshold:
            self.regime = "trend"
        else:
            self.regime = "range"

        # ── Check trailing stop first ────────────────────────────
        if self.current_position_side and self.trailing_stop:
            if self.current_position_side == "Buy" and close <= self.trailing_stop:
                self.current_position_side = None
                self.trailing_stop = None
                return Signal(side="Sell", qty_usd=self.order_size_usd,
                              reason=f"dr_trailing_stop_{self.regime}")
            elif self.current_position_side == "Sell" and close >= self.trailing_stop:
                self.current_position_side = None
                self.trailing_stop = None
                return Signal(side="Buy", qty_usd=self.order_size_usd,
                              reason=f"dr_trailing_stop_{self.regime}")

        # Update trailing stop
        if self.current_position_side == "Buy":
            ns = close - atr * self.atr_sl_mult
            if self.trailing_stop is None or ns > self.trailing_stop:
                self.trailing_stop = ns
        elif self.current_position_side == "Sell":
            ns = close + atr * self.atr_sl_mult
            if self.trailing_stop is None or ns < self.trailing_stop:
                self.trailing_stop = ns

        # ── Vol expansion → close positions, no new trades ───────
        if self.regime == "flat":
            if self.current_position_side:
                side = "Sell" if self.current_position_side == "Buy" else "Buy"
                self.current_position_side = None
                self.trailing_stop = None
                return Signal(side=side, qty_usd=self.order_size_usd,
                              reason="dr_vol_expansion_exit")
            return None

        # ── Trend regime: Donchian breakout ──────────────────────
        if self.regime == "trend":
            don_hi = self._donchian_high()
            don_lo = self._donchian_low()

            if self.current_position_side is None:
                if close >= don_hi:
                    sl = close - atr * self.atr_sl_mult
                    return Signal(side="Buy", qty_usd=self.order_size_usd,
                                  stop_loss=sl,
                                  reason="dr_trend_don_long")
                elif close <= don_lo:
                    sl = close + atr * self.atr_sl_mult
                    return Signal(side="Sell", qty_usd=self.order_size_usd,
                                  stop_loss=sl,
                                  reason="dr_trend_don_short")
            elif self.current_position_side == "Buy":
                exit_lo = self._donchian_exit_low()
                if close <= exit_lo:
                    self.current_position_side = None
                    self.trailing_stop = None
                    return Signal(side="Sell", qty_usd=self.order_size_usd,
                                  reason="dr_trend_don_exit_long")
            elif self.current_position_side == "Sell":
                exit_hi = self._donchian_exit_high()
                if close >= exit_hi:
                    self.current_position_side = None
                    self.trailing_stop = None
                    return Signal(side="Buy", qty_usd=self.order_size_usd,
                                  reason="dr_trend_don_exit_short")
            return None

        # ── Range regime: Bollinger Band reversion ───────────────
        if self.regime == "range":
            bb_upper, bb_mid, bb_lower = self._bb()
            if bb_upper is None:
                return None

            if self.current_position_side is None:
                if close <= bb_lower:
                    return Signal(side="Buy", qty_usd=self.order_size_usd,
                                  reason="dr_range_bb_long")
                elif close >= bb_upper:
                    return Signal(side="Sell", qty_usd=self.order_size_usd,
                                  reason="dr_range_bb_short")
            elif self.current_position_side == "Buy":
                if close >= bb_mid:
                    self.current_position_side = None
                    self.trailing_stop = None
                    return Signal(side="Sell", qty_usd=self.order_size_usd,
                                  reason="dr_range_bb_exit_long")
            elif self.current_position_side == "Sell":
                if close <= bb_mid:
                    self.current_position_side = None
                    self.trailing_stop = None
                    return Signal(side="Buy", qty_usd=self.order_size_usd,
                                  reason="dr_range_bb_exit_short")
            return None

        return None

    def on_position_update(self, position: dict):
        size = float(position.get("size", 0))
        if size == 0:
            self.current_position_side = None
            self.trailing_stop = None
            self.entry_price = 0.0
        else:
            self.current_position_side = position.get("side", "Buy")
            self.entry_price = float(position.get("avgPrice", 0))
