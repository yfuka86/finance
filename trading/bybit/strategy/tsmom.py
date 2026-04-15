"""Vol-Targeted Time-Series Momentum (TSMOM) strategy.

References:
- Moskowitz, Ooi, Pedersen (2012) "Time Series Momentum"
- Bianchi & Babiak (2022) "Cryptocurrencies and Momentum"
- SSRN 4825389: Cryptocurrency Volume-Weighted TSMOM (Sharpe 2.17)

Logic:
  - Signal = sign of blended lookback return (14d + 28d average)
  - Position size = target_vol / realized_vol (vol-targeting)
  - Long-only: positive signal → full position, negative → cash
  - No training data needed: purely rule-based with past returns only
"""
import logging
import math
from collections import deque
from typing import Optional

import numpy as np

from .base import BaseStrategy, Signal

logger = logging.getLogger(__name__)


class TSMOMStrategy(BaseStrategy):
    """
    Vol-targeted time-series momentum.

    Entry: Blended lookback return > 0 → long.
    Exit: Blended lookback return <= 0 → close (go to cash).
    Sizing: target_vol / realized_vol, capped at max_leverage.
    """

    def __init__(self, config):
        super().__init__(config)
        self.lookback_short = config.tsmom_lookback_short   # 14 bars
        self.lookback_long = config.tsmom_lookback_long     # 28 bars
        self.vol_window = config.tsmom_vol_window           # 30 bars
        self.vol_target = config.tsmom_vol_target           # annualized target
        self.max_leverage = config.tsmom_max_leverage       # 1.5
        self.order_size_usd = config.tsmom_order_size_usd   # base size

        max_len = max(self.lookback_long, self.vol_window) + 10
        self.closes = deque(maxlen=max_len)

        # State
        self.current_position_side = None  # "Buy" or None
        self.entry_price = 0.0
        self.current_sizing = 0.0  # current vol-adjusted multiplier

    def _log_returns(self):
        """Compute log returns from close prices."""
        c = list(self.closes)
        return [math.log(c[i] / c[i - 1]) for i in range(1, len(c))]

    def _signal(self) -> float:
        """Blended lookback return signal. >0 = long, <=0 = cash."""
        c = list(self.closes)
        n = len(c)
        if n < self.lookback_long + 1:
            return 0.0
        # Return over short lookback
        ret_short = (c[-1] / c[-1 - self.lookback_short]) - 1
        # Return over long lookback
        ret_long = (c[-1] / c[-1 - self.lookback_long]) - 1
        # Average (blend)
        return (ret_short + ret_long) / 2.0

    def _realized_vol(self) -> float:
        """Annualized realized vol from recent log returns."""
        rets = self._log_returns()
        if len(rets) < self.vol_window:
            return 0.0
        recent = rets[-self.vol_window:]
        std = np.std(recent)
        if std == 0:
            return 0.0
        # Annualize: depends on bar frequency, approximate with sqrt(365*bars_per_day)
        # For generality, use sqrt(252) as proxy (adjusted in config via vol_target)
        return std * math.sqrt(365)

    def _vol_sizing(self) -> float:
        """Position size multiplier based on vol targeting."""
        rv = self._realized_vol()
        if rv <= 0:
            return 1.0
        raw = self.vol_target / rv
        return min(raw, self.max_leverage)

    def on_kline(self, kline: dict) -> Optional[Signal]:
        if isinstance(kline, dict):
            close = float(kline.get("close", kline.get("c", 0)))
            confirm = kline.get("confirm", True)
        else:
            close = float(kline[4])
            confirm = True

        if not confirm:
            return None

        self.closes.append(close)

        # Need enough history
        if len(self.closes) < self.lookback_long + 2:
            return None

        sig = self._signal()
        sizing = self._vol_sizing()

        want_long = sig > 0

        if want_long and self.current_position_side is None:
            # Enter long
            qty = self.order_size_usd * sizing
            logger.info(f"TSMOM entry LONG: sig={sig:.4f} sizing={sizing:.2f} qty=${qty:.0f}")
            return Signal(
                side="Buy",
                qty_usd=qty,
                reason=f"tsmom_long_sig={sig:.4f}_vol={sizing:.2f}",
            )

        elif not want_long and self.current_position_side == "Buy":
            # Exit: signal turned negative
            logger.info(f"TSMOM exit: sig={sig:.4f}")
            return Signal(
                side="Sell",
                qty_usd=self.order_size_usd * self.current_sizing,
                reason=f"tsmom_exit_sig={sig:.4f}",
            )

        elif want_long and self.current_position_side == "Buy":
            # Rebalance check: sizing changed significantly (>20%)
            if self.current_sizing > 0:
                ratio = sizing / self.current_sizing
                if abs(ratio - 1.0) > 0.20:
                    diff_usd = self.order_size_usd * (sizing - self.current_sizing)
                    if diff_usd > 0:
                        logger.info(f"TSMOM rebalance up: {diff_usd:.0f}")
                        return Signal(side="Buy", qty_usd=abs(diff_usd),
                                      reason="tsmom_rebal_up")
                    else:
                        logger.info(f"TSMOM rebalance down: {diff_usd:.0f}")
                        return Signal(side="Sell", qty_usd=abs(diff_usd),
                                      reason="tsmom_rebal_down")

        # Update current sizing for rebalance tracking
        if want_long:
            self.current_sizing = sizing

        return None

    def on_position_update(self, position: dict):
        size = float(position.get("size", 0))
        if size == 0:
            self.current_position_side = None
            self.entry_price = 0.0
            self.current_sizing = 0.0
        else:
            self.current_position_side = position.get("side", "Buy")
            self.entry_price = float(position.get("avgPrice", 0))
