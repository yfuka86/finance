"""Risk management module with kill switches and position limits."""
import logging
import time
from dataclasses import dataclass, field

from .config import BybitConfig

logger = logging.getLogger(__name__)


@dataclass
class RiskState:
    """Tracks risk metrics in real-time."""
    equity_start: float = 0.0         # Equity at bot start
    equity_high: float = 0.0          # High-water mark
    equity_current: float = 0.0       # Current equity
    daily_pnl: float = 0.0            # Realized P&L today
    unrealized_pnl: float = 0.0       # Unrealized P&L
    position_usd: float = 0.0         # Current position size in USD
    num_trades_today: int = 0
    trading_enabled: bool = True
    kill_reason: str = ""
    last_check_time: float = 0.0


class RiskManager:
    """Enforces risk limits and triggers kill switches."""

    def __init__(self, config: BybitConfig):
        self.config = config
        self.state = RiskState()

    def initialize(self, equity: float):
        self.state.equity_start = equity
        self.state.equity_high = equity
        self.state.equity_current = equity
        self.state.trading_enabled = True
        self.state.kill_reason = ""
        logger.info(f"Risk manager initialized: equity={equity:.2f}")

    def check(self, equity: float, unrealized_pnl: float, position_usd: float) -> bool:
        """Run all risk checks. Returns True if trading is allowed."""
        self.state.equity_current = equity
        self.state.unrealized_pnl = unrealized_pnl
        self.state.position_usd = abs(position_usd)
        self.state.last_check_time = time.time()

        if equity > self.state.equity_high:
            self.state.equity_high = equity

        # Check max drawdown from high-water mark
        if self.state.equity_high > 0:
            dd_pct = (self.state.equity_high - equity) / self.state.equity_high * 100
            if dd_pct >= self.config.max_drawdown_pct:
                self._kill(f"Max drawdown reached: {dd_pct:.2f}% >= {self.config.max_drawdown_pct}%")
                return False

        # Check daily loss
        daily_pnl = equity - self.state.equity_start + unrealized_pnl
        self.state.daily_pnl = daily_pnl
        if daily_pnl <= -self.config.max_daily_loss_usd:
            self._kill(f"Daily loss limit: {daily_pnl:.2f} <= -{self.config.max_daily_loss_usd}")
            return False

        # Check position size
        if abs(position_usd) > self.config.max_position_usd:
            logger.warning(
                f"Position size {abs(position_usd):.2f} exceeds limit {self.config.max_position_usd}"
            )
            # Don't kill, just warn (engine should reduce position)

        return self.state.trading_enabled

    def can_open_position(self, side: str, qty_usd: float) -> bool:
        """Check if a new order is allowed."""
        if not self.state.trading_enabled:
            return False

        new_position = self.state.position_usd + qty_usd
        if new_position > self.config.max_position_usd:
            logger.warning(
                f"Order rejected: would exceed max position "
                f"({new_position:.2f} > {self.config.max_position_usd})"
            )
            return False

        return True

    def _kill(self, reason: str):
        """Trigger kill switch - disable all trading."""
        self.state.trading_enabled = False
        self.state.kill_reason = reason
        logger.critical(f"KILL SWITCH: {reason}")

    def reset_daily(self):
        """Reset daily counters (call at start of new trading day)."""
        self.state.equity_start = self.state.equity_current
        self.state.daily_pnl = 0.0
        self.state.num_trades_today = 0
        if not self.state.trading_enabled and "daily" in self.state.kill_reason.lower():
            self.state.trading_enabled = True
            self.state.kill_reason = ""
            logger.info("Daily risk limits reset, trading re-enabled")

    def status(self) -> dict:
        s = self.state
        dd_pct = 0
        if s.equity_high > 0:
            dd_pct = (s.equity_high - s.equity_current) / s.equity_high * 100
        return {
            "trading_enabled": s.trading_enabled,
            "equity": s.equity_current,
            "daily_pnl": s.daily_pnl,
            "drawdown_pct": dd_pct,
            "position_usd": s.position_usd,
            "kill_reason": s.kill_reason,
        }
