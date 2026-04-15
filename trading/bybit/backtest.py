"""
Backtesting engine for Bybit strategies.

Uses the same strategy classes as live trading (backtest-live parity).
Fetches historical klines from Bybit public API (no auth needed).
月単位でキャッシュし、同じデータの再取得を避ける。
"""
import calendar
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from pybit.unified_trading import HTTP

from .config import BybitConfig
from .data_cache import has_month, load_month, save_month, load_months
from .strategy import STRATEGIES
from .strategy.base import BaseStrategy, Signal
from .strategy.grid import GridStrategy

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Record of a single trade."""
    timestamp: pd.Timestamp
    side: str
    price: float
    qty: float
    qty_usd: float
    fee: float
    reason: str
    pnl: float = 0.0  # Realized PnL for closing trades


@dataclass
class BacktestResult:
    """Complete backtest output."""
    config: dict
    equity_curve: pd.Series
    trades: list[Trade]
    daily_returns: pd.Series
    metrics: dict
    klines_df: pd.DataFrame


# ── Fee model ────────────────────────────────────────────────────

MAKER_FEE = 0.0002   # 0.02%
TAKER_FEE = 0.00055  # 0.055%


def _calc_fee(qty_usd: float, is_maker: bool) -> float:
    return qty_usd * (MAKER_FEE if is_maker else TAKER_FEE)


# ── Data fetching (月単位キャッシュ付き) ─────────────────────────

def _fetch_raw_klines(http: HTTP, symbol: str, interval: str,
                      start_ms: int, end_ms: int,
                      category: str = "linear", limit: int = 1000,
                      on_progress: Optional[callable] = None,
                      base_count: int = 0) -> list:
    """Bybit API から生の kline データをページネーション付きで取得する。"""
    all_klines = []
    cursor_end = end_ms
    max_retries = 5

    while True:
        params = dict(category=category, symbol=symbol, interval=interval, limit=limit)
        params["start"] = start_ms
        if cursor_end:
            params["end"] = cursor_end

        klines = None
        for attempt in range(max_retries):
            try:
                resp = http.get_kline(**params)
                klines = resp["result"]["list"]
                break
            except Exception as e:
                err_str = str(e).lower()
                if "rate limit" in err_str or "x-bapi-limit" in err_str or "403" in err_str or "10006" in err_str:
                    wait = 2 ** attempt
                    logger.warning(f"Rate limit hit, waiting {wait}s (attempt {attempt+1}/{max_retries})")
                    if on_progress:
                        on_progress(base_count + len(all_klines), None,
                                    f"レート制限... {wait}秒待機中 ({attempt+1}/{max_retries})")
                    time.sleep(wait)
                else:
                    raise

        if klines is None:
            raise RuntimeError(f"APIレート制限超過: {max_retries}回リトライ後も失敗")

        if not klines:
            break

        all_klines.extend(klines)

        oldest_ts = int(klines[-1][0])
        if oldest_ts <= start_ms:
            break
        cursor_end = oldest_ts - 1
        time.sleep(0.25)
        if len(klines) < limit:
            break

    return all_klines


def _raw_to_df(raw_klines: list) -> pd.DataFrame:
    """生の kline リストを DataFrame に変換する。"""
    df = pd.DataFrame(raw_klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume", "turnover"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        df[col] = df[col].astype(float)
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    df = df.set_index("timestamp")
    return df


def _month_range(start_year: int, start_month: int,
                 end_year: int, end_month: int) -> list[tuple[int, int]]:
    """(year, month) のリストを返す。"""
    months = []
    y, m = start_year, start_month
    while (y, m) <= (end_year, end_month):
        months.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def fetch_klines(symbol: str = "BTCUSDT", interval: str = "1",
                 start: Optional[str] = None, end: Optional[str] = None,
                 category: str = "linear", limit: int = 1000,
                 on_progress: Optional[callable] = None) -> pd.DataFrame:
    """
    月単位キャッシュ付きで過去データを取得する。

    キャッシュ済みの月はスキップし、未取得の月のみ API から取得して保存する。
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    sy, sm = start_ts.year, start_ts.month
    ey, em = end_ts.year, end_ts.month

    all_months = _month_range(sy, sm, ey, em)
    total_months = len(all_months)

    # キャッシュ確認
    cached_frames = []
    missing_months = []
    for y, m in all_months:
        df = load_month(symbol, interval, y, m)
        if df is not None:
            cached_frames.append(df)
        else:
            missing_months.append((y, m))

    if on_progress:
        cached_count = sum(len(f) for f in cached_frames)
        on_progress(cached_count, None,
                    f"キャッシュ確認: {total_months - len(missing_months)}/{total_months}ヶ月取得済み")

    # 未取得の月を API から取得
    http = HTTP() if missing_months else None
    fetched_total = sum(len(f) for f in cached_frames)

    for i, (y, m) in enumerate(missing_months):
        last_day = calendar.monthrange(y, m)[1]
        m_start_ms = int(pd.Timestamp(f"{y:04d}-{m:02d}-01").timestamp() * 1000)
        m_end_ms = int(pd.Timestamp(f"{y:04d}-{m:02d}-{last_day} 23:59:59").timestamp() * 1000)

        if on_progress:
            pct = (total_months - len(missing_months) + i) / total_months * 0.95
            on_progress(fetched_total, pct,
                        f"データ取得中... {y:04d}-{m:02d} "
                        f"({total_months - len(missing_months) + i + 1}/{total_months}ヶ月)")

        raw = _fetch_raw_klines(
            http, symbol, interval, m_start_ms, m_end_ms,
            category=category, limit=limit,
            on_progress=on_progress, base_count=fetched_total,
        )

        if raw:
            df_month = _raw_to_df(raw)
            save_month(symbol, interval, y, m, df_month)
            cached_frames.append(df_month)
            fetched_total += len(df_month)

    if not cached_frames:
        raise ValueError(f"No klines fetched for {symbol} {interval}")

    # 全月を結合
    df = pd.concat(cached_frames).sort_index()
    df = df[~df.index.duplicated(keep="first")]

    # 指定範囲でフィルタ
    df = df[start_ts:end_ts]

    if on_progress:
        on_progress(len(df), 1.0, f"データ取得完了: {len(df):,}本 ({total_months}ヶ月)")

    logger.info(f"Loaded {len(df)} klines for {symbol} ({df.index[0]} to {df.index[-1]})")
    return df


# ── Backtest engine ──────────────────────────────────────────────

class Backtester:
    """
    Simulates strategy execution on historical data.

    - Feeds klines to strategy one-by-one (event-driven, same as live)
    - Tracks position, equity, and fills
    - Applies realistic fee model (maker/taker)
    - Supports slippage simulation
    """

    def __init__(self, config: BybitConfig, slippage_bps: float = 1.0):
        self.config = config
        self.slippage_bps = slippage_bps

        strategy_cls = STRATEGIES.get(config.strategy)
        if not strategy_cls:
            raise ValueError(f"Unknown strategy: {config.strategy}")
        self.strategy: BaseStrategy = strategy_cls(config)

        # Simulation state
        self.equity = 10000.0  # Starting capital
        self.cash = self.equity
        self.position_qty = 0.0   # Positive = long, negative = short
        self.position_side = None  # "Buy" or "Sell"
        self.entry_price = 0.0
        self.trades: list[Trade] = []
        self.equity_history: list[tuple] = []  # (timestamp, equity)

    def run(self, klines_df: pd.DataFrame, initial_equity: float = 10000.0,
            on_progress: Optional[callable] = None) -> BacktestResult:
        """
        Run backtest on historical klines.

        Args:
            klines_df: DataFrame from fetch_klines()
            initial_equity: Starting capital in USDT
            on_progress: Optional callback(processed, total, pct, message) for progress
        """
        self.equity = initial_equity
        self.cash = initial_equity
        self.position_qty = 0.0
        self.position_side = None
        self.entry_price = 0.0
        self.trades = []
        self.equity_history = []

        total = len(klines_df)
        report_interval = max(total // 100, 1)  # Report ~100 times

        for idx, (ts, row) in enumerate(klines_df.iterrows()):
            if on_progress and idx % report_interval == 0:
                pct = idx / total
                pnl = self.equity - initial_equity
                sign = "+" if pnl >= 0 else ""
                on_progress(
                    idx, total, pct,
                    f"シミュレーション中... {idx:,}/{total:,}本 "
                    f"| 損益: {sign}${pnl:,.2f} | 取引: {len(self.trades)}回"
                )
            kline = {
                "open": str(row["open"]),
                "high": str(row["high"]),
                "low": str(row["low"]),
                "close": str(row["close"]),
                "volume": str(row["volume"]),
                "confirm": True,
            }

            current_price = row["close"]

            # Update equity with mark-to-market
            unrealized = self.position_qty * (current_price - self.entry_price) if self.position_qty != 0 else 0
            self.equity = self.cash + unrealized
            self.equity_history.append((ts, self.equity))

            # Feed to strategy
            signal = self.strategy.on_kline(kline)

            # For grid strategy: initialize on first candle
            if isinstance(self.strategy, GridStrategy) and not self.strategy.grid_initialized:
                self.strategy.initialize_grid(current_price)
                grid_signals = self.strategy.get_initial_orders(current_price)
                for gs in grid_signals:
                    self._execute_grid_signal(gs, current_price, ts)
                continue

            if signal:
                self._execute_signal(signal, current_price, ts)

            # For grid: check if any pending grid orders would fill
            if isinstance(self.strategy, GridStrategy):
                self._check_grid_fills(row, ts)

        # Close any remaining position at last price
        if self.position_qty != 0:
            last_price = klines_df.iloc[-1]["close"]
            last_ts = klines_df.index[-1]
            self._close_position(last_price, last_ts, "backtest_end")

        # Build results
        equity_series = pd.Series(
            {ts: eq for ts, eq in self.equity_history},
            name="equity"
        )

        daily_equity = equity_series.resample("D").last().dropna()
        daily_returns = daily_equity.pct_change().dropna()

        metrics = self._compute_metrics(equity_series, daily_returns)

        config_dict = {
            "strategy": self.config.strategy,
            "symbol": self.config.symbol,
            "initial_equity": initial_equity,
            "slippage_bps": self.slippage_bps,
            "num_klines": len(klines_df),
            "period": f"{klines_df.index[0]} ~ {klines_df.index[-1]}",
        }
        # Add strategy-specific params
        if self.config.strategy == "momentum":
            config_dict.update({
                "fast_ema": self.config.mom_fast_period,
                "slow_ema": self.config.mom_slow_period,
                "atr_period": self.config.mom_atr_period,
                "atr_mult": self.config.mom_atr_multiplier,
            })
        elif self.config.strategy == "market_maker":
            config_dict.update({
                "spread_bps": self.config.mm_spread_bps,
                "num_levels": self.config.mm_num_levels,
            })
        elif self.config.strategy == "grid":
            config_dict.update({
                "grid_range_pct": self.config.grid_upper_pct,
                "num_grids": self.config.grid_num_grids,
            })

        return BacktestResult(
            config=config_dict,
            equity_curve=equity_series,
            trades=self.trades,
            daily_returns=daily_returns,
            metrics=metrics,
            klines_df=klines_df,
        )

    def _execute_signal(self, signal: Signal, current_price: float, ts: pd.Timestamp):
        """Execute a trading signal in simulation."""
        is_maker = signal.price is not None
        exec_price = signal.price if is_maker else current_price

        # Apply slippage for market orders
        if not is_maker:
            slip = exec_price * self.slippage_bps / 10000
            exec_price = exec_price + slip if signal.side == "Buy" else exec_price - slip

        qty = signal.qty_usd / exec_price
        fee = _calc_fee(signal.qty_usd, is_maker)

        # Close existing position if reversing
        if self.position_qty != 0:
            close_side = "Sell" if self.position_qty > 0 else "Buy"
            if signal.side == close_side or signal.reason.startswith("trailing_stop"):
                pnl = self.position_qty * (exec_price - self.entry_price)
                self.cash += pnl - fee
                self.trades.append(Trade(
                    timestamp=ts, side=signal.side, price=exec_price,
                    qty=abs(self.position_qty), qty_usd=abs(self.position_qty) * exec_price,
                    fee=fee, reason=signal.reason, pnl=pnl,
                ))
                self.position_qty = 0
                self.position_side = None
                self.entry_price = 0

                # Notify strategy
                self.strategy.on_position_update({"size": "0", "side": "", "avgPrice": "0"})

                # If signal was just a close (trailing stop), return
                if signal.reason.startswith("trailing_stop"):
                    return

                # For reversal, reduce remaining qty
                qty = max(0, qty - abs(self.trades[-1].qty))
                if qty * exec_price < 5:  # Below min notional
                    return

        # Open new position
        if signal.side == "Buy":
            self.position_qty = qty
            self.position_side = "Buy"
        else:
            self.position_qty = -qty
            self.position_side = "Sell"

        self.entry_price = exec_price
        self.cash -= fee

        self.trades.append(Trade(
            timestamp=ts, side=signal.side, price=exec_price,
            qty=qty, qty_usd=qty * exec_price, fee=fee,
            reason=signal.reason, pnl=0,
        ))

        self.strategy.on_position_update({
            "size": str(abs(self.position_qty)),
            "side": signal.side,
            "avgPrice": str(exec_price),
        })

    def _close_position(self, price: float, ts: pd.Timestamp, reason: str):
        """Force close position."""
        if self.position_qty == 0:
            return
        pnl = self.position_qty * (price - self.entry_price)
        fee = _calc_fee(abs(self.position_qty) * price, False)
        self.cash += pnl - fee
        side = "Sell" if self.position_qty > 0 else "Buy"
        self.trades.append(Trade(
            timestamp=ts, side=side, price=price,
            qty=abs(self.position_qty), qty_usd=abs(self.position_qty) * price,
            fee=fee, reason=reason, pnl=pnl,
        ))
        self.position_qty = 0
        self.equity = self.cash

    def _execute_grid_signal(self, signal: Signal, current_price: float, ts: pd.Timestamp):
        """Simplified grid execution for backtest."""
        pass  # Grid orders are passive; checked in _check_grid_fills

    def _check_grid_fills(self, row: pd.Series, ts: pd.Timestamp):
        """Check if price crossed any grid levels during this candle."""
        if not isinstance(self.strategy, GridStrategy):
            return
        low, high = row["low"], row["high"]
        for i, price in enumerate(self.strategy.grid_prices):
            if low <= price <= high:
                if i not in self.strategy.filled_levels and price < row["close"]:
                    # Buy filled
                    qty = self.strategy.order_size_usd / price
                    fee = _calc_fee(self.strategy.order_size_usd, True)
                    self.position_qty += qty
                    self.cash -= fee
                    self.entry_price = price if self.position_qty == qty else (
                        (self.entry_price * (self.position_qty - qty) + price * qty) / self.position_qty
                    )
                    self.strategy.filled_levels.add(i)
                    self.trades.append(Trade(
                        timestamp=ts, side="Buy", price=price, qty=qty,
                        qty_usd=qty * price, fee=fee, reason=f"grid_fill_L{i}", pnl=0,
                    ))
                elif i in self.strategy.filled_levels and price > row["close"]:
                    # Sell filled
                    qty = self.strategy.order_size_usd / price
                    fee = _calc_fee(self.strategy.order_size_usd, True)
                    pnl_per_unit = price - self.entry_price if self.position_qty > 0 else 0
                    pnl = qty * pnl_per_unit
                    self.position_qty -= qty
                    self.cash += pnl - fee
                    self.strategy.filled_levels.discard(i)
                    self.trades.append(Trade(
                        timestamp=ts, side="Sell", price=price, qty=qty,
                        qty_usd=qty * price, fee=fee, reason=f"grid_fill_L{i}", pnl=pnl,
                    ))

    @staticmethod
    def _compute_metrics(equity_curve: pd.Series, daily_returns: pd.Series) -> dict:
        """Compute standard performance metrics."""
        if len(daily_returns) < 2:
            return {}

        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
        n_days = (equity_curve.index[-1] - equity_curve.index[0]).days
        n_years = max(n_days / 365.25, 1 / 365.25)
        annualized_return = ((equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / n_years) - 1) * 100

        ann_vol = daily_returns.std() * np.sqrt(365) * 100  # Crypto = 365 days
        sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(365)) if daily_returns.std() > 0 else 0

        # Max drawdown
        cum_max = equity_curve.cummax()
        drawdown = (equity_curve - cum_max) / cum_max * 100
        max_dd = drawdown.min()

        # Sortino ratio
        downside = daily_returns[daily_returns < 0]
        sortino = (daily_returns.mean() / downside.std() * np.sqrt(365)) if len(downside) > 0 and downside.std() > 0 else 0

        # Calmar ratio
        calmar = annualized_return / abs(max_dd) if max_dd != 0 else 0

        return {
            "total_return_pct": round(total_return, 2),
            "annualized_return_pct": round(annualized_return, 2),
            "annualized_vol_pct": round(ann_vol, 2),
            "sharpe_ratio": round(sharpe, 3),
            "sortino_ratio": round(sortino, 3),
            "calmar_ratio": round(calmar, 3),
            "max_drawdown_pct": round(max_dd, 2),
            "n_trades": 0,  # Filled by caller
            "win_rate_pct": 0,
            "avg_trade_pnl": 0,
            "profit_factor": 0,
            "n_days": n_days,
        }


def run_backtest(
    strategy: str = "momentum",
    symbol: str = "BTCUSDT",
    interval: str = "5",
    start: str = "2024-01-01",
    end: str = "2024-12-31",
    initial_equity: float = 10000.0,
    slippage_bps: float = 1.0,
    on_fetch_progress: Optional[callable] = None,
    on_sim_progress: Optional[callable] = None,
    **strategy_params,
) -> BacktestResult:
    """
    Convenience function to run a full backtest.

    Args:
        strategy: Strategy name (momentum, market_maker, grid)
        symbol: Trading pair
        interval: Kline interval in minutes (1, 5, 15, 60, 240, etc.)
        start: Start date
        end: End date
        initial_equity: Starting capital
        slippage_bps: Slippage in basis points
        on_fetch_progress: callback(count, pct, message) for data fetch progress
        on_sim_progress: callback(processed, total, pct, message) for simulation progress
        **strategy_params: Override strategy parameters

    Returns:
        BacktestResult with equity curve, trades, and metrics
    """
    config = BybitConfig.from_env(strategy=strategy, symbol=symbol, **strategy_params)

    logger.info(f"Fetching klines: {symbol} {interval}m {start} ~ {end}")
    klines_df = fetch_klines(symbol=symbol, interval=interval, start=start, end=end,
                             on_progress=on_fetch_progress)

    bt = Backtester(config, slippage_bps=slippage_bps)
    result = bt.run(klines_df, initial_equity=initial_equity, on_progress=on_sim_progress)

    # Fill trade-level metrics
    closing_trades = [t for t in result.trades if t.pnl != 0]
    if closing_trades:
        wins = [t for t in closing_trades if t.pnl > 0]
        losses = [t for t in closing_trades if t.pnl < 0]
        result.metrics["n_trades"] = len(result.trades)
        result.metrics["n_closing_trades"] = len(closing_trades)
        result.metrics["win_rate_pct"] = round(len(wins) / len(closing_trades) * 100, 1)
        result.metrics["avg_trade_pnl"] = round(np.mean([t.pnl for t in closing_trades]), 2)
        total_profit = sum(t.pnl for t in wins) if wins else 0
        total_loss = abs(sum(t.pnl for t in losses)) if losses else 1
        result.metrics["profit_factor"] = round(total_profit / total_loss, 2) if total_loss > 0 else float("inf")
        result.metrics["total_fees"] = round(sum(t.fee for t in result.trades), 2)

    logger.info(f"Backtest complete: {result.metrics}")
    return result
