"""Bybit trading bot configuration."""
import os
from dataclasses import dataclass, field


@dataclass
class BybitConfig:
    # API credentials (from environment variables)
    api_key: str = ""
    api_secret: str = ""

    # Network: "testnet", "demo", "mainnet"
    network: str = "testnet"

    # Trading pair
    symbol: str = "BTCUSDT"
    category: str = "linear"  # linear (USDT perp), inverse, spot

    # Risk management
    max_position_usd: float = 1000.0      # Max position size in USD
    max_drawdown_pct: float = 5.0          # Kill switch at -5%
    max_daily_loss_usd: float = 50.0       # Daily loss limit
    risk_per_trade_pct: float = 1.0        # Risk per trade (% of equity)
    max_leverage: int = 5                  # Max leverage

    # Strategy defaults
    strategy: str = "momentum"  # momentum, market_maker, grid

    # Market maker params
    mm_spread_bps: float = 10.0            # Spread in basis points
    mm_order_size_usd: float = 100.0       # Order size per side
    mm_num_levels: int = 3                 # Number of order levels
    mm_level_spacing_bps: float = 5.0      # Spacing between levels

    # Momentum params
    mom_fast_period: int = 10              # Fast EMA period
    mom_slow_period: int = 30              # Slow EMA period
    mom_signal_period: int = 9             # Signal line period
    mom_atr_period: int = 14              # ATR for stop loss
    mom_atr_multiplier: float = 2.0        # ATR multiplier for stops
    mom_order_size_usd: float = 200.0      # Order size

    # Grid params
    grid_upper_pct: float = 3.0            # Upper bound (% from mid)
    grid_lower_pct: float = 3.0            # Lower bound (% from mid)
    grid_num_grids: int = 10               # Number of grid levels
    grid_order_size_usd: float = 50.0      # Size per grid order

    # Bollinger Band mean reversion params
    bb_period: int = 20
    bb_std_mult: float = 2.0
    bb_adx_period: int = 14
    bb_adx_threshold: float = 25.0
    bb_order_size_usd: float = 200.0

    # RSI mean reversion params
    rsi_period: int = 2
    rsi_oversold: float = 10.0
    rsi_overbought: float = 90.0
    rsi_exit_level: float = 55.0
    rsi_order_size_usd: float = 200.0

    # Donchian breakout params
    don_entry_period: int = 20
    don_exit_period: int = 10
    don_atr_period: int = 20
    don_atr_sl_mult: float = 2.0
    don_order_size_usd: float = 200.0

    # Volatility breakout (Dual Thrust) params
    vb_lookback: int = 4
    vb_k_long: float = 0.5
    vb_k_short: float = 0.5
    vb_atr_period: int = 14
    vb_atr_sl_mult: float = 2.0
    vb_session_bars: int = 24
    vb_order_size_usd: float = 200.0

    # Ichimoku params
    ichi_tenkan: int = 9
    ichi_kijun: int = 26
    ichi_senkou_b: int = 52
    ichi_order_size_usd: float = 200.0

    # MACD + ADX params
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    macd_adx_period: int = 14
    macd_adx_threshold: float = 20.0
    macd_atr_period: int = 14
    macd_atr_sl_mult: float = 2.0
    macd_order_size_usd: float = 200.0

    # R-Breaker params
    rb_f1: float = 0.35
    rb_f2: float = 0.07
    rb_f3: float = 0.25
    rb_session_bars: int = 24
    rb_order_size_usd: float = 200.0

    # Trend Regime params
    tr_ma_period: int = 50
    tr_adx_period: int = 14
    tr_adx_threshold: float = 25.0
    tr_atr_period: int = 14
    tr_atr_sl_mult: float = 3.0
    tr_slope_lookback: int = 5
    tr_order_size_usd: float = 200.0

    # Mean Reversion Filtered params
    mrf_bb_period: int = 20
    mrf_bb_mult: float = 2.5
    mrf_rsi_period: int = 14
    mrf_rsi_oversold: float = 25.0
    mrf_rsi_overbought: float = 75.0
    mrf_vol_mult: float = 1.5
    mrf_atr_sl_mult: float = 1.5
    mrf_order_size_usd: float = 200.0

    # TSMOM params
    tsmom_lookback_short: int = 14
    tsmom_lookback_long: int = 28
    tsmom_vol_window: int = 30
    tsmom_vol_target: float = 0.50       # annualized
    tsmom_max_leverage: float = 1.5
    tsmom_order_size_usd: float = 5000.0

    # Dual Regime params
    dr_adx_period: int = 14
    dr_adx_threshold: float = 25.0
    dr_vol_ratio_limit: float = 1.5
    dr_don_entry_period: int = 20
    dr_don_exit_period: int = 10
    dr_bb_period: int = 20
    dr_bb_std_mult: float = 2.0
    dr_atr_period: int = 14
    dr_atr_sl_mult: float = 3.0
    dr_order_size_usd: float = 5000.0

    # MTF RSI(2) params
    mtf_rsi_period: int = 2
    mtf_rsi_entry_long: float = 5.0
    mtf_rsi_entry_short: float = 95.0
    mtf_rsi_exit_long: float = 60.0
    mtf_rsi_exit_short: float = 40.0
    mtf_atr_period: int = 14
    mtf_atr_sl_mult: float = 2.5
    mtf_atr_tp_mult: float = 4.0
    mtf_htf_bars: int = 16               # 15m × 16 = 4H
    mtf_htf_ema_fast: int = 20
    mtf_htf_ema_slow: int = 50
    mtf_order_size_usd: float = 3000.0
    mtf_min_hold: int = 8                # min bars before RSI exit
    mtf_cooldown: int = 12               # cooldown bars after exit
    mtf_trend_gap_pct: float = 0.0       # min EMA gap % (0=disabled)
    mtf_dd_limit_pct: float = 0.0        # pause at this DD % (0=disabled)
    mtf_vol_filter: float = 0.0          # pause if ATR > median*this (0=disabled)

    # VWAP Reversion params
    vwap_entry_sd: float = 2.0
    vwap_stop_sd: float = 3.0
    vwap_rsi_period: int = 6
    vwap_rsi_long: float = 25.0
    vwap_rsi_short: float = 75.0
    vwap_reset_bars: int = 96             # 15m × 96 = 24h
    vwap_max_move_pct: float = 4.0
    vwap_order_size_usd: float = 3000.0
    vwap_min_dev_pct: float = 0.3
    vwap_cooldown: int = 8

    # Volume Spike params
    vs_vol_lookback: int = 200
    vs_vol_zscore: float = 4.0
    vs_wick_ratio: float = 0.65
    vs_min_range_pct: float = 0.15
    vs_max_hold: int = 12                 # 5m × 12 = 60min
    vs_cooldown: int = 12
    vs_stop_buffer_pct: float = 0.1
    vs_tp_wick_mult: float = 2.0
    vs_order_size_usd: float = 2000.0

    # MTF Confluence params
    cfl_rsi_period: int = 14
    cfl_rsi_entry: float = 40.0
    cfl_vol_lookback: int = 60
    cfl_vol_mult: float = 1.5
    cfl_ema_15m: int = 20
    cfl_bars_per_15m: int = 3              # 5m×3=15m
    cfl_ema_fast: int = 20
    cfl_ema_slow: int = 50
    cfl_adx_period: int = 14
    cfl_adx_threshold: float = 25.0
    cfl_bars_per_1h: int = 12              # 5m×12=1H
    cfl_sl_pct: float = 1.0               # 1% stop loss
    cfl_tp_pct: float = 2.0               # 2% take profit
    cfl_max_hold: int = 144                # 5m×144=12h
    cfl_cooldown: int = 36                 # 5m×36=3h
    cfl_order_size_usd: float = 3000.0

    # Engine
    heartbeat_interval: int = 30           # Seconds between heartbeats
    order_refresh_interval: int = 60       # Seconds between order refreshes

    # Logging
    log_level: str = "INFO"
    log_file: str = "bybit_bot.log"

    @classmethod
    def from_env(cls, **overrides) -> "BybitConfig":
        """Create config from environment variables with optional overrides."""
        cfg = cls(
            api_key=os.environ.get("BYBIT_API_KEY", ""),
            api_secret=os.environ.get("BYBIT_API_SECRET", ""),
            network=os.environ.get("BYBIT_NETWORK", "testnet"),
            symbol=os.environ.get("BYBIT_SYMBOL", "BTCUSDT"),
        )
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, type(getattr(cfg, k))(v))
        return cfg

    @property
    def is_testnet(self) -> bool:
        return self.network in ("testnet", "demo")

    @property
    def base_url(self) -> str:
        urls = {
            "mainnet": "https://api.bybit.com",
            "testnet": "https://api-testnet.bybit.com",
            "demo": "https://api-demo.bybit.com",
        }
        return urls[self.network]

    @property
    def ws_url(self) -> str:
        urls = {
            "mainnet": "wss://stream.bybit.com",
            "testnet": "wss://stream-testnet.bybit.com",
            "demo": "wss://stream-demo.bybit.com",
        }
        return urls[self.network]
