#!/usr/bin/env python3
"""
Bybit Trading Bot - Entry Point

Usage:
    # Testnet (default - safe for testing)
    python -m trading.bybit.bot --strategy momentum --symbol BTCUSDT

    # With custom config
    python -m trading.bybit.bot --strategy market_maker --symbol ETHUSDT --spread 15

    # Mainnet (real money - be careful!)
    BYBIT_NETWORK=mainnet python -m trading.bybit.bot --strategy grid

Environment variables:
    BYBIT_API_KEY       API key
    BYBIT_API_SECRET    API secret
    BYBIT_NETWORK       testnet (default) | demo | mainnet
    BYBIT_SYMBOL        Trading pair (default: BTCUSDT)
"""
import argparse
import logging
import signal
import sys

from .config import BybitConfig
from .engine import TradingEngine


def setup_logging(config: BybitConfig):
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handlers = [logging.StreamHandler(sys.stdout)]
    if config.log_file:
        handlers.append(logging.FileHandler(config.log_file))
    logging.basicConfig(level=config.log_level, format=fmt, handlers=handlers)


def parse_args():
    p = argparse.ArgumentParser(description="Bybit Trading Bot")
    p.add_argument("--strategy", default="momentum",
                   choices=["momentum", "market_maker", "grid"])
    p.add_argument("--symbol", default=None)
    p.add_argument("--network", default=None, choices=["testnet", "demo", "mainnet"])

    # Risk params
    p.add_argument("--max-position", type=float, default=None,
                   help="Max position size in USD")
    p.add_argument("--max-drawdown", type=float, default=None,
                   help="Max drawdown %% before kill switch")
    p.add_argument("--leverage", type=int, default=None)

    # Strategy-specific
    p.add_argument("--spread", type=float, default=None,
                   help="Market maker spread in bps")
    p.add_argument("--fast-ema", type=int, default=None)
    p.add_argument("--slow-ema", type=int, default=None)
    p.add_argument("--grids", type=int, default=None)
    p.add_argument("--grid-range", type=float, default=None,
                   help="Grid range %% above/below mid")
    p.add_argument("--order-size", type=float, default=None,
                   help="Order size in USD")

    return p.parse_args()


def main():
    args = parse_args()

    overrides = {"strategy": args.strategy}
    if args.symbol:
        overrides["symbol"] = args.symbol
    if args.network:
        overrides["network"] = args.network
    if args.max_position:
        overrides["max_position_usd"] = args.max_position
    if args.max_drawdown:
        overrides["max_drawdown_pct"] = args.max_drawdown
    if args.leverage:
        overrides["max_leverage"] = args.leverage
    if args.spread:
        overrides["mm_spread_bps"] = args.spread
    if args.fast_ema:
        overrides["mom_fast_period"] = args.fast_ema
    if args.slow_ema:
        overrides["mom_slow_period"] = args.slow_ema
    if args.grids:
        overrides["grid_num_grids"] = args.grids
    if args.grid_range:
        overrides["grid_upper_pct"] = args.grid_range
        overrides["grid_lower_pct"] = args.grid_range
    if args.order_size:
        overrides["mom_order_size_usd"] = args.order_size
        overrides["mm_order_size_usd"] = args.order_size
        overrides["grid_order_size_usd"] = args.order_size

    config = BybitConfig.from_env(**overrides)
    setup_logging(config)

    logger = logging.getLogger(__name__)

    if not config.api_key or not config.api_secret:
        logger.error(
            "BYBIT_API_KEY and BYBIT_API_SECRET must be set.\n"
            "  1. Go to https://testnet.bybit.com (testnet) or https://bybit.com (mainnet)\n"
            "  2. Create API key with 'Contract' permission\n"
            "  3. export BYBIT_API_KEY=xxx BYBIT_API_SECRET=yyy"
        )
        sys.exit(1)

    if config.network == "mainnet":
        logger.warning("=" * 60)
        logger.warning("  MAINNET MODE - REAL MONEY AT RISK")
        logger.warning(f"  Symbol: {config.symbol}")
        logger.warning(f"  Strategy: {config.strategy}")
        logger.warning(f"  Max position: ${config.max_position_usd}")
        logger.warning(f"  Max drawdown: {config.max_drawdown_pct}%")
        logger.warning("=" * 60)

    engine = TradingEngine(config)

    # Graceful shutdown on Ctrl+C
    def shutdown(sig, frame):
        logger.info("Shutdown signal received")
        engine.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        engine.start()
    except Exception as e:
        logger.critical(f"Engine crashed: {e}", exc_info=True)
        engine.stop()
        sys.exit(1)


if __name__ == "__main__":
    main()
