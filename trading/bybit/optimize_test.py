#!/usr/bin/env python3
"""
広範なパラメータ探索で年利20%超を達成する戦略を見つける。
2025年1-3月でテスト。複数シンボル・時間足・パラメータを試す。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trading.bybit.backtest import run_backtest

TESTS = [
    # ── Trend Regime (低頻度・大トレンド狙い) ──────────────
    # 4H
    {"name": "TrendRegime 4H MA20 ADX20", "strategy": "trend_regime", "symbol": "BTCUSDT", "interval": "240",
     "tr_ma_period": 20, "tr_adx_threshold": 20.0, "tr_atr_sl_mult": 3.0, "tr_slope_lookback": 3, "tr_order_size_usd": 200},
    {"name": "TrendRegime 4H MA50 ADX25", "strategy": "trend_regime", "symbol": "BTCUSDT", "interval": "240",
     "tr_ma_period": 50, "tr_adx_threshold": 25.0, "tr_atr_sl_mult": 4.0, "tr_slope_lookback": 5, "tr_order_size_usd": 200},
    {"name": "TrendRegime 4H MA20 ADX15", "strategy": "trend_regime", "symbol": "BTCUSDT", "interval": "240",
     "tr_ma_period": 20, "tr_adx_threshold": 15.0, "tr_atr_sl_mult": 2.5, "tr_slope_lookback": 3, "tr_order_size_usd": 200},
    # 1H
    {"name": "TrendRegime 1H MA20 ADX20", "strategy": "trend_regime", "symbol": "BTCUSDT", "interval": "60",
     "tr_ma_period": 20, "tr_adx_threshold": 20.0, "tr_atr_sl_mult": 3.0, "tr_slope_lookback": 5, "tr_order_size_usd": 200},
    {"name": "TrendRegime 1H MA30 ADX25", "strategy": "trend_regime", "symbol": "BTCUSDT", "interval": "60",
     "tr_ma_period": 30, "tr_adx_threshold": 25.0, "tr_atr_sl_mult": 3.5, "tr_slope_lookback": 5, "tr_order_size_usd": 200},
    # ETH
    {"name": "TrendRegime 4H ETH MA20", "strategy": "trend_regime", "symbol": "ETHUSDT", "interval": "240",
     "tr_ma_period": 20, "tr_adx_threshold": 20.0, "tr_atr_sl_mult": 3.0, "tr_slope_lookback": 3, "tr_order_size_usd": 200},
    {"name": "TrendRegime 1H ETH MA20", "strategy": "trend_regime", "symbol": "ETHUSDT", "interval": "60",
     "tr_ma_period": 20, "tr_adx_threshold": 20.0, "tr_atr_sl_mult": 3.0, "tr_slope_lookback": 5, "tr_order_size_usd": 200},

    # ── Donchian Breakout (タートル) ───────────────────────
    {"name": "Donchian 4H 20/10", "strategy": "donchian_breakout", "symbol": "BTCUSDT", "interval": "240",
     "don_entry_period": 20, "don_exit_period": 10, "don_atr_sl_mult": 3.0, "don_order_size_usd": 200},
    {"name": "Donchian 4H 10/5", "strategy": "donchian_breakout", "symbol": "BTCUSDT", "interval": "240",
     "don_entry_period": 10, "don_exit_period": 5, "don_atr_sl_mult": 2.5, "don_order_size_usd": 200},
    {"name": "Donchian 1H 20/10 wide", "strategy": "donchian_breakout", "symbol": "BTCUSDT", "interval": "60",
     "don_entry_period": 20, "don_exit_period": 10, "don_atr_sl_mult": 4.0, "don_order_size_usd": 200},
    {"name": "Donchian 4H ETH 20/10", "strategy": "donchian_breakout", "symbol": "ETHUSDT", "interval": "240",
     "don_entry_period": 20, "don_exit_period": 10, "don_atr_sl_mult": 3.0, "don_order_size_usd": 200},

    # ── MACD+ADX (トレンドフォロー) ──────────────────────
    {"name": "MACD+ADX 4H 12/26 ADX15", "strategy": "macd_adx", "symbol": "BTCUSDT", "interval": "240",
     "macd_fast": 12, "macd_slow": 26, "macd_signal": 9, "macd_adx_threshold": 15.0, "macd_atr_sl_mult": 3.0, "macd_order_size_usd": 200},
    {"name": "MACD+ADX 4H 8/21 ADX20", "strategy": "macd_adx", "symbol": "BTCUSDT", "interval": "240",
     "macd_fast": 8, "macd_slow": 21, "macd_signal": 5, "macd_adx_threshold": 20.0, "macd_atr_sl_mult": 3.0, "macd_order_size_usd": 200},
    {"name": "MACD+ADX 1H 12/26 ADX15", "strategy": "macd_adx", "symbol": "BTCUSDT", "interval": "60",
     "macd_fast": 12, "macd_slow": 26, "macd_signal": 9, "macd_adx_threshold": 15.0, "macd_atr_sl_mult": 3.5, "macd_order_size_usd": 200},
    {"name": "MACD+ADX 4H ETH ADX15", "strategy": "macd_adx", "symbol": "ETHUSDT", "interval": "240",
     "macd_fast": 12, "macd_slow": 26, "macd_signal": 9, "macd_adx_threshold": 15.0, "macd_atr_sl_mult": 3.0, "macd_order_size_usd": 200},

    # ── Mean Reversion Filtered ─────────────────────────
    {"name": "MRF 4H BB2.5 RSI25/75", "strategy": "mean_reversion_filtered", "symbol": "BTCUSDT", "interval": "240",
     "mrf_bb_period": 20, "mrf_bb_mult": 2.5, "mrf_rsi_oversold": 25.0, "mrf_rsi_overbought": 75.0,
     "mrf_vol_mult": 1.2, "mrf_atr_sl_mult": 2.0, "mrf_order_size_usd": 200},
    {"name": "MRF 1H BB2.0 RSI20/80", "strategy": "mean_reversion_filtered", "symbol": "BTCUSDT", "interval": "60",
     "mrf_bb_period": 20, "mrf_bb_mult": 2.0, "mrf_rsi_oversold": 20.0, "mrf_rsi_overbought": 80.0,
     "mrf_vol_mult": 1.5, "mrf_atr_sl_mult": 1.5, "mrf_order_size_usd": 200},
    {"name": "MRF 4H ETH BB2.5", "strategy": "mean_reversion_filtered", "symbol": "ETHUSDT", "interval": "240",
     "mrf_bb_period": 20, "mrf_bb_mult": 2.5, "mrf_rsi_oversold": 25.0, "mrf_rsi_overbought": 75.0,
     "mrf_vol_mult": 1.2, "mrf_atr_sl_mult": 2.0, "mrf_order_size_usd": 200},

    # ── Volatility Breakout (longer timeframes) ──────────
    {"name": "VolBK 4H k=0.6", "strategy": "volatility_breakout", "symbol": "BTCUSDT", "interval": "240",
     "vb_lookback": 4, "vb_k_long": 0.6, "vb_k_short": 0.6, "vb_atr_sl_mult": 3.0, "vb_session_bars": 6, "vb_order_size_usd": 200},
    {"name": "VolBK 1H k=0.5", "strategy": "volatility_breakout", "symbol": "BTCUSDT", "interval": "60",
     "vb_lookback": 4, "vb_k_long": 0.5, "vb_k_short": 0.5, "vb_atr_sl_mult": 3.0, "vb_session_bars": 24, "vb_order_size_usd": 200},

    # ── Ichimoku (longer timeframes) ────────────────────
    {"name": "Ichimoku D 9/26/52", "strategy": "ichimoku", "symbol": "BTCUSDT", "interval": "D",
     "ichi_tenkan": 9, "ichi_kijun": 26, "ichi_senkou_b": 52, "ichi_order_size_usd": 200},
    {"name": "Ichimoku 4H 10/30/60", "strategy": "ichimoku", "symbol": "BTCUSDT", "interval": "240",
     "ichi_tenkan": 10, "ichi_kijun": 30, "ichi_senkou_b": 60, "ichi_order_size_usd": 200},

    # ── Momentum (optimized for longer timeframes) ──────
    {"name": "Momentum 4H 20/50 ATR3", "strategy": "momentum", "symbol": "BTCUSDT", "interval": "240",
     "mom_fast_period": 20, "mom_slow_period": 50, "mom_atr_period": 14, "mom_atr_multiplier": 3.0, "mom_order_size_usd": 200},
    {"name": "Momentum 1H 10/30 ATR3", "strategy": "momentum", "symbol": "BTCUSDT", "interval": "60",
     "mom_fast_period": 10, "mom_slow_period": 30, "mom_atr_period": 14, "mom_atr_multiplier": 3.0, "mom_order_size_usd": 200},

    # ── SOL tests ────────────────────────────────────────
    {"name": "Donchian 4H SOL 20/10", "strategy": "donchian_breakout", "symbol": "SOLUSDT", "interval": "240",
     "don_entry_period": 20, "don_exit_period": 10, "don_atr_sl_mult": 3.0, "don_order_size_usd": 200},
    {"name": "TrendRegime 4H SOL MA20", "strategy": "trend_regime", "symbol": "SOLUSDT", "interval": "240",
     "tr_ma_period": 20, "tr_adx_threshold": 20.0, "tr_atr_sl_mult": 3.0, "tr_slope_lookback": 3, "tr_order_size_usd": 200},
]


def main():
    START = "2025-01-01"
    END = "2025-03-31"
    INITIAL = 10000.0
    THRESHOLD = 20.0

    results = []
    total = len(TESTS)

    for i, test in enumerate(TESTS):
        name = test.pop("name")
        strategy = test.pop("strategy")
        symbol = test.pop("symbol")
        interval = test.pop("interval")

        print(f"[{i+1}/{total}] {name}  ({symbol} {interval})")
        try:
            result = run_backtest(
                strategy=strategy, symbol=symbol, interval=interval,
                start=START, end=END, initial_equity=INITIAL, slippage_bps=1.0,
                **test,
            )
            m = result.metrics
            ann = m.get("annualized_return_pct", 0)
            if ann != ann:  # NaN check
                ann = -999
            print(f"  -> Ret: {m.get('total_return_pct',0):+.2f}% | Ann: {ann:+.2f}% | "
                  f"Sharpe: {m.get('sharpe_ratio',0):.3f} | MaxDD: {m.get('max_drawdown_pct',0):.2f}% | "
                  f"Trades: {m.get('n_trades',0)}")
            results.append((name, symbol, interval, ann, m))
        except Exception as e:
            print(f"  -> FAILED: {e}")
            results.append((name, symbol, interval, -999, {}))

    # Sort and display
    results.sort(key=lambda x: x[3], reverse=True)
    print("\n" + "=" * 100)
    print(f"{'Name':45s} {'Symbol':10s} {'TF':5s} {'Ann%':>8s} {'Sharpe':>8s} {'MaxDD%':>8s} {'Trades':>7s}")
    print("=" * 100)
    for name, sym, tf, ann, m in results:
        marker = "✓" if ann >= THRESHOLD else " "
        print(f"{marker} {name:43s} {sym:10s} {tf:5s} {ann:+8.2f} "
              f"{m.get('sharpe_ratio',0):8.3f} {m.get('max_drawdown_pct',0):8.2f} "
              f"{m.get('n_trades',0):7d}")

    passed = [(n, s, t, a) for n, s, t, a, _ in results if a >= THRESHOLD]
    print(f"\n年利{THRESHOLD}%以上: {len(passed)}/{len(results)}")
    for n, s, t, a in passed:
        print(f"  ✓ {n} ({s} {t}) -> {a:+.2f}%")


if __name__ == "__main__":
    main()
