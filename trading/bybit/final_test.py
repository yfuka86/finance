#!/usr/bin/env python3
"""
最終最適化テスト。
- ポジションサイズ: 資本の30-50% ($3000-5000 / $10000)
- 全戦略を4H/1Hの長い時間足で
- BTC/ETH/SOL
- 2025年1-3月
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trading.bybit.backtest import run_backtest

# 大きめポジション + 低頻度 = 少ない手数料 + 大きなリターン
TESTS = [
    # ── Donchian (best performer from previous tests) ────
    {"name": "Don BTC 1H 20/10 sz3k", "strategy": "donchian_breakout", "symbol": "BTCUSDT", "interval": "60",
     "don_entry_period": 20, "don_exit_period": 10, "don_atr_sl_mult": 4.0, "don_order_size_usd": 3000},
    {"name": "Don BTC 4H 10/5 sz5k", "strategy": "donchian_breakout", "symbol": "BTCUSDT", "interval": "240",
     "don_entry_period": 10, "don_exit_period": 5, "don_atr_sl_mult": 3.0, "don_order_size_usd": 5000},
    {"name": "Don BTC 1H 30/15 sz3k", "strategy": "donchian_breakout", "symbol": "BTCUSDT", "interval": "60",
     "don_entry_period": 30, "don_exit_period": 15, "don_atr_sl_mult": 4.0, "don_order_size_usd": 3000},
    {"name": "Don SOL 4H 20/10 sz5k", "strategy": "donchian_breakout", "symbol": "SOLUSDT", "interval": "240",
     "don_entry_period": 20, "don_exit_period": 10, "don_atr_sl_mult": 3.0, "don_order_size_usd": 5000},
    {"name": "Don ETH 1H 20/10 sz3k", "strategy": "donchian_breakout", "symbol": "ETHUSDT", "interval": "60",
     "don_entry_period": 20, "don_exit_period": 10, "don_atr_sl_mult": 4.0, "don_order_size_usd": 3000},

    # ── MACD+ADX ────────────────────────────────────────
    {"name": "MACD BTC 1H ADX15 sz3k", "strategy": "macd_adx", "symbol": "BTCUSDT", "interval": "60",
     "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
     "macd_adx_threshold": 15.0, "macd_atr_sl_mult": 3.5, "macd_order_size_usd": 3000},
    {"name": "MACD BTC 4H ADX15 sz5k", "strategy": "macd_adx", "symbol": "BTCUSDT", "interval": "240",
     "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
     "macd_adx_threshold": 15.0, "macd_atr_sl_mult": 3.0, "macd_order_size_usd": 5000},
    {"name": "MACD ETH 1H ADX15 sz3k", "strategy": "macd_adx", "symbol": "ETHUSDT", "interval": "60",
     "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
     "macd_adx_threshold": 15.0, "macd_atr_sl_mult": 3.5, "macd_order_size_usd": 3000},
    {"name": "MACD SOL 1H ADX15 sz3k", "strategy": "macd_adx", "symbol": "SOLUSDT", "interval": "60",
     "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
     "macd_adx_threshold": 15.0, "macd_atr_sl_mult": 3.5, "macd_order_size_usd": 3000},

    # ── TrendRegime ─────────────────────────────────────
    {"name": "TR BTC 4H MA20 sz5k", "strategy": "trend_regime", "symbol": "BTCUSDT", "interval": "240",
     "tr_ma_period": 20, "tr_adx_threshold": 20.0, "tr_atr_sl_mult": 3.0, "tr_slope_lookback": 3, "tr_order_size_usd": 5000},
    {"name": "TR BTC 1H MA20 sz3k", "strategy": "trend_regime", "symbol": "BTCUSDT", "interval": "60",
     "tr_ma_period": 20, "tr_adx_threshold": 20.0, "tr_atr_sl_mult": 3.0, "tr_slope_lookback": 5, "tr_order_size_usd": 3000},
    {"name": "TR SOL 4H MA20 sz5k", "strategy": "trend_regime", "symbol": "SOLUSDT", "interval": "240",
     "tr_ma_period": 20, "tr_adx_threshold": 20.0, "tr_atr_sl_mult": 3.0, "tr_slope_lookback": 3, "tr_order_size_usd": 5000},
    {"name": "TR ETH 4H MA20 sz5k", "strategy": "trend_regime", "symbol": "ETHUSDT", "interval": "240",
     "tr_ma_period": 20, "tr_adx_threshold": 20.0, "tr_atr_sl_mult": 3.0, "tr_slope_lookback": 3, "tr_order_size_usd": 5000},

    # ── Momentum (long TF, big size) ────────────────────
    {"name": "Mom BTC 4H 20/50 ATR3 sz5k", "strategy": "momentum", "symbol": "BTCUSDT", "interval": "240",
     "mom_fast_period": 20, "mom_slow_period": 50, "mom_atr_period": 14, "mom_atr_multiplier": 3.0, "mom_order_size_usd": 5000},
    {"name": "Mom BTC 1H 10/30 ATR3 sz3k", "strategy": "momentum", "symbol": "BTCUSDT", "interval": "60",
     "mom_fast_period": 10, "mom_slow_period": 30, "mom_atr_period": 14, "mom_atr_multiplier": 3.0, "mom_order_size_usd": 3000},
    {"name": "Mom SOL 4H 20/50 ATR3 sz5k", "strategy": "momentum", "symbol": "SOLUSDT", "interval": "240",
     "mom_fast_period": 20, "mom_slow_period": 50, "mom_atr_period": 14, "mom_atr_multiplier": 3.0, "mom_order_size_usd": 5000},
    {"name": "Mom ETH 1H 10/30 ATR3 sz3k", "strategy": "momentum", "symbol": "ETHUSDT", "interval": "60",
     "mom_fast_period": 10, "mom_slow_period": 30, "mom_atr_period": 14, "mom_atr_multiplier": 3.0, "mom_order_size_usd": 3000},

    # ── MRF (Bollinger+RSI combo) ───────────────────────
    {"name": "MRF BTC 4H sz3k", "strategy": "mean_reversion_filtered", "symbol": "BTCUSDT", "interval": "240",
     "mrf_bb_period": 20, "mrf_bb_mult": 2.0, "mrf_rsi_oversold": 30.0, "mrf_rsi_overbought": 70.0,
     "mrf_vol_mult": 1.0, "mrf_atr_sl_mult": 2.0, "mrf_order_size_usd": 3000},
    {"name": "MRF BTC 1H sz3k", "strategy": "mean_reversion_filtered", "symbol": "BTCUSDT", "interval": "60",
     "mrf_bb_period": 20, "mrf_bb_mult": 2.0, "mrf_rsi_oversold": 25.0, "mrf_rsi_overbought": 75.0,
     "mrf_vol_mult": 1.0, "mrf_atr_sl_mult": 1.5, "mrf_order_size_usd": 3000},
    {"name": "MRF ETH 1H sz3k", "strategy": "mean_reversion_filtered", "symbol": "ETHUSDT", "interval": "60",
     "mrf_bb_period": 20, "mrf_bb_mult": 2.0, "mrf_rsi_oversold": 25.0, "mrf_rsi_overbought": 75.0,
     "mrf_vol_mult": 1.0, "mrf_atr_sl_mult": 1.5, "mrf_order_size_usd": 3000},
    {"name": "MRF SOL 4H sz3k", "strategy": "mean_reversion_filtered", "symbol": "SOLUSDT", "interval": "240",
     "mrf_bb_period": 20, "mrf_bb_mult": 2.0, "mrf_rsi_oversold": 30.0, "mrf_rsi_overbought": 70.0,
     "mrf_vol_mult": 1.0, "mrf_atr_sl_mult": 2.0, "mrf_order_size_usd": 3000},

    # ── RSI Reversion (short-term) ──────────────────────
    {"name": "RSI BTC 1H p14 sz3k", "strategy": "rsi_reversion", "symbol": "BTCUSDT", "interval": "60",
     "rsi_period": 14, "rsi_oversold": 25.0, "rsi_overbought": 75.0, "rsi_exit_level": 50.0, "rsi_order_size_usd": 3000},
    {"name": "RSI ETH 1H p14 sz3k", "strategy": "rsi_reversion", "symbol": "ETHUSDT", "interval": "60",
     "rsi_period": 14, "rsi_oversold": 25.0, "rsi_overbought": 75.0, "rsi_exit_level": 50.0, "rsi_order_size_usd": 3000},

    # ── BB Reversion ────────────────────────────────────
    {"name": "BB BTC 1H 2σ sz3k", "strategy": "bollinger_reversion", "symbol": "BTCUSDT", "interval": "60",
     "bb_period": 20, "bb_std_mult": 2.0, "bb_adx_threshold": 25.0, "bb_order_size_usd": 3000},
    {"name": "BB BTC 4H 2σ sz5k", "strategy": "bollinger_reversion", "symbol": "BTCUSDT", "interval": "240",
     "bb_period": 20, "bb_std_mult": 2.0, "bb_adx_threshold": 25.0, "bb_order_size_usd": 5000},

    # ── Volatility Breakout ─────────────────────────────
    {"name": "VB BTC 4H k0.6 sz5k", "strategy": "volatility_breakout", "symbol": "BTCUSDT", "interval": "240",
     "vb_lookback": 4, "vb_k_long": 0.6, "vb_k_short": 0.6, "vb_atr_sl_mult": 3.0, "vb_session_bars": 6, "vb_order_size_usd": 5000},

    # ── R-Breaker ────────────────────────────────────────
    {"name": "RB BTC 1H sz3k", "strategy": "rbreaker", "symbol": "BTCUSDT", "interval": "60",
     "rb_f1": 0.35, "rb_f2": 0.07, "rb_f3": 0.25, "rb_session_bars": 24, "rb_order_size_usd": 3000},
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

        print(f"[{i+1}/{total}] {name}")
        try:
            result = run_backtest(
                strategy=strategy, symbol=symbol, interval=interval,
                start=START, end=END, initial_equity=INITIAL, slippage_bps=1.0,
                **test,
            )
            m = result.metrics
            ann = m.get("annualized_return_pct", 0)
            if ann != ann:
                ann = -999
            print(f"  Ret: {m.get('total_return_pct',0):+.2f}% Ann: {ann:+.2f}% "
                  f"S: {m.get('sharpe_ratio',0):.3f} DD: {m.get('max_drawdown_pct',0):.2f}% "
                  f"T: {m.get('n_trades',0)} W: {m.get('win_rate_pct',0):.0f}%")
            results.append((name, ann, m))
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append((name, -999, {}))

    results.sort(key=lambda x: x[1], reverse=True)
    print("\n" + "=" * 90)
    print(f"{'Name':40s} {'Ann%':>8s} {'Sharpe':>8s} {'MaxDD%':>8s} {'Trades':>7s} {'WR%':>6s}")
    print("=" * 90)
    for name, ann, m in results:
        marker = "✓" if ann >= THRESHOLD else " "
        print(f"{marker} {name:38s} {ann:+8.2f} {m.get('sharpe_ratio',0):8.3f} "
              f"{m.get('max_drawdown_pct',0):8.2f} {m.get('n_trades',0):7d} {m.get('win_rate_pct',0):5.0f}%")

    passed = [n for n, a, _ in results if a >= THRESHOLD]
    print(f"\n年利{THRESHOLD}%以上: {len(passed)}/{len(results)}")
    for n in passed:
        print(f"  ✓ {n}")


if __name__ == "__main__":
    main()
