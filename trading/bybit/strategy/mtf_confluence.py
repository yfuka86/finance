"""Multi-Timeframe Confluence Strategy — 5m/15m/1H条件合流で発火。

3つの短期時間足で異なるインジケータを監視し、
全条件が同時に揃った高確信度の瞬間のみエントリーする。

== 時間足ごとの役割 ==

1H (方向フィルタ):
  - EMA(fast) vs EMA(slow) でトレンド方向を決定
  - ADX >= threshold でトレンド強度を確認
  → bullish / bearish / neutral

15m (セットアップ):
  - 15m EMA(20) でトレンド継続を確認
  - ロング: 15m終値 > 15m EMA(20)
  - ショート: 15m終値 < 15m EMA(20)

5m (トリガー):
  - RSI(14) がトレンド方向への押し目/戻り
    (ロング: RSI < rsi_entry / ショート: RSI > 100-rsi_entry)
  - 出来高が直近平均の vol_mult 倍以上
  → 全3条件合流で発火 (トレンド押し目買い/戻り売り)

== 決済ルール ==
  - 損切り: エントリー価格 × sl_pct% (パーセント固定SL)
  - 利確: エントリー価格 × tp_pct% (パーセント固定TP)
  - 時間切れ: max_hold本で強制決済
  - クールダウン: 決済後 cooldown本は新規エントリー禁止

== 入力タイムフレーム ==
  5m推奨。内部で 5m×3=15m、5m×12=1H に集約。
"""
import logging
from collections import deque
from typing import Optional

import numpy as np

from .base import BaseStrategy, Signal

logger = logging.getLogger(__name__)


class MTFConfluenceStrategy(BaseStrategy):

    def __init__(self, config):
        super().__init__(config)

        # --- 5m trigger params ---
        self.rsi_period = getattr(config, "cfl_rsi_period", 14)
        self.rsi_entry = getattr(config, "cfl_rsi_entry", 40.0)
        self.vol_lookback = getattr(config, "cfl_vol_lookback", 60)
        self.vol_mult = getattr(config, "cfl_vol_mult", 1.5)

        # --- 15m setup params ---
        self.ema_15m_period = getattr(config, "cfl_ema_15m", 20)
        self.bars_per_15m = getattr(config, "cfl_bars_per_15m", 3)  # 5m×3=15m

        # --- 1H filter params ---
        self.ema_fast = getattr(config, "cfl_ema_fast", 20)
        self.ema_slow = getattr(config, "cfl_ema_slow", 50)
        self.adx_period = getattr(config, "cfl_adx_period", 14)
        self.adx_threshold = getattr(config, "cfl_adx_threshold", 25.0)
        self.bars_per_1h = getattr(config, "cfl_bars_per_1h", 12)  # 5m×12=1H

        # --- Risk params (percentage-based) ---
        self.sl_pct = getattr(config, "cfl_sl_pct", 1.0)    # 1% stop loss
        self.tp_pct = getattr(config, "cfl_tp_pct", 2.0)    # 2% take profit
        self.max_hold = getattr(config, "cfl_max_hold", 144)  # 5m×144=12h
        self.cooldown = getattr(config, "cfl_cooldown", 36)   # 5m×36=3h
        self.order_size_usd = getattr(config, "cfl_order_size_usd", 3000.0)

        # --- 5m data ---
        buf = max(self.rsi_period + 2, self.vol_lookback + 5, 80)
        self.closes_5m = deque(maxlen=buf)
        self.highs_5m = deque(maxlen=buf)
        self.lows_5m = deque(maxlen=buf)
        self.volumes_5m = deque(maxlen=self.vol_lookback + 5)

        # --- 15m aggregation ---
        self._15m_count = 0
        self._15m_high = 0.0
        self._15m_low = float("inf")
        self._15m_open = 0.0
        self._15m_vol = 0.0
        buf_15m = self.ema_15m_period + 10
        self.closes_15m = deque(maxlen=buf_15m)
        self._ema_15m_val = None

        # --- 1H aggregation ---
        self._1h_count = 0
        self._1h_high = 0.0
        self._1h_low = float("inf")
        self._1h_open = 0.0
        buf_1h = max(self.ema_slow + 20, self.adx_period * 3)
        self.closes_1h = deque(maxlen=buf_1h)
        self.highs_1h = deque(maxlen=buf_1h)
        self.lows_1h = deque(maxlen=buf_1h)
        self._ema_fast_val = None
        self._ema_slow_val = None

        # --- State ---
        self.current_position_side = None
        self.entry_price = 0.0
        self.stop_loss = None
        self.take_profit = None
        self._bars_in_trade = 0
        self._cooldown_remaining = 0
        self.trend = None  # "bull" / "bear" / None

    # ── 15m aggregation ──────────────────────────────────────────

    def _push_15m(self, close, high, low, volume):
        if self._15m_count == 0:
            self._15m_open = close
            self._15m_high = high
            self._15m_low = low
            self._15m_vol = volume
        else:
            self._15m_high = max(self._15m_high, high)
            self._15m_low = min(self._15m_low, low)
            self._15m_vol += volume

        self._15m_count += 1
        if self._15m_count >= self.bars_per_15m:
            self.closes_15m.append(close)
            self._15m_count = 0
            self._update_15m_ema(close)

    def _update_15m_ema(self, close):
        n = len(self.closes_15m)
        k = 2 / (self.ema_15m_period + 1)
        if self._ema_15m_val is None and n >= self.ema_15m_period:
            self._ema_15m_val = np.mean(list(self.closes_15m)[-self.ema_15m_period:])
        elif self._ema_15m_val is not None:
            self._ema_15m_val = self._ema_15m_val * (1 - k) + close * k

    # ── 1H aggregation ───────────────────────────────────────────

    def _push_1h(self, close, high, low):
        if self._1h_count == 0:
            self._1h_open = close
            self._1h_high = high
            self._1h_low = low
        else:
            self._1h_high = max(self._1h_high, high)
            self._1h_low = min(self._1h_low, low)

        self._1h_count += 1
        if self._1h_count >= self.bars_per_1h:
            self.closes_1h.append(close)
            self.highs_1h.append(self._1h_high)
            self.lows_1h.append(self._1h_low)
            self._1h_count = 0
            self._update_1h_indicators(close)

    def _update_1h_indicators(self, close):
        n = len(self.closes_1h)
        k_f = 2 / (self.ema_fast + 1)
        k_s = 2 / (self.ema_slow + 1)

        if self._ema_fast_val is None and n >= self.ema_fast:
            self._ema_fast_val = np.mean(list(self.closes_1h)[-self.ema_fast:])
        elif self._ema_fast_val is not None:
            self._ema_fast_val = self._ema_fast_val * (1 - k_f) + close * k_f

        if self._ema_slow_val is None and n >= self.ema_slow:
            self._ema_slow_val = np.mean(list(self.closes_1h)[-self.ema_slow:])
        elif self._ema_slow_val is not None:
            self._ema_slow_val = self._ema_slow_val * (1 - k_s) + close * k_s

        if self._ema_fast_val is not None and self._ema_slow_val is not None:
            if self._ema_fast_val > self._ema_slow_val:
                self.trend = "bull"
            elif self._ema_fast_val < self._ema_slow_val:
                self.trend = "bear"
            else:
                self.trend = None

    # ── Indicators ───────────────────────────────────────────────

    def _rsi_5m(self) -> Optional[float]:
        n = self.rsi_period
        if len(self.closes_5m) < n + 1:
            return None
        c = list(self.closes_5m)
        gains, losses = 0.0, 0.0
        for i in range(-n, 0):
            diff = c[i] - c[i - 1]
            if diff > 0:
                gains += diff
            else:
                losses -= diff
        ag = gains / n
        al = losses / n
        if al == 0:
            return 100.0
        return 100 - 100 / (1 + ag / al)

    def _adx_1h(self) -> Optional[float]:
        n = self.adx_period
        if len(self.closes_1h) < n + 2 or len(self.highs_1h) < n + 2:
            return None
        h = list(self.highs_1h)
        lo = list(self.lows_1h)
        c = list(self.closes_1h)

        plus_dm, minus_dm, tr_list = [], [], []
        for i in range(-n - 1, 0):
            up = h[i] - h[i - 1]
            dn = lo[i - 1] - lo[i]
            plus_dm.append(up if up > dn and up > 0 else 0)
            minus_dm.append(dn if dn > up and dn > 0 else 0)
            tr = max(h[i] - lo[i], abs(h[i] - c[i - 1]), abs(lo[i] - c[i - 1]))
            tr_list.append(tr)

        atr = np.mean(tr_list[-n:])
        if atr == 0:
            return None
        plus_di = np.mean(plus_dm[-n:]) / atr * 100
        minus_di = np.mean(minus_dm[-n:]) / atr * 100
        di_sum = plus_di + minus_di
        if di_sum == 0:
            return None
        dx = abs(plus_di - minus_di) / di_sum * 100
        return dx

    def _vol_spike(self, volume) -> bool:
        if len(self.volumes_5m) < self.vol_lookback:
            return False
        past = list(self.volumes_5m)[-(self.vol_lookback + 1):-1]
        avg = np.mean(past)
        if avg == 0:
            return False
        return volume >= avg * self.vol_mult

    # ── Main logic ───────────────────────────────────────────────

    def on_kline(self, kline: dict) -> Optional[Signal]:
        if isinstance(kline, dict):
            close = float(kline.get("close", kline.get("c", 0)))
            high = float(kline.get("high", kline.get("h", 0)))
            low = float(kline.get("low", kline.get("l", 0)))
            volume = float(kline.get("volume", kline.get("v", 0)))
            confirm = kline.get("confirm", True)
        else:
            close = float(kline[4])
            high = float(kline[2])
            low = float(kline[3])
            volume = float(kline[5])
            confirm = True

        if not confirm:
            return None

        # Feed 5m
        self.closes_5m.append(close)
        self.highs_5m.append(high)
        self.lows_5m.append(low)
        self.volumes_5m.append(volume)

        # Aggregate to 15m, 1H
        self._push_15m(close, high, low, volume)
        self._push_1h(close, high, low)

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

        # Compute 5m RSI
        rsi = self._rsi_5m()
        if rsi is None:
            return None

        # ── Manage existing position ─────────────────────────────
        if self.current_position_side:
            self._bars_in_trade += 1

            if self.current_position_side == "Buy":
                if self.stop_loss and close <= self.stop_loss:
                    self._exit()
                    return Signal(side="Sell", qty_usd=self.order_size_usd,
                                  reason="trailing_stop_cfl_sl_long")
                if self.take_profit and close >= self.take_profit:
                    self._exit()
                    return Signal(side="Sell", qty_usd=self.order_size_usd,
                                  reason="trailing_stop_cfl_tp_long")
                if self._bars_in_trade >= self.max_hold:
                    self._exit()
                    return Signal(side="Sell", qty_usd=self.order_size_usd,
                                  reason="trailing_stop_cfl_timeout_long")

            elif self.current_position_side == "Sell":
                if self.stop_loss and close >= self.stop_loss:
                    self._exit()
                    return Signal(side="Buy", qty_usd=self.order_size_usd,
                                  reason="trailing_stop_cfl_sl_short")
                if self.take_profit and close <= self.take_profit:
                    self._exit()
                    return Signal(side="Buy", qty_usd=self.order_size_usd,
                                  reason="trailing_stop_cfl_tp_short")
                if self._bars_in_trade >= self.max_hold:
                    self._exit()
                    return Signal(side="Buy", qty_usd=self.order_size_usd,
                                  reason="trailing_stop_cfl_timeout_short")
            return None

        # ── No position — check entry ────────────────────────────
        if self._cooldown_remaining > 0:
            return None

        # --- Condition 1: 1H trend + ADX ---
        if self.trend is None:
            return None
        adx = self._adx_1h()
        if adx is None or adx < self.adx_threshold:
            return None

        # --- Condition 2: 15m EMA alignment ---
        if self._ema_15m_val is None:
            return None
        last_15m = list(self.closes_15m)[-1] if self.closes_15m else None
        if last_15m is None:
            return None

        # --- Condition 3: 5m RSI pullback + volume spike ---
        has_vol = self._vol_spike(volume)

        # === LONG: bull 1H + 15m above EMA + 5m RSI pullback + vol spike ===
        if (self.trend == "bull"
                and last_15m > self._ema_15m_val
                and rsi < self.rsi_entry
                and has_vol):
            sl = close * (1 - self.sl_pct / 100)
            tp = close * (1 + self.tp_pct / 100)
            self.stop_loss = sl
            self.take_profit = tp
            self._bars_in_trade = 0
            return Signal(side="Buy", qty_usd=self.order_size_usd,
                          stop_loss=sl, take_profit=tp,
                          reason="cfl_confluence_long")

        # === SHORT: bear 1H + 15m below EMA + 5m RSI spike + vol spike ===
        if (self.trend == "bear"
                and last_15m < self._ema_15m_val
                and rsi > (100 - self.rsi_entry)
                and has_vol):
            sl = close * (1 + self.sl_pct / 100)
            tp = close * (1 - self.tp_pct / 100)
            self.stop_loss = sl
            self.take_profit = tp
            self._bars_in_trade = 0
            return Signal(side="Sell", qty_usd=self.order_size_usd,
                          stop_loss=sl, take_profit=tp,
                          reason="cfl_confluence_short")

        return None

    def _exit(self):
        self.current_position_side = None
        self.stop_loss = None
        self.take_profit = None
        self._cooldown_remaining = self.cooldown

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
