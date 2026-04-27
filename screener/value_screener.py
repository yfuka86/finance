"""
Value-Reversal Screener
=======================
J-Quants API のみで完結するバリュー×テクニカル反転スクリーナー。
yfinance 不使用。

データソース:
  - /v2/equities/bars/daily      → OHLCV (テクニカル分析)
  - /v2/fins/summary             → PER / PBR / 現預金
  - /v2/equities/master          → 銘柄名・業種

Usage:
    python -m screener.run
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import io

from data.collectors.config import JQUANTS_API_KEY, JQUANTS_BASE
from data.collectors.jquants import (
    fetch_equities_master,
    fetch_fins_summary,
    fetch_bars_daily,
)

# ── Cache directory ───────────────────────────────────────────

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


# ── JPX master (日本語銘柄名・業種) ───────────────────────────

_JPX_CACHE: pd.DataFrame | None = None


def fetch_jpx_listed() -> pd.DataFrame:
    """JPX公開の上場銘柄一覧 (コード / 銘柄名 / 業種 / 市場区分)."""
    global _JPX_CACHE
    if _JPX_CACHE is not None:
        return _JPX_CACHE
    url = ("https://www.jpx.co.jp/markets/statistics-equities/"
           "misc/tvdivq0000001vg2-att/data_j.xls")
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        df = pd.read_excel(io.BytesIO(r.content), engine="xlrd")
        df["コード"] = df["コード"].astype(str).str[:4]
        _JPX_CACHE = df
        return df
    except Exception as e:
        print(f"  JPX一覧取得失敗: {e}")
        return pd.DataFrame()


# ── J-Quants bars/daily helpers ───────────────────────────────

def _jq_bars(params: dict) -> pd.DataFrame:
    """Paginated fetch from /v2/equities/bars/daily with retry on 429."""
    headers = {"x-api-key": JQUANTS_API_KEY}
    records = []
    while True:
        for attempt in range(4):
            r = requests.get(f"{JQUANTS_BASE}/v2/equities/bars/daily",
                             params=params, headers=headers, timeout=30)
            if r.status_code == 429:
                wait = 2 ** attempt + 1
                time.sleep(wait)
                continue
            break
        if r.status_code != 200:
            break
        body = r.json()
        records.extend(body.get("data", []))
        pk = body.get("pagination_key")
        if not pk:
            break
        params["pagination_key"] = pk
        time.sleep(0.3)
    return pd.DataFrame(records) if records else pd.DataFrame()


def fetch_all_stocks(date_str: str) -> pd.DataFrame:
    """全銘柄の OHLCV を取得 (date=YYYYMMDD)."""
    return _jq_bars({"date": date_str.replace("-", "")})


def fetch_stock_history(code: str, from_d: str, to_d: str) -> pd.DataFrame:
    """個別銘柄の日足を取得."""
    return _jq_bars({
        "code": code.replace(".T", ""),
        "from": from_d.replace("-", ""),
        "to": to_d.replace("-", ""),
    })


# ── Bulk history cache ────────────────────────────────────────

def _bulk_history_cache_path(from_d: str, to_d: str) -> str:
    return os.path.join(CACHE_DIR, f"bars_{from_d}_{to_d}.parquet")


def fetch_bulk_history(from_d: str, to_d: str) -> pd.DataFrame:
    """全銘柄の日足を日単位で一括取得。各日ごとにキャッシュ。"""
    cache = _bulk_history_cache_path(from_d, to_d)
    if os.path.exists(cache):
        print(f"  キャッシュ読込: {os.path.basename(cache)}")
        return pd.read_parquet(cache)

    # 日付リストを生成
    start = dt.datetime.strptime(from_d, "%Y-%m-%d").date()
    end = dt.datetime.strptime(to_d, "%Y-%m-%d").date()
    all_frames = []

    # 個別日キャッシュを確認しつつ取得
    d = start
    days_total = (end - start).days + 1
    days_done = 0
    while d <= end:
        if d.weekday() >= 5:  # skip weekends
            d += dt.timedelta(days=1)
            continue
        ds = d.strftime("%Y-%m-%d")
        day_cache = os.path.join(CACHE_DIR, f"bars_day_{ds}.parquet")

        if os.path.exists(day_cache):
            df = pd.read_parquet(day_cache)
        else:
            df = _jq_bars({"date": ds.replace("-", "")})
            if not df.empty:
                df["Code"] = df["Code"].astype(str).str[:4]
                for col in ("C", "O", "H", "L", "Vo", "Va",
                            "AdjC", "AdjO", "AdjH", "AdjL"):
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                df.to_parquet(day_cache)

        if not df.empty:
            all_frames.append(df)
        days_done += 1
        if days_done % 10 == 0:
            print(f"    {days_done} 日取得済 ({ds})")
        d += dt.timedelta(days=1)

    if not all_frames:
        return pd.DataFrame()

    result = pd.concat(all_frames, ignore_index=True)
    # 結合キャッシュも保存
    result.to_parquet(cache)
    print(f"  → {len(result)} レコード ({result['Code'].nunique()} 銘柄)")
    return result


# ── fins/summary cache ────────────────────────────────────────

def _fins_cache_path(date_str: str) -> str:
    return os.path.join(CACHE_DIR, f"fins_{date_str}.json")


def _load_fins_cache(date_str: str) -> dict:
    path = _fins_cache_path(date_str)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    # gzip分割ファイルからの自動復元
    import gzip as _gzip
    parts = sorted(glob.glob(os.path.join(CACHE_DIR, f"fins_{date_str}_part*.json.gz")))
    if parts:
        merged = {}
        for p in parts:
            with _gzip.open(p, "rt", encoding="utf-8") as f:
                merged.update(json.load(f))
        # 復元したJSONを書き出して次回高速化
        _save_fins_cache(date_str, merged)
        print(f"  fins cache restored from {len(parts)} parts ({len(merged)} codes)")
        return merged
    return {}


def _save_fins_cache(date_str: str, data: dict):
    path = _fins_cache_path(date_str)
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False)


def _fetch_single_fins(code: str) -> tuple[str, list]:
    """1銘柄の fins/summary を取得 → (code, records_list)."""
    try:
        df = fetch_fins_summary(code)
        if df.empty:
            return code, []
        return code, df.to_dict("records")
    except Exception:
        return code, []


# ── Next earnings estimator ──────────────────────────────────

def _estimate_next_earnings(latest, today: dt.date) -> str | None:
    """fins/summary の CurFYEn / CurPerType から次回決算発表推定日を算出。"""
    try:
        fy_end_str = str(latest.get("CurFYEn", ""))
        per_type = str(latest.get("CurPerType", ""))
        if not fy_end_str or len(fy_end_str) < 10 or not per_type:
            return None

        fy_end = dt.datetime.strptime(fy_end_str[:10], "%Y-%m-%d").date()
        fy_start_str = str(latest.get("CurFYSt", ""))
        if fy_start_str and len(fy_start_str) >= 10:
            fy_start = dt.datetime.strptime(fy_start_str[:10], "%Y-%m-%d").date()
        else:
            fy_start = fy_end.replace(year=fy_end.year - 1) + dt.timedelta(days=1)

        total_days = (fy_end - fy_start).days
        q1_end = fy_start + dt.timedelta(days=total_days // 4)
        q2_end = fy_start + dt.timedelta(days=total_days // 2)
        q3_end = fy_start + dt.timedelta(days=total_days * 3 // 4)

        if per_type == "FY":
            next_per_end = fy_end + dt.timedelta(days=total_days // 4)
        elif per_type == "3Q":
            next_per_end = fy_end
        elif per_type == "2Q":
            next_per_end = q3_end
        elif per_type == "1Q":
            next_per_end = q2_end
        else:
            return None

        est_date = next_per_end + dt.timedelta(days=45)
        if est_date < today:
            est_date = est_date + dt.timedelta(days=total_days // 4)

        return est_date.strftime("%Y-%m-%d")
    except Exception:
        return None


def _build_split_factor_map(out: dict):
    """キャッシュ済みバーデータから銘柄ごとの分割係数を算出。

    分割前の日付では C/AdjC > 1 になるため、期間中の最大値を分割係数とする。
    fins/summary の EPS/BPS/ShOutFY はこの係数で調整する。
    """
    cache_dir = os.path.join(os.path.dirname(__file__), "..", "data", "cache")
    combined = sorted(glob.glob(os.path.join(cache_dir, "bars_20*.parquet")))
    if not combined:
        return
    bars = pd.read_parquet(combined[-1])
    bars["Code"] = bars["Code"].astype(str).str[:4]
    for col in ("C", "AdjC"):
        bars[col] = pd.to_numeric(bars[col], errors="coerce")
    valid = bars[(bars["C"] > 0) & (bars["AdjC"] > 0)].copy()
    valid["ratio"] = valid["C"] / valid["AdjC"]
    max_ratio = valid.groupby("Code")["ratio"].max()
    for code, ratio in max_ratio.items():
        if ratio > 1.01:
            out[code] = round(ratio, 6)


# ── J-Quants fundamentals (with cache + concurrency) ─────────

def fetch_fundamentals_jq(codes: list, quote_date: str) -> dict:
    """J-Quants API で PER / PBR / 時価総額 / 現預金を取得。キャッシュ+並列取得。"""
    out = {}

    # 1) 終値を bars/daily (全銘柄一括) から取得
    #    C=生の終値, AdjC=分割調整済み終値
    #    分割係数 = C / AdjC (分割前の銘柄は >1, 例: 1:5分割 → 係数5)
    #    fins/summary の EPS/BPS/ShOutFY は分割前の値なので調整が必要
    print(f"  bars/daily 取得中 ({quote_date})...")
    dq = fetch_bars_daily(date=quote_date)
    close_map = {}  # code -> 現在の終値 (C, 生値)
    split_factor_map = {}  # code -> 分割係数 (C / AdjC)
    if not dq.empty:
        dq["Code"] = dq["Code"].astype(str).str[:4]
        for col in ("AdjC", "C"):
            if col in dq.columns:
                dq[col] = pd.to_numeric(dq[col], errors="coerce")
        if "C" in dq.columns:
            close_map = dict(zip(dq["Code"], dq["C"]))
        # 分割係数は過去データの C/AdjC 最大値から推定
        # (分割前の日付では C/AdjC = 分割倍率, 分割後は 1.0)
        _build_split_factor_map(split_factor_map)

    # 2) equities/master で銘��名・業種取得
    print("  equities/master 取得中...")
    info_df = fetch_equities_master()
    name_map = {}
    sector_map = {}
    if not info_df.empty:
        info_df["Code"] = info_df["Code"].astype(str).str[:4]
        if "CoName" in info_df.columns:
            name_map = dict(zip(info_df["Code"], info_df["CoName"]))
        if "S33Nm" in info_df.columns:
            sector_map = dict(zip(info_df["Code"], info_df["S33Nm"]))

    # 3) fins/summary — キャッシュ活用 + 並列取得
    today = dt.date.today()
    cache_key = quote_date.replace("-", "")
    cache = _load_fins_cache(cache_key)

    # キャッシュにない銘柄を特定
    missing_codes = [c for c in codes if c not in cache]
    cached_count = len(codes) - len(missing_codes)
    if cached_count > 0:
        print(f"  fins/summary キャッシュ: {cached_count} 銘柄")

    if missing_codes:
        print(f"  fins/summary 取得中 ({len(missing_codes)} 銘柄)...")
        # 逐次取得 (レートリミット対応: 0.15s間隔)
        for i, code in enumerate(missing_codes):
            _, records = _fetch_single_fins(code)
            cache[code] = records
            if (i + 1) % 100 == 0:
                print(f"    {i+1}/{len(missing_codes)}")
                # 100件ごとに中間キャッシュ保存
                _save_fins_cache(cache_key, cache)
            time.sleep(0.15)
        _save_fins_cache(cache_key, cache)
        print(f"  → fins/summary 完了 ({len(missing_codes)} 銘柄取得)")

    # 4) 各銘柄のファンダメンタルを計算
    n_split_adj = 0
    for code in codes:
        entry = {
            "Name": name_map.get(code, ""),
            "Sector": sector_map.get(code, ""),
            "PER": None, "fPER": None, "PBR": None,
            "MarketCap": None, "Cash": None, "EarningsDate": None,
        }

        records = cache.get(code, [])
        if records:
            stmts = pd.DataFrame(records)
            if "DiscDate" in stmts.columns:
                stmts = stmts.sort_values("DiscDate", ascending=False)
            latest = stmts.iloc[0]
            close_price = close_map.get(code)

            # 分割調整: fins/summary の per-share 数値は分割前基準
            # split_factor > 1 の場合、EPS/BPS を割り、ShOutFY を掛ける
            sf = split_factor_map.get(code, 1.0)
            if sf > 1.01:
                n_split_adj += 1

            eps = pd.to_numeric(latest.get("EPS"), errors="coerce")
            if pd.notna(eps) and eps > 0 and close_price and close_price > 0:
                entry["PER"] = round(close_price / (eps / sf), 1)

            feps = pd.to_numeric(latest.get("FEPS"), errors="coerce")
            if pd.notna(feps) and feps > 0 and close_price and close_price > 0:
                entry["fPER"] = round(close_price / (feps / sf), 1)

            bps = pd.to_numeric(latest.get("BPS"), errors="coerce")
            if not (pd.notna(bps) and bps > 0):
                eq = pd.to_numeric(latest.get("Eq"), errors="coerce")
                shares = pd.to_numeric(latest.get("ShOutFY"), errors="coerce")
                tr_shares = pd.to_numeric(latest.get("TrShFY"), errors="coerce")
                if pd.notna(eq) and pd.notna(shares) and shares > 0:
                    net_shares = (shares * sf) - ((tr_shares * sf) if pd.notna(tr_shares) else 0)
                    if net_shares > 0:
                        bps = eq / net_shares
            else:
                # BPS も分割前基準なので調整
                bps = bps / sf
            if pd.notna(bps) and bps > 0 and close_price and close_price > 0:
                entry["PBR"] = round(close_price / bps, 2)

            shares = pd.to_numeric(latest.get("ShOutFY"), errors="coerce")
            if pd.notna(shares) and close_price and close_price > 0:
                entry["MarketCap"] = close_price * shares * sf

            cash = pd.to_numeric(latest.get("CashEq"), errors="coerce")
            if pd.notna(cash):
                entry["Cash"] = cash

            ta = pd.to_numeric(latest.get("TA"), errors="coerce")
            eq = pd.to_numeric(latest.get("Eq"), errors="coerce")
            if pd.notna(cash) and pd.notna(ta) and pd.notna(eq):
                entry["NetCash"] = cash - (ta - eq)  # 現金 - 負債

            entry["EarningsDate"] = _estimate_next_earnings(latest, today)

        out[code] = entry

    if n_split_adj > 0:
        print(f"  → 株式分割調整: {n_split_adj} 銘柄")
    return out


# ── Default parameters ────────────────────────────────────────

DEFAULTS = dict(
    per_min=3.0,
    per_max=15.0,
    pbr_min=0.1,
    pbr_max=2.5,
    mix_max=30.0,
    turnover_min=1e7,    # 最低売買代金 1000万円/日
    rsi_period=14,
    rsi_oversold=35,
    rsi_upper=55,
    macd_fast=12,
    macd_slow=26,
    macd_signal=9,
    ma_short=5,
    ma_long=25,
    lookback=60,
    vol_short=5,
    vol_long=20,
    earnings_window=45,
    top_n=0,
    max_scan=0,
)


# ── Technical helpers ─────────────────────────────────────────

def _rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    ag = pd.Series(gain).ewm(span=period, min_periods=period).mean().values
    al = pd.Series(loss).ewm(span=period, min_periods=period).mean().values
    rs = np.where(al > 0, ag / al, 100.0)
    return 100.0 - 100.0 / (1.0 + rs)


def _macd(close: np.ndarray, fast=12, slow=26, sig=9):
    s = pd.Series(close)
    ef = s.ewm(span=fast, min_periods=fast).mean()
    es = s.ewm(span=slow, min_periods=slow).mean()
    line = ef - es
    signal = line.ewm(span=sig, min_periods=sig).mean()
    return line.values, signal.values, (line - signal).values


def _sma(close: np.ndarray, w: int) -> np.ndarray:
    return pd.Series(close).rolling(w, min_periods=w).mean().values


# ── Screener ──────────────────────────────────────────────────

class ValueReversalScreener:
    """
    Pass 1  J-Quants bars/daily: 全銘柄 → 流動性フィルタ → テクニカル分析 (一括データ)
    Pass 2  J-Quants fins/summary: PER/PBR/MIX/現預金 (キャッシュ+並列)
    Pass 3  複合スコア (フィルタはダッシュボード側)
    """

    def __init__(self, **kw):
        self.p = {**DEFAULTS, **kw}
        self.today = dt.date.today()

    # ── Pass 1: price/volume from J-Quants ────────────────────

    def _latest_universe(self) -> pd.DataFrame:
        """直近営業日の全銘柄 OHLCV を取得."""
        print("[1/5] 全銘柄データ取得中...")
        for off in range(10):
            d = self.today - dt.timedelta(days=off)
            ds = d.strftime("%Y-%m-%d")
            df = fetch_all_stocks(ds)
            if not df.empty:
                self._quote_date = ds
                df["Code"] = df["Code"].astype(str).str[:4]
                for col in ("C", "O", "H", "L", "Vo", "Va",
                            "AdjC", "AdjO", "AdjH", "AdjL"):
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                print(f"  {ds}: {len(df)} 銘柄")
                return df
        raise RuntimeError("直近の株価データが取得できません")

    def _liquidity_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """売買代金フィルタ."""
        print("[2/5] 流動性フィルタ...")
        before = len(df)
        df = df[df["Va"] >= self.p["turnover_min"]].copy()
        print(f"  {before} → {len(df)} 銘柄 (売買代金 >= {self.p['turnover_min']/1e8:.1f}億円)")
        return df

    def _load_bulk_history(self) -> dict[str, pd.DataFrame]:
        """全銘柄の日足を一括取得し、銘柄別に分割して返す。"""
        p = self.p
        from_d = (self.today - dt.timedelta(days=p["lookback"] * 2)).strftime("%Y-%m-%d")
        to_d = self.today.strftime("%Y-%m-%d")

        print(f"[3a/5] 日足データ一括取得中 ({from_d} → {to_d})...")
        bulk = fetch_bulk_history(from_d, to_d)
        if bulk.empty:
            return {}

        # 銘柄ごとに分割
        result = {}
        for code, grp in bulk.groupby("Code"):
            result[code] = grp.sort_values("Date").reset_index(drop=True)
        print(f"  → {len(result)} 銘柄の履歴データ取得完了")
        return result

    def _score_technical_from_data(self, hist: pd.DataFrame) -> dict | None:
        """事前取得済みの履歴データからテクニカルスコアを算出。"""
        p = self.p
        if len(hist) < p["macd_slow"] + p["macd_signal"]:
            return None

        close = pd.to_numeric(hist["AdjC"], errors="coerce").dropna().values
        vol = pd.to_numeric(hist["Vo"], errors="coerce").dropna().values
        va = pd.to_numeric(hist["Va"], errors="coerce").dropna().values
        dates = hist["Date"].values
        if len(close) < 30:
            return None

        # RSI
        rsi = _rsi(close, p["rsi_period"])
        rsi_now = rsi[-1]
        rsi_5ago = rsi[-6] if len(rsi) >= 6 else rsi_now
        if rsi_5ago <= p["rsi_oversold"] and rsi_now > rsi_5ago:
            rsi_sc = 1.0
        elif rsi_now <= p["rsi_oversold"]:
            rsi_sc = 0.5
        elif rsi_now <= p["rsi_upper"]:
            rsi_sc = 0.3
        else:
            rsi_sc = 0.0

        # MACD
        _, _, mh = _macd(close, p["macd_fast"], p["macd_slow"], p["macd_signal"])
        if np.isnan(mh[-1]) or np.isnan(mh[-2]):
            macd_sc = 0.0
        elif mh[-1] > 0 and mh[-2] <= 0:
            macd_sc = 1.0
        elif mh[-1] > mh[-2]:
            macd_sc = 0.5
        else:
            macd_sc = 0.0

        # MA cross
        ms = _sma(close, p["ma_short"])
        ml = _sma(close, p["ma_long"])
        if np.isnan(ms[-1]) or np.isnan(ml[-1]):
            ma_sc = 0.0
        else:
            ma_sc = 0.0
            if close[-1] > ms[-1] and ms[-1] > ml[-1]:
                ma_sc = 1.0
            elif close[-1] > ms[-1]:
                ma_sc = 0.5
            if (len(ms) >= 2 and not np.isnan(ms[-2]) and not np.isnan(ml[-2])
                    and ms[-2] <= ml[-2] and ms[-1] > ml[-1]):
                ma_sc = min(ma_sc + 0.5, 1.0)

        # Bottom bounce
        lb = min(p["lookback"], len(close))
        lo, hi = np.nanmin(close[-lb:]), np.nanmax(close[-lb:])
        rng = hi - lo
        pos = (close[-1] - lo) / rng if rng > 0 else 0.5
        bottom_sc = max(0.0, 1.0 - abs(pos - 0.25) * 3)

        # Volume pickup
        vs = np.nanmean(vol[-p["vol_short"]:])
        vl = np.nanmean(vol[-p["vol_long"]:])
        vr = vs / vl if vl > 0 else 1.0
        vol_sc = min(1.0, max(0.0, (vr - 1.0) / 0.5))

        tech = rsi_sc * 0.25 + macd_sc * 0.25 + ma_sc * 0.25 + bottom_sc * 0.25

        # 直近10日の売買代金 (億円) と日付
        n_spark = min(10, len(va))
        va_spark = [round(float(v) / 1e8, 1) for v in va[-n_spark:]]
        va_dates = [str(d)[:10] for d in dates[-n_spark:]]
        va_latest = round(float(va[-1]) / 1e8, 1) if len(va) > 0 else 0
        va_avg5 = round(float(np.nanmean(va[-5:])) / 1e8, 1) if len(va) >= 5 else va_latest
        va_avg20 = round(float(np.nanmean(va[-20:])) / 1e8, 1) if len(va) >= 20 else va_avg5

        return dict(
            Close=float(close[-1]),
            RSI=round(float(rsi_now), 1),
            RSI_sc=round(rsi_sc, 2),
            MACD_hist=round(float(mh[-1]), 4) if not np.isnan(mh[-1]) else 0.0,
            MACD_sc=round(macd_sc, 2),
            MA_sc=round(ma_sc, 2),
            PricePos=round(float(pos), 2),
            Bottom_sc=round(bottom_sc, 2),
            VolRatio=round(float(vr), 2),
            Vol_sc=round(vol_sc, 2),
            TechScore=round(tech, 2),
            Va_latest=va_latest,
            Va_avg5=va_avg5,
            Va_avg20=va_avg20,
            Va_spark=va_spark,
            Va_dates=va_dates,
        )

    def _run_technical_pass(self, universe: pd.DataFrame,
                            history: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """一括取得済み履歴データでテクニカルスコアを算出。"""
        p = self.p
        codes = universe["Code"].tolist()
        max_scan = p["max_scan"]
        if max_scan > 0 and len(codes) > max_scan:
            universe = universe.nlargest(max_scan, "Va")
            codes = universe["Code"].tolist()

        print(f"[3b/5] テクニカル分析中... ({len(codes)} 銘柄)")
        rows = []
        for i, code in enumerate(codes):
            hist = history.get(code)
            if hist is None or hist.empty:
                continue
            sc = self._score_technical_from_data(hist)
            if sc is None:
                continue
            row = universe[universe["Code"] == code].iloc[0]
            rows.append({
                "Code": code,
                "Turnover_M": round(float(row["Va"]) / 1e6, 0),
                **sc,
            })
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(codes)} ({len(rows)} 候補)")
        print(f"  → テクニカル候補: {len(rows)} 銘柄")
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    # ── Pass 2: fundamentals from J-Quants ────────────────────

    def _enrich_fundamentals(self, df: pd.DataFrame) -> pd.DataFrame:
        """テクニカル上位銘柄の PER/PBR/時価総額/現預金を J-Quants から取得."""
        codes = df["Code"].tolist()
        print(f"[4/5] ファンダメンタル取得中 (J-Quants, {len(codes)} 銘柄)...")
        fdata = fetch_fundamentals_jq(codes, self._quote_date)

        for c, fd in fdata.items():
            mask = df["Code"] == c
            if not mask.any():
                continue
            df.loc[mask, "Name"] = fd.get("Name", "")
            df.loc[mask, "Sector"] = fd.get("Sector", "")
            if fd.get("PER") is not None:
                df.loc[mask, "PER"] = round(fd["PER"], 1)
            if fd.get("fPER") is not None:
                df.loc[mask, "fPER"] = round(fd["fPER"], 1)
            if fd.get("PBR") is not None:
                df.loc[mask, "PBR"] = round(fd["PBR"], 2)
            if fd.get("MarketCap") is not None:
                mc = fd["MarketCap"]
                df.loc[mask, "MarketCap_B"] = round(mc / 1e8, 1) if mc else None
            if fd.get("Cash") is not None and fd.get("MarketCap"):
                df.loc[mask, "CashRatio"] = round(
                    fd["Cash"] / fd["MarketCap"] * 100, 1)
            if fd.get("NetCash") is not None and fd.get("MarketCap"):
                df.loc[mask, "NetCashRatio"] = round(
                    fd["NetCash"] / fd["MarketCap"] * 100, 1)
            if fd.get("EarningsDate"):
                df.loc[mask, "EarningsDate"] = fd["EarningsDate"]

        # ── EDINET BS 詳細から広義ネットキャッシュを上書き ──
        from screener.edinet_bs import fetch_bs_batch
        print("[4b/5] EDINET BS 詳細取得中...")
        bs_data = fetch_bs_batch(codes)
        n_bs = 0
        for c, bsd in bs_data.items():
            mask = df["Code"] == c
            if not mask.any():
                continue
            nc_info = bsd.get("net_cash_info")
            if nc_info is None:
                continue
            mc_raw = fdata.get(c, {}).get("MarketCap")
            if mc_raw and mc_raw > 0:
                # 広義ネットキャッシュ比率で上書き
                df.loc[mask, "NetCashRatio"] = round(
                    nc_info["net_cash"] / mc_raw * 100, 1)
                n_bs += 1
            # BS 内訳列 (ダッシュボード表示用)
            bd = nc_info["breakdown"]
            df.loc[mask, "BS_NearCash"] = round(nc_info["near_cash"] / 1e8, 1)
            df.loc[mask, "BS_Debt"] = round(nc_info["interest_debt"] / 1e8, 1)
            df.loc[mask, "BS_NetCash"] = round(nc_info["net_cash"] / 1e8, 1)
        if n_bs:
            print(f"  → EDINET BS で広義ネットキャッシュ更新: {n_bs} 銘柄")
        return df

    # ── Pass 3: composite score (no pre-filter) ────────────────

    def _value_filter_and_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """MIX を算出し、複合スコアを付与 (フィルタはダッシュボード側)."""
        print("[5/5] スコアリング...")

        before = len(df)

        # MIX 計算のみ (フィルタは掛けない)
        if "PER" in df.columns and "PBR" in df.columns:
            both = df["PER"].notna() & df["PBR"].notna()
            df.loc[both, "MIX"] = df.loc[both, "PER"] * df.loc[both, "PBR"]

        print(f"  {before} 銘柄")

        if df.empty:
            return df

        # ── 7軸評価でスコアリング ──
        from screener.report_format import evaluate_stock

        eval_results = []
        for _, row in df.iterrows():
            ev = evaluate_stock(row.to_dict())
            bd = ev["breakdown"]
            eval_results.append({
                "Sc_Valuation": bd.get("valuation", 0),
                "Sc_Financial": bd.get("financial", 0),
                "Sc_Growth": bd.get("growth", 0),
                "Sc_Technical": bd.get("technical", 0),
                "Sc_Supply": bd.get("supply", 0),
                "Sc_Catalyst": bd.get("catalyst", 0),
                "Sc_Risk": bd.get("risk", 0),
                "TotalScore": ev["total_score"],
            })
        eval_df = pd.DataFrame(eval_results, index=df.index)
        df = pd.concat([df, eval_df], axis=1)

        # ── ファンダメンタル総合 (テクニカル除外) ──
        # バリュー(20) + 財務(15) + 成長性(15) + カタリスト(10) + リスク(10) = 70点満点
        df["FundaScore"] = (
            df["Sc_Valuation"]
            + df["Sc_Financial"]
            + df["Sc_Growth"]
            + df["Sc_Catalyst"]
            + df["Sc_Risk"]
        ).round(1)

        # テクニカル込みの総合 (参考用に残す)
        df["Score"] = df["TotalScore"]

        df = df.sort_values("FundaScore", ascending=False)
        if self.p["top_n"] > 0:
            df = df.head(self.p["top_n"])
        df = df.reset_index(drop=True)
        df.index += 1
        return df

    # ── Main ──────────────────────────────────────────────────

    def screen(self) -> pd.DataFrame:
        print("=" * 60)
        print("  Value-Reversal Screener")
        print(f"  {self.today}")
        print("=" * 60)

        universe = self._latest_universe()
        liquid = self._liquidity_filter(universe)
        if liquid.empty:
            return pd.DataFrame()

        # 一括で全銘柄の日足を取得 (API 1回 vs 2950回)
        history = self._load_bulk_history()

        tech = self._run_technical_pass(liquid, history)
        if tech.empty:
            return pd.DataFrame()

        enriched = self._enrich_fundamentals(tech)

        # JPX日本語名をマージ
        jpx = fetch_jpx_listed()
        if not jpx.empty:
            name_map = dict(zip(jpx["コード"], jpx["銘柄名"]))
            sector_map = dict(zip(jpx["コード"], jpx["33業種区分"]))
            enriched["NameJP"] = enriched["Code"].map(name_map).fillna("")
            enriched["SectorJP"] = enriched["Code"].map(sector_map).fillna("")

        result = self._value_filter_and_score(enriched)
        return result


# ── Display ───────────────────────────────────────────────────

def _text_spark(vals: list) -> str:
    """数値リストを簡易テキストスパークラインに変換."""
    if not vals:
        return ""
    bars = " ▁▂▃▄▅▆▇█"
    mx = max(vals) if max(vals) > 0 else 1
    return "".join(bars[min(8, int(v / mx * 8))] for v in vals)


def format_results(df: pd.DataFrame) -> str:
    if df.empty:
        return "候補銘柄なし"

    # 日本語名があればそちらを使う
    out = df.copy()
    if "NameJP" in out.columns:
        out["Name"] = out["NameJP"].fillna(out.get("Name", ""))
    if "SectorJP" in out.columns:
        out["Sector"] = out["SectorJP"].fillna(out.get("Sector", ""))

    rename = {
        "Code": "コード", "Name": "銘柄名", "Sector": "業種",
        "PER": "PER", "fPER": "fPER", "PBR": "PBR", "MIX": "MIX",
        "MarketCap_B": "時価総額(億)", "CashRatio": "現金%",
        "Sc_Valuation": "割安", "Sc_Financial": "財務", "Sc_Growth": "成長",
        "Sc_Catalyst": "触媒", "Sc_Risk": "リスク",
        "FundaScore": "Fスコア",
        "Va_latest": "代金(億)",
        "EarningsDate": "決算予定",
    }
    show = [c for c in rename if c in out.columns]
    disp = out[show].copy()
    if "Name" in disp.columns:
        disp["Name"] = disp["Name"].astype(str).apply(
            lambda s: s[:10] + ".." if len(s) > 12 else s)
    if "Sector" in disp.columns:
        disp["Sector"] = disp["Sector"].astype(str).apply(
            lambda s: s[:6] + ".." if len(s) > 8 else s)
    disp = disp.rename(columns=rename)

    lines = [
        "=" * 140,
        "  VALUE SCREENER — ファンダメンタル重視 (テクニカル除外)",
        "=" * 140, "",
        disp.to_string(), "",
    ]

    # 出来高推移 (テキストスパークライン)
    if "Va_spark" in out.columns:
        lines.append("-" * 130)
        lines.append("  売買代金推移 (直近10営業日, 億円)")
        lines.append("")
        for _, r in out.iterrows():
            code = r.get("Code", "")
            name = str(r.get("Name", ""))[:10]
            spark_vals = r.get("Va_spark")
            if isinstance(spark_vals, str):
                import ast as _ast
                try:
                    spark_vals = _ast.literal_eval(spark_vals)
                except Exception:
                    spark_vals = []
            if not isinstance(spark_vals, list):
                spark_vals = []
            spark = _text_spark(spark_vals)
            nums = " ".join(f"{v:5.0f}" for v in spark_vals) if spark_vals else ""
            lines.append(f"  {code} {name:<12s} {spark}  {nums}")
        lines.append("")

    lines.extend([
        "-" * 140,
        "Fスコア = 割安(20) + 財務(15) + 成長(15) + 触媒(10) + リスク(10) = 70点満点",
        "割安: PER/fPER/PBR/MIX  |  財務: 現金比率+資産裏付け  |  成長: fPER/PER比+推定ROE",
        "触媒: 決算近接度  |  リスク: 流動性+バリュートラップ兆候 (減点方式)",
        "",
    ])
    return "\n".join(lines)
