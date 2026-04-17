"""
Value-Reversal Screener
=======================
J-Quants bars/daily API でテクニカル + 出来高スクリーニング。
yfinance で PER / PBR / MIX / 現預金を補完。

Usage:
    python -m screener.run
"""
from __future__ import annotations

import datetime as dt
import time
import numpy as np
import pandas as pd
import requests

from data.collectors.config import JQUANTS_API_KEY, JQUANTS_BASE

import io


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
        time.sleep(0.5)
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


# ── yfinance fundamental helpers ──────────────────────────────

def fetch_fundamentals_yf(codes: list) -> dict:
    """
    yfinance で PER / PBR / 時価総額 / 現預金 / 銘柄名 / セクターを取得。
    codes: ['7203', '6758', ...]  (4桁コード)
    """
    import yfinance as yf
    out = {}
    for i, c in enumerate(codes):
        ticker_str = f"{c}.T"
        try:
            info = yf.Ticker(ticker_str).info
        except Exception:
            continue
        out[c] = {
            "Name": info.get("shortName", ""),
            "Sector": info.get("sector", ""),
            "PER": info.get("trailingPE"),
            "fPER": info.get("forwardPE"),
            "PBR": info.get("priceToBook"),
            "MarketCap": info.get("marketCap"),
            "Cash": info.get("totalCash"),
            "EarningsDate": None,
        }
        # earnings date
        ed = info.get("earningsTimestamp") or info.get("mostRecentQuarter")
        if ed and isinstance(ed, (int, float)):
            out[c]["EarningsDate"] = dt.datetime.fromtimestamp(ed).strftime("%Y-%m-%d")
        if (i + 1) % 10 == 0:
            print(f"  yfinance {i+1}/{len(codes)}")
        time.sleep(0.25)
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
    top_n=30,
    max_scan=250,
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
    Pass 1  J-Quants: 全銘柄 → 流動性フィルタ → テクニカル分析
    Pass 2  yfinance: 上位候補の PER/PBR/MIX/現預金
    Pass 3  バリューフィルタ + 複合スコア
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
                # normalise code to 4 digits
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

    def _score_technical(self, code: str) -> dict | None:
        """個別銘柄の価格履歴からテクニカル+出来高スコアを算出."""
        p = self.p
        fr = (self.today - dt.timedelta(days=p["lookback"] * 2)).strftime("%Y-%m-%d")
        to = self.today.strftime("%Y-%m-%d")

        hist = fetch_stock_history(code, fr, to)
        if hist.empty or len(hist) < p["macd_slow"] + p["macd_signal"]:
            return None
        hist = hist.sort_values("Date")
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

    def _run_technical_pass(self, universe: pd.DataFrame) -> pd.DataFrame:
        """全候補のテクニカルスコアを算出."""
        p = self.p
        codes = universe["Code"].tolist()
        # Limit to avoid too many API calls
        max_scan = p["max_scan"]
        if len(codes) > max_scan:
            # sort by turnover (Va) descending, take top max_scan
            universe = universe.nlargest(max_scan, "Va")
            codes = universe["Code"].tolist()

        print(f"[3/5] テクニカル分析中... ({len(codes)} 銘柄)")
        rows = []
        for i, code in enumerate(codes):
            sc = self._score_technical(code)
            if sc is None:
                continue
            row = universe[universe["Code"] == code].iloc[0]
            rows.append({
                "Code": code,
                "Turnover_M": round(float(row["Va"]) / 1e6, 0),
                **sc,
            })
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(codes)} ({len(rows)} 候補)")
            time.sleep(0.5)  # rate limit guard
        print(f"  → テクニカル候補: {len(rows)} 銘柄")
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    # ── Pass 2: fundamentals from yfinance ────────────────────

    def _enrich_fundamentals(self, df: pd.DataFrame) -> pd.DataFrame:
        """テクニカル上位銘柄の PER/PBR/MIX/現預金を取得."""
        p = self.p
        codes = df["Code"].tolist()
        print(f"[4/5] ファンダメンタル取得中 (yfinance, {len(codes)} 銘柄)...")
        fdata = fetch_fundamentals_yf(codes)

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
                df.loc[mask, "MarketCap_B"] = round(fd["MarketCap"] / 1e9, 1)
            if fd.get("Cash") is not None and fd.get("MarketCap"):
                df.loc[mask, "CashRatio"] = round(
                    fd["Cash"] / fd["MarketCap"] * 100, 1)
            if fd.get("EarningsDate"):
                df.loc[mask, "EarningsDate"] = fd["EarningsDate"]
        return df

    # ── Pass 3: value filter & composite score ────────────────

    def _value_filter_and_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """PER/PBR/MIX でフィルタし、複合スコアを算出."""
        print("[5/5] バリューフィルタ & スコアリング...")
        p = self.p

        # Require fundamental data
        before = len(df)
        if "PBR" in df.columns:
            df = df[df["PBR"].notna()]
        if "PER" in df.columns:
            has_per = df["PER"].notna()
            per_ok = (df["PER"] >= p["per_min"]) & (df["PER"] <= p["per_max"])
            df = df[~has_per | per_ok]
        if "PBR" in df.columns:
            pbr_ok = (df["PBR"] >= p["pbr_min"]) & (df["PBR"] <= p["pbr_max"])
            df = df[pbr_ok]

        # MIX
        if "PER" in df.columns and "PBR" in df.columns:
            both = df["PER"].notna() & df["PBR"].notna()
            df.loc[both, "MIX"] = df.loc[both, "PER"] * df.loc[both, "PBR"]
            has_mix = df["MIX"].notna() if "MIX" in df.columns else pd.Series(False, index=df.index)
            if has_mix.any():
                df = df[~has_mix | (df["MIX"] <= p["mix_max"])]

        print(f"  {before} → {len(df)} 銘柄")

        if df.empty:
            return df

        # ── Composite score ──
        def _ni(s):
            mn, mx = s.min(), s.max()
            return 1.0 - (s - mn) / (mx - mn) if mx > mn else pd.Series(0.5, index=s.index)

        def _n(s):
            mn, mx = s.min(), s.max()
            return (s - mn) / (mx - mn) if mx > mn else pd.Series(0.5, index=s.index)

        vs = pd.Series(0.0, index=df.index)
        nv = 0
        for col in ("MIX", "PER", "PBR"):
            if col in df.columns and df[col].notna().any():
                vs += _ni(df[col].fillna(df[col].max()))
                nv += 1
        if "CashRatio" in df.columns and df["CashRatio"].notna().any():
            vs += _n(df["CashRatio"].fillna(0))
            nv += 1
        df["ValueScore"] = (vs / max(nv, 1)).round(2)

        # catalyst bonus
        cat = pd.Series(0.0, index=df.index)
        if "EarningsDate" in df.columns:
            cat = df["EarningsDate"].apply(lambda x: 0.1 if pd.notna(x) and x else 0.0)

        # 40% value + 40% tech + 15% volume + 5% catalyst
        df["Score"] = (
            df["ValueScore"] * 0.40
            + df["TechScore"] * 0.40
            + df["Vol_sc"] * 0.15
            + cat * 0.05
        ).round(3)

        df = df.sort_values("Score", ascending=False).head(self.p["top_n"])
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

        tech = self._run_technical_pass(liquid)
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
        "PER": "PER", "PBR": "PBR", "MIX": "MIX",
        "MarketCap_B": "時価総額(億)", "CashRatio": "現金%",
        "RSI": "RSI", "TechScore": "Tech", "VolRatio": "出来高比",
        "Va_latest": "代金(億)", "Va_avg5": "5d平均", "Va_avg20": "20d平均",
        "ValueScore": "Value", "Score": "総合",
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
        "=" * 130,
        "  VALUE-REVERSAL SCREENER RESULTS",
        "=" * 130, "",
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
        "-" * 130,
        "MIX = PER x PBR  (Graham基準 < 22.5, 許容上限 30)",
        "Tech = RSI反転 + MACDクロス + MA転換 + 底値反発",
        "総合 = Value 40% + Tech 40% + 出来高 15% + カタリスト 5%",
        "",
    ])
    return "\n".join(lines)
