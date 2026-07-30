"""Trainable cross-sectional daily model (overnight gap + 需給 + 先物).

Predicts the tradable open->close return from features known at the open, forms a
dollar-neutral long/short book, and evaluates it walk-forward (train on past
years, test the next). Training runs locally.

Features build on the robust overnight-gap reversal and let the model combine it
with:
  * gap magnitude / prior-day dynamics / realised vol / liquidity,
  * (optional) index-futures overnight moves (先物/US via NK225F & DJIAF night
    session) so the model can tell a whole-market gap (may persist) from an
    idiosyncratic one (tends to revert),
  * (optional) sector short-selling ratio (需給).
Everything is point-in-time: only the residual gap and lagged variables enter.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .daily_gap import _sharpe, build_gap_panel

# All features must be known at the OPEN (point-in-time). vol uses the LAGGED
# 20d vol (ivol), never vol20 which includes today's not-yet-known close.
# amihud20 (出来高由来の非流動性, lagged) is the one volume feature that survived
# IS-pick -> OOS confirmation (+0.2-0.3 net Sharpe at 3/7bps, adversarially verified).
BASE_FEATURES = ["residual_gap", "gap_abs", "prev_intraday", "prev_resid_gap",
                 "ivol", "liq_rank", "amihud20"]
FUT_FEATURES = ["nk_night", "dow_night", "gap_x_nk"]


# Sections that are ordinary listed companies (individual stocks). Everything else
# (その他 = ETF/ETN/上場投信/REIT/インフラファンド, plus TOKYO PRO MARKET) is excluded.
_INDIVIDUAL_SECTIONS = {"プライム", "スタンダード", "グロース"}


def load_master(path: str = "data/jp_daily_history/master.parquet") -> pd.DataFrame | None:
    """Code -> name / 33-sector / market, with an is_fund flag (投資信託系を除外用)."""
    try:
        m = pd.read_parquet(path)
    except (OSError, FileNotFoundError):
        return None
    m = m.rename(columns={"Code": "symbol", "CoName": "name", "S33Nm": "sector",
                          "MktNm": "market", "S33": "s33_code", "MrgnNm": "margin"})
    m["symbol"] = m["symbol"].astype(str)
    m["is_fund"] = ~m["market"].isin(_INDIVIDUAL_SECTIONS)
    m["shortable"] = m["margin"].eq("貸借")  # 制度信用でショート可能
    scale_map = {"TOPIX Core30": 5, "TOPIX Large70": 4, "TOPIX Mid400": 3,
                 "TOPIX Small 1": 2, "TOPIX Small 2": 1, "-": 0}
    m["scale_ord"] = m["ScaleCat"].map(scale_map).fillna(0)
    return m[["symbol", "name", "sector", "s33_code", "market", "is_fund",
              "shortable", "scale_ord"]].drop_duplicates("symbol")


def _load_topix_returns() -> pd.DataFrame | None:
    """TOPIX (0000): close-to-close return (beta factor) + open->close return (hedge leg)."""
    import glob
    files = sorted(glob.glob("data/jp_derivatives/indices_*.parquet"))
    if not files:
        return None
    idx = pd.concat([pd.read_parquet(f, columns=["Date", "Code", "O", "C"]) for f in files],
                    ignore_index=True)
    t = idx[idx["Code"].astype(str) == "0000"].drop_duplicates("Date").copy()
    if t.empty:
        return None
    t["date"] = pd.to_datetime(t["Date"])
    t = t.sort_values("date")
    t["mret"] = t["C"].pct_change()
    t["topix_oc"] = t["C"] / t["O"] - 1
    return t[["date", "mret", "topix_oc"]]


def _load_sector_index_gaps() -> pd.DataFrame | None:
    """Per (date, s33_code): the 33-sector TOPIX index overnight gap (O/prevC − 1).

    Index codes 0040-0060 map to the 33 industry (s33) codes by canonical ordinal
    order — verified empirically via Hungarian assignment on return correlations
    (perfect bijection). PIT: today's index OPEN vs yesterday's index close.
    """
    import glob
    files = sorted(glob.glob("data/jp_derivatives/indices_*.parquet"))
    if not files:
        return None
    idx = pd.concat([pd.read_parquet(f, columns=["Date", "Code", "O", "C"]) for f in files],
                    ignore_index=True).drop_duplicates(["Date", "Code"])
    idx["Code"] = idx["Code"].astype(str)
    sec = idx[(idx["Code"] >= "0040") & (idx["Code"] <= "0060")].copy()
    if sec.empty:
        return None
    sec["date"] = pd.to_datetime(sec["Date"])
    sec = sec.sort_values(["Code", "date"])
    sec["prev_c"] = sec.groupby("Code")["C"].shift(1)
    sec["sector_index_gap"] = sec["O"] / sec["prev_c"] - 1
    master = load_master()
    if master is None:
        return None
    s33_sorted = sorted(c for c in master["s33_code"].dropna().unique() if c != "9999")
    index_sorted = sorted(sec["Code"].unique())
    if len(s33_sorted) != len(index_sorted):
        return None
    code_map = dict(zip(index_sorted, s33_sorted))
    sec["s33_code"] = sec["Code"].map(code_map)
    return sec[["date", "s33_code", "sector_index_gap"]].dropna()


def _load_sector_short_z() -> pd.DataFrame | None:
    """Per (date, 33-sector) short-selling-ratio z-score, lagged 1 day (PIT)."""
    import glob
    files = sorted(glob.glob("data/jp_derivatives/short_ratio_*.parquet"))
    if not files:
        return None
    sr = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    sr = sr.drop_duplicates(["Date", "S33"]).copy()
    sr["date"] = pd.to_datetime(sr["Date"])
    sr = sr.sort_values(["S33", "date"])
    tot = sr["SellExShortVa"] + sr["ShrtWithResVa"] + sr["ShrtNoResVa"]
    ratio = (sr["ShrtWithResVa"] + sr["ShrtNoResVa"]).div(tot.replace(0, np.nan))
    lag = ratio.groupby(sr["S33"]).shift(1)  # published after close -> use prior day
    mean = lag.groupby(sr["S33"]).transform(lambda s: s.rolling(60, min_periods=20).mean())
    std = lag.groupby(sr["S33"]).transform(lambda s: s.rolling(60, min_periods=20).std())
    sr["sector_short_ratio_z"] = (lag - mean).div(std.replace(0, np.nan))
    return sr[["date", "S33", "sector_short_ratio_z"]].rename(columns={"S33": "s33_code"})


_PANEL_CACHE_DIR = "data/cache_panels"
_PANEL_SCHEMA_VERSION = 7  # パネル列を追加/変更したらインクリメント（キャッシュ自動無効化）
_PANEL_INPUT_GLOBS = (
    "data/cache/bars_day_*.parquet", "data/jp_intraday_reference/daily_20260528_20260724.parquet",
    "data/jp_daily_history/daily_adj_*.parquet", "data/jp_daily_history/master.parquet",
    "data/jp_derivatives/indices_*.parquet", "data/jp_derivatives/futures_*.parquet",
    "data/jp_derivatives/short_ratio_*.parquet", "data/jp_intraday_reference/share_snapshots.csv",
    "data/jp_flows/margin_alert_*.parquet",
)


def load_panel_cached(min_value_yen: float = 5e8, markets: tuple | None = None,
                      min_mktcap_yen: float | None = None, max_mktcap_yen: float | None = None,
                      with_futures: bool = True) -> pd.DataFrame:
    """build_daily_features のディスクキャッシュ版（入力ファイルのmtime指紋がキー）.

    コールド~13s → ウォーム~1s。入力データ・パラメータが変わると自動で再構築。
    キャッシュは直近6件のみ保持（重複保存の膨張防止）。
    """
    import glob as _glob
    import hashlib
    import json as _json
    import os
    from pathlib import Path

    files = []
    for pat in _PANEL_INPUT_GLOBS:
        files += sorted(_glob.glob(pat))
    stat = [(f, os.path.getmtime(f), os.path.getsize(f)) for f in files]
    key = _json.dumps([_PANEL_SCHEMA_VERSION, stat, min_value_yen, markets, min_mktcap_yen,
                       max_mktcap_yen, with_futures], default=str)
    h = hashlib.md5(key.encode()).hexdigest()[:16]
    cdir = Path(_PANEL_CACHE_DIR)
    cdir.mkdir(parents=True, exist_ok=True)
    path = cdir / f"panel_{h}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    from trading.jp_intraday.daily_gap import load_existing_daily
    ov = None
    if with_futures:
        try:
            from trading.jp_intraday.futures_context import build_overnight_features
            futs = sorted(_glob.glob("data/jp_derivatives/futures_*.parquet"))
            if futs:
                fut = pd.concat([pd.read_parquet(f) for f in futs], ignore_index=True)
                ov = build_overnight_features(fut.drop_duplicates(["Date", "Code"]))
        except Exception:
            ov = None
    panel = build_daily_features(load_existing_daily(), min_value_yen=min_value_yen,
                                 futures_overnight=ov, markets=markets,
                                 min_mktcap_yen=min_mktcap_yen, max_mktcap_yen=max_mktcap_yen)
    panel.to_parquet(path, index=False)
    old = sorted(cdir.glob("panel_*.parquet"), key=os.path.getmtime)[:-6]
    for f in old:
        f.unlink(missing_ok=True)
    return panel


def _load_short_restrictions() -> pd.DataFrame | None:
    """(date, symbol) の売り建て規制フラグ（PIT: 公表翌日から適用・7日キャリー）.

    J-Quants margin_alert（日々公表・規制銘柄残高）の PubReason から、売り建てが
    実務上できない/危険な銘柄を判定:
      Restricted(取引所規制=増担保等) / RestrictedByJSF(日証金貸株申込制限=売建不能の主犯)。
    指定中は定期的に再公表されるため、各公表から7暦日先までフラグを維持（週末・祝日跨ぎ）。
    """
    import glob
    files = sorted(glob.glob("data/jp_flows/margin_alert_*.parquet"))
    if not files:
        return None
    a = pd.concat([pd.read_parquet(f, columns=["PubDate", "Code", "PubReason"])
                   for f in files], ignore_index=True)
    hard = a["PubReason"].str.contains("'Restricted': '1'") | \
        a["PubReason"].str.contains("'RestrictedByJSF': '1'")
    a = a[hard].copy()
    if a.empty:
        return None
    a["pub"] = pd.to_datetime(a["PubDate"])
    a["symbol"] = a["Code"].astype(str)
    rows = []
    for off in range(1, 8):                      # 公表翌日〜+7暦日にフラグ適用
        r = a[["symbol", "pub"]].copy()
        r["date"] = r["pub"] + pd.Timedelta(days=off)
        rows.append(r[["date", "symbol"]])
    out = pd.concat(rows, ignore_index=True).drop_duplicates()
    out["short_restricted"] = True
    return out


def _load_share_snapshots() -> pd.DataFrame | None:
    """PIT株数スナップショット（symbol4桁, known_at=開示日, shares）。時価総額フィルタ用."""
    try:
        s = pd.read_csv("data/jp_intraday_reference/share_snapshots.csv")
    except (OSError, FileNotFoundError):
        return None
    s["known_at"] = pd.to_datetime(s["known_at"])
    s["sym4"] = s["symbol"].astype(str)
    return s.sort_values("known_at")[["sym4", "known_at", "shares"]]


def build_daily_features(daily: pd.DataFrame, min_value_yen: float = 5e8,
                         futures_overnight: pd.DataFrame | None = None,
                         individual_only: bool = True,
                         markets: tuple | None = None,
                         min_mktcap_yen: float | None = None,
                         max_mktcap_yen: float | None = None) -> pd.DataFrame:
    """Feature panel with optional universe constraints.

    markets: 例 ("プライム",) — 現行masterの市場区分でフィルタ（区分の履歴は取得不能
      なため現在区分ベース。上場廃止銘柄は区分不明で除外される点に留意=軽い生存バイアス）。
    min/max_mktcap_yen: 時価総額バンド（PIT株数×前日終値。株数開示は2021-04以降のため
      それ以前の日付は各銘柄の最初の開示値で近似）。
    """
    p = build_gap_panel(daily, min_value_yen=min_value_yen)
    # 市場の営業日インデックス（フィルタ前の全銘柄の日付集合＝取引所カレンダー相当）
    _dcol = "Date" if "Date" in daily.columns else "date"
    daily_dates = pd.to_datetime(daily[_dcol]).unique()
    # Restrict to individual stocks (drop ETF/ETN/REIT/投信 & PRO MARKET). Symbols
    # absent from the current master are DELISTED individual names -> keep them
    # (is_fund NaN -> False) so the backtest is not survivorship-biased.
    master = load_master()
    if master is not None:
        p = p.merge(master[["symbol", "name", "sector", "s33_code", "is_fund",
                            "shortable", "scale_ord", "market"]], on="symbol", how="left")
        if individual_only:
            p = p[p["is_fund"] != True]  # noqa: E712 — keep NaN (delisted) and False
        if markets:
            p = p[p["market"].isin(markets)]
    snaps = _load_share_snapshots()
    if snaps is not None:                       # mktcap_yen は常設列（ライブの時価総額フロア用）
        p["sym4"] = p["symbol"].astype(str).map(lambda s: s[:-1] if len(s) == 5 else s)
        p = p.sort_values("date")
        p = pd.merge_asof(p, snaps, left_on="date", right_on="known_at",
                          by="sym4", direction="backward")
        # 2021-04以前は最初の開示値で近似（株数は緩やかにしか変わらない前提）
        first = snaps.drop_duplicates("sym4", keep="first").set_index("sym4")["shares"]
        p["shares"] = p["shares"].fillna(p["sym4"].map(first))
        p["mktcap_yen"] = p["prev_close"] * p["shares"]
        if min_mktcap_yen:
            p = p[p["mktcap_yen"] >= min_mktcap_yen]
        if max_mktcap_yen:
            p = p[p["mktcap_yen"] <= max_mktcap_yen]
        p = p.drop(columns=["known_at"], errors="ignore")
    restr = _load_short_restrictions()
    if restr is not None:                       # 売り建て規制フラグ（PIT・公表翌日から適用）
        p = p.merge(restr, on=["date", "symbol"], how="left")
        p["short_restricted"] = p["short_restricted"].fillna(False).astype(bool)
    else:
        p["short_restricted"] = False
    p["sector"] = p.get("sector", pd.Series(index=p.index, dtype="object")).fillna("unknown")
    p["name"] = p.get("name", pd.Series(index=p.index, dtype="object")).fillna(p["symbol"])
    sh = p["shortable"] if "shortable" in p.columns else pd.Series(True, index=p.index)
    p["shortable"] = sh.fillna(True).astype(bool)
    p["scale_ord"] = p.get("scale_ord", pd.Series(index=p.index, dtype="float")).fillna(0)

    def _groll(series: pd.Series, window: int, minp: int, fn: str = "std",
               shift: int = 0) -> pd.Series:
        """groupby(symbol).rolling を Cython 実装で（transform(lambda) の高速置換）."""
        r = getattr(series.groupby(p["symbol"]).rolling(window, min_periods=minp), fn)()
        r = r.reset_index(level=0, drop=True)
        if shift:
            r = r.groupby(p["symbol"]).shift(shift)
        return r

    # Cross-sectional stats recomputed on the (individual-stock) universe.
    p["residual_gap"] = p["overnight_gap"].sub(p.groupby("date")["overnight_gap"].transform("mean"))
    p["sector_resid_gap"] = p["residual_gap"].sub(
        p.groupby(["date", "sector"])["residual_gap"].transform("mean"))
    p["target"] = p["intraday_ret"].sub(p.groupby("date")["intraday_ret"].transform("mean"))

    # Per-symbol features (use the cleaned residual_gap).
    p = p.sort_values(["symbol", "date"])
    g = p.groupby("symbol", sort=False)
    p["prev_intraday"] = g["intraday_ret"].shift(1)
    p["prev_resid_gap"] = g["residual_gap"].shift(1)
    # 空売り価格規制トリガー判定用（前日安値 vs 前日の基準値=前々日終値）
    p["prev_low"] = g["low"].shift(1)
    p["prev_close2"] = g["close"].shift(2)
    # 保有区分別のフォワードリターン（戦略が"取りに行く"リターン。特徴量ではない）
    # overnight: 当日引け→翌日寄り / cc1: 当日引け→翌日引け。シグナルは当日引けまでの情報のみ。
    # 汚染ガード（2026-07-30 R7の検証で強化）: shift(-1)はフィルタ済みパネル上のため、
    # 流動性フィルタ等で翌行が抜けた銘柄では「翌日」が実際には数セッション先になり、
    # 複数日分のリターンが「一晩」として計上されて偽アルファを生む。
    # 旧ガード（暦日差<=4日）では2〜4暦日の飛びを素通りさせ、選択窓の3.57%の行が
    # 真の翌営業日を指していなかった（該当行の平均+60.4bps vs 真値+16.8bps）。
    # 正しい判定は「取引所カレンダー上で真に隣接するセッションか」なので、
    # 全銘柄の日付集合から市場の営業日インデックスを作り、連番+1のみを有効とする。
    _sessions = pd.Index(sorted(daily_dates)) if daily_dates is not None else \
        pd.Index(sorted(p["date"].unique()))
    _sess_no = pd.Series(range(len(_sessions)), index=_sessions)
    _cur_no = p["date"].map(_sess_no)
    _next_no = _cur_no.groupby(p["symbol"]).shift(-1)
    _fwd_ok = _next_no.eq(_cur_no + 1)          # 真の翌営業日のみ
    p["ret_on_fwd"] = (g["open"].shift(-1) / p["close"] - 1).where(_fwd_ok)
    p["ret_cc_fwd"] = (g["close"].shift(-1) / p["close"] - 1).where(_fwd_ok)
    p["gap_abs"] = p["residual_gap"].abs()
    cc = g["close"].pct_change(fill_method=None)
    p["ret"] = cc
    p["vol20"] = _groll(cc, 20, 10, "std")
    p["vol20_floor"] = p["vol20"].clip(lower=0.005)
    # PIT inverse-vol for risk-parity sizing (exclude today's unknown close).
    p["ivol"] = _groll(cc, 20, 10, "std", shift=1).clip(lower=0.005)
    p["liq_rank"] = p.groupby("date")["prev_value"].rank(pct=True)
    # Amihud illiquidity (出来高/売買代金ベース), PIT: rolling20 of |ret|/value, lagged 1d.
    amihud = cc.abs() / p["value"].replace(0, np.nan)
    p["amihud20"] = _groll(amihud, 20, 10, "mean", shift=1)
    # Self-normalised gap (gap vs the symbol's own overnight-gap volatility).
    p["gap_vol60"] = _groll(p["residual_gap"], 60, 20, "std").clip(lower=0.005)
    p["gap_z"] = p["residual_gap"] / p["gap_vol60"]
    # Beta vs TOPIX (rolling 60d, lagged 1 day = known at the open).
    # ベクトル化: cov/var を rolling 和で構成（ddof=1・ペア有効数nはpandas cov準拠）。
    topix = _load_topix_returns()
    if topix is not None:
        p = p.merge(topix, on="date", how="left").sort_values(["symbol", "date"])
        x = p["ret"].where(p["mret"].notna())
        y = p["mret"].where(p["ret"].notna())
        n = _groll(x.notna().astype(float), 60, 40, "sum")
        sx, sy = _groll(x, 60, 40, "sum"), _groll(y, 60, 40, "sum")
        sxy = _groll(x * y, 60, 40, "sum")
        cov = (sxy - sx * sy / n) / (n - 1)
        m = p["mret"]
        n2 = _groll(m.notna().astype(float), 60, 40, "sum")
        sm, smm = _groll(m, 60, 40, "sum"), _groll(m * m, 60, 40, "sum")
        var = (smm - sm * sm / n2) / (n2 - 1)
        p["beta"] = (cov / var).groupby(p["symbol"]).shift(1)
    # Sector short-selling-ratio z (需給), merged by 33-sector code.
    short_z = _load_sector_short_z()
    if short_z is not None and "s33_code" in p.columns:
        p = p.merge(short_z, on=["date", "s33_code"], how="left")
    # Sector-INDEX overnight gap -> index-based idiosyncratic gap (2021-09+ coverage).
    sig = _load_sector_index_gaps()
    if sig is not None and "s33_code" in p.columns:
        p = p.merge(sig, on=["date", "s33_code"], how="left")
        p["idio_gap2"] = p["overnight_gap"] - p["sector_index_gap"]
    if futures_overnight is not None:
        fo = futures_overnight.reset_index().rename(columns={"cash_day": "date", "index": "date"})
        fo["date"] = pd.to_datetime(fo["date"])
        cols = [c for c in ("nk_night", "dow_night") if c in fo.columns]
        p = p.merge(fo[["date"] + cols], on="date", how="left")
        for c in cols:  # a missing overnight = no move; keep the stock row
            p[c] = p[c].fillna(0.0)
        if "nk_night" in p.columns:
            p["gap_x_nk"] = p["residual_gap"] * p["nk_night"]
    return p.reset_index(drop=True)


def _ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    return np.linalg.solve(x.T @ x + np.eye(x.shape[1]) * alpha, x.T @ y)


def _ls_weights(frame: pd.DataFrame, pred: pd.Series, quantile: float) -> pd.Series:
    rank = pred.groupby(frame["date"]).rank(pct=True)
    long, short = rank.ge(1 - quantile), rank.le(quantile)
    nl = long.groupby(frame["date"]).transform("sum")
    ns = short.groupby(frame["date"]).transform("sum")
    both = nl.gt(0) & ns.gt(0)
    w = pd.Series(0.0, index=frame.index)
    w.loc[both & long] = 0.5 / nl.loc[both & long]
    w.loc[both & short] = -0.5 / ns.loc[both & short]
    return w


def walk_forward_returns(panel: pd.DataFrame, features: list[str], quantile: float = 0.05,
                         alpha: float = 10.0, cost_bps_side: float = 3.0,
                         gross_leverage: float = 1.0, min_train_years: int = 2) -> pd.DataFrame:
    """Pooled OOS daily returns (date, gross, net) at a chosen gross leverage.

    ``gross_leverage`` scales the dollar-neutral book (1.0 = 0.5 long + 0.5 short
    of capital). Trading cost scales with it too. Positions are opened at the
    open and closed at the close every day — flat overnight, no overnight risk.
    """
    panel = panel.copy()
    panel["year"] = panel["date"].dt.year
    years = sorted(panel["year"].unique())
    daily = []
    for i in range(min_train_years, len(years)):
        test_year = years[i]
        train = panel[panel["year"] < test_year].dropna(subset=features + ["target"])
        test = panel[panel["year"].eq(test_year)].copy()
        if len(train) < 5000 or test.empty:
            continue
        mean = train[features].mean()
        std = train[features].std().replace(0, 1).fillna(1)
        beta = _ridge(((train[features] - mean) / std).to_numpy(), train["target"].to_numpy(), alpha)
        xte = (test[features] - mean) / std
        valid = xte.notna().all(axis=1)
        pred = pd.Series(np.nan, index=test.index)
        pred.loc[valid] = xte.loc[valid].to_numpy() @ beta
        w = _ls_weights(test, pred.fillna(pred.mean()), quantile) * gross_leverage
        gross = (w * test["intraday_ret"]).groupby(test["date"]).sum()
        expo = w.abs().groupby(test["date"]).sum()
        net = gross.sub(expo * 2 * cost_bps_side / 10_000)
        daily.append(pd.DataFrame({"date": gross.index, "gross": gross.values, "net": net.values}))
    return pd.concat(daily, ignore_index=True) if daily else pd.DataFrame(columns=["date", "gross", "net"])


def walk_forward_predictions(panel: pd.DataFrame, features: list[str], alpha: float = 10.0,
                             min_train_years: int = 2, target: str = "demeaned") -> pd.DataFrame:
    """Pooled OOS predictions (date, symbol, pred, intraday_ret) — train past, predict next.

    ``target``: "demeaned" (cross-sectionally demeaned return) or "rank"
    (per-date pct rank − 0.5; kills fat-tail noise, OOS-verified +0.8-0.9 Sh).
    Ridge output does not depend on quantile/leverage/cost, so callers cache this
    once and build many portfolios from it cheaply via ``portfolio_returns``.
    """
    panel = panel.copy()
    panel["year"] = panel["date"].dt.year
    if target == "rank":
        panel["target"] = panel.groupby("date")["intraday_ret"].rank(pct=True) - 0.5
    years = sorted(panel["year"].unique())
    out = []
    for i in range(min_train_years, len(years)):
        test_year = years[i]
        train = panel[panel["year"] < test_year].dropna(subset=features + ["target"])
        test = panel[panel["year"].eq(test_year)].copy()
        if len(train) < 5000 or test.empty:
            continue
        mean = train[features].mean()
        std = train[features].std().replace(0, 1).fillna(1)
        beta = _ridge(((train[features] - mean) / std).to_numpy(), train["target"].to_numpy(), alpha)
        xte = (test[features] - mean) / std
        valid = xte.notna().all(axis=1)
        pred = pd.Series(np.nan, index=test.index)
        pred.loc[valid] = xte.loc[valid].to_numpy() @ beta
        test["pred"] = pred.fillna(pred.mean())
        out.append(test[["date", "symbol", "pred", "intraday_ret"]])
    if out:
        return pd.concat(out, ignore_index=True)
    # 空でも型を保つ（object dateだと下流のmergeが型衝突で落ちる）
    return pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"),
                         "symbol": pd.Series(dtype=object),
                         "pred": pd.Series(dtype=float),
                         "intraday_ret": pd.Series(dtype=float)})


def portfolio_returns(preds: pd.DataFrame, quantile: float = 0.05,
                      gross_leverage: float = 1.0, cost_bps_side: float = 3.0) -> pd.DataFrame:
    """Daily open->close returns from cached predictions (flat overnight)."""
    w = _ls_weights(preds, preds["pred"], quantile) * gross_leverage
    gross = (w * preds["intraday_ret"]).groupby(preds["date"]).sum()
    expo = w.abs().groupby(preds["date"]).sum()
    net = gross.sub(expo * 2 * cost_bps_side / 10_000)
    out = pd.DataFrame({"date": gross.index, "gross": gross.values, "net": net.values})
    return out


def annualized_stats(daily: pd.DataFrame, col: str = "net") -> dict:
    """Annualised return / vol / Sharpe / max drawdown from a daily return series."""
    r = daily[col].astype(float) if len(daily) else pd.Series(dtype=float)
    n = len(r)
    if n == 0:  # 空でもキーを揃える（制約が強すぎてデータ無しの場合など）
        return {"ann_return": 0.0, "ann_vol": 0.0, "sharpe": 0.0,
                "max_drawdown": 0.0, "total_return": 0.0, "days": 0, "win_rate": 0.0}
    ann_ret = float(r.mean() * 252)
    ann_vol = float(r.std(ddof=1) * np.sqrt(252))
    equity = (1 + r).cumprod()
    dd = float((equity / equity.cummax() - 1).min())
    return {"ann_return": ann_ret, "ann_vol": ann_vol,
            "sharpe": ann_ret / ann_vol if ann_vol else 0.0, "max_drawdown": dd,
            "total_return": float(equity.iloc[-1] - 1), "days": n,
            "win_rate": float((r > 0).mean())}


def walk_forward(panel: pd.DataFrame, features: list[str], quantile: float = 0.1,
                 alpha: float = 10.0, cost_bps_side: float = 3.0,
                 min_train_years: int = 2) -> pd.DataFrame:
    """Expanding-window walk-forward: train on all prior years, test the next."""
    panel = panel.copy()
    panel["year"] = panel["date"].dt.year
    years = sorted(panel["year"].unique())
    rows = []
    for i in range(min_train_years, len(years)):
        test_year = years[i]
        train = panel[panel["year"] < test_year].dropna(subset=features + ["target"])
        test = panel[panel["year"].eq(test_year)].copy()
        if len(train) < 5000 or test.empty:
            continue
        mean = train[features].mean()
        std = train[features].std().replace(0, 1).fillna(1)
        xtr = ((train[features] - mean) / std).to_numpy()
        beta = _ridge(xtr, train["target"].to_numpy(), alpha)
        xte = (test[features] - mean) / std
        valid = xte.notna().all(axis=1)
        pred = pd.Series(np.nan, index=test.index)
        pred.loc[valid] = xte.loc[valid].to_numpy() @ beta
        w = _ls_weights(test, pred.fillna(pred.mean()), quantile)
        gross = (w * test["intraday_ret"]).groupby(test["date"]).sum()
        expo = w.abs().groupby(test["date"]).sum()
        net = gross.sub(expo * 2 * cost_bps_side / 10_000)
        rows.append({"test_year": test_year, "train_rows": len(train),
                     "gross_sharpe": _sharpe(gross), "net_sharpe": _sharpe(net),
                     "gross_sum": float(gross.sum()), "net_sum": float(net.sum())})
    out = pd.DataFrame(rows)
    if len(out):
        pooled = out[["gross_sharpe", "net_sharpe"]].mean()
        out.loc[len(out)] = {"test_year": "MEAN", "train_rows": 0,
                             "gross_sharpe": pooled["gross_sharpe"],
                             "net_sharpe": pooled["net_sharpe"],
                             "gross_sum": out["gross_sum"].sum(), "net_sum": out["net_sum"].sum()}
    return out
