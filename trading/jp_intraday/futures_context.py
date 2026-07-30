"""Index-futures overnight context (先物) for JP intraday research.

JPX index futures trade a day session (M*) and a night session (E*, ~16:30 JST
to ~06:00 JST next morning) that spans the entire US session. So for a given
futures trade date D, the night-session return ``EC/MC - 1`` is the overnight
(US-driven) move, realised before the *next* cash open. That makes it a clean,
market-level overnight factor — cleaner than reconstructing it from cash gaps.

DJIAF gives the Dow's move (a US proxy) and NKVIF the Nikkei vol index (regime),
both without any non-JPX data source.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def front_month(futures: pd.DataFrame, product: str) -> pd.DataFrame:
    """Nearest non-expired contract per date (ties broken by open interest)."""
    f = futures[futures["ProdCat"].astype(str).eq(product)].copy()
    f["Date"] = pd.to_datetime(f["Date"])
    f["LTD"] = pd.to_datetime(f["LTD"])
    f = f[f["LTD"] >= f["Date"]]
    f = f.sort_values(["Date", "LTD", "OI"], ascending=[True, True, False])
    return f.groupby("Date", as_index=False).first()


def overnight_factor(futures: pd.DataFrame, product: str = "NK225F") -> pd.DataFrame:
    """Per cash day D: the futures night-session (overnight) and day returns.

    **セッションの並び順（2026-07-30 実データで訂正）**: 1レコード(取引日 D)は
    「ナイト（D-1夕17:00 → D朝06:00, US時間帯を含む）」が**先**で、その後に
    「日中（08:45→15:45, C == Settle）」が来る。実証: 日通し始値 O == EO が99.9%、
    EC_D/C_{D-1}-1 と**同日**の現物ギャップの相関 +0.264（翌日ギャップとは −0.016）。
    したがって:
      overnight (D朝06:00に確定・D寄付き前に使える) = ``EC_D / C_{D-1} - 1``
      day session (D 15:45に確定)                   = ``C_D / EO_D - 1``
    旧実装は ``EC_D/C_D - 1`` を D+1 に寄せており、これは「D の日中リターンの符号反転」
    をほぼ意味する別物だった（真のオーバーナイト因子は一度も使われていなかった）。
    本番の ensemble_core は FUT_FEATURES を使わないため過去の本番成績には影響しない。
    (M* 日中列はこのデータセットでは 100% 欠損のため O/C/EO/EC を使う。)
    """
    f = front_month(futures, product).sort_values("Date")
    close = f["C"].replace(0, np.nan)
    eo = f["EO"].replace(0, np.nan)
    ec = f["EC"].replace(0, np.nan)
    out = pd.DataFrame({"cash_day": f["Date"].to_numpy()})
    # ナイトは前営業日の日中引け → 当日朝06:00（当日の寄付き前に確定＝PIT合法）
    out["night_ret"] = (ec.to_numpy() / close.shift(1).to_numpy()) - 1.0
    # 日中セッションは当日のナイト明け（EO）→ 引け
    out["day_ret"] = (close.div(eo).sub(1)).to_numpy()
    out["settle"] = f["Settle"].to_numpy()
    out["oi"] = f["OI"].to_numpy()
    return out.dropna(subset=["cash_day"]).set_index("cash_day")


def build_overnight_features(futures: pd.DataFrame) -> pd.DataFrame:
    """Combined overnight panel keyed by cash day: Nikkei/TOPIX/Dow/VI."""
    nk = overnight_factor(futures, "NK225F")[["night_ret", "day_ret"]].rename(
        columns={"night_ret": "nk_night", "day_ret": "nk_day"})
    tp = overnight_factor(futures, "TOPIXF")[["night_ret"]].rename(
        columns={"night_ret": "topix_night"})
    dj = overnight_factor(futures, "DJIAF")[["night_ret", "day_ret"]].rename(
        columns={"night_ret": "dow_night", "day_ret": "dow_day"})
    vi = overnight_factor(futures, "NKVIF")[["settle"]].rename(columns={"settle": "nkvi"})
    return nk.join([tp, dj, vi], how="outer").sort_index()
