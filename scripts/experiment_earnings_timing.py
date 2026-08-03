#!/usr/bin/env python3
"""Pre-earnings drift, hedged with the liquid market basket.

Frozen in docs/PREREGISTER_EARNINGS_TIMING.md.

The seven families before this one died the same way: the alpha sat on the short
side, in names that cannot be borrowed. Here the alpha is deliberately on the
LONG leg (stocks approaching their expected announcement) and the short leg is
just the equal-weight liquid borrowable universe acting as a hedge, so the borrow
constraint has nothing to bite on.

The J-Quants earnings-calendar endpoint has no history (it ignores every
parameter and returns only today's 79 rows), so the expected announcement date is
rebuilt from last year's actual DiscDate for the same fiscal quarter — which uses
no current-year information.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_value_event_v1 import load_fins
from trading.jp_intraday.daily_model import load_panel_cached

WINDOW_LO, WINDOW_HI = 1, 5          # 予想発表日まで N 営業日
DELAY_DAYS = 3                        # 副仕様B: 何営業日過ぎたら「遅延」とみなすか
COST_BPS, MIN_NAMES = 1.0, 5
SELECTION = ("2022-01-01", "2024-12-31")
CONFIRM = ("2025-01-01", "2099-12-31")
OUT = Path("data/jp_earnings_timing")
QUARTERS = ("1Q", "2Q", "3Q", "FY")


def announcements() -> pd.DataFrame:
    """(symbol, quarter, fiscal-year) -> actual announcement date."""
    f = load_fins()
    f = f[f["DocType"].astype(str).str.contains("FinancialStatements", na=False)].copy()
    f["date"] = pd.to_datetime(f["DiscDate"], errors="coerce")
    f["symbol"] = f["Code"].astype(str).str[:4]
    f["q"] = f["DocType"].astype(str).str.extract(r"^(1Q|2Q|3Q|FY)", expand=False)
    f["fy"] = pd.to_datetime(f["CurFYEn"], errors="coerce")
    f = f.dropna(subset=["date", "symbol", "q", "fy"])
    f = f.sort_values(["symbol", "q", "fy", "date"])
    return f.drop_duplicates(["symbol", "q", "fy"], keep="first")[
        ["symbol", "q", "fy", "date"]]


def expected_dates(ann: pd.DataFrame, sessions: pd.Index) -> pd.DataFrame:
    """Last year's same-quarter announcement + 364d, rolled to a session.

    Uses only the prior fiscal year, so nothing from the current year leaks in.
    """
    a = ann.sort_values(["symbol", "q", "fy"]).copy()
    g = a.groupby(["symbol", "q"], sort=False)
    a["prev_date"] = g["date"].shift(1)
    a["prev_fy"] = g["fy"].shift(1)
    gap = (a["fy"] - a["prev_fy"]).dt.days
    a = a[gap.between(300, 430)].copy()          # 連続する年度のみ
    raw = a["prev_date"] + pd.Timedelta(days=364)
    idx = sessions.searchsorted(raw, side="left")
    ok = idx < len(sessions)
    a = a[ok].copy()
    a["expected"] = sessions[idx[ok]]
    return a[["symbol", "q", "fy", "date", "expected"]]


def basket_series(panel: pd.DataFrame, exp: pd.DataFrame, sessions: pd.Index,
                  mode: str) -> pd.DataFrame:
    """Daily equal-weight basket return minus the equal-weight liquid market."""
    pos = pd.Series(range(len(sessions)), index=sessions)
    exp = exp.copy()
    exp["e_i"] = exp["expected"].map(pos)
    exp["a_i"] = exp["date"].map(pos)
    # ★パネルの symbol は5桁("13010")、fins 側は4桁("1301")。揃えないと交差がゼロになる
    # （保有構造データでも同じ罠を踏んだ）。パネルは sym4 列を持っている。
    key = "sym4" if "sym4" in panel.columns else "symbol"
    pv = panel.assign(_k=panel[key].astype(str).str[:4])
    ret = pv.pivot_table(index="date", columns="_k", values="ret", aggfunc="mean")
    liquid = ret.notna()
    rows, prev = [], set()
    for i, day in enumerate(sessions):
        if day not in ret.index:
            continue
        if mode == "pre":
            # 予想発表日まで 1..5 営業日、かつ**まだ発表されていない**
            m = exp[(exp["e_i"] - i).between(WINDOW_LO, WINDOW_HI)
                    & (exp["a_i"].isna() | (exp["a_i"] > i))]
        else:
            # 予想日を DELAY_DAYS 以上過ぎてまだ未発表
            m = exp[((i - exp["e_i"]) >= DELAY_DAYS)
                    & ((i - exp["e_i"]) < DELAY_DAYS + 10)
                    & (exp["a_i"].isna() | (exp["a_i"] > i))]
        names = set(m["symbol"]) & set(ret.columns[liquid.loc[day].values])
        if len(names) < MIN_NAMES:
            prev = set()
            continue
        r = ret.loc[day, list(names)].dropna()
        mkt = ret.loc[day].dropna()
        if r.empty or mkt.empty:
            prev = set()
            continue
        turn = len(names ^ prev) / max(len(names), 1)
        gross = float(r.mean() - mkt.mean())
        sign = 1.0 if mode == "pre" else -1.0
        rows.append({"date": day, "n": len(r), "turnover": turn,
                     "gross": sign * gross,
                     "net": sign * gross - turn * 2 * COST_BPS / 1e4})
        prev = names
    return pd.DataFrame(rows)


def full_calendar_sharpe(df: pd.DataFrame, sessions: pd.Index, lo: str, hi: str,
                         col: str = "net") -> dict:
    """稼働しない日を0リターンとして全暦日で測る（資金が遊ぶ前提）.

    ★当初は稼働日だけで年率化していたが、それは「休んでいる日に資金を別戦略へ
    回せる」前提の数字。単独運用なら全暦日で測るのが正しく、稼働率が4割なので
    Sharpe はおよそ √0.4 倍になる。両方を報告して読み手が選べるようにする。
    """
    idx = sessions[(sessions >= lo) & (sessions <= hi)]
    v = df.set_index("date")[col].reindex(idx).fillna(0.0)
    if len(v) < 50 or v.std() == 0:
        return {"sharpe": None}
    eq = (1 + v).cumprod()
    return {"sessions": int(len(v)), "active_ratio": round(float((v != 0).mean()), 3),
            "sharpe": round(float(v.mean() / v.std() * 252 ** .5), 3),
            "ann_return": round(float(v.mean() * 252), 4),
            "max_drawdown": round(float((eq / eq.cummax() - 1).min()), 4)}


def stats(df: pd.DataFrame, lo: str, hi: str) -> dict:
    d = df[df["date"].between(lo, hi)]
    r = d["net"].dropna()
    if len(r) < 50 or r.std() == 0:
        return {"days": int(len(r)), "sharpe": None}
    by = d.assign(y=d["date"].dt.year).groupby("y")["net"].mean() * 252
    return {"days": int(len(r)), "sharpe": round(float(r.mean() / r.std() * 252 ** .5), 3),
            "ann_net": round(float(r.mean() * 252), 4),
            "ann_gross": round(float(d["gross"].mean() * 252), 4),
            "median_names": int(d["n"].median()), "mean_turnover": round(float(d["turnover"].mean()), 3),
            "negative_years": int((by < 0).sum()),
            "by_year": {int(k): round(float(v), 4) for k, v in by.items()}}


def main() -> None:
    panel = load_panel_cached(min_value_yen=1e9)
    panel = panel[panel["shortable"].fillna(True) & ~panel["short_restricted"].fillna(False)]
    sessions = pd.Index(sorted(panel["date"].unique()))
    exp = expected_dates(announcements(), sessions)
    out = {"spec": {"window": [WINDOW_LO, WINDOW_HI], "cost_bps_side": COST_BPS,
                    "min_names": MIN_NAMES, "delay_days": DELAY_DAYS},
           "events_with_expected_date": int(len(exp)),
           "note": "earnings-calendar endpoint has no history; expected date rebuilt "
                   "from last year's same-quarter DiscDate + 364d"}
    for mode, label in [("pre", "A_pre_earnings"), ("delay", "B_delayed_announcement")]:
        s = basket_series(panel, exp, sessions, mode)
        out[label] = {"selection": stats(s, *SELECTION),
                      "confirmation": stats(s, *CONFIRM),
                      "selection_full_calendar": full_calendar_sharpe(s, sessions, *SELECTION),
                      "confirmation_full_calendar": full_calendar_sharpe(s, sessions, *CONFIRM)}
        out[label]["convention_note"] = (
            "selection/confirmation は稼働日のみで年率化（事前登録の実装）。"
            "*_full_calendar は休止日を0として全暦日で測ったもの。単独運用なら後者。")
        if not s.empty:
            s.to_csv(OUT / f"series_{mode}.csv", index=False) if OUT.exists() else None
        print(label, json.dumps(out[label], ensure_ascii=False), flush=True)
    a = out["A_pre_earnings"]["selection"]
    failed = []
    if (a.get("sharpe") or -9) < 1.0:
        failed.append("selection_sharpe_lt_1.0")
    if a.get("negative_years", 9) > 1:
        failed.append("selection_negative_years_gt_1")
    c = out["A_pre_earnings"]["confirmation"]
    cfailed = []
    if (c.get("sharpe") or -9) < 1.0:
        cfailed.append("confirmation_sharpe_lt_1.0")
    if c.get("negative_years", 9) > 1:
        cfailed.append("confirmation_negative_years_gt_1")
    out["failed_criteria"] = {"selection": failed, "confirmation": cfailed}
    out["decision"] = ("NO_GO" if (failed or cfailed)
                       else "PENDING_FULL_RISK_TESTS")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str),
                                      encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
