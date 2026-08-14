#!/usr/bin/env python3
"""Crash dip-buy: long-only, quote-free, judged on excess return over the market.

Specification frozen in docs/PREREGISTER_CRASH_DIPBUY.md. Do not edit thresholds,
holding length, name count or the score after reading the numbers.

Long-only by design: the last four families all died because their alpha sat in
stocks that cannot be borrowed. Beta is the obvious failure mode here instead, so
the verdict is taken on **excess return over an equal-weight market book held for
the same sessions**, never on raw return.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trading.jp_intraday.daily_model import load_panel_cached
from trading.jp_intraday.extra_features import attach_extra_features
from trading.jp_intraday.flow_features import attach_flow_features
from trading.jp_intraday.futures_context import build_overnight_features

TRIGGER, HOLD, NAMES = -0.030, 10, 20
CAPITAL, COST_ROUND_TRIP, LOT = 2e7, 0.0020, 100
MIN_EPISODES, MAX_TOP_SHARE = 30, .20
SCORE = [("fall_5d", -1), ("spike_1d", -1), ("xt_ssr_to_so", +1),
         ("flow_close_z", -1), ("equity_ratio", +1)]
OUT = Path("data/jp_crash_dipbuy")


def _z(frame: pd.DataFrame, col: str) -> pd.Series:
    g = frame.groupby("date")[col]
    return ((frame[col] - g.transform("mean")) / g.transform("std").replace(0, np.nan)).clip(-5, 5)


def attach_equity_ratio(p: pd.DataFrame) -> pd.DataFrame:
    """自己資本比率 Eq/TA を、その日までに開示済みの直近決算から付ける（PIT）."""
    from scripts.run_value_event_v1 import load_fins
    f = load_fins()
    f["disc_date"] = pd.to_datetime(f["DiscDate"], errors="coerce")
    f["sym4"] = f["Code"].astype(str).str[:4]
    for c in ("Eq", "TA"):
        f[c] = pd.to_numeric(f.get(c), errors="coerce")
    f = f.dropna(subset=["disc_date", "sym4"]).sort_values(
        [c for c in ["sym4", "disc_date", "DiscTime", "DiscNo"] if c in f])
    f[["Eq", "TA"]] = f.groupby("sym4", sort=False)[["Eq", "TA"]].ffill()
    f = f.drop_duplicates(["sym4", "disc_date"], keep="last")
    f["equity_ratio"] = f["Eq"] / f["TA"].replace(0, np.nan)
    f = f.dropna(subset=["equity_ratio"]).sort_values("disc_date")
    left = p.copy()
    left["sym4"] = left["symbol"].astype(str).str[:4]
    merged = pd.merge_asof(left.sort_values("date"),
                           f[["sym4", "disc_date", "equity_ratio"]],
                           left_on="date", right_on="disc_date", by="sym4",
                           direction="backward", allow_exact_matches=False)
    return merged.drop(columns=["disc_date"])


def build(min_value_yen: float = 1e9) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    p = load_panel_cached(min_value_yen=min_value_yen)
    p = attach_extra_features(attach_flow_features(p))
    p = p.sort_values(["symbol", "date"])
    p = attach_equity_ratio(p)
    # fall_5d must be computed on the *unfiltered* history? The panel is already
    # liquidity-filtered, so use the panel's own close series but require the
    # 5-session window to be calendar-plausible (guards removed-row stitching).
    g = p.groupby("symbol", sort=False)
    prev5 = g["close"].shift(5)
    span = (p["date"] - g["date"].shift(5)).dt.days
    p["fall_5d"] = (p["close"] / prev5 - 1).where(span.between(5, 12))
    p["spike_1d"] = p["ret"]          # D日当日の下げ＝短期スパイク
    mkt = p.groupby("date")["ret"].mean()          # equal-weight market, close-to-close
    fut = pd.concat([pd.read_parquet(f) for f in
                     sorted(glob.glob("data/jp_derivatives/futures_*.parquet"))],
                    ignore_index=True)
    ov = build_overnight_features(fut.drop_duplicates(["Date", "Code"]))
    ov.index = pd.to_datetime(ov.index)
    return p, mkt, ov["dow_night"]


def path_return(bars: pd.DataFrame, tp_mult: float, sl_mult: float, sigma: float) -> float:
    """Adjusted-price path return with volatility-scaled TP/SL.

    Conservative by construction: when a session's range contains both barriers we
    cannot tell which printed first on a daily bar, so we assume the stop did.
    A gap through a barrier fills at that session's open, not at the barrier.
    """
    entry = bars["open"].iloc[0]
    if not np.isfinite(entry) or entry <= 0 or not np.isfinite(sigma) or sigma <= 0:
        return np.nan
    tp, sl = entry * (1 + tp_mult * sigma), entry * (1 - sl_mult * sigma)
    for i in range(len(bars)):
        o, h, l = bars["open"].iloc[i], bars["high"].iloc[i], bars["low"].iloc[i]
        if i > 0 and np.isfinite(o):
            if o <= sl:
                return o / entry - 1
            if o >= tp:
                return o / entry - 1
        hit_sl = np.isfinite(l) and l <= sl
        hit_tp = np.isfinite(h) and h >= tp
        if hit_sl:                      # stop wins ties — never the optimistic side
            return sl / entry - 1
        if hit_tp:
            return tp / entry - 1
    return bars["close"].iloc[-1] / entry - 1


def episodes(p, mkt, dow, trigger, hold, names, us_stabilized=False,
             use_barriers=True, entry_px="open") -> pd.DataFrame:
    sessions = pd.Index(sorted(p["date"].unique()))
    crash_days = [d for d in sessions if mkt.get(d, 0) <= trigger]
    px_all = p.set_index(["date", "symbol"])
    rows = []
    for d in crash_days:
        i = sessions.get_loc(d)
        if i + 1 + hold >= len(sessions):
            continue
        entry_d, exit_d = sessions[i + 1], sessions[i + 1 + hold]
        if us_stabilized and not (dow.get(entry_d, np.nan) > -0.010):
            continue
        sel = p[p["date"].eq(d)].dropna(subset=["fall_5d"]).copy()
        if sel.empty:
            continue
        score = pd.Series(0.0, index=sel.index)
        for col, sign in SCORE:
            if col in sel.columns:
                score = score + sign * _z(sel, col).fillna(0.0)
        sel["_s"] = score
        picks = sel.nlargest(names, "_s")["symbol"].tolist()
        try:
            entry = px_all.loc[entry_d]
            exit_ = px_all.loc[exit_d]
        except KeyError:
            continue
        budget = CAPITAL / names
        window = sessions[i + 1: i + 2 + hold]
        rets, weights = [], []
        for s in picks:
            if s not in entry.index:
                continue
            # 未調整価格は単元数の決定にのみ使う（調整済価格は建値ではない）。
            raw = entry.loc[s, "raw_open"]
            raw = entry.loc[s, "open"] if not np.isfinite(raw) else raw
            if not np.isfinite(raw) or raw <= 0:
                continue
            if entry_px == "vwap":
                va, vo = entry.loc[s, "value"], entry.loc[s, "raw_volume"]
                if np.isfinite(va) and np.isfinite(vo) and vo > 0:
                    raw = va / vo
            units = int(budget // (raw * LOT))
            if units < 1:
                continue
            bars = p[p["symbol"].eq(s) & p["date"].isin(window)].sort_values("date")
            if bars.empty:
                continue
            if entry_px == "vwap":
                # 実VWAP = 売買代金/出来高（未調整）。経路は調整済で見るので、
                # その日の 調整済終値/未調整終値 を係数にして同じ基準へ移す。
                bars = bars.copy()
                raw_c, adj_c = entry.loc[s, "raw_close"], bars["close"].iloc[0]
                if np.isfinite(raw_c) and raw_c > 0 and np.isfinite(adj_c):
                    bars.iloc[0, bars.columns.get_loc("open")] = raw * (adj_c / raw_c)
            if use_barriers:
                sigma = entry.loc[s, "ivol"] * np.sqrt(hold)
                r = path_return(bars, 2.0, 1.5, sigma)
            else:
                r = bars["close"].iloc[-1] / bars["open"].iloc[0] - 1
            if not np.isfinite(r):
                continue
            notional = units * LOT * raw
            rets.append(r * notional)
            weights.append(notional)
        if not weights:
            continue
        book = float(np.sum(weights))
        gross = float(np.sum(rets)) / book
        mkt_leg = float((1 + mkt.loc[entry_d:exit_d].iloc[1:]).prod() - 1)
        rows.append({"crash_date": d, "entry": entry_d, "exit": exit_d,
                     "n_filled": len(weights), "deployed_yen": book,
                     "gross": gross, "net": gross - COST_ROUND_TRIP,
                     "market": mkt_leg, "excess": gross - COST_ROUND_TRIP - mkt_leg})
    return pd.DataFrame(rows)


def report(ep: pd.DataFrame) -> dict:
    if ep.empty:
        return {"episodes": 0}
    x = ep["excess"]
    pos = x[x > 0].sum()
    return {"episodes": int(len(ep)),
            "excess_mean": round(float(x.mean()), 4),
            "excess_median": round(float(x.median()), 4),
            "win_rate": round(float((x > 0).mean()), 4),
            "top_episode_profit_share": round(float(x.max() / pos), 4) if pos > 0 else None,
            "excess_worst": round(float(x.min()), 4),
            "raw_net_mean": round(float(ep["net"].mean()), 4),
            "market_mean": round(float(ep["market"].mean()), 4),
            "median_filled": int(ep["n_filled"].median())}


def main() -> None:
    p, mkt, dow = build()
    out = {"spec": {"trigger": TRIGGER, "hold": HOLD, "names": NAMES,
                    "cost_round_trip": COST_ROUND_TRIP, "capital": CAPITAL},
           "primary": {}, "sensitivity": {}}
    main_ep = episodes(p, mkt, dow, TRIGGER, HOLD, NAMES)
    out["primary"] = report(main_ep)
    r = out["primary"]
    failed = []
    if r.get("episodes", 0) < MIN_EPISODES:
        failed.append("episodes_lt_30")
    if not (r.get("excess_median") or -1) > 0:
        failed.append("excess_median_not_positive")
    if (r.get("top_episode_profit_share") or 1) >= MAX_TOP_SHARE:
        failed.append("top_episode_share_ge_20pct")
    if not (r.get("excess_mean") or -1) > 0:
        failed.append("excess_mean_not_positive")
    if (r.get("win_rate") or 0) < .50:
        failed.append("win_rate_lt_50pct")
    out["failed_criteria"] = failed
    out["decision"] = "NO_GO" if failed else "PENDING_FULL_RISK_TESTS"

    for label, kw in [("trigger_-4pct", dict(trigger=-.04)), ("trigger_-5pct", dict(trigger=-.05)),
                      ("hold_5", dict(hold=5)), ("hold_20", dict(hold=20)),
                      ("names_10", dict(names=10)), ("names_40", dict(names=40)),
                      ("us_stabilized", dict(us_stabilized=True)),
                      ("no_barriers", dict(use_barriers=False)),
                      ("vwap_entry", dict(entry_px="vwap"))]:
        kw = {"trigger": TRIGGER, "hold": HOLD, "names": NAMES, **kw}
        out["sensitivity"][label] = report(episodes(p, mkt, dow, **kw))

    if not main_ep.empty:
        by_year = main_ep.assign(y=main_ep["crash_date"].dt.year).groupby("y")["excess"]
        out["by_year"] = {int(k): round(float(v), 4) for k, v in by_year.mean().items()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str),
                                      encoding="utf-8")
    if not main_ep.empty:
        main_ep.to_csv(OUT / "episodes.csv", index=False)
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
