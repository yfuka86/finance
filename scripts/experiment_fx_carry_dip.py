#!/usr/bin/env python3
"""Buy the dip in positive-carry FX pairs.

Frozen in docs/PREREGISTER_FX_CARRY_DIP.md.

Unconditional carry died because the +2.3%/yr of swap was eaten by the spot move
(Sharpe 0.138 selection / 0.069 confirmation, DD -22.8%). This turns that failure
around: carry trades die in crashes, so enter *after* one — cheaper entry, and the
swap keeps accruing while waiting.

★The 2011-2025 windows were already consumed by the unconditional carry run, so a
pass here is a **selection-grade** result, not a confirmation. Judging it needs
fresh forward data.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from trading.fx.swap import load_short_rates, rollover_days_series

PAIRS = {"EUR": "EURUSD", "JPY": "USDJPY", "GBP": "GBPUSD", "CHF": "USDCHF",
         "AUD": "AUDUSD", "CAD": "USDCAD", "NZD": "NZDUSD"}
# pair の建値方向: base/quote。EURUSD は base=EUR quote=USD。
BASE_QUOTE = {"EURUSD": ("EUR", "USD"), "USDJPY": ("USD", "JPY"), "GBPUSD": ("GBP", "USD"),
              "USDCHF": ("USD", "CHF"), "AUDUSD": ("AUD", "USD"), "USDCAD": ("USD", "CAD"),
              "NZDUSD": ("NZD", "USD")}
Z_ENTRY, HOLD, MAX_POS, LOOKBACK, VOLWIN = -2.0, 20, 4, 5, 60
SELECTION, REFERENCE = ("2011-01-01", "2019-12-31"), ("2020-01-01", "2099-12-31")
OUT = Path("data/fx_carry_dip")


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mid, hs, op = {}, {}, {}
    for pair in PAIRS.values():
        d = pd.read_parquet(f"data/fx_dukascopy/{pair}_day.parquet")
        d["date"] = pd.to_datetime(d["ts"]).dt.tz_localize(None).dt.normalize()
        d = d.drop_duplicates("date").set_index("date").sort_index()
        mid[pair] = d["mid"]
        hs[pair] = d["spread"] / d["mid"] / 2.0
        op[pair] = (d["open_bid"] + d["open_ask"]) / 2
    mid = pd.DataFrame(mid).dropna(how="all")
    rates = load_short_rates(index=mid.index)
    return mid, pd.DataFrame(hs), pd.DataFrame(op), rates


def run(mid, hs, op, rates, z_entry=Z_ENTRY, hold=HOLD, max_pos=MAX_POS,
        haircut_bp=0.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = mid.index
    roll = rollover_days_series(idx)
    lg = np.log(mid)
    ret = lg.diff()
    daily = pd.Series(0.0, index=idx)
    open_pos: list[dict] = []
    trades = []
    for i, day in enumerate(idx):
        if i < VOLWIN + LOOKBACK + 1:
            continue
        # 満期到来分を落とす
        for p in [p for p in open_pos if p["exit_i"] <= i]:
            trades.append({"pair": p["pair"], "entry": p["entry_day"], "exit": day,
                           "sign": p["sign"], "ret": p["cum"]})
            open_pos.remove(p)
        # 既存建玉の当日損益（価格＋スワップ）
        pnl = 0.0
        for p in open_pos:
            r = p["sign"] * ret.loc[day, p["pair"]]
            b, q = BASE_QUOTE[p["pair"]]
            carry = p["sign"] * (rates.loc[day, b] - rates.loc[day, q])
            sw = (carry - haircut_bp / 1e4) * roll.loc[day] / 365.0
            pnl += (r + sw) * p["w"]
            p["cum"] += r + sw
        # 新規エントリ判定（当日終値まででシグナル、翌営業日は次ループの建玉）
        if len(open_pos) < max_pos and i + hold < len(idx):
            held = {p["pair"] for p in open_pos}
            cands = []
            for ccy, pair in PAIRS.items():
                if pair in held:
                    continue
                b, q = BASE_QUOTE[pair]
                diff = rates.loc[day, b] - rates.loc[day, q]
                sign = 1 if diff > 0 else -1          # 正キャリーになる向き
                if diff == 0 or not np.isfinite(diff):
                    continue
                r5 = sign * (lg.loc[day, pair] - lg.iloc[i - LOOKBACK][pair])
                sd = (sign * ret[pair]).iloc[i - VOLWIN:i].std() * np.sqrt(LOOKBACK)
                if not np.isfinite(sd) or sd == 0:
                    continue
                z = r5 / sd
                if z <= z_entry:
                    cands.append((z, pair, sign))
            for z, pair, sign in sorted(cands)[: max_pos - len(open_pos)]:
                w = 1.0 / max_pos
                pnl -= hs.loc[day, pair] * w * 2      # 往復スプレッド（入口で計上）
                open_pos.append({"pair": pair, "sign": sign, "w": w, "cum": 0.0,
                                 "exit_i": i + hold, "entry_day": day})
        daily.loc[day] = pnl
    return daily.to_frame("net"), pd.DataFrame(trades)


def stats(daily: pd.DataFrame, trades: pd.DataFrame, lo: str, hi: str) -> dict:
    r = daily.loc[lo:hi, "net"]
    if len(r) < 100 or r.std() == 0:
        return {"days": int(len(r)), "sharpe": None}
    eq = (1 + r).cumprod()
    t = trades[trades["entry"].between(lo, hi)] if len(trades) else trades
    tr = t["ret"] if len(t) else pd.Series(dtype=float)
    pos = tr[tr > 0].sum() if len(tr) else 0
    drop10 = r.copy()
    if len(t) >= 10:
        worst = t.nlargest(10, "ret")["exit"]
        drop10 = r[~r.index.isin(worst)]
    by = r.groupby(r.index.year).sum()
    return {"days": int(len(r)), "trades": int(len(t)),
            "sharpe": round(float(r.mean() / r.std() * 252 ** .5), 3),
            "ann_return": round(float(r.mean() * 252), 4),
            "max_drawdown": round(float((eq / eq.cummax() - 1).min()), 4),
            "trade_mean": round(float(tr.mean()), 4) if len(tr) else None,
            "trade_win_rate": round(float(tr.gt(0).mean()), 4) if len(tr) else None,
            "top5_trade_profit_share": round(float(tr.nlargest(5).sum() / pos), 4)
            if pos > 0 else None,
            "sharpe_ex_top10": round(float(drop10.mean() / drop10.std() * 252 ** .5), 3)
            if drop10.std() else None,
            "negative_years": int((by < 0).sum()), "years": int(len(by)),
            "by_year": {int(k): round(float(v), 4) for k, v in by.items()}}


def main() -> None:
    mid, hs, op, rates = load()
    daily, trades = run(mid, hs, op, rates)
    out = {"spec": {"z_entry": Z_ENTRY, "hold": HOLD, "max_pos": MAX_POS,
                    "lookback": LOOKBACK, "vol_window": VOLWIN},
           "window_note": "2011-2025 は無条件キャリーで消費済み。合格しても選択段階の結果で、"
                          "判定には新規フォワードが要る",
           "selection": stats(daily, trades, *SELECTION),
           "reference": stats(daily, trades, *REFERENCE)}
    s = out["selection"]
    failed = []
    if (s.get("sharpe") or -9) < 1.0:
        failed.append("selection_sharpe_lt_1.0")
    if s.get("negative_years", 9) > max(1, s.get("years", 1) // 3):
        failed.append("selection_too_many_negative_years")
    if (s.get("top5_trade_profit_share") or 1) >= .20:
        failed.append("selection_top5_trade_share_ge_20pct")
    if (s.get("sharpe_ex_top10") or -9) < 0.5:
        failed.append("selection_sharpe_ex_top10_lt_0.5")
    out["failed_criteria"] = failed
    out["decision"] = "NO_GO" if failed else "SELECTION_PASS_NEEDS_FORWARD"
    out["sensitivity"] = {}
    for lab, kw in [("z-1.5", dict(z_entry=-1.5)), ("z-2.5", dict(z_entry=-2.5)),
                    ("hold10", dict(hold=10)), ("hold40", dict(hold=40)),
                    ("pos2", dict(max_pos=2)), ("pos7", dict(max_pos=7)),
                    ("haircut50", dict(haircut_bp=50))]:
        d2, t2 = run(mid, hs, op, rates, **kw)
        out["sensitivity"][lab] = stats(d2, t2, *SELECTION)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str),
                                      encoding="utf-8")
    if len(trades):
        trades.to_csv(OUT / "trades.csv", index=False)
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
