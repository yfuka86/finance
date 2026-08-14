#!/usr/bin/env python3
"""Oversold interaction: market-state x stock-oversold longs, TP shaping, ML/MLP.

Frozen in docs/PREREGISTER_OVERSOLD_INTERACTION.md. Rule grid: market 5d state
{<=0,-2%,-4%} x hold {1,3,5} x TP {none, close-trigger 2x ivol20 -> next open};
long-only cash, all auctions. ML cells: Ridge / MLP on interaction features,
top-decile 3d books. Judged on leg-matched equal-weight market EXCESS.
Selection 2018-2024; confirmation 2025+ opens once for a passing cell.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from trading.jp_intraday.daily_model import load_panel_cached

SEL_END = pd.Timestamp("2024-12-31")
COST_OPEN, COST_CLOSE = 1.5e-4, 0.5e-4
TP_MULT = 2.0
OUT = Path("data/jp_oversold_interaction")


def build_arrays():
    p = load_panel_cached(min_value_yen=1e9)[
        ["date", "symbol", "intraday_ret", "ret", "ret_on_fwd", "ivol",
         "prev_close", "raw_close"]].copy()
    dates_all = pd.Index(sorted(p["date"].unique()))
    syms_all = pd.Index(sorted(p["symbol"].astype(str).unique()))
    piv = lambda c: p.pivot_table(index="date", columns="symbol", values=c,
                                  aggfunc="last").reindex(index=dates_all,
                                                          columns=syms_all)
    INTRA, CC, ONF, IVOL = piv("intraday_ret"), piv("ret"), piv("ret_on_fwd"), piv("ivol")
    # 5d return z, 1d z, rsi from adjusted closes reconstructed via cc chain
    logcc = np.log1p(CC)
    r5 = np.expm1(logcc.rolling(5).sum())
    z = lambda M: M.sub(M.mean(axis=1), axis=0).div(M.std(axis=1), axis=0)
    Z5, Z1, ZIV = z(r5), z(CC), z(IVOL)
    d = CC.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-CC.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    RSI = 100 - 100 / (1 + d / dn.replace(0, np.nan))
    tp = pd.read_parquet("data/jp_derivatives/topix_index_2008_2026.parquet")
    tp["Date"] = pd.to_datetime(tp["Date"])
    tpx5 = tp.set_index("Date")["C"].sort_index().pct_change(5).reindex(CC.index)
    mkt_intra = INTRA.mean(axis=1)
    mkt_cc = CC.mean(axis=1)
    mkt_onf = ONF.mean(axis=1)
    return {"INTRA": INTRA, "CC": CC, "ONF": ONF, "IVOL": IVOL, "Z5": Z5,
            "Z1": Z1, "ZIV": ZIV, "RSI": RSI, "M": tpx5,
            "mkt_intra": mkt_intra, "mkt_cc": mkt_cc, "mkt_onf": mkt_onf,
            "dates": CC.index, "panel": p}


def simulate(A, members: pd.DataFrame, h: int, use_tp: bool,
             lo=None, hi=None) -> pd.Series:
    """members: bool DataFrame (trigger date x symbol). Entry next session open.

    Returns full-calendar daily EXCESS series (0 when flat). Leg matching:
    day1 intra vs mkt_intra; later days cc vs mkt_cc; TP exit day onf vs mkt_onf.
    """
    dates = A["dates"]
    n = len(dates)
    strat = np.zeros(n)
    bench = np.zeros(n)
    weightsum = np.zeros(n)
    INTRA, CC, ONF, IVOL = (A[k].to_numpy() for k in ("INTRA", "CC", "ONF", "IVOL"))
    mi, mc, mo = (A[k].to_numpy() for k in ("mkt_intra", "mkt_cc", "mkt_onf"))
    mem = members.reindex(index=dates, columns=A["CC"].columns).fillna(False).to_numpy()
    for i in np.nonzero(mem.any(axis=1))[0]:
        if lo is not None and (dates[i] < lo or dates[i] > hi):
            continue
        if i + h + 1 >= n:
            continue
        cols = np.nonzero(mem[i])[0]
        e = i + 1                                   # entry session (open)
        r_day1 = INTRA[e, cols]
        ok = ~np.isnan(r_day1)
        cols = cols[ok]
        if len(cols) == 0:
            continue
        cum = 1 + INTRA[e, cols]                     # price-relative TP tracker
        thresh = 1 + TP_MULT * IVOL[i, cols] if use_tp else np.full(len(cols), np.inf)
        alive = np.ones(len(cols), dtype=bool)
        strat[e] += np.nansum(INTRA[e, cols]) - len(cols) * COST_OPEN
        bench[e] += len(cols) * mi[e]
        weightsum[e] += len(cols)
        if h == 1:
            strat[e] -= len(cols) * COST_CLOSE
            continue
        for k in range(1, h):
            s = e + k
            hit = alive & (cum >= thresh)
            if hit.any():                            # exit at next open (onf leg)
                r = ONF[s - 1, cols[hit]]
                r = np.where(np.isnan(r), 0.0, r)
                strat[s] += r.sum() - hit.sum() * COST_OPEN
                bench[s] += hit.sum() * mo[s - 1]
                weightsum[s] += hit.sum()
                alive &= ~hit
            if not alive.any():
                break
            r = CC[s, cols[alive]]
            r = np.where(np.isnan(r), 0.0, r)
            strat[s] += r.sum()
            bench[s] += alive.sum() * mc[s]
            weightsum[s] += alive.sum()
            cum[alive] *= 1 + np.clip(r, -1, None)
        if alive.any():                              # final close exit
            strat[e + h - 1] -= alive.sum() * COST_CLOSE
    with np.errstate(invalid="ignore"):
        daily = np.where(weightsum > 0, (strat - bench) / np.maximum(weightsum, 1), 0.0)
    return pd.Series(daily, index=dates)


def battery(ex: pd.Series, lo, hi) -> dict:
    ex = ex.loc[lo:hi]
    active = (ex != 0)
    if active.sum() < 100 or ex.std() == 0:
        return {"ir": None, "active_days_per_year": round(float(
            active.sum() / max(len(ex) / 252, 1e-9)), 1)}
    yearly = ex.groupby(ex.index.year).sum()
    top5 = float(ex.nlargest(5).sum() / ex.sum()) if ex.sum() > 0 else None
    ex10 = ex.drop(ex.nlargest(10).index)
    out = {"ir": round(float(ex.mean() / ex.std() * 252 ** .5), 3),
           "excess_ann_pct": round(float(ex.mean() * 252 * 100), 2),
           "active_days_per_year": round(float(active.sum() / (len(ex) / 252)), 1),
           "neg_years": int((yearly < 0).sum()), "years": int(len(yearly)),
           "top5_share": None if top5 is None else round(top5, 3),
           "ir_ex_top10": round(float(ex10.mean() / ex10.std() * 252 ** .5), 3)}
    out["criteria"] = {"ir_ge_07": bool((out["ir"] or -9) >= .7),
                       "freq_ge_60py": bool(out["active_days_per_year"] >= 60),
                       "neg_years_le_third": bool(out["years"] > 0
                                                  and out["neg_years"] * 3 <= out["years"]),
                       "top5_lt_20pct": bool((out["top5_share"] or 9) < .20),
                       "ir_ex10_ge_035": bool((out["ir_ex_top10"] or -9) >= .35)}
    return out


def ml_members(A, kind: str) -> pd.DataFrame:
    from sklearn.neural_network import MLPRegressor
    feats = {}
    M = A["M"]
    for name, Z in (("z5", A["Z5"]), ("z1", A["Z1"]), ("ziv", A["ZIV"]),
                    ("rsi", A["RSI"] / 50 - 1)):
        feats[name] = Z
        feats[name + "xM"] = Z.mul(M, axis=0)
    INTRA, CC = A["INTRA"], A["CC"]
    tgt = (1 + INTRA.shift(-1)) * (1 + CC.shift(-2)) * (1 + CC.shift(-3)) - 1
    tgt = tgt.sub(tgt.mean(axis=1), axis=0)
    rows = []
    dates = A["dates"]
    years = dates.year.unique()
    stack = pd.concat({k: v.stack() for k, v in feats.items()}, axis=1)
    stack["target"] = tgt.stack(dropna=False).reindex(stack.index)
    stack["M"] = M.reindex(stack.index.get_level_values(0)).to_numpy()
    stack = stack.dropna()
    preds = []
    rng = np.random.default_rng(0)
    for y in sorted(years):
        if y < 2020:
            continue
        tr = stack[stack.index.get_level_values(0).year < y]
        te = stack[stack.index.get_level_values(0).year == y]
        if len(tr) < 50000 or te.empty:
            continue
        if len(tr) > 400_000:
            tr = tr.iloc[rng.choice(len(tr), 400_000, replace=False)]
        X = tr.drop(columns="target").to_numpy()
        mu, sd = X.mean(0), X.std(0) + 1e-12
        Xn = (X - mu) / sd
        yv = tr["target"].to_numpy()
        Xt = (te.drop(columns="target").to_numpy() - mu) / sd
        if kind == "ridge":
            beta = np.linalg.solve(Xn.T @ Xn + 30.0 * np.eye(Xn.shape[1]), Xn.T @ yv)
            pr = Xt @ beta
        else:
            m = MLPRegressor(hidden_layer_sizes=(32, 16), early_stopping=True,
                             max_iter=200, random_state=0)
            m.fit(Xn, yv)
            pr = m.predict(Xt)
        preds.append(pd.Series(pr, index=te.index))
    if not preds:
        return pd.DataFrame()
    P = pd.concat(preds).unstack()
    thr = P.quantile(.9, axis=1)
    return P.ge(thr, axis=0)


def main() -> None:
    A = build_arrays()
    lo, hi = pd.Timestamp("2018-01-01"), SEL_END
    summary = {"spec": "docs/PREREGISTER_OVERSOLD_INTERACTION.md",
               "selection": {}, "confirmation": "UNOPENED"}
    oversold = A["Z5"].le(A["Z5"].quantile(.1, axis=1), axis=0)
    passing = []
    cells = []
    for mlab, mthr in (("A0_m0", 0.0), ("A1_m2", -.02), ("A2_m4", -.04)):
        mmask = A["M"].le(mthr)
        mem = oversold[mmask.reindex(oversold.index).fillna(False)].fillna(False)
        for h in (1, 3, 5):
            for use_tp in ((False,) if h == 1 else (False, True)):
                cells.append((f"{mlab}_h{h}" + ("_tp" if use_tp else ""),
                              mem, h, use_tp))
    for kind in ("ridge", "mlp"):
        memk = ml_members(A, kind)
        if len(memk):
            cells.append((f"ML_{kind}_h3", memk, 3, False))
    for name, mem, h, use_tp in cells:
        ex = simulate(A, mem, h, use_tp)
        b = battery(ex, lo, hi)
        summary["selection"][name] = b
        if b.get("criteria") and all(b["criteria"].values()):
            passing.append((name, mem, h, use_tp))
    summary["passing_cells"] = [c[0] for c in passing]
    if passing:
        name, mem, h, use_tp = passing[0]
        summary["frozen_cell"] = name
        ex = simulate(A, mem, h, use_tp)
        summary["confirmation"] = {name: battery(
            ex, pd.Timestamp("2025-01-01"), pd.Timestamp("2026-12-31"))}
        conf = summary["confirmation"][name]
        summary["decision"] = ("GO_PENDING_USER_APPROVAL"
                               if conf.get("criteria") and all(conf["criteria"].values())
                               else "NO_GO_AT_CONFIRMATION")
    else:
        summary["decision"] = "NO_GO_AT_SELECTION"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
