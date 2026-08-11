#!/usr/bin/env python3
"""Export trade-level detail for the oversold-interaction strategy dashboard.

Re-simulates the two headline cells (ML_ridge_h3 and A0_m0_h5_tp) with full
trade recording over the FULL period 2018..present. 2025+ display was an
explicit user decision (2026-08-11): showing it consumes this family's
confirmation window, which is recorded in AGENTS.md. Output:
data/jp_oversold_interaction/detail.json for scripts/build_finance_site.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experiment_oversold_interaction import (build_arrays, ml_members,
                                                     COST_OPEN, COST_CLOSE,
                                                     TP_MULT, SEL_END)

OUT = Path("data/jp_oversold_interaction")
LO = pd.Timestamp("2018-01-01")


def fundamentals_for(syms: list[str]) -> dict:
    """PIT fundamental snapshot per 5-digit panel symbol (latest disclosures).

    fins codes are 4-char (width trap!): match on sym[:4]. Growth/ROE use the
    latest FULL-YEAR rows (quarterly NP is cumulative and would mislead);
    BPS/forecast dividend use the latest disclosed state of any row.
    """
    import numpy as np
    from scripts.run_value_event_v1 import load_fins
    from trading.jp_intraday.daily_gap import load_existing_daily
    f = load_fins()
    f["sym4"] = f["Code"].astype(str).str[:4]
    want4 = {s[:4] for s in syms}
    f = f[f["sym4"].isin(want4)].copy()
    f["disc"] = pd.to_datetime(f["DiscDate"], errors="coerce")
    for c in ("Sales", "OP", "NP", "Eq", "BPS", "DivAnn", "FDivAnn"):
        f[c] = pd.to_numeric(f.get(c), errors="coerce")
    f = f.sort_values(["sym4", "disc"])
    st = f.groupby("sym4")[["BPS", "FDivAnn"]].last()
    fy = f[f["CurPerType"] == "FY"].dropna(subset=["Sales"]).sort_values(["sym4", "disc"])
    # groupby.nth returns original-index rows (silent mismatch); rank from end instead
    fy = fy.assign(_rn=fy.groupby("sym4").cumcount(ascending=False))
    last_fy = fy[fy["_rn"] == 0].set_index("sym4")[["Sales", "OP", "NP", "Eq", "DivAnn"]]
    prev_fy = fy[fy["_rn"] == 1].set_index("sym4")[["Sales", "OP", "DivAnn"]]
    d = load_existing_daily()
    d = d[d["Code"].astype(str).isin(set(syms))]
    px = (d.assign(Date=pd.to_datetime(d["Date"]))
          .sort_values("Date").groupby(d["Code"].astype(str))["raw_close"].last())
    out = {}
    for s5 in syms:
        s4 = s5[:4]
        r = {}
        close = float(px.get(s5, np.nan))
        bps = float(st["BPS"].get(s4, np.nan))
        fdiv = float(st["FDivAnn"].get(s4, np.nan))
        lf = last_fy.loc[s4] if s4 in last_fy.index else None
        pf = prev_fy.loc[s4] if s4 in prev_fy.index else None
        r["pbr"] = round(close / bps, 2) if close > 0 and bps and bps > 0 else None
        if lf is not None and pd.notna(lf["NP"]) and lf["Eq"]:
            r["roe_pct"] = round(float(lf["NP"] / lf["Eq"]) * 100, 1)
        if lf is not None and pf is not None:
            if pd.notna(lf["Sales"]) and pf["Sales"]:
                r["sales_yoy_pct"] = round(float(lf["Sales"] / pf["Sales"] - 1) * 100, 1)
            if pd.notna(lf["OP"]) and pf["OP"] and pf["OP"] > 0:
                r["op_yoy_pct"] = round(float(lf["OP"] / pf["OP"] - 1) * 100, 1)
            prev_div = float(lf["DivAnn"]) if pd.notna(lf["DivAnn"]) else None
            if prev_div is not None and pd.notna(fdiv):
                r["div_up"] = bool(fdiv > prev_div)
        if pd.notna(fdiv) and close > 0:
            r["div_yield_pct"] = round(fdiv / close * 100, 2)
        out[s5] = r
    return out


def names_map() -> dict:
    m = pd.read_parquet("data/jp_daily_history/master.parquet",
                        columns=["Code", "CoName", "S33Nm"])
    m["Code"] = m["Code"].astype(str)
    return {r.Code: (r.CoName, r.S33Nm) for r in m.itertuples()}


def simulate_recorded(A, members: pd.DataFrame, h: int, use_tp: bool):
    dates = A["dates"]
    syms = A["CC"].columns.to_numpy()
    n = len(dates)
    strat = np.zeros(n); bench = np.zeros(n); wsum = np.zeros(n)
    INTRA, CC, ONF, IVOL = (A[k].to_numpy() for k in ("INTRA", "CC", "ONF", "IVOL"))
    mi, mc, mo = (A[k].to_numpy() for k in ("mkt_intra", "mkt_cc", "mkt_onf"))
    mem = members.reindex(index=dates, columns=A["CC"].columns).fillna(False).to_numpy()
    trades = []
    for i in np.nonzero(mem.any(axis=1))[0]:
        if dates[i] < LO or i + h + 1 >= n:
            continue
        cols = np.nonzero(mem[i])[0]
        e = i + 1
        ok = ~np.isnan(INTRA[e, cols])
        cols = cols[ok]
        if len(cols) == 0:
            continue
        cum = 1 + INTRA[e, cols]
        thresh = 1 + TP_MULT * IVOL[i, cols] if use_tp else np.full(len(cols), np.inf)
        alive = np.ones(len(cols), dtype=bool)
        exit_day = np.full(len(cols), e + h - 1)
        exit_reason = np.array(["期日引け"] * len(cols), dtype=object)
        tot = 1 + INTRA[e, cols]
        strat[e] += np.nansum(INTRA[e, cols]) - len(cols) * COST_OPEN
        bench[e] += len(cols) * mi[e]; wsum[e] += len(cols)
        if h > 1:
            for k in range(1, h):
                s = e + k
                hit = alive & (cum >= thresh)
                if hit.any():
                    r = np.where(np.isnan(ONF[s - 1, cols[hit]]), 0.0,
                                 ONF[s - 1, cols[hit]])
                    strat[s] += r.sum() - hit.sum() * COST_OPEN
                    bench[s] += hit.sum() * mo[s - 1]; wsum[s] += hit.sum()
                    tot[hit] *= 1 + r
                    exit_day[hit] = s; exit_reason[hit] = "利確(翌寄)"
                    alive &= ~hit
                if not alive.any():
                    break
                r = np.where(np.isnan(CC[s, cols[alive]]), 0.0, CC[s, cols[alive]])
                strat[s] += r.sum()
                bench[s] += alive.sum() * mc[s]; wsum[s] += alive.sum()
                tot[alive] *= 1 + r
                cum[alive] *= 1 + np.clip(r, -1, None)
            if alive.any():
                strat[e + h - 1] -= alive.sum() * COST_CLOSE
        else:
            strat[e] -= len(cols) * COST_CLOSE
        for j, c in enumerate(cols):
            trades.append({"trigger": str(dates[i].date()),
                           "entry": str(dates[e].date()),
                           "exit": str(dates[exit_day[j]].date()),
                           "sym": str(syms[c]), "ret": round(float(tot[j] - 1), 5),
                           "reason": str(exit_reason[j])})
    with np.errstate(invalid="ignore"):
        daily = np.where(wsum > 0, (strat - bench) / np.maximum(wsum, 1), 0.0)
    ex = pd.Series(daily, index=dates).loc[LO:]
    return ex, trades


def pack(ex: pd.Series, trades: list, nm: dict) -> dict:
    for t in trades:
        name, sec = nm.get(t["sym"], ("", ""))
        t["name"], t["sector"] = name, sec
    tr = pd.DataFrame(trades)
    cum = (1 + ex).cumprod()
    monthly = ex.groupby(ex.index.to_period("M")).sum()
    agg_sym = (tr.groupby(["sym", "name"])
               .agg(n=("ret", "size"), mean_ret=("ret", "mean"),
                    win=("ret", lambda r: (r > 0).mean()))
               .reset_index().sort_values("n", ascending=False))
    def window_stats(lo, hi):
        w = ex.loc[lo:hi]
        if len(w) < 60 or w.std() == 0:
            return {}
        return {"ir": round(float(w.mean() / w.std() * 252 ** .5), 2),
                "excess_ann_pct": round(float(w.mean() * 252 * 100), 2)}
    return {
        "sel_stats": window_stats("2018-01-01", str(SEL_END.date())),
        "post_stats": window_stats("2025-01-01", "2027-12-31"),
        "daily_cum": [[str(d.date()), round(float(v), 5)]
                      for d, v in cum.iloc[::3].items()],
        "monthly": [[str(p), round(float(v) * 100, 2)] for p, v in monthly.items()],
        "trades_recent": tr.sort_values("entry").tail(400).to_dict("records"),
        "top_winners": tr.nlargest(25, "ret").to_dict("records"),
        "top_losers": tr.nsmallest(25, "ret").to_dict("records"),
        "most_traded": agg_sym.head(20).round(4).to_dict("records"),
        "n_trades": int(len(tr)),
        "win_rate": round(float((tr["ret"] > 0).mean()), 3),
        "mean_ret_bps": round(float(tr["ret"].mean() * 1e4), 1),
        "tp_exit_share": round(float((tr["reason"] == "利確(翌寄)").mean()), 3),
    }


def main() -> None:
    A = build_arrays()
    nm = names_map()
    out = {}
    # X11 (z20 x low-ivol x h5 x TP, no market gate): the 2nd sweep's closest cell.
    # Candidates are tomorrow's opening-auction buys (signal only, no fwd P&L calc
    # beyond the already-disclosed window; display frozen at seal date if sealed).
    from scripts.oversold_sweep_harness import build as hbuild, members_for
    HA = hbuild(1e9)
    x11_mem = members_for(HA, {"dip": "z20", "ivol": "lo", "market": "none"})
    ex, tr = simulate_recorded(HA, x11_mem, 5, True)
    # SEALED 2026-08-12: display frozen at the seal date (no-peek forward).
    seal = pd.Timestamp("2026-08-11")
    tr = [t for t in tr if t["entry"] <= "2026-08-11"]
    out["X11_z20_lo_h5tp"] = pack(ex.loc[:seal], tr, nm)
    out["X11_z20_lo_h5tp"]["sealed_note"] = "成績表示は封印日2026-08-11で凍結（判定2028-08-12）"
    last_day = x11_mem.index[x11_mem.any(axis=1)][-1]
    cand = x11_mem.columns[x11_mem.loc[last_day]].tolist()
    fund = fundamentals_for(sorted(cand))
    out["x11_candidates"] = {
        "signal_date": str(pd.Timestamp(last_day).date()),
        "entry": "翌営業日の寄成（現物ロング・5営業日 or +2×ivol20で翌寄利確）",
        "names": [{"sym": c, "name": nm.get(c, ("", ""))[0],
                   "sector": nm.get(c, ("", ""))[1], **fund.get(c, {})}
                  for c in sorted(cand)]}
    oversold = A["Z5"].le(A["Z5"].quantile(.1, axis=1), axis=0)
    mmask = A["M"].le(0.0)
    mem_rule = oversold[mmask.reindex(oversold.index).fillna(False)].fillna(False)
    ex, tr = simulate_recorded(A, mem_rule, 5, True)
    out["A0_m0_h5_tp"] = pack(ex, tr, nm)
    mem_ml = ml_members(A, "ridge")
    ex, tr = simulate_recorded(A, mem_ml, 3, False)
    out["ML_ridge_h3"] = pack(ex, tr, nm)
    (OUT / "detail.json").write_text(
        json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: {"n_trades": v["n_trades"], "win": v["win_rate"],
                          "mean_bps": v["mean_ret_bps"]}
                      for k, v in out.items() if "n_trades" in v},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
