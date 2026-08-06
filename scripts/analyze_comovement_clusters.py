#!/usr/bin/env python3
"""Do statistical co-movement clusters beat S33 sectors as a grouping? (DIAGNOSTIC)

Follow-up to analyze_sector_momentum.py: S33 explains only 5-6% of cross-sectional
variance. Here stocks are clustered by trailing-252d return correlation
(hierarchical, average linkage, K groups), formed point-in-time each year and
evaluated ONLY on the following year — in-sample the statistical grouping wins by
construction, so every number reported is out-of-sample.

Questions:
  A. OOS variance share: cluster membership vs S33 on identical universe/periods
  B. OOS within-group mean pairwise correlation: clusters vs S33
  C. Cluster stability year over year (pair survival), and
  D. Cluster momentum persistence: does a hot cluster stay hot? (rank IC)

No costs, no execution, no strategy claim.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from trading.jp_intraday.daily_gap import load_existing_daily
from trading.jp_intraday.daily_model import load_master

K = 33                       # S33と同数＝土俵を揃える
MIN_VALUE = 5e8
OUT = Path("data/jp_comovement_clusters")


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    d = load_existing_daily()
    d["Date"] = pd.to_datetime(d["Date"])
    d["symbol"] = d["Code"].astype(str)
    m = load_master()
    d = d.merge(m[["symbol", "s33_code", "is_fund"]], on="symbol", how="left")
    d = d[(d["is_fund"] != True) & d["s33_code"].notna()]          # noqa: E712
    px = d.pivot_table(index="Date", columns="symbol", values="AdjC", aggfunc="last")
    val = d.pivot_table(index="Date", columns="symbol", values="Va", aggfunc="last")
    sec = m.set_index("symbol")["s33_code"]
    return px, val, sec


def clusters_for(ret: pd.DataFrame) -> pd.Series:
    """Hierarchical clustering on 1-corr distance, K groups."""
    c = ret.corr(min_periods=150)
    keep = c.columns[c.notna().mean() > .9]
    c = c.loc[keep, keep].fillna(0.0).clip(-1, 1)
    dist = squareform(((1 - c) / 2).values, checks=False)
    # ★average linkage は退化した（1クラスタに88%が吸われ中央値サイズ1）。
    # Ward は均衡した分割を作る金融の標準（相関距離への適用は近似だが実務慣行）。
    lab = fcluster(linkage(dist, method="ward"), t=K, criterion="maxclust")
    return pd.Series(lab, index=c.columns)


def var_share(returns: pd.Series, groups: pd.Series) -> float | None:
    r = returns.dropna()
    r = r[(r > -0.9) & (r < 4.0)]
    g = groups.reindex(r.index).dropna()
    r = r[g.index]
    if len(r) < 300 or r.var() == 0:
        return None
    return float(r.groupby(g).transform("mean").var() / r.var())


def within_corr(ret: pd.DataFrame, groups: pd.Series, n_pairs: int = 4000) -> float:
    """Mean pairwise correlation inside groups (sampled deterministically)."""
    rng = np.random.RandomState(0)
    cors = []
    for _, members in groups.groupby(groups):
        syms = [s for s in members.index if s in ret.columns]
        if len(syms) < 2:
            continue
        for _ in range(min(60, len(syms) * 2)):
            a, b = rng.choice(syms, 2, replace=False)
            v = ret[[a, b]].dropna()
            if len(v) > 150:
                cors.append(float(v[a].corr(v[b])))
            if len(cors) >= n_pairs:
                break
    return float(np.nanmean(cors))


def main() -> None:
    px, val, sec = load()
    years = range(2019, 2027)
    ret_d = px.pct_change(fill_method=None)
    A, B, C, D = [], [], [], []
    prev_cl = None
    for y in years:
        form = ret_d.loc[f"{y-1}-01-01":f"{y-1}-12-31"]
        liq = val.loc[f"{y-1}-01-01":f"{y-1}-12-31"].median()
        uni = [s for s in form.columns
               if liq.get(s, 0) >= MIN_VALUE and form[s].notna().sum() >= 200]
        cl = clusters_for(form[uni])
        ev = ret_d.loc[f"{y}-01-01":f"{y}-12-31"]
        # A: OOS 分散シェア（1m/3m）を同一ユニバースで cluster vs S33
        for months, step in (("1m", 1), ("3m", 3)):
            rr = px.loc[f"{y}-01-01":f"{y}-12-31"].resample("ME").last().pct_change(step)
            for ym in rr.index[step::step]:
                row = rr.loc[ym].reindex(cl.index)
                vs_cl = var_share(row, cl)
                vs_s33 = var_share(row, sec.reindex(cl.index))
                if vs_cl is not None and vs_s33 is not None:
                    A.append({"year": y, "h": months, "cluster": vs_cl, "s33": vs_s33})
        # B: OOS 平均ペア相関
        B.append({"year": y, "cluster": within_corr(ev, cl),
                  "s33": within_corr(ev, sec.reindex(cl.index).dropna())})
        # C: クラスタの年次安定性（同居ペアの生存率）
        if prev_cl is not None:
            common = cl.index.intersection(prev_cl.index)
            if len(common) > 300:
                rng = np.random.RandomState(1)
                same_prev = same_now = 0
                for _ in range(6000):
                    a, b = rng.choice(common, 2, replace=False)
                    if prev_cl[a] == prev_cl[b]:
                        same_prev += 1
                        same_now += int(cl[a] == cl[b])
                C.append({"year": y, "pair_survival": same_now / max(1, same_prev)})
        prev_cl = cl
        # D: クラスタ・モメンタムの持続性（12-1形成→翌1/3m・PITクラスタ）
        mom = (px.shift(21) / px.shift(252) - 1).loc[f"{y}-01-01":f"{y}-12-31"]
        eom = px.loc[f"{y-1}-12-01":f"{y}-12-31"].resample("ME").last()
        for i in range(1, len(eom.index) - 3):
            ym = eom.index[i]
            if ym.year != y:
                continue
            mrow = mom.loc[:ym].iloc[-1].reindex(cl.index)
            cmom = mrow.groupby(cl).mean()
            fut1 = (eom.iloc[i + 1] / eom.iloc[i] - 1).reindex(cl.index).groupby(cl).mean()
            fut3 = (eom.iloc[min(i + 3, len(eom) - 1)] / eom.iloc[i] - 1
                    ).reindex(cl.index).groupby(cl).mean()
            for h, fut in (("1m", fut1), ("3m", fut3)):
                v = pd.concat([cmom, fut], axis=1, keys=["m", "f"]).dropna()
                if len(v) >= 20:
                    D.append({"year": y, "h": h,
                              "ic": float(v["m"].corr(v["f"], method="spearman"))})
    a = pd.DataFrame(A)
    b = pd.DataFrame(B)
    c = pd.DataFrame(C)
    dd = pd.DataFrame(D)
    out = {"note": "DIAGNOSTIC. Clusters formed on year Y-1 correlations (K=33), "
                   "all numbers evaluated on year Y (out-of-sample).",
           "A_oos_variance_share": {
               h: {"cluster": round(float(a[a.h.eq(h)].cluster.mean()), 4),
                   "s33": round(float(a[a.h.eq(h)].s33.mean()), 4),
                   "periods": int(a.h.eq(h).sum())} for h in ("1m", "3m")},
           "B_oos_within_group_pairwise_corr": {
               "cluster": round(float(b.cluster.mean()), 3),
               "s33": round(float(b.s33.mean()), 3),
               "by_year": b.round(3).to_dict("records")},
           "C_cluster_pair_survival_yoy": {
               "mean": round(float(c.pair_survival.mean()), 3),
               "by_year": c.round(3).to_dict("records")},
           "D_cluster_momentum_ic": {
               h: {"mean_ic": round(float(dd[dd.h.eq(h)].ic.mean()), 3),
                   "t": round(float(dd[dd.h.eq(h)].ic.mean()
                              / dd[dd.h.eq(h)].ic.std()
                              * np.sqrt(max(1, dd.h.eq(h).sum() // int(h[0])))), 2),
                   "n": int(dd.h.eq(h).sum())} for h in ("1m", "3m")}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
