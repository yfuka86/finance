#!/usr/bin/env python3
"""X11 sealed-forward verdict -- SEALED until 2028-08-12. Run ONCE.

Preregistered in docs/PREREGISTER_OVERSOLD_X11_FORWARD.md. No override flag
exists by design. Recomputes everything from raw data with the frozen spec;
the candidate ledger is an integrity cross-check, not the data source.
Criteria (fixed 2026-08-12, concentration relaxed by explicit user decision):
excess IR >= 0.5, both calendar years positive, top-5-day share < 30%,
ex-top10 IR >= 0.2, >= 60 active days/year.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd

JUDGMENT_DATE = dt.date(2028, 8, 12)
SEAL = (pd.Timestamp("2026-08-12"), pd.Timestamp("2028-08-11"))


def main() -> None:
    if dt.date.today() < JUDGMENT_DATE:
        raise SystemExit(
            f"SEALED until {JUDGMENT_DATE}. Today is {dt.date.today()}. "
            "No override exists; opening early would unseal the forward test.")
    from scripts.oversold_sweep_harness import build, members_for
    from scripts.experiment_oversold_interaction import simulate
    A = build(1e9)
    mem = members_for(A, {"dip": "z20", "ivol": "lo", "market": "none"})
    ex = simulate(A, mem, 5, True).loc[SEAL[0]:SEAL[1]]
    active = ex[ex != 0]
    yearly = ex.groupby(ex.index.year).sum()
    top5 = float(ex.nlargest(5).sum() / ex.sum()) if ex.sum() > 0 else None
    ex10 = ex.drop(ex.nlargest(10).index)
    ir = float(ex.mean() / ex.std() * 252 ** .5)
    ir10 = float(ex10.mean() / ex10.std() * 252 ** .5)
    crit = {"ir_ge_05": ir >= .5,
            "both_years_positive": bool((yearly > 0).all()),
            "top5_lt_30pct": bool(top5 is not None and top5 < .30),
            "ir_ex10_ge_02": ir10 >= .2,
            "freq_ge_60py": bool(len(active) / (len(ex) / 252) >= 60)}
    verdict = {"judged_at": dt.datetime.now(dt.timezone.utc).isoformat(),
               "window": [str(SEAL[0].date()), str(SEAL[1].date())],
               "ir": round(ir, 3), "excess_ann_pct": round(float(ex.mean() * 252 * 100), 2),
               "yearly_pct": {int(y): round(float(v) * 100, 2) for y, v in yearly.items()},
               "top5_share": None if top5 is None else round(top5, 3),
               "ir_ex_top10": round(ir10, 3), "criteria": crit,
               "decision": "GO_PENDING_USER_APPROVAL" if all(crit.values()) else
               "NO_GO_PERMANENT_CLOSE"}
    out = Path("data/jp_oversold_x11_forward/verdict.json")
    out.open("x", encoding="utf-8").write(
        json.dumps(verdict, ensure_ascii=False, indent=2))
    print(json.dumps(verdict, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
