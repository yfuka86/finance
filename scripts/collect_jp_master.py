"""Collect the equity master (銘柄マスタ) into data/jp_daily_history/master.parquet.

daily_model.load_master() expects columns Code / CoName / S33Nm / MktNm /
MrgnNm / ScaleCat (market section, margin/貸借 flag, TOPIX scale category).
Run SOLO after other collectors (429 avoidance). Idempotent: overwrites the
single snapshot file, which is the intended behaviour for a current-state master.
"""
import time
from pathlib import Path

import jquantsapi
from data.collectors.config import JQUANTS_API_KEY

OUT = Path("data/jp_daily_history/master.parquet")


def _retry(fn, tries=8, base=2.0):
    for i in range(tries):
        try:
            return fn()
        except Exception:  # noqa: BLE001
            if i == tries - 1:
                raise
            time.sleep(min(base ** i, 90))


def main() -> None:
    client = jquantsapi.ClientV2(api_key=JQUANTS_API_KEY)
    m = _retry(client.get_eq_master)
    need = {"Code", "CoName", "S33Nm", "MktNm", "MrgnNm", "ScaleCat"}
    missing = need - set(m.columns)
    if missing:
        raise SystemExit(f"master is missing expected columns: {missing} "
                         f"(got {sorted(m.columns)})")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    m.to_parquet(OUT, index=False)
    print(f"wrote {OUT} rows={len(m)} 貸借={int(m['MrgnNm'].eq('貸借').sum())}")


if __name__ == "__main__":
    main()
