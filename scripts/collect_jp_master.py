"""銘柄マスタ (master.parquet) を J-Quants から取得する。

`daily_model.load_master()` が要求する data/jp_daily_history/master.parquet を
`get_eq_master`（上場銘柄一覧）の生出力そのままのスキーマで生成する:
  Date, Code, CoName, CoNameEn, S17, S17Nm, S33, S33Nm, ScaleCat, Mkt, MktNm, Mrgn, MrgnNm
MrgnNm（貸借区分）はショート可否判定、MktNm は個別株フィルタに必須。

マスタは「現時点のスナップショット」が正: 現行マスタに無い銘柄＝上場廃止は
個別株として残す設計（サバイバーシップ回避、AGENTS.md 参照）のため、
日次バーと違い過去分の収集は不要。当日取得済みならスキップする（--force で強制）。

実行:  PYTHONPATH=. python scripts/collect_jp_master.py [--force] [--out PATH]
"""
import argparse
import datetime as dt
from pathlib import Path

import pandas as pd

import jquantsapi
from data.collectors.config import JQUANTS_API_KEY

DEFAULT_OUT = Path("data/jp_daily_history/master.parquet")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--force", action="store_true", help="当日取得済みでも再取得")
    args = ap.parse_args()

    if args.out.exists() and not args.force:
        cur = pd.read_parquet(args.out, columns=["Date"])
        newest = pd.to_datetime(cur["Date"]).max()
        if pd.notna(newest) and newest.date() >= dt.date.today() - dt.timedelta(days=1):
            print(f"skip: {args.out} は最新 (Date={newest.date()})。--force で再取得。")
            return

    client = jquantsapi.ClientV2(api_key=JQUANTS_API_KEY)
    m = client.get_eq_master()
    if m.empty:
        raise SystemExit("get_eq_master が空を返しました（APIキー/プランを確認）")
    need = {"Code", "CoName", "S33", "S33Nm", "MktNm", "MrgnNm", "ScaleCat"}
    missing = need - set(m.columns)
    if missing:
        raise SystemExit(f"想定列が欠落: {missing}（jquantsapi の版差を確認）")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    m.to_parquet(args.out, index=False)
    funds = (~m["MktNm"].isin(["プライム", "スタンダード", "グロース"])).sum()
    print(f"wrote {args.out}: {len(m)}銘柄（うち非・個別株 {funds}） Date={m['Date'].iloc[0]}")


if __name__ == "__main__":
    main()
