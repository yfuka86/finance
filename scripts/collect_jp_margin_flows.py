"""需給データ収集: 週次信用残・売買内訳(日次信用フロー)・空売り残高報告(大口).

J-Quants Premium。**単独実行**（並列は429でデータ欠損）・冪等・日次逐次+強バックオフ+
インクリメンタルフラッシュで再開可能（既存の日次コレクタと同方式）。
  - margin_interest_{Y}.parquet : 週次銘柄別 信用買残/売残（LongVol/ShrtVol・制度/一般）
  - breakdown_{Y}.parquet       : 日次銘柄別 売買内訳（信用新規/返済・実弾空売り代金/数量）
  - short_positions_{Y}.parquet : 空売り残高報告（大口機関の銘柄別ポジション・開示日）

実行:  PYTHONPATH=. python scripts/collect_jp_margin_flows.py
"""
import time
from pathlib import Path

import pandas as pd

import jquantsapi
from data.collectors.config import JQUANTS_API_KEY

START_YEAR, END_YEAR = 2018, 2026
OUT = Path("data/jp_flows")
PACE = 0.6          # 呼び出し間隔（レート制限余裕）
FLUSH_EVERY = 40    # 日次ループのフラッシュ間隔


def _retry(fn, tries=10, base=2.0, cap=300.0):
    for i in range(tries):
        try:
            return fn()
        except Exception:  # noqa: BLE001 - 429含む。長い停止にも耐える
            if i == tries - 1:
                raise
            time.sleep(min(base ** i, cap))


def _trading_days(client, year: int) -> list[str]:
    cal = _retry(lambda: client.get_mkt_calendar(from_yyyymmdd=f"{year}0101",
                                                 to_yyyymmdd=f"{year}1231"))
    return sorted(cal[cal["HolDiv"].astype(str).isin(["1", "2"])]["Date"].astype(str))


def _collect_daily(client, year: int, path: Path, date_col: str, fetch) -> None:
    """日次逐次収集（既存分スキップ・定期フラッシュ・再開可能）."""
    days = _trading_days(client, year)
    have: set = set()
    frames = []
    if path.exists():
        cur = pd.read_parquet(path)
        frames.append(cur)
        have = set(pd.to_datetime(cur[date_col]).dt.strftime("%Y-%m-%d"))
    todo = [d for d in days if d not in have]
    if not todo:
        print(f"{year} {path.stem.rsplit('_', 1)[0]}: complete, skip", flush=True)
        return
    got = 0
    for i, d in enumerate(todo):
        time.sleep(PACE)
        try:
            df = _retry(lambda: fetch(d.replace("-", "")))
        except Exception as exc:  # noqa: BLE001 - 開示なし日/一時障害は飛ばして続行
            print(f"  {d}: skip ({type(exc).__name__})", flush=True)
            continue
        if len(df):
            frames.append(df)
            got += len(df)
        if (i + 1) % FLUSH_EVERY == 0 and frames:
            pd.concat(frames, ignore_index=True).to_parquet(path, index=False)
    if frames:
        pd.concat(frames, ignore_index=True).to_parquet(path, index=False)
    print(f"{year} {path.stem.rsplit('_', 1)[0]}: +{got} rows ({len(todo)} days fetched)",
          flush=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    client = jquantsapi.ClientV2(api_key=JQUANTS_API_KEY)
    this_year = pd.Timestamp.today().year

    for year in range(START_YEAR, END_YEAR + 1):
        # 週次信用残: 週次粒度なので年レンジ一括（失敗時は強バックオフで再試行）
        p = OUT / f"margin_interest_{year}.parquet"
        if year >= this_year or not p.exists():
            df = _retry(lambda: client.get_mkt_margin_interest_range(
                start_dt=f"{year}0101", end_dt=f"{year}1231"))
            if len(df):
                df.to_parquet(p, index=False)
            print(f"{year} margin_interest: {len(df)} rows", flush=True)
            time.sleep(5)

        # 売買内訳: 日次逐次（レンジAPIは内部で高速連打し429になるため使わない）
        _collect_daily(client, year, OUT / f"breakdown_{year}.parquet", "Date",
                       lambda dd: client.get_mkt_breakdown(date_yyyymmdd=dd))

        # 増担保・規制銘柄（日々公表銘柄残高＋規制区分。売り建て可否ガードの正本）
        p = OUT / f"margin_alert_{year}.parquet"
        if year >= this_year or not p.exists():
            df = _retry(lambda: client.get_mkt_margin_alert_range(
                start_dt=f"{year}0101", end_dt=f"{year}1231"))
            if len(df):
                df["PubReason"] = df["PubReason"].astype(str)  # dict列→str(parquet安全)
                df.to_parquet(p, index=False)
            print(f"{year} margin_alert: {len(df)} rows", flush=True)
            time.sleep(5)

        # 空売り残高報告: 開示日ごと
        _collect_daily(client, year, OUT / f"short_positions_{year}.parquet", "DiscDate",
                       lambda dd: client.get_mkt_short_sale_report(disclosed_date=dd))

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
