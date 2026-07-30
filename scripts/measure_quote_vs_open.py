"""プレオープン気配(08:5x CalcPrice) vs 実寄値 の乖離を実測する。

実運用のプラン（test/prod）は est_price=プラン生成時の板気配 を持って
https://trade.a-tokyo.jp に記録される。これを J-Quants の実寄値（無調整O）と
突き合わせ、「気配で銘柄選択→実寄値で約定」の選択ノイズを日次で定量化する。

シグナルの主成分は残差ギャップなので、乖離は bps だけでなく
「ギャップ計算に与える影響（乖離/前日終値）」でも報告する。

実行:  PYTHONPATH=. python scripts/measure_quote_vs_open.py [--days 10]
出力例: 日付ごとの n銘柄・乖離bps分布(中央値/p90/最大)・サイド別。
数日分貯まったら live/README の判断基準（ノイズ予算）と突き合わせること。
"""
import argparse
import json
import urllib.request

import pandas as pd

REPORTS_URL = "https://trade.a-tokyo.jp/api/reports?limit=200"


def _load_reports() -> list[dict]:
    with urllib.request.urlopen(REPORTS_URL, timeout=30) as r:
        rows = json.loads(r.read())
    return rows if isinstance(rows, list) else rows.get("reports", [])


def _daily_opens() -> pd.DataFrame:
    """無調整の寄値（気配と同じスケール）。daily_adj_* は生O列を保持している。"""
    import glob
    frames = []
    for f in sorted(glob.glob("data/jp_daily_history/daily_adj_202[5-9].parquet")):
        d = pd.read_parquet(f, columns=["Date", "Code", "O"])
        frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["Date"]).dt.date
    out["Code"] = out["Code"].astype(str)
    return out[["date", "Code", "O"]].dropna()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=10)
    args = ap.parse_args()

    plans = [r for r in _load_reports()
             if r.get("event") in ("plan", "entry") and r.get("env") in ("test", "prod")]
    if not plans:
        print("実運用（test/prod）のプラン記録がまだありません。mockは対象外。")
        return
    opens = _daily_opens()

    seen_dates = set()
    for r in sorted(plans, key=lambda x: x.get("received", ""), reverse=True):
        # planイベントのest_price=プラン生成時CalcPrice。取引日はreceived(UTC)→JSTの日付
        ts = pd.Timestamp(r.get("received")).tz_localize("UTC").tz_convert("Asia/Tokyo")
        tdate = ts.date()
        if tdate in seen_dates or len(seen_dates) >= args.days:
            continue
        seen_dates.add(tdate)
        rows = (r.get("data") or {}).get("plan") or (r.get("data") or {}).get("orders") or []
        df = pd.DataFrame(rows)
        if df.empty or "est_price" not in df.columns:
            continue
        df["Code"] = df["symbol"].astype(str)
        m = df.merge(opens[opens["date"] == tdate], on="Code", how="left").dropna(subset=["O"])
        if m.empty:
            print(f"{tdate}: 実寄値データ未着（翌日以降に再実行）")
            continue
        dev = (m["O"] / m["est_price"] - 1) * 1e4
        print(f"\n{tdate} ({r.get('env')}) n={len(m)}/{len(df)}銘柄")
        print(f"  気配→寄値乖離bps: 中央値{dev.abs().median():.1f} / p90 {dev.abs().quantile(0.9):.1f}"
              f" / 最大 {dev.abs().max():.1f} / 平均符号付き {dev.mean():+.1f}")
        for side in ("LONG", "SHORT"):
            s = dev[m["side_label"] == side]
            if len(s):
                print(f"  {side}: 中央値|{s.abs().median():.1f}| 最大|{s.abs().max():.1f}|")


if __name__ == "__main__":
    main()
