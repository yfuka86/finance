"""板寄せの成立がPUSH配信でいつ観測されるかを実測する（発注ゼロ・read-only）。

目的: 「取引所→kabuステーション→手元」の下り遅延の実測。
  引け(15:30)の板寄せ成立は取引所内で 15:30:00.000。その約定プリント
  （CurrentPriceTime が 15:30:00 のメッセージ）が手元に届いた受信時刻との差が
  下り遅延そのもの。これが小さければ、逆方向（発注→取引所）も同程度と推定でき、
  「08:59:5X 発注で板寄せに間に合うか」の判断材料になる。

    PYTHONPATH=. python scripts/capture_auction_push.py --auction 15:30
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import time
from pathlib import Path

from trading.jp_intraday.daily_gap import load_existing_daily
from trading.jp_intraday.daily_model import build_daily_features
from trading.jp_intraday.live.config import LiveConfig
from trading.jp_intraday.live.kabu_client import KabuClient
from trading.jp_intraday.live.push_experiment import wait_until
from trading.jp_intraday.live.push_feed import PushBoardFeed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--auction", default="15:30", help="板寄せ時刻 (09:00/12:30/15:30)")
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()
    hh, mm = args.auction.split(":")
    auction_epoch = dt.datetime.now().replace(hour=int(hh), minute=int(mm),
                                              second=0, microsecond=0).timestamp()

    cfg = LiveConfig.from_env()
    panel = build_daily_features(load_existing_daily(), min_value_yen=cfg.min_value_yen)
    last = panel[panel["date"].eq(panel["date"].max())]
    syms = list(last.nlargest(args.n, "prev_value")["symbol"])   # 流動性上位=確実に寄る
    c = KabuClient(cfg.api_password, cfg.order_password, env="prod")
    c.authenticate()

    with PushBoardFeed(c, syms) as feed:
        feed.record_history = True
        print(f"{len(syms)}銘柄を登録し {args.auction} の板寄せを観測します…")
        wait_until(f"{args.auction}:59")          # 成立後1分まで録る
        wait_until((dt.datetime.fromtimestamp(auction_epoch) +
                    dt.timedelta(seconds=90)).strftime("%H:%M:%S"))
        hist = list(feed.history)
        n_msg = feed.messages

    # 板寄せプリント = CurrentPriceTime がちょうど板寄せ時刻のメッセージ（銘柄ごとに最初の1本）
    prints = {}
    tgt = f"T{args.auction}:00"
    for recv, sym, cpt, cp, opx, clx in hist:
        if cpt and tgt in str(cpt) and sym not in prints:
            prints[sym] = recv - auction_epoch
    lat = sorted(prints.values())

    day = dt.date.today().isoformat()
    out = Path("data/live_reports") / f"auction_push_{args.auction.replace(':','')}_{day}.json"
    out.write_text(json.dumps({
        "day": day, "auction": args.auction, "n_symbols": len(syms),
        "messages": n_msg, "n_prints": len(lat),
        "latency_s": {"min": lat[0] if lat else None,
                      "median": statistics.median(lat) if lat else None,
                      "p90": lat[int(len(lat) * .9)] if lat else None,
                      "max": lat[-1] if lat else None},
        "per_symbol": {k: round(v, 3) for k, v in prints.items()},
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n受信 {n_msg}件 / 板寄せプリントを検出 {len(lat)}/{len(syms)}銘柄")
    if lat:
        print(f"板寄せ成立({args.auction}:00.000) → PUSH受信までの遅延:")
        print(f"  最小 {lat[0]:.2f}秒 / 中央値 {statistics.median(lat):.2f}秒 / "
              f"p90 {lat[int(len(lat)*.9)]:.2f}秒 / 最大 {lat[-1]:.2f}秒")
        print("  ※これは下り(取引所→手元)。発注の上りも同程度のオーダーと推定できる")
    print(f"保存: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
