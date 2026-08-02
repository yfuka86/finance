"""PUSH配信（WebSocket）が実際に動くかを確認する。発注なし・read-only。

    PYTHONPATH=. python scripts/check_push_feed.py [--n 50] [--seconds 60]

見るもの:
  1. 登録とWebSocket接続ができるか
  2. **PUSHメッセージが実際に飛んでくるか**（板が動く時間帯でしか飛ばない）
  3. スナップショットのスメア（最古と最新の差）が何秒に収まるか
     ← RESTの1周は686銘柄で十数分。ここが数秒なら「同時性のある観測」が成立する
"""
from __future__ import annotations

import argparse
import time

from trading.jp_intraday.daily_gap import load_existing_daily
from trading.jp_intraday.daily_model import build_daily_features
from trading.jp_intraday.live.config import LiveConfig
from trading.jp_intraday.live.kabu_client import KabuClient
from trading.jp_intraday.live.push_feed import MAX_REGISTERED, PushBoardFeed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=MAX_REGISTERED, help="登録銘柄数（最大50）")
    ap.add_argument("--seconds", type=int, default=60, help="観測秒数")
    args = ap.parse_args()

    cfg = LiveConfig.from_env()
    panel = build_daily_features(load_existing_daily(), min_value_yen=cfg.min_value_yen)
    last = panel[panel["date"].eq(panel["date"].max())]
    syms = list(last["symbol"])[:min(args.n, MAX_REGISTERED)]
    print(f"ユニバース{len(last)}銘柄 → 先頭{len(syms)}銘柄をPUSH登録します")

    client = KabuClient(cfg.api_password, cfg.order_password, env=cfg.env)
    client.authenticate()
    t0 = time.time()
    with PushBoardFeed(client, syms) as feed:
        ready = feed.wait_ready(timeout=90)
        print(f"  初期値の充足: {'OK' if ready else '未充足'} "
              f"({len(feed.snapshot())}/{len(syms)}銘柄・{time.time()-t0:.0f}秒)")
        print(f"  {args.seconds}秒間 PUSH を観測します…")
        start_msgs = feed.messages
        for i in range(args.seconds):
            time.sleep(1)
            if (i + 1) % 15 == 0:
                print(f"    {i+1}秒: 受信 {feed.messages - start_msgs}件 "
                      f"スメア {feed.smear_seconds():.1f}秒")
        got = feed.messages - start_msgs
        snap = feed.snapshot()
        ages = sorted(v["age_s"] for v in snap.values())

    print(f"\n結果: PUSH受信 {got}件 / {args.seconds}秒")
    if snap:
        print(f"  スナップショット {len(snap)}銘柄 ・ 経過秒 中央値{ages[len(ages)//2]:.1f} "
              f"最大{ages[-1]:.1f} ・ スメア {ages[-1]-ages[0]:.1f}秒")
    if got == 0:
        print("  → PUSHが1件も来ていない。板が動かない時間帯（引け後・休日）なら正常。"
              "\n     場中/寄前(08:00以降)に再実行して確認すること。")
        return 2
    print("  → PUSH配信は生きている。50銘柄までなら同時性のある観測が可能。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
