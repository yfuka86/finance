"""50銘柄PUSH実験の実行（朝・発注なし・read-only）。

    PYTHONPATH=. python scripts/run_push_experiment.py [--select strategy] [--dry]

流れ:
  1. 全ユニバースをREST1周（スメアあり・各銘柄の取得時刻も記録）
  2. 候補50銘柄を選定（--select strategy|absgap|liquidity）
  3. PUSH登録 → 08:50/08:55/08:59 に同時スナップショット
  4. data/live_reports/push_experiment_YYYY-MM-DD.json に保存
夕方に scripts/analyze_push_experiment.py で実寄値と突き合わせる。
"""
from __future__ import annotations

import argparse
import datetime as dt
import time

import pandas as pd

from trading.jp_intraday.daily_gap import load_existing_daily
from trading.jp_intraday.daily_model import build_daily_features
from trading.jp_intraday.live import push_experiment as px
from trading.jp_intraday.live.config import LiveConfig
from trading.jp_intraday.live.kabu_client import KabuClient
from trading.jp_intraday.live.push_feed import PushBoardFeed


def sweep_all(client, symbols) -> dict:
    """全銘柄をREST1周。**各銘柄の取得時刻を残す**（スメアの実測値になる）。"""
    quotes, at = {}, {}
    t0 = time.time()
    for i, s in enumerate(symbols, 1):
        try:
            b = client.board(s)
        except Exception:  # noqa: BLE001
            continue
        q = px.quote_from_board(b)
        if q > 0:
            quotes[s] = q
            at[s] = time.time()
        if i % 90 == 0:
            el = time.time() - t0
            print(f"  sweep {i}/{len(symbols)} 経過{el:.0f}秒 "
                  f"残り~{el/i*(len(symbols)-i):.0f}秒", flush=True)
    return {"quotes": quotes, "at": at, "started": t0, "finished": time.time()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", default="strategy",
                    choices=["strategy", "absgap", "liquidity"])
    ap.add_argument("--n", type=int, default=px.MAX_PUSH)
    ap.add_argument("--dry", action="store_true",
                    help="時刻待ちをせず即座に1回スナップショット（動作確認用）")
    ap.add_argument("--names-per-side", type=int, default=8,
                    help="本番基準の銘柄数/側（既定8＝本番構成。.envの縮小設定に引きずられない）")
    ap.add_argument("--max-sweep", type=int, default=0,
                    help="1周する銘柄数の上限（0=全部。夜間のスモークテスト用）")
    args = ap.parse_args()

    cfg = LiveConfig.from_env()
    print(f"CONFIG: {cfg.summary()}")
    panel = build_daily_features(load_existing_daily(), min_value_yen=cfg.min_value_yen)
    last = panel[panel["date"].eq(panel["date"].max())].copy()
    syms = list(last["symbol"])
    if args.max_sweep:
        syms = syms[:args.max_sweep]
        last = last[last["symbol"].isin(syms)].copy()
        print(f"※--max-sweep {args.max_sweep}: スイープを絞っています（スモークテスト用）")
    print(f"ユニバース {len(syms)}銘柄 / データ日 {panel['date'].max().date()}")

    client = KabuClient(cfg.api_password, cfg.order_password, env=cfg.env)
    client.authenticate()

    print("① 全銘柄をREST1周（スメアあり）…")
    sweep = sweep_all(client, syms)
    smear = (max(sweep["at"].values()) - min(sweep["at"].values())) if sweep["at"] else 0
    print(f"   取得 {len(sweep['quotes'])}/{len(syms)}銘柄 ・ 所要 "
          f"{sweep['finished']-sweep['started']:.0f}秒 ・ **スメア {smear:.0f}秒**")

    nps = args.names_per_side
    print(f"② 候補{args.n}銘柄を選定（方式={args.select}・{nps}銘柄/側/スリーブ）…")
    diag = {}
    if args.select == "strategy":
        # 本番と同じ建て方（スリーブごとに上位/下位→統合）で建玉を作り、
        # 残り枠を次点で埋める＝2パス方式の1パス目そのもの
        chosen, diag = px.select_for_ensemble(last, sweep["quotes"], cfg.strategy,
                                              nps, args.n)
        early_book = set(diag["book"])
        for m, b in diag["per_sleeve"].items():
            print(f"   スリーブ {m}: 建玉{len(b)}銘柄")
    else:
        scored = px.score_frame(last, sweep["quotes"], cfg.strategy)
        if "prev_value" not in scored:
            scored["prev_value"] = last.set_index("symbol")["prev_value"].reindex(
                scored["symbol"]).values
        chosen = px.select_symbols(scored, args.select, args.n, nps)
        early_book = px.book_from_scores(scored, nps)
    print(f"   候補 {len(chosen)}銘柄 / 早い1周での建玉 {len(early_book)}銘柄 "
          f"（うち候補に含まれる {len(early_book & set(chosen))}）"
          f"{' / 次点はn=' + str(diag['depth_used']) + 'まで採用' if diag else ''}")

    record = {
        "day": dt.date.today().isoformat(),
        "select": args.select,
        "names_per_side": nps,
        "select_diag": diag,
        "universe_n": len(syms),
        "sweep": {"n": len(sweep["quotes"]), "smear_s": smear,
                  "started": sweep["started"], "finished": sweep["finished"],
                  "quotes": sweep["quotes"],
                  "at": {k: v for k, v in sweep["at"].items() if k in chosen}},
        "early_book": sorted(early_book),
        "chosen": chosen,
        "snapshots": {},
    }
    px.save(record, dry=args.dry)

    print("③ PUSH登録して同時スナップショット…")
    with PushBoardFeed(client, chosen) as feed:
        feed.wait_ready(timeout=120)
        times = ["now"] if args.dry else list(px.SNAP_TIMES)
        for hhmm in times:
            if hhmm != "now":
                px.wait_until(hhmm)
            snap = feed.snapshot()
            record["snapshots"][hhmm] = {
                "taken": time.time(),
                # ★smear は「気配が動いていない時間」であって測定誤差ではない。
                #   同時性の担保は health()（接続・全銘柄が配信を受けているか）で見る。
                "quiet_s": feed.smear_seconds(),
                "health": feed.health(),
                "push_updated_symbols": feed.push_updated(),
                "push_messages": feed.messages,
                "quotes": {k: {"q": px.quote_from_board(v["board"]),
                               "age_s": round(v["age_s"], 2),
                               "source": v["source"], "updates": v["updates"],
                               "bid": v["board"].get("BidPrice"),
                               "ask": v["board"].get("AskPrice"),
                               "calc": v["board"].get("CalcPrice")}
                           for k, v in snap.items()},
            }
            h = feed.health()
            print(f"   {hhmm}: {len(snap)}銘柄（PUSH更新済み {feed.push_updated()}）"
                  f" ・ 健全性 {'OK' if h['ok'] else 'NG:' + str(h['never_pushed'][:3])}"
                  f" ・ 静止時間 {feed.smear_seconds():.0f}秒（誤差ではない）"
                  f" ・ PUSH受信 {feed.messages}件")
            px.save(record, dry=args.dry)

    p = px.save(record, dry=args.dry)
    print(f"\n保存: {p}")
    print("夕方に `PYTHONPATH=. python scripts/analyze_push_experiment.py` で判定材料を出す")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
