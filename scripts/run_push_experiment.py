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
import json
import datetime as dt
import time

import pandas as pd

from trading.jp_intraday.daily_gap import load_existing_daily
from trading.jp_intraday.daily_model import build_daily_features
from trading.jp_intraday.live import push_experiment as px
from trading.jp_intraday.live.config import LiveConfig
from trading.jp_intraday.live.kabu_client import KabuClient, to_kabu_symbol
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
        # RESTバースト対象（候補50の外側・近いtier優先）。08:57以降に叩く
        burst_names = px.burst_list(last, sweep["quotes"], cfg.strategy, nps, chosen)
        print(f"   バースト対象 {len(burst_names)}銘柄（時間内に叩けた分だけ記録）")
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
        # 全メッセージ履歴を録る: 後から**任意の時刻**の選択を再構成できる
        # （スナップショット9点ではなく連続曲線）。寄値もPUSHの約定プリント
        # (CurrentPriceTime=09:00:00) から直接取れる。
        feed.record_history = True
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

            # ── RESTバースト: 08:57スナップの直後に候補外~120銘柄を直前気配で叩く ──
            # 器（候補50）が実効一致率の律速（A=17-50%）なので、最後の2分でプールを
            # ~170銘柄に広げる。GET /board は自動登録で50枠を食うため、PUSH登録を
            # 一旦解除して叩き、寄値プリント回収のため最後に candidates を再登録する。
            if hhmm == "08:57" and not args.dry and burst_names:
                client.unregister_all()
                client._board_calls = 0          # board()の45件ごと自動解除と整合させる
                deadline = dt.datetime.now().replace(hour=8, minute=59, second=15,
                                                     microsecond=0).timestamp()
                burst: dict = {}
                for s in burst_names:
                    if time.time() >= deadline:
                        break
                    try:
                        b = client.board(s)      # 45件ごとにunregister_allしてくれる
                    except Exception:  # noqa: BLE001
                        continue
                    q = px.quote_from_board(b)
                    if q > 0:
                        burst[s] = {"q": q, "at": round(time.time(), 2),
                                    "bid": b.get("BidPrice"), "ask": b.get("AskPrice")}
                record["burst"] = {"n_target": len(burst_names), "n_got": len(burst),
                                   "quotes": burst}
                print(f"   burst: {len(burst)}/{len(burst_names)}銘柄を取得"
                      f"（プール={len(burst) + len(chosen)}）")
                # 寄値プリント回収のため再登録（PUSH再開）
                try:
                    client.unregister_all()
                    client._request("PUT", "/register", json={
                        "Symbols": [{"Symbol": to_kabu_symbol(s), "Exchange": 1}
                                    for s in chosen]})
                    print("   burst後にcandidates 50を再登録（寄値プリント回収用）")
                except Exception as exc:  # noqa: BLE001
                    print(f"   再登録に失敗: {exc}")
                px.save(record, dry=args.dry)

        if not args.dry:
            # 寄付きの約定プリント(09:00:00台のCurrentPriceTime)まで録り切ってから
            # 履歴を書き出す。**withブロックの中で行う**（外に出るとfeedが閉じて履歴が消える。
            # 2026-08-05: このブロックがサイレントな置換失敗で欠落し履歴を1日分失った）
            # 09:06まで録る: 特別気配の銘柄は寄りが数分遅れる。09:00:40打ち切りだと
            # 寄値プリントが29/50しか取れず（2026-08-06実測）、遅れて寄る銘柄＝
            # 大ギャップ銘柄＝一番測りたい銘柄が抜ける上方バイアスになる
            px.wait_until("09:06:00")
            hist_path = px.out_path().with_name(
                px.out_path().stem.replace("push_experiment", "push_history") + ".jsonl")
            with hist_path.open("w", encoding="utf-8") as hf:
                for row in feed.history:
                    hf.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            record["history_file"] = str(hist_path)
            record["history_rows"] = len(feed.history)
            print(f"   受信履歴 {len(feed.history)}行 → {hist_path}")

    p = px.save(record, dry=args.dry)
    print(f"\n保存: {p}")
    print("夕方に `PYTHONPATH=. python scripts/analyze_push_experiment.py` で判定材料を出す")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
