"""寄付き直前の発注が板寄せに間に合うかを実測する（発注時刻のラダーテスト）。

なぜ必要か:
  APIの受理(実測3ms)と板寄せ参加は別物。2026-07-29/30 の実障害では、09:17送信の
  寄成が Result=0 で受理されたのに板寄せに乗れず State5 で全滅した。
  「08:59:50 に決定 → 発注」という運用が成立するかは、**実際に寄付き直前に
  発注して寄値で約定するか**でしか検証できない。

設計（実弾・ただし極小）:
  銘柄: 8918 ランド（~¥10 × 100株 = 約¥1,000/注文。発注経路プローブの実績銘柄・
        売買高上位で09:00に確実に寄る）
  送信: 寄成(13)買いを 08:59:30 / 08:59:50 / 08:59:57 / 09:00:05(負の対照) の4本
  判定: 09:00:45 に /orders を照会 →
        寄値で約定 = 板寄せに参加できた / 0約定・State5 = 間に合わなかった
  後始末: 約定分は市場成行(10)で即返済（建玉を残さない）
  総リスク: 想定元本 ~¥4,000・往復スプレッド数円〜数十円

実行（安全ゲート: --arm が無ければ何もしない）:
  PYTHONPATH=. python scripts/verify_auction_cutoff.py --arm
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

from trading.jp_intraday.live.config import LiveConfig
from trading.jp_intraday.live.kabu_client import KabuAPIError, KabuClient
from trading.jp_intraday.live.push_experiment import wait_until

SYMBOL = "8918"
QTY = 100
MAX_NOTIONAL = 50_000          # これを超える銘柄なら中止（想定は~¥1,000）
LADDER = ("08:59:30", "08:59:50", "08:59:57", "09:00:05")
OUT = Path("data/live_reports")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="store_true", help="実弾で実行（無ければ表示のみ）")
    args = ap.parse_args()

    cfg = LiveConfig.from_env()
    c = KabuClient(cfg.api_password, cfg.order_password, env="prod")
    c.authenticate()

    b = c.board(SYMBOL)
    px = float(b.get("PreviousClose") or b.get("CurrentPrice") or 0)
    notional = px * QTY
    print(f"銘柄 {SYMBOL} {b.get('SymbolName')} 前日終値¥{px} × {QTY}株 = 想定¥{notional:,.0f}/注文")
    if not (0 < notional <= MAX_NOTIONAL):
        print(f"中止: 想定元本が上限¥{MAX_NOTIONAL:,}を超過")
        return 2
    if not args.arm:
        print("（--arm 未指定なので発注しません。ラダー:", ", ".join(LADDER), "）")
        return 0

    day = dt.date.today().isoformat()
    rec = {"day": day, "symbol": SYMBOL, "qty": QTY, "prev_close": px, "orders": []}
    for t in LADDER:
        wait_until(t)
        sent_at = dt.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        t0 = time.perf_counter()
        try:
            r = c.send_margin_open(SYMBOL, "2", QTY, front_order_type=13,
                                   margin_type=3, account_type=cfg.account_type)
            entry = {"target": t, "sent_at": sent_at,
                     "rtt_ms": round((time.perf_counter() - t0) * 1000),
                     "order_id": r.get("OrderId"), "result": 0}
        except KabuAPIError as e:
            entry = {"target": t, "sent_at": sent_at,
                     "rtt_ms": round((time.perf_counter() - t0) * 1000),
                     "order_id": None, "result": str(e)[:120]}
        rec["orders"].append(entry)
        print(f"  {t} 送信 {sent_at} rtt={entry['rtt_ms']}ms -> {entry.get('order_id') or entry['result']}")

    wait_until("09:00:45")
    open_px = None
    try:
        open_px = c.board(SYMBOL).get("OpeningPrice")
    except Exception:  # noqa: BLE001
        pass
    rec["opening_price"] = open_px
    orders = {o.get("ID"): o for o in c.orders(product=2)}
    print(f"\n寄値 = {open_px}")
    for e in rec["orders"]:
        o = orders.get(e.get("order_id"), {})
        e["state"] = o.get("State")
        e["cum_qty"] = o.get("CumQty")
        e["fill_price"] = o.get("Price")
        filled = bool(e["cum_qty"]) and float(e["cum_qty"] or 0) > 0
        verdict = ("✅板寄せ参加（寄値で約定）" if filled and open_px and
                   abs(float(e.get("fill_price") or 0) - float(open_px)) < 1e-9
                   else ("⚠️約定したが寄値と不一致" if filled else "❌不参加（未約定）"))
        e["verdict"] = verdict
        print(f"  {e['target']}: state={e['state']} fill={e['cum_qty']}@{e['fill_price']} → {verdict}")

    # ── 後始末: 建玉を市場成行で即返済 ─────────────────────────
    closed = []
    for p in c.positions(product=2):
        if str(p.get("Symbol")) != SYMBOL:
            continue
        free = int(float(p.get("LeavesQty") or 0) - float(p.get("HoldQty") or 0))
        if free <= 0:
            continue
        try:
            r = c.send_margin_close(SYMBOL, "1", free, p["ExecutionID"],
                                    front_order_type=10, margin_type=3,
                                    account_type=cfg.account_type)
            closed.append({"hold_id": p["ExecutionID"], "qty": free,
                           "order_id": r.get("OrderId")})
        except KabuAPIError as e:
            closed.append({"hold_id": p["ExecutionID"], "qty": free, "error": str(e)[:120]})
        time.sleep(0.2)
    rec["closes"] = closed
    print(f"返済注文: {len(closed)}件")

    out = OUT / f"auction_cutoff_{day}.json"
    out.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"保存: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
