"""実効コストの脚別実測（寄り/引けを分離して記録する）.

なぜ分けるか（2026-07-30 R9の実測）: 引け板寄せ(15:30)は日次売買代金の13.5%を占め、
寄付板寄せ(9:00)の**3.4倍厚い**。つまり同じ建玉でも参加率は寄り側が4倍で、
実効コストは構造的に非対称なはず。往復合算で持つと入口の悪さが出口に隠される。

測る量（インプリメンテーション・ショートフォール）:
  entry脚: (実約定価格 − 公式寄値) / 公式寄値 × 符号   ※買いは高く買うほど正=コスト
  exit脚 : (公式引値 − 実約定価格) / 公式引値 × 符号   ※売りは安く売るほど正=コスト
板寄せは単一約定価格なので理論上ゼロで、**残るのは自分の注文が約定価格を動かした分
（マーケットインパクト）**。したがってこの実測値がそのままインパクトの推定になる。

公式寄値/引値は J-Quants の日次バーから翌営業日に取得する（当日は kabu の
board から取れる CurrentPrice を暫定値として使い、翌日に確定値で上書きする）。

実行:
  python -m trading.jp_intraday.live.run_live cost        # 当日分を記録（15:40以降）
  PYTHONPATH=. python scripts/analyze_effective_cost.py   # 蓄積分を集計・閾値判定
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

_STATE_DIR = Path("data/live_reports")
SIDE_BUY, SIDE_SELL = "2", "1"


def _fills_from_orders(orders: list) -> list[dict]:
    """kabu /orders のレスポンスから約定明細を取り出す（防御的パース）.

    Details は環境差があるため、RecType/Price/Qty の組で「約定(RecType=8)」を拾い、
    取れない場合は注文サマリの Price/CumQty にフォールバックする。
    """
    out = []
    for o in orders or []:
        sym = str(o.get("Symbol", ""))
        side = str(o.get("Side", ""))
        # front order type: 13=寄成(entry) / 16=引成(exit) を脚の判定に使う
        front = str(o.get("FrontOrderType", o.get("OrdType", "")))
        details = o.get("Details") or []
        px_qty = []
        for d in details if isinstance(details, list) else []:
            if str(d.get("RecType")) == "8":            # 8 = 約定
                p, q = d.get("Price"), d.get("Qty")
                if p and q:
                    px_qty.append((float(p), float(q)))
        if not px_qty:
            p, q = o.get("Price"), o.get("CumQty") or o.get("OrderQty")
            if p and q:
                px_qty.append((float(p), float(q)))
        if not px_qty:
            continue
        qty = sum(q for _, q in px_qty)
        vwap = sum(p * q for p, q in px_qty) / qty if qty else None
        if vwap:
            out.append({"symbol": sym, "side": side, "front": front,
                        "fill_px": vwap, "qty": qty})
    return out


def leg_slippage_bps(fill_px: float, ref_px: float, side: str, leg: str) -> float:
    """1脚の実効コスト(bps・正=コスト). leg は "entry" か "exit"."""
    if not fill_px or not ref_px:
        return float("nan")
    raw = (fill_px / ref_px - 1.0) * 1e4
    # entry: 買いは高いほどコスト / 売建は安いほどコスト
    # exit : 買戻しは高いほどコスト / 売却は安いほどコスト（符号は entry と逆）
    sign = 1.0 if side == SIDE_BUY else -1.0
    return raw * sign if leg == "entry" else -raw * sign


def record_daily_costs(client, cfg, ref_prices: dict | None = None,
                       day: str | None = None) -> dict:
    """当日の約定を脚別に集計して JSONL に追記し、サマリを返す.

    ref_prices: {kabu_symbol: {"open": x, "close": y}}。None なら kabu の board から
    暫定取得（当日中の速報値。翌日 analyze_effective_cost.py が J-Quants の確定値で上書き）。
    """
    import datetime as dt
    day = day or dt.date.today().isoformat()
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = _STATE_DIR / f"cost_{day}.jsonl"

    fills = _fills_from_orders(client.orders(product=2))
    rows = []
    for f in fills:
        leg = "entry" if f["front"] in ("13", "10") else "exit" if f["front"] == "16" else "?"
        ref = (ref_prices or {}).get(f["symbol"], {})
        if not ref:
            try:
                b = client.board(f["symbol"])
                ref = {"open": b.get("OpeningPrice"), "close": b.get("ClosingPrice")}
            except Exception:  # noqa: BLE001
                ref = {}
        ref_px = ref.get("open") if leg == "entry" else ref.get("close")
        rows.append({
            "day": day, "symbol": f["symbol"], "side": f["side"], "leg": leg,
            "fill_px": f["fill_px"], "qty": f["qty"], "ref_px": ref_px,
            "slip_bps": leg_slippage_bps(f["fill_px"], float(ref_px or 0), f["side"], leg),
            "notional": f["fill_px"] * f["qty"],
        })
    with path.open("a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    d = pd.DataFrame(rows)
    summary = {"day": day, "n_fills": len(rows), "log": str(path)}
    if len(d):
        d = d[d["slip_bps"].notna()]
        for leg in ("entry", "exit"):
            sub = d[d["leg"] == leg]
            if len(sub):
                w = sub["notional"]
                summary[f"{leg}_bps_wavg"] = float((sub["slip_bps"] * w).sum() / w.sum())
                summary[f"{leg}_bps_median"] = float(sub["slip_bps"].median())
                summary[f"{leg}_n"] = int(len(sub))
        if {"entry_bps_wavg", "exit_bps_wavg"} <= summary.keys():
            summary["roundtrip_bps"] = summary["entry_bps_wavg"] + summary["exit_bps_wavg"]
    return summary
