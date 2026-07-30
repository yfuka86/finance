"""板寄せ執行の監視指標（2026-07-30 全面改訂）.

**重要な訂正**: 当初この module は「実約定価格 vs 公式寄値/引値」の乖離を
実効コストとして測ろうとしていたが、**これは原理的に測れない**。
寄成・引成は板寄せ＝単一約定価格なので、約定できた限り fill == 公式寄値/引値 で
**乖離は構造的にゼロ**になる。自分のインパクトは公式価格そのものに内包されており、
バックテストも同じ価格を使うため、実現P&Lとの突合からは分離不能。

したがってこの module が測るのは**コストそのものではなく、コスト前提が崩れる兆候**:
  1. **参加率** = 自分の約定代金 / その板寄せの総約定代金
     → インパクトの唯一の観測可能な driver（実測: 寄り¥加重1.0-1.3% / 引け0.44%）
  2. **数量約定率** = 約定株数 / 発注株数（寄らず・部分約定の検知）
  3. **|slip|** = fill vs 公式価格の乖離。**0が正常**。非ゼロは
     SOR が PTS へ回した等の「板寄せ外約定」のシグナル（コストではなく異常検知）
  4. プレミアム料（一日信用）の発生 — ここだけは桁が違う（上限100bps/日）

監視閾値（live/README と同一）: 参加率 p90 が 寄り≤3% / 引け≤1.5%、
数量約定率 ≥98%、|slip| ≤0.5bps。いずれかを外れたらコスト前提の再検討。

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
                        "fill_px": vwap, "qty": qty,
                        "ordered_qty": float(o.get("OrderQty") or 0) or None})
    return out


def leg_slippage_bps(fill_px: float, ref_px: float, side: str, leg: str) -> float:
    """fill と公式板寄せ価格の乖離(bps・正=不利側).

    **板寄せで約定した限りこれは 0 になる**（単一約定価格のため）。
    したがって「コストの実測値」ではなく、**板寄せ外約定（SORのPTS迂回等）の
    異常検知指標**として使う。leg は "entry" か "exit"。
    """
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
        # 参加率の分母（その銘柄のその板寄せの総約定代金）を board から拾えれば記録する。
        # kabu の板情報に板寄せ出来高が無い環境では None のまま（翌日J-Quantsで補完）。
        auction_val = None
        try:
            b = client.board(f["symbol"])
            vol = b.get("OpeningPriceVolume") if leg == "entry" else b.get("ClosingPriceVolume")
            if vol and ref_px:
                auction_val = float(vol) * float(ref_px)
        except Exception:  # noqa: BLE001
            pass
        notional = f["fill_px"] * f["qty"]
        rows.append({
            "day": day, "symbol": f["symbol"], "side": f["side"], "leg": leg,
            "fill_px": f["fill_px"], "qty": f["qty"], "ref_px": ref_px,
            "ordered_qty": f.get("ordered_qty"),
            "fill_ratio": (f["qty"] / f["ordered_qty"]) if f.get("ordered_qty") else None,
            "slip_bps": leg_slippage_bps(f["fill_px"], float(ref_px or 0), f["side"], leg),
            "notional": notional,
            "auction_value": auction_val,
            "participation_pct": (notional / auction_val * 100) if auction_val else None,
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
        for leg in ("entry", "exit"):
            sub = d[(d["leg"] == leg) & d["participation_pct"].notna()]
            if len(sub):
                summary[f"{leg}_participation_p50"] = float(sub["participation_pct"].median())
                summary[f"{leg}_participation_p90"] = float(sub["participation_pct"].quantile(0.9))
        fr = d["fill_ratio"].dropna()
        if len(fr):
            summary["fill_ratio_mean"] = float(fr.mean())
            summary["fill_ratio_min"] = float(fr.min())
    return summary
