"""
Trade executor: PCA_SUB シグナルに基づいて kabuステーションAPI で発注する。
VPS上で毎営業日実行する想定。

バックテスト (backtest/strategies/pca_sub.py) は JP を Open→Close で評価している。
つまり寄付きで建てて大引けで決済する日次の往復取引であり、
executor も entry / exit の2フェーズで動く。

実行タイミング:
  entry: 8:55頃に起動 → 9:00 寄付き成行で建てる
  exit : 14:55頃に起動 → 大引け前に成行で全決済

  python -m trading.executor entry           # ドライラン (発注しない)
  python -m trading.executor entry --live    # 実発注
  python -m trading.executor exit  --live    # 全決済

ショートは信用売り (CashMargin=2)。往復するのでロングも既定で信用建てにしている
(現物で同日に買って売ると差金決済の制約を受けるため)。
"""
import argparse
import datetime
import sys

import numpy as np

from backtest.strategies.pca_sub import compute_signal_latest
from data.collect import collect
from data.collectors.config import JP_TICKERS
from trading import broker as bk
from trading.broker import KabuApiError, KabuStationClient
from trading.config import KABU_MARGIN_TRADE_TYPE, TARGET_NOTIONAL_PER_LEG

# JP ETF の銘柄コード (kabuステーションAPI は4桁コード。"1617.T" -> "1617")
JP_CODES = {t: t.replace(".T", "") for t in JP_TICKERS}

# /symbol が売買単位を返さない場合のフォールバック (NEXT FUNDS TOPIX-17 は 10口)
DEFAULT_TRADING_UNIT = 10


def compute_today_signal(us_ret=None, jp_ret=None, **params):
    """
    直近のUSリターンから、次のJPセッション向けの PCA_SUB シグナルを計算する。

    バックテストと同一の実装 (pca_sub.compute_signal_latest) を呼ぶので、
    シグナル定義が検証結果とズレない。
    """
    if us_ret is None or jp_ret is None:
        us_ret, jp_ret, _jp_am, _jp_pm = collect()
    if us_ret.empty or jp_ret.empty:
        raise RuntimeError(
            f"リターン行列が空です (US={us_ret.shape}, JP={jp_ret.shape})。"
            " data/raw/ を削除して `python -m data.collect` を再実行してください。"
        )
    return compute_signal_latest(us_ret, jp_ret, **params)


# --- 建玉サイジング ---

def _trading_unit(client, code):
    try:
        info = client.symbol(code)
    except KabuApiError as e:
        print(f"    /symbol 取得失敗 ({e.message}) → 売買単位 {DEFAULT_TRADING_UNIT} を仮定")
        return DEFAULT_TRADING_UNIT
    unit = info.get("TradingUnit")
    if not unit:
        print(f"    売買単位が空 → {DEFAULT_TRADING_UNIT} を仮定")
        return DEFAULT_TRADING_UNIT
    return int(unit)


def _reference_price(client, code):
    """成行サイジング用の参照価格。現在値→前日終値の順で拾う。"""
    try:
        b = client.board(code)
    except KabuApiError as e:
        print(f"    /board 取得失敗: {e.message}")
        return None
    for field in ("CurrentPrice", "PreviousClose", "OpeningPrice"):
        if b.get(field):
            return float(b[field])
    return None


def size_orders(client, targets, notional_per_leg=TARGET_NOTIONAL_PER_LEG):
    """
    ターゲット (ticker, side) から発注数量を決める。

    Returns: list of dict(ticker, code, side, qty, price, notional)
    """
    plans = []
    for ticker, side in targets:
        code = JP_CODES[ticker]
        price = _reference_price(client, code)
        unit = _trading_unit(client, code)
        if price is None:
            print(f"  [SKIP] {ticker} ({code}): 参照価格が取得できず数量を決められません")
            continue
        lots = int(notional_per_leg // (price * unit))
        if lots < 1:
            print(f"  [SKIP] {ticker} ({code}): 1単位 {price * unit:,.0f}円 > "
                  f"上限 {notional_per_leg:,.0f}円")
            continue
        qty = lots * unit
        plans.append({
            "ticker": ticker, "code": code, "side": side, "qty": qty,
            "price": price, "notional": qty * price,
        })
    return plans


# --- 発注 ---

def execute_entries(client, plans, live=False, margin_trade_type=KABU_MARGIN_TRADE_TYPE,
                    long_with_cash=False):
    """
    寄付き成行でエントリー。

    long_with_cash=True にするとロングだけ現物買いにする
    (同日決済しない運用に切り替える場合のみ)。
    """
    results = []
    for p in plans:
        tag = "BUY " if p["side"] == "LONG" else "SELL"
        desc = (f"{tag} {p['ticker']} ({p['code']}) qty={p['qty']} "
                f"@~{p['price']:,.0f} = {p['notional']:,.0f}円")
        if p["side"] == "LONG":
            kind = "現物買い" if long_with_cash else "信用新規買い"
        else:
            kind = "信用新規売り"
        print(f"  {desc}  [{kind}]")

        if not live:
            results.append({**p, "kind": kind, "result": "dry-run"})
            continue

        try:
            if p["side"] == "LONG":
                if long_with_cash:
                    r = client.buy_market(p["code"], p["qty"])
                else:
                    r = client.margin_buy_open(
                        p["code"], p["qty"], margin_trade_type=margin_trade_type
                    )
            else:
                r = client.margin_sell_open(
                    p["code"], p["qty"], margin_trade_type=margin_trade_type
                )
            print(f"    -> OrderId={r.get('OrderId')}")
            results.append({**p, "kind": kind, "result": r})
        except KabuApiError as e:
            print(f"    -> 発注失敗: {e}")
            results.append({**p, "kind": kind, "error": str(e)})
    return results


def execute_exits(client, live=False, margin_trade_type=KABU_MARGIN_TRADE_TYPE):
    """
    信用建玉をすべて成行返済する。現物保有には手を出さない。
    """
    positions = client.positions(product=2)  # 2=信用
    if not positions:
        print("  信用建玉なし")
        return []

    results = []
    for pos in positions:
        code = pos.get("Symbol")
        hold_id = pos.get("HoldID")
        qty = int(pos.get("LeavesQty") or pos.get("Qty") or 0)
        pos_side = str(pos.get("Side"))  # 建玉の売買区分
        if qty <= 0 or not hold_id:
            continue

        # 建玉と反対側で返済する
        close_side = bk.SIDE_BUY if pos_side == bk.SIDE_SELL else bk.SIDE_SELL
        kind = "売建の返済買い" if pos_side == bk.SIDE_SELL else "買建の返済売り"
        print(f"  CLOSE {code} qty={qty} HoldID={hold_id}  [{kind}]")

        if not live:
            results.append({"code": code, "qty": qty, "result": "dry-run"})
            continue

        try:
            r = client.margin_close(
                code, qty, close_side,
                margin_trade_type=margin_trade_type,
                close_positions=[{"HoldID": hold_id, "Qty": qty}],
            )
            print(f"    -> OrderId={r.get('OrderId')}")
            results.append({"code": code, "qty": qty, "result": r})
        except KabuApiError as e:
            print(f"    -> 返済失敗: {e}")
            results.append({"code": code, "qty": qty, "error": str(e)})
    return results


# --- シグナル → ターゲット ---

def signal_to_targets(sig):
    """compute_signal_latest の結果をロング/ショートの銘柄リストに変換する。"""
    weights, tickers = sig["weights"], sig["jp_tickers"]
    longs = [tickers[i] for i in np.where(weights > 0)[0]]
    shorts = [tickers[i] for i in np.where(weights < 0)[0]]
    return ([(t, "LONG") for t in longs] + [(t, "SHORT") for t in shorts],
            longs, shorts)


def preflight_margin_sell(client, shorts):
    """
    ショート予定銘柄が信用売建可能かを確認する。
    検証環境は /symbol が空を返すので判定不能。その場合は警告のみ。
    """
    print("\n--- 信用売建可否チェック ---")
    blocked, unknown = [], []
    for ticker in shorts:
        code = JP_CODES[ticker]
        try:
            ok, flags = client.can_margin_sell(code)
        except KabuApiError as e:
            print(f"  {ticker} ({code}): 確認失敗 {e.message}")
            unknown.append(ticker)
            continue
        if all(v is None for v in flags.values()):
            print(f"  {ticker} ({code}): 判定不能 (フィールドが空)")
            unknown.append(ticker)
        elif ok:
            print(f"  {ticker} ({code}): 売建可 {flags}")
        else:
            print(f"  {ticker} ({code}): 売建不可 {flags}")
            blocked.append(ticker)
    if blocked:
        print(f"\n  [警告] 売建不可 {len(blocked)}銘柄: {blocked}")
    if unknown:
        print(f"  [注意] 判定不能 {len(unknown)}銘柄 (検証環境では常にこうなります)")
    return blocked


# --- エントリポイント ---

def run_entry(client, args):
    print("\n--- シグナル計算 ---")
    sig = compute_today_signal(fresh_us=args.fresh_us)
    print(f"  USセッション: {sig['us_date']}  直近JPセッション: {sig['jp_date']}")
    print(f"  サンプル数: {sig['n_days']}日")
    print(f"  fresh_us: {sig['fresh_us']}"
          + ("  [注意] USが未更新のため1日古いシグナルにフォールバック"
             if sig["stale_us"] else ""))
    if sig["used_fallback_C_full"]:
        print("  [警告] C_full を指定期間から作れず、先頭600日で代用しています "
              "(その期間は in-sample)")

    order = np.argsort(sig["signal"])[::-1]
    print("\n  シグナル (降順):")
    for i in order:
        t = sig["jp_tickers"][i]
        w = sig["weights"][i]
        mark = "LONG " if w > 0 else ("SHORT" if w < 0 else "  -  ")
        print(f"    {mark} {t:>8}  signal={sig['signal'][i]:+.4f}  w={w:+.4f}")

    targets, longs, shorts = signal_to_targets(sig)
    print(f"\n  ロング {len(longs)}銘柄 / ショート {len(shorts)}銘柄")

    blocked = preflight_margin_sell(client, shorts)
    if blocked and args.live and not args.allow_blocked:
        print("\n売建不可の銘柄があるため中止しました。"
              "片側だけ建てるとマーケットニュートラルが崩れます。"
              "\n強行するには --allow-blocked を付けてください。")
        return None

    print("\n--- 発注数量 ---")
    plans = size_orders(client, targets, args.notional)
    if not plans:
        print("  発注可能な銘柄がありません")
        return None

    n_long = sum(1 for p in plans if p["side"] == "LONG")
    n_short = len(plans) - n_long
    print(f"\n  合計 {len(plans)}件 (ロング{n_long} / ショート{n_short}) "
          f"想定約定代金 {sum(p['notional'] for p in plans):,.0f}円")
    if n_long != n_short:
        print("  [警告] ロングとショートの本数が揃っていません "
              "(サイジングで一部スキップ)。ネットエクスポージャーが残ります。")

    print(f"\n--- エントリー {'(実発注)' if args.live else '(ドライラン)'} ---")
    return execute_entries(client, plans, live=args.live,
                           margin_trade_type=args.margin_type,
                           long_with_cash=args.long_with_cash)


def run_exit(client, args):
    print(f"\n--- 決済 {'(実発注)' if args.live else '(ドライラン)'} ---")
    return execute_exits(client, live=args.live, margin_trade_type=args.margin_type)


def main(argv=None):
    parser = argparse.ArgumentParser(description="PCA_SUB 自動発注")
    parser.add_argument("mode", choices=["entry", "exit", "status"],
                        help="entry=寄付き建て, exit=大引け決済, status=口座照会のみ")
    parser.add_argument("--live", action="store_true",
                        help="実際に発注する (既定はドライラン)")
    parser.add_argument("--notional", type=float, default=TARGET_NOTIONAL_PER_LEG,
                        help=f"1銘柄あたりの発注代金上限 (既定 {TARGET_NOTIONAL_PER_LEG:,.0f}円)")
    parser.add_argument("--margin-type", type=int, default=KABU_MARGIN_TRADE_TYPE,
                        dest="margin_type",
                        help="信用区分 1=制度 2=一般長期 3=一般デイトレ")
    parser.add_argument("--long-with-cash", action="store_true", dest="long_with_cash",
                        help="ロングを現物買いにする (同日決済しない場合のみ)")
    parser.add_argument("--allow-blocked", action="store_true", dest="allow_blocked",
                        help="信用売建不可の銘柄があっても続行する")
    parser.add_argument("--fresh-us", action="store_true", dest="fresh_us",
                        help="取引直前のUSセッションをシグナルに使う "
                             "(既定はバックテスト同様1日古いUSを使う)")
    args = parser.parse_args(argv)

    print(f"=== Trade Executor [{args.mode}]: {datetime.datetime.now()} ===")
    if not args.live:
        print("*** ドライラン: 発注はしません (--live で実発注) ***")

    client = KabuStationClient()
    print(f"接続先: {client.base}")
    try:
        client.auth()
    except (ValueError, KabuApiError) as e:
        print(f"認証失敗: {e}")
        return 1
    print(f"認証成功: token={client.token[:8]}...")

    try:
        print(f"買付余力: {client.wallet()}")
        if args.mode == "status":
            print(f"信用建可能額: {client.margin_wallet()}")
            print(f"現物保有: {client.positions(product=1)}")
            print(f"信用建玉: {client.positions(product=2)}")
            print(f"注文一覧: {client.orders()}")
        elif args.mode == "entry":
            run_entry(client, args)
        else:
            run_exit(client, args)
    except KabuApiError as e:
        print(f"APIエラー: {e}")
        return 1
    except (RuntimeError, ValueError) as e:
        print(f"中断: {e}")
        return 1

    print("\n=== 完了 ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
