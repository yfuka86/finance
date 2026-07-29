"""寄付き前の発注経路プローブ: 口座が信用新規を受け付ける状態かを1発で確認する。

約定し得ない指値 (超低位株の値幅下限付近) を1件だけ送り、受理されたら即取消。
結果は stdout とダッシュボード (event=state, probe フィールド) に送る。
口座手続き未完了 (100203/100368 等) の検知が目的。本番運用の 08:50 に
タスクスケジューラから1回実行する想定。手動実行も可。

実行:  PYTHONPATH=. python scripts/preflight_order_probe.py
終了コード: 発注受理(=経路OK)=0 / 拒否=1
"""
import datetime as dt
import sys
import time

from trading.jp_intraday.live.config import LiveConfig
from trading.jp_intraday.live.kabu_client import KabuAPIError, KabuClient
from trading.jp_intraday.live import reporter

# 超低位・値幅内で絶対に約定しない指値 (株価~10円に対して6円の買い指値)
PROBE_SYMBOL, PROBE_PRICE, PROBE_QTY = "8918", 6, 100


def main() -> int:
    cfg = LiveConfig.from_env()
    if cfg.env != "prod" or not cfg.orders_enabled:
        print(f"skip: 実発注が有効ではありません ({cfg.summary()})")
        return 0

    c = KabuClient(cfg.api_password, cfg.order_password, env="prod")
    c.authenticate()
    # 09:00前は寄指(前場)=21 で板寄せ経路まで検証、それ以降は通常指値=20。
    # 市場は信用新規で必須の SOR(9) (東証直指定は100368で拒否される・実測)。
    front = 21 if dt.datetime.now().time() < dt.time(9, 0) else 20
    body = {
        "Password": cfg.order_password, "Symbol": PROBE_SYMBOL,
        "Exchange": KabuClient.EXCHANGE_SOR,
        "SecurityType": 1, "Side": "2", "CashMargin": 2,
        "MarginTradeType": cfg.margin_type, "DelivType": 0,
        "AccountType": cfg.account_type, "Qty": PROBE_QTY,
        "FrontOrderType": front, "Price": PROBE_PRICE, "ExpireDay": 0,
    }
    result: dict = {"probe": "order_path", "symbol": PROBE_SYMBOL}
    ok = False
    try:
        r = c._send(body)
        oid = r.get("OrderId")
        result.update({"accepted": True, "order_id": oid})
        ok = True
        time.sleep(0.8)
        try:
            c.cancel_order(oid)
            result["cancelled"] = True
        except KabuAPIError as e:
            # 取消失敗は要手動確認 (約定リスクは指値幅的に無いが放置しない)
            result.update({"cancelled": False, "cancel_error": str(e)[:200]})
    except KabuAPIError as e:
        result.update({"accepted": False, "error": str(e)[:250]})

    print("PROBE:", result)
    reporter.report(cfg, "state", {"order_path_probe": result},
                    dt.datetime.now().isoformat(timespec="seconds"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
