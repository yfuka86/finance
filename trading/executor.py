"""
Trade executor: PCA_SUBシグナルに基づいてkabuステーションAPIで発注する。
VPS上で毎朝実行する想定。

実行タイミング:
  - 米国市場クローズ後 (日本時間 6:00~8:30)
  - シグナル計算 → 8:55頃に注文準備 → 9:00寄付き成行
"""
import datetime
import json
import os
import numpy as np
import pandas as pd
from trading.broker import KabuStationClient
from data.collectors.config import US_TICKERS, JP_TICKERS

# JP ETF の銘柄コード (kabuステーション用、.Tなし末尾0付き)
JP_CODES = {t: t.replace(".T", "0") for t in JP_TICKERS}
# 例: "1617.T" -> "16170"


def compute_today_signal(us_ret: pd.DataFrame, jp_ret: pd.DataFrame):
    """
    直近のUSリターンからPCA_SUBシグナルを計算。
    backtest/strategies/pca_sub.py のロジックを最新1日分だけ実行する。
    """
    # TODO: backtest.strategies.pca_sub からコアロジックを呼び出す
    # ここではスケルトンとして、ダミーシグナルを返す
    raise NotImplementedError("シグナル計算を実装してください")


def execute_trades(signal: np.ndarray, jp_tickers: list[str], client: KabuStationClient):
    """
    シグナルに基づいて寄付き成行注文を発注。

    signal: 各JPセクターの予測シグナル (正=ロング, 負=ショート)
    """
    q = 0.3
    n = max(1, int(np.ceil(len(jp_tickers) * q)))
    ranked = np.argsort(signal)[::-1]
    long_set = set(ranked[:n])
    short_set = set(ranked[-n:])

    orders = []
    for i, ticker in enumerate(jp_tickers):
        code = JP_CODES[ticker]
        if i in long_set:
            print(f"  BUY  {ticker} ({code})")
            result = client.buy_market(code, qty=1)
            orders.append({"ticker": ticker, "side": "BUY", "result": result})
        elif i in short_set:
            print(f"  SELL {ticker} ({code})")
            # 現物売りは保有がないとできない → 信用売りに変更する場合は send_order を直接使う
            # ここではスケルトンとしてスキップ
            print(f"    (信用売り未実装 - スキップ)")
            orders.append({"ticker": ticker, "side": "SELL", "result": "skipped"})

    return orders


def main():
    print(f"=== Trade Executor: {datetime.datetime.now()} ===")

    client = KabuStationClient()
    client.auth()
    print(f"認証成功: token={client.token[:8]}...")

    wallet = client.wallet()
    print(f"買付余力: {wallet}")

    # TODO: 実際のシグナル計算
    # signal = compute_today_signal(us_ret, jp_ret)
    # execute_trades(signal, JP_TICKERS, client)

    print("=== 完了 ===")


if __name__ == "__main__":
    main()
