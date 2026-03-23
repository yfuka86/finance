"""
eスマート証券 kabuステーションAPI クライアント
"""
import json
import requests
from trading.config import KABU_API_BASE, KABU_API_PASSWORD


class KabuStationClient:
    """kabuステーション REST API クライアント"""

    def __init__(self):
        self.base = KABU_API_BASE
        self.token = None

    def auth(self, password: str = None):
        """トークン取得"""
        pw = password or KABU_API_PASSWORD
        if not pw:
            raise ValueError("APIパスワードが設定されていません。環境変数 KABU_API_PASSWORD を設定してください。")
        resp = requests.post(
            f"{self.base}/token",
            headers={"content-type": "application/json"},
            data=json.dumps({"APIPassword": pw}),
        )
        resp.raise_for_status()
        self.token = resp.json()["Token"]
        return self.token

    def _headers(self):
        if not self.token:
            raise RuntimeError("未認証です。auth() を先に呼んでください。")
        return {"X-API-KEY": self.token, "content-type": "application/json"}

    # --- 照会系 ---

    def wallet(self):
        """買付余力"""
        resp = requests.get(f"{self.base}/wallet/cash", headers=self._headers())
        resp.raise_for_status()
        return resp.json()

    def positions(self):
        """保有銘柄一覧"""
        resp = requests.get(f"{self.base}/positions", headers=self._headers())
        resp.raise_for_status()
        return resp.json()

    def orders(self):
        """注文一覧"""
        resp = requests.get(f"{self.base}/orders", headers=self._headers())
        resp.raise_for_status()
        return resp.json()

    def board(self, symbol: str, exchange: int = 1):
        """板情報 (exchange: 1=東証)"""
        resp = requests.get(
            f"{self.base}/board/{symbol}@{exchange}", headers=self._headers()
        )
        resp.raise_for_status()
        return resp.json()

    # --- 発注系 ---

    def send_order(self, payload: dict):
        """
        注文発注 (汎用)
        payload例 (現物買い):
        {
            "Symbol": "16170",     # 銘柄コード
            "Exchange": 1,          # 1=東証
            "SecurityType": 1,      # 1=株式
            "Side": "2",            # 1=売, 2=買
            "CashMargin": 1,        # 1=現物, 2=新規信用, 3=返済信用
            "DelivType": 2,         # 2=お預り金
            "AccountType": 2,       # 2=特定
            "Qty": 1,               # 数量
            "FrontOrderType": 10,   # 10=成行
            "Price": 0,             # 成行なら0
            "ExpireDay": 0,         # 0=当日
        }
        """
        resp = requests.post(
            f"{self.base}/sendorder",
            headers=self._headers(),
            data=json.dumps(payload),
        )
        resp.raise_for_status()
        return resp.json()

    def cancel_order(self, order_id: str):
        """注文取消"""
        resp = requests.put(
            f"{self.base}/cancelorder",
            headers=self._headers(),
            data=json.dumps({"OrderId": order_id}),
        )
        resp.raise_for_status()
        return resp.json()

    # --- 便利メソッド ---

    def buy_market(self, symbol: str, qty: int, exchange: int = 1):
        """成行で現物買い"""
        return self.send_order({
            "Symbol": symbol,
            "Exchange": exchange,
            "SecurityType": 1,
            "Side": "2",
            "CashMargin": 1,
            "DelivType": 2,
            "AccountType": 2,
            "Qty": qty,
            "FrontOrderType": 10,
            "Price": 0,
            "ExpireDay": 0,
        })

    def sell_market(self, symbol: str, qty: int, exchange: int = 1):
        """成行で現物売り"""
        return self.send_order({
            "Symbol": symbol,
            "Exchange": exchange,
            "SecurityType": 1,
            "Side": "1",
            "CashMargin": 1,
            "DelivType": 2,
            "AccountType": 2,
            "Qty": qty,
            "FrontOrderType": 10,
            "Price": 0,
            "ExpireDay": 0,
        })
