"""
eスマート証券 kabuステーションAPI 設定
"""
import os

# kabuステーションAPI
# 本番: http://localhost:18080/kabusapi
# 検証: http://localhost:18081/kabusapi
KABU_API_BASE = os.environ.get("KABU_API_BASE", "http://localhost:18080/kabusapi")
KABU_API_PASSWORD = os.environ.get("KABU_API_PASSWORD", "")  # 環境変数から読み込み

# レート制限対策: 全リクエストの最小間隔(秒)と実行回数エラー時のリトライ回数
KABU_MIN_INTERVAL = float(os.environ.get("KABU_MIN_INTERVAL", "1.2"))
KABU_MAX_RETRIES = int(os.environ.get("KABU_MAX_RETRIES", "5"))

# 信用取引区分: 1=制度信用, 2=一般信用(長期), 3=一般信用(デイトレード)
KABU_MARGIN_TRADE_TYPE = int(os.environ.get("KABU_MARGIN_TRADE_TYPE", "1"))

# 1銘柄あたりの発注代金上限(円)。ポジションサイジングの基準。
TARGET_NOTIONAL_PER_LEG = float(os.environ.get("TARGET_NOTIONAL_PER_LEG", "100000"))
