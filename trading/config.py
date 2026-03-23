"""
eスマート証券 kabuステーションAPI 設定
"""
import os

# kabuステーションAPI
KABU_API_BASE = os.environ.get("KABU_API_BASE", "http://localhost:18080/kabusapi")
KABU_API_PASSWORD = os.environ.get("KABU_API_PASSWORD", "")  # 環境変数から読み込み
