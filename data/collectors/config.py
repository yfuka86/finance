"""
Ticker definitions and data source configuration.
"""
import os

# --- API Keys ---
JQUANTS_API_KEY = os.environ.get(
    "JQUANTS_API_KEY", "2gOAlBhCb2vjpcfNYm0AKV8hhhmfsvyYdTnRNLQm6aI"
)
JQUANTS_BASE = "https://api.jquants.com"

EDINET_API_KEY = os.environ.get(
    "EDINET_API_KEY", "a5e9c7dad00c4554ba9ca23ac3f62c79"
)

# --- US Sector ETFs (Select Sector SPDR, 11 GICS sectors) ---
US_TICKERS = [
    "XLB",   # Materials
    "XLE",   # Energy
    "XLF",   # Financials
    "XLI",   # Industrials
    "XLK",   # Information Technology
    "XLP",   # Consumer Staples
    "XLU",   # Utilities
    "XLV",   # Health Care
    "XLY",   # Consumer Discretionary
    "XLC",   # Communication Services
    "XLRE",  # Real Estate
]

# --- Japanese Sector ETFs (NEXT FUNDS TOPIX-17) ---
JP_TICKERS = [
    "1617.T",  # 食品
    "1618.T",  # エネルギー資源
    "1619.T",  # 建設・資材
    "1620.T",  # 素材・化学
    "1621.T",  # 医薬品
    "1622.T",  # 自動車・輸送機
    "1623.T",  # 鉄鋼・非鉄
    "1624.T",  # 機械
    "1625.T",  # 電機・精密
    "1626.T",  # 情報通信・サービスその他
    "1627.T",  # 電力・ガス
    "1628.T",  # 運輸・物流
    "1629.T",  # 商社・卸売
    "1630.T",  # 小売
    "1631.T",  # 銀行
    "1632.T",  # 金融(除く銀行)
    "1633.T",  # 不動産
]

# --- Cyclical / Defensive labels (for PCA prior vectors) ---
US_CYCLICAL = ["XLB", "XLE", "XLF", "XLRE"]
US_DEFENSIVE = ["XLK", "XLP", "XLU", "XLV"]
JP_CYCLICAL = ["1618.T", "1625.T", "1629.T", "1631.T"]
JP_DEFENSIVE = ["1617.T", "1621.T", "1627.T", "1630.T"]

# --- Date range ---
START_DATE = "2010-01-01"
END_DATE = "2025-12-31"

# --- Paths ---
RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "raw")
