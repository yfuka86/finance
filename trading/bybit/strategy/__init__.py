from .base import BaseStrategy
from .momentum import MomentumStrategy
from .market_maker import MarketMakerStrategy
from .grid import GridStrategy
from .bollinger_reversion import BollingerReversionStrategy
from .rsi_reversion import RSIReversionStrategy
from .donchian_breakout import DonchianBreakoutStrategy
from .volatility_breakout import VolatilityBreakoutStrategy
from .ichimoku import IchimokuStrategy
from .macd_adx import MACDADXStrategy
from .rbreaker import RBreakerStrategy
from .trend_regime import TrendRegimeStrategy
from .mean_reversion_filtered import MeanReversionFilteredStrategy
from .tsmom import TSMOMStrategy
from .dual_regime import DualRegimeStrategy
from .mtf_rsi2 import MTFRsi2Strategy
from .vwap_reversion import VWAPReversionStrategy
from .volume_spike import VolumeSpikeStrategy
from .mtf_confluence import MTFConfluenceStrategy

STRATEGIES = {
    "momentum": MomentumStrategy,
    "market_maker": MarketMakerStrategy,
    "grid": GridStrategy,
    "bollinger_reversion": BollingerReversionStrategy,
    "rsi_reversion": RSIReversionStrategy,
    "donchian_breakout": DonchianBreakoutStrategy,
    "volatility_breakout": VolatilityBreakoutStrategy,
    "ichimoku": IchimokuStrategy,
    "macd_adx": MACDADXStrategy,
    "rbreaker": RBreakerStrategy,
    "trend_regime": TrendRegimeStrategy,
    "mean_reversion_filtered": MeanReversionFilteredStrategy,
    "tsmom": TSMOMStrategy,
    "dual_regime": DualRegimeStrategy,
    "mtf_rsi2": MTFRsi2Strategy,
    "vwap_reversion": VWAPReversionStrategy,
    "volume_spike": VolumeSpikeStrategy,
    "mtf_confluence": MTFConfluenceStrategy,
}
