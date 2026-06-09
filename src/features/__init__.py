"""
因子工程模块
============

技术因子计算，支持：
- 趋势因子（MA, EMA, MACD等）
- 动量因子（RSI, ROC, Momentum等）
- 量价因子（OBV, MFI, Volume等）
- 波动率因子（ATR, Bollinger Bands等）

使用TA-Lib提供高性能计算。
"""

from src.features.base import BaseCalculator, TechnicalFactorCalculator
from src.features.momentum import MomentumFactors
from src.features.volume import VolumeFactors
from src.features.trend import TrendFactors

__all__ = [
    'BaseCalculator',
    'TechnicalFactorCalculator',
    'MomentumFactors',
    'VolumeFactors',
    'TrendFactors',
]
