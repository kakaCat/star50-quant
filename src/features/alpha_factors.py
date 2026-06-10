"""
Alpha因子库
============

实现经典Alpha因子和自定义Alpha因子。
参考WorldQuant 101 Alphas。
"""

import pandas as pd
import numpy as np
from typing import List, Dict


class AlphaFactorCalculator:
    """
    Alpha因子计算器

    提供多种Alpha因子计算方法。
    """

    def __init__(self):
        """初始化"""
        self.factor_registry = {
            'alpha_001': self.calculate_alpha_001,
            'alpha_006': self.calculate_alpha_006,
            'alpha_053': self.calculate_alpha_053,
            'alpha_054': self.calculate_alpha_054,
            'alpha_101': self.calculate_alpha_101,
            'alpha_custom_price_volume_divergence': self.calculate_custom_price_volume_divergence,
            'alpha_custom_ma_acceleration': self.calculate_custom_ma_acceleration,
            'alpha_custom_volume_surge': self.calculate_custom_volume_surge,
            'alpha_custom_volatility_ratio': self.calculate_custom_volatility_ratio,
            'alpha_custom_trend_strength': self.calculate_custom_trend_strength
        }

    def calculate_alpha_001(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        Alpha#1: 动量反转
        (-1 × close.pct_change(period))

        逻辑: 过去N天涨太多的股票倾向于回调
        """
        result = df.copy()
        result['alpha_001'] = -1 * result.groupby('ts_code')['close'].pct_change(period)
        return result

    def calculate_alpha_006(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        Alpha#6: 相对强弱
        close.pct_change(period) / volume.rolling(period).mean()

        逻辑: 价格涨幅相对于成交量的比率
        """
        result = df.copy()

        price_change = result.groupby('ts_code')['close'].pct_change(period)
        volume_ma = result.groupby('ts_code')['volume'].transform(
            lambda x: x.rolling(period, min_periods=1).mean()
        )

        result['alpha_006'] = price_change / (volume_ma + 1e-6)
        return result

    def calculate_alpha_053(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Alpha#53: 高低价差/成交量
        (close - low) / volume

        逻辑: 相对最低价的位置除以成交量
        """
        result = df.copy()
        result['alpha_053'] = (result['close'] - result['low']) / (result['volume'] + 1e-6)
        return result

    def calculate_alpha_054(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Alpha#54: 价格位置
        (close - open) / (high - low + ε)

        逻辑: 收盘价在当日高低范围的相对位置
        """
        result = df.copy()
        result['alpha_054'] = (result['close'] - result['open']) / (result['high'] - result['low'] + 1e-6)
        return result

    def calculate_alpha_101(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Alpha#101: 价格动量强度
        (close - open) / (open + ε)

        逻辑: 日内涨跌幅
        """
        result = df.copy()
        result['alpha_101'] = (result['close'] - result['open']) / (result['open'] + 1e-6)
        return result

    def calculate_custom_price_volume_divergence(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        自定义Alpha: 价量背离
        close.rank() - volume.rank()

        逻辑: 价格排名与成交量排名的差异
        """
        result = df.copy()

        price_rank = result.groupby('trade_date')['close'].rank(pct=True)
        volume_rank = result.groupby('trade_date')['volume'].rank(pct=True)

        result['alpha_custom_price_volume_divergence'] = price_rank - volume_rank
        return result

    def calculate_custom_ma_acceleration(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        自定义Alpha: MA加速度
        ma.pct_change(5).pct_change(5)

        逻辑: 均线变化率的变化率（二阶导数）
        """
        result = df.copy()

        ma = result.groupby('ts_code')['close'].transform(
            lambda x: x.rolling(period, min_periods=1).mean()
        )
        ma_velocity = ma.groupby(result['ts_code']).pct_change(5, fill_method=None)
        ma_acceleration = ma_velocity.groupby(result['ts_code']).pct_change(5, fill_method=None)

        result['alpha_custom_ma_acceleration'] = ma_acceleration
        return result

    def calculate_custom_volume_surge(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        自定义Alpha: 成交量突增
        (volume - volume.rolling(period).mean()) / volume.rolling(period).std()

        逻辑: 成交量的Z-score
        """
        result = df.copy()

        grouped = result.groupby('ts_code')['volume']
        volume_ma = grouped.transform(lambda x: x.rolling(period, min_periods=1).mean())
        volume_std = grouped.transform(lambda x: x.rolling(period, min_periods=1).std())

        result['alpha_custom_volume_surge'] = (result['volume'] - volume_ma) / (volume_std + 1e-6)
        return result

    def calculate_custom_volatility_ratio(self, df: pd.DataFrame, short: int = 5, long: int = 20) -> pd.DataFrame:
        """
        自定义Alpha: 波动率比率
        std(short) / std(long)

        逻辑: 短期波动率相对长期波动率
        """
        result = df.copy()

        grouped = result.groupby('ts_code')['close']
        std_short = grouped.transform(lambda x: x.pct_change().rolling(short, min_periods=1).std())
        std_long = grouped.transform(lambda x: x.pct_change().rolling(long, min_periods=1).std())

        result['alpha_custom_volatility_ratio'] = std_short / (std_long + 1e-6)
        return result

    def calculate_custom_trend_strength(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        自定义Alpha: 趋势强度
        (close - close.rolling(period).min()) / (close.rolling(period).max() - close.rolling(period).min())

        逻辑: 价格在N天最高最低价之间的相对位置
        """
        result = df.copy()

        grouped = result.groupby('ts_code')['close']
        rolling_min = grouped.transform(lambda x: x.rolling(period, min_periods=1).min())
        rolling_max = grouped.transform(lambda x: x.rolling(period, min_periods=1).max())

        result['alpha_custom_trend_strength'] = (result['close'] - rolling_min) / (rolling_max - rolling_min + 1e-6)
        return result

    def calculate_batch(self, df: pd.DataFrame, alpha_list: List[str]) -> pd.DataFrame:
        """
        批量计算多个Alpha因子

        Args:
            df: 输入数据
            alpha_list: Alpha因子名称列表

        Returns:
            包含所有Alpha因子的DataFrame
        """
        result = df.copy()

        for alpha_name in alpha_list:
            if alpha_name not in self.factor_registry:
                raise ValueError(f"Unknown alpha factor: {alpha_name}")

            calculator_func = self.factor_registry[alpha_name]
            result = calculator_func(result)

        return result

    def get_available_alphas(self) -> List[str]:
        """返回所有可用的Alpha因子名称"""
        return list(self.factor_registry.keys())
