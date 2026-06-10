"""
协方差估计器
============

使用滚动窗口计算样本协方差矩阵。
"""

import numpy as np
import pandas as pd
from typing import Optional


class CovarianceEstimator:
    """
    样本协方差估计器

    使用历史收益率的滚动窗口计算协方差矩阵。
    """

    def __init__(self, window: int = 252):
        """
        初始化

        Args:
            window: 滚动窗口大小（交易日数），默认252天≈1年
        """
        self.window = window

    def estimate(
        self,
        returns: pd.DataFrame,
        method: str = 'sample'
    ) -> np.ndarray:
        """
        计算协方差矩阵

        Args:
            returns: 日收益率 DataFrame [n_days, n_stocks]
                     列名为股票代码，索引为日期
            method: 估计方法，目前只支持 'sample'

        Returns:
            协方差矩阵 [n_stocks, n_stocks]

        Raises:
            ValueError: 如果数据不足或method不支持
        """
        if method != 'sample':
            raise ValueError(f"不支持的方法: {method}，目前只支持 'sample'")

        # 处理缺失值：forward fill
        returns_filled = returns.ffill()

        # 如果仍有NaN（首行缺失），用0填充
        returns_filled = returns_filled.fillna(0)

        # 取最近window天
        if len(returns_filled) > self.window:
            recent_returns = returns_filled.iloc[-self.window:]
        else:
            recent_returns = returns_filled

        # 检查数据是否足够
        if len(recent_returns) < 2:
            raise ValueError(f"数据不足: 需要至少2天，实际{len(recent_returns)}天")

        # 计算样本协方差
        cov_matrix = recent_returns.cov().values

        # 确保正定性（添加小的正则化项）
        epsilon = 1e-8
        cov_matrix = cov_matrix + epsilon * np.eye(cov_matrix.shape[0])

        return cov_matrix

    def estimate_correlation(
        self,
        returns: pd.DataFrame
    ) -> np.ndarray:
        """
        计算相关系数矩阵

        Args:
            returns: 日收益率 DataFrame

        Returns:
            相关系数矩阵 [n_stocks, n_stocks]
        """
        returns_filled = returns.ffill().fillna(0)

        if len(returns_filled) > self.window:
            recent_returns = returns_filled.iloc[-self.window:]
        else:
            recent_returns = returns_filled

        corr_matrix = recent_returns.corr().values

        return corr_matrix
