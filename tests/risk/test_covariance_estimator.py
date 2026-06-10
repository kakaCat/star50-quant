"""
协方差估计器测试
"""

import numpy as np
import pandas as pd
import pytest
from src.risk.covariance_estimator import CovarianceEstimator


class TestCovarianceEstimator:
    """协方差估计器测试套件"""

    def test_sample_covariance_shape(self):
        """测试输出形状正确"""
        # 创建测试数据：10天 × 5只股票
        returns = pd.DataFrame(
            np.random.randn(10, 5),
            columns=['stock1', 'stock2', 'stock3', 'stock4', 'stock5']
        )

        estimator = CovarianceEstimator(window=10)
        cov_matrix = estimator.estimate(returns)

        # 验证形状
        assert cov_matrix.shape == (5, 5)

    def test_covariance_positive_definite(self):
        """测试协方差矩阵正定"""
        returns = pd.DataFrame(
            np.random.randn(50, 5),
            columns=[f'stock{i}' for i in range(5)]
        )

        estimator = CovarianceEstimator(window=50)
        cov_matrix = estimator.estimate(returns)

        # 验证正定：所有特征值>0
        eigenvalues = np.linalg.eigvals(cov_matrix)
        assert np.all(eigenvalues > 0)

    def test_covariance_symmetric(self):
        """测试协方差矩阵对称"""
        returns = pd.DataFrame(
            np.random.randn(30, 3),
            columns=['A', 'B', 'C']
        )

        estimator = CovarianceEstimator(window=30)
        cov_matrix = estimator.estimate(returns)

        # 验证对称
        assert np.allclose(cov_matrix, cov_matrix.T)

    def test_rolling_window(self):
        """测试滚动窗口计算"""
        # 创建100天数据
        returns = pd.DataFrame(
            np.random.randn(100, 3),
            columns=['X', 'Y', 'Z']
        )

        # 使用最近50天
        estimator = CovarianceEstimator(window=50)
        cov_matrix = estimator.estimate(returns)

        # 手工计算最近50天协方差
        recent_returns = returns.iloc[-50:]
        expected_cov = recent_returns.cov().values

        # 验证结果接近
        assert np.allclose(cov_matrix, expected_cov, rtol=1e-5)

    def test_missing_data_forward_fill(self):
        """测试缺失值处理（forward fill）"""
        # 创建带缺失值的数据
        returns = pd.DataFrame({
            'stock1': [0.01, 0.02, np.nan, 0.03, 0.01],
            'stock2': [0.02, np.nan, 0.01, 0.02, 0.03],
            'stock3': [0.01, 0.01, 0.02, 0.01, 0.02]
        })

        estimator = CovarianceEstimator(window=5)
        cov_matrix = estimator.estimate(returns)

        # 验证无NaN
        assert not np.isnan(cov_matrix).any()

        # 验证形状
        assert cov_matrix.shape == (3, 3)
