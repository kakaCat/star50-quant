"""
测试MoE Parquet数据加载器
========================
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.models.alpha.moe_data_loader import MoEParquetDataLoader


class TestMoEParquetDataLoader:
    """测试Parquet数据加载器"""

    def test_initialization(self):
        """测试初始化"""
        loader = MoEParquetDataLoader()
        assert loader.stock_data_path is not None
        assert loader.index_data_path is not None

    def test_load_data_if_exists(self):
        """测试数据加载（如果文件存在）"""
        loader = MoEParquetDataLoader()

        # 检查文件是否存在
        stock_path = Path(loader.stock_data_path)
        index_path = Path(loader.index_data_path)

        if stock_path.exists() and index_path.exists():
            stock_df, index_df = loader.load_data()

            # 验证数据结构
            assert len(stock_df) > 0
            assert len(index_df) > 0
            assert 'trade_date' in stock_df.columns
            assert 'trade_date' in index_df.columns
            assert 'ts_code' in stock_df.columns
        else:
            pytest.skip("Parquet files not found, skipping test")

    def test_build_stock_features_with_mock_data(self):
        """测试个股特征构建（使用模拟数据）"""
        loader = MoEParquetDataLoader()

        # 创建模拟数据
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        stocks = ['stock_1', 'stock_2']

        data = []
        for stock in stocks:
            for i, date in enumerate(dates):
                data.append({
                    'ts_code': stock,
                    'trade_date': date,
                    'hfq_open': 10 + np.random.randn() * 0.5,
                    'hfq_high': 11 + np.random.randn() * 0.5,
                    'hfq_low': 9 + np.random.randn() * 0.5,
                    'hfq_close': 10 + np.random.randn() * 0.5,
                    'vol': 1000000 + np.random.randn() * 100000
                })

        stock_df = pd.DataFrame(data)
        features = loader.build_stock_features(stock_df, windows=[5, 10])

        # 验证
        assert len(features) > 0
        assert 'ts_code' in features.columns
        assert 'trade_date' in features.columns
        assert 'mom_5d' in features.columns
        assert 'vol_5d' in features.columns

    def test_build_regime_features_with_mock_data(self):
        """测试环境特征构建（使用模拟数据）"""
        loader = MoEParquetDataLoader()

        # 创建模拟指数数据
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        index_data = []
        for date in dates:
            index_data.append({
                'trade_date': date,
                'open': 3000 + np.random.randn() * 10,
                'high': 3010 + np.random.randn() * 10,
                'low': 2990 + np.random.randn() * 10,
                'close': 3000 + np.random.randn() * 10,
                'vol': 1000000000 + np.random.randn() * 10000000
            })
        index_df = pd.DataFrame(index_data)

        # 创建模拟股票数据
        stocks = ['stock_1', 'stock_2']
        stock_data = []
        for stock in stocks:
            for date in dates:
                stock_data.append({
                    'ts_code': stock,
                    'trade_date': date,
                    'hfq_close': 10 + np.random.randn() * 0.5
                })
        stock_df = pd.DataFrame(stock_data)

        features = loader.build_regime_features(index_df, stock_df)

        # 验证
        assert len(features) > 0
        assert 'trade_date' in features.columns
        assert 'idx_ret_1d' in features.columns
        assert 'cs_dispersion' in features.columns

    def test_calculate_beta_with_mock_data(self):
        """测试Beta计算（使用模拟数据）"""
        loader = MoEParquetDataLoader()

        # 创建模拟数据
        dates = pd.date_range('2020-01-01', periods=100, freq='D')

        # 股票数据
        stock_data = []
        for date in dates:
            stock_data.append({
                'ts_code': 'stock_1',
                'trade_date': date,
                'hfq_open': 10 + np.random.randn() * 0.5,
                'hfq_close': 10 + np.random.randn() * 0.5
            })
        stock_df = pd.DataFrame(stock_data)

        # 指数数据
        index_data = []
        for date in dates:
            index_data.append({
                'trade_date': date,
                'open': 3000 + np.random.randn() * 10,
                'close': 3000 + np.random.randn() * 10
            })
        index_df = pd.DataFrame(index_data)

        result = loader.calculate_beta_and_residual(
            stock_df, index_df, forward_days=5, beta_window=20
        )

        # 验证
        assert len(result) > 0
        assert 'beta' in result.columns
        assert 'residual_return' in result.columns
        assert not result['beta'].isna().all()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
