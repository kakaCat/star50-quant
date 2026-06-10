"""增强数据加载器测试"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.models.data_loader import FactorDataLoader


def test_build_dataset_with_feature_engineering():
    """测试集成特征工程的数据集构建"""
    # Mock数据库连接和查询
    with patch.object(FactorDataLoader, 'connect'), \
         patch.object(FactorDataLoader, 'close'), \
         patch.object(FactorDataLoader, 'load_factors') as mock_load_factors, \
         patch.object(FactorDataLoader, 'load_prices') as mock_load_prices:

        # 准备mock数据（30个原始因子）
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        stocks = [f'{i:06d}.SH' for i in range(5)]

        # Mock因子数据
        factor_data = []
        factor_names = ['close', 'open', 'high', 'low', 'volume', 'macd', 'rsi6', 'rsi12', 'rsi24',
                       'obv', 'volume_ratio', 'momentum_5', 'momentum_10', 'momentum_20',
                       'ma5', 'ma10', 'ma20', 'atr14', 'volume_ma5', 'volume_ma10',
                       'volume_ma20', 'mfi14', 'roc_5', 'roc_10', 'roc_20']

        for stock in stocks:
            for date in dates:
                for fname in factor_names:
                    factor_data.append({
                        'ts_code': stock,
                        'factor_date': date,
                        'factor_name': fname,
                        'factor_value': np.random.randn()
                    })

        mock_load_factors.return_value = pd.DataFrame(factor_data)

        # Mock价格数据
        price_data = []
        for stock in stocks:
            for date in dates:
                price_data.append({
                    'ts_code': stock,
                    'trade_date': date,
                    'close': 100 + np.random.randn() * 10
                })
        mock_load_prices.return_value = pd.DataFrame(price_data)

        # 测试调用
        loader = FactorDataLoader()
        features, labels = loader.build_dataset(
            start_date='2024-01-01',
            end_date='2024-12-31',
            forward_days=5,
            enable_feature_engineering=True
        )

        # 检查特征数量显著增加
        feature_cols = [c for c in features.columns if c not in ['ts_code', 'factor_date']]
        assert len(feature_cols) > 100  # 应该有100+特征

        print(f"增强后特征数: {len(feature_cols)}")
