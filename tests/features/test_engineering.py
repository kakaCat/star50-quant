"""特征工程模块测试"""

import pytest
import pandas as pd
import numpy as np
import yaml
from src.features.engineering import FeatureEngineer


@pytest.fixture
def sample_factor_data():
    """创建测试因子数据"""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    n_stocks = 10

    data = []
    for stock_id in range(n_stocks):
        for date in dates:
            data.append({
                'ts_code': f'{stock_id:06d}.SH',
                'factor_date': date,
                'close': 100 + np.random.randn() * 10,
                'open': 100 + np.random.randn() * 10,
                'high': 105 + np.random.randn() * 10,
                'low': 95 + np.random.randn() * 10,
                'volume': 1000000 + np.random.randn() * 100000,
                'macd': np.random.randn(),
                'rsi6': 50 + np.random.randn() * 20,
                'rsi12': 50 + np.random.randn() * 20,
                'rsi24': 50 + np.random.randn() * 20,
                'obv': np.random.randn() * 1000,
                'volume_ratio': 1 + np.random.randn() * 0.2,
                'momentum_5': np.random.randn(),
                'momentum_10': np.random.randn(),
                'momentum_20': np.random.randn(),
                'ma5': 100 + np.random.randn() * 10,
                'ma10': 100 + np.random.randn() * 10,
                'ma20': 100 + np.random.randn() * 10,
                'atr14': 5 + np.random.randn(),
                'volume_ma5': 1000000 + np.random.randn() * 100000,
                'volume_ma10': 1000000 + np.random.randn() * 100000,
                'volume_ma20': 1000000 + np.random.randn() * 100000,
                'mfi14': 50 + np.random.randn() * 20,
                'roc_5': np.random.randn(),
                'roc_10': np.random.randn(),
                'roc_20': np.random.randn()
            })

    df = pd.DataFrame(data)
    return df


def test_feature_engineer_initialization():
    """测试FeatureEngineer初始化"""
    engineer = FeatureEngineer('../configs/feature_config.yaml')
    assert engineer.config is not None
    assert 'feature_engineering' in engineer.config


def test_add_cross_features(sample_factor_data):
    """测试交叉特征生成"""
    engineer = FeatureEngineer('../configs/feature_config.yaml')
    result = engineer.add_cross_features(sample_factor_data)

    # 检查是否生成了交叉特征
    assert 'rsi6_x_volume_ratio' in result.columns
    assert 'macd_x_volume_ma5' in result.columns

    # 检查计算正确性
    expected = sample_factor_data['rsi6'] * sample_factor_data['volume_ratio']
    pd.testing.assert_series_equal(result['rsi6_x_volume_ratio'], expected, check_names=False)


def test_add_temporal_features(sample_factor_data):
    """测试时序衍生特征"""
    engineer = FeatureEngineer('../configs/feature_config.yaml')
    result = engineer.add_temporal_features(sample_factor_data)

    # 检查因子动量
    assert 'volume_ma20_momentum_5d' in result.columns

    # 检查因子波动
    assert 'volume_ma20_volatility_20d' in result.columns


def test_add_nonlinear_features(sample_factor_data):
    """测试非线性变换"""
    engineer = FeatureEngineer('../configs/feature_config.yaml')
    result = engineer.add_nonlinear_features(sample_factor_data)

    # 检查Log变换
    assert 'volume_ma20_log' in result.columns

    # 检查Rank归一化
    assert 'rsi6_rank' in result.columns
    assert result['rsi6_rank'].min() >= 0
    assert result['rsi6_rank'].max() <= 1


def test_add_cross_sectional_features(sample_factor_data):
    """测试截面统计特征"""
    engineer = FeatureEngineer('../configs/feature_config.yaml')
    result = engineer.add_cross_sectional_features(sample_factor_data)

    # 检查分位数特征
    assert 'volume_ma20_quantile' in result.columns
    assert result['volume_ma20_quantile'].min() >= 0
    assert result['volume_ma20_quantile'].max() <= 1

    # 检查Z-score特征
    assert 'volume_ma20_zscore' in result.columns


def test_add_alpha_factors(sample_factor_data):
    """测试Alpha因子添加"""
    engineer = FeatureEngineer('../configs/feature_config.yaml')
    result = engineer.add_alpha_factors(sample_factor_data)

    # 检查Alpha因子
    assert 'alpha_001' in result.columns
    assert 'alpha_006' in result.columns


def test_transform_complete(sample_factor_data):
    """测试完整特征转换流程"""
    engineer = FeatureEngineer('../configs/feature_config.yaml')
    result = engineer.transform(sample_factor_data)

    # 检查原始特征数量
    original_cols = [c for c in sample_factor_data.columns if c not in ['ts_code', 'factor_date']]

    # 检查增强后特征数量（应该显著增加）
    enhanced_cols = [c for c in result.columns if c not in ['ts_code', 'factor_date']]
    assert len(enhanced_cols) > len(original_cols) * 3  # 至少3倍

    # 检查无NaN和Inf（除了前几行因为rolling）
    result_clean = result.iloc[30:]  # 跳过前30行（warm-up period）
    numeric_cols = result_clean.select_dtypes(include=[np.number]).columns
    assert not result_clean[numeric_cols].isna().all().any()
    assert not np.isinf(result_clean[numeric_cols]).any().any()


def test_get_feature_names(sample_factor_data):
    """测试特征名称获取"""
    engineer = FeatureEngineer('../configs/feature_config.yaml')
    engineer.transform(sample_factor_data)

    feature_names = engineer.get_feature_names()
    assert isinstance(feature_names, list)
    assert len(feature_names) > 100  # 应该有100+特征
