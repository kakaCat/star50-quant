"""Alpha因子库测试"""

import pytest
import pandas as pd
import numpy as np
from src.features.alpha_factors import AlphaFactorCalculator


@pytest.fixture
def sample_data():
    """创建测试数据"""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    n_stocks = 10

    data = []
    for stock_id in range(n_stocks):
        for date in dates:
            data.append({
                'ts_code': f'{stock_id:06d}.SH',
                'trade_date': date,
                'close': 100 + np.random.randn() * 10,
                'open': 100 + np.random.randn() * 10,
                'high': 105 + np.random.randn() * 10,
                'low': 95 + np.random.randn() * 10,
                'volume': 1000000 + np.random.randn() * 100000
            })

    df = pd.DataFrame(data)
    return df


def test_alpha_001(sample_data):
    """测试Alpha#1: 动量反转"""
    calculator = AlphaFactorCalculator()
    result = calculator.calculate_alpha_001(sample_data, period=20)

    assert 'alpha_001' in result.columns
    assert len(result) == len(sample_data)
    assert not result['alpha_001'].isna().all()


def test_alpha_006(sample_data):
    """测试Alpha#6: 相对强弱"""
    calculator = AlphaFactorCalculator()
    result = calculator.calculate_alpha_006(sample_data, period=20)

    assert 'alpha_006' in result.columns
    assert len(result) == len(sample_data)
    assert not result['alpha_006'].isna().all()


def test_alpha_053(sample_data):
    """测试Alpha#53: 高低价差/成交量"""
    calculator = AlphaFactorCalculator()
    result = calculator.calculate_alpha_053(sample_data)

    assert 'alpha_053' in result.columns
    assert len(result) == len(sample_data)
    # 检查计算逻辑
    expected = (sample_data['close'] - sample_data['low']) / (sample_data['volume'] + 1e-6)
    pd.testing.assert_series_equal(result['alpha_053'], expected, check_names=False)


def test_alpha_054(sample_data):
    """测试Alpha#54: 价格位置"""
    calculator = AlphaFactorCalculator()
    result = calculator.calculate_alpha_054(sample_data)

    assert 'alpha_054' in result.columns
    assert len(result) == len(sample_data)
    # 检查计算逻辑（不检查范围，因为high-low可能很小）
    expected = (sample_data['close'] - sample_data['open']) / (sample_data['high'] - sample_data['low'] + 1e-6)
    pd.testing.assert_series_equal(result['alpha_054'], expected, check_names=False)


def test_alpha_custom_price_volume_divergence(sample_data):
    """测试自定义Alpha: 价量背离"""
    calculator = AlphaFactorCalculator()
    result = calculator.calculate_custom_price_volume_divergence(sample_data)

    assert 'alpha_custom_price_volume_divergence' in result.columns
    assert len(result) == len(sample_data)


def test_batch_calculation(sample_data):
    """测试批量计算所有Alpha因子"""
    calculator = AlphaFactorCalculator()
    alpha_list = ['alpha_001', 'alpha_006', 'alpha_053', 'alpha_054']

    result = calculator.calculate_batch(sample_data, alpha_list)

    for alpha_name in alpha_list:
        assert alpha_name in result.columns

    assert len(result) == len(sample_data)
