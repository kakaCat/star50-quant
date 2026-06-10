"""
集成模型预测器测试
"""

import numpy as np
import pandas as pd
import pytest
import os
from src.models.ensemble_predictor import EnsemblePredictor


class TestEnsemblePredictor:
    """集成模型预测器测试套件"""

    @pytest.fixture
    def mock_model_dir(self):
        """创建模拟模型目录"""
        # 注意：这个测试需要真实的模型文件
        # 在实际环境中，应该使用phase2训练好的模型
        model_dir = 'models/phase2_ensemble/'
        if not os.path.exists(model_dir):
            pytest.skip(f"模型目录不存在: {model_dir}")
        return model_dir

    def test_load_models(self, mock_model_dir):
        """测试模型加载"""
        predictor = EnsemblePredictor(mock_model_dir)

        # 验证15个基础模型已加载
        assert len(predictor.base_models) == 15

        # 验证权重已加载
        assert len(predictor.weights) == 15
        assert np.isclose(predictor.weights.sum(), 1.0)

    def test_predict_shape(self, mock_model_dir):
        """测试预测输出形状"""
        predictor = EnsemblePredictor(mock_model_dir)

        # 创建测试特征：10只股票 × 9个因子
        features = pd.DataFrame(
            np.random.randn(10, 9),
            columns=[
                'momentum_5', 'momentum_10', 'momentum_20',
                'volatility_10', 'volatility_20',
                'volume_ratio', 'atr_ratio', 'ma_ratio', 'rsi14'
            ]
        )
        features['ts_code'] = [f'stock{i}' for i in range(10)]

        # 预测
        result = predictor.predict(features)

        # 验证输出
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 10
        assert 'ts_code' in result.columns
        assert 'alpha' in result.columns

    def test_ic_weighted_fusion(self, mock_model_dir):
        """测试IC加权融合"""
        predictor = EnsemblePredictor(mock_model_dir)

        # 创建测试特征
        features = pd.DataFrame(
            np.random.randn(5, 9),
            columns=[
                'momentum_5', 'momentum_10', 'momentum_20',
                'volatility_10', 'volatility_20',
                'volume_ratio', 'atr_ratio', 'ma_ratio', 'rsi14'
            ]
        )
        features['ts_code'] = [f'stock{i}' for i in range(5)]

        # 预测
        result = predictor.predict(features)

        # 验证alpha值合理（不是NaN或Inf）
        assert not result['alpha'].isna().any()
        assert not np.isinf(result['alpha']).any()
