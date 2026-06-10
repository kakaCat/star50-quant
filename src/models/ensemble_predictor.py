"""
集成模型预测器
==============

加载Phase 2训练好的15个LightGBM模型，使用IC加权融合生成Alpha预测。
"""

import numpy as np
import pandas as pd
import os
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False


class EnsemblePredictor:
    """
    集成模型预测器

    加载15个基础LightGBM模型 + IC加权元学习器，生成Alpha预测。
    """

    def __init__(self, model_dir: str):
        """
        初始化

        Args:
            model_dir: 模型目录路径，包含15个模型文件和权重文件
        """
        if not LGBM_AVAILABLE:
            raise ImportError("LightGBM未安装，请运行: pip install lightgbm")

        self.model_dir = model_dir
        self.base_models = {}
        self.weights = None

        # 窗口和配置
        self.windows = [1, 3, 5, 10, 20]
        self.configs = ['default', 'regularized', 'deep']

        # 加载模型
        self._load_models()

    def _load_models(self):
        """加载所有基础模型和权重"""
        print(f"加载集成模型从: {self.model_dir}")

        # 加载15个基础模型
        model_count = 0
        for window in self.windows:
            for config in self.configs:
                model_key = f'w{window}_{config}'
                model_path = os.path.join(self.model_dir, f'{model_key}.txt')

                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"模型文件不存在: {model_path}")

                # 加载LightGBM模型
                model = lgb.Booster(model_file=model_path)
                self.base_models[model_key] = model
                model_count += 1

        print(f"  ✓ 加载 {model_count} 个基础模型")

        # 加载IC权重
        weights_path = os.path.join(self.model_dir, 'ic_weights.npy')
        if os.path.exists(weights_path):
            self.weights = np.load(weights_path)
            print(f"  ✓ 加载IC权重")
        else:
            # 如果没有权重文件，使用等权
            print(f"  ⚠ 权重文件不存在，使用等权")
            self.weights = np.ones(15) / 15

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        生成Alpha预测

        Args:
            features: 特征DataFrame [n_stocks, 9]
                     必须包含列: momentum_5, momentum_10, momentum_20,
                                 volatility_10, volatility_20,
                                 volume_ratio, atr_ratio, ma_ratio, rsi14
                     可选列: ts_code (股票代码)

        Returns:
            预测结果 DataFrame: {ts_code, alpha}
        """
        # 提取特征列
        feature_cols = [
            'momentum_5', 'momentum_10', 'momentum_20',
            'volatility_10', 'volatility_20',
            'volume_ratio', 'atr_ratio', 'ma_ratio', 'rsi14'
        ]

        # 验证特征列存在
        missing_cols = set(feature_cols) - set(features.columns)
        if missing_cols:
            raise ValueError(f"缺失特征列: {missing_cols}")

        # 提取特征矩阵
        X = features[feature_cols].values

        # 检查NaN
        if np.isnan(X).any():
            # Forward fill处理NaN
            features_filled = features[feature_cols].ffill().fillna(0)
            X = features_filled.values

        # 收集所有基础模型预测
        base_predictions = []

        for window in self.windows:
            for config in self.configs:
                model_key = f'w{window}_{config}'
                model = self.base_models[model_key]

                # 预测
                pred = model.predict(X)
                base_predictions.append(pred)

        # 堆叠成矩阵 [n_stocks, 15]
        base_predictions = np.column_stack(base_predictions)

        # IC加权融合
        alpha = (base_predictions * self.weights).sum(axis=1)

        # 构建结果DataFrame
        result = pd.DataFrame({
            'alpha': alpha
        })

        # 如果有ts_code，添加到结果
        if 'ts_code' in features.columns:
            result['ts_code'] = features['ts_code'].values
            result = result[['ts_code', 'alpha']]  # 调整列顺序

        return result

    def save_weights(self, weights: np.ndarray, filepath: str):
        """
        保存IC权重

        Args:
            weights: IC权重数组 [15]
            filepath: 保存路径
        """
        np.save(filepath, weights)
        print(f"✓ 权重已保存到: {filepath}")
