"""
Alpha预测模型模块
==================

深度学习模型用于预测股票未来超额收益率。

支持模型：
- LightGBM / XGBoost (Baseline)
- LSTM / GRU (时序模型)
- Temporal Fusion Transformer (TFT)
"""

from src.models.data_loader import FactorDataLoader
from src.models.lgbm_model import LightGBMAlphaModel

# 可选导入PyTorch模型
try:
    from src.models.lstm_model import LSTMAlphaModel
    __all__ = [
        'FactorDataLoader',
        'LightGBMAlphaModel',
        'LSTMAlphaModel',
    ]
except ImportError:
    __all__ = [
        'FactorDataLoader',
        'LightGBMAlphaModel',
    ]
