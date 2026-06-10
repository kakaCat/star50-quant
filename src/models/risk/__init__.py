"""
风险模型模块
============

深度学习风险建模，包括：
- 自编码器风险模型
- 传统Barra风险模型
"""

try:
    from src.models.risk.deep_risk_model import DeepRiskModel
    __all__ = ['DeepRiskModel']
except ImportError:
    __all__ = []
    print("Warning: Risk models require PyTorch")
