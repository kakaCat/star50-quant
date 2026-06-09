"""
组合优化模块
============

使用cvxpy实现组合优化，包括：
- 风险调整后收益最大化
- 权重约束
- 换手率控制
"""

try:
    from src.optimization.portfolio_optimizer import PortfolioOptimizer
    __all__ = ['PortfolioOptimizer']
except ImportError as e:
    __all__ = []
    print(f"Warning: Portfolio optimizer requires cvxpy: {e}")
