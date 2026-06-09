"""
回测模块
========

实现策略回测和评估，包括：
- 回测引擎
- 业绩归因
- 风险分析
"""

from src.backtest.backtest_engine import BacktestEngine

__all__ = ['BacktestEngine']
