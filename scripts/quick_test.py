#!/usr/bin/env python3
"""
XGBoost快速测试版本
==================

使用较少数据和迭代快速验证系统
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime
from pathlib import Path

from src.data.parquet_loader import ParquetDataLoader


def quick_test():
    print("="*70)
    print("XGBoost快速测试")
    print("="*70)

    # 1. 加载少量数据
    print("\n1. 加载数据（2023-2024年）...")
    loader = ParquetDataLoader(data_dir='data/raw')
    data, feature_cols = loader.load_and_prepare(
        start_date='2023-01-01',
        end_date='2024-06-30',
        forward_days=5
    )

    print(f"\n数据准备完成:")
    print(f"  样本数: {len(data)}")
    print(f"  特征数: {len(feature_cols)}")
    print(f"  特征列: {feature_cols[:5]}... (显示前5个)")

    # 2. 简单训练测试
    print("\n2. 测试XGBoost训练...")

    X = data[feature_cols].values
    y = data['forward_return'].values

    # 简单的时序分割
    split_point = int(len(X) * 0.8)
    X_train, X_test = X[:split_point], X[split_point:]
    y_train, y_test = y[:split_point], y[split_point:]

    # 训练简单模型
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        'max_depth': 6,
        'learning_rate': 0.05,
        'objective': 'reg:squarederror',
        'seed': 42
    }

    print("\n  训练中...")
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=50,
        evals=[(dtest, 'test')],
        verbose_eval=10
    )

    # 预测和评估
    y_pred = model.predict(dtest)
    ic = np.corrcoef(y_pred, y_test)[0, 1]

    print(f"\n  测试集IC: {ic:.4f}")

    # 3. 简单的回测测试
    print("\n3. 测试策略回测...")

    test_data = data.iloc[split_point:].copy()
    test_data['predicted_alpha'] = y_pred

    # 按日期分组，选择top 15%
    portfolio_returns = []

    for date, group in test_data.groupby('trade_date'):
        n_top = max(1, int(len(group) * 0.15))
        top_stocks = group.nlargest(n_top, 'predicted_alpha')
        portfolio_return = top_stocks['forward_return'].mean()
        portfolio_returns.append(portfolio_return)

    cumulative_nav = np.cumprod(1 + np.array(portfolio_returns))
    total_return = cumulative_nav[-1] - 1

    print(f"  累计收益: {total_return:.2%}")
    print(f"  调仓次数: {len(portfolio_returns)}")

    print("\n" + "="*70)
    print("✓ 快速测试通过！系统运行正常")
    print("="*70)

    return {
        'ic': ic,
        'total_return': total_return,
        'n_samples': len(data),
        'n_features': len(feature_cols)
    }


if __name__ == '__main__':
    results = quick_test()
    print(f"\n测试结果: {results}")
