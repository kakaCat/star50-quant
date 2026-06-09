#!/usr/bin/env python3
"""
模型评估脚本
============

评估已训练的Alpha模型性能，包括IC分析、分层回测等。
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from src.models.data_loader import FactorDataLoader
from src.models.lgbm_model import LightGBMAlphaModel


def calculate_ic_series(predictions, actuals, dates):
    """
    计算IC时间序列

    Args:
        predictions: 预测值
        actuals: 实际值
        dates: 日期

    Returns:
        IC序列DataFrame
    """
    df = pd.DataFrame({
        'date': dates,
        'pred': predictions,
        'actual': actuals
    })

    # 按日期计算IC
    ic_series = df.groupby('date').apply(
        lambda x: np.corrcoef(x['pred'], x['actual'])[0, 1]
    )

    return pd.DataFrame({
        'date': ic_series.index,
        'ic': ic_series.values
    })


def evaluate_model(model_path: str = 'models/lgbm_alpha.txt'):
    """
    评估模型

    Args:
        model_path: 模型路径
    """
    print("="*60)
    print("Alpha模型评估")
    print("="*60)

    # 加载模型
    print("\n1. 加载模型...")
    model = LightGBMAlphaModel()
    model.load(model_path)

    # 加载数据
    print("\n2. 加载验证数据...")
    with FactorDataLoader() as loader:
        features, labels = loader.build_dataset(
            start_date='2020-01-01',
            end_date='2024-12-31',
            forward_days=5
        )

    # 划分验证集（后20%）
    split_date = features['factor_date'].quantile(0.8)
    val_features = features[features['factor_date'] > split_date].copy()
    val_labels = labels[labels['factor_date'] > split_date].copy()

    print(f"验证集大小: {len(val_features)} 样本")

    # 预测
    print("\n3. 生成预测...")
    predictions = model.predict(val_features)

    # 合并预测和实际值
    val_features['prediction'] = predictions
    results = val_features.merge(
        val_labels[['ts_code', 'factor_date', 'forward_return']],
        on=['ts_code', 'factor_date'],
        how='inner'
    )

    # 整体指标
    print("\n" + "="*60)
    print("整体性能指标")
    print("="*60)

    overall_ic = np.corrcoef(results['prediction'], results['forward_return'])[0, 1]
    rmse = np.sqrt(np.mean((results['prediction'] - results['forward_return'])**2))

    print(f"IC (Information Coefficient): {overall_ic:.4f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"预测值范围: [{results['prediction'].min():.4f}, {results['prediction'].max():.4f}]")
    print(f"实际值范围: [{results['forward_return'].min():.4f}, {results['forward_return'].max():.4f}]")

    # IC时间序列
    print("\n4. 计算IC时间序列...")
    ic_series = calculate_ic_series(
        results['prediction'].values,
        results['forward_return'].values,
        results['factor_date'].values
    )

    print(f"\nIC统计:")
    print(f"  均值: {ic_series['ic'].mean():.4f}")
    print(f"  标准差: {ic_series['ic'].std():.4f}")
    print(f"  IR (IC均值/IC标准差): {ic_series['ic'].mean() / ic_series['ic'].std():.4f}")
    print(f"  IC>0的比例: {(ic_series['ic'] > 0).sum() / len(ic_series):.2%}")

    # 分层回测
    print("\n5. 分层回测...")
    print("="*60)

    # 按预测值分5层
    results['quantile'] = pd.qcut(results['prediction'], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])

    quantile_stats = results.groupby('quantile')['forward_return'].agg(['mean', 'std', 'count'])
    quantile_stats['sharpe'] = quantile_stats['mean'] / quantile_stats['std']

    print("\n分层收益统计:")
    print(quantile_stats)

    print(f"\n多空收益 (Q5 - Q1): {quantile_stats.loc['Q5', 'mean'] - quantile_stats.loc['Q1', 'mean']:.4f}")

    # 保存结果
    print("\n6. 保存评估结果...")

    # 保存IC时间序列
    ic_series.to_csv('models/ic_series.csv', index=False)
    print("  ✓ IC时间序列保存至: models/ic_series.csv")

    # 保存分层统计
    quantile_stats.to_csv('models/quantile_stats.csv')
    print("  ✓ 分层统计保存至: models/quantile_stats.csv")

    # 绘制IC时间序列图
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(ic_series['date'], ic_series['ic'])
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.title('IC Time Series')
    plt.ylabel('IC')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.plot(ic_series['date'], ic_series['ic'].cumsum())
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.title('Cumulative IC')
    plt.ylabel('Cumulative IC')
    plt.xlabel('Date')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('models/ic_analysis.png', dpi=150)
    print("  ✓ IC分析图保存至: models/ic_analysis.png")

    # 绘制分层收益图
    plt.figure(figsize=(10, 6))
    quantile_stats['mean'].plot(kind='bar')
    plt.title('Average Returns by Prediction Quantile')
    plt.xlabel('Quantile')
    plt.ylabel('Average Forward Return')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('models/quantile_returns.png', dpi=150)
    print("  ✓ 分层收益图保存至: models/quantile_returns.png")

    print("\n" + "="*60)
    print("✓ 模型评估完成!")
    print("="*60)


if __name__ == '__main__':
    evaluate_model()
