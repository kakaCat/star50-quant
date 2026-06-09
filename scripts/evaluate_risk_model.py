#!/usr/bin/env python3
"""
风险模型评估脚本
================

评估深度风险模型效果，包括：
- 风险因子分析
- 风险分解
- 协方差矩阵预测准确性
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_risk_model():
    """评估风险模型"""
    print("="*60)
    print("深度风险模型评估")
    print("="*60)

    # 加载结果
    print("\n1. 加载模型结果...")
    factor_exposures = pd.read_csv('models/risk/factor_exposures.csv', index_col=0)
    factor_returns = pd.read_csv('models/risk/factor_returns.csv', index_col=0)
    factor_cov = pd.read_csv('models/risk/factor_covariance.csv', index_col=0)
    specific_risk = pd.read_csv('models/risk/specific_risk.csv')

    print(f"  因子暴露: {factor_exposures.shape}")
    print(f"  因子收益率: {factor_returns.shape}")
    print(f"  因子协方差: {factor_cov.shape}")
    print(f"  特质风险: {len(specific_risk)}")

    # 分析因子暴露
    print("\n" + "="*60)
    print("2. 因子暴露分析")
    print("="*60)

    # 计算每个因子的集中度
    factor_concentration = factor_exposures.abs().sum(axis=0).sort_values(ascending=False)
    print("\n因子集中度（绝对值求和）:")
    print(factor_concentration)

    # 因子相关性
    factor_corr = factor_returns.corr()
    print(f"\n因子间相关性:")
    print(f"  平均相关性: {factor_corr.abs().mean().mean():.4f}")
    print(f"  最大相关性: {factor_corr.abs().values[np.triu_indices_from(factor_corr.values, k=1)].max():.4f}")

    # 特质风险分析
    print("\n" + "="*60)
    print("3. 特质风险分析")
    print("="*60)

    print(f"\n特质波动率统计:")
    print(specific_risk['specific_volatility'].describe())

    # 找出高特质风险的股票
    high_risk_stocks = specific_risk.nlargest(10, 'specific_volatility')
    print(f"\nTop 10 高特质风险股票:")
    print(high_risk_stocks[['ts_code', 'specific_volatility']])

    # 风险分解
    print("\n" + "="*60)
    print("4. 风险分解")
    print("="*60)

    # 计算系统性风险和特质风险占比
    total_var = np.trace(factor_cov.values) + specific_risk['specific_variance'].sum()
    systematic_var = np.trace(factor_cov.values)
    specific_var_total = specific_risk['specific_variance'].sum()

    print(f"\n总风险: {total_var:.6f}")
    print(f"  系统性风险: {systematic_var:.6f} ({systematic_var/total_var*100:.2f}%)")
    print(f"  特质风险: {specific_var_total:.6f} ({specific_var_total/total_var*100:.2f}%)")

    # 各因子的方差贡献
    factor_var_contrib = np.diag(factor_cov.values)
    factor_var_df = pd.DataFrame({
        'Factor': factor_cov.columns,
        'Variance': factor_var_contrib,
        'Contribution': factor_var_contrib / systematic_var * 100
    }).sort_values('Variance', ascending=False)

    print(f"\n各因子方差贡献:")
    print(factor_var_df)

    # 可视化
    print("\n" + "="*60)
    print("5. 生成可视化")
    print("="*60)

    # 1. 因子暴露热力图
    plt.figure(figsize=(12, 10))
    sns.heatmap(factor_exposures.T, cmap='RdBu_r', center=0,
                cbar_kws={'label': 'Exposure'})
    plt.title('Factor Exposures Heatmap')
    plt.xlabel('Stocks')
    plt.ylabel('Factors')
    plt.tight_layout()
    plt.savefig('models/risk/factor_exposures_heatmap.png', dpi=150)
    print("  ✓ 因子暴露热力图: models/risk/factor_exposures_heatmap.png")

    # 2. 因子相关性矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(factor_corr, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True)
    plt.title('Factor Correlation Matrix')
    plt.tight_layout()
    plt.savefig('models/risk/factor_correlation.png', dpi=150)
    print("  ✓ 因子相关性矩阵: models/risk/factor_correlation.png")

    # 3. 特质风险分布
    plt.figure(figsize=(10, 6))
    plt.hist(specific_risk['specific_volatility'], bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Specific Volatility')
    plt.ylabel('Frequency')
    plt.title('Distribution of Specific Risk')
    plt.axvline(specific_risk['specific_volatility'].mean(),
                color='r', linestyle='--', label='Mean')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('models/risk/specific_risk_distribution.png', dpi=150)
    print("  ✓ 特质风险分布: models/risk/specific_risk_distribution.png")

    # 4. 因子方差贡献
    plt.figure(figsize=(10, 6))
    plt.bar(factor_var_df['Factor'], factor_var_df['Contribution'])
    plt.xlabel('Risk Factor')
    plt.ylabel('Variance Contribution (%)')
    plt.title('Factor Variance Contribution')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('models/risk/factor_variance_contribution.png', dpi=150)
    print("  ✓ 因子方差贡献: models/risk/factor_variance_contribution.png")

    # 5. 风险分解饼图
    plt.figure(figsize=(8, 8))
    labels = ['Systematic Risk', 'Specific Risk']
    sizes = [systematic_var/total_var*100, specific_var_total/total_var*100]
    colors = ['#ff9999', '#66b3ff']
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.2f%%',
            startangle=90)
    plt.title('Risk Decomposition')
    plt.tight_layout()
    plt.savefig('models/risk/risk_decomposition.png', dpi=150)
    print("  ✓ 风险分解饼图: models/risk/risk_decomposition.png")

    # 6. 因子收益率时间序列
    fig, axes = plt.subplots(5, 2, figsize=(15, 12))
    axes = axes.flatten()

    for i, col in enumerate(factor_returns.columns):
        axes[i].plot(factor_returns.index, factor_returns[col], linewidth=0.5)
        axes[i].set_title(col)
        axes[i].grid(True, alpha=0.3)
        axes[i].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('models/risk/factor_returns_timeseries.png', dpi=150)
    print("  ✓ 因子收益率时间序列: models/risk/factor_returns_timeseries.png")

    print("\n" + "="*60)
    print("✓ 风险模型评估完成!")
    print("="*60)


if __name__ == '__main__':
    evaluate_risk_model()
