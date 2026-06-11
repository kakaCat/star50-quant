#!/usr/bin/env python3
"""
阶段3：完整回测（最终修正版）
===============================

正确的数据处理和回测逻辑
"""

import sys
import os
sys.path.insert(0, 'star50-quant')

import pandas as pd
import numpy as np
from scipy import stats

print("="*70)
print("阶段3：完整回测（最终修正版）")
print("="*70)
print()

# 加载数据
from src.models.data_loader import FactorDataLoader
from src.models.xgb_model import XGBoostAlphaModel

print("1. 加载数据...")
loader = FactorDataLoader(db_name='star50_quant')
loader.connect()

factors_long = loader.load_factors('2023-01-01', '2024-12-31')
features = loader.pivot_factors(factors_long)

prices = loader.load_prices('2023-01-01', '2025-01-31')
labels = loader.calculate_returns(prices, forward_days=5)

labels['factor_date'] = pd.to_datetime(labels['trade_date'])
features['factor_date'] = pd.to_datetime(features['factor_date'])

# 合并数据
data = features.merge(
    labels[['ts_code', 'factor_date', 'forward_return']],
    on=['ts_code', 'factor_date'],
    how='inner'
).dropna()

print(f"  原始样本: {len(data)}")

# 保存原始forward_return（在标准化之前）
original_returns = data[['ts_code', 'factor_date', 'forward_return']].copy()

# 清洗forward_return - 过滤和winsorize
print("  清洗forward_return数据...")
data = data[(data['forward_return'] > -0.5) & (data['forward_return'] < 0.5)]
print(f"  过滤后: {len(data)}")

# Winsorize forward_return
data['forward_return_clean'] = stats.mstats.winsorize(
    data['forward_return'].values,
    limits=[0.01, 0.01]
)

print(f"  Forward return stats after cleaning:")
print(f"    Mean: {data['forward_return_clean'].mean():.4f}")
print(f"    Std: {data['forward_return_clean'].std():.4f}")
print(f"    Min: {data['forward_return_clean'].min():.4f}")
print(f"    Max: {data['forward_return_clean'].max():.4f}")

# 标准化特征（不包括forward_return）
feature_cols = [col for col in data.columns
                if col not in ['ts_code', 'factor_date', 'forward_return', 'forward_return_clean']]

data_features = data[['ts_code', 'factor_date'] + feature_cols].copy()
data_features = loader.winsorize(data_features)
data_features = loader.standardize(data_features)

# 重新合并清洗后的returns
data_final = data_features.merge(
    data[['ts_code', 'factor_date', 'forward_return_clean']],
    on=['ts_code', 'factor_date'],
    how='inner'
)

loader.close()

print(f"✓ 数据处理完成: {len(data_final)} 样本")
print(f"  日期范围: {data_final['factor_date'].min()} 至 {data_final['factor_date'].max()}")
print(f"  股票数: {data_final['ts_code'].nunique()}")
print()

# 加载模型
print("2. 加载最佳模型...")
model = XGBoostAlphaModel()
model.load('./tuning_results/full_run_fixed/xgb_best_model_20260610_211911.json')
print("✓ 模型加载完成")
print()

# 预测
print("3. 生成预测...")
temp_features = data_final[['ts_code', 'factor_date'] + feature_cols].copy()
predictions = model.predict(temp_features)
data_final['alpha'] = predictions
print("✓ 预测完成")
print()

print("="*70)
print("回测说明")
print("="*70)
print("forward_return = 未来5个交易日的收益率")
print("回测策略: 每5个交易日调仓一次")
print("交易成本: 单次调仓双边0.2%")
print()

# 回测策略
print("="*70)
print("4. 测试不同策略配置")
print("="*70)
print()

strategies = [
    {'name': '10%等权', 'select_pct': 0.10, 'weight': 'equal'},
    {'name': '15%等权', 'select_pct': 0.15, 'weight': 'equal'},
    {'name': '20%等权', 'select_pct': 0.20, 'weight': 'equal'},
    {'name': '10%信号加权', 'select_pct': 0.10, 'weight': 'signal'},
    {'name': '15%信号加权', 'select_pct': 0.15, 'weight': 'signal'},
    {'name': '20%信号加权', 'select_pct': 0.20, 'weight': 'signal'},
]

results = []

for strategy in strategies:
    print(f"测试策略: {strategy['name']}")

    data_sorted = data_final.sort_values('factor_date')
    unique_dates = sorted(data_sorted['factor_date'].unique())
    rebalance_dates = unique_dates[::5]  # 每5天调仓

    period_returns = []
    period_ics = []

    for date in rebalance_dates:
        group = data_sorted[data_sorted['factor_date'] == date]
        if len(group) < 5:
            continue

        # 选股
        n_select = max(int(len(group) * strategy['select_pct']), 1)
        selected = group.nlargest(n_select, 'alpha')

        # 权重
        if strategy['weight'] == 'equal':
            weights = np.ones(len(selected)) / len(selected)
        else:  # signal
            alphas = selected['alpha'].values
            alphas = alphas - alphas.min() + 0.01
            weights = alphas / alphas.sum()

        # 组合收益（5日期收益）
        portfolio_return = np.dot(weights, selected['forward_return_clean'].values)
        portfolio_return -= 0.002  # 双边0.2%交易成本

        period_returns.append(portfolio_return)

        # IC
        ic = np.corrcoef(selected['alpha'].values, selected['forward_return_clean'].values)[0, 1]
        if not np.isnan(ic):
            period_ics.append(ic)

    returns_array = np.array(period_returns)

    # 计算指标
    mean_ic = np.mean(period_ics) if period_ics else 0

    # 累积收益
    cumulative = np.cumprod(1 + returns_array)
    total_return = cumulative[-1] - 1

    # 年化收益（每个period是5天）
    n_periods = len(returns_array)
    periods_per_year = 252 / 5  # 约50个period

    if total_return > -0.99:
        annual_return = (1 + total_return) ** (periods_per_year / n_periods) - 1
        annual_return = np.clip(annual_return, -0.99, 10.0)  # 限制在合理范围
    else:
        annual_return = -0.99

    # 最大回撤
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0

    # IR
    mean_return = np.mean(returns_array)
    std_return = np.std(returns_array)
    ir = (mean_return * np.sqrt(periods_per_year)) / std_return if std_return > 0 else 0

    # 综合评分
    ic_score = max(0, mean_ic / 0.04)
    ir_score = max(0, ir / 1.5)
    return_score = max(0, annual_return / 0.35) if annual_return > 0 else 0
    drawdown_score = max(0, 1 + max_drawdown / 0.2) if max_drawdown >= -0.2 else 0

    composite_score = (ic_score * 0.25 + ir_score * 0.25 +
                      return_score * 0.25 + drawdown_score * 0.25)

    print(f"  调仓次数:  {n_periods}")
    print(f"  IC:        {mean_ic:.4f}")
    print(f"  IR:        {ir:.2f}")
    print(f"  年化收益:  {annual_return:.2%}")
    print(f"  最大回撤:  {max_drawdown:.2%}")
    print(f"  综合评分:  {composite_score:.4f}")

    # 检查达标
    targets_met = 0
    if mean_ic > 0.04:
        print(f"  ✓ IC达标")
        targets_met += 1
    if ir >= 1.5:
        print(f"  ✓ IR达标")
        targets_met += 1
    if annual_return > 0.35:
        print(f"  ✓ 年化收益达标")
        targets_met += 1
    if max_drawdown >= -0.20:
        print(f"  ✓ 最大回撤达标")
        targets_met += 1

    print(f"  目标达成: {targets_met}/4")
    print()

    results.append({
        'strategy': strategy['name'],
        'select_pct': strategy['select_pct'],
        'weight_method': strategy['weight'],
        'n_periods': n_periods,
        'ic': mean_ic,
        'ir': ir,
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'composite_score': composite_score,
        'targets_met': targets_met
    })

# 找最佳策略
results_df = pd.DataFrame(results)
best_idx = results_df['composite_score'].idxmax()
best = results_df.loc[best_idx]

print("="*70)
print("最佳策略")
print("="*70)
print(f"策略:      {best['strategy']}")
print(f"选股比例:  {best['select_pct']:.0%}")
print(f"权重方式:  {best['weight_method']}")
print(f"调仓次数:  {int(best['n_periods'])}次（2年测试期）")
print()
print(f"IC:        {best['ic']:.4f}  (目标 > 0.04)")
print(f"IR:        {best['ir']:.2f}  (目标 >= 1.5)")
print(f"年化收益:  {best['annual_return']:.2%}  (目标 > 35%)")
print(f"最大回撤:  {best['max_drawdown']:.2%}  (目标 <= -20%)")
print(f"综合评分:  {best['composite_score']:.4f}")
print()

targets_met = int(best['targets_met'])
print(f"目标达成: {targets_met}/4")

if targets_met == 4:
    print()
    print("🎉 恭喜！所有目标已达成！")
    print("下一步：进入阶段5（生产部署）")
elif targets_met >= 2:
    print()
    print("⚠️ 部分目标达成")
    print("建议：进入阶段4（策略优化）")
else:
    print()
    print("❌ 大部分目标未达成")
    print("建议：审查数据质量或特征工程")

print()
print("="*70)
print("阶段3完成！")
print("="*70)

# 保存结果
results_df.to_csv('backtest_results_stage3_final.csv', index=False)
print()
print("✓ 结果已保存到: backtest_results_stage3_final.csv")
