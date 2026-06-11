#!/usr/bin/env python3
"""
阶段3：完整回测（修正版）
=========================

正确处理5日前瞻收益的回测
"""

import sys
import os
sys.path.insert(0, 'star50-quant')

import pandas as pd
import numpy as np
from datetime import timedelta
import json
from scipy import stats

print("="*70)
print("阶段3：完整回测（修正版）")
print("="*70)
print()

# 加载数据
from src.models.data_loader import FactorDataLoader
from src.models.xgb_model import XGBoostAlphaModel

print("1. 加载数据...")
loader = FactorDataLoader(db_name='star50_quant')
loader.connect()

# 使用2023-2024年数据测试
factors_long = loader.load_factors('2023-01-01', '2024-12-31')
features = loader.pivot_factors(factors_long)

prices = loader.load_prices('2023-01-01', '2025-01-31')
labels = loader.calculate_returns(prices, forward_days=5)

labels['factor_date'] = pd.to_datetime(labels['trade_date'])
features['factor_date'] = pd.to_datetime(features['factor_date'])

data = features.merge(
    labels[['ts_code', 'factor_date', 'forward_return']],
    on=['ts_code', 'factor_date'],
    how='inner'
).dropna()

print(f"  原始样本: {len(data)}")

# 数据清洗
data = data[
    (data['forward_return'] > -0.5) &
    (data['forward_return'] < 0.5)
]
print(f"  过滤极端值后: {len(data)}")

data['forward_return'] = stats.mstats.winsorize(
    data['forward_return'].values,
    limits=[0.01, 0.01]
)

data = loader.winsorize(data.copy())
data = loader.standardize(data)
loader.close()

print(f"✓ 数据加载完成: {len(data)} 样本")
print(f"  日期范围: {data['factor_date'].min()} 至 {data['factor_date'].max()}")
print(f"  股票数: {data['ts_code'].nunique()}")
print()

# 加载最佳模型
print("2. 加载最佳模型...")
model = XGBoostAlphaModel()
model.load('./tuning_results/full_run_fixed/xgb_best_model_20260610_211911.json')
print("✓ 模型加载完成")
print()

# 预测
print("3. 生成预测...")
temp_features = data[['ts_code', 'factor_date'] +
                     [col for col in data.columns
                      if col not in ['ts_code', 'factor_date', 'forward_return', 'alpha']]].copy()

predictions = model.predict(temp_features)
data['alpha'] = predictions
print("✓ 预测完成")
print()

# 关键理解：forward_return是5日收益，不是单日收益
print("="*70)
print("重要说明")
print("="*70)
print("forward_return = 未来5个交易日的收益率")
print("回测策略: 每周调仓一次（持有5天）")
print()

# 回测不同策略配置
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

    # 按日期排序
    data_sorted = data.sort_values('factor_date')

    # 每5天调仓一次（因为forward_return是5日收益）
    unique_dates = sorted(data_sorted['factor_date'].unique())

    # 选择调仓日期（每隔5个交易日）
    rebalance_dates = unique_dates[::5]

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

        # 组合收益（这是5日收益）
        portfolio_return = np.dot(weights, selected['forward_return'].values)
        portfolio_return -= 0.01  # 5日交易成本（双边0.2% * 5）

        period_returns.append(portfolio_return)

        # IC
        ic = np.corrcoef(selected['alpha'].values, selected['forward_return'].values)[0, 1]
        if not np.isnan(ic):
            period_ics.append(ic)

    returns_array = np.array(period_returns)

    # 计算指标
    mean_ic = np.mean(period_ics) if period_ics else 0

    # 累积收益曲线
    cumulative = np.cumprod(1 + returns_array)
    total_return = cumulative[-1] - 1

    # 年化收益（每个period是5天，一年约252/5=50个period）
    n_periods = len(returns_array)
    periods_per_year = 252 / 5
    annual_return = (1 + total_return) ** (periods_per_year / n_periods) - 1

    # 最大回撤
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0

    # IR（基于5日period）
    mean_return = np.mean(returns_array)
    std_return = np.std(returns_array)
    ir = (mean_return * np.sqrt(periods_per_year)) / std_return if std_return > 0 else 0

    # 综合评分
    composite_score = (
        (mean_ic / 0.04) * 0.25 +
        (ir / 1.5) * 0.25 +
        (annual_return / 0.35) * 0.25 +
        ((1 + max_drawdown / 0.2) if max_drawdown >= -0.2 else 0) * 0.25
    )

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
print(f"调仓次数:  {int(best['n_periods'])}次（约每周一次）")
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
    print("建议：审查方法或特征工程")

print()
print("="*70)
print("阶段3完成！")
print("="*70)

# 保存结果
results_df.to_csv('backtest_results_stage3_fixed.csv', index=False)
print()
print("✓ 结果已保存到: backtest_results_stage3_fixed.csv")
