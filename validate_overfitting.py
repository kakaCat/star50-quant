#!/usr/bin/env python3
"""
过拟合验证 - 严格的样本外测试
================================

训练集: 2019-2022（用于训练和调参）
测试集: 2024（完全样本外）
"""

import sys
sys.path.insert(0, 'star50-quant')

import pandas as pd
import numpy as np
from scipy import stats

print("="*70)
print("过拟合验证 - 严格样本外测试")
print("="*70)
print()

from src.models.data_loader import FactorDataLoader
from src.models.xgb_model import XGBoostAlphaModel
import xgboost as xgb

# 1. 重新训练模型（只用2019-2022）
print("="*70)
print("步骤1: 重新训练模型（只用2019-2022数据）")
print("="*70)
print()

loader = FactorDataLoader(db_name='star50_quant')
loader.connect()

print("加载训练数据（2019-2022）...")
factors_train = loader.load_factors('2019-01-01', '2022-12-31')
features_train = loader.pivot_factors(factors_train)

prices_train = loader.load_prices('2019-01-01', '2023-02-28')
labels_train = loader.calculate_returns(prices_train, forward_days=5)

labels_train['factor_date'] = pd.to_datetime(labels_train['trade_date'])
features_train['factor_date'] = pd.to_datetime(features_train['factor_date'])

train_data = features_train.merge(
    labels_train[['ts_code', 'factor_date', 'forward_return']],
    on=['ts_code', 'factor_date'],
    how='inner'
).dropna()

print(f"  训练集样本: {len(train_data)}")
print(f"  日期范围: {train_data['factor_date'].min()} 至 {train_data['factor_date'].max()}")

# 清洗
train_data = train_data[(train_data['forward_return'] > -0.5) & (train_data['forward_return'] < 0.5)]
train_data['forward_return_clean'] = stats.mstats.winsorize(
    train_data['forward_return'].values,
    limits=[0.01, 0.01]
)

feature_cols = [col for col in train_data.columns
                if col not in ['ts_code', 'factor_date', 'forward_return', 'forward_return_clean']]

train_features = train_data[['ts_code', 'factor_date'] + feature_cols].copy()
train_features = loader.winsorize(train_features)
train_features = loader.standardize(train_features)

train_final = train_features.merge(
    train_data[['ts_code', 'factor_date', 'forward_return_clean']],
    on=['ts_code', 'factor_date'],
    how='inner'
)

print(f"✓ 训练数据准备完成")
print()

# 训练新模型（使用阶段1的最佳参数）
print("训练XGBoost模型...")
best_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'booster': 'gbtree',
    'max_depth': 7,
    'learning_rate': 0.0885,
    'subsample': 0.9446,
    'colsample_bytree': 0.9290,
    'colsample_bylevel': 0.6672,
    'min_child_weight': 4,
    'gamma': 0.0005,
    'reg_alpha': 2.9637,
    'reg_lambda': 2.7327,
    'seed': 42,
    'n_jobs': -1,
    'verbosity': 0
}

X_train = train_final[feature_cols].values
y_train = train_final['forward_return_clean'].values

dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)

model_new = xgb.train(
    best_params,
    dtrain,
    num_boost_round=152,
    verbose_eval=False
)

print("✓ 模型训练完成")
print()

# 2. 样本外测试（2024年）
print("="*70)
print("步骤2: 样本外测试（2024年，模型从未见过）")
print("="*70)
print()

print("加载测试数据（2024年）...")
factors_test = loader.load_factors('2024-01-01', '2024-12-31')
features_test = loader.pivot_factors(factors_test)

prices_test = loader.load_prices('2024-01-01', '2025-01-31')
labels_test = loader.calculate_returns(prices_test, forward_days=5)

labels_test['factor_date'] = pd.to_datetime(labels_test['trade_date'])
features_test['factor_date'] = pd.to_datetime(features_test['factor_date'])

test_data = features_test.merge(
    labels_test[['ts_code', 'factor_date', 'forward_return']],
    on=['ts_code', 'factor_date'],
    how='inner'
).dropna()

print(f"  测试集样本: {len(test_data)}")
print(f"  日期范围: {test_data['factor_date'].min()} 至 {test_data['factor_date'].max()}")

# 清洗测试数据
test_data = test_data[(test_data['forward_return'] > -0.5) & (test_data['forward_return'] < 0.5)]
test_data['forward_return_clean'] = stats.mstats.winsorize(
    test_data['forward_return'].values,
    limits=[0.01, 0.01]
)

test_features = test_data[['ts_code', 'factor_date'] + feature_cols].copy()
test_features = loader.winsorize(test_features)
test_features = loader.standardize(test_features)

test_final = test_features.merge(
    test_data[['ts_code', 'factor_date', 'forward_return_clean']],
    on=['ts_code', 'factor_date'],
    how='inner'
)

loader.close()

print(f"✓ 测试数据准备完成")
print()

# 预测
print("生成预测...")
X_test = test_final[feature_cols].values
dtest = xgb.DMatrix(X_test, feature_names=feature_cols)
predictions = model_new.predict(dtest)
test_final['alpha'] = predictions

print("✓ 预测完成")
print()

# 3. 回测（20%等权策略）
print("="*70)
print("步骤3: 样本外回测（20%等权策略）")
print("="*70)
print()

test_sorted = test_final.sort_values('factor_date')
unique_dates = sorted(test_sorted['factor_date'].unique())
rebalance_dates = unique_dates[::5]

period_returns = []
period_ics = []

for date in rebalance_dates:
    group = test_sorted[test_sorted['factor_date'] == date]
    if len(group) < 5:
        continue

    n_select = max(int(len(group) * 0.20), 1)
    selected = group.nlargest(n_select, 'alpha')
    weights = np.ones(len(selected)) / len(selected)

    portfolio_return = np.dot(weights, selected['forward_return_clean'].values)
    portfolio_return -= 0.002
    period_returns.append(portfolio_return)

    ic = np.corrcoef(selected['alpha'].values, selected['forward_return_clean'].values)[0, 1]
    if not np.isnan(ic):
        period_ics.append(ic)

returns_array = np.array(period_returns)

# 计算指标
mean_ic = np.mean(period_ics) if period_ics else 0
cumulative = np.cumprod(1 + returns_array)
total_return = cumulative[-1] - 1

n_periods = len(returns_array)
periods_per_year = 252 / 5

if total_return > -0.99:
    annual_return = (1 + total_return) ** (periods_per_year / n_periods) - 1
    annual_return = np.clip(annual_return, -0.99, 10.0)
else:
    annual_return = -0.99

running_max = np.maximum.accumulate(cumulative)
drawdown = (cumulative - running_max) / running_max
max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0

mean_return = np.mean(returns_array)
std_return = np.std(returns_array)
ir = (mean_return * np.sqrt(periods_per_year)) / std_return if std_return > 0 else 0

print("样本外测试结果（2024年）:")
print(f"  调仓次数:  {n_periods}")
print(f"  IC:        {mean_ic:.4f}  (目标 > 0.04)")
print(f"  IR:        {ir:.2f}  (目标 >= 1.5)")
print(f"  年化收益:  {annual_return:.2%}  (目标 > 35%)")
print(f"  最大回撤:  {max_drawdown:.2%}  (目标 <= -20%)")
print()

# 4. 对比分析
print("="*70)
print("步骤4: 样本内 vs 样本外对比")
print("="*70)
print()

print("| 数据集 | IC | IR | 年化收益 | 最大回撤 |")
print("|--------|----|----|----------|----------|")
print(f"| 样本内（2023-2024，有泄露） | 0.136 | 3.36 | 230.76% | -18.57% |")
print(f"| **样本外（2024，无泄露）** | **{mean_ic:.3f}** | **{ir:.2f}** | **{annual_return:.2%}** | **{max_drawdown:.2%}** |")
print()

# 性能下降分析
ic_drop = (0.136 - mean_ic) / 0.136 * 100
ir_drop = (3.36 - ir) / 3.36 * 100
return_drop = (2.3076 - annual_return) / 2.3076 * 100 if annual_return > 0 else 100

print("性能下降分析:")
print(f"  IC下降: {ic_drop:.1f}%")
print(f"  IR下降: {ir_drop:.1f}%")
print(f"  年化收益下降: {return_drop:.1f}%")
print()

if ic_drop > 30 or return_drop > 50:
    print("⚠️ 结论: 存在严重过拟合！")
    print("   样本外性能大幅下降，说明原始测试有数据泄露")
elif ic_drop > 15 or return_drop > 30:
    print("⚠️ 结论: 存在一定过拟合")
    print("   样本外性能有所下降，但可能仍然可用")
else:
    print("✅ 结论: 模型泛化能力良好")
    print("   样本外性能与样本内接近")

print()
print("="*70)
print("验证完成")
print("="*70)
