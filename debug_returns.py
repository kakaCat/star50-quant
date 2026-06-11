#!/usr/bin/env python3
"""Debug: Check actual daily returns in backtest"""

import sys
sys.path.insert(0, 'star50-quant')

import pandas as pd
import numpy as np
from src.models.data_loader import FactorDataLoader
from src.models.xgb_model import XGBoostAlphaModel
from scipy import stats

# Load and clean data
loader = FactorDataLoader(db_name='star50_quant')
loader.connect()

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

print(f"Original samples: {len(data)}")

# Clean data
data = data[
    (data['forward_return'] > -0.5) &
    (data['forward_return'] < 0.5)
]
print(f"After filtering: {len(data)}")

data['forward_return'] = stats.mstats.winsorize(
    data['forward_return'].values,
    limits=[0.01, 0.01]
)

data = loader.winsorize(data.copy())
data = loader.standardize(data)
loader.close()

# Load model and predict
model = XGBoostAlphaModel()
model.load('./tuning_results/full_run_fixed/xgb_best_model_20260610_211911.json')

temp_features = data[['ts_code', 'factor_date'] +
                     [col for col in data.columns
                      if col not in ['ts_code', 'factor_date', 'forward_return', 'alpha']]].copy()

predictions = model.predict(temp_features)
data['alpha'] = predictions

# Simulate one strategy
print("\nSimulating 20% equal weight strategy...")
daily_returns = []

for date, group in data.groupby('factor_date'):
    if len(group) < 5:
        continue

    # Select top 20%
    n_select = max(int(len(group) * 0.20), 1)
    selected = group.nlargest(n_select, 'alpha')

    # Equal weight
    weights = np.ones(len(selected)) / len(selected)

    # Portfolio return
    portfolio_return = np.dot(weights, selected['forward_return'].values)
    portfolio_return -= 0.002  # Trading cost
    portfolio_return = np.clip(portfolio_return, -0.15, 0.15)

    daily_returns.append(portfolio_return)

returns_array = np.array(daily_returns)

print(f"\nDaily returns statistics:")
print(f"  Count: {len(returns_array)}")
print(f"  Mean: {np.mean(returns_array):.4f}")
print(f"  Std: {np.std(returns_array):.4f}")
print(f"  Min: {np.min(returns_array):.4f}")
print(f"  Max: {np.max(returns_array):.4f}")
print(f"  Median: {np.median(returns_array):.4f}")
print()

# Show distribution
print("Return distribution:")
for pct in [1, 5, 25, 50, 75, 95, 99]:
    val = np.percentile(returns_array, pct)
    print(f"  {pct:2d}%: {val:7.4f}")

print("\nCumulative calculation test:")
# Method 1: Direct product
cumulative_prod = np.cumprod(1 + returns_array)
print(f"  Direct product final: {cumulative_prod[-1]:.2f}x")

# Method 2: Log sum
log_returns = np.log1p(returns_array)
cumulative_log = np.sum(log_returns)
print(f"  Log sum: {cumulative_log:.4f}")
print(f"  Exp(log sum): {np.exp(cumulative_log):.2f}x")

# Proper annualization
n_days = len(returns_array)
mean_daily = np.mean(returns_array)
annual_mean = mean_daily * 252
print(f"\nProper calculation:")
print(f"  Mean daily return: {mean_daily:.4f} ({mean_daily*100:.2f}%)")
print(f"  Simple annualization: {annual_mean:.4f} ({annual_mean*100:.2f}%)")
print(f"  Compound annualization: {(1 + mean_daily)**252 - 1:.4f} ({((1 + mean_daily)**252 - 1)*100:.2f}%)")

# IR calculation
std_daily = np.std(returns_array)
ir = (mean_daily * np.sqrt(252)) / std_daily
print(f"\nIR:")
print(f"  Daily std: {std_daily:.4f}")
print(f"  IR: {ir:.2f}")
