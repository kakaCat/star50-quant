#!/usr/bin/env python3
"""
简化验证：只用核心原始因子
========================

测试假设：复杂特征工程引入噪音，简单因子可能更有效
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from src.models.lgbm_model import LightGBMAlphaModel
import warnings
warnings.filterwarnings('ignore')


def calculate_simple_factors(df):
    """计算最简单的因子"""
    print("计算核心因子...")
    df = df.sort_values(['ts_code', 'trade_date'])
    result = []

    for ts_code, group in df.groupby('ts_code'):
        group = group.sort_values('trade_date').copy()

        # 只计算最核心的因子，添加滞后避免当期信息
        group['momentum_5'] = group['close'].pct_change(5).shift(1)
        group['momentum_10'] = group['close'].pct_change(10).shift(1)
        group['momentum_20'] = group['close'].pct_change(20).shift(1)

        group['volatility_10'] = group['close'].pct_change().rolling(10).std().shift(1)
        group['volatility_20'] = group['close'].pct_change().rolling(20).std().shift(1)

        group['volume_ratio'] = (group['vol'] / group['vol'].rolling(20).mean()).shift(1)

        high_low = group['high'] - group['low']
        high_close = np.abs(group['high'] - group['close'].shift())
        low_close = np.abs(group['low'] - group['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        group['atr_ratio'] = (true_range.rolling(14).mean() / group['close']).shift(1)

        group['factor_date'] = group['trade_date']
        result.append(group)

    return pd.concat(result, ignore_index=True)


def prepare_dataset(df, forward_days=5):
    """准备数据集"""
    labels_data = []
    for ts_code, group in df.groupby('ts_code'):
        group = group.sort_values('trade_date').copy()
        group['forward_return'] = group['close'].pct_change(forward_days).shift(-forward_days)
        for _, row in group.iterrows():
            labels_data.append({
                'ts_code': row['ts_code'],
                'factor_date': row['factor_date'],
                'forward_return': row['forward_return']
            })

    labels = pd.DataFrame(labels_data)

    factor_cols = ['ts_code', 'factor_date',
                   'momentum_5', 'momentum_10', 'momentum_20',
                   'volatility_10', 'volatility_20',
                   'volume_ratio', 'atr_ratio']

    features = df[factor_cols].copy()
    features = features.dropna()
    labels = labels.dropna()

    return features, labels


def train_and_evaluate(features_df, labels_df, label, params=None):
    """训练并评估"""
    print("\n" + "="*80)
    print(f"{label}")
    print("="*80)

    dataset = features_df.merge(labels_df, on=['ts_code', 'factor_date'], how='inner')
    split_date = dataset['factor_date'].quantile(0.8)

    train_data = dataset[dataset['factor_date'] <= split_date]
    val_data = dataset[dataset['factor_date'] > split_date]

    feature_cols = [c for c in dataset.columns if c not in ['ts_code', 'factor_date', 'forward_return']]

    train_features = train_data[['ts_code', 'factor_date'] + feature_cols]
    train_labels = train_data[['ts_code', 'factor_date', 'forward_return']]
    val_features = val_data[['ts_code', 'factor_date'] + feature_cols]
    val_labels = val_data[['ts_code', 'factor_date', 'forward_return']]

    print(f"\nTrain: {len(train_features)} samples, {len(feature_cols)} features")
    print(f"Val: {len(val_features)} samples")

    model = LightGBMAlphaModel(params=params)
    model.train(train_features, train_labels, num_boost_round=200,
                early_stopping_rounds=20, verbose_eval=50)

    val_pred = model.predict(val_features)
    val_true = val_labels['forward_return'].values

    ic = np.corrcoef(val_pred, val_true)[0, 1]

    val_df = val_features.copy()
    val_df['pred'] = val_pred
    val_df['true'] = val_true

    ic_series = []
    for date in val_df['factor_date'].unique():
        mask = val_df['factor_date'] == date
        if mask.sum() >= 5:
            daily_ic = np.corrcoef(val_df.loc[mask, 'pred'], val_df.loc[mask, 'true'])[0, 1]
            if not np.isnan(daily_ic):
                ic_series.append(daily_ic)

    ic_mean = np.mean(ic_series)
    ic_std = np.std(ic_series)
    ic_ir = ic_mean / (ic_std + 1e-6)
    ic_positive_ratio = (np.array(ic_series) > 0).mean()

    print(f"\n{'='*80}")
    print(f"验证集结果:")
    print(f"  IC (整体): {ic:.4f}")
    print(f"  IC (日度均值): {ic_mean:.4f}")
    print(f"  IC (标准差): {ic_std:.4f}")
    print(f"  IR: {ic_ir:.4f}")
    print(f"  IC>0比例: {ic_positive_ratio:.2%}")
    print(f"{'='*80}\n")

    return {
        'ic': ic,
        'ic_mean': ic_mean,
        'ic_std': ic_std,
        'ic_ir': ic_ir,
        'ic_positive_ratio': ic_positive_ratio,
        'n_features': len(feature_cols)
    }


def main():
    """主函数"""
    print("\n" + "="*80)
    print("简化验证：核心因子 + 正则化")
    print("="*80)

    print("\n加载数据...")
    df = pd.read_parquet('data/raw/star50_daily_hfq_data_6yrs.parquet')
    df_with_factors = calculate_simple_factors(df)
    features, labels = prepare_dataset(df_with_factors)

    # 测试不同配置
    print("\n" + "="*80)
    print("对比测试")
    print("="*80)

    # 1. 默认参数
    default_results = train_and_evaluate(
        features.copy(),
        labels.copy(),
        label="[1] 核心7因子 - 默认参数"
    )

    # 2. 增强正则化
    regularized_params = {
        'learning_rate': 0.03,
        'max_depth': 4,
        'num_leaves': 15,
        'min_child_samples': 100,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.5,
        'reg_lambda': 1.0,
    }

    regularized_results = train_and_evaluate(
        features.copy(),
        labels.copy(),
        label="[2] 核心7因子 - 强正则化",
        params=regularized_params
    )

    # 对比结果
    print("\n" + "="*80)
    print("结果对比")
    print("="*80)

    results_df = pd.DataFrame([
        {'配置': '默认参数', **default_results},
        {'配置': '强正则化', **regularized_results}
    ])

    print(f"\n{results_df.to_string(index=False)}")

    # 验收
    print("\n" + "="*80)
    print("验收标准")
    print("="*80)

    target_ic = 0.025
    best_ic = max(default_results['ic_mean'], regularized_results['ic_mean'])

    if best_ic >= target_ic:
        print(f"✓ PASS: 最佳IC={best_ic:.4f} >= 目标{target_ic}")
    else:
        print(f"⚠ 当前最佳IC={best_ic:.4f}, 目标={target_ic}")
        print(f"\n分析:")
        print(f"  - 简化到7个核心因子")
        print(f"  - 所有因子都加了1天滞后")
        print(f"  - 使用了正则化")
        if best_ic > 0:
            print(f"  - IC为正，方向正确，需要更多数据或更好的因子")


if __name__ == '__main__':
    main()
