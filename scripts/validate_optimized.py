#!/usr/bin/env python3
"""
优化后验证脚本
==============

使用Top 50特征 + 修复后的特征工程代码
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from src.features.engineering import FeatureEngineer
from src.models.lgbm_model import LightGBMAlphaModel
import warnings
warnings.filterwarnings('ignore')


# Top 50 推荐特征
TOP_50_FEATURES = [
    'atr14_momentum_20d', 'atr14_momentum_10d', 'momentum_5', 'ma20_momentum_10d',
    'roc_5_x_mfi14', 'momentum_10_rank', 'mfi14_momentum_20d', 'ma20_momentum_5d',
    'rsi6_rank', 'atr14_quantile', 'volume_ma20_momentum_20d', 'ma20_momentum_20d',
    'atr14_zscore', 'alpha_001', 'atr14_volatility_20d', 'rsi12_momentum_5d',
    'rsi12_momentum_20d', 'alpha_006', 'atr14', 'macd_zscore', 'rsi12_rank',
    'atr14_momentum_5d', 'volume_ma20_momentum_10d', 'alpha_054', 'alpha_101',
    'alpha_custom_price_volume_divergence', 'macd_rank', 'momentum_20',
    'macd_x_volume_ma10', 'momentum_10_volatility_20d', 'atr14_log', 'rsi24',
    'rsi6', 'momentum_10', 'volume_ma10_momentum_20d', 'ma20', 'roc_20_x_mfi14',
    'ma20_slope_x_rsi12', 'ma10_slope_x_momentum_10', 'macd_x_volume_ma5',
    'mfi14_volatility_20d', 'macd_div_atr14', 'roc_5', 'macd_x_volume_ma20',
    'rsi12_volatility_20d', 'ma5_slope_x_momentum_5', 'volume_ma5_momentum_5d',
    'alpha_custom_trend_strength', 'macd', 'roc_20'
]


def calculate_technical_factors(df):
    """从价格数据计算技术因子"""
    print("计算技术因子...")
    df = df.sort_values(['ts_code', 'trade_date'])
    result = []

    for ts_code, group in df.groupby('ts_code'):
        group = group.sort_values('trade_date').copy()
        group['volume'] = group['vol']

        ema12 = group['close'].ewm(span=12, adjust=False).mean()
        ema26 = group['close'].ewm(span=26, adjust=False).mean()
        group['macd'] = ema12 - ema26

        delta = group['close'].diff()
        for period in [6, 12, 24]:
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / (loss + 1e-6)
            group[f'rsi{period}'] = 100 - (100 / (1 + rs))

        obv = (np.sign(group['close'].diff()) * group['volume']).fillna(0).cumsum()
        group['obv'] = obv
        group['volume_ma5'] = group['volume'].rolling(5).mean()
        group['volume_ratio'] = group['volume'] / (group['volume_ma5'] + 1e-6)
        group['momentum_5'] = group['close'].pct_change(5)
        group['momentum_10'] = group['close'].pct_change(10)
        group['momentum_20'] = group['close'].pct_change(20)
        group['ma5'] = group['close'].rolling(5).mean()
        group['ma10'] = group['close'].rolling(10).mean()
        group['ma20'] = group['close'].rolling(20).mean()

        high_low = group['high'] - group['low']
        high_close = np.abs(group['high'] - group['close'].shift())
        low_close = np.abs(group['low'] - group['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        group['atr14'] = true_range.rolling(14).mean()

        group['volume_ma10'] = group['volume'].rolling(10).mean()
        group['volume_ma20'] = group['volume'].rolling(20).mean()

        typical_price = (group['high'] + group['low'] + group['close']) / 3
        money_flow = typical_price * group['volume']
        positive_flow = money_flow.where(typical_price > typical_price.shift(), 0).rolling(14).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(), 0).rolling(14).sum()
        mfi = 100 - (100 / (1 + positive_flow / (negative_flow + 1e-6)))
        group['mfi14'] = mfi

        group['roc_5'] = group['close'].pct_change(5) * 100
        group['roc_10'] = group['close'].pct_change(10) * 100
        group['roc_20'] = group['close'].pct_change(20) * 100
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

    factor_cols = ['ts_code', 'factor_date', 'close', 'open', 'high', 'low', 'volume',
                   'macd', 'rsi6', 'rsi12', 'rsi24', 'obv', 'volume_ratio',
                   'momentum_5', 'momentum_10', 'momentum_20',
                   'ma5', 'ma10', 'ma20', 'atr14',
                   'volume_ma5', 'volume_ma10', 'volume_ma20',
                   'mfi14', 'roc_5', 'roc_10', 'roc_20']

    features = df[factor_cols].copy()
    features = features.dropna()
    labels = labels.dropna()

    return features, labels


def filter_top_features(features_df, top_features):
    """过滤保留Top特征"""
    available_features = [f for f in top_features if f in features_df.columns]
    keep_cols = ['ts_code', 'factor_date'] + available_features
    return features_df[keep_cols]


def train_and_evaluate(features_df, labels_df, feature_filter, label):
    """训练并评估模型"""
    print("\n" + "="*80)
    print(f"{label}")
    print("="*80)

    # 应用特征过滤
    if feature_filter == 'top50':
        features_df = filter_top_features(features_df, TOP_50_FEATURES)

    dataset = features_df.merge(labels_df, on=['ts_code', 'factor_date'], how='inner')
    split_date = dataset['factor_date'].quantile(0.8)

    train_data = dataset[dataset['factor_date'] <= split_date]
    val_data = dataset[dataset['factor_date'] > split_date]

    feature_cols = [c for c in dataset.columns if c not in ['ts_code', 'factor_date', 'forward_return']]

    train_features = train_data[['ts_code', 'factor_date'] + feature_cols]
    train_labels = train_data[['ts_code', 'factor_date', 'forward_return']]
    val_features = val_data[['ts_code', 'factor_date'] + feature_cols]
    val_labels = val_data[['ts_code', 'factor_date', 'forward_return']]

    print(f"\nTrain: {len(train_features)} samples")
    print(f"Val: {len(val_features)} samples")
    print(f"Features: {len(feature_cols)}")

    model = LightGBMAlphaModel()
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
    print("Phase 1 优化验证")
    print("="*80)

    print("\n加载数据...")
    df = pd.read_parquet('data/raw/star50_daily_hfq_data_6yrs.parquet')
    df_with_factors = calculate_technical_factors(df)
    features, labels = prepare_dataset(df_with_factors)

    print("\n应用特征工程（修复后）...")
    engineer = FeatureEngineer('../configs/feature_config.yaml')
    enhanced_features = engineer.transform(features.copy())

    # 测试三种配置
    print("\n" + "="*80)
    print("对比测试")
    print("="*80)

    # 1. Baseline (25因子)
    baseline_results = train_and_evaluate(
        features.copy(),
        labels.copy(),
        feature_filter=None,
        label="[1] Baseline: 25个原始因子"
    )

    # 2. Enhanced All (修复后的全部特征)
    enhanced_all_results = train_and_evaluate(
        enhanced_features.copy(),
        labels.copy(),
        feature_filter=None,
        label="[2] Enhanced All: 修复后的全部特征"
    )

    # 3. Enhanced Top50 (精选特征)
    enhanced_top50_results = train_and_evaluate(
        enhanced_features.copy(),
        labels.copy(),
        feature_filter='top50',
        label="[3] Enhanced Top50: 精选50个特征"
    )

    # 对比结果
    print("\n" + "="*80)
    print("优化效果对比")
    print("="*80)

    results_df = pd.DataFrame([
        {'配置': 'Baseline (25)', **baseline_results},
        {'配置': 'Enhanced All (124)', **enhanced_all_results},
        {'配置': 'Enhanced Top50 (50)', **enhanced_top50_results}
    ])

    print(f"\n{results_df.to_string(index=False)}")

    # 验收判断
    print("\n" + "="*80)
    print("验收标准")
    print("="*80)

    target_ic = 0.025
    best_ic = max(baseline_results['ic_mean'],
                  enhanced_all_results['ic_mean'],
                  enhanced_top50_results['ic_mean'])

    if best_ic >= target_ic:
        print(f"✓ PASS: 最佳IC={best_ic:.4f} >= 目标{target_ic}")
    else:
        print(f"⚠ 当前最佳IC={best_ic:.4f}, 目标={target_ic}")
        print(f"改进幅度: {((best_ic - baseline_results['ic_mean']) / abs(baseline_results['ic_mean'])) * 100:+.1f}%")


if __name__ == '__main__':
    main()
