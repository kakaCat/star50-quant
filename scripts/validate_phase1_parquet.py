#!/usr/bin/env python3
"""
Phase 1验证脚本（使用Parquet数据）
===================================

从原始价格数据计算技术因子，然后验证特征工程效果。
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from src.features.engineering import FeatureEngineer
from src.models.lgbm_model import LightGBMAlphaModel


def calculate_technical_factors(df):
    """从价格数据计算技术因子"""
    print("计算技术因子...")

    df = df.sort_values(['ts_code', 'trade_date'])
    result = []

    for ts_code, group in df.groupby('ts_code'):
        group = group.sort_values('trade_date').copy()

        # 价格和成交量
        group['volume'] = group['vol']

        # MACD
        ema12 = group['close'].ewm(span=12, adjust=False).mean()
        ema26 = group['close'].ewm(span=26, adjust=False).mean()
        group['macd'] = ema12 - ema26

        # RSI
        delta = group['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=6).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=6).mean()
        rs = gain / (loss + 1e-6)
        group['rsi6'] = 100 - (100 / (1 + rs))

        gain = (delta.where(delta > 0, 0)).rolling(window=12).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=12).mean()
        rs = gain / (loss + 1e-6)
        group['rsi12'] = 100 - (100 / (1 + rs))

        gain = (delta.where(delta > 0, 0)).rolling(window=24).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=24).mean()
        rs = gain / (loss + 1e-6)
        group['rsi24'] = 100 - (100 / (1 + rs))

        # OBV
        obv = (np.sign(group['close'].diff()) * group['volume']).fillna(0).cumsum()
        group['obv'] = obv

        # 量比
        group['volume_ma5'] = group['volume'].rolling(5).mean()
        group['volume_ratio'] = group['volume'] / (group['volume_ma5'] + 1e-6)

        # 动量
        group['momentum_5'] = group['close'].pct_change(5)
        group['momentum_10'] = group['close'].pct_change(10)
        group['momentum_20'] = group['close'].pct_change(20)

        # 均线
        group['ma5'] = group['close'].rolling(5).mean()
        group['ma10'] = group['close'].rolling(10).mean()
        group['ma20'] = group['close'].rolling(20).mean()

        # ATR
        high_low = group['high'] - group['low']
        high_close = np.abs(group['high'] - group['close'].shift())
        low_close = np.abs(group['low'] - group['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        group['atr14'] = true_range.rolling(14).mean()

        # 成交量均线
        group['volume_ma10'] = group['volume'].rolling(10).mean()
        group['volume_ma20'] = group['volume'].rolling(20).mean()

        # MFI
        typical_price = (group['high'] + group['low'] + group['close']) / 3
        money_flow = typical_price * group['volume']
        positive_flow = money_flow.where(typical_price > typical_price.shift(), 0).rolling(14).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(), 0).rolling(14).sum()
        mfi = 100 - (100 / (1 + positive_flow / (negative_flow + 1e-6)))
        group['mfi14'] = mfi

        # ROC
        group['roc_5'] = group['close'].pct_change(5) * 100
        group['roc_10'] = group['close'].pct_change(10) * 100
        group['roc_20'] = group['close'].pct_change(20) * 100

        result.append(group)

    result_df = pd.concat(result, ignore_index=True)

    # 重命名factor_date
    result_df['factor_date'] = result_df['trade_date']

    print(f"  - 计算完成，共{len(result_df)}条记录")

    return result_df


def prepare_dataset(df, forward_days=5):
    """准备训练数据集"""
    print(f"\n准备数据集（forward_days={forward_days}）...")

    # 计算未来收益率
    labels_data = []
    for ts_code, group in df.groupby('ts_code'):
        group = group.sort_values('trade_date').copy()
        group['forward_return'] = (group['close'].shift(-forward_days) - group['close']) / group['close']

        for _, row in group.iterrows():
            labels_data.append({
                'ts_code': row['ts_code'],
                'factor_date': row['factor_date'],
                'forward_return': row['forward_return']
            })

    labels = pd.DataFrame(labels_data)

    # 选择因子列
    factor_cols = ['ts_code', 'factor_date', 'close', 'open', 'high', 'low', 'volume',
                   'macd', 'rsi6', 'rsi12', 'rsi24', 'obv', 'volume_ratio',
                   'momentum_5', 'momentum_10', 'momentum_20',
                   'ma5', 'ma10', 'ma20', 'atr14',
                   'volume_ma5', 'volume_ma10', 'volume_ma20',
                   'mfi14', 'roc_5', 'roc_10', 'roc_20']

    features = df[factor_cols].copy()

    # 删除NaN
    features = features.dropna()
    labels = labels.dropna()

    print(f"  - 特征: {len(features)} 样本, {len(factor_cols)-2} 个因子")
    print(f"  - 标签: {len(labels)} 样本")

    return features, labels


def train_and_evaluate(features_df, labels_df, enable_features, label):
    """训练并评估模型"""
    print("\n" + "="*80)
    print(f"{label}")
    print("="*80)

    # 特征工程（如果启用）
    if enable_features:
        engineer = FeatureEngineer('../configs/feature_config.yaml')
        features_df = engineer.transform(features_df)

    # 合并特征和标签
    dataset = features_df.merge(
        labels_df,
        on=['ts_code', 'factor_date'],
        how='inner'
    )

    # 时间序列划分
    split_date = dataset['factor_date'].quantile(0.8)
    train_data = dataset[dataset['factor_date'] <= split_date]
    val_data = dataset[dataset['factor_date'] > split_date]

    feature_cols = [c for c in dataset.columns
                   if c not in ['ts_code', 'factor_date', 'forward_return']]

    train_features = train_data[['ts_code', 'factor_date'] + feature_cols]
    train_labels = train_data[['ts_code', 'factor_date', 'forward_return']]
    val_features = val_data[['ts_code', 'factor_date'] + feature_cols]
    val_labels = val_data[['ts_code', 'factor_date', 'forward_return']]

    print(f"\nTrain: {len(train_features)} samples")
    print(f"Val: {len(val_features)} samples")
    print(f"Features: {len(feature_cols)}")

    # 训练模型
    model = LightGBMAlphaModel()
    model.train(
        features=train_features,
        labels=train_labels,
        num_boost_round=200,
        early_stopping_rounds=20,
        verbose_eval=50
    )

    # 验证集评估
    val_pred = model.predict(val_features)
    val_true = val_labels['forward_return'].values

    # 计算IC
    ic = np.corrcoef(val_pred, val_true)[0, 1]

    # 按日期计算IC序列
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
    print("Phase 1 验证: 特征工程效果对比（真实数据）")
    print("="*80)

    # 加载原始数据
    print("\n加载数据...")
    df = pd.read_parquet('data/raw/star50_daily_hfq_data_6yrs.parquet')
    print(f"  - {len(df)} 条记录")
    print(f"  - {df['ts_code'].nunique()} 只股票")
    print(f"  - {df['trade_date'].min()} 到 {df['trade_date'].max()}")

    # 计算技术因子
    df_with_factors = calculate_technical_factors(df)

    # 准备数据集
    features, labels = prepare_dataset(df_with_factors, forward_days=5)

    # Baseline: 25因子
    baseline_results = train_and_evaluate(
        features.copy(),
        labels.copy(),
        enable_features=False,
        label="Baseline: 25个技术因子"
    )

    # Enhanced: 特征工程
    enhanced_results = train_and_evaluate(
        features.copy(),
        labels.copy(),
        enable_features=True,
        label="Enhanced: 特征工程增强"
    )

    # 对比结果
    print("\n" + "="*80)
    print("Phase 1 验证结果对比")
    print("="*80)

    print(f"\n{'指标':<20} {'Baseline':<15} {'Enhanced':<15} {'提升':<15}")
    print("-" * 80)

    metrics = [
        ('特征数', 'n_features', ''),
        ('IC (整体)', 'ic', '.4f'),
        ('IC (日度均值)', 'ic_mean', '.4f'),
        ('IC 标准差', 'ic_std', '.4f'),
        ('IR', 'ic_ir', '.4f'),
        ('IC>0比例', 'ic_positive_ratio', '.2%')
    ]

    for label, key, fmt in metrics:
        baseline_val = baseline_results[key]
        enhanced_val = enhanced_results[key]

        if fmt:
            baseline_str = f"{baseline_val:{fmt}}"
            enhanced_str = f"{enhanced_val:{fmt}}"
        else:
            baseline_str = str(baseline_val)
            enhanced_str = str(enhanced_val)

        if key in ['ic', 'ic_mean', 'ic_ir']:
            if abs(baseline_val) > 1e-6:
                improvement = ((enhanced_val - baseline_val) / abs(baseline_val)) * 100
                improvement_str = f"{improvement:+.1f}%"
            else:
                improvement_str = "-"
        else:
            improvement_str = "-"

        print(f"{label:<20} {baseline_str:<15} {enhanced_str:<15} {improvement_str:<15}")

    # 验收判断
    print("\n" + "="*80)
    print("Phase 1 验收标准")
    print("="*80)

    target_ic = 0.028
    achieved_ic = enhanced_results['ic_mean']

    if achieved_ic >= target_ic:
        print(f"✓ PASS: IC={achieved_ic:.4f} >= 目标{target_ic}")
        print(f"✓ Phase 1完成，准备进入Phase 2（模型集成）")
    else:
        print(f"✗ 当前: IC={achieved_ic:.4f}, 目标={target_ic}")
        print(f"提升幅度: {((achieved_ic - baseline_results['ic_mean']) / abs(baseline_results['ic_mean'])) * 100:+.1f}%")
        print(f"\n特征工程系统已成功实现并集成")


if __name__ == '__main__':
    main()
