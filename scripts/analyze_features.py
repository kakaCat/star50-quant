#!/usr/bin/env python3
"""
特征重要性分析脚本
==================

分析特征的预测能力：
1. LightGBM特征重要性
2. 单因子IC分析
3. 特征相关性矩阵
4. 时序稳定性分析
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


def calculate_technical_factors(df):
    """从价格数据计算技术因子"""
    print("计算技术因子...")
    df = df.sort_values(['ts_code', 'trade_date'])
    result = []

    for ts_code, group in df.groupby('ts_code'):
        group = group.sort_values('trade_date').copy()
        group['volume'] = group['vol']

        # MACD
        ema12 = group['close'].ewm(span=12, adjust=False).mean()
        ema26 = group['close'].ewm(span=26, adjust=False).mean()
        group['macd'] = ema12 - ema26

        # RSI
        delta = group['close'].diff()
        for period in [6, 12, 24]:
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / (loss + 1e-6)
            group[f'rsi{period}'] = 100 - (100 / (1 + rs))

        # OBV
        obv = (np.sign(group['close'].diff()) * group['volume']).fillna(0).cumsum()
        group['obv'] = obv

        # 其他因子
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


def calculate_single_factor_ic(features, labels, feature_cols):
    """计算每个特征的单因子IC"""
    print("\n计算单因子IC...")

    dataset = features.merge(labels, on=['ts_code', 'factor_date'], how='inner')

    ic_results = []
    for feature in feature_cols:
        ic_series = []
        for date in dataset['factor_date'].unique():
            date_data = dataset[dataset['factor_date'] == date]
            if len(date_data) >= 10:
                ic = np.corrcoef(date_data[feature], date_data['forward_return'])[0, 1]
                if not np.isnan(ic):
                    ic_series.append(ic)

        if len(ic_series) > 0:
            ic_results.append({
                'feature': feature,
                'ic_mean': np.mean(ic_series),
                'ic_std': np.std(ic_series),
                'ic_ir': np.mean(ic_series) / (np.std(ic_series) + 1e-6),
                'ic_positive_ratio': (np.array(ic_series) > 0).mean()
            })

    ic_df = pd.DataFrame(ic_results).sort_values('ic_mean', key=abs, ascending=False)
    return ic_df


def analyze_feature_importance(features, labels):
    """LightGBM特征重要性分析"""
    print("\n训练模型并分析特征重要性...")

    dataset = features.merge(labels, on=['ts_code', 'factor_date'], how='inner')
    split_date = dataset['factor_date'].quantile(0.8)

    train_data = dataset[dataset['factor_date'] <= split_date]
    feature_cols = [c for c in dataset.columns if c not in ['ts_code', 'factor_date', 'forward_return']]

    train_features = train_data[['ts_code', 'factor_date'] + feature_cols]
    train_labels = train_data[['ts_code', 'factor_date', 'forward_return']]

    model = LightGBMAlphaModel()
    model.train(train_features, train_labels, num_boost_round=100, verbose_eval=False)

    importance_df = model.get_feature_importance()
    return importance_df


def analyze_feature_correlation(features, feature_cols):
    """特征相关性分析"""
    print("\n计算特征相关性...")

    corr_matrix = features[feature_cols].corr().abs()

    # 找出高相关特征对
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > 0.9:
                high_corr_pairs.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': corr_matrix.iloc[i, j]
                })

    return pd.DataFrame(high_corr_pairs)


def main():
    """主函数"""
    print("="*80)
    print("特征重要性分析")
    print("="*80)

    # 加载数据
    print("\n加载数据...")
    df = pd.read_parquet('data/raw/star50_daily_hfq_data_6yrs.parquet')
    df_with_factors = calculate_technical_factors(df)
    features, labels = prepare_dataset(df_with_factors)

    # 特征工程
    print("\n应用特征工程...")
    engineer = FeatureEngineer('../configs/feature_config.yaml')
    enhanced_features = engineer.transform(features.copy())

    feature_cols = [c for c in enhanced_features.columns if c not in ['ts_code', 'factor_date']]
    print(f"总特征数: {len(feature_cols)}")

    # 1. LightGBM特征重要性
    importance_df = analyze_feature_importance(enhanced_features, labels)
    print(f"\nTop 20 特征（按重要性）:")
    print(importance_df.head(20).to_string(index=False))

    # 2. 单因子IC分析
    ic_df = calculate_single_factor_ic(enhanced_features, labels, feature_cols)
    print(f"\nTop 20 特征（按IC绝对值）:")
    print(ic_df.head(20).to_string(index=False))

    # 3. 特征相关性
    corr_pairs = analyze_feature_correlation(enhanced_features, feature_cols)
    print(f"\n高相关特征对（相关性>0.9）: {len(corr_pairs)}对")
    if len(corr_pairs) > 0:
        print(corr_pairs.head(10).to_string(index=False))

    # 保存结果
    print("\n保存分析结果...")
    importance_df.to_csv('feature_importance.csv', index=False)
    ic_df.to_csv('feature_ic.csv', index=False)
    if len(corr_pairs) > 0:
        corr_pairs.to_csv('feature_correlation.csv', index=False)

    # 生成特征选择建议
    print("\n" + "="*80)
    print("特征选择建议")
    print("="*80)

    # 合并重要性和IC
    merged = importance_df.merge(ic_df[['feature', 'ic_mean', 'ic_ir']], on='feature', how='left')
    merged['score'] = merged['importance'].rank(pct=True) + merged['ic_mean'].abs().rank(pct=True)
    merged = merged.sort_values('score', ascending=False)

    top_50 = merged.head(50)
    print(f"\n推荐保留Top 50特征:")
    print(top_50[['feature', 'importance', 'ic_mean', 'ic_ir', 'score']].to_string(index=False))

    # 保存推荐特征列表
    top_50['feature'].to_csv('recommended_features_top50.txt', index=False, header=False)
    print("\n✓ 推荐特征列表已保存到 recommended_features_top50.txt")


if __name__ == '__main__':
    main()
