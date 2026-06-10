"""
多窗口数据加载器
================

为集成学习准备多窗口预测数据集。
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict


class MultiWindowDataLoader:
    """
    多窗口数据加载器

    为不同预测窗口生成标签，使用相同的特征集。
    """

    def __init__(self, windows: List[int] = [1, 3, 5, 10, 20]):
        """
        初始化

        Args:
            windows: 预测窗口列表（天数）
        """
        self.windows = windows

    def calculate_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算核心因子

        Args:
            df: 原始价格数据

        Returns:
            包含因子的DataFrame
        """
        print("计算核心因子...")
        df = df.sort_values(['ts_code', 'trade_date'])
        result = []

        for ts_code, group in df.groupby('ts_code'):
            group = group.sort_values('trade_date').copy()

            # 9个核心因子
            group['momentum_5'] = group['close'].pct_change(5)
            group['momentum_10'] = group['close'].pct_change(10)
            group['momentum_20'] = group['close'].pct_change(20)

            group['volatility_10'] = group['close'].pct_change().rolling(10).std()
            group['volatility_20'] = group['close'].pct_change().rolling(20).std()

            group['volume_ratio'] = group['vol'] / group['vol'].rolling(20).mean()

            high_low = group['high'] - group['low']
            high_close = np.abs(group['high'] - group['close'].shift())
            low_close = np.abs(group['low'] - group['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            group['atr_ratio'] = true_range.rolling(14).mean() / group['close']

            group['ma5'] = group['close'].rolling(5).mean()
            group['ma20'] = group['close'].rolling(20).mean()
            group['ma_ratio'] = group['ma5'] / group['ma20']

            delta = group['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-6)
            group['rsi14'] = 100 - (100 / (1 + rs))

            group['factor_date'] = group['trade_date']
            result.append(group)

        return pd.concat(result, ignore_index=True)

    def generate_multi_window_labels(self, df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
        """
        为多个窗口生成标签

        Args:
            df: 包含价格的DataFrame

        Returns:
            字典 {window: labels_df}
        """
        print(f"生成多窗口标签: {self.windows}")

        labels_dict = {}

        for window in self.windows:
            print(f"  - Window {window}天...")
            labels_data = []

            for ts_code, group in df.groupby('ts_code'):
                group = group.sort_values('trade_date').copy()
                # 计算未来N天收益率
                group['forward_return'] = group['close'].pct_change(window).shift(-window)

                for _, row in group.iterrows():
                    labels_data.append({
                        'ts_code': row['ts_code'],
                        'factor_date': row['factor_date'],
                        'forward_return': row['forward_return'],
                        'window': window
                    })

            labels_df = pd.DataFrame(labels_data).dropna()
            labels_dict[window] = labels_df
            print(f"    → {len(labels_df)} 样本")

        return labels_dict

    def build_multi_window_dataset(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[int, pd.DataFrame]]:
        """
        构建多窗口数据集

        Args:
            df: 原始价格数据（包含ts_code, trade_date, open, high, low, close, vol）

        Returns:
            (features, labels_dict)
            - features: 特征DataFrame
            - labels_dict: {window: labels_df}
        """
        print("\n" + "="*60)
        print("构建多窗口数据集")
        print("="*60)

        # 计算特征
        df_with_factors = self.calculate_factors(df)

        # 提取特征列
        feature_cols = [
            'ts_code', 'factor_date',
            'momentum_5', 'momentum_10', 'momentum_20',
            'volatility_10', 'volatility_20',
            'volume_ratio', 'atr_ratio',
            'ma_ratio', 'rsi14'
        ]

        features = df_with_factors[feature_cols].copy().dropna()
        print(f"\n特征: {len(features)} 样本, {len(feature_cols)-2} 个因子")

        # 生成多窗口标签
        labels_dict = self.generate_multi_window_labels(df_with_factors)

        print("\n" + "="*60)
        print("数据集构建完成")
        print("="*60)

        return features, labels_dict

    def split_train_val(
        self,
        features: pd.DataFrame,
        labels_dict: Dict[int, pd.DataFrame],
        train_ratio: float = 0.8
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]]:
        """
        时间序列划分训练集和验证集

        Args:
            features: 特征DataFrame
            labels_dict: 标签字典
            train_ratio: 训练集比例

        Returns:
            (train_features, val_features, split_labels_dict)
            - split_labels_dict: {window: (train_labels, val_labels)}
        """
        split_date = features['factor_date'].quantile(train_ratio)

        train_features = features[features['factor_date'] <= split_date]
        val_features = features[features['factor_date'] > split_date]

        split_labels_dict = {}
        for window, labels in labels_dict.items():
            train_labels = labels[labels['factor_date'] <= split_date]
            val_labels = labels[labels['factor_date'] > split_date]
            split_labels_dict[window] = (train_labels, val_labels)

        print(f"\n时间划分:")
        print(f"  训练集: {len(train_features)} 样本")
        print(f"  验证集: {len(val_features)} 样本")
        print(f"  分割日期: {split_date.date()}")

        return train_features, val_features, split_labels_dict


if __name__ == '__main__':
    """测试多窗口数据加载器"""
    import sys
    import os

    print("\n测试多窗口数据加载器")
    print("="*60)

    # 加载数据
    data_path = os.path.join(os.path.dirname(__file__), '../../data/raw/star50_daily_hfq_data_6yrs.parquet')
    df = pd.read_parquet(data_path)
    print(f"原始数据: {len(df)} 行, {df['ts_code'].nunique()} 只股票")

    # 构建多窗口数据集
    loader = MultiWindowDataLoader(windows=[1, 3, 5, 10, 20])
    features, labels_dict = loader.build_multi_window_dataset(df)

    # 划分训练验证集
    train_features, val_features, split_labels = loader.split_train_val(
        features, labels_dict
    )

    # 验证
    print("\n验证结果:")
    for window in loader.windows:
        train_labels, val_labels = split_labels[window]
        print(f"  Window {window}天: 训练{len(train_labels)}, 验证{len(val_labels)}")

    print("\n✓ 多窗口数据加载器测试通过")
