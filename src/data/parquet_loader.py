"""
Parquet数据加载器
================

从Parquet文件加载股票和指数数据，构造超额收益标签
"""

import pandas as pd
import numpy as np
from typing import Tuple
from scipy import stats
from pathlib import Path


class ParquetDataLoader:
    """
    基于Parquet文件的数据加载器

    负责：
    - 加载股票和指数后复权数据
    - 计算5日超额收益标签
    - 数据预处理（去极值、标准化）
    """

    def __init__(self, data_dir: str = 'data/raw'):
        """
        初始化数据加载器

        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = Path(data_dir)
        self.stock_file = self.data_dir / 'star50_daily_hfq_data_6yrs.parquet'
        self.index_file = self.data_dir / 'star50_index_daily_6yrs.parquet'

    def load_stock_data(
        self,
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        加载股票后复权数据

        Args:
            start_date: 开始日期（可选）
            end_date: 结束日期（可选）

        Returns:
            DataFrame，列包含：ts_code, trade_date, hfq_close等
        """
        df = pd.read_parquet(self.stock_file)
        df['trade_date'] = pd.to_datetime(df['trade_date'])

        if start_date:
            df = df[df['trade_date'] >= start_date]
        if end_date:
            df = df[df['trade_date'] <= end_date]

        return df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)

    def load_index_data(
        self,
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        加载科创50指数数据

        Args:
            start_date: 开始日期（可选）
            end_date: 结束日期（可选）

        Returns:
            DataFrame，列包含：trade_date, close
        """
        df = pd.read_parquet(self.index_file)
        df['trade_date'] = pd.to_datetime(df['trade_date'])

        if start_date:
            df = df[df['trade_date'] >= start_date]
        if end_date:
            df = df[df['trade_date'] <= end_date]

        return df[['trade_date', 'close']].sort_values('trade_date').reset_index(drop=True)

    def calculate_excess_returns(
        self,
        stock_df: pd.DataFrame,
        index_df: pd.DataFrame,
        forward_days: int = 5
    ) -> pd.DataFrame:
        """
        计算超额收益（个股收益 - 指数收益）

        Args:
            stock_df: 股票数据
            index_df: 指数数据
            forward_days: 前向天数

        Returns:
            DataFrame，列：ts_code, trade_date, forward_return（超额收益）
        """
        # 计算个股未来N日收益
        stock_returns = []

        for ts_code, group in stock_df.groupby('ts_code'):
            group = group.sort_values('trade_date').reset_index(drop=True)
            group['future_close'] = group['hfq_close'].shift(-forward_days)
            group['stock_return'] = (group['future_close'] / group['hfq_close'] - 1)
            stock_returns.append(group[['ts_code', 'trade_date', 'stock_return']])

        stock_ret_df = pd.concat(stock_returns, ignore_index=True)

        # 计算指数未来N日收益
        index_df = index_df.sort_values('trade_date').reset_index(drop=True)
        index_df['future_close'] = index_df['close'].shift(-forward_days)
        index_df['index_return'] = (index_df['future_close'] / index_df['close'] - 1)
        index_ret_df = index_df[['trade_date', 'index_return']]

        # 合并计算超额收益
        result = stock_ret_df.merge(index_ret_df, on='trade_date', how='inner')
        result['forward_return'] = result['stock_return'] - result['index_return']

        return result[['ts_code', 'trade_date', 'forward_return']].dropna()

    def calculate_basic_factors(self, stock_df: pd.DataFrame) -> pd.DataFrame:
        """
        计算基础技术因子

        Args:
            stock_df: 股票数据

        Returns:
            DataFrame，包含因子列
        """
        factors_list = []

        for ts_code, group in stock_df.groupby('ts_code'):
            group = group.sort_values('trade_date').reset_index(drop=True)

            # 价格相关因子
            group['ma5'] = group['hfq_close'].rolling(5).mean()
            group['ma10'] = group['hfq_close'].rolling(10).mean()
            group['ma20'] = group['hfq_close'].rolling(20).mean()
            group['ma60'] = group['hfq_close'].rolling(60).mean()

            # 成交量因子
            group['vol_ma5'] = group['vol'].rolling(5).mean()
            group['vol_ma10'] = group['vol'].rolling(10).mean()
            group['vol_ma20'] = group['vol'].rolling(20).mean()

            # 动量因子
            group['return_1d'] = group['pct_chg'] / 100
            group['return_5d'] = group['hfq_close'].pct_change(5)
            group['return_10d'] = group['hfq_close'].pct_change(10)
            group['return_20d'] = group['hfq_close'].pct_change(20)

            # 波动率因子
            group['volatility_5d'] = group['return_1d'].rolling(5).std()
            group['volatility_20d'] = group['return_1d'].rolling(20).std()

            # 价格相对位置
            group['price_to_ma20'] = group['hfq_close'] / group['ma20']

            factors_list.append(group)

        return pd.concat(factors_list, ignore_index=True)

    def prepare_ml_data(
        self,
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        准备机器学习数据（正确的处理顺序）

        处理流程：
        1. 合并特征和标签
        2. 清洗标签（去极值，不标准化）
        3. 标准化特征（不包括标签）
        4. 返回最终数据

        Args:
            features_df: 特征DataFrame
            labels_df: 标签DataFrame

        Returns:
            处理后的DataFrame
        """
        # 1. 合并数据
        data = features_df.merge(
            labels_df[['ts_code', 'trade_date', 'forward_return']],
            on=['ts_code', 'trade_date'],
            how='inner'
        )

        print(f"原始样本数: {len(data)}")

        # 2. 删除NaN
        data = data.dropna()
        print(f"删除NaN后: {len(data)}")

        # 3. 清洗forward_return（去除极端值）
        data = data[
            (data['forward_return'] > -0.5) &
            (data['forward_return'] < 0.5)
        ]
        print(f"过滤极端值后: {len(data)}")

        # Winsorize标签（1%分位数截断）
        data['forward_return_clean'] = stats.mstats.winsorize(
            data['forward_return'].values,
            limits=[0.01, 0.01]
        )

        print(f"\nForward return统计:")
        print(f"  Mean: {data['forward_return_clean'].mean():.4f}")
        print(f"  Std: {data['forward_return_clean'].std():.4f}")
        print(f"  Min: {data['forward_return_clean'].min():.4f}")
        print(f"  Max: {data['forward_return_clean'].max():.4f}")

        # 4. 标准化特征（不包括标签和元数据）
        meta_cols = ['ts_code', 'trade_date', 'forward_return', 'forward_return_clean']
        feature_cols = [col for col in data.columns if col not in meta_cols]

        # 对每个特征进行标准化
        for col in feature_cols:
            # MAD去极值
            median = data[col].median()
            mad = np.median(np.abs(data[col] - median))
            if mad > 0:
                data[col] = data[col].clip(
                    lower=median - 3 * mad,
                    upper=median + 3 * mad
                )

            # Z-Score标准化
            mean = data[col].mean()
            std = data[col].std()
            if std > 0:
                data[col] = (data[col] - mean) / std

        # 5. 重命名标签列（保留forward_return_clean作为最终标签）
        # 删除原始的forward_return列
        if 'forward_return' in data.columns and 'forward_return_clean' in data.columns:
            data = data.drop(columns=['forward_return'])

        # 将forward_return_clean重命名为forward_return
        data = data.rename(columns={'forward_return_clean': 'forward_return'})

        print(f"\n最终数据shape: {data.shape}")
        print(f"特征数: {len(feature_cols)}")

        return data

    def load_and_prepare(
        self,
        start_date: str = '2020-01-01',
        end_date: str = '2024-12-31',
        forward_days: int = 5
    ) -> Tuple[pd.DataFrame, list]:
        """
        一站式加载和准备数据

        Args:
            start_date: 开始日期
            end_date: 结束日期
            forward_days: 前向天数

        Returns:
            (prepared_data, feature_columns)
        """
        print("="*70)
        print("加载和准备数据")
        print("="*70)

        # 1. 加载原始数据
        print("\n1. 加载原始数据...")
        stock_df = self.load_stock_data(start_date, end_date)
        index_df = self.load_index_data(start_date, end_date)

        print(f"  股票数据: {len(stock_df)} 条, {stock_df['ts_code'].nunique()} 只股票")
        print(f"  指数数据: {len(index_df)} 条")
        print(f"  日期范围: {stock_df['trade_date'].min()} ~ {stock_df['trade_date'].max()}")

        # 2. 计算因子
        print("\n2. 计算技术因子...")
        features_df = self.calculate_basic_factors(stock_df)

        # 3. 计算超额收益
        print("\n3. 计算超额收益...")
        labels_df = self.calculate_excess_returns(stock_df, index_df, forward_days)

        # 4. 准备ML数据
        print("\n4. 准备机器学习数据...")
        prepared_data = self.prepare_ml_data(features_df, labels_df)

        # 提取特征列
        meta_cols = ['ts_code', 'trade_date', 'forward_return']
        feature_cols = [col for col in prepared_data.columns if col not in meta_cols]

        print("\n" + "="*70)
        print("数据准备完成！")
        print("="*70)

        return prepared_data, feature_cols
