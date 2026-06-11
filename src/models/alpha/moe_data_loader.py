"""
MoE模型专用数据加载器（基于Parquet）
====================================

从Parquet文件加载数据，构建MoE训练数据集。
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
from pathlib import Path


class MoEParquetDataLoader:
    """
    MoE模型数据加载器（Parquet版本）

    负责：
    - 从Parquet文件加载股票和指数数据
    - 计算Beta并剥离市场收益
    - 分离个股特征和环境特征
    - 构建训练/测试数据集
    """

    def __init__(
        self,
        stock_data_path: str = 'data/raw/star50_daily_hfq_data_6yrs.parquet',
        index_data_path: str = 'data/raw/star50_index_daily_6yrs.parquet'
    ):
        """
        初始化数据加载器

        Args:
            stock_data_path: 股票数据Parquet文件路径
            index_data_path: 指数数据Parquet文件路径
        """
        self.stock_data_path = stock_data_path
        self.index_data_path = index_data_path

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        加载股票和指数数据

        Returns:
            (stock_df, index_df)
        """
        print(f"Loading stock data from {self.stock_data_path}...")
        stock_df = pd.read_parquet(self.stock_data_path)

        print(f"Loading index data from {self.index_data_path}...")
        index_df = pd.read_parquet(self.index_data_path)

        # 确保日期列为datetime类型
        if 'trade_date' in stock_df.columns:
            stock_df['trade_date'] = pd.to_datetime(stock_df['trade_date'])
        if 'trade_date' in index_df.columns:
            index_df['trade_date'] = pd.to_datetime(index_df['trade_date'])

        print(f"Stock data: {len(stock_df)} rows, {len(stock_df.columns)} columns")
        print(f"Index data: {len(index_df)} rows, {len(index_df.columns)} columns")

        return stock_df, index_df

    def build_stock_features(
        self,
        stock_df: pd.DataFrame,
        windows: List[int] = [3, 5, 10, 20]
    ) -> pd.DataFrame:
        """
        构建个股特征

        Args:
            stock_df: 股票数据
            windows: 时间窗口列表

        Returns:
            包含个股特征的DataFrame
        """
        print("Building stock features...")

        df = stock_df.copy()
        df = df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
        grouped = df.groupby('ts_code')

        # 基础收益率
        df['ret_1d'] = grouped['hfq_close'].pct_change(1)

        # 日内波动
        df['hl_spread'] = (df['hfq_high'] - df['hfq_low']) / df['hfq_close']
        df['gap_open'] = df['hfq_open'] / grouped['hfq_close'].shift(1) - 1

        # 时间窗口特征
        stock_features = ['hl_spread', 'gap_open']

        for window in windows:
            # 动量
            col_mom = f'mom_{window}d'
            df[col_mom] = grouped['hfq_close'].pct_change(window)
            stock_features.append(col_mom)

            # 波动率
            col_vol = f'vol_{window}d'
            df[col_vol] = grouped['ret_1d'].transform(lambda x: x.rolling(window).std())
            stock_features.append(col_vol)

            # 偏度
            col_skew = f'skew_{window}d'
            df[col_skew] = grouped['ret_1d'].transform(lambda x: x.rolling(window).skew())
            stock_features.append(col_skew)

            # 成交量动量
            col_vol_ma = f'vol_ma_{window}d'
            df[col_vol_ma] = grouped['vol'].transform(lambda x: x.rolling(window).mean())
            col_vol_mom = f'vol_mom_{window}d'
            df[col_vol_mom] = df['vol'] / df[col_vol_ma] - 1
            stock_features.append(col_vol_mom)

            # 量价相关性
            col_pv_corr = f'pv_corr_{window}d'
            df[col_pv_corr] = grouped.apply(
                lambda x: x['ret_1d'].rolling(window).corr(x['vol'])
            ).reset_index(level=0, drop=True)
            stock_features.append(col_pv_corr)

        # 偏离度
        df['bias_20d'] = df['hfq_close'] / grouped['hfq_close'].transform(
            lambda x: x.rolling(20).mean()
        ) - 1
        stock_features.append('bias_20d')

        print(f"Created {len(stock_features)} stock features")

        # 截面标准化
        print("Applying cross-sectional standardization...")
        for f in stock_features:
            df[f] = df.groupby('trade_date')[f].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-8)
            )
            df[f] = df[f].fillna(0)

        return df[['ts_code', 'trade_date'] + stock_features]

    def build_regime_features(
        self,
        index_df: pd.DataFrame,
        stock_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        构建环境特征

        Args:
            index_df: 指数数据
            stock_df: 股票数据（用于计算截面统计量）

        Returns:
            包含环境特征的DataFrame
        """
        print("Building regime features...")

        df = index_df.copy()
        df = df.sort_values('trade_date').reset_index(drop=True)

        # 指数收益率
        df['idx_ret_1d'] = df['close'].pct_change(1)

        # 指数偏离度
        df['idx_bias_20d'] = df['close'] / df['close'].rolling(20).mean() - 1
        df['idx_bias_60d'] = df['close'] / df['close'].rolling(60).mean() - 1

        # 指数波动率
        df['idx_vol_20d'] = df['idx_ret_1d'].rolling(20).std()

        # 成交量动量
        df['idx_vol_mom'] = df['vol'] / df['vol'].rolling(60).mean() - 1

        # 截面离散度（从股票数据计算）
        stock_df['ret_1d'] = stock_df.groupby('ts_code')['hfq_close'].pct_change(1)
        cs_dispersion = stock_df.groupby('trade_date')['ret_1d'].std().reset_index()
        cs_dispersion.columns = ['trade_date', 'cs_dispersion']
        df = df.merge(cs_dispersion, on='trade_date', how='left')

        regime_features = [
            'idx_ret_1d', 'idx_bias_20d', 'idx_bias_60d',
            'idx_vol_20d', 'idx_vol_mom', 'cs_dispersion'
        ]

        print(f"Created {len(regime_features)} regime features")

        # 时序标准化
        print("Applying time-series standardization...")
        for f in regime_features:
            mean = df[f].mean()
            std = df[f].std()
            if std > 0:
                df[f] = (df[f] - mean) / std
            df[f] = df[f].fillna(0)

        return df[['trade_date'] + regime_features]

    def calculate_beta_and_residual(
        self,
        stock_df: pd.DataFrame,
        index_df: pd.DataFrame,
        forward_days: int = 5,
        beta_window: int = 60
    ) -> pd.DataFrame:
        """
        计算Beta并剥离市场收益

        Args:
            stock_df: 股票数据
            index_df: 指数数据
            forward_days: 预测未来N天收益率
            beta_window: Beta计算滚动窗口

        Returns:
            包含beta和residual的DataFrame
        """
        print(f"Calculating Beta (window={beta_window}) and residual returns...")

        # 计算未来收益率
        stock_df = stock_df.sort_values(['ts_code', 'trade_date']).copy()
        grouped = stock_df.groupby('ts_code')
        stock_df['forward_return'] = (
            grouped['hfq_close'].shift(-forward_days) / grouped['hfq_open'].shift(-1) - 1
        )

        # 计算指数收益率
        index_df = index_df.sort_values('trade_date').copy()
        index_df['idx_ret_1d'] = index_df['close'].pct_change(1)

        # 计算未来指数收益率
        index_df['idx_future_ret'] = (
            index_df['close'].shift(-forward_days) / index_df['open'].shift(-1) - 1
        )

        # 合并数据
        merged = stock_df[['ts_code', 'trade_date', 'forward_return']].merge(
            index_df[['trade_date', 'idx_ret_1d', 'idx_future_ret']],
            on='trade_date',
            how='inner'
        )

        # 按股票分组计算Beta
        result_list = []
        for ts_code, group in merged.groupby('ts_code'):
            group = group.sort_values('trade_date').copy()

            # 滚动协方差和方差
            group['cov_60d'] = group['forward_return'].rolling(beta_window).cov(
                group['idx_ret_1d']
            )
            group['idx_var_60d'] = group['idx_ret_1d'].rolling(beta_window).var()

            # Beta
            group['beta'] = group['cov_60d'] / (group['idx_var_60d'] + 1e-8)
            group['beta'] = group['beta'].fillna(1.0)

            # Residual = 股票收益 - Beta × 指数收益
            group['residual_return'] = (
                group['forward_return'] - group['beta'] * group['idx_future_ret']
            )

            result_list.append(group)

        result_df = pd.concat(result_list, ignore_index=True)

        print(f"Residual returns calculated for {len(result_df)} samples")

        return result_df[['ts_code', 'trade_date', 'forward_return', 'beta', 'residual_return']]

    def build_moe_dataset(
        self,
        forward_days: int = 5,
        beta_window: int = 60,
        windows: List[int] = [3, 5, 10, 20]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        构建完整的MoE数据集

        Args:
            forward_days: 预测未来N天收益率
            beta_window: Beta计算滚动窗口
            windows: 特征计算时间窗口

        Returns:
            (stock_features, regime_features, labels)
        """
        print("\n" + "="*70)
        print("Building MoE Dataset from Parquet Files")
        print("="*70 + "\n")

        # 1. 加载数据
        stock_df, index_df = self.load_data()

        # 2. 构建个股特征
        stock_features = self.build_stock_features(stock_df, windows)

        # 3. 构建环境特征
        regime_features = self.build_regime_features(index_df, stock_df)

        # 4. 计算Beta和Residual
        labels = self.calculate_beta_and_residual(
            stock_df, index_df, forward_days, beta_window
        )

        # 5. 删除缺失值
        stock_features = stock_features.dropna()
        regime_features = regime_features.dropna()
        labels = labels.dropna()

        print("\n" + "="*70)
        print("Dataset Summary:")
        print(f"  Stock features: {len(stock_features)} samples")
        print(f"  Regime features: {len(regime_features)} dates")
        print(f"  Labels: {len(labels)} samples")
        print("="*70 + "\n")

        return stock_features, regime_features, labels
