"""
因子数据加载器
==============

从数据库加载因子和收益率数据，构建训练集。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import psycopg2
import os


class FactorDataLoader:
    """
    因子数据加载器

    负责：
    - 从数据库加载因子数据
    - 计算标签（未来收益率）
    - 数据预处理（去极值、标准化）
    - 构建时序数据集
    """

    def __init__(self, db_name: str = 'star50_quant'):
        """
        初始化数据加载器

        Args:
            db_name: 数据库名称
        """
        self.db_name = db_name
        self.conn = None

    def connect(self):
        """连接数据库"""
        if self.conn is None:
            user = os.getenv('USER', 'mac')
            self.conn = psycopg2.connect(
                dbname=self.db_name,
                user=user,
                host='localhost'
            )

    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def load_factors(
        self,
        start_date: str,
        end_date: str,
        stocks: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        加载因子数据

        Args:
            start_date: 开始日期
            end_date: 结束日期
            stocks: 股票代码列表（None表示所有股票）

        Returns:
            DataFrame，列：ts_code, factor_date, factor_name, factor_value
        """
        self.connect()

        query = """
            SELECT ts_code, factor_date, factor_name, factor_value
            FROM factor_values
            WHERE factor_date >= %s AND factor_date <= %s
        """
        params = [start_date, end_date]

        if stocks:
            query += " AND ts_code = ANY(%s)"
            params.append(stocks)

        query += " ORDER BY factor_date, ts_code, factor_name"

        df = pd.read_sql(query, self.conn, params=params)
        return df

    def load_prices(
        self,
        start_date: str,
        end_date: str,
        stocks: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        加载价格数据

        Args:
            start_date: 开始日期
            end_date: 结束日期
            stocks: 股票代码列表

        Returns:
            DataFrame，列：ts_code, trade_date, close
        """
        self.connect()

        query = """
            SELECT ts_code, trade_date, close
            FROM stock_daily
            WHERE trade_date >= %s AND trade_date <= %s
        """
        params = [start_date, end_date]

        if stocks:
            query += " AND ts_code = ANY(%s)"
            params.append(stocks)

        query += " ORDER BY trade_date, ts_code"

        df = pd.read_sql(query, self.conn, params=params)
        return df

    def calculate_returns(
        self,
        prices: pd.DataFrame,
        forward_days: int = 5
    ) -> pd.DataFrame:
        """
        计算未来收益率

        Args:
            prices: 价格DataFrame
            forward_days: 前向天数

        Returns:
            DataFrame，列：ts_code, trade_date, forward_return
        """
        # 按股票分组计算收益率
        returns_list = []

        for ts_code, group in prices.groupby('ts_code'):
            group = group.sort_values('trade_date').copy()

            # 计算未来收益率
            group['forward_return'] = (
                group['close'].shift(-forward_days) / group['close'] - 1
            )

            returns_list.append(group[['ts_code', 'trade_date', 'forward_return']])

        returns_df = pd.concat(returns_list, ignore_index=True)
        return returns_df

    def pivot_factors(self, factors_df: pd.DataFrame) -> pd.DataFrame:
        """
        将因子数据透视为宽表格式

        Args:
            factors_df: 长格式因子DataFrame

        Returns:
            宽格式DataFrame，列：ts_code, factor_date, factor1, factor2, ...
        """
        wide_df = factors_df.pivot_table(
            index=['ts_code', 'factor_date'],
            columns='factor_name',
            values='factor_value'
        ).reset_index()

        return wide_df

    def winsorize(self, df: pd.DataFrame, n_sigma: float = 3.0) -> pd.DataFrame:
        """
        去极值（MAD方法）

        Args:
            df: 因子DataFrame
            n_sigma: MAD倍数

        Returns:
            去极值后的DataFrame
        """
        factor_cols = [col for col in df.columns if col not in ['ts_code', 'factor_date']]

        for col in factor_cols:
            median = df[col].median()
            mad = (df[col] - median).abs().median()

            if mad > 0:
                upper = median + n_sigma * mad
                lower = median - n_sigma * mad
                df[col] = df[col].clip(lower, upper)

        return df

    def standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        标准化（截面Z-Score）

        Args:
            df: 因子DataFrame

        Returns:
            标准化后的DataFrame
        """
        factor_cols = [col for col in df.columns if col not in ['ts_code', 'factor_date']]

        # 按日期分组标准化
        for date, group in df.groupby('factor_date'):
            for col in factor_cols:
                mean = group[col].mean()
                std = group[col].std()

                if std > 0:
                    df.loc[group.index, col] = (group[col] - mean) / std
                else:
                    df.loc[group.index, col] = 0

        return df

    def build_dataset(
        self,
        start_date: str,
        end_date: str,
        forward_days: int = 5,
        lookback_days: int = 20,
        stocks: Optional[List[str]] = None,
        enable_feature_engineering: bool = False,
        feature_config_path: str = '../configs/feature_config.yaml'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        构建完整的训练数据集

        Args:
            start_date: 开始日期
            end_date: 结束日期
            forward_days: 预测未来N天收益率
            lookback_days: 回看N天历史（用于时序模型）
            stocks: 股票列表
            enable_feature_engineering: 是否启用特征工程（30因子→160+特征）
            feature_config_path: 特征工程配置文件路径

        Returns:
            (features_df, labels_df)
        """
        print(f"Loading factors from {start_date} to {end_date}...")
        factors_df = self.load_factors(start_date, end_date, stocks)

        print(f"Loading prices...")
        prices_df = self.load_prices(start_date, end_date, stocks)

        print(f"Calculating forward returns (forward_days={forward_days})...")
        returns_df = self.calculate_returns(prices_df, forward_days)

        print(f"Pivoting factors to wide format...")
        factors_wide = self.pivot_factors(factors_df)

        print(f"Preprocessing: winsorize and standardize...")
        factors_wide = self.winsorize(factors_wide)
        factors_wide = self.standardize(factors_wide)

        # 特征工程集成
        if enable_feature_engineering:
            print(f"\n{'='*60}")
            print(f"启用特征工程: 30因子 → 160+特征")
            print(f"{'='*60}")
            from src.features.engineering import FeatureEngineer

            engineer = FeatureEngineer(feature_config_path)
            factors_wide = engineer.transform(factors_wide)

            # 特征验证
            validation_results = engineer.validate_features(factors_wide)
            print(f"\n特征验证结果:")
            print(f"  - 总特征数: {validation_results['total_features']}")
            print(f"  - NaN问题特征: {len(validation_results['nan_ratio'])}")
            print(f"  - Inf问题特征: {len(validation_results['inf_count'])}")
            print(f"  - 高相关特征对: {len(validation_results['high_correlation_pairs'])}")
            print(f"{'='*60}\n")

        print(f"Merging features and labels...")
        # 合并因子和收益率
        dataset = factors_wide.merge(
            returns_df,
            left_on=['ts_code', 'factor_date'],
            right_on=['ts_code', 'trade_date'],
            how='inner'
        )

        # 删除缺失值
        dataset = dataset.dropna()

        # 分离特征和标签
        feature_cols = [col for col in dataset.columns
                       if col not in ['ts_code', 'factor_date', 'trade_date', 'forward_return']]

        features = dataset[['ts_code', 'factor_date'] + feature_cols]
        labels = dataset[['ts_code', 'factor_date', 'forward_return']]

        print(f"Dataset built: {len(dataset)} samples, {len(feature_cols)} features")

        return features, labels

    def get_stock_list(self) -> List[str]:
        """获取所有股票代码"""
        self.connect()

        query = "SELECT DISTINCT ts_code FROM stock_daily ORDER BY ts_code"
        df = pd.read_sql(query, self.conn)

        return df['ts_code'].tolist()
