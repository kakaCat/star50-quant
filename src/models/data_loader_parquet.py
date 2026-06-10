"""
因子数据加载器（基于Parquet文件）
==================================

从本地parquet文件加载股票数据，计算因子和收益率，构建训练集。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from pathlib import Path


class FactorDataLoader:
    """
    因子数据加载器

    负责：
    - 从parquet文件加载股票数据
    - 计算标签（未来收益率）
    - 数据预处理（去极值、标准化）
    - 构建时序数据集
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

        # 检查文件是否存在
        if not self.stock_file.exists():
            raise FileNotFoundError(f"股票数据文件不存在: {self.stock_file}")
        if not self.index_file.exists():
            raise FileNotFoundError(f"指数数据文件不存在: {self.index_file}")

        # 缓存数据
        self._stock_data = None
        self._index_data = None

    def _load_stock_data(self) -> pd.DataFrame:
        """加载股票数据（带缓存）"""
        if self._stock_data is None:
            print(f"Loading stock data from {self.stock_file}...")
            self._stock_data = pd.read_parquet(self.stock_file)
            self._stock_data['trade_date'] = pd.to_datetime(self._stock_data['trade_date'])
        return self._stock_data

    def _load_index_data(self) -> pd.DataFrame:
        """加载指数数据（带缓存）"""
        if self._index_data is None:
            print(f"Loading index data from {self.index_file}...")
            self._index_data = pd.read_parquet(self.index_file)
            self._index_data['trade_date'] = pd.to_datetime(self._index_data['trade_date'])
        return self._index_data

    def load_prices(
        self,
        start_date: str,
        end_date: str,
        stocks: Optional[List[str]] = None,
        use_hfq: bool = True
    ) -> pd.DataFrame:
        """
        加载价格数据

        Args:
            start_date: 开始日期
            end_date: 结束日期
            stocks: 股票代码列表（None表示所有股票）
            use_hfq: 是否使用后复权价格

        Returns:
            DataFrame，列：ts_code, trade_date, close
        """
        df = self._load_stock_data()

        # 筛选日期
        df = df[(df['trade_date'] >= start_date) & (df['trade_date'] <= end_date)].copy()

        # 筛选股票
        if stocks:
            df = df[df['ts_code'].isin(stocks)]

        # 选择价格列
        price_col = 'hfq_close' if use_hfq else 'close'
        result = df[['ts_code', 'trade_date', price_col]].copy()
        result.columns = ['ts_code', 'trade_date', 'close']

        return result.sort_values(['trade_date', 'ts_code']).reset_index(drop=True)

    def load_stock_features(
        self,
        start_date: str,
        end_date: str,
        stocks: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        加载股票原始特征（OHLCV）

        Args:
            start_date: 开始日期
            end_date: 结束日期
            stocks: 股票代码列表

        Returns:
            DataFrame，包含OHLCV等基础数据
        """
        df = self._load_stock_data()

        # 筛选日期
        df = df[(df['trade_date'] >= start_date) & (df['trade_date'] <= end_date)].copy()

        # 筛选股票
        if stocks:
            df = df[df['ts_code'].isin(stocks)]

        return df.sort_values(['trade_date', 'ts_code']).reset_index(drop=True)

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

    def calculate_basic_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算基本因子（如果还没有计算过）

        Args:
            df: 包含OHLCV的DataFrame

        Returns:
            添加了基本因子的DataFrame
        """
        result_list = []

        for ts_code, group in df.groupby('ts_code'):
            group = group.sort_values('trade_date').copy()

            # 计算收益率
            group['return_1d'] = group['hfq_close'].pct_change(1)
            group['return_5d'] = group['hfq_close'].pct_change(5)
            group['return_10d'] = group['hfq_close'].pct_change(10)
            group['return_20d'] = group['hfq_close'].pct_change(20)

            # 计算波动率
            group['volatility_5d'] = group['return_1d'].rolling(5).std()
            group['volatility_20d'] = group['return_1d'].rolling(20).std()

            # 计算均线
            group['ma5'] = group['hfq_close'].rolling(5).mean()
            group['ma10'] = group['hfq_close'].rolling(10).mean()
            group['ma20'] = group['hfq_close'].rolling(20).mean()

            # 计算量价因子
            group['volume_ratio'] = group['vol'] / group['vol'].rolling(20).mean()
            group['amount_ratio'] = group['amount'] / group['amount'].rolling(20).mean()

            result_list.append(group)

        return pd.concat(result_list, ignore_index=True)

    def load_factors_from_file(
        self,
        factor_file: str,
        start_date: str,
        end_date: str,
        stocks: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        从因子文件加载预计算的因子

        Args:
            factor_file: 因子文件路径（parquet或csv）
            start_date: 开始日期
            end_date: 结束日期
            stocks: 股票代码列表

        Returns:
            DataFrame，长格式：ts_code, factor_date, factor_name, factor_value
        """
        factor_path = Path(factor_file)

        if not factor_path.exists():
            raise FileNotFoundError(f"因子文件不存在: {factor_path}")

        # 根据文件类型加载
        if factor_path.suffix == '.parquet':
            df = pd.read_parquet(factor_path)
        else:
            df = pd.read_csv(factor_path)

        df['factor_date'] = pd.to_datetime(df['factor_date'])

        # 筛选日期
        df = df[(df['factor_date'] >= start_date) & (df['factor_date'] <= end_date)]

        # 筛选股票
        if stocks:
            df = df[df['ts_code'].isin(stocks)]

        return df

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
        df = df.copy()
        factor_cols = [col for col in df.columns if col not in ['ts_code', 'factor_date', 'trade_date']]

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
        df = df.copy()
        date_col = 'factor_date' if 'factor_date' in df.columns else 'trade_date'
        factor_cols = [col for col in df.columns if col not in ['ts_code', 'factor_date', 'trade_date']]

        # 按日期分组标准化
        for date, group in df.groupby(date_col):
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
        factor_file: Optional[str] = None,
        use_basic_factors: bool = True,
        stocks: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        构建完整的训练数据集

        Args:
            start_date: 开始日期
            end_date: 结束日期
            forward_days: 预测未来N天收益率
            factor_file: 预计算因子文件路径（可选）
            use_basic_factors: 是否计算基本因子
            stocks: 股票列表

        Returns:
            (features_df, labels_df)
        """
        print(f"Building dataset from {start_date} to {end_date}...")

        if factor_file:
            # 从文件加载因子
            print(f"Loading factors from {factor_file}...")
            factors_df = self.load_factors_from_file(factor_file, start_date, end_date, stocks)
            factors_wide = self.pivot_factors(factors_df)
        elif use_basic_factors:
            # 计算基本因子
            print(f"Calculating basic factors...")
            stock_data = self.load_stock_features(start_date, end_date, stocks)
            factors_wide = self.calculate_basic_factors(stock_data)

            # 重命名日期列
            if 'trade_date' in factors_wide.columns:
                factors_wide = factors_wide.rename(columns={'trade_date': 'factor_date'})
        else:
            raise ValueError("必须指定factor_file或启用use_basic_factors")

        print(f"Loading prices...")
        prices_df = self.load_prices(start_date, end_date, stocks, use_hfq=True)

        print(f"Calculating forward returns (forward_days={forward_days})...")
        returns_df = self.calculate_returns(prices_df, forward_days)

        print(f"Preprocessing: winsorize and standardize...")
        # 提取因子列
        id_cols = ['ts_code', 'factor_date']
        factor_cols = [col for col in factors_wide.columns if col not in id_cols]

        factors_only = factors_wide[id_cols + factor_cols].copy()
        factors_only = self.winsorize(factors_only)
        factors_only = self.standardize(factors_only)

        print(f"Merging features and labels...")
        # 合并因子和收益率
        dataset = factors_only.merge(
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
        df = self._load_stock_data()
        return sorted(df['ts_code'].unique().tolist())

    def get_date_range(self) -> Tuple[datetime, datetime]:
        """获取数据的日期范围"""
        df = self._load_stock_data()
        return df['trade_date'].min(), df['trade_date'].max()


# 保持向后兼容的上下文管理器接口
class FactorDataLoaderCompat(FactorDataLoader):
    """兼容旧版本的数据加载器（支持with语句）"""

    def connect(self):
        """兼容接口：不需要连接"""
        pass

    def close(self):
        """兼容接口：不需要关闭"""
        pass

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
