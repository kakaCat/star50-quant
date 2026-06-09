"""
数据加载模块
提供从本地parquet文件加载股票和指数数据的功能
"""
from pathlib import Path
from typing import Optional, List
import pandas as pd
from loguru import logger


class DataLoader:
    """数据加载器"""

    def __init__(self, data_dir: str = "data/raw"):
        """
        初始化数据加载器

        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise ValueError(f"数据目录不存在: {self.data_dir}")

    def load_stock_data(
        self,
        filename: str = "star50_daily_hfq_data_6yrs.parquet",
        stock_codes: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        加载股票日线数据

        Args:
            filename: parquet文件名
            stock_codes: 股票代码列表，为None则加载全部
            start_date: 开始日期，格式'YYYY-MM-DD'
            end_date: 结束日期，格式'YYYY-MM-DD'

        Returns:
            股票数据DataFrame，包含以下字段：
            - ts_code: 股票代码
            - trade_date: 交易日期
            - open, high, low, close: 原始价格
            - pre_close: 前收盘价
            - change, pct_chg: 涨跌额和涨跌幅
            - vol, amount: 成交量和成交额
            - adj_factor: 复权因子
            - hfq_open, hfq_high, hfq_low, hfq_close: 后复权价格
        """
        file_path = self.data_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {file_path}")

        logger.info(f"加载股票数据: {file_path}")
        df = pd.read_parquet(file_path)

        # 筛选股票代码
        if stock_codes is not None:
            df = df[df['ts_code'].isin(stock_codes)]
            logger.info(f"筛选股票代码: {len(stock_codes)} 只")

        # 筛选日期范围
        if start_date is not None:
            df = df[df['trade_date'] >= pd.to_datetime(start_date)]
            logger.info(f"开始日期: {start_date}")

        if end_date is not None:
            df = df[df['trade_date'] <= pd.to_datetime(end_date)]
            logger.info(f"结束日期: {end_date}")

        logger.info(f"加载完成，数据形状: {df.shape}")
        return df

    def load_index_data(
        self,
        filename: str = "star50_index_daily_6yrs.parquet",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        加载指数日线数据

        Args:
            filename: parquet文件名
            start_date: 开始日期，格式'YYYY-MM-DD'
            end_date: 结束日期，格式'YYYY-MM-DD'

        Returns:
            指数数据DataFrame，包含以下字段：
            - ts_code: 指数代码
            - trade_date: 交易日期
            - open, high, low, close: OHLC价格
            - pre_close: 前收盘价
            - change, pct_chg: 涨跌额和涨跌幅
            - vol, amount: 成交量和成交额
        """
        file_path = self.data_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {file_path}")

        logger.info(f"加载指数数据: {file_path}")
        df = pd.read_parquet(file_path)

        # 筛选日期范围
        if start_date is not None:
            df = df[df['trade_date'] >= pd.to_datetime(start_date)]
            logger.info(f"开始日期: {start_date}")

        if end_date is not None:
            df = df[df['trade_date'] <= pd.to_datetime(end_date)]
            logger.info(f"结束日期: {end_date}")

        logger.info(f"加载完成，数据形状: {df.shape}")
        return df

    def get_stock_list(self, filename: str = "star50_daily_hfq_data_6yrs.parquet") -> List[str]:
        """
        获取数据中的股票代码列表

        Args:
            filename: parquet文件名

        Returns:
            股票代码列表
        """
        file_path = self.data_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {file_path}")

        df = pd.read_parquet(file_path, columns=['ts_code'])
        stock_codes = sorted(df['ts_code'].unique().tolist())
        logger.info(f"共有 {len(stock_codes)} 只股票")
        return stock_codes

    def get_date_range(self, filename: str = "star50_daily_hfq_data_6yrs.parquet") -> tuple:
        """
        获取数据的日期范围

        Args:
            filename: parquet文件名

        Returns:
            (开始日期, 结束日期) 元组
        """
        file_path = self.data_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {file_path}")

        df = pd.read_parquet(file_path, columns=['trade_date'])
        start_date = df['trade_date'].min()
        end_date = df['trade_date'].max()
        logger.info(f"日期范围: {start_date} 至 {end_date}")
        return start_date, end_date


def load_star50_data(
    stock_codes: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    include_index: bool = True
) -> dict:
    """
    便捷函数：加载Star50股票和指数数据

    Args:
        stock_codes: 股票代码列表，为None则加载全部
        start_date: 开始日期，格式'YYYY-MM-DD'
        end_date: 结束日期，格式'YYYY-MM-DD'
        include_index: 是否包含指数数据

    Returns:
        包含'stocks'和'index'键的字典
    """
    loader = DataLoader()

    result = {
        'stocks': loader.load_stock_data(
            stock_codes=stock_codes,
            start_date=start_date,
            end_date=end_date
        )
    }

    if include_index:
        result['index'] = loader.load_index_data(
            start_date=start_date,
            end_date=end_date
        )

    return result


if __name__ == "__main__":
    # 使用示例
    loader = DataLoader()

    # 获取股票列表
    stocks = loader.get_stock_list()
    print(f"股票列表: {stocks[:5]}... (共{len(stocks)}只)")

    # 获取日期范围
    date_range = loader.get_date_range()
    print(f"日期范围: {date_range}")

    # 加载所有数据
    df_stocks = loader.load_stock_data()
    print(f"\n股票数据: {df_stocks.shape}")
    print(df_stocks.head())

    # 加载指数数据
    df_index = loader.load_index_data()
    print(f"\n指数数据: {df_index.shape}")
    print(df_index.head())

    # 使用便捷函数
    data = load_star50_data(
        stock_codes=['688001.SH', '688008.SH'],
        start_date='2024-01-01',
        end_date='2024-12-31'
    )
    print(f"\n筛选后的股票数据: {data['stocks'].shape}")
    print(f"筛选后的指数数据: {data['index'].shape}")
