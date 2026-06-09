# src/data/collectors/index_data.py
import pandas as pd
from typing import Optional
from loguru import logger
import akshare as ak


class IndexDataCollector:
    """指数数据采集器"""

    def __init__(self):
        self.source = "akshare"
        logger.info(f"初始化指数数据采集器: {self.source}")

    def collect_index_daily(
        self,
        index_code: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        采集指数日频数据

        Args:
            index_code: 指数代码，如 '000688' (科创50)
            start_date: 开始日期 YYYY-MM-DD
            end_date: 结束日期 YYYY-MM-DD

        Returns:
            指数数据DataFrame
        """
        logger.info(f"开始采集指数数据: {index_code}")
        logger.info(f"日期范围: {start_date} 至 {end_date}")

        try:
            # 获取指数历史数据
            df = ak.stock_zh_index_daily(symbol=f"sh{index_code}")

            if df.empty:
                logger.warning(f"指数 {index_code}: 无数据")
                return pd.DataFrame()

            # 筛选日期范围
            df['date'] = pd.to_datetime(df['date'])
            df = df[
                (df['date'] >= start_date) &
                (df['date'] <= end_date)
            ]

            # 重命名列
            df = df.rename(columns={
                'date': 'trade_date',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume',
                'amount': 'amount'
            })

            # 添加指数代码
            df['ts_code'] = f"{index_code}.SH"

            # 计算涨跌幅
            df['pct_change'] = df['close'].pct_change() * 100

            # 选择需要的列
            columns = [
                'ts_code', 'trade_date', 'open', 'high', 'low', 'close',
                'volume', 'amount', 'pct_change'
            ]
            df = df[columns]

            logger.info(f"指数 {index_code}: 采集 {len(df)} 条记录")
            return df

        except Exception as e:
            logger.error(f"指数 {index_code}: 采集失败 - {e}")
            return pd.DataFrame()

    def collect_star50_index(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        采集科创50指数数据

        Args:
            start_date: 开始日期 YYYY-MM-DD
            end_date: 结束日期 YYYY-MM-DD

        Returns:
            科创50指数数据DataFrame
        """
        return self.collect_index_daily("000688", start_date, end_date)
