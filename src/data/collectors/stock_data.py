# src/data/collectors/stock_data.py
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional
from loguru import logger
import akshare as ak


class StockDataCollector:
    """股票数据采集器（使用akshare，免费无需token）"""

    def __init__(self):
        self.source = "akshare"
        logger.info(f"初始化股票数据采集器: {self.source}")

    def collect_daily_data(
        self,
        ts_codes: List[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        采集股票日频数据

        Args:
            ts_codes: 股票代码列表，如 ['688001', '688002']
            start_date: 开始日期 YYYY-MM-DD
            end_date: 结束日期 YYYY-MM-DD

        Returns:
            包含所有股票数据的DataFrame
        """
        logger.info(f"开始采集 {len(ts_codes)} 只股票的日频数据")
        logger.info(f"日期范围: {start_date} 至 {end_date}")

        all_data = []

        for i, ts_code in enumerate(ts_codes, 1):
            try:
                # 转换为akshare格式（去掉.SH/.SZ后缀）
                symbol = ts_code.split('.')[0]

                # 获取股票历史数据
                df = ak.stock_zh_a_hist(
                    symbol=symbol,
                    period="daily",
                    start_date=start_date.replace('-', ''),
                    end_date=end_date.replace('-', ''),
                    adjust="qfq"  # 前复权
                )

                if df.empty:
                    logger.warning(f"[{i}/{len(ts_codes)}] {ts_code}: 无数据")
                    continue

                # 重命名列
                df = df.rename(columns={
                    '日期': 'trade_date',
                    '开盘': 'open',
                    '最高': 'high',
                    '最低': 'low',
                    '收盘': 'close',
                    '成交量': 'volume',
                    '成交额': 'amount',
                    '换手率': 'turnover_rate',
                    '振幅': 'amplitude',
                    '涨跌幅': 'pct_change'
                })

                # 添加股票代码
                df['ts_code'] = ts_code

                # 选择需要的列
                columns = [
                    'ts_code', 'trade_date', 'open', 'high', 'low', 'close',
                    'volume', 'amount', 'turnover_rate', 'amplitude', 'pct_change'
                ]
                df = df[columns]

                all_data.append(df)

                logger.info(f"[{i}/{len(ts_codes)}] {ts_code}: 采集 {len(df)} 条记录")

            except Exception as e:
                logger.error(f"[{i}/{len(ts_codes)}] {ts_code}: 采集失败 - {e}")
                continue

        if not all_data:
            logger.warning("未采集到任何数据")
            return pd.DataFrame()

        # 合并所有数据
        result = pd.concat(all_data, ignore_index=True)
        logger.info(f"采集完成，共 {len(result)} 条记录")

        return result

    def get_star50_components(self) -> List[str]:
        """
        获取科创50成分股列表

        Returns:
            股票代码列表
        """
        try:
            # 获取科创50成分股
            df = ak.index_stock_cons(symbol="000688")

            # 提取股票代码并添加交易所后缀
            ts_codes = [f"{code}.SH" for code in df['品种代码'].tolist()]

            logger.info(f"获取科创50成分股: {len(ts_codes)} 只")
            return ts_codes

        except Exception as e:
            logger.error(f"获取科创50成分股失败: {e}")
            return []
