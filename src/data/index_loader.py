"""
科创50指数数据加载器
===================

加载科创50指数行情数据，用于计算超额收益标签和相对指标。
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
import psycopg2
from psycopg2.extras import execute_values


class IndexDataLoader:
    """
    科创50指数数据加载器
    """

    def __init__(self, db_name: str = 'star50_quant'):
        """
        初始化

        Args:
            db_name: 数据库名
        """
        self.db_name = db_name
        self.conn = None

    def connect(self):
        """连接数据库（使用与FactorDataLoader相同的方式）"""
        self.conn = psycopg2.connect(dbname=self.db_name)

    def close(self):
        """关闭连接"""
        if self.conn:
            self.conn.close()

    def load_index_daily(
        self,
        index_code: str = '000688.SH',
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        加载指数日行情数据

        Args:
            index_code: 指数代码（默认科创50: 000688.SH）
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            DataFrame with columns: trade_date, close, open, high, low, volume, amount
        """
        query = """
        SELECT
            trade_date,
            close,
            open,
            high,
            low,
            volume,
            amount
        FROM index_daily
        WHERE ts_code = %s
        """

        params = [index_code]

        if start_date:
            query += " AND trade_date >= %s"
            params.append(start_date)

        if end_date:
            query += " AND trade_date <= %s"
            params.append(end_date)

        query += " ORDER BY trade_date"

        df = pd.read_sql(query, self.conn, params=params)
        df['trade_date'] = pd.to_datetime(df['trade_date'])

        return df

    def calculate_index_returns(
        self,
        prices: pd.DataFrame,
        forward_days: int = 5
    ) -> pd.DataFrame:
        """
        计算指数收益率

        Args:
            prices: 指数价格数据
            forward_days: 前瞻天数

        Returns:
            DataFrame with columns: trade_date, forward_return
        """
        df = prices.copy()
        df = df.sort_values('trade_date')

        # 未来N日收益率
        df['forward_return'] = df['close'].shift(-forward_days) / df['close'] - 1

        # 当日收益率
        df['daily_return'] = df['close'].pct_change()

        return df[['trade_date', 'close', 'daily_return', 'forward_return']].dropna()

    def calculate_index_volatility(
        self,
        prices: pd.DataFrame,
        window: int = 20
    ) -> pd.DataFrame:
        """
        计算指数波动率

        Args:
            prices: 指数价格数据
            window: 滚动窗口

        Returns:
            DataFrame with columns: trade_date, volatility
        """
        df = prices.copy()
        df = df.sort_values('trade_date')

        # 日收益率
        df['return'] = df['close'].pct_change()

        # 滚动波动率
        df['volatility'] = df['return'].rolling(window).std() * np.sqrt(252)

        return df[['trade_date', 'volatility']].dropna()

    def create_index_table_if_not_exists(self):
        """
        创建指数行情表（如果不存在）
        """
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS index_daily (
            id SERIAL PRIMARY KEY,
            ts_code VARCHAR(20) NOT NULL,
            trade_date DATE NOT NULL,
            close DECIMAL(10, 2),
            open DECIMAL(10, 2),
            high DECIMAL(10, 2),
            low DECIMAL(10, 2),
            volume BIGINT,
            amount DECIMAL(20, 2),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(ts_code, trade_date)
        );

        CREATE INDEX IF NOT EXISTS idx_index_daily_code_date
        ON index_daily(ts_code, trade_date);
        """

        cursor = self.conn.cursor()
        cursor.execute(create_table_sql)
        self.conn.commit()
        cursor.close()

        print("✓ Index daily table created/verified")

    def insert_index_data(self, data: pd.DataFrame):
        """
        插入指数数据

        Args:
            data: 包含ts_code, trade_date, close等列的DataFrame
        """
        if len(data) == 0:
            return

        cursor = self.conn.cursor()

        # 准备数据
        values = [
            (
                row['ts_code'],
                row['trade_date'].strftime('%Y-%m-%d') if isinstance(row['trade_date'], pd.Timestamp) else row['trade_date'],
                float(row['close']),
                float(row.get('open', row['close'])),
                float(row.get('high', row['close'])),
                float(row.get('low', row['close'])),
                int(row.get('volume', 0)),
                float(row.get('amount', 0))
            )
            for _, row in data.iterrows()
        ]

        # 批量插入（忽略重复）
        insert_sql = """
        INSERT INTO index_daily
        (ts_code, trade_date, close, open, high, low, volume, amount)
        VALUES %s
        ON CONFLICT (ts_code, trade_date) DO NOTHING
        """

        execute_values(cursor, insert_sql, values)
        self.conn.commit()
        cursor.close()

        print(f"✓ Inserted {len(data)} index records")

    def collect_star50_index_from_akshare(
        self,
        start_date: str = '2019-01-01',
        end_date: str = '2024-12-31'
    ):
        """
        从AkShare采集科创50指数数据

        Args:
            start_date: 开始日期
            end_date: 结束日期
        """
        try:
            import akshare as ak
        except ImportError:
            print("❌ AkShare not installed. Install with: pip install akshare")
            return

        print(f"Collecting STAR 50 Index data from {start_date} to {end_date}...")

        try:
            # 获取科创50指数数据
            # 使用stock_zh_index_daily接口
            df = ak.stock_zh_index_daily(symbol="sh000688")

            # 重命名列
            df.columns = ['date', 'open', 'close', 'high', 'low', 'volume', 'amount']

            # 过滤日期
            df['date'] = pd.to_datetime(df['date'])
            df = df[
                (df['date'] >= start_date) &
                (df['date'] <= end_date)
            ]

            # 添加ts_code
            df['ts_code'] = '000688.SH'
            df = df.rename(columns={'date': 'trade_date'})

            print(f"✓ Collected {len(df)} records")

            # 插入数据库
            self.create_index_table_if_not_exists()
            self.insert_index_data(df)

            return df

        except Exception as e:
            print(f"❌ Error collecting index data: {e}")
            return None


def test_index_loader():
    """测试指数数据加载器"""
    print("="*70)
    print("测试科创50指数数据加载器")
    print("="*70)
    print()

    loader = IndexDataLoader(db_name='star50_quant')
    loader.connect()

    # 1. 采集数据（如果需要）
    print("1. 采集科创50指数数据...")
    loader.collect_star50_index_from_akshare('2019-01-01', '2024-12-31')
    print()

    # 2. 加载数据
    print("2. 加载指数数据...")
    index_data = loader.load_index_daily('000688.SH', '2023-01-01', '2024-12-31')
    print(f"✓ 加载了 {len(index_data)} 条记录")
    print(index_data.head())
    print()

    # 3. 计算收益率
    print("3. 计算指数收益率...")
    index_returns = loader.calculate_index_returns(index_data, forward_days=5)
    print(f"✓ 计算了 {len(index_returns)} 个收益率")
    print(index_returns.head())
    print()

    # 4. 计算波动率
    print("4. 计算指数波动率...")
    index_vol = loader.calculate_index_volatility(index_data, window=20)
    print(f"✓ 计算了 {len(index_vol)} 个波动率")
    print(index_vol.head())
    print()

    loader.close()

    print("="*70)
    print("测试完成")
    print("="*70)


if __name__ == '__main__':
    test_index_loader()
