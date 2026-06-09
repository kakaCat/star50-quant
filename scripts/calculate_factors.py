#!/usr/bin/env python3
"""
因子计算脚本
============

从数据库读取股票K线数据，计算技术因子并保存到数据库。

用法:
    python scripts/calculate_factors.py --stock 688009.SH --start 2024-01-01 --end 2024-12-31
    python scripts/calculate_factors.py --all  # 计算所有股票
"""

import sys
import os
import argparse
import logging
from datetime import datetime, date
from typing import List, Dict, Any

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import psycopg2
from psycopg2.extras import execute_values

from src.features.momentum import MomentumFactors
from src.features.volume import VolumeFactors
from src.features.trend import TrendFactors

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FactorCalculator:
    """因子计算管理器"""

    def __init__(self):
        """初始化因子计算器"""
        self.momentum_factors = MomentumFactors(precision=4)
        self.volume_factors = VolumeFactors(precision=4)
        self.trend_factors = TrendFactors(precision=4)

        # 定义要计算的因子列表
        self.factor_methods = {
            # 动量因子
            'macd': self.momentum_factors.macd,
            'macd_signal': self.momentum_factors.macd_signal,
            'macd_histogram': self.momentum_factors.macd_histogram,
            'rsi6': self.momentum_factors.rsi6,
            'rsi12': self.momentum_factors.rsi12,
            'rsi24': self.momentum_factors.rsi24,
            'roc_5': self.momentum_factors.roc_5,
            'roc_10': self.momentum_factors.roc_10,
            'roc_20': self.momentum_factors.roc_20,
            'momentum_5': self.momentum_factors.momentum_5,
            'momentum_10': self.momentum_factors.momentum_10,
            'momentum_20': self.momentum_factors.momentum_20,

            # 量价因子
            'obv': self.volume_factors.obv,
            'mfi14': self.volume_factors.mfi14,
            'vwap': self.volume_factors.vwap,
            'volume_ma5': self.volume_factors.volume_ma5,
            'volume_ma10': self.volume_factors.volume_ma10,
            'volume_ma20': self.volume_factors.volume_ma20,
            'volume_ratio': self.volume_factors.volume_ratio,

            # 趋势因子
            'ma5': self.trend_factors.ma5,
            'ma10': self.trend_factors.ma10,
            'ma20': self.trend_factors.ma20,
            'ma60': self.trend_factors.ma60,
            'ema5': self.trend_factors.ema5,
            'ema10': self.trend_factors.ema10,
            'ema20': self.trend_factors.ema20,
            'boll_upper': self.trend_factors.boll_upper,
            'boll_middle': self.trend_factors.boll_middle,
            'boll_lower': self.trend_factors.boll_lower,
            'atr14': self.trend_factors.atr14,
        }

    def calculate_factors(self, klines: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        计算所有因子

        Args:
            klines: K线数据列表

        Returns:
            因子字典 {factor_name: factor_value}
        """
        factors = {}

        for factor_name, factor_method in self.factor_methods.items():
            try:
                result = factor_method(klines)
                factors[factor_name] = result['value']
            except Exception as e:
                logger.warning(f"Failed to calculate {factor_name}: {e}")
                factors[factor_name] = None

        return factors


class DatabaseManager:
    """数据库管理器"""

    def __init__(self):
        """初始化数据库连接"""
        user = os.getenv('USER', 'mac')
        self.conn = psycopg2.connect(
            dbname='star50_quant',
            user=user,
            host='localhost'
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.close()

    def get_stock_list(self) -> List[str]:
        """获取所有股票代码"""
        with self.conn.cursor() as cur:
            cur.execute("SELECT DISTINCT ts_code FROM stock_daily ORDER BY ts_code")
            stocks = [row[0] for row in cur.fetchall()]
        return stocks

    def get_klines(self, ts_code: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """
        获取K线数据

        Args:
            ts_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            K线数据列表
        """
        query = """
            SELECT trade_date, open, high, low, close, volume, amount
            FROM stock_daily
            WHERE ts_code = %s
              AND trade_date >= %s
              AND trade_date <= %s
            ORDER BY trade_date ASC
        """

        with self.conn.cursor() as cur:
            cur.execute(query, (ts_code, start_date, end_date))
            rows = cur.fetchall()

        klines = []
        for row in rows:
            klines.append({
                'date': row[0],
                'open': float(row[1]),
                'high': float(row[2]),
                'low': float(row[3]),
                'close': float(row[4]),
                'volume': float(row[5]),
                'amount': float(row[6]) if row[6] else 0.0
            })

        return klines

    def save_factors(self, ts_code: str, factor_date: date, factors: Dict[str, float]):
        """
        保存因子到数据库

        Args:
            ts_code: 股票代码
            factor_date: 因子日期
            factors: 因子字典
        """
        # 准备插入数据
        values = []
        for factor_name, factor_value in factors.items():
            if factor_value is not None:
                values.append((ts_code, factor_date, factor_name, factor_value))

        if not values:
            return

        # 使用 UPSERT
        query = """
            INSERT INTO factor_values (ts_code, factor_date, factor_name, factor_value)
            VALUES %s
            ON CONFLICT (ts_code, factor_date, factor_name)
            DO UPDATE SET factor_value = EXCLUDED.factor_value
        """

        with self.conn.cursor() as cur:
            execute_values(cur, query, values)
            self.conn.commit()

    def create_factor_table(self):
        """创建因子表（如果不存在）"""
        query = """
        CREATE TABLE IF NOT EXISTS factor_values (
            ts_code VARCHAR(20) NOT NULL,
            factor_date DATE NOT NULL,
            factor_name VARCHAR(50) NOT NULL,
            factor_value DOUBLE PRECISION,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (ts_code, factor_date, factor_name)
        );

        CREATE INDEX IF NOT EXISTS idx_factor_values_date ON factor_values(factor_date);
        CREATE INDEX IF NOT EXISTS idx_factor_values_name ON factor_values(factor_name);
        """

        with self.conn.cursor() as cur:
            cur.execute(query)
            self.conn.commit()

        logger.info("Factor table created/verified")


def calculate_stock_factors(
    ts_code: str,
    start_date: str,
    end_date: str,
    window_size: int = 60
):
    """
    计算单只股票的因子

    Args:
        ts_code: 股票代码
        start_date: 开始日期
        end_date: 结束日期
        window_size: 滚动窗口大小（天）
    """
    logger.info(f"Calculating factors for {ts_code} from {start_date} to {end_date}")

    calculator = FactorCalculator()

    with DatabaseManager() as db:
        # 获取K线数据
        klines = db.get_klines(ts_code, start_date, end_date)

        if len(klines) < window_size:
            logger.warning(f"Insufficient data for {ts_code}: {len(klines)} < {window_size}")
            return

        logger.info(f"Loaded {len(klines)} K-lines for {ts_code}")

        # 滚动计算因子
        factor_count = 0
        for i in range(window_size - 1, len(klines)):
            window_data = klines[max(0, i - window_size + 1):i + 1]
            current_date = klines[i]['date']

            try:
                # 计算因子
                factors = calculator.calculate_factors(window_data)

                # 保存到数据库
                db.save_factors(ts_code, current_date, factors)
                factor_count += 1

                if factor_count % 100 == 0:
                    logger.info(f"  Processed {factor_count}/{len(klines) - window_size + 1} days")

            except Exception as e:
                logger.error(f"Error calculating factors for {ts_code} on {current_date}: {e}")

        logger.info(f"✓ Completed {ts_code}: {factor_count} days processed")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='计算技术因子')
    parser.add_argument('--stock', type=str, help='股票代码（如 688009.SH）')
    parser.add_argument('--all', action='store_true', help='计算所有股票')
    parser.add_argument('--start', type=str, default='2020-01-01', help='开始日期')
    parser.add_argument('--end', type=str, default='2024-12-31', help='结束日期')
    parser.add_argument('--window', type=int, default=60, help='滚动窗口大小（天）')

    args = parser.parse_args()

    # 创建因子表
    with DatabaseManager() as db:
        db.create_factor_table()

    if args.all:
        # 计算所有股票
        with DatabaseManager() as db:
            stocks = db.get_stock_list()

        logger.info(f"Found {len(stocks)} stocks to process")

        for i, ts_code in enumerate(stocks, 1):
            logger.info(f"\n[{i}/{len(stocks)}] Processing {ts_code}")
            try:
                calculate_stock_factors(ts_code, args.start, args.end, args.window)
            except Exception as e:
                logger.error(f"Failed to process {ts_code}: {e}")

        logger.info(f"\n✓ All done! Processed {len(stocks)} stocks")

    elif args.stock:
        # 计算单只股票
        calculate_stock_factors(args.stock, args.start, args.end, args.window)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
