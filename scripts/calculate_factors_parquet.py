#!/usr/bin/env python3
"""
因子计算脚本（基于Parquet文件）
================================

从parquet文件读取股票K线数据，计算技术因子并保存到文件。

用法:
    python scripts/calculate_factors_parquet.py --stock 688009.SH --start 2024-01-01 --end 2024-12-31
    python scripts/calculate_factors_parquet.py --all  # 计算所有股票
    python scripts/calculate_factors_parquet.py --all --output data/processed/factors.parquet
"""

import sys
import os
import argparse
import logging
from datetime import datetime, date
from typing import List, Dict, Any
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np

from src.features.momentum import MomentumFactors
from src.features.volume import VolumeFactors
from src.features.trend import TrendFactors
from src.data.loaders import DataLoader

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


class ParquetFactorManager:
    """基于Parquet文件的因子管理器"""

    def __init__(self, data_dir: str = 'data/raw'):
        """初始化"""
        self.loader = DataLoader(data_dir=data_dir)
        self.stock_data = None

    def load_data(self, start_date: str, end_date: str, stock_codes: List[str] = None):
        """加载股票数据"""
        logger.info(f"Loading stock data from {start_date} to {end_date}...")
        self.stock_data = self.loader.load_stock_data(
            stock_codes=stock_codes,
            start_date=start_date,
            end_date=end_date
        )
        logger.info(f"Loaded {len(self.stock_data)} records")

    def get_stock_list(self) -> List[str]:
        """获取所有股票代码"""
        return self.loader.get_stock_list()

    def get_klines(self, ts_code: str) -> List[Dict[str, Any]]:
        """
        获取单只股票的K线数据

        Args:
            ts_code: 股票代码

        Returns:
            K线数据列表
        """
        df = self.stock_data[self.stock_data['ts_code'] == ts_code].copy()
        df = df.sort_values('trade_date')

        klines = []
        for _, row in df.iterrows():
            klines.append({
                'date': row['trade_date'].date(),
                'open': float(row['hfq_open']),
                'high': float(row['hfq_high']),
                'low': float(row['hfq_low']),
                'close': float(row['hfq_close']),
                'volume': float(row['vol']),
                'amount': float(row['amount']) if pd.notna(row['amount']) else 0.0
            })

        return klines

    def save_factors(self, factor_data: List[Dict], output_file: str):
        """
        保存因子到文件

        Args:
            factor_data: 因子数据列表 [{ts_code, factor_date, factor_name, factor_value}]
            output_file: 输出文件路径
        """
        df = pd.DataFrame(factor_data)

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix == '.parquet':
            df.to_parquet(output_path, index=False)
        elif output_path.suffix == '.csv':
            df.to_csv(output_path, index=False)
        else:
            raise ValueError(f"Unsupported file format: {output_path.suffix}")

        logger.info(f"Saved {len(df)} factor records to {output_path}")


def calculate_stock_factors(
    ts_code: str,
    manager: ParquetFactorManager,
    window_size: int = 60
) -> List[Dict]:
    """
    计算单只股票的因子

    Args:
        ts_code: 股票代码
        manager: Parquet因子管理器
        window_size: 滚动窗口大小（天）

    Returns:
        因子数据列表
    """
    logger.info(f"Calculating factors for {ts_code}")

    calculator = FactorCalculator()

    # 获取K线数据
    klines = manager.get_klines(ts_code)

    if len(klines) < window_size:
        logger.warning(f"Insufficient data for {ts_code}: {len(klines)} < {window_size}")
        return []

    logger.info(f"Loaded {len(klines)} K-lines for {ts_code}")

    # 滚动计算因子
    factor_data = []
    factor_count = 0

    for i in range(window_size - 1, len(klines)):
        window_data = klines[max(0, i - window_size + 1):i + 1]
        current_date = klines[i]['date']

        try:
            # 计算因子
            factors = calculator.calculate_factors(window_data)

            # 转换为长格式
            for factor_name, factor_value in factors.items():
                if factor_value is not None:
                    factor_data.append({
                        'ts_code': ts_code,
                        'factor_date': current_date,
                        'factor_name': factor_name,
                        'factor_value': factor_value
                    })

            factor_count += 1

            if factor_count % 100 == 0:
                logger.info(f"  Processed {factor_count}/{len(klines) - window_size + 1} days")

        except Exception as e:
            logger.error(f"Error calculating factors for {ts_code} on {current_date}: {e}")

    logger.info(f"✓ Completed {ts_code}: {factor_count} days processed")

    return factor_data


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='计算技术因子（基于Parquet文件）')
    parser.add_argument('--stock', type=str, help='股票代码（如 688009.SH）')
    parser.add_argument('--all', action='store_true', help='计算所有股票')
    parser.add_argument('--start', type=str, default='2020-01-01', help='开始日期')
    parser.add_argument('--end', type=str, default='2025-12-31', help='结束日期')
    parser.add_argument('--window', type=int, default=60, help='滚动窗口大小（天）')
    parser.add_argument('--output', type=str, default='data/processed/factors.parquet',
                       help='输出文件路径')
    parser.add_argument('--data-dir', type=str, default='data/raw',
                       help='数据目录')

    args = parser.parse_args()

    # 初始化管理器
    manager = ParquetFactorManager(data_dir=args.data_dir)

    all_factor_data = []

    if args.all:
        # 获取所有股票
        stocks = manager.get_stock_list()
        logger.info(f"Found {len(stocks)} stocks to process")

        # 加载所有数据
        manager.load_data(args.start, args.end)

        # 计算所有股票
        for i, ts_code in enumerate(stocks, 1):
            logger.info(f"\n[{i}/{len(stocks)}] Processing {ts_code}")
            try:
                factor_data = calculate_stock_factors(ts_code, manager, args.window)
                all_factor_data.extend(factor_data)
            except Exception as e:
                logger.error(f"Failed to process {ts_code}: {e}")

        logger.info(f"\n✓ All done! Processed {len(stocks)} stocks")

    elif args.stock:
        # 计算单只股票
        manager.load_data(args.start, args.end, stock_codes=[args.stock])
        factor_data = calculate_stock_factors(args.stock, manager, args.window)
        all_factor_data.extend(factor_data)

    else:
        parser.print_help()
        return

    # 保存结果
    if all_factor_data:
        manager.save_factors(all_factor_data, args.output)
        logger.info(f"\n✓ Saved {len(all_factor_data)} factor records to {args.output}")
    else:
        logger.warning("No factor data to save")


if __name__ == '__main__':
    main()
