# scripts/collect_data.py
import sys
import argparse
from pathlib import Path
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config, load_env, setup_logger
from src.data.database import DatabaseManager
from src.data.collectors import StockDataCollector, IndexDataCollector
import pandas as pd


def save_to_database(df: pd.DataFrame, table_name: str, db_manager: DatabaseManager, logger):
    """
    保存数据到数据库

    Args:
        df: 数据DataFrame
        table_name: 表名
        db_manager: 数据库管理器
        logger: 日志对象
    """
    if df.empty:
        logger.warning(f"数据为空，跳过保存到 {table_name}")
        return

    try:
        with db_manager.get_session() as session:
            # 使用pandas to_sql保存
            df.to_sql(
                table_name,
                db_manager.engine,
                if_exists='append',
                index=False,
                method='multi',
                chunksize=1000
            )
        logger.info(f"成功保存 {len(df)} 条记录到 {table_name}")
    except Exception as e:
        logger.error(f"保存数据到 {table_name} 失败: {e}")
        raise


def collect_stock_data(args, logger):
    """采集股票数据"""
    logger.info("=== 开始采集股票数据 ===")

    # 加载配置
    load_env()
    data_config = load_config("configs/data_config.yaml")
    db_config = load_config("configs/db_config.yaml")

    # 初始化采集器
    stock_collector = StockDataCollector()

    # 获取科创50成分股
    logger.info("获取科创50成分股列表")
    ts_codes = stock_collector.get_star50_components()

    if not ts_codes:
        logger.error("获取成分股失败")
        return

    logger.info(f"共 {len(ts_codes)} 只成分股")

    # 获取日期范围
    start_date = args.start_date or data_config['date_range']['start']
    end_date = args.end_date or data_config['date_range']['end']

    # 采集数据
    logger.info(f"采集日期范围: {start_date} 至 {end_date}")
    df = stock_collector.collect_daily_data(ts_codes, start_date, end_date)

    if df.empty:
        logger.warning("未采集到任何股票数据")
        return

    # 保存到数据库
    logger.info("保存数据到数据库")
    db_manager = DatabaseManager(db_config)
    db_manager.connect()

    try:
        save_to_database(df, 'stock_daily', db_manager, logger)
        logger.info("=== 股票数据采集完成 ===")
    finally:
        db_manager.disconnect()


def collect_index_data(args, logger):
    """采集指数数据"""
    logger.info("=== 开始采集指数数据 ===")

    # 加载配置
    load_env()
    data_config = load_config("configs/data_config.yaml")
    db_config = load_config("configs/db_config.yaml")

    # 初始化采集器
    index_collector = IndexDataCollector()

    # 获取日期范围
    start_date = args.start_date or data_config['date_range']['start']
    end_date = args.end_date or data_config['date_range']['end']

    # 采集科创50指数数据
    logger.info(f"采集科创50指数数据: {start_date} 至 {end_date}")
    df = index_collector.collect_star50_index(start_date, end_date)

    if df.empty:
        logger.warning("未采集到指数数据")
        return

    # 保存到数据库
    logger.info("保存数据到数据库")
    db_manager = DatabaseManager(db_config)
    db_manager.connect()

    try:
        save_to_database(df, 'index_daily', db_manager, logger)
        logger.info("=== 指数数据采集完成 ===")
    finally:
        db_manager.disconnect()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='数据采集脚本')
    parser.add_argument('--type', choices=['stock', 'index', 'all'], default='all',
                       help='采集类型: stock(股票), index(指数), all(全部)')
    parser.add_argument('--start-date', type=str, help='开始日期 YYYY-MM-DD')
    parser.add_argument('--end-date', type=str, help='结束日期 YYYY-MM-DD')
    parser.add_argument('--log-level', default='INFO', help='日志级别')

    args = parser.parse_args()

    # 设置日志
    logger = setup_logger(log_level=args.log_level)

    try:
        if args.type in ['stock', 'all']:
            collect_stock_data(args, logger)

        if args.type in ['index', 'all']:
            collect_index_data(args, logger)

        logger.info("=== 所有数据采集完成 ===")

    except Exception as e:
        logger.error(f"数据采集失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
