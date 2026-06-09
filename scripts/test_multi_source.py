#!/usr/bin/env python
"""测试多数据源采集器"""
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.collectors.multi_source import MultiSourceCollector
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=== 测试多数据源采集器 ===")

    # 初始化
    collector = MultiSourceCollector()

    # 测试获取成分股
    logger.info("1. 测试获取科创50成分股...")
    try:
        components = collector.get_star50_components()
        logger.info(f"✓ 获取到 {len(components)} 只成分股")
        logger.info(f"前5只: {components[:5]}")
    except Exception as e:
        logger.error(f"✗ 获取成分股失败: {e}")
        return

    # 测试采集2只股票
    logger.info("\n2. 测试采集2只股票数据 (2024-01-01 ~ 2024-01-31)...")
    test_codes = components[:2]
    logger.info(f"测试股票: {test_codes}")

    try:
        df = collector.collect_daily_data(test_codes, '2024-01-01', '2024-01-31')
        logger.info(f"✓ 采集成功: {len(df)} 条记录")

        if not df.empty:
            logger.info(f"\n数据预览:")
            logger.info(f"\n{df.head()}")
    except Exception as e:
        logger.error(f"✗ 采集失败: {e}")

    # 输出统计
    logger.info("\n3. 统计信息:")
    stats = collector.get_stats()
    logger.info(f"数据源成功次数: {stats['sources']}")
    logger.info(f"熔断器状态: {stats['circuit_breakers']}")

    logger.info("\n=== 测试完成 ===")


if __name__ == '__main__':
    main()
