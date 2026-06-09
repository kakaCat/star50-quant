"""
快速对比：PostgreSQL vs Parquet
测试两种数据加载方式的性能和易用性
"""
import time
from src.models.data_loader_parquet import FactorDataLoader

def test_parquet_loader():
    """测试Parquet加载器"""
    print("=" * 80)
    print("测试 Parquet 数据加载器")
    print("=" * 80)

    start_time = time.time()

    # 初始化
    loader = FactorDataLoader(data_dir='data/raw')
    print(f"✓ 初始化完成（无需数据库连接）")

    # 获取股票列表
    stocks = loader.get_stock_list()
    print(f"✓ 股票列表: {len(stocks)} 只")

    # 获取日期范围
    start_date, end_date = loader.get_date_range()
    print(f"✓ 日期范围: {start_date.date()} 至 {end_date.date()}")

    # 加载价格数据
    prices = loader.load_prices('2024-01-01', '2024-12-31', stocks=stocks[:10])
    print(f"✓ 加载价格: {len(prices)} 条记录")

    # 构建训练数据集
    features, labels = loader.build_dataset(
        start_date='2024-01-01',
        end_date='2024-12-31',
        forward_days=5,
        use_basic_factors=True
    )
    print(f"✓ 构建数据集: {features.shape[0]} 样本, {features.shape[1]-2} 特征")

    elapsed = time.time() - start_time
    print(f"\n总耗时: {elapsed:.2f} 秒")

    return features, labels


def show_comparison():
    """显示对比"""
    print("\n" + "=" * 80)
    print("PostgreSQL vs Parquet 对比")
    print("=" * 80)

    comparison = """
┌─────────────────┬──────────────────────────┬──────────────────────────┐
│     特性        │      PostgreSQL          │         Parquet          │
├─────────────────┼──────────────────────────┼──────────────────────────┤
│ 环境配置        │ 需要安装和配置数据库      │ 只需要parquet文件         │
│ 依赖            │ psycopg2-binary          │ pyarrow                  │
│ 启动时间        │ 需要连接数据库（~100ms）  │ 即开即用（<10ms）         │
│ 数据加载        │ SQL查询（网络开销）       │ 本地文件读取（更快）      │
│ 数据更新        │ INSERT/UPDATE            │ 替换文件                 │
│ 部署难度        │ 需要部署数据库服务        │ 复制文件即可             │
│ 版本控制        │ 需要数据库迁移脚本        │ 可用git-lfs管理          │
│ 数据分享        │ 需要导出/导入             │ 直接分享文件             │
│ 云端支持        │ 需要RDS等服务             │ S3/OSS对象存储           │
│ 适用场景        │ 生产环境，高并发写入      │ 数据分析，机器学习       │
└─────────────────┴──────────────────────────┴──────────────────────────┘
    """
    print(comparison)

    print("\n推荐使用 Parquet 的场景:")
    print("  ✓ 量化研究和回测")
    print("  ✓ 机器学习模型训练")
    print("  ✓ 本地开发和实验")
    print("  ✓ 数据分析和可视化")
    print("  ✓ 快速原型开发")

    print("\n仍需 PostgreSQL 的场景:")
    print("  ✓ 实时数据写入")
    print("  ✓ 多用户并发访问")
    print("  ✓ 复杂的关联查询")
    print("  ✓ 事务性操作")


def show_usage_examples():
    """显示使用示例"""
    print("\n" + "=" * 80)
    print("使用示例")
    print("=" * 80)

    examples = """
# 1. 基本数据加载
from src.models.data_loader_parquet import FactorDataLoader

loader = FactorDataLoader()
prices = loader.load_prices('2024-01-01', '2024-12-31')

# 2. 构建训练数据集
features, labels = loader.build_dataset(
    start_date='2024-01-01',
    end_date='2024-12-31',
    forward_days=5,
    use_basic_factors=True
)

# 3. 使用预计算因子
features, labels = loader.build_dataset(
    start_date='2024-01-01',
    end_date='2024-12-31',
    factor_file='data/processed/factors.parquet'
)

# 4. 兼容旧代码（使用with语句）
from src.models.data_loader_parquet import FactorDataLoaderCompat

with FactorDataLoaderCompat() as loader:
    features, labels = loader.build_dataset('2024-01-01', '2024-12-31')
    """
    print(examples)


if __name__ == '__main__':
    # 运行测试
    features, labels = test_parquet_loader()

    # 显示对比
    show_comparison()

    # 显示使用示例
    show_usage_examples()

    print("\n" + "=" * 80)
    print("迁移完成！现在可以使用Parquet文件进行数据加载了。")
    print("详细迁移指南: docs/MIGRATION_TO_PARQUET.md")
    print("=" * 80)
