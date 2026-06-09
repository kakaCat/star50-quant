"""
数据加载使用示例
展示如何使用DataLoader加载和处理Star50数据
"""
import sys
from pathlib import Path

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.loaders import DataLoader, load_star50_data
import pandas as pd


def example_1_basic_loading():
    """示例1: 基本数据加载"""
    print("=" * 80)
    print("示例1: 基本数据加载")
    print("=" * 80)

    loader = DataLoader()

    # 加载全部股票数据
    df_stocks = loader.load_stock_data()
    print(f"\n股票数据形状: {df_stocks.shape}")
    print(f"股票数量: {df_stocks['ts_code'].nunique()}")
    print(f"\n前5行数据:")
    print(df_stocks.head())

    # 加载指数数据
    df_index = loader.load_index_data()
    print(f"\n指数数据形状: {df_index.shape}")
    print(f"\n前5行数据:")
    print(df_index.head())


def example_2_filtered_loading():
    """示例2: 筛选特定股票和日期"""
    print("\n" + "=" * 80)
    print("示例2: 筛选特定股票和日期")
    print("=" * 80)

    loader = DataLoader()

    # 加载特定股票和日期范围
    stock_codes = ['688001.SH', '688008.SH', '688012.SH']
    df = loader.load_stock_data(
        stock_codes=stock_codes,
        start_date='2024-01-01',
        end_date='2024-12-31'
    )

    print(f"\n数据形状: {df.shape}")
    print(f"股票: {df['ts_code'].unique()}")
    print(f"日期范围: {df['trade_date'].min()} 至 {df['trade_date'].max()}")
    print(f"\n示例数据:")
    print(df.head())


def example_3_convenient_function():
    """示例3: 使用便捷函数"""
    print("\n" + "=" * 80)
    print("示例3: 使用便捷函数")
    print("=" * 80)

    # 使用便捷函数同时加载股票和指数数据
    data = load_star50_data(
        stock_codes=['688001.SH', '688008.SH'],
        start_date='2024-01-01',
        include_index=True
    )

    print(f"\n股票数据: {data['stocks'].shape}")
    print(f"指数数据: {data['index'].shape}")


def example_4_data_analysis():
    """示例4: 基本数据分析"""
    print("\n" + "=" * 80)
    print("示例4: 基本数据分析")
    print("=" * 80)

    loader = DataLoader()

    # 获取股票列表
    stock_codes = loader.get_stock_list()
    print(f"\n共有 {len(stock_codes)} 只股票")
    print(f"前10只: {stock_codes[:10]}")

    # 获取日期范围
    start_date, end_date = loader.get_date_range()
    print(f"\n日期范围: {start_date} 至 {end_date}")

    # 加载数据并计算统计信息
    df = loader.load_stock_data()

    print(f"\n数据统计:")
    print(f"总记录数: {len(df):,}")
    print(f"平均日涨跌幅: {df['pct_chg'].mean():.4f}%")
    print(f"涨跌幅标准差: {df['pct_chg'].std():.4f}%")

    # 计算每只股票的统计
    stock_stats = df.groupby('ts_code').agg({
        'pct_chg': ['mean', 'std'],
        'vol': 'mean',
        'trade_date': 'count'
    }).round(4)
    stock_stats.columns = ['平均涨跌幅(%)', '涨跌幅标准差(%)', '平均成交量', '交易天数']

    print(f"\n各股票统计信息 (前10只):")
    print(stock_stats.head(10))


def example_5_hfq_price_usage():
    """示例5: 使用后复权价格"""
    print("\n" + "=" * 80)
    print("示例5: 使用后复权价格进行回测")
    print("=" * 80)

    loader = DataLoader()

    # 加载单只股票
    df = loader.load_stock_data(
        stock_codes=['688001.SH'],
        start_date='2024-01-01'
    )

    # 使用后复权价格计算收益
    df = df.sort_values('trade_date')
    df['hfq_return'] = df['hfq_close'].pct_change() * 100

    print(f"\n股票: 688001.SH")
    print(f"期间: {df['trade_date'].min()} 至 {df['trade_date'].max()}")
    print(f"\n后复权价格走势:")
    print(df[['trade_date', 'hfq_close', 'hfq_return']].tail(10))

    # 计算累计收益
    total_return = (df['hfq_close'].iloc[-1] / df['hfq_close'].iloc[0] - 1) * 100
    print(f"\n累计收益率: {total_return:.2f}%")


def example_6_align_stock_and_index():
    """示例6: 对齐股票和指数数据"""
    print("\n" + "=" * 80)
    print("示例6: 对齐股票和指数数据")
    print("=" * 80)

    data = load_star50_data(
        stock_codes=['688001.SH'],
        start_date='2024-01-01',
        include_index=True
    )

    # 合并股票和指数数据
    df_stock = data['stocks'][['trade_date', 'ts_code', 'hfq_close', 'pct_chg']].copy()
    df_stock.columns = ['trade_date', 'ts_code', 'stock_close', 'stock_pct_chg']

    df_index = data['index'][['trade_date', 'close', 'pct_chg']].copy()
    df_index.columns = ['trade_date', 'index_close', 'index_pct_chg']

    # 按日期合并
    df_merged = pd.merge(df_stock, df_index, on='trade_date', how='inner')

    print(f"\n对齐后的数据:")
    print(df_merged.head(10))

    # 计算相对表现
    df_merged['relative_return'] = df_merged['stock_pct_chg'] - df_merged['index_pct_chg']

    print(f"\n相对指数的超额收益:")
    print(f"平均: {df_merged['relative_return'].mean():.4f}%")
    print(f"累计: {df_merged['relative_return'].sum():.4f}%")


if __name__ == "__main__":
    # 运行所有示例
    example_1_basic_loading()
    example_2_filtered_loading()
    example_3_convenient_function()
    example_4_data_analysis()
    example_5_hfq_price_usage()
    example_6_align_stock_and_index()

    print("\n" + "=" * 80)
    print("所有示例运行完成!")
    print("=" * 80)
