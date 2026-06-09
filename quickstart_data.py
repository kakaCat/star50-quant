"""
快速开始：加载Star50数据
这是一个最简单的数据加载脚本
"""
from src.data.loaders import load_star50_data

# 加载所有数据
print("正在加载Star50数据...")
data = load_star50_data()

print(f"\n✓ 加载完成!")
print(f"  - 股票数据: {data['stocks'].shape[0]:,} 条记录，{data['stocks']['ts_code'].nunique()} 只股票")
print(f"  - 指数数据: {data['index'].shape[0]:,} 条记录")
print(f"  - 日期范围: {data['stocks']['trade_date'].min().date()} 至 {data['stocks']['trade_date'].max().date()}")

print(f"\n股票列表前10只:")
print(data['stocks']['ts_code'].unique()[:10].tolist())

print(f"\n最新数据预览:")
latest_date = data['stocks']['trade_date'].max()
latest_data = data['stocks'][data['stocks']['trade_date'] == latest_date]
print(latest_data[['ts_code', 'trade_date', 'hfq_close', 'pct_chg', 'vol']].head())
