# Star50量化投资系统 - 数据集成说明

## 数据文件

项目已集成以下历史数据文件（位于 `data/raw/`）：

### 1. star50_daily_hfq_data_6yrs.parquet
- **描述**: Star50成分股日线后复权数据
- **时间范围**: 2020-01-02 至 2025-12-31（6年）
- **股票数量**: 121只
- **记录数**: 150,378条
- **字段说明**:
  - `ts_code`: 股票代码（如688001.SH）
  - `trade_date`: 交易日期
  - `open, high, low, close`: 原始OHLC价格
  - `pre_close`: 前收盘价
  - `change, pct_chg`: 涨跌额和涨跌幅(%)
  - `vol, amount`: 成交量(手)和成交额(千元)
  - `adj_factor`: 复权因子
  - `hfq_open, hfq_high, hfq_low, hfq_close`: 后复权OHLC价格

### 2. star50_index_daily_6yrs.parquet
- **描述**: Star50指数(000688.SH)日线数据
- **时间范围**: 2020-01-02 至 2025-12-31（6年）
- **记录数**: 1,455条
- **字段说明**:
  - `ts_code`: 指数代码(000688.SH)
  - `trade_date`: 交易日期
  - `open, high, low, close`: OHLC价格
  - `pre_close`: 前收盘价
  - `change, pct_chg`: 涨跌额和涨跌幅(%)
  - `vol, amount`: 成交量和成交额

## 使用方式

### 1. 基本加载

```python
from src.data.loaders import DataLoader

loader = DataLoader()

# 加载所有股票数据
df_stocks = loader.load_stock_data()

# 加载指数数据
df_index = loader.load_index_data()
```

### 2. 筛选加载

```python
# 加载特定股票和日期范围
df = loader.load_stock_data(
    stock_codes=['688001.SH', '688008.SH'],
    start_date='2024-01-01',
    end_date='2024-12-31'
)
```

### 3. 使用便捷函数

```python
from src.data.loaders import load_star50_data

# 同时加载股票和指数数据
data = load_star50_data(
    stock_codes=['688001.SH', '688008.SH'],
    start_date='2024-01-01',
    include_index=True
)

df_stocks = data['stocks']
df_index = data['index']
```

### 4. 获取元数据

```python
# 获取股票列表
stock_codes = loader.get_stock_list()
print(f"共有 {len(stock_codes)} 只股票")

# 获取日期范围
start_date, end_date = loader.get_date_range()
print(f"日期范围: {start_date} 至 {end_date}")
```

## 应用场景

### 1. 因子计算
使用后复权价格计算技术因子，避免分红送股对因子值的影响：

```python
from src.features.momentum import MomentumCalculator

df = loader.load_stock_data(start_date='2024-01-01')
calc = MomentumCalculator()

# 使用后复权价格计算动量因子
for stock_code, group in df.groupby('ts_code'):
    group = group.sort_values('trade_date')
    momentum = calc.calculate(group[['hfq_close']])
```

### 2. Alpha模型训练
准备训练数据集：

```python
# 加载训练期数据
train_data = load_star50_data(
    start_date='2020-01-01',
    end_date='2023-12-31',
    include_index=True
)

# 计算标签（未来收益）
train_data['stocks']['future_return'] = (
    train_data['stocks']
    .groupby('ts_code')['hfq_close']
    .pct_change(5)  # 5日收益
    .shift(-5)      # 未来5日
)
```

### 3. 回测系统
使用历史数据进行策略回测：

```python
# 加载回测期数据
backtest_data = load_star50_data(
    start_date='2024-01-01',
    end_date='2024-12-31',
    include_index=True
)

# 按日期遍历进行回测
for date in backtest_data['stocks']['trade_date'].unique():
    daily_data = backtest_data['stocks'][
        backtest_data['stocks']['trade_date'] == date
    ]
    # 执行选股和组合构建逻辑
```

### 4. 基准对比
对比策略表现与指数基准：

```python
# 计算相对指数的超额收益
df_stock = data['stocks'][['trade_date', 'pct_chg']]
df_index = data['index'][['trade_date', 'pct_chg']]

merged = pd.merge(df_stock, df_index, on='trade_date', suffixes=('_stock', '_index'))
merged['alpha'] = merged['pct_chg_stock'] - merged['pct_chg_index']
```

## 数据质量说明

- **完整性**: 6年完整交易日数据，无缺失日期
- **准确性**: 使用后复权价格，已处理分红送股影响
- **时效性**: 数据截至2025-12-31
- **覆盖面**: 包含Star50所有历史成分股（121只）

## 下一步工作

基于这些数据，可以进行：

1. **因子工程**: 计算量价、技术、基本面等各类因子
2. **特征工程**: 构建机器学习特征矩阵
3. **模型训练**: 训练Alpha预测模型
4. **策略回测**: 验证选股和组合策略效果
5. **风险分析**: 评估策略风险特征和归因

## 更多示例

详细使用示例请参考：[examples/load_data_example.py](../examples/load_data_example.py)

运行示例：
```bash
cd /Users/mac/Documents/ai/bisai/star50-quant
python examples/load_data_example.py
```
