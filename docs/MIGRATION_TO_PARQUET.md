# Parquet数据迁移指南

## 从PostgreSQL迁移到Parquet文件

项目已经完成从PostgreSQL数据库到Parquet文件的迁移，提供更简单、更快速的数据访问方式。

## 变化对比

### 旧方式（PostgreSQL）

**数据加载器**：`src/models/data_loader.py`
```python
from src.models.data_loader import FactorDataLoader

with FactorDataLoader(db_name='star50_quant') as loader:
    factors = loader.load_factors(start_date, end_date)
    prices = loader.load_prices(start_date, end_date)
```

**因子计算**：`scripts/calculate_factors.py`
```bash
python scripts/calculate_factors.py --all --start 2020-01-01
```

**依赖**：
- PostgreSQL数据库
- psycopg2-binary
- 需要配置数据库连接

### 新方式（Parquet）

**数据加载器**：`src/models/data_loader_parquet.py`
```python
from src.models.data_loader_parquet import FactorDataLoader

loader = FactorDataLoader(data_dir='data/raw')
# 或使用兼容接口
with FactorDataLoaderCompat(data_dir='data/raw') as loader:
    features, labels = loader.build_dataset(start_date, end_date)
```

**因子计算**：`scripts/calculate_factors_parquet.py`
```bash
python scripts/calculate_factors_parquet.py --all --start 2020-01-01
```

**依赖**：
- 仅需parquet文件
- pyarrow
- 无需数据库配置

## 迁移步骤

### 1. 更新数据加载代码

**旧代码**：
```python
from src.models.data_loader import FactorDataLoader

with FactorDataLoader(db_name='star50_quant') as loader:
    features, labels = loader.build_dataset(
        start_date='2024-01-01',
        end_date='2024-12-31'
    )
```

**新代码**：
```python
from src.models.data_loader_parquet import FactorDataLoader

loader = FactorDataLoader(data_dir='data/raw')
features, labels = loader.build_dataset(
    start_date='2024-01-01',
    end_date='2024-12-31',
    use_basic_factors=True  # 或指定factor_file
)
```

### 2. 更新因子计算脚本

**旧方式**：
```bash
python scripts/calculate_factors.py --all
```

**新方式**：
```bash
python scripts/calculate_factors_parquet.py --all --output data/processed/factors.parquet
```

### 3. 更新训练脚本

如果你的训练脚本使用了旧的数据加载器，需要替换导入：

```python
# 旧代码
from src.models.data_loader import FactorDataLoader

# 新代码
from src.models.data_loader_parquet import FactorDataLoader
# 或者为了完全兼容，使用：
from src.models.data_loader_parquet import FactorDataLoaderCompat as FactorDataLoader
```

## API对比

### FactorDataLoader

| 方法 | PostgreSQL版本 | Parquet版本 | 变化 |
|------|---------------|-------------|------|
| `__init__` | `db_name='star50_quant'` | `data_dir='data/raw'` | 参数改变 |
| `connect()` | 连接数据库 | 无操作 | 不再需要 |
| `close()` | 关闭连接 | 无操作 | 不再需要 |
| `load_prices()` | 从数据库查询 | 从parquet读取 | 新增`use_hfq`参数 |
| `load_factors()` | 从数据库查询 | 改为`load_factors_from_file()` | 需要指定文件 |
| `build_dataset()` | 从数据库构建 | 从parquet构建 | 新增参数 |

### 新增功能

**Parquet版本新增**：
```python
# 加载原始股票特征
loader.load_stock_features(start_date, end_date, stocks)

# 计算基本因子
loader.calculate_basic_factors(df)

# 获取日期范围
start, end = loader.get_date_range()
```

## 优势

### 性能
- ✅ **无需数据库**：不需要PostgreSQL服务
- ✅ **加载更快**：Parquet列式存储，读取速度快
- ✅ **部署简单**：只需复制parquet文件

### 便捷性
- ✅ **文件即数据**：数据和代码在一起
- ✅ **版本控制**：可以用git-lfs管理数据版本
- ✅ **易于分享**：直接分享parquet文件

### 开发体验
- ✅ **环境配置简单**：无需配置数据库
- ✅ **调试方便**：可以直接查看文件
- ✅ **云端友好**：可以存储到S3等对象存储

## 迁移检查清单

- [ ] 数据文件已放置到`data/raw/`目录
  - [ ] `star50_daily_hfq_data_6yrs.parquet`
  - [ ] `star50_index_daily_6yrs.parquet`
- [ ] 已安装`pyarrow`依赖
- [ ] 数据加载代码已更新为新版本
- [ ] 因子计算脚本已更新
- [ ] 训练脚本已更新导入
- [ ] 已测试新代码能正常运行

## 测试迁移

运行以下命令测试新的数据加载：

```bash
# 测试数据加载
python quickstart_data.py

# 测试因子计算（单只股票）
python scripts/calculate_factors_parquet.py --stock 688001.SH --start 2024-01-01 --end 2024-12-31

# 测试数据集构建
python -c "
from src.models.data_loader_parquet import FactorDataLoader
loader = FactorDataLoader()
features, labels = loader.build_dataset('2024-01-01', '2024-12-31', use_basic_factors=True)
print(f'Features: {features.shape}, Labels: {labels.shape}')
"
```

## 保留旧代码

旧的PostgreSQL版本文件保留在原位置：
- `src/models/data_loader.py`（旧版）
- `scripts/calculate_factors.py`（旧版）

如果需要回退，可以继续使用这些文件。

## 完全切换

当确认新版本工作正常后，可以：

1. 备份旧文件
2. 将新版本重命名为主版本
3. 移除PostgreSQL依赖

```bash
# 备份旧文件
mv src/models/data_loader.py src/models/data_loader_pg_backup.py
mv scripts/calculate_factors.py scripts/calculate_factors_pg_backup.py

# 使用新文件作为主版本
cp src/models/data_loader_parquet.py src/models/data_loader.py
cp scripts/calculate_factors_parquet.py scripts/calculate_factors.py

# 更新requirements.txt，移除psycopg2-binary（可选）
```

## 常见问题

### Q: 如何加载已计算的因子？

```python
loader = FactorDataLoader()
features, labels = loader.build_dataset(
    start_date='2024-01-01',
    end_date='2024-12-31',
    factor_file='data/processed/factors.parquet'
)
```

### Q: 数据更新怎么办？

直接替换parquet文件即可，或者重新运行数据采集脚本生成新文件。

### Q: 可以混用PostgreSQL和Parquet吗？

可以，两个版本的加载器API类似，可以根据需要选择使用。

### Q: Parquet文件太大怎么办？

可以：
1. 按年份分片存储
2. 使用压缩（parquet默认支持）
3. 只加载需要的列和日期范围

## 后续优化

可以考虑的优化方向：

1. **分片存储**：按年份或股票分片
2. **增量更新**：只更新增量数据
3. **预计算因子**：将常用因子预计算并缓存
4. **云端存储**：使用S3/OSS存储大文件
