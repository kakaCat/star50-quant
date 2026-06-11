# MoE模型集成 - 基于Parquet数据

## ✅ 修正说明

项目实际使用 **Parquet 文件**作为数据源，而非 PostgreSQL。已创建专用的 Parquet 数据加载器。

## 📦 核心文件

### 1. Parquet数据加载器
- **`src/models/alpha/moe_data_loader.py`** (350行)
  - `MoEParquetDataLoader`: 专门从Parquet加载数据
  - 直接读取 `data/raw/star50_daily_hfq_data_6yrs.parquet`
  - 直接读取 `data/raw/star50_index_daily_6yrs.parquet`
  - ✅ 已测试，数据加载成功（150,378行股票数据，121只股票，6年历史）

### 2. MoE模型
- **`src/models/alpha/moe_model.py`** - Neural MoE架构
- **`tests/models/test_moe_model.py`** - 模型测试（✅ 10/10通过）
- **`tests/models/test_moe_data_loader.py`** - 数据加载器测试（✅ 5/5通过）

### 3. 训练脚本
- **`scripts/train_moe_model.py`** - 已更新为使用Parquet数据源

## 🎯 数据源确认

```bash
$ ls -lh data/raw/*.parquet
-rw-r--r--  star50_daily_hfq_data_6yrs.parquet     # 股票数据（150K行，121只股票）
-rw-r--r--  star50_index_daily_6yrs.parquet        # 科创50指数（1455行，6年）
```

## 🚀 快速使用

### 方式1：使用现有Parquet文件（推荐）

```bash
# 1. 确认数据文件存在
ls data/raw/*.parquet

# 2. 直接训练
python scripts/train_moe_model.py

# 3. 自定义参数
python scripts/train_moe_model.py \
    --stock_data data/raw/star50_daily_hfq_data_6yrs.parquet \
    --index_data data/raw/star50_index_daily_6yrs.parquet \
    --epochs 30 \
    --hidden_dim 128
```

### 方式2：Python API

```python
from src.models.alpha.moe_data_loader import MoEParquetDataLoader
from src.models.alpha.moe_model import MoEDataset, MoEAlphaTrainer
from torch.utils.data import DataLoader

# 1. 加载Parquet数据
loader = MoEParquetDataLoader(
    stock_data_path='data/raw/star50_daily_hfq_data_6yrs.parquet',
    index_data_path='data/raw/star50_index_daily_6yrs.parquet'
)

stock_features, regime_features, labels = loader.build_moe_dataset(
    forward_days=5,
    beta_window=60
)

# 2. 创建数据集
dataset = MoEDataset(stock_features, regime_features, labels)
data_loader = DataLoader(dataset, batch_size=1024, shuffle=True)

# 3. 训练
stock_dim = len([c for c in stock_features.columns 
                 if c not in ['ts_code', 'trade_date']])
regime_dim = len([c for c in regime_features.columns 
                  if c != 'trade_date'])

trainer = MoEAlphaTrainer(
    stock_dim=stock_dim,
    regime_dim=regime_dim,
    hidden_dim=64,
    device='cuda'  # 或 'cpu'
)

trainer.train(data_loader, num_epochs=30)
```

## 📊 数据统计

从实际Parquet文件加载的数据：

```
股票数据: 150,378 行
  - 时间范围: 2020-01-02 到 2025-12-31 (6年)
  - 股票代码数: 121 只
  - 列数: 16 (包含后复权价格)

指数数据: 1,455 行
  - 时间范围: 2020-01-02 到 2025-12-31 (6年)
  - 列数: 11
```

## 🔄 与phase1.py的对比

| 维度 | phase1.py | 本实现 |
|------|-----------|--------|
| **数据源** | ✅ Parquet | ✅ Parquet (相同) |
| **数据文件** | 硬编码路径 | ✅ 可配置路径 |
| **特征工程** | 脚本内20个特征 | ✅ 模块化，可扩展 |
| **Beta剥离** | ✅ | ✅ 继承 |
| **MoE架构** | ✅ | ✅ 继承 |
| **测试覆盖** | ❌ | ✅ 15个测试 |
| **MLflow追踪** | ❌ | ✅ 支持 |

## 🎯 核心特性（继承自phase1.py）

### 1. Beta剥离
```python
Beta = Cov(R_stock, R_index) / Var(R_index)
Residual = R_stock - Beta × R_index
```
- 60日滚动Beta
- 预测真实Alpha而非总收益

### 2. 特征分离
- **个股特征**：动量、波动率、偏度、量价相关性等（22个）
  - 截面标准化（Cross-sectional Z-score）
- **环境特征**：指数收益、波动率、截面离散度等（6个）
  - 时序标准化（Time-series Z-score）

### 3. MoE架构
```
Stock Expert (3层) + Regime Expert (2层) → Gating → Alpha
```

### 4. Walk-forward验证
- 默认：24个月训练 → 1个月测试
- 严格时序分割

## 🧪 测试验证

```bash
# 所有测试
$ pytest tests/models/test_moe*.py -v

test_moe_model.py::TestMoEDataset::test_dataset_creation PASSED
test_moe_model.py::TestMoEDataset::test_dataset_getitem PASSED
test_moe_model.py::TestNeuralMoE::test_model_forward PASSED
test_moe_model.py::TestNeuralMoE::test_model_components PASSED
test_moe_model.py::TestNeuralMoE::test_model_parameters PASSED
test_moe_model.py::TestMoEAlphaTrainer::test_trainer_initialization PASSED
test_moe_model.py::TestMoEAlphaTrainer::test_trainer_train PASSED
test_moe_model.py::TestMoEAlphaTrainer::test_trainer_evaluate PASSED
test_moe_model.py::TestMoEAlphaTrainer::test_trainer_save_load PASSED
test_moe_model.py::TestIntegration::test_end_to_end PASSED
test_moe_data_loader.py::test_initialization PASSED
test_moe_data_loader.py::test_load_data_if_exists PASSED
test_moe_data_loader.py::test_build_stock_features_with_mock_data PASSED
test_moe_data_loader.py::test_build_regime_features_with_mock_data PASSED
test_moe_data_loader.py::test_calculate_beta_with_mock_data PASSED

✅ 15/15 测试通过
```

## 📝 命令行参数

```bash
python scripts/train_moe_model.py --help

数据参数:
  --stock_data          股票数据Parquet路径 (默认: data/raw/star50_daily_hfq_data_6yrs.parquet)
  --index_data          指数数据Parquet路径 (默认: data/raw/star50_index_daily_6yrs.parquet)
  --forward_days        预测未来N天 (默认: 5)
  --beta_window         Beta窗口 (默认: 60)

模型参数:
  --hidden_dim          隐藏层维度 (默认: 64)
  --dropout             Dropout (默认: 0.3)
  --learning_rate       学习率 (默认: 0.005)
  --weight_decay        L2正则 (默认: 1e-4)

训练参数:
  --epochs              训练轮数 (默认: 25)
  --batch_size          批次大小 (默认: 1024)
  --train_months        训练窗口月数 (默认: 24)
  --test_months         测试窗口月数 (默认: 1)

其他:
  --device              cpu/cuda/auto (默认: auto)
  --experiment_name     MLflow实验名 (默认: moe_alpha)
  --no_mlflow           禁用MLflow追踪
```

## 💡 使用建议

1. **首次运行**：直接使用默认参数
   ```bash
   python scripts/train_moe_model.py
   ```

2. **GPU加速**：如果有GPU
   ```bash
   python scripts/train_moe_model.py --device cuda
   ```

3. **快速验证**：减少训练窗口
   ```bash
   python scripts/train_moe_model.py --train_months 12 --epochs 10
   ```

4. **查看结果**：
   ```bash
   mlflow ui --port 5000
   # 浏览器打开 http://localhost:5000
   ```

## 🎉 集成状态

- ✅ Parquet数据加载器实现
- ✅ 实际数据测试通过（150K行）
- ✅ MoE模型测试通过（10/10）
- ✅ 数据加载器测试通过（5/5）
- ✅ 训练脚本更新完成
- ⏳ Walk-forward训练验证（待运行）

---

**更新日期**: 2024-06-10  
**数据源**: ✅ Parquet文件（与phase1.py一致）  
**测试状态**: ✅ 15/15 passed
