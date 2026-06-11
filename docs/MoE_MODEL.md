# Neural MoE Alpha Model - 集成文档

## 概述

本模块将 `phase1.py` 中的 Neural Mixture of Experts (MoE) 模型集成到项目架构中，实现了：

1. **Beta剥离**：计算60日滚动Beta，剥离市场收益得到真实Alpha作为标签
2. **特征分离**：区分个股特征和环境特征，分别处理
3. **MoE架构**：Stock Expert + Regime Expert + Gating Network
4. **Walk-forward验证**：24个月训练 → 1个月测试的滚动窗口
5. **MLflow集成**：实验追踪和模型管理

## 核心创新点

### 1. Beta剥离（来自 phase1.py）

传统模型预测的是**总收益率** = Beta × 市场收益 + Alpha，但我们真正想要的是**纯Alpha**。

```python
# 60日滚动Beta计算
Beta = Cov(R_stock, R_index) / Var(R_index)

# 真实Alpha（residual）
Residual = R_stock - Beta × R_index
```

**优势**：
- 剥离市场贝塔，专注预测真实选股能力
- 更符合指数增强策略的目标（获取超额收益）
- 降低模型对市场波动的依赖

### 2. 特征分离处理

**个股特征**（Stock Features）：
- 动量因子（momentum_5d, momentum_10d, momentum_20d）
- 波动率因子（volatility_5d, volatility_20d）
- 技术指标（RSI, MACD, Volume Ratio）
- **处理方式**：截面标准化（Cross-sectional Z-score）

**环境特征**（Regime Features）：
- 指数收益率（index_return_1d）
- 市场波动率（index_volatility_20d）
- 截面离散度（market_dispersion）
- 市场宽度（market_breadth）
- **处理方式**：时序标准化（Time-series Z-score）

### 3. MoE网络架构

```
Input: Stock Features (个股) + Regime Features (环境)
         |                           |
         v                           v
   Stock Expert (3层MLP)    Regime Expert (2层MLP)
         |                           |
         v                           v
    Stock Pred                  Regime Pred
         |                           |
         +---------------------------+
                      |
                 Gating Network (动态加权)
                      |
                      v
                  Alpha Prediction
```

**为什么有效？**
- 不同市场状态下，个股因子和市场因子的重要性不同
- Gating Network学习动态调整两个专家的权重
- 牛市可能更依赖Stock Expert，熊市更依赖Regime Expert

## 文件结构

```
star50-quant/
├── src/models/alpha/
│   └── moe_model.py              # MoE模型实现
├── src/models/
│   └── data_loader.py            # 扩展数据加载器（增加Beta剥离）
├── scripts/
│   └── train_moe_model.py        # 训练脚本
├── configs/
│   └── moe_config.yaml           # 配置文件
├── tests/models/
│   └── test_moe_model.py         # 单元测试
└── docs/
    └── MoE_MODEL.md              # 本文档
```

## 快速开始

### 1. 环境准备

```bash
cd star50-quant
source venv/bin/activate
pip install scipy  # 如果还没安装
```

### 2. 数据准备

确保数据库中有：
- `factor_values` 表：因子数据
- `stock_daily` 表：个股价格数据
- `index_daily` 表：指数数据（科创50: 000688.SH）

```bash
# 如果数据库为空，先运行数据收集
python scripts/collect_data.py --type all
python scripts/calculate_factors.py --start_date 2019-01-01 --end_date 2024-12-31
```

### 3. 训练模型

```bash
# 基础训练（使用默认参数）
python scripts/train_moe_model.py

# 自定义参数
python scripts/train_moe_model.py \
    --start_date 2019-01-01 \
    --end_date 2024-12-31 \
    --epochs 30 \
    --hidden_dim 128 \
    --learning_rate 0.001 \
    --train_months 36 \
    --test_months 1
```

### 4. 查看结果

```bash
# 查看MLflow实验结果
mlflow ui --port 5000
# 浏览器打开 http://localhost:5000

# 预测结果保存在
ls outputs/moe_predictions/
```

## 参数说明

### 数据参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--start_date` | 2019-01-01 | 训练数据开始日期 |
| `--end_date` | 2024-12-31 | 训练数据结束日期 |
| `--index_code` | 000688.SH | 基准指数代码（科创50） |
| `--forward_days` | 5 | 预测未来N天收益率 |
| `--beta_window` | 60 | Beta计算滚动窗口 |

### 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--hidden_dim` | 64 | 隐藏层维度 |
| `--dropout` | 0.3 | Dropout比例 |
| `--learning_rate` | 0.005 | 学习率 |
| `--weight_decay` | 1e-4 | L2正则化系数 |

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 25 | 每个窗口训练轮数 |
| `--batch_size` | 1024 | 批次大小 |
| `--train_months` | 24 | Walk-forward训练窗口（月） |
| `--test_months` | 1 | Walk-forward测试窗口（月） |

## 评估指标

模型使用以下指标评估：

1. **Rank IC**（秩相关系数）
   - 预测值与真实residual的Spearman相关系数
   - 每日计算，然后取均值
   - **目标**：>0.03（优秀），>0.02（良好）

2. **ICIR**（Information Coefficient IR）
   - ICIR = Mean(IC) / Std(IC)
   - 衡量IC的稳定性
   - **目标**：>0.5（优秀），>0.3（良好）

3. **IC > 0 占比**
   - IC为正的交易日占比
   - **目标**：>60%

## 与原始 phase1.py 的对比

| 维度 | phase1.py | 本实现 |
|------|-----------|--------|
| **数据源** | Parquet缓存文件 | PostgreSQL数据库 |
| **特征数量** | ~20个内置特征 | 30个基础因子（可扩展到160+） |
| **特征工程** | 脚本内硬编码 | 模块化（`src/features/`） |
| **Beta剥离** | ✅ 实现 | ✅ 继承并改进 |
| **MoE架构** | ✅ 实现 | ✅ 继承 |
| **实验追踪** | ❌ 无 | ✅ MLflow集成 |
| **测试覆盖** | ❌ 无 | ✅ 单元测试 + 集成测试 |
| **可扩展性** | ❌ 单文件脚本 | ✅ 模块化架构 |

## 使用示例

### 1. Python API调用

```python
from src.models.data_loader import FactorDataLoader
from src.models.alpha.moe_model import MoEDataset, MoEAlphaTrainer
from torch.utils.data import DataLoader

# 加载数据
loader = FactorDataLoader()
stock_features, regime_features, labels = loader.build_moe_dataset(
    start_date='2020-01-01',
    end_date='2023-12-31',
    forward_days=5,
    beta_window=60
)

# 创建数据集
dataset = MoEDataset(stock_features, regime_features, labels)
data_loader = DataLoader(dataset, batch_size=512, shuffle=True)

# 训练模型
stock_dim = len([c for c in stock_features.columns if c not in ['ts_code', 'factor_date']])
regime_dim = len([c for c in regime_features.columns if c != 'factor_date'])

trainer = MoEAlphaTrainer(
    stock_dim=stock_dim,
    regime_dim=regime_dim,
    hidden_dim=64,
    device='cuda'  # 或 'cpu'
)

trainer.train(data_loader, num_epochs=30)

# 预测
predictions = trainer.predict(test_stock_x, test_regime_x)
```

### 2. 集成到回测流程

```python
# 在回测脚本中使用MoE预测
from src.models.alpha.moe_model import MoEAlphaTrainer

# 1. 加载训练好的模型
trainer = MoEAlphaTrainer(stock_dim=160, regime_dim=10, device='cpu')
trainer.load('models/moe_model_20241210.pth')

# 2. 每日预测alpha
for date in trading_dates:
    # 获取当日特征
    stock_x = get_stock_features(date)
    regime_x = get_regime_features(date)
    
    # 预测alpha
    alpha_predictions = trainer.predict(stock_x, regime_x)
    
    # 3. 结合风险模型做组合优化
    portfolio_weights = optimize_portfolio(
        alpha=alpha_predictions,
        risk_matrix=risk_model.predict(stock_x),
        constraints=constraints
    )
    
    # 4. 执行交易
    execute_trades(portfolio_weights)
```

## 进阶用法

### 1. 特征工程集成

启用项目的特征工程模块（30因子→160+特征）：

```python
from src.features.engineering import FeatureEngineer

# 加载因子
factors_wide = loader.pivot_factors(factors_df)

# 特征扩展
engineer = FeatureEngineer('configs/feature_config.yaml')
features_expanded = engineer.transform(factors_wide)  # 160+特征

# 然后用于MoE训练
```

### 2. 超参数调优

```bash
# 使用不同超参数组合
for hidden in 32 64 128; do
    for lr in 0.001 0.005 0.01; do
        python scripts/train_moe_model.py \
            --hidden_dim $hidden \
            --learning_rate $lr \
            --experiment_name "moe_tuning_${hidden}_${lr}"
    done
done

# MLflow会自动记录所有实验
```

### 3. 自定义特征分组

修改 `configs/moe_config.yaml`：

```yaml
features:
  stock_features:
    - momentum_5d
    - momentum_10d
    - volatility_20d
    - rsi_14
    # ... 添加更多个股特征
  
  regime_features:
    - index_return_1d
    - index_volatility_20d
    - market_dispersion
    # ... 添加更多环境特征
```

## 常见问题

### Q1: 为什么IC很低？

**可能原因**：
1. 因子质量不足：检查30个基础因子的IC
2. Beta窗口不合适：尝试30/90/120日
3. 过拟合：增加dropout或减少hidden_dim
4. 数据质量问题：检查是否有未来数据泄漏

### Q2: 训练很慢怎么办？

**优化方法**：
1. 使用GPU：`--device cuda`
2. 增大batch_size：`--batch_size 2048`
3. 减少训练窗口：`--train_months 12`
4. 减少epoch：`--epochs 15`

### Q3: 如何与LSTM模型对比？

```bash
# 训练LSTM
python scripts/train_alpha_model.py --model lstm

# 训练MoE
python scripts/train_moe_model.py

# MLflow对比
mlflow ui  # 在UI中对比IC/ICIR指标
```

### Q4: 可以用于A股主板吗？

可以！只需修改：
1. `--index_code 000300.SH`（改为沪深300）
2. 调整Beta窗口（主板波动较小，可用120日）
3. 调整交易成本参数

## 下一步计划

1. **集成RL优化器**：结合MoE预测和RL动态调仓
2. **多时间尺度**：同时预测5日/10日/20日alpha
3. **Ensemble**：MoE + LSTM + LightGBM集成
4. **在线学习**：增量更新模型权重

## 参考文献

- [Mixture of Experts原始论文](https://arxiv.org/abs/1701.06538)
- WorldQuant Alpha因子库
- 科创50指数增强策略实践

## 贡献者

集成完成日期：2024-12-10
测试状态：✅ 10/10 passed
