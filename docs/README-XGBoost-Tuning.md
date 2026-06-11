# XGBoost 超参数调优系统

## 概述

完整的XGBoost超参数调优系统，支持三种调参方法，目标是实现：
- **IC > 0.04**: 信息系数（预测能力）
- **IR >= 1.5**: 信息比率（风险调整后的超额收益）
- **年化收益 > 35%**: 策略盈利能力
- **最大回撤 <= 20%**: 风险控制

## 快速开始

### 1. 测试系统（使用合成数据）

```bash
cd star50-quant
python scripts/test_tuning_system.py
```

**预期输出**: ✅ 4/4 测试通过

### 2. 真实数据调参

#### 准备数据
```bash
python scripts/calculate_factors.py --start_date 2020-01-01 --end_date 2024-12-31
```

#### 运行调参（推荐：贝叶斯优化）
```bash
python scripts/tune_xgb_model.py --method bayesian --n_iter 50
```

**预计时间**: 30-60分钟（取决于数据量和硬件配置）

## 三种调参方法

### 🎯 贝叶斯优化（推荐）
- **最高效**: 利用历史试验信息智能搜索
- **适用场景**: 复杂参数空间，中等预算（50-100次试验）
- **技术**: TPE (Tree-structured Parzen Estimator)

```bash
python scripts/tune_xgb_model.py --method bayesian --n_iter 50
```

### 🎲 随机搜索
- **快速探索**: 随机采样参数空间
- **适用场景**: 初步探索，快速验证（100-200次试验）
- **优点**: 简单、覆盖广

```bash
python scripts/tune_xgb_model.py --method random --n_iter 100
```

### 📊 网格搜索
- **完整遍历**: 遍历所有参数组合
- **适用场景**: 小参数空间精调（<50个组合）
- **优点**: 可重现、完整

```bash
python scripts/tune_xgb_model.py --method grid
```

## 高级配置

### 自定义目标权重

根据策略优先级调整各指标权重：

```bash
python scripts/tune_xgb_model.py \
    --method bayesian \
    --ic_weight 0.5 \
    --ir_weight 0.3 \
    --return_weight 0.15 \
    --drawdown_weight 0.05
```

### 调整数据范围和交叉验证

```bash
python scripts/tune_xgb_model.py \
    --method bayesian \
    --start_date 2021-01-01 \
    --end_date 2024-12-31 \
    --cv_folds 5
```

### 指定输出目录

```bash
python scripts/tune_xgb_model.py \
    --method bayesian \
    --output_dir tuning_results/xgb_v1
```

## 输出文件

```
tuning_results/
├── xgb_tuning_bayesian_20250610_143020.json       # 最佳参数和评分
├── xgb_tuning_bayesian_20250610_143020_trials.csv # 所有试验记录
├── xgb_best_model_20250610_143020.json            # 最佳模型
├── xgb_feature_importance_20250610_143020.csv     # 特征重要性
└── xgb_predictions_20250610_143020.csv            # 验证集预测
```

## 结果分析

### 查看最佳参数

```python
import json

with open('tuning_results/xgb_tuning_bayesian_20250610_143020.json') as f:
    result = json.load(f)
    print("最佳参数:", result['best_params'])
    print("最佳评分:", result['best_score'])
```

### 分析试验过程

```python
import pandas as pd

trials = pd.read_csv('tuning_results/xgb_tuning_bayesian_20250610_143020_trials.csv')
print(trials[['ic', 'ir', 'annual_return', 'max_drawdown', 'composite_score']].describe())
```

### 特征重要性

```python
importance = pd.read_csv('tuning_results/xgb_feature_importance_20250610_143020.csv')
print("Top 20 重要特征:")
print(importance.head(20))
```

## 调参策略

### 三阶段策略

#### 阶段1: 粗调（1-2小时）
```bash
python scripts/tune_xgb_model.py --method random --n_iter 50
```
目标：快速探索参数空间，识别有效区域

#### 阶段2: 精调（2-4小时）
```bash
python scripts/tune_xgb_model.py --method bayesian --n_iter 100
```
目标：基于粗调结果，智能搜索最优参数

#### 阶段3: 验证
```bash
# 如果参数空间较小，可用网格搜索验证关键参数
python scripts/tune_xgb_model.py --method grid
```

## 常见问题

### Q: 目标无法达成怎么办？

**A**: 依次尝试：

1. **增加迭代次数**: `--n_iter 100` 或更多
2. **调整目标权重**: 根据业务优先级调整
3. **优化特征工程**: 
   - 检查特征质量（缺失值、异常值）
   - 添加更多有效因子
   - 使用特征选择
4. **扩大参数空间**: 修改 `scripts/tune_xgb_model.py` 中的 `define_param_space()`

### Q: 调参时间太长？

**A**: 优化方案：

1. 减少交叉验证折数: `--cv_folds 3`
2. 使用随机搜索: `--method random --n_iter 30`
3. 缩小数据范围: `--start_date 2022-01-01`

### Q: 过拟合怎么办？

**A**: 增强正则化：

- 增大参数空间中 `reg_alpha` 和 `reg_lambda` 的范围
- 减小 `max_depth`
- 降低 `subsample` 和 `colsample_bytree`

## 文档

- **完整指南**: [docs/xgboost-tuning-guide.md](docs/xgboost-tuning-guide.md)
- **实现总结**: [docs/xgboost-tuning-summary.md](docs/xgboost-tuning-summary.md)

## 技术架构

### 核心模块

1. **XGBoostAlphaModel** (`src/models/xgb_model.py`)
   - XGBoost模型封装
   - 训练、预测、交叉验证
   - 特征重要性分析

2. **ObjectiveFunction** (`src/models/hyperparameter_tuning.py`)
   - 多目标综合评估
   - 时间序列交叉验证
   - 策略模拟

3. **HyperparameterTuner** (`src/models/hyperparameter_tuning.py`)
   - 三种调参方法实现
   - 结果记录和分析
   - 参数空间管理

### 参数空间

| 参数 | 范围 | 作用 |
|------|------|------|
| max_depth | 3-10 | 树深度 |
| learning_rate | 0.005-0.3 | 学习率 |
| subsample | 0.5-1.0 | 样本采样 |
| colsample_bytree | 0.5-1.0 | 特征采样（树） |
| colsample_bylevel | 0.5-1.0 | 特征采样（层） |
| min_child_weight | 1-15 | 最小样本权重和 |
| gamma | 0.0-1.0 | 最小损失减少 |
| reg_alpha | 0.0-3.0 | L1正则化 |
| reg_lambda | 0.0-5.0 | L2正则化 |
| num_boost_round | 50-300 | 提升轮数 |

## 性能基准

基于合成数据测试（500样本，20特征，5折交叉验证）：

| 方法 | 试验次数 | 时间 | 最佳评分 |
|------|----------|------|----------|
| 随机搜索 | 5 | ~2秒 | 0.538 |
| 网格搜索 | 8 | ~3秒 | 0.500 |
| 贝叶斯优化 | 5 | ~2秒 | 0.300 |

*实际数据的时间会显著增加（10-100倍）*

## 下一步

完成调参后：

1. **完整回测**
   ```bash
   python scripts/run_backtest.py --model_path tuning_results/xgb_best_model_*.json
   ```

2. **特征工程优化**
   - 基于特征重要性分析
   - 移除低重要性特征
   - 添加新的有效因子

3. **模型集成**
   - XGBoost + LSTM + LightGBM
   - Stacking/Blending策略

4. **组合优化集成**
   - 将alpha预测输入portfolio optimization
   - 结合风险模型

## 依赖

- xgboost >= 2.0.0
- optuna >= 3.0.0
- scikit-learn >= 1.2.0
- pandas >= 1.5.0
- numpy >= 1.23.0

## 测试状态

✅ **所有测试通过**

- ✅ 目标函数测试
- ✅ 随机搜索测试
- ✅ 网格搜索测试
- ✅ 贝叶斯优化测试

运行测试：
```bash
python scripts/test_tuning_system.py
```

## 许可

本调参系统是科创50指数增强策略系统的一部分。
