# XGBoost超参数调优指南

## 目标

优化XGBoost模型参数，实现以下目标：
- **IC > 0.04**: 预测能力（信息系数）
- **IR >= 1.5**: 信息比率（风险调整后的超额收益）
- **年化收益 > 35%**: 策略盈利能力
- **最大回撤 <= 20%**: 风险控制

## 快速开始

### 1. 安装依赖

```bash
cd star50-quant
pip install xgboost optuna scikit-learn
```

### 2. 准备数据

确保已经运行过因子计算脚本：

```bash
python scripts/calculate_factors.py --start_date 2020-01-01 --end_date 2024-12-31
```

### 3. 运行调参

#### 贝叶斯优化（推荐）

最高效的方法，适合复杂参数空间：

```bash
python scripts/tune_xgb_model.py --method bayesian --n_iter 50
```

#### 随机搜索

适合快速探索参数空间：

```bash
python scripts/tune_xgb_model.py --method random --n_iter 100
```

#### 网格搜索

遍历所有组合，适合小参数空间：

```bash
python scripts/tune_xgb_model.py --method grid
```

## 调参方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **随机搜索** | 简单快速，覆盖广 | 不考虑历史结果 | 初步探索 |
| **网格搜索** | 完整遍历，可重现 | 组合爆炸，效率低 | 小参数空间精调 |
| **贝叶斯优化** | 高效，智能搜索 | 需要更多依赖 | 复杂参数空间 |

## 参数空间配置

### XGBoost关键参数

#### 树结构参数
- `max_depth` (3-10): 树的最大深度，控制模型复杂度
- `min_child_weight` (1-15): 叶子节点最小样本权重和，防止过拟合

#### 随机采样参数
- `subsample` (0.5-1.0): 训练样本采样比例
- `colsample_bytree` (0.5-1.0): 每棵树的特征采样比例
- `colsample_bylevel` (0.5-1.0): 每层分裂的特征采样比例

#### 正则化参数
- `gamma` (0.0-1.0): 分裂节点的最小损失减少，越大越保守
- `reg_alpha` (0.0-3.0): L1正则化，增加稀疏性
- `reg_lambda` (0.0-5.0): L2正则化，平滑权重

#### 学习参数
- `learning_rate` (0.005-0.3): 学习率，越小需要更多轮数
- `num_boost_round` (50-300): 提升轮数

## 目标函数说明

综合评分 = IC权重 × IC得分 + IR权重 × IR得分 + 年化收益权重 × 收益得分 + 回撤权重 × 回撤得分

### 默认权重
- IC权重: 0.4 (最重要，预测能力)
- IR权重: 0.3 (风险调整收益)
- 年化收益权重: 0.2 (绝对收益)
- 回撤权重: 0.1 (风险控制)

### 自定义权重

```bash
python scripts/tune_xgb_model.py \
    --method bayesian \
    --ic_weight 0.5 \
    --ir_weight 0.3 \
    --return_weight 0.1 \
    --drawdown_weight 0.1
```

## 高级用法

### 1. 调整数据范围

```bash
python scripts/tune_xgb_model.py \
    --method bayesian \
    --start_date 2021-01-01 \
    --end_date 2024-12-31 \
    --cv_folds 5
```

### 2. 增加搜索强度

```bash
# 贝叶斯优化：更多试验次数
python scripts/tune_xgb_model.py --method bayesian --n_iter 100

# 随机搜索：更密集采样
python scripts/tune_xgb_model.py --method random --n_iter 200
```

### 3. 指定输出目录

```bash
python scripts/tune_xgb_model.py \
    --method bayesian \
    --output_dir tuning_results/xgb_v1
```

## 输出文件

调参完成后会生成以下文件：

```
tuning_results/
├── xgb_tuning_bayesian_20250610_143020.json     # 最佳参数和评分
├── xgb_tuning_bayesian_20250610_143020_trials.csv  # 所有试验记录
├── xgb_best_model_20250610_143020.json          # 最佳模型
├── xgb_feature_importance_20250610_143020.csv   # 特征重要性
└── xgb_predictions_20250610_143020.csv          # 验证集预测
```

## 结果分析

### 1. 查看最佳参数

```python
import json

with open('tuning_results/xgb_tuning_bayesian_20250610_143020.json') as f:
    result = json.load(f)
    print("最佳参数:", result['best_params'])
    print("最佳评分:", result['best_score'])
```

### 2. 分析试验过程

```python
import pandas as pd
import matplotlib.pyplot as plt

trials = pd.read_csv('tuning_results/xgb_tuning_bayesian_20250610_143020_trials.csv')

# 查看评分分布
print(trials[['ic', 'ir', 'annual_return', 'max_drawdown', 'composite_score']].describe())

# 可视化参数重要性
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].scatter(trials['max_depth'], trials['composite_score'])
axes[0, 0].set_xlabel('max_depth')
axes[0, 0].set_ylabel('Composite Score')

axes[0, 1].scatter(trials['learning_rate'], trials['composite_score'])
axes[0, 1].set_xlabel('learning_rate')
axes[0, 1].set_ylabel('Composite Score')

axes[1, 0].scatter(trials['subsample'], trials['composite_score'])
axes[1, 0].set_xlabel('subsample')
axes[1, 0].set_ylabel('Composite Score')

axes[1, 1].scatter(trials['reg_lambda'], trials['composite_score'])
axes[1, 1].set_xlabel('reg_lambda')
axes[1, 1].set_ylabel('Composite Score')

plt.tight_layout()
plt.savefig('tuning_analysis.png')
```

### 3. 特征重要性分析

```python
import pandas as pd

importance = pd.read_csv('tuning_results/xgb_feature_importance_20250610_143020.csv')
print("Top 20 重要特征:")
print(importance.head(20))
```

## 调参策略

### 阶段1: 粗调（Coarse Tuning）

使用随机搜索快速探索参数空间：

```bash
python scripts/tune_xgb_model.py --method random --n_iter 50
```

### 阶段2: 精调（Fine Tuning）

基于粗调结果，缩小参数范围，使用贝叶斯优化：

```bash
# 修改 scripts/tune_xgb_model.py 中的 define_param_space()
# 缩小参数范围到最优区域附近
python scripts/tune_xgb_model.py --method bayesian --n_iter 100
```

### 阶段3: 验证（Validation）

使用最佳参数进行完整回测验证：

```bash
python scripts/run_backtest.py --model_path tuning_results/xgb_best_model_20250610_143020.json
```

## 常见问题

### Q1: 目标无法达成怎么办？

**A1**: 依次尝试以下方法：

1. **增加迭代次数**: `--n_iter 100` 或更多
2. **扩大参数空间**: 修改 `define_param_space()` 函数
3. **调整目标权重**: 根据业务优先级调整权重
4. **优化特征工程**: 
   - 检查特征质量（缺失值、异常值）
   - 添加更多有效因子
   - 使用特征选择（移除噪音特征）
5. **检查数据质量**: 
   - 确认数据完整性
   - 检查标签构造（forward_return）
   - 验证数据时间对齐

### Q2: 调参时间太长？

**A2**: 优化方案：

1. 减少交叉验证折数: `--cv_folds 3`
2. 使用随机搜索: `--method random --n_iter 30`
3. 缩小数据范围: `--start_date 2022-01-01`
4. 减少num_boost_round上限

### Q3: 训练集表现好但验证集差？

**A3**: 过拟合问题，增强正则化：

- 增大 `reg_alpha` 和 `reg_lambda`
- 减小 `max_depth`
- 降低 `subsample` 和 `colsample_bytree`
- 增大 `min_child_weight`

### Q4: IC和IR冲突怎么办？

**A4**: IC和IR有时存在权衡：

- IC高但IR低：预测有效但不稳定，增大正则化
- IR高但IC低：稳定但预测能力弱，增加模型复杂度
- 调整权重平衡：根据策略风格选择侧重点

## 下一步

1. **模型集成**: 将XGBoost与LSTM、LightGBM集成
2. **在线学习**: 实现增量更新机制
3. **组合优化**: 将alpha预测输入到portfolio optimization
4. **风险建模**: 结合风险模型进行风险预算
5. **完整回测**: 使用最佳模型进行全周期回测

## 参考资源

- XGBoost官方文档: https://xgboost.readthedocs.io/
- Optuna教程: https://optuna.readthedocs.io/
- 时间序列交叉验证: https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split
