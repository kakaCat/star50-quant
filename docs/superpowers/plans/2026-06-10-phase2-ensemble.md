# Phase 2: 模型集成方案

## 目标

通过集成学习提升预测能力：
- **当前**: IC = 0.0178 (单模型)
- **目标**: IC = 0.03-0.04 (集成后)
- **提升**: 约70-120%

## 核心策略

### 1. 多窗口预测

预测不同时间窗口的收益率：
- **1天**: 捕获短期反转
- **3天**: 短期趋势
- **5天**: 中期趋势（当前）
- **10天**: 中期动量
- **20天**: 长期趋势

**原理**: 不同窗口捕获不同频率的信号，组合后更稳定

### 2. 多模型集成

为每个窗口训练多个模型：
- **LightGBM** (默认)
- **LightGBM** (高正则化)
- **LightGBM** (深度树)

如果资源允许，可选添加：
- **XGBoost**
- **CatBoost**
- **Random Forest**
- **Linear Ridge**

**基础配置**: 5窗口 × 3模型 = **15个基础模型**

### 3. 元学习器

使用Ridge回归融合基础模型：
- 输入：15个基础模型的预测
- 输出：最终预测
- 优势：自动学习最优权重，防止过拟合

## 架构设计

```
                           ┌─────────────┐
                           │   原始数据   │
                           └──────┬──────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │    特征工程（9个核心因子）  │
                    └─────────────┬─────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          │                       │                       │
    ┌─────▼─────┐          ┌─────▼─────┐         ┌─────▼─────┐
    │  Window=1 │          │  Window=5 │         │ Window=20 │
    │   Label   │          │   Label   │         │   Label   │
    └─────┬─────┘          └─────┬─────┘         └─────┬─────┘
          │                       │                       │
    ┌─────┴─────┐          ┌─────┴─────┐         ┌─────┴─────┐
    │ LightGBM1 │          │ LightGBM1 │         │ LightGBM1 │
    │ LightGBM2 │          │ LightGBM2 │         │ LightGBM2 │
    │ LightGBM3 │          │ LightGBM3 │         │ LightGBM3 │
    └─────┬─────┘          └─────┬─────┘         └─────┬─────┘
          │                       │                       │
          └───────────────────────┼───────────────────────┘
                                  │
                          ┌───────▼───────┐
                          │ Meta-Learner  │
                          │  (Ridge)      │
                          └───────┬───────┘
                                  │
                          ┌───────▼───────┐
                          │  最终预测      │
                          └───────────────┘
```

## 实施步骤

### Step 1: 准备多窗口数据集

**任务**:
- 为每个窗口(1/3/5/10/20天)生成标签
- 保持相同的特征集（9个核心因子）
- 时间序列划分：80%训练，20%验证

**文件**: `src/models/multi_window_loader.py`

### Step 2: 实现基础模型训练器

**任务**:
- 封装LightGBM多配置训练
- 支持不同超参数配置
- 统一的接口和预测方法

**文件**: `src/models/base_models.py`

### Step 3: 实现集成训练流程

**任务**:
- 训练所有基础模型
- 收集基础模型预测
- 训练Ridge元学习器
- 保存完整模型集合

**文件**: `src/models/ensemble_trainer.py`

### Step 4: 模型评估

**任务**:
- 计算每个基础模型的IC
- 计算集成模型的IC
- 对比单模型 vs 集成效果
- 生成性能报告

**文件**: `scripts/train_ensemble.py`

### Step 5: 验证与优化

**任务**:
- Walk-Forward验证
- 时间稳定性分析
- 超参数网格搜索（可选）
- 最终性能评估

**文件**: `scripts/validate_ensemble.py`

## 技术细节

### 多窗口标签生成

```python
def generate_multi_window_labels(df, windows=[1, 3, 5, 10, 20]):
    """为多个窗口生成标签"""
    labels = {}
    for window in windows:
        labels[f'forward_{window}d'] = df.groupby('ts_code')['close'].apply(
            lambda x: x.pct_change(window).shift(-window)
        )
    return labels
```

### 基础模型配置

```python
base_models_config = {
    'lgbm_default': {
        'learning_rate': 0.05,
        'max_depth': 6,
        'num_leaves': 31,
    },
    'lgbm_regularized': {
        'learning_rate': 0.03,
        'max_depth': 4,
        'num_leaves': 15,
        'reg_alpha': 0.5,
        'reg_lambda': 1.0,
    },
    'lgbm_deep': {
        'learning_rate': 0.02,
        'max_depth': 8,
        'num_leaves': 63,
    }
}
```

### Ridge元学习器

```python
from sklearn.linear_model import Ridge

# 基础模型预测作为特征
meta_features = np.column_stack([
    model1.predict(X), model2.predict(X), ...
])

# 训练元学习器
meta_learner = Ridge(alpha=1.0)
meta_learner.fit(meta_features, y)
```

## 预期效果

### 基础模型（单窗口）
- Window=1: IC ~ 0.010-0.015
- Window=3: IC ~ 0.015-0.020
- Window=5: IC ~ 0.018-0.022 ✓ (已验证)
- Window=10: IC ~ 0.015-0.020
- Window=20: IC ~ 0.012-0.018

### 集成模型
- **目标IC**: 0.03-0.04
- **IC>0比例**: >60%
- **IR**: >0.3
- **时间稳定性**: 月度IC标准差 <0.10

### 提升来源
1. **多窗口互补**: 不同时间尺度信号叠加
2. **模型多样性**: 不同超参数捕获不同模式
3. **元学习器**: 自适应权重，防止过拟合

## 时间估算

- Step 1: 1小时（数据准备）
- Step 2: 1小时（基础模型）
- Step 3: 2小时（集成训练）
- Step 4: 1小时（评估）
- Step 5: 1小时（验证）

**总计**: 约6小时

## 验收标准

### 必须达成
- ✓ 集成模型IC日度 > 0.025
- ✓ 集成模型IC > 任何单基础模型
- ✓ IC>0比例 > 55%
- ✓ 时间稳定性: 验证集各月IC无系统性偏差

### 期望达成
- IC日度 > 0.030
- IC>0比例 > 60%
- IR > 0.3

## 风险与应对

### 风险1: 集成后没有提升
**原因**: 基础模型过于相似
**应对**: 增加模型多样性（不同算法、不同特征子集）

### 风险2: 元学习器过拟合
**原因**: Ridge正则化不足
**应对**: 增大alpha参数，使用交叉验证选择

### 风险3: 计算资源不足
**原因**: 15个模型训练耗时
**应对**: 减少基础模型数量（保留3窗口×2模型=6个）

## 下一步行动

1. 创建`src/models/multi_window_loader.py`
2. 实现多窗口数据加载
3. 训练第一个多窗口模型验证可行性
4. 如果效果好，继续完整集成流程
