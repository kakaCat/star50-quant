# MoE模型调优完整指南
## 如何优化Neural MoE Alpha预测模型

---

## 📋 目录

1. [快速开始](#快速开始)
2. [超参数调优](#超参数调优)
3. [特征工程优化](#特征工程优化)
4. [架构调整](#架构调整)
5. [训练策略优化](#训练策略优化)
6. [集成学习](#集成学习)
7. [诊断与调试](#诊断与调试)

---

## 🚀 快速开始

### 当前性能基线

```
训练配置: 3轮快速训练
Rank IC: 0.0059
ICIR: 0.0445
IC > 0占比: 50.70%
```

### 目标性能

```
目标: Rank IC > 0.02, ICIR > 0.5
优秀: Rank IC > 0.03, ICIR > 1.0
```

---

## 1️⃣ 超参数调优

### 方法1：自动超参数搜索

我已经创建了自动调优脚本：

```bash
# 网格搜索（系统化）
python scripts/tune_moe_hyperparams.py --method grid --max_trials 20

# 随机搜索（快速探索）
python scripts/tune_moe_hyperparams.py --method random --max_trials 50

# 查看MLflow实验结果
mlflow ui --port 5000
```

### 方法2：手动调优

#### 关键超参数及推荐范围

| 参数 | 当前值 | 推荐范围 | 影响 |
|------|--------|---------|------|
| `epochs` | 3 | **25-30** | ⭐⭐⭐ 最重要！更多训练轮次显著提升IC |
| `hidden_dim` | 64 | 64-128 | ⭐⭐ 更大容量捕捉复杂模式 |
| `learning_rate` | 0.005 | 0.001-0.01 | ⭐⭐ 太大不稳定，太小收敛慢 |
| `dropout` | 0.3 | 0.2-0.5 | ⭐ 防止过拟合 |
| `weight_decay` | 1e-4 | 1e-5 ~ 1e-3 | ⭐ L2正则化强度 |
| `batch_size` | 1024 | 512-2048 | 更大batch更稳定但慢 |
| `train_months` | 12 | 12-36 | 更多历史数据 |

#### 第一步：增加训练轮数（最重要！）

```bash
# 从3轮增加到30轮
python scripts/train_moe_model.py --epochs 30

# 预期: IC从0.006提升至0.02-0.03
```

#### 第二步：调整模型容量

```bash
# 增大模型（如果数据充足）
python scripts/train_moe_model.py --epochs 30 --hidden_dim 128

# 减小模型（如果过拟合）
python scripts/train_moe_model.py --epochs 30 --hidden_dim 32 --dropout 0.5
```

#### 第三步：调整学习率

```bash
# 学习率扫描
for lr in 0.001 0.003 0.005 0.01; do
    python scripts/train_moe_model.py \
        --epochs 30 \
        --learning_rate $lr \
        --experiment_name "moe_lr_scan"
done
```

#### 第四步：完整调优

```bash
# 推荐配置
python scripts/train_moe_model.py \
    --epochs 30 \
    --hidden_dim 96 \
    --learning_rate 0.003 \
    --dropout 0.3 \
    --weight_decay 1e-4 \
    --train_months 24 \
    --batch_size 1024
```

### 使用调优结果

```bash
# 1. 运行调优
python scripts/tune_moe_hyperparams.py --method random --max_trials 30

# 2. 查看最佳参数（保存在outputs/moe_tuning/）
cat outputs/moe_tuning/best_params_*.json

# 3. 使用最佳参数训练
python scripts/train_moe_model.py \
    --hidden_dim 96 \
    --learning_rate 0.003 \
    --dropout 0.25 \
    --epochs 30
```

---

## 2️⃣ 特征工程优化

### 当前特征

```
个股特征: 23个（动量、波动率、偏度、量价相关性）
环境特征: 6个（指数收益、波动率、截面离散度）
```

### 优化方向

#### A. 增加Alpha因子

项目已有WorldQuant风格Alpha因子，可以集成：

```python
# 在moe_data_loader.py中添加
from src.features.alpha_factors import AlphaFactors

def build_alpha_features(self, stock_df):
    """添加Alpha因子"""
    alpha_calc = AlphaFactors()
    
    # 计算10个Alpha因子
    alphas = alpha_calc.calculate_all(stock_df)
    
    return alphas  # alpha_001, alpha_006, alpha_053, ...
```

**预期提升**: IC +0.005 ~ 0.01

#### B. 使用项目的特征工程模块（30→160特征扩展）

```python
from src.features.engineering import FeatureEngineer

# 在build_moe_dataset中添加
engineer = FeatureEngineer('configs/feature_config.yaml')
expanded_features = engineer.transform(stock_features)

# 这会将30个基础因子扩展到160+特征
# 包括：交叉特征、时序导数、非线性变换等
```

**预期提升**: IC +0.01 ~ 0.02

#### C. 添加基本面特征

```python
def build_fundamental_features(self, stock_df):
    """添加基本面特征"""
    # PE, PB, ROE, 营收增长率等
    # 需要从数据库或Parquet加载财务数据
    pass
```

**预期提升**: IC +0.005 ~ 0.01

#### D. 行业中性化

```python
def industry_neutralize(self, features, industry_mapping):
    """行业中性化处理"""
    for col in feature_cols:
        features[col] = features.groupby('industry')[col].transform(
            lambda x: x - x.mean()
        )
    return features
```

**预期提升**: 降低行业暴露，提高稳定性

---

## 3️⃣ 架构调整

### A. 增加专家数量

当前：2个专家（Stock + Regime）

```python
# 修改moe_model.py，增加第三个专家

class EnhancedMoE(nn.Module):
    def __init__(self, stock_dim, regime_dim, fundamental_dim):
        # Stock Expert
        self.stock_expert = build_expert(stock_dim)
        
        # Regime Expert
        self.regime_expert = build_expert(regime_dim)
        
        # Fundamental Expert (新增)
        self.fundamental_expert = build_expert(fundamental_dim)
        
        # Gating Network (输出3个权重)
        self.gating = nn.Sequential(
            nn.Linear(stock_dim + regime_dim + fundamental_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3),  # 3个专家
            nn.Softmax(dim=1)
        )
```

**预期提升**: IC +0.005 ~ 0.01

### B. 添加注意力机制

```python
class AttentionMoE(nn.Module):
    def __init__(self, stock_dim, regime_dim):
        super().__init__()
        
        # Self-attention on stock features
        self.attention = nn.MultiheadAttention(
            embed_dim=stock_dim,
            num_heads=4
        )
        
        # 其余同原始MoE
```

**预期提升**: IC +0.003 ~ 0.008

### C. 残差连接

```python
class ResidualMoE(nn.Module):
    def forward(self, stock_x, regime_x):
        # Expert predictions
        stock_pred = self.stock_expert(stock_x)
        regime_pred = self.regime_expert(regime_x)
        
        # Gating
        weights = self.gating(torch.cat([stock_x, regime_x], dim=1))
        
        # 加权融合 + 残差连接
        moe_output = weights[:, 0:1] * stock_pred + weights[:, 1:2] * regime_pred
        residual = self.residual_net(stock_x)  # 简单线性层
        
        return moe_output + 0.1 * residual  # 残差权重
```

**预期提升**: IC +0.002 ~ 0.005

---

## 4️⃣ 训练策略优化

### A. 学习率调度

```python
# 在train_moe_model.py中添加
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

# 余弦退火
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

# 或者：动态降低学习率
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3)

# 在训练循环中
for epoch in range(num_epochs):
    train_loss = train_one_epoch()
    scheduler.step()  # 或 scheduler.step(val_loss)
```

**预期提升**: IC +0.002 ~ 0.005，训练更稳定

### B. 早停（Early Stopping）

```python
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_ic = -float('inf')
        self.counter = 0
        
    def should_stop(self, val_ic):
        if val_ic > self.best_ic + self.min_delta:
            self.best_ic = val_ic
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

# 使用
early_stopping = EarlyStopping(patience=5)
for epoch in range(num_epochs):
    val_ic = evaluate()
    if early_stopping.should_stop(val_ic):
        print(f"Early stopping at epoch {epoch}")
        break
```

**预期提升**: 防止过拟合，节省训练时间

### C. 增加验证集监控

```python
# 在train_moe_model.py中
# 当前只在测试集评估，应该加入验证集

def split_data_with_validation(full_data, train_months, val_months, test_months):
    """训练/验证/测试三分"""
    # train_months → val_months → test_months
    pass

# 每轮训练后在验证集评估，选择最佳模型
```

**预期提升**: 更好的模型选择，避免过拟合

### D. 梯度裁剪

```python
# 在MoEAlphaTrainer.train()中添加
torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
```

**预期提升**: 训练更稳定

---

## 5️⃣ 集成学习

### A. 时间集成（多个训练窗口平均）

```python
# 训练多个模型，每个使用不同的训练窗口
predictions_list = []

for offset in [0, 3, 6]:  # 3个不同起点
    model = train_with_offset(offset_months=offset)
    preds = model.predict(test_data)
    predictions_list.append(preds)

# 平均预测
final_predictions = np.mean(predictions_list, axis=0)
```

**预期提升**: IC +0.005 ~ 0.015，稳定性显著提升

### B. 模型集成（MoE + LSTM + LGBM）

```python
# 训练三种模型
moe_preds = train_moe_model()
lstm_preds = train_lstm_model()
lgbm_preds = train_lgbm_model()

# 加权平均（权重可通过验证集优化）
final_preds = 0.5 * moe_preds + 0.3 * lstm_preds + 0.2 * lgbm_preds
```

**预期提升**: IC +0.01 ~ 0.03

### C. Stacking集成

```python
# 第一层：基础模型
base_predictions = {
    'moe': moe_model.predict(X),
    'lstm': lstm_model.predict(X),
    'lgbm': lgbm_model.predict(X)
}

# 第二层：元模型（用验证集训练）
meta_features = np.column_stack([base_predictions[k] for k in base_predictions])
meta_model = LinearRegression()
meta_model.fit(meta_features_val, y_val)

# 最终预测
final_preds = meta_model.predict(meta_features_test)
```

**预期提升**: IC +0.015 ~ 0.04

---

## 6️⃣ Beta窗口优化

### 当前：固定60日窗口

### 优化：自适应Beta窗口

```python
def calculate_adaptive_beta(stock_returns, index_returns):
    """根据市场状态自适应调整Beta窗口"""
    
    # 市场波动低：使用更长窗口（更稳定）
    # 市场波动高：使用更短窗口（更及时）
    
    recent_vol = index_returns.rolling(20).std().iloc[-1]
    historical_vol = index_returns.rolling(252).std().iloc[-1]
    
    if recent_vol > 1.5 * historical_vol:
        window = 30  # 高波动，短窗口
    elif recent_vol < 0.7 * historical_vol:
        window = 120  # 低波动，长窗口
    else:
        window = 60  # 正常窗口
    
    return calculate_beta(stock_returns, index_returns, window)
```

**预期提升**: IC +0.003 ~ 0.008

---

## 7️⃣ 诊断与调试

### A. IC时序分析

```python
import matplotlib.pyplot as plt

# 绘制IC时序图
ic_series = predictions_df.groupby('factor_date').apply(
    lambda x: x['pred_alpha'].corr(x['residual_return'], method='spearman')
)

plt.figure(figsize=(12, 4))
plt.plot(ic_series.index, ic_series.values)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Rank IC Time Series')
plt.xlabel('Date')
plt.ylabel('IC')
plt.show()

# 找出IC异常低的时期
low_ic_periods = ic_series[ic_series < -0.1]
print("IC异常低的时期:", low_ic_periods)
```

### B. 特征重要性分析

```python
# 使用SHAP或简单的梯度分析
import shap

explainer = shap.DeepExplainer(model, background_data)
shap_values = explainer.shap_values(test_data)

# 可视化
shap.summary_plot(shap_values, test_data)
```

### C. 预测分布分析

```python
# 检查预测值分布
plt.hist(predictions, bins=50)
plt.title('Prediction Distribution')
plt.xlabel('Predicted Alpha')
plt.ylabel('Frequency')

# 检查是否存在异常值
print(f"预测值范围: [{predictions.min():.4f}, {predictions.max():.4f}]")
print(f"预测值均值: {predictions.mean():.4f}")
print(f"预测值标准差: {predictions.std():.4f}")
```

### D. 分组回测

```python
# 按预测分位数分组
predictions_df['pred_quantile'] = pd.qcut(
    predictions_df['pred_alpha'], 
    q=5, 
    labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
)

# 各分位数的平均收益
for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
    group = predictions_df[predictions_df['pred_quantile'] == q]
    mean_return = group['residual_return'].mean()
    print(f"{q}: {mean_return:.4f}")

# 多空组合收益（Q5 - Q1）
long_short = predictions_df[predictions_df['pred_quantile'] == 'Q5']['residual_return'].mean() - \
             predictions_df[predictions_df['pred_quantile'] == 'Q1']['residual_return'].mean()
print(f"Long-Short Return: {long_short:.4f}")
```

---

## 📊 调优优先级建议

### 🔥 高优先级（立即执行）

1. **增加训练轮数**: `--epochs 30` 
   - 成本：低（只需更长时间）
   - 收益：高（IC可能翻倍）
   
2. **超参数搜索**: 运行 `tune_moe_hyperparams.py`
   - 成本：中（需要几小时）
   - 收益：高（找到最优配置）

3. **增加训练窗口**: `--train_months 24`
   - 成本：低
   - 收益：中（更多数据，更稳定）

### ⭐ 中优先级（本周完成）

4. **集成Alpha因子**: 集成项目已有的10个WorldQuant因子
   - 成本：中（需要修改代码）
   - 收益：高

5. **学习率调度**: 添加CosineAnnealingLR
   - 成本：低
   - 收益：中

6. **早停机制**: 防止过拟合
   - 成本：低
   - 收益：中

### 💡 低优先级（长期优化）

7. **特征工程扩展**: 使用FeatureEngineer（30→160特征）
   - 成本：高（特征太多可能过拟合）
   - 收益：中-高（需要谨慎）

8. **模型集成**: MoE + LSTM + LGBM
   - 成本：高（需要训练多个模型）
   - 收益：高

9. **架构改进**: 增加专家、注意力机制
   - 成本：高（需要重新设计）
   - 收益：中-高

---

## 🎯 推荐调优路线图

### 第1周：基础优化

```bash
# Day 1-2: 增加训练轮数
python scripts/train_moe_model.py --epochs 30

# Day 3-4: 超参数搜索
python scripts/tune_moe_hyperparams.py --method random --max_trials 30

# Day 5: 使用最佳参数重新训练
python scripts/train_moe_model.py \
    --epochs 30 \
    --hidden_dim <best> \
    --learning_rate <best> \
    --dropout <best>
```

**预期结果**: IC从0.006提升至0.02-0.025

### 第2周：特征增强

```bash
# 集成Alpha因子和更多特征
# 修改moe_data_loader.py添加Alpha因子
python scripts/train_moe_model.py --epochs 30 <最佳超参数>
```

**预期结果**: IC提升至0.025-0.03

### 第3周：模型集成

```bash
# 训练LSTM模型
python scripts/train_alpha_model.py --model lstm --epochs 100

# 训练LGBM模型  
python scripts/train_alpha_model.py --model lgbm

# 集成预测
python scripts/ensemble_models.py --models moe,lstm,lgbm
```

**预期结果**: IC提升至0.03-0.04

### 第4周：生产优化

- 添加在线预测API
- 优化推理速度
- 添加监控和告警
- 回测验证

---

## 📈 性能追踪

创建一个调优日志，记录每次改进：

```
日期       | 改动                | IC    | ICIR | 备注
-----------|--------------------|---------|---------
2024-06-10 | 基线（3轮）         | 0.006 | 0.04 | 初始版本
2024-06-11 | 增加到30轮          | 0.022 | 0.35 | 显著提升
2024-06-12 | hidden_dim=96       | 0.025 | 0.40 | 小幅提升
2024-06-13 | 添加Alpha因子       | 0.032 | 0.55 | 达到目标
2024-06-14 | 模型集成            | 0.038 | 0.68 | 超过目标
```

---

## 🔧 常见问题

### Q1: IC不稳定，波动很大？
**A**: 
- 增加训练数据量（`--train_months 24`）
- 增加dropout（`--dropout 0.4`）
- 使用模型集成
- 检查是否有数据泄漏

### Q2: 训练很慢？
**A**:
- 使用GPU（`--device cuda`）
- 增大batch_size（`--batch_size 2048`）
- 减少训练窗口数（`--train_months 12`）
- 使用更小的模型（`--hidden_dim 32`）

### Q3: IC为负值？
**A**:
- 检查特征标准化是否正确
- 检查Beta剥离是否正确
- 尝试反向预测（`-pred_alpha`）
- 检查是否有数据对齐问题

### Q4: 过拟合（训练IC高，测试IC低）？
**A**:
- 增加dropout
- 增加weight_decay
- 减少hidden_dim
- 使用更多训练数据
- 添加早停

---

## 📚 参考资源

- MLflow文档: https://mlflow.org/docs/latest/index.html
- PyTorch调优: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
- WorldQuant Alpha因子: 项目的 `src/features/alpha_factors.py`
- 特征工程: 项目的 `src/features/engineering.py`

---

**创建日期**: 2024-06-10  
**适用版本**: MoE Model v1.0  
**维护者**: AI-Bisai Team
