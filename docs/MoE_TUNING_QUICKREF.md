# 🎯 MoE模型调优 - 快速参考卡

## 📊 当前性能
```
基线（3轮训练）:
  Rank IC: 0.0059
  ICIR: 0.0445
  IC > 0占比: 50.70%
```

## 🎯 目标性能
```
目标: IC > 0.02, ICIR > 0.5
优秀: IC > 0.03, ICIR > 1.0
```

---

## ⚡ 快速开始（3种方式）

### 方式1️⃣：自动化调优（推荐）
```bash
# 一键运行完整调优流程（约2-3小时）
bash scripts/quick_tune_moe.sh

# 包含：基线测试 → 增加训练轮数 → 超参数搜索 → 最佳参数训练
```

### 方式2️⃣：手动超参数搜索
```bash
# 随机搜索30组超参数（约1-2小时）
python scripts/tune_moe_hyperparams.py --method random --max_trials 30

# 使用最佳参数训练
python scripts/train_moe_model.py \
    --hidden_dim 96 \
    --learning_rate 0.003 \
    --dropout 0.3 \
    --epochs 30
```

### 方式3️⃣：最简单的改进
```bash
# 只增加训练轮数（10分钟见效）
python scripts/train_moe_model.py --epochs 30

# 预期: IC从0.006提升至0.02+
```

---

## 🔥 优先级排序

### 🥇 最高优先级（立即执行，ROI最高）

#### 1. 增加训练轮数
```bash
python scripts/train_moe_model.py --epochs 30
```
- **成本**: ⭐ 低（只需更多时间）
- **收益**: ⭐⭐⭐⭐⭐ 极高（IC可能提升3-5倍）
- **预期**: IC 0.006 → 0.02-0.03

#### 2. 超参数搜索
```bash
python scripts/tune_moe_hyperparams.py --method random --max_trials 30
```
- **成本**: ⭐⭐ 中（需要1-2小时）
- **收益**: ⭐⭐⭐⭐ 高（找到最优配置）
- **预期**: IC +0.005 ~ 0.01

#### 3. 增加训练窗口
```bash
python scripts/train_moe_model.py --train_months 24
```
- **成本**: ⭐ 低
- **收益**: ⭐⭐⭐ 中-高（更稳定）
- **预期**: ICIR +0.1 ~ 0.2

---

### 🥈 中等优先级（本周完成）

#### 4. 集成Alpha因子
- 修改 `src/models/alpha/moe_data_loader.py`
- 添加项目已有的10个WorldQuant Alpha因子
- **预期**: IC +0.005 ~ 0.01

#### 5. 学习率调度
- 在训练循环添加 `CosineAnnealingLR`
- **预期**: IC +0.002 ~ 0.005

#### 6. 早停机制
- 添加EarlyStopping防止过拟合
- **预期**: 训练更稳定

---

### 🥉 低优先级（长期优化）

#### 7. 特征工程扩展（30→160特征）
- **成本**: ⭐⭐⭐ 高
- **收益**: ⭐⭐⭐⭐ 高（但需谨慎，可能过拟合）
- **预期**: IC +0.01 ~ 0.02

#### 8. 模型集成（MoE + LSTM + LGBM）
- **成本**: ⭐⭐⭐⭐ 很高
- **收益**: ⭐⭐⭐⭐⭐ 极高
- **预期**: IC +0.01 ~ 0.03

#### 9. 架构改进（增加专家、注意力机制）
- **成本**: ⭐⭐⭐⭐ 很高
- **收益**: ⭐⭐⭐ 中-高
- **预期**: IC +0.005 ~ 0.01

---

## 📈 推荐调优路线

### 第1天：基础优化
```bash
# 1. 增加训练轮数（30分钟）
python scripts/train_moe_model.py --epochs 30

# 2. 查看结果
tail -50 outputs/moe_predictions/predictions_*.csv
```
**预期结果**: IC 0.006 → 0.02

---

### 第2-3天：超参数调优
```bash
# 1. 运行超参数搜索（2小时）
python scripts/tune_moe_hyperparams.py --method random --max_trials 30

# 2. 查看最佳参数
cat outputs/moe_tuning/best_params_*.json

# 3. 使用最佳参数训练
python scripts/train_moe_model.py \
    --epochs 30 \
    --hidden_dim <best> \
    --learning_rate <best> \
    --dropout <best>
```
**预期结果**: IC 0.02 → 0.025-0.03

---

### 第1周：特征增强
- 集成Alpha因子
- 添加基本面特征
- 行业中性化

**预期结果**: IC 0.025 → 0.03+

---

### 第2周：模型集成
- 训练LSTM和LGBM模型
- 构建Ensemble
- 优化集成权重

**预期结果**: IC 0.03 → 0.035-0.04

---

## 🔍 关键超参数

| 参数 | 当前 | 推荐 | 说明 |
|------|------|------|------|
| `epochs` | 3 | **25-30** | ⭐⭐⭐ 最重要！ |
| `hidden_dim` | 64 | 64-128 | 模型容量 |
| `learning_rate` | 0.005 | 0.001-0.01 | 收敛速度 |
| `dropout` | 0.3 | 0.2-0.5 | 防过拟合 |
| `train_months` | 12 | 12-36 | 训练窗口 |
| `batch_size` | 1024 | 512-2048 | 训练稳定性 |

---

## 🐛 常见问题速查

### Q: IC不稳定，波动大？
```bash
# 增加训练数据和正则化
python scripts/train_moe_model.py \
    --train_months 24 \
    --dropout 0.4 \
    --weight_decay 1e-3
```

### Q: 训练太慢？
```bash
# 使用GPU + 更大batch
python scripts/train_moe_model.py \
    --device cuda \
    --batch_size 2048
```

### Q: IC为负？
- 检查特征标准化
- 检查数据对齐
- 尝试反向预测 `-pred_alpha`

### Q: 过拟合？
```bash
# 增加正则化
python scripts/train_moe_model.py \
    --dropout 0.5 \
    --weight_decay 1e-3 \
    --hidden_dim 32
```

---

## 📊 性能追踪模板

创建 `outputs/tuning_log.md`：

```markdown
| 日期 | 改动 | IC | ICIR | 备注 |
|------|------|-----|------|------|
| 06-10 | 基线（3轮） | 0.006 | 0.04 | 初始版本 |
| 06-11 | 30轮训练 | 0.022 | 0.35 | 显著提升 |
| 06-12 | 超参数调优 | 0.028 | 0.45 | 达到目标 |
| 06-13 | 添加Alpha因子 | 0.035 | 0.60 | 优秀水平 |
```

---

## 📚 完整文档

- **详细调优指南**: [docs/MoE_TUNING_GUIDE.md](../docs/MoE_TUNING_GUIDE.md)
- **使用说明**: [docs/MoE_PARQUET_README.md](../docs/MoE_PARQUET_README.md)
- **技术文档**: [docs/MoE_MODEL.md](../docs/MoE_MODEL.md)

---

## 🎯 30分钟快速提升方案

如果只有30分钟，按这个顺序做：

```bash
# 1. 增加训练轮数（15分钟）
python scripts/train_moe_model.py --epochs 30

# 2. 查看结果（5分钟）
tail -50 outputs/moe_predictions/predictions_*.csv | grep "Rank IC"

# 3. 如果IC>0.02，再增加训练窗口（10分钟）
python scripts/train_moe_model.py --epochs 30 --train_months 24
```

**预期**: IC从0.006提升至0.02+，ICIR从0.04提升至0.3+

---

**创建日期**: 2024-06-10  
**版本**: v1.0  
**维护**: AI-Bisai Team
