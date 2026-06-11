# XGBoost调参系统 - 命令速查表

## 🚀 快速开始

### 1. 测试系统
```bash
cd star50-quant
python scripts/test_tuning_system.py
```

### 2. 准备数据
```bash
python scripts/calculate_factors.py --start_date 2020-01-01 --end_date 2024-12-31
```

### 3. 开始调参
```bash
# 推荐：贝叶斯优化
python scripts/tune_xgb_model.py --method bayesian --n_iter 50

# 快速探索：随机搜索
python scripts/tune_xgb_model.py --method random --n_iter 100

# 精确搜索：网格搜索
python scripts/tune_xgb_model.py --method grid
```

---

## 📋 常用命令

### 基本调参
```bash
# 贝叶斯优化（默认50次试验）
python scripts/tune_xgb_model.py --method bayesian --n_iter 50

# 随机搜索（100次试验）
python scripts/tune_xgb_model.py --method random --n_iter 100

# 网格搜索（遍历所有组合）
python scripts/tune_xgb_model.py --method grid
```

### 自定义数据范围
```bash
# 使用2021-2024年数据
python scripts/tune_xgb_model.py \
    --method bayesian \
    --start_date 2021-01-01 \
    --end_date 2024-12-31
```

### 调整交叉验证
```bash
# 使用3折交叉验证（更快）
python scripts/tune_xgb_model.py \
    --method bayesian \
    --cv_folds 3

# 使用5折交叉验证（更稳定）
python scripts/tune_xgb_model.py \
    --method bayesian \
    --cv_folds 5
```

### 自定义目标权重
```bash
# 更重视IC（预测能力）
python scripts/tune_xgb_model.py \
    --method bayesian \
    --ic_weight 0.5 \
    --ir_weight 0.3 \
    --return_weight 0.15 \
    --drawdown_weight 0.05

# 更重视收益
python scripts/tune_xgb_model.py \
    --method bayesian \
    --ic_weight 0.3 \
    --ir_weight 0.2 \
    --return_weight 0.4 \
    --drawdown_weight 0.1
```

### 指定输出目录
```bash
python scripts/tune_xgb_model.py \
    --method bayesian \
    --output_dir tuning_results/xgb_v1
```

### 完整配置示例
```bash
python scripts/tune_xgb_model.py \
    --method bayesian \
    --n_iter 100 \
    --start_date 2021-01-01 \
    --end_date 2024-12-31 \
    --cv_folds 5 \
    --ic_weight 0.4 \
    --ir_weight 0.3 \
    --return_weight 0.2 \
    --drawdown_weight 0.1 \
    --output_dir tuning_results/experiment_001
```

---

## 📊 查看结果

### 列出输出文件
```bash
ls -lh tuning_results/
```

### 查看最佳参数
```bash
cat tuning_results/xgb_tuning_bayesian_*.json | python -m json.tool
```

### 查看试验历史
```bash
# 安装csvkit（如果需要）
pip install csvkit

# 查看CSV
csvlook tuning_results/xgb_tuning_bayesian_*_trials.csv | less

# 或使用pandas
python -c "
import pandas as pd
df = pd.read_csv('tuning_results/xgb_tuning_bayesian_*_trials.csv')
print(df[['ic', 'ir', 'annual_return', 'max_drawdown', 'composite_score']].describe())
"
```

### 查看特征重要性
```bash
head -20 tuning_results/xgb_feature_importance_*.csv
```

---

## 🔧 故障排查

### 检查系统状态
```bash
python scripts/check_tuning_system.py
```

### 测试调参系统
```bash
python scripts/test_tuning_system.py
```

### 检查依赖
```bash
pip list | grep -E "xgboost|optuna|sklearn|pandas|numpy"
```

### 安装缺失依赖
```bash
pip install -r requirements.txt
```

### 重新安装特定包
```bash
pip install --upgrade xgboost optuna
```

---

## 📈 三阶段调参策略

### 阶段1: 粗调（1-2小时）
```bash
# 快速探索50次
python scripts/tune_xgb_model.py --method random --n_iter 50
```

### 阶段2: 精调（2-4小时）
```bash
# 贝叶斯优化100次
python scripts/tune_xgb_model.py --method bayesian --n_iter 100
```

### 阶段3: 验证
```bash
# 完整回测
python scripts/run_backtest.py \
    --model_path tuning_results/xgb_best_model_*.json
```

---

## 🎯 目标

调参目标（可通过权重调整优先级）：
- **IC > 0.04**: 信息系数（预测能力）
- **IR >= 1.5**: 信息比率（风险调整收益）
- **年化收益 > 35%**: 策略盈利能力
- **最大回撤 <= 20%**: 风险控制

---

## 📚 文档链接

- **快速开始**: `docs/README-XGBoost-Tuning.md`
- **详细指南**: `docs/xgboost-tuning-guide.md`
- **技术总结**: `docs/xgboost-tuning-summary.md`
- **交付报告**: `docs/DELIVERY-SUMMARY.md`

---

## 💡 性能优化

### 加快调参速度
```bash
# 方法1: 减少CV折数
python scripts/tune_xgb_model.py --method bayesian --cv_folds 3

# 方法2: 使用更少数据
python scripts/tune_xgb_model.py --method bayesian --start_date 2023-01-01

# 方法3: 减少迭代次数
python scripts/tune_xgb_model.py --method random --n_iter 30
```

### 提高调参精度
```bash
# 方法1: 增加迭代次数
python scripts/tune_xgb_model.py --method bayesian --n_iter 150

# 方法2: 增加CV折数
python scripts/tune_xgb_model.py --method bayesian --cv_folds 10

# 方法3: 使用更多数据
python scripts/tune_xgb_model.py --method bayesian --start_date 2019-01-01
```

---

## 🆘 帮助

### 查看所有参数
```bash
python scripts/tune_xgb_model.py --help
```

### 常见错误

**错误**: "数据加载失败"  
**解决**: 先运行 `python scripts/calculate_factors.py`

**错误**: "No module named 'optuna'"  
**解决**: 运行 `pip install optuna`

**错误**: "调参时间太长"  
**解决**: 减少 `--cv_folds` 或 `--n_iter`

**错误**: "目标无法达成"  
**解决**: 增加 `--n_iter` 或调整目标权重

---

**提示**: 将此文件保存为书签，方便随时查阅！
