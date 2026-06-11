# XGBoost调参系统 - 完整流程化指南

## 📋 总览

```
┌─────────────────────────────────────────────────────────────┐
│                   量化策略开发完整流程                        │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
   [数据准备]          [模型调参]          [策略优化]
     阶段0              阶段1-2             阶段3-4
        │                   │                   │
        └───────────────────┴───────────────────┘
                            │
                       [生产部署]
                        阶段5
```

---

## 🎯 目标设定

### 最终目标
- ✅ IC > 0.04 (信息系数，预测能力)
- ⏳ IR >= 1.5 (信息比率，风险调整收益)
- ⏳ 年化收益 > 35%
- ⏳ 最大回撤 <= 20%

### 阶段目标分解
| 阶段 | 主要目标 | 评估指标 | 预期结果 |
|------|----------|----------|----------|
| 阶段0 | 数据就绪 | 数据质量 | 可用数据 |
| 阶段1 | 模型调参 | IC | IC > 0.08 |
| 阶段2 | 模型训练 | IC, 泛化 | 稳定模型 |
| 阶段3 | 策略回测 | IR, 收益, 回撤 | 达标策略 |
| 阶段4 | 策略优化 | 全部指标 | 所有达标 |
| 阶段5 | 生产部署 | 实盘表现 | 稳定运行 |

---

## 阶段0: 数据准备 📊

### 目标
准备5年历史数据（2020-2024），包含价格、成交量、技术因子

### 流程

#### 步骤0.1: 检查数据库 ✓
```bash
# 检查数据库连接
python -c "
from src.models.data_loader import FactorDataLoader
loader = FactorDataLoader(db_name='star50_quant')
loader.connect()
print('✓ 数据库连接成功')
loader.close()
"
```

**检查点**: 数据库可连接

#### 步骤0.2: 检查原始数据 ✓
```bash
# 检查行情数据
python -c "
import psycopg2
import os
conn = psycopg2.connect(dbname='star50_quant', user=os.getenv('USER'), host='localhost')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*), MIN(trade_date), MAX(trade_date) FROM stock_daily')
count, min_date, max_date = cursor.fetchone()
print(f'行情数据: {count:,} 条')
print(f'日期范围: {min_date} 至 {max_date}')
conn.close()
"
```

**检查点**: 
- ✅ 有足够的历史数据（5年）
- ✅ 覆盖50只股票

#### 步骤0.3: 计算技术因子 ✓
```bash
# 计算30个技术因子
python scripts/calculate_factors.py \
    --all \
    --start 2020-01-02 \
    --end 2024-12-31
```

**耗时**: ~10分钟  
**检查点**: 
- ✅ factor_values表有120万+条记录
- ✅ 30个技术因子完整

**完成标志**: 
```
✓ All done! Processed 50 stocks
```

---

## 阶段1: 模型调参（IC优化）🔧

### 目标
找到能最大化IC的XGBoost参数

### 原理
- **只关注IC**（预测能力），不管策略收益
- 使用时间序列交叉验证
- 贝叶斯优化智能搜索参数空间

### 流程

#### 步骤1.1: 快速验证（10次，5分钟）
```bash
# 用2024年数据快速测试
python scripts/tune_xgb_model.py \
    --method bayesian \
    --n_iter 10 \
    --start_date 2024-01-01 \
    --end_date 2024-12-31 \
    --cv_folds 3 \
    --ic_weight 1.0 \
    --ir_weight 0.0 \
    --return_weight 0.0 \
    --drawdown_weight 0.0 \
    --output_dir tuning_results/quick_test
```

**检查点**: 
- ✅ 系统运行正常
- ✅ IC > 0.04

#### 步骤1.2: 完整调参（50次，15-20分钟）
```bash
# 用5年数据完整调参
python scripts/tune_xgb_model.py \
    --method bayesian \
    --n_iter 50 \
    --start_date 2020-01-02 \
    --end_date 2024-12-31 \
    --cv_folds 5 \
    --ic_weight 1.0 \
    --ir_weight 0.0 \
    --return_weight 0.0 \
    --drawdown_weight 0.0 \
    --output_dir tuning_results/ic_optimization
```

**检查点**: 
- ✅ IC > 0.08（优秀水平）
- ✅ 参数重要性分析合理

**完成标志**:
```
✓ 模型调参完成
  最佳IC: 0.0953
```

#### 步骤1.3: 分析结果
```bash
# 查看最佳参数
cat tuning_results/ic_optimization/xgb_tuning_bayesian_*.json

# 查看试验历史
python -c "
import pandas as pd
trials = pd.read_csv('tuning_results/ic_optimization/xgb_tuning_bayesian_*_trials.csv')
print('IC统计:')
print(trials['ic'].describe())
print(f'\nTop 5 IC:')
print(trials.nlargest(5, 'ic')[['ic']])
"

# 查看特征重要性
head -20 tuning_results/ic_optimization/xgb_feature_importance_*.csv
```

**决策点**: IC是否足够高？
- ✅ IC > 0.08 → 继续阶段2
- ❌ IC < 0.08 → 优化特征工程后重试

---

## 阶段2: 模型训练（最终模型）🎓

### 目标
用最佳参数训练最终模型

### 流程

#### 步骤2.1: 训练最终模型
```python
# 在Python中执行
from src.models.xgb_model import XGBoostAlphaModel
from src.models.data_loader import FactorDataLoader
import json

# 加载最佳参数
with open('tuning_results/ic_optimization/xgb_tuning_bayesian_*.json') as f:
    result = json.load(f)
    best_params = result['best_params']

# 加载数据
loader = FactorDataLoader(db_name='star50_quant')
loader.connect()
# ... 数据加载和预处理 ...

# 训练模型
model = XGBoostAlphaModel(params=best_params)
model.train(train_features, train_labels, 
            val_features, val_labels,
            num_boost_round=best_params['num_boost_round'],
            early_stopping_rounds=20)

# 保存模型
model.save('models/xgboost_final_model.json')
```

#### 步骤2.2: 验证模型
```python
# 验证集评估
predictions = model.predict(val_features)
ic = np.corrcoef(predictions, val_labels['forward_return'])[0, 1]
print(f'验证集IC: {ic:.4f}')
```

**检查点**: 
- ✅ 验证集IC接近训练集IC（不过拟合）
- ✅ 特征重要性合理

**完成标志**:
```
✓ 模型训练完成
  验证集IC: 0.0931
```

---

## 阶段3: 完整回测（真实评估）📈

### 目标
在完整的回测框架中评估真实的IR、收益、回撤

### 重要性
⚠️ **关键阶段** - 这里才能得到真实的策略表现

### 流程

#### 步骤3.1: 检查回测框架
```bash
# 检查是否有现成的回测脚本
ls scripts/run_backtest.py

# 如果没有，需要先实现
```

**如果没有回测框架**: 需要实现完整的回测引擎（见附录A）

#### 步骤3.2: 运行基准回测
```bash
# 使用默认策略配置
python scripts/run_backtest.py \
    --model_path models/xgboost_final_model.json \
    --start_date 2020-01-02 \
    --end_date 2024-12-31 \
    --strategy_config configs/strategy_baseline.yaml \
    --output_dir backtest_results/baseline
```

**strategy_baseline.yaml**:
```yaml
selection:
  method: top_pct
  pct: 0.2          # 选择top 20%

weighting:
  method: equal     # 等权配置

rebalance:
  frequency: daily  # 每日再平衡

costs:
  commission: 0.0003  # 单边手续费0.03%
  slippage: 0.0002    # 滑点0.02%

constraints:
  max_position: 0.1   # 单只股票最大10%
  min_position: 0.01  # 最小1%
```

#### 步骤3.3: 分析回测结果
```python
import pandas as pd

# 加载回测结果
results = pd.read_csv('backtest_results/baseline/performance_metrics.csv')

print('回测结果:')
print(f"IC:        {results['ic'].mean():.4f}")
print(f"IR:        {results['ir'][0]:.2f}")
print(f"年化收益:  {results['annual_return'][0]:.2%}")
print(f"最大回撤:  {results['max_drawdown'][0]:.2%}")
print(f"夏普比率: {results['sharpe_ratio'][0]:.2f}")
```

**决策点**: 是否达标？
- ✅ 所有指标达标 → 阶段5（部署）
- ⏳ 部分未达标 → 阶段4（优化）
- ❌ 全部未达标 → 回到阶段1（重新调参或特征工程）

---

## 阶段4: 策略优化（参数调优）⚙️

### 目标
通过优化策略参数达到所有目标

### 流程

#### 步骤4.1: 定义参数网格
```yaml
# configs/strategy_param_grid.yaml
selection_pct: [0.10, 0.15, 0.20, 0.25, 0.30]
weighting_method: ['equal', 'signal', 'signal_squared', 'risk_adjusted']
rebalance_frequency: ['daily', 'weekly', 'monthly']
```

#### 步骤4.2: 网格搜索
```bash
# 遍历所有策略配置
python scripts/optimize_strategy_params.py \
    --model_path models/xgboost_final_model.json \
    --param_grid configs/strategy_param_grid.yaml \
    --backtest_period 2020-01-02:2024-12-31 \
    --output_dir strategy_optimization
```

**耗时**: 根据组合数（5×4×3=60种配置）

#### 步骤4.3: 分析最佳策略
```python
# 加载所有结果
results = pd.read_csv('strategy_optimization/all_results.csv')

# 综合评分
results['composite_score'] = (
    (results['ic'] / 0.04) * 0.2 +
    (results['ir'] / 1.5) * 0.3 +
    (results['annual_return'] / 0.35) * 0.3 +
    ((1 + results['max_drawdown'] / 0.2).clip(0, 1)) * 0.2
)

# 最佳策略
best = results.loc[results['composite_score'].idxmax()]

print('最佳策略配置:')
print(f"选股比例: {best['selection_pct']:.0%}")
print(f"权重方式: {best['weighting_method']}")
print(f"再平衡:   {best['rebalance_frequency']}")
print()
print('表现:')
print(f"IC:       {best['ic']:.4f}")
print(f"IR:       {best['ir']:.2f}")
print(f"年化:     {best['annual_return']:.2%}")
print(f"回撤:     {best['max_drawdown']:.2%}")
```

**决策点**: 是否达标？
- ✅ 达标 → 阶段5
- ⏳ 接近 → 步骤4.4（进阶优化）
- ❌ 差距大 → 考虑模型集成或组合优化

#### 步骤4.4: 进阶优化（可选）

**A. 组合优化**
```bash
python scripts/optimize_portfolio.py \
    --alpha_predictions strategy_optimization/best_predictions.csv \
    --risk_model data/risk_covariance.csv \
    --method mean_variance \
    --target_return 0.35 \
    --max_drawdown 0.20
```

**B. 模型集成**
```bash
# 训练多个模型
python scripts/train_alpha_model.py --model lstm
python scripts/train_alpha_model.py --model lgbm

# 集成
python scripts/ensemble_models.py \
    --models xgboost,lstm,lgbm \
    --method weighted_average \
    --weights 0.5,0.3,0.2
```

**C. 风险控制**
```yaml
# configs/risk_control.yaml
stop_loss:
  enabled: true
  threshold: -0.05  # 单股跌5%止损

position_limits:
  max_single: 0.08   # 单股最大8%
  max_industry: 0.30  # 单行业最大30%

volatility_scaling:
  enabled: true
  target_vol: 0.15    # 目标年化波动15%
```

---

## 阶段5: 生产部署（实盘）🚀

### 目标
将策略部署到生产环境

### 流程

#### 步骤5.1: 样本外测试
```bash
# 用最新数据测试（2025年）
python scripts/run_backtest.py \
    --model_path models/xgboost_final_model.json \
    --strategy_config strategy_optimization/best_config.yaml \
    --start_date 2025-01-01 \
    --end_date 2025-03-31 \
    --output_dir out_of_sample_test
```

**检查点**: 样本外表现不能严重衰减

#### 步骤5.2: 实盘模拟
```bash
# 纸上交易1-2个月
python scripts/paper_trading.py \
    --model_path models/xgboost_final_model.json \
    --strategy_config strategy_optimization/best_config.yaml \
    --duration 60  # 60天
```

#### 步骤5.3: 小资金实盘
```bash
# 用小资金真实交易
python scripts/live_trading.py \
    --model_path models/xgboost_final_model.json \
    --strategy_config strategy_optimization/best_config.yaml \
    --capital 100000  # 10万测试
    --max_loss 5000   # 最大损失5千
```

#### 步骤5.4: 监控和维护
```bash
# 每日监控
python scripts/monitor_performance.py --daily

# 每周报告
python scripts/generate_report.py --weekly

# 每月模型更新
python scripts/retrain_model.py --monthly
```

---

## 📊 完整流程图

```
[开始]
   │
   ▼
┌─────────────────┐
│  阶段0: 数据准备  │  ← 检查数据库、计算因子
└────────┬────────┘
         │ ✓ 120万+因子记录
         ▼
┌─────────────────┐
│ 阶段1: 模型调参  │  ← 贝叶斯优化，50次试验
│  (只看IC)       │
└────────┬────────┘
         │ ✓ IC > 0.08
         ▼
┌─────────────────┐
│ 阶段2: 模型训练  │  ← 用最佳参数训练最终模型
└────────┬────────┘
         │ ✓ 验证IC稳定
         ▼
┌─────────────────┐
│ 阶段3: 完整回测  │  ← 真实评估IR/收益/回撤
└────────┬────────┘
         │
    ┌────┴────┐
    │ 是否达标？│
    └────┬────┘
         │
    ┌────┼────┐
    ▼    │    ▼
   是    │   否
    │    │    │
    │    │    ▼
    │    │ ┌─────────────────┐
    │    │ │ 阶段4: 策略优化  │ ← 网格搜索策略参数
    │    │ │  - 选股比例      │
    │    │ │  - 权重方式      │
    │    │ │  - 再平衡频率    │
    │    │ └────────┬────────┘
    │    │          │ ✓ 所有指标达标
    │    └──────────┘
    │
    ▼
┌─────────────────┐
│ 阶段5: 生产部署  │  ← 样本外测试→纸上交易→实盘
└────────┬────────┘
         │
         ▼
    [持续监控]
```

---

## 🎯 关键决策点

### 决策点1: 数据是否就绪？（阶段0→1）
- ✅ 是 → 继续
- ❌ 否 → 补充数据采集

### 决策点2: IC是否足够高？（阶段1→2）
- ✅ IC > 0.08 → 继续
- ⏳ 0.04 < IC < 0.08 → 可继续，但预期收益较低
- ❌ IC < 0.04 → 优化特征工程

### 决策点3: 回测是否达标？（阶段3→4/5）
- ✅ 全部达标 → 跳到阶段5
- ⏳ 部分达标 → 阶段4优化
- ❌ 全部未达标 → 回到阶段1

### 决策点4: 策略优化后是否达标？（阶段4→5）
- ✅ 达标 → 阶段5
- ⏳ 接近 → 进阶优化（组合优化/模型集成）
- ❌ 差距大 → 重新评估方法论

---

## 📋 检查清单

### 阶段0完成检查
- [ ] 数据库连接正常
- [ ] stock_daily表有5年数据
- [ ] factor_values表有120万+记录
- [ ] 30个技术因子完整

### 阶段1完成检查
- [ ] 快速验证IC > 0.04
- [ ] 完整调参IC > 0.08
- [ ] 参数重要性分析合理
- [ ] 特征重要性已保存

### 阶段2完成检查
- [ ] 最终模型已训练
- [ ] 验证集IC稳定
- [ ] 模型文件已保存
- [ ] 无明显过拟合

### 阶段3完成检查
- [ ] 回测框架可用
- [ ] 基准回测已运行
- [ ] 所有指标已计算
- [ ] 决策：是否需要阶段4

### 阶段4完成检查
- [ ] 参数网格已定义
- [ ] 所有配置已测试
- [ ] 最佳策略已确定
- [ ] 所有目标已达成

### 阶段5完成检查
- [ ] 样本外测试通过
- [ ] 纸上交易正常
- [ ] 小资金实盘运行
- [ ] 监控系统上线

---

## ⏱️ 时间预估

| 阶段 | 任务 | 预计时间 | 实际时间 |
|------|------|----------|----------|
| **阶段0** | 数据准备 | 1小时 | ✅ 已完成 |
| **阶段1** | 模型调参 | 0.5小时 | ✅ 已完成 |
| **阶段2** | 模型训练 | 0.2小时 | ✅ 已完成 |
| **阶段3** | 完整回测 | 1-2小时 | ⏳ 待实施 |
| **阶段4** | 策略优化 | 3-5小时 | ⏳ 待实施 |
| **阶段5** | 生产部署 | 1-2周 | ⏳ 待实施 |

**总计**: 约5-8小时开发 + 1-2周部署

---

## 💡 最佳实践

### 1. 增量迭代
不要一次性追求完美，先跑通基本流程，再逐步优化

### 2. 版本管理
每个阶段保存结果和配置，便于回溯

### 3. 详细记录
记录每次实验的配置和结果，建立实验日志

### 4. 风险控制
在每个阶段设置检查点，避免错误传播

### 5. 自动化
尽可能自动化重复性工作（调参、回测、报告）

---

## 📚 附录

### 附录A: 实现回测框架

如果没有现成的回测框架，需要实现：

```python
# src/backtest/backtest_engine.py

class BacktestEngine:
    def __init__(self, model, strategy_config):
        self.model = model
        self.config = strategy_config
    
    def run(self, start_date, end_date):
        # 1. 加载数据
        data = self.load_data(start_date, end_date)
        
        # 2. 每日循环
        for date in self.trading_days:
            # 2.1 生成信号
            signals = self.model.predict(data[date])
            
            # 2.2 选股
            selected = self.select_stocks(signals)
            
            # 2.3 计算权重
            weights = self.calculate_weights(selected)
            
            # 2.4 模拟交易
            self.execute_trades(selected, weights)
            
            # 2.5 更新组合
            self.update_portfolio(date)
        
        # 3. 计算绩效
        metrics = self.calculate_metrics()
        return metrics
```

### 附录B: 常见问题

**Q: 阶段1的IC很高，但阶段3的收益很低？**  
A: 正常，因为阶段1只评估预测能力，阶段3才评估策略执行。需要在阶段4优化策略。

**Q: 需要多少历史数据？**  
A: 至少3年，推荐5年以上。数据越多，模型越稳定。

**Q: 可以跳过某些阶段吗？**  
A: 不建议。每个阶段都有其目的，跳过可能导致问题。

**Q: 回测表现好，实盘表现差怎么办？**  
A: 检查过拟合、交易成本、市场环境变化等因素。

---

## 📞 快速入口

### 从头开始
```bash
# 1. 准备数据
python scripts/calculate_factors.py --all --start 2020-01-02 --end 2024-12-31

# 2. 模型调参
python scripts/tune_xgb_model.py --method bayesian --n_iter 50 \
    --ic_weight 1.0 --ir_weight 0.0 --return_weight 0.0 --drawdown_weight 0.0

# 3. 完整回测（需要先有回测框架）
python scripts/run_backtest.py --model_path models/xxx.json
```

### 继续未完成的工作
```bash
# 查看当前进度
cat docs/COMPLETE-CHECKLIST.md

# 从阶段3开始
python scripts/run_backtest.py --model_path tuning_results/full_run_fixed/xgb_best_model_*.json
```

---

**最后更新**: 2026-06-10  
**状态**: 阶段0-2已完成✅，阶段3-5待实施⏳  
**当前位置**: 需要实现或使用完整回测框架进入阶段3
