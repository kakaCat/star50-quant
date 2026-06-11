# 科创50指数增强策略 - 重构方案

**基于标准Walk-Forward框架的正确实现**

---

## 🎯 核心改进

### 与原方案的关键区别

| 维度 | 原方案（错误） | 新方案（正确） |
|------|--------------|--------------|
| **标签定义** | 绝对收益 | **超额收益**（个股 - 指数） |
| **训练方式** | 一次性训练 | **滚动固定窗口** |
| **数据使用** | 训练集=测试集 | 严格时序分离 |
| **预测能力** | IC = -0.009（失败） | 目标 IC > 0.03 |

---

## 📋 完整实施流程

### 第一阶段：数据准备与特征工程

#### 1.1 数据加载
```python
# 时间范围：2019-01-01 至 2024-12-31
# - 科创50成分股后复权日行情
# - 科创50指数日行情
# - 个股基本面数据（可选）
```

#### 1.2 基础因子构造

**量价因子**（30个基础因子已有）：
- 价格：MA5/10/20/60, EMA5/10/20, BOLL
- 动量：MACD, RSI14/28, CCI, KDJ
- 波动：ATR14, 历史波动率
- 成交量：成交量MA, OBV, 量价背离
- 市场微观：换手率、振幅

**相对指数因子**（新增）：
```python
# 1. 相对Beta
rolling_beta = cov(stock_return, index_return) / var(index_return)

# 2. 相对强度
relative_strength = stock_return - index_return  # 不同周期

# 3. 相对波动
relative_volatility = stock_vol / index_vol

# 4. 相对相关性
rolling_correlation = corr(stock_return, index_return)

# 5. 超额收益动量
excess_return_5d = stock_return_5d - index_return_5d
excess_return_20d = stock_return_20d - index_return_20d
```

#### 1.3 Alpha因子（已有10个WorldQuant风格因子）
```python
# alpha_001, alpha_006, alpha_053, alpha_054, alpha_101
# + 5个自定义因子
```

#### 1.4 特征扩展（已有30→160扩展）
保持现有特征工程，但添加相对指标。

---

### 第二阶段：标签构造（关键改进）

#### 2.1 超额收益标签

```python
def calculate_excess_return_label(prices, index_prices, forward_days=5):
    """
    构造超额收益标签
    
    关键：个股未来收益 - 指数未来收益
    """
    # 个股未来5日收益
    stock_forward_return = (prices.shift(-forward_days) / prices - 1)
    
    # 指数未来5日收益
    index_forward_return = (index_prices.shift(-forward_days) / index_prices - 1)
    
    # 超额收益 = 个股 - 指数
    excess_return = stock_forward_return - index_forward_return
    
    return excess_return
```

**优势**：
- ✅ 剥离市场Beta，只预测Alpha
- ✅ 天然适合指数增强策略
- ✅ 避免混淆择时和选股能力

---

### 第三阶段：因子预处理

#### 3.1 截面处理（每个交易日独立处理）

```python
def preprocess_factors_cross_sectional(data, factor_cols):
    """
    截面预处理（比时序标准化更重要）
    """
    processed = data.copy()
    
    for date in data['trade_date'].unique():
        date_mask = data['trade_date'] == date
        date_data = data.loc[date_mask, factor_cols]
        
        # 1. 缺失值填充（截面中位数）
        filled = date_data.fillna(date_data.median())
        
        # 2. MAD去极值（3倍MAD）
        median = filled.median()
        mad = (filled - median).abs().median()
        lower = median - 3 * mad
        upper = median + 3 * mad
        winsorized = filled.clip(lower, upper, axis=1)
        
        # 3. Z-Score标准化（截面）
        mean = winsorized.mean()
        std = winsorized.std()
        standardized = (winsorized - mean) / std
        
        processed.loc[date_mask, factor_cols] = standardized
    
    return processed
```

**为什么截面处理更重要**：
- 横截面选股需要的是**相对排名**，不是绝对值
- 时序标准化会破坏截面可比性
- MAD去极值比简单clip更robust

---

### 第四阶段：滚动窗口训练（核心改进）

#### 4.1 Walk-Forward框架

```python
class WalkForwardBacktest:
    """
    滚动固定窗口回测框架
    """
    def __init__(
        self,
        train_window=756,      # 训练窗口：3年 ≈ 756个交易日
        predict_window=20,     # 预测窗口：1个月
        rebalance_freq=5       # 调仓频率：每周
    ):
        self.train_window = train_window
        self.predict_window = predict_window
        self.rebalance_freq = rebalance_freq
    
    def generate_windows(self, data):
        """
        生成训练-预测窗口序列
        """
        dates = sorted(data['trade_date'].unique())
        windows = []
        
        start_idx = self.train_window
        while start_idx + self.predict_window <= len(dates):
            train_dates = dates[start_idx - self.train_window : start_idx]
            predict_dates = dates[start_idx : start_idx + self.predict_window]
            
            windows.append({
                'train_start': train_dates[0],
                'train_end': train_dates[-1],
                'predict_start': predict_dates[0],
                'predict_end': predict_dates[-1]
            })
            
            # 滑动窗口（每次前进predict_window）
            start_idx += self.predict_window
        
        return windows
    
    def run(self, data, model_params):
        """
        执行滚动回测
        """
        windows = self.generate_windows(data)
        predictions_all = []
        
        for i, window in enumerate(windows):
            print(f"Window {i+1}/{len(windows)}: "
                  f"Train {window['train_start']} to {window['train_end']}, "
                  f"Predict {window['predict_start']} to {window['predict_end']}")
            
            # 1. 准备训练数据
            train_data = data[
                (data['trade_date'] >= window['train_start']) &
                (data['trade_date'] <= window['train_end'])
            ]
            
            # 2. 训练模型
            model = self.train_model(train_data, model_params)
            
            # 3. 预测
            predict_data = data[
                (data['trade_date'] >= window['predict_start']) &
                (data['trade_date'] <= window['predict_end'])
            ]
            predictions = self.predict(model, predict_data)
            predictions_all.append(predictions)
        
        return pd.concat(predictions_all)
```

**关键点**：
- ✅ 每个窗口只用历史数据训练
- ✅ 预测期数据模型从未见过
- ✅ 避免look-ahead bias
- ✅ 模型定期更新，适应市场变化

---

### 第五阶段：模型训练

#### 5.1 XGBoost配置（借鉴LightGBM思路）

```python
def get_xgboost_params(trial=None):
    """
    XGBoost参数（类似LightGBM的设置）
    """
    if trial is None:  # 默认参数
        return {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 150,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 100,
            'reg_alpha': 1.0,
            'reg_lambda': 1.0,
            'seed': 42
        }
    else:  # 贝叶斯优化参数
        return {
            'max_depth': trial.suggest_int('max_depth', 4, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'n_estimators': trial.suggest_int('n_estimators', 100, 300),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 50, 200),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 5.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 5.0),
        }
```

---

### 第六阶段：组合构建与回测

#### 6.1 选股与组合

```python
def construct_portfolio(predictions, top_quantile=0.15):
    """
    根据预测Alpha构建组合
    
    Args:
        predictions: 包含['date', 'stock', 'alpha', 'excess_return']
        top_quantile: 选股比例（0.15 = 前15%）
    """
    portfolios = []
    
    for date in predictions['date'].unique():
        date_pred = predictions[predictions['date'] == date]
        
        # 选取预测Alpha最高的top_quantile
        n_select = max(int(len(date_pred) * top_quantile), 1)
        selected = date_pred.nlargest(n_select, 'alpha')
        
        # 等权组合
        selected['weight'] = 1.0 / len(selected)
        
        # 组合超额收益
        portfolio_excess_return = (selected['excess_return'] * selected['weight']).sum()
        
        portfolios.append({
            'date': date,
            'excess_return': portfolio_excess_return,
            'n_stocks': len(selected)
        })
    
    return pd.DataFrame(portfolios)
```

#### 6.2 绩效评估

```python
def evaluate_strategy(portfolio_returns, benchmark_returns):
    """
    评估策略表现
    """
    # 1. 超额收益序列
    excess_returns = portfolio_returns - benchmark_returns
    
    # 2. 累计净值
    cumulative_nav = (1 + excess_returns).cumprod()
    
    # 3. 年化收益
    total_days = len(excess_returns)
    total_return = cumulative_nav.iloc[-1] - 1
    annual_return = (1 + total_return) ** (252 / total_days) - 1
    
    # 4. 最大回撤
    running_max = cumulative_nav.expanding().max()
    drawdown = (cumulative_nav - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # 5. 卡玛比率
    calmar_ratio = annual_return / abs(max_drawdown)
    
    # 6. 夏普比率
    sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    
    # 7. IC和Rank IC（需要预测值）
    # 在预测阶段计算
    
    return {
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
        'sharpe_ratio': sharpe_ratio,
        'cumulative_nav': cumulative_nav
    }
```

---

### 第七阶段：参数优化

#### 7.1 两阶段优化策略

**阶段1：贝叶斯优化（粗搜索）**
```python
def objective(trial):
    """
    优化目标：最大化卡玛比率
    """
    # 模型参数
    model_params = get_xgboost_params(trial)
    
    # 组合参数
    top_quantile = trial.suggest_float('top_quantile', 0.10, 0.25)
    
    # 运行回测
    results = walk_forward_backtest(
        data_train_val,  # 只用训练+验证集
        model_params,
        top_quantile
    )
    
    # 返回卡玛比率（收益/最大回撤）
    return results['calmar_ratio']

# 运行优化
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
```

**阶段2：网格搜索（细搜索）**
```python
# 围绕贝叶斯最优参数±10%进行网格搜索
best_params = study.best_params

grid = {
    'learning_rate': [best_params['learning_rate'] * x for x in [0.9, 1.0, 1.1]],
    'n_estimators': [best_params['n_estimators'] + x for x in [-20, 0, 20]],
    'top_quantile': [best_params['top_quantile'] + x for x in [-0.02, 0, 0.02]]
}
```

---

### 第八阶段：严格验证

#### 8.1 三期划分

```python
# 训练期（用于模型训练和参数优化）
train_period = '2019-01-01' to '2022-12-31'

# 验证期（用于选择最佳参数组合）
validation_period = '2023-01-01' to '2023-12-31'

# 测试期（完全hold-out，不参与任何决策）
test_period = '2024-01-01' to '2024-12-31'
```

#### 8.2 防止过拟合检查

```python
def check_overfitting(train_results, val_results, test_results):
    """
    检查过拟合程度
    """
    metrics = ['ic', 'calmar_ratio', 'annual_return']
    
    print("Performance Comparison:")
    print(f"{'Metric':<20} {'Train':<12} {'Validation':<12} {'Test':<12} {'Val Drop':<12} {'Test Drop':<12}")
    print("="*80)
    
    for metric in metrics:
        train_val = train_results[metric]
        val_val = val_results[metric]
        test_val = test_results[metric]
        
        val_drop = (train_val - val_val) / train_val * 100
        test_drop = (val_val - test_val) / val_val * 100
        
        print(f"{metric:<20} {train_val:<12.4f} {val_val:<12.4f} {test_val:<12.4f} "
              f"{val_drop:<12.1f}% {test_drop:<12.1f}%")
    
    # 判断标准
    if abs(val_drop) > 30 or abs(test_drop) > 30:
        print("\n⚠️ Warning: Significant performance degradation detected!")
        print("   Possible overfitting or regime change.")
    else:
        print("\n✅ Model generalization appears reasonable.")
```

---

## 📊 预期结果目标

基于参考案例，调整我们的目标：

### 保守目标（更现实）

| 指标 | 原目标 | 新目标 | 参考案例 |
|------|-------|-------|---------|
| **IC** | > 0.04 | **> 0.03** | 0.0313 |
| **Rank IC** | - | **> 0.02** | 0.0236 |
| **年化收益** | > 35% | **> 30%** | 59.66% |
| **最大回撤** | <= -20% | **<= -30%** | -35.78% |
| **卡玛比率** | - | **> 1.0** | 1.67 |
| **夏普比率** | - | **> 1.2** | 1.61 |

### 评估标准

**一级指标（必须达标）**：
- IC > 0.03（预测能力）
- 卡玛比率 > 1.0（风险调整收益）

**二级指标（参考）**：
- 年化收益 > 30%
- 最大回撤 < -30%
- 夏普比率 > 1.2

---

## 🚀 实施计划

### Phase 1: 数据准备（1-2天）
- [ ] 加载科创50指数数据
- [ ] 计算相对指标（Beta、相关性等）
- [ ] 构造超额收益标签
- [ ] 实现截面预处理函数

### Phase 2: 框架搭建（2-3天）
- [ ] 实现WalkForwardBacktest类
- [ ] 验证窗口生成逻辑
- [ ] 测试单个窗口训练-预测流程

### Phase 3: 默认模型baseline（1天）
- [ ] 使用默认参数运行完整回测
- [ ] 计算IC、卡玛比率等指标
- [ ] 建立baseline（预期IC≈0.02-0.03）

### Phase 4: 参数优化（2-3天）
- [ ] 贝叶斯优化（50次试验）
- [ ] 网格搜索细化
- [ ] 记录每次优化结果

### Phase 5: 严格验证（1天）
- [ ] 在2024年测试集上验证
- [ ] 对比train/val/test性能
- [ ] 检查过拟合程度

### Phase 6: 结果分析（1天）
- [ ] 绘制净值曲线
- [ ] 分年度/月度统计
- [ ] IC时序分析
- [ ] 持仓周转率分析

---

## 💡 关键经验教训

### 从失败中学到的

1. **标签定义至关重要**
   - ❌ 绝对收益包含Beta，混淆择时和选股
   - ✅ 超额收益剥离Beta，聚焦Alpha

2. **数据泄露是隐形杀手**
   - ❌ 训练集=测试集导致虚假高收益
   - ✅ 滚动窗口确保真正样本外

3. **优化目标要匹配实际目标**
   - ❌ 只优化IC忽略风险
   - ✅ 优化卡玛比率平衡收益和回撤

4. **验证比训练更重要**
   - ❌ 样本内表现再好也可能是过拟合
   - ✅ 必须在真正hold-out集上验证

### 从参考案例学到的

1. **截面预处理很重要**
   - MAD去极值比简单clip更robust
   - Z-Score标准化保证可比性

2. **滚动窗口是标准做法**
   - 3年训练窗口（756个交易日）
   - 1个月预测窗口（20个交易日）
   - 每周调仓（5个交易日）

3. **相对指标不可或缺**
   - Beta、相关性、相对强度
   - 这些是指数增强的核心

4. **现实目标设定**
   - IC=0.03已经是不错的水平
   - 年化30-60%是合理区间
   - 回撤30-40%可以接受

---

## 📁 代码结构

```
star50-quant/
├── src/
│   ├── data/
│   │   ├── loaders.py                 # 已有
│   │   └── index_loader.py            # 新增：加载指数数据
│   ├── features/
│   │   ├── engineering.py             # 已有
│   │   ├── relative_features.py       # 新增：相对指标
│   │   └── cross_sectional.py         # 新增：截面预处理
│   ├── labels/
│   │   └── excess_return.py           # 新增：超额收益标签
│   ├── models/
│   │   ├── xgb_model.py              # 已有
│   │   └── walk_forward.py            # 新增：滚动窗口框架
│   └── backtest/
│       ├── portfolio.py               # 新增：组合构建
│       └── evaluation.py              # 新增：绩效评估
└── scripts/
    ├── run_walk_forward_backtest.py   # 新增：主回测脚本
    ├── optimize_parameters.py         # 新增：参数优化
    └── validate_overfitting.py        # 已有：过拟合验证
```

---

## ✅ 成功标准

### 最低要求（必须满足）
- ✅ IC > 0.03（样本外测试集）
- ✅ IC的t统计量显著（p < 0.05）
- ✅ 卡玛比率 > 1.0
- ✅ 样本外性能不低于样本内50%

### 优秀水平（努力达到）
- ✅ IC > 0.04
- ✅ 年化收益 > 40%
- ✅ 卡玛比率 > 1.5
- ✅ 最大回撤 < -25%

---

**下一步**: 立即开始Phase 1，预计7-10天完成全部实施和验证。

