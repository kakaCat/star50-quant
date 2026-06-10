# Phase 3: 回测与部署 - 设计文档

## 概述

**目标**: 将IC=0.0343的集成模型转化为完整的指数增强策略，通过回测验证其在真实交易环境下的表现。

**范围**: Alpha预测 → 风险估计 → 组合优化 → 回测评估的端到端流程

**成功标准**:
- 回测系统正常运行（无报错）
- 年化收益 >15%（超越基准）
- 跟踪误差控制在8%以内
- 信息比率 IR >0.5

## 架构设计

### 整体流程

```
历史数据
   ↓
核心9因子计算
   ↓
集成模型预测 (15个LightGBM + IC加权)
   ↓
Alpha信号 (IC=0.0343)
   ↓
样本协方差估计 (252天滚动窗口)
   ↓
组合优化 (cvxpy)
 - 目标: max(Alpha - λ*风险)
 - 约束: 跟踪误差≤5%, 换手率≤30%, 个股≤5%
   ↓
每周再平衡 (周一开盘)
   ↓
回测引擎
 - 交易成本: 0.15%双边
 - 科创板规则: ±20%涨跌停
   ↓
性能评估
 - 年化收益、夏普比率、信息比率
 - IC转化效率分析
```

### 核心组件

**1. 风险估计模块** (新增)
- 文件: `src/risk/covariance_estimator.py`
- 功能: 滚动窗口计算样本协方差矩阵
- 输入: 日收益率矩阵 [n_days, n_stocks]
- 输出: 协方差矩阵 [n_stocks, n_stocks]
- 方法: 样本协方差（简单、透明、适合50只股票）

**2. 集成模型预测器** (新增)
- 文件: `src/models/ensemble_predictor.py`
- 功能: 加载训练好的集成模型，生成Alpha预测
- 输入: 当日9个核心因子 [n_stocks, 9]
- 输出: Alpha预测 {ts_code, alpha}
- 模型: 15个LightGBM基础模型 + IC加权元学习器

**3. 组合优化流程** (改造现有)
- 文件: `src/optimization/portfolio_optimizer.py` (已有)
- 使用: `optimize_with_tracking_error()` 方法
- 配置: 
  - 风险厌恶系数 λ=2.0
  - 跟踪误差上限=5%
  - 单周换手率上限=30%
  - 个股权重上限=5%

**4. 回测引擎** (改造现有)
- 文件: `src/backtest/backtest_engine.py` (已有)
- 修改: 支持周度再平衡 (rebalance_freq='W-MON')
- 功能: 模拟交易、计算成本、记录净值

**5. 主流程脚本** (新增)
- 文件: `scripts/run_phase3_backtest.py`
- 功能: 整合所有组件，执行完整回测流程

## 数据流设计

### 时间划分

- **训练集**: 2020-01-01 ~ 2024-12-27 (集成模型训练，已完成)
- **回测期**: 2024-12-28 ~ 2025-12-31 (外样本验证)
- **风险估计**: 滚动252天窗口

### 每周再平衡流程

**周一开盘前:**
1. 加载过去252天收益率 → 计算协方差矩阵
2. 读取当前9个核心因子 → 集成模型预测Alpha
3. 获取上周五持仓权重
4. 组合优化:
   - 目标: maximize (w·alpha - λ·w'·Σ·w)
   - 约束: TE≤5%, turnover≤30%, w_i≤5%
5. 生成目标权重

**周一开盘:**
6. 按目标权重下单（模拟）
7. 扣除交易成本（0.15%双边 + 0.05%滑点）
8. 更新持仓

**周内 (周二~周五):**
9. 每日记录组合市值变化
10. 不触发交易

### 关键参数

| 参数 | 值 | 说明 |
|-----|---|-----|
| 初始资金 | 1000万 | |
| 再平衡频率 | 每周一 | 平衡信号利用和交易成本 |
| 风险厌恶系数 λ | 2.0 | 初始值，可调 |
| 跟踪误差上限 | 5% | 年化 |
| 单周换手率上限 | 30% | |
| 个股权重上限 | 5% | |
| 交易成本 | 0.15% | 双边（佣金+印花税） |
| 滑点 | 0.05% | |
| 涨跌停限制 | ±20% | 科创板 |
| 风险估计窗口 | 252天 | 约1年交易日 |

## 评估指标体系

### 绝对收益指标

- **累计收益率**: 期末净值/期初净值 - 1
- **年化收益率**: (1+累计收益)^(252/交易日数) - 1
  - **目标**: >30% (最终目标)
  - **Phase 3验收**: >15%
- **年化波动率**: std(日收益) × √252
- **夏普比率**: 年化收益 / 年化波动
  - **目标**: >1.5 (最终目标)
- **最大回撤**: max(peak - valley) / peak
- **卡玛比率**: 年化收益 / 最大回撤

### 相对收益指标

- **累计超额收益**: 组合累计收益 - 基准累计收益
- **年化跟踪误差**: std(超额收益) × √252
  - **目标**: 4-6% (最终目标)
  - **Phase 3验收**: <8%
- **信息比率 IR**: 年化超额收益 / 跟踪误差
  - **目标**: >1.0 (最终目标)
  - **Phase 3验收**: >0.5
- **胜率**: 日收益>基准的天数占比

### 交易指标

- **日均换手率**: 平均每日交易量 / 组合市值
- **周均换手率**: 平均每周交易量 / 组合市值
- **累计交易成本**: 总交易成本 / 期初资金
- **持仓集中度**: 1 / Σ(w_i²) (有效股票数)

### IC转化效率

- **IC vs 收益相关性**: corr(Alpha预测, 实际收益)
- **Alpha预测准确率**: 预测正确方向的比例
- **IC衰减分析**: 不同窗口(1/5/10/20天)的IC表现

### 归因分析

- **Alpha贡献**: w·alpha (预测收益)
- **风险成本**: λ·w'·Σ·w (风险惩罚)
- **交易成本侵蚀**: 累计交易成本占比
- **选股效应**: 个股选择的贡献
- **择时效应**: 再平衡时机的贡献

## 技术实现

### 文件结构

```
star50-quant/
├── src/
│   ├── risk/                    # 新增模块
│   │   ├── __init__.py
│   │   └── covariance_estimator.py
│   ├── models/
│   │   ├── ensemble_predictor.py   # 新增
│   │   ├── multi_window_loader.py  # 已有
│   │   └── lgbm_model.py           # 已有
│   ├── optimization/               # 已有
│   │   └── portfolio_optimizer.py
│   └── backtest/                   # 已有
│       └── backtest_engine.py
├── scripts/
│   ├── run_phase3_backtest.py   # 新增主脚本
│   └── train_ensemble.py        # 已有
├── configs/
│   └── phase3_config.yaml       # 新增配置
├── models/
│   └── phase2_ensemble/         # 已训练的模型
│       ├── w1_default.txt
│       ├── w1_regularized.txt
│       └── ... (15个模型)
└── results/                     # 新增结果目录
    └── phase3/
        ├── backtest_results.csv
        ├── trades.csv
        ├── metrics.json
        └── plots/
            ├── net_value.png
            ├── drawdown.png
            └── ic_vs_return.png
```

### 配置文件

`configs/phase3_config.yaml`:

```yaml
# 回测配置
backtest:
  start_date: '2024-12-28'
  end_date: '2025-12-31'
  initial_capital: 10000000
  rebalance_freq: 'W-MON'  # 每周一

# 风险估计配置
risk:
  estimation_window: 252  # 1年交易日
  method: 'sample'        # 样本协方差

# 组合优化配置
optimization:
  risk_aversion: 2.0
  max_tracking_error: 0.05    # 5%
  max_turnover: 0.30          # 30%
  max_weight: 0.05            # 5%
  min_weight: 0.0

# 交易成本配置
trading:
  commission_rate: 0.0015     # 0.15%双边
  slippage: 0.0005            # 0.05%
  price_limit: 0.20           # ±20%

# 集成模型配置
ensemble:
  model_dir: 'models/phase2_ensemble/'
  n_base_models: 15
  windows: [1, 3, 5, 10, 20]
```

### 核心类设计

#### 1. CovarianceEstimator

```python
class CovarianceEstimator:
    """样本协方差估计器"""
    
    def __init__(self, window: int = 252):
        self.window = window
    
    def estimate(
        self,
        returns: pd.DataFrame,
        method: str = 'sample'
    ) -> np.ndarray:
        """
        计算协方差矩阵
        
        Args:
            returns: 日收益率 [n_days, n_stocks]
            method: 'sample'
        
        Returns:
            协方差矩阵 [n_stocks, n_stocks]
        """
        # 取最近window天
        # 处理缺失值 (forward fill)
        # 计算样本协方差
        # 确保正定性
```

#### 2. EnsemblePredictor

```python
class EnsemblePredictor:
    """集成模型预测器"""
    
    def __init__(self, model_dir: str):
        # 加载15个基础模型
        # 加载IC权重
    
    def predict(
        self,
        features: pd.DataFrame
    ) -> pd.DataFrame:
        """
        生成Alpha预测
        
        Args:
            features: 9个核心因子 [n_stocks, 9]
        
        Returns:
            {ts_code, alpha}
        """
        # 对每个基础模型预测
        # IC加权融合
```

#### 3. 主流程伪代码

```python
# scripts/run_phase3_backtest.py

def main():
    # 1. 加载配置
    config = load_config('configs/phase3_config.yaml')
    
    # 2. 加载数据
    prices = load_prices(config.start_date, config.end_date)
    features = calculate_features(prices)  # 9个核心因子
    
    # 3. 初始化组件
    predictor = EnsemblePredictor(config.ensemble.model_dir)
    risk_estimator = CovarianceEstimator(config.risk.estimation_window)
    optimizer = PortfolioOptimizer(
        risk_aversion=config.optimization.risk_aversion,
        max_weight=config.optimization.max_weight,
        max_turnover=config.optimization.max_turnover
    )
    backtester = BacktestEngine(
        initial_capital=config.backtest.initial_capital,
        commission_rate=config.trading.commission_rate,
        slippage=config.trading.slippage
    )
    
    # 4. 计算基准权重 (科创50等权)
    benchmark_weights = calculate_benchmark_weights(prices)
    
    # 5. 逐周回测循环
    rebalance_dates = get_rebalance_dates(
        config.start_date,
        config.end_date,
        freq='W-MON'
    )
    
    weights_series = []
    previous_weights = None
    
    for date in rebalance_dates:
        # 5.1 风险估计
        historical_returns = get_historical_returns(prices, date, 252)
        covariance = risk_estimator.estimate(historical_returns)
        
        # 5.2 Alpha预测
        daily_features = features[features['factor_date'] == date]
        alpha = predictor.predict(daily_features)
        
        # 5.3 组合优化
        result = optimizer.optimize_with_tracking_error(
            alpha=alpha['alpha'].values,
            covariance=covariance,
            benchmark_weights=benchmark_weights,
            max_tracking_error=config.optimization.max_tracking_error,
            previous_weights=previous_weights
        )
        
        # 5.4 记录权重
        weights_series.append({
            'date': date,
            'weights': result['weights'],
            'alpha': alpha['alpha'].values
        })
        
        previous_weights = result['weights']
    
    # 6. 运行回测
    backtest_results = backtester.run_backtest(
        weights_series=pd.DataFrame(weights_series),
        prices=prices,
        benchmark_weights=benchmark_weights
    )
    
    # 7. 计算指标
    metrics = backtester.calculate_metrics(
        backtest_results['portfolio']
    )
    
    # 8. 归因分析
    attribution = backtester.calculate_attribution(
        backtest_results['portfolio'],
        weights_series,
        alpha_series
    )
    
    # 9. 保存结果
    save_results(backtest_results, metrics, attribution)
    
    # 10. 可视化
    plot_results(backtest_results, metrics)
    
    # 11. 验收检查
    validate_phase3(metrics)
```

## 错误处理与边界情况

### 数据缺失处理

**因子缺失**
- 策略: Forward fill最近可用值
- 限制: 最多填充5个交易日，超过则排除该股票

**价格缺失（停牌）**
- 优化时: 将该股票从可投资池中排除
- 回测时: 持仓按停牌前价格计算市值，恢复交易后按实际价格调整

**新股上市**
- 策略: 上市满30个交易日后纳入可投资池
- 原因: 新股波动大，因子不稳定

### 优化失败处理

**cvxpy求解失败**
- Level 1: 保持上期权重（不交易）
- Level 2: 放宽跟踪误差至8%重试
- Level 3: 降级为等权组合
- 记录失败原因，生成警告日志

**跟踪误差约束无解**
- 原因: Alpha信号与基准偏离过大
- 处理: 逐步放宽约束（5% → 6% → 8%）
- 记录实际跟踪误差

### 极端市场情况

**涨跌停无法成交**
- 买入遇涨停: 按涨停价部分成交，剩余资金保留现金
- 卖出遇跌停: 延后至次日继续卖出
- 记录未成交量

**流动性不足**
- 约束: 单只股票买入量≤日成交量10%
- 策略: 优先调整流动性好的股票
- 记录流动性冲击成本

### 模型预测异常

**Alpha预测NaN**
- 原因: 因子缺失或模型输入异常
- 处理: 使用该股票历史Alpha均值填充
- 警告日志

**Alpha预测方差过大**
- 检测: |alpha| > 3σ
- 处理: 截断到±3σ
- 原因: 避免极端值主导组合

**所有Alpha为负**
- 处理: 仍执行优化（可能全配现金或基准权重）
- 警告: 模型可能失效

## 测试策略

### 单元测试

`tests/risk/test_covariance_estimator.py`:
- `test_sample_covariance_positive_definite()`: 验证协方差矩阵正定
- `test_rolling_window()`: 验证滚动窗口计算
- `test_missing_data_handling()`: 验证缺失值处理

`tests/models/test_ensemble_predictor.py`:
- `test_load_models()`: 验证模型加载
- `test_predict_shape()`: 验证预测输出形状
- `test_ic_weighted_fusion()`: 验证IC加权融合逻辑

`tests/integration/test_phase3_pipeline.py`:
- `test_end_to_end_single_week()`: 测试单周完整流程
- `test_optimization_convergence()`: 测试优化收敛性

### 回测验证

**手工计算验证**
- 选择第一个再平衡日（2024-12-30）
- 手工计算协方差、Alpha、优化结果
- 对比程序输出，确保无误

**基准对比**
- 等权组合: 验证超额收益合理性
- Buy & Hold: 验证交易成本计算

**边界测试**
- 极端IC值 (±0.1)
- 极端波动率 (年化50%)
- 全部涨停/跌停场景

## 可交付成果

### 代码

1. `src/risk/covariance_estimator.py` - 协方差估计器 (新增)
2. `src/models/ensemble_predictor.py` - 集成模型预测器 (新增)
3. `scripts/run_phase3_backtest.py` - 主回测脚本 (新增)
4. `configs/phase3_config.yaml` - 配置文件 (新增)
5. `src/backtest/backtest_engine.py` - 支持周度再平衡 (修改)
6. 单元测试套件 (3-5个测试文件)

### 文档

`docs/superpowers/reports/2026-06-10-phase3-results.md`:
- 回测指标汇总
- 净值曲线图
- 问题分析
- 优化建议

### 数据产出

1. `results/phase3/backtest_results.csv` - 逐日组合净值
2. `results/phase3/trades.csv` - 交易记录
3. `results/phase3/metrics.json` - 评估指标
4. `results/phase3/plots/` - 可视化图表
   - `net_value.png` - 净值曲线
   - `drawdown.png` - 回撤曲线
   - `ic_vs_return.png` - IC vs 收益散点图
   - `turnover.png` - 换手率时序图

## 时间估算

| 任务 | 工作量 | 说明 |
|-----|-------|-----|
| 协方差估计器 | 30分钟 | 简单的样本协方差实现 |
| 集成模型预测器 | 45分钟 | 加载15个模型+IC加权融合 |
| 主回测脚本 | 1小时 | 整合现有组件 |
| 回测引擎改造 | 30分钟 | 支持周度再平衡 |
| 测试与调试 | 1小时 | 单元测试+端到端验证 |
| 结果分析与文档 | 45分钟 | 报告撰写+可视化 |
| **总计** | **4-5小时** | |

## 风险与依赖

### 关键假设

1. **IC稳定性**: 集成模型在外样本期保持IC>0.02
   - 训练集IC=0.0343
   - 保守估计外样本IC=0.02-0.025
   
2. **再平衡频率**: 周度足以捕获Alpha信号
   - 模型包含1-20天多窗口
   - 周度平衡短期和中期信号
   
3. **协方差稳定性**: 50只股票用252天样本足够
   - 样本数(252) > 参数数(1225)的20%
   - 科创板成分股相对稳定

### 潜在问题

**1. IC衰减**
- 风险: 外样本IC可能降至0.01-0.02
- 影响: 超额收益大幅下降
- 缓解: 先用2024年12月验证集评估IC，提前预警

**2. 交易成本侵蚀**
- 估算: 0.15% × 30%换手 × 52周 ≈ 2.3%年化成本
- 影响: 侵蚀超额收益
- 缓解: 优化器显式考虑换手率约束

**3. 跟踪误差超标**
- 风险: 5%约束可能过紧，导致优化无解
- 影响: 频繁触发fallback，策略退化为等权
- 缓解: 分阶段放宽约束（5% → 8%）

**4. 市场环境变化**
- 风险: 2025年市场风格与训练期不同
- 影响: 因子失效，IC大幅下降
- 缓解: 监控逐月IC，及时预警

## Phase 3验收标准

### 必须全部达成

1. ✓ 回测系统正常运行（无报错，完整输出）
2. ✓ 年化收益 >15%（超越基准）
3. ✓ 跟踪误差控制在8%以内
4. ✓ 信息比率 IR >0.5
5. ✓ 最大回撤 <20%
6. ✓ 交易成本 <3%（累计）

### 如果未达标

**年化收益<15%**
- 检查IC衰减程度
- 分析交易成本侵蚀
- 考虑调整风险厌恶系数λ或换手率约束

**跟踪误差>8%**
- 检查优化约束是否合理
- 分析Alpha信号波动性
- 考虑收紧个股权重上限

**IR<0.5**
- IC转化效率低
- 可能需要改进Alpha模型或优化流程
- 分析归因：选股 vs 交易成本

## 后续优化方向（Phase 4）

如果Phase 3验收通过，Phase 4可以探索：

1. **风险模型升级**: 样本协方差 → Ledoit-Wolf收缩估计
2. **再平衡频率优化**: 固定周度 → 自适应触发
3. **Alpha模型改进**: 
   - 添加行业/风格因子
   - 尝试深度学习模型（LSTM/Transformer）
   - 在线学习与模型更新
4. **交易执行优化**: 
   - VWAP/TWAP算法交易
   - 流动性感知的订单拆分
5. **多资产扩展**: 科创50 → 科创100 → 全市场

---

**文档版本**: v1.0  
**创建日期**: 2026-06-10  
**作者**: Kiro AI  
**状态**: 待审核
