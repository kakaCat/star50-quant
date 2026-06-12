# LightGBM 固定窗口与滚动窗口优化报告

**实验日期**: 2026-06-12
**项目阶段**: 阶段二 Alpha 预测模型
**数据来源**: `data/raw/star50_daily_hfq_data_6yrs.parquet`、`data/raw/star50_index_daily_6yrs.parquet`
**标签**: `label_5d`，个股未来 5 日相对科创 50 指数的超额收益
**核心目标**: 在基础量价因子和指数特征基础上，验证 LightGBM 是否能提供稳定的截面排序能力，并比较固定窗口与滚动窗口训练方式。

---

## 1. 实验路径

本轮实验从原始 Parquet 数据开始，流程保持一致：

1. 读取个股后复权行情和科创 50 指数行情。
2. 生成基础量价因子、指数状态因子和未来超额收益标签。
3. 做截面预处理：缺失值填充、MAD 去极值、Z-Score 标准化。
4. 用 LightGBM 训练 Alpha 排序模型。
5. 取预测 Alpha 排名前 15% 股票，等权持有并每 5 个交易日调仓。
6. 输出净值、回撤、卡玛比率、IC、Rank IC、ICIR 等指标。

当前实验使用的特征仍以量价特征为主，不包含市值、行业、财务、研发、专利、分析师预期等基本面或另类数据。

---

## 2. 代码与结果位置

### 核心脚本

| 用途 | 路径 |
|---|---|
| 阶段一特征生成 | `scripts/02_build_stage1_features.py` |
| 阶段一截面预处理 | `scripts/03_preprocess_stage1_features.py` |
| Alpha baseline | `scripts/04_train_alpha_baseline.py` |
| 默认固定窗口回测 | `scripts/05_lightgbm_fixed_window_backtest.py` |
| Optuna 贝叶斯优化 | `scripts/06_optimize_lightgbm_fixed_window.py` |
| 网格搜索优化 | `scripts/07_grid_search_lightgbm_fixed_window.py` |
| Spyder 固定窗口一体化脚本 | `scripts/08_spyder_lightgbm_pipeline.py` |
| IC 图表生成 | `scripts/09_generate_lightgbm_ic_report.py` |
| Spyder 滚动窗口一体化脚本 | `scripts/10_spyder_lightgbm_rolling_pipeline.py` |

### 汇总结果

| 文件 | 说明 |
|---|---|
| `tuning_results/lightgbm_window/metrics_summary.csv` | 固定窗口、Optuna、网格搜索、滚动窗口指标总表 |
| `tuning_results/lightgbm_window/best_params.csv` | 当前最佳 LightGBM 参数 |
| `tuning_results/lightgbm_window/rolling_metrics.csv` | 已跑通的滚动窗口指标 |
| `tuning_results/lightgbm_window/rolling_nav.png` | 已跑通的滚动窗口净值图 |
| `results/lightgbm/lightgbm_results_20260612.json` | 机器可读最终结果 |

---

## 3. 参数优化过程

### 3.1 默认固定窗口

默认版本采用：

- 训练窗口：756 个交易日，约 3 年。
- 预测窗口：20 个交易日。
- 调仓频率：5 个交易日。
- 选股方式：按预测 Alpha 排名前 15% 等权持有。

该版本先用于确认模型、标签、回测链路能完整跑通。

### 3.2 贝叶斯优化

第二步使用 Optuna 做贝叶斯优化，目标是降低回撤并提高风险调整后收益。贝叶斯优化后的结果提升了累计净值、年化收益和卡玛比率，但最大回撤没有明显下降，说明单纯参数搜索不能完全解决回撤问题。

### 3.3 网格搜索

第三步改用聚焦网格搜索，在已知较优区域附近搜索：

| 参数 | 最优值 |
|---|---:|
| `n_estimators` | 180 |
| `learning_rate` | 0.045 |
| `num_leaves` | 62 |
| `min_child_samples` | 120 |
| `top_quantile` | 0.15 |
| `subsample` | 0.708884 |
| `colsample_bytree` | 0.913066 |
| `reg_alpha` | 0.000601 |
| `reg_lambda` | 0.483284 |

网格搜索版本在固定窗口框架下表现最好，因此滚动窗口版本沿用这组参数。

### 3.4 滚动窗口

滚动窗口版本保持其它逻辑不变，只把预测窗口从 20 个交易日改为 5 个交易日：

- 训练窗口仍为 756 个交易日。
- 每 5 个交易日重新训练一次模型。
- 每次只预测下一段约 5 个交易日。
- 调仓频率仍为 5 个交易日。

这样做的含义是让模型训练、预测和持仓周期更一致，减少“用 20 日预测信号做 5 日执行”带来的信号错配。

---

## 4. 结果对比

| 实验 | 累计净值 | 年化收益 | 最大回撤 | 卡玛比率 | 夏普比率 | IC | Rank IC |
|---|---:|---:|---:|---:|---:|---:|---:|
| 默认固定窗口 | 2.2476 | 37.97% | -40.72% | 0.9325 | 1.0447 | 0.0272 | 0.0190 |
| Optuna 固定窗口 | 2.6701 | 47.75% | -41.26% | 1.1573 | 1.2882 | 0.0231 | 0.0159 |
| 网格搜索固定窗口 | 3.2448 | 59.66% | -35.78% | 1.6672 | 1.6094 | 0.0313 | 0.0236 |
| 网格参数滚动窗口 | 8.8650 | 138.06% | -28.10% | 4.9130 | 3.6114 | 0.0721 | 0.0550 |

滚动窗口版本在这组历史数据上明显优于固定窗口，主要体现在：

- 累计净值从 3.2448 提升到 8.8650。
- 最大回撤从 -35.78% 降到 -28.10%。
- 卡玛比率从 1.6672 提升到 4.9130。
- IC 从 0.0313 提升到 0.0721，Rank IC 从 0.0236 提升到 0.0550。

---

## 5. Spyder 运行方式

固定窗口：

```python
import os
os.chdir("/Users/elize/Desktop/量化")
%run spyder_lightgbm_pipeline.py
```

滚动窗口：

```python
import os
os.chdir("/Users/elize/Desktop/量化")
%run spyder_lightgbm_rolling_pipeline.py
```

如需在项目目录运行，可将两个 Parquet 文件放在项目 `data/raw/` 下，并使用 `scripts/05` 至 `scripts/10` 中对应脚本。

---

## 6. 重要限制

这组结果用于模型路径验证和阶段二实验记录，尚不是最终指数增强策略结果。

主要限制：

1. 当前数据只有量价和指数行情，没有基本面、行业、市值、研发、专利、分析师预期等信息。
2. 当前回测没有加入真实交易成本，脚本中 `TRANSACTION_COST = 0.0`。
3. 当前回测没有处理科创板涨跌停买卖约束、停牌、冲击成本和真实成分股权重约束。
4. 当前组合是 Top 15% 等权持有，还没有接入阶段四的组合优化器、行业中性、风格暴露约束和个股偏离约束。
5. 滚动窗口重训频率更高，表现更好但计算成本也更高，后续需要做更严格的 walk-forward 验证和交易成本压力测试。

---

## 7. 阶段结论

本轮 LightGBM 实验说明：

- 基础量价因子在科创 50 样本上具备一定截面预测能力。
- 参数优化可以改善固定窗口模型，但收益和回撤改善有限。
- 让预测窗口与调仓周期一致后，滚动窗口模型的 IC、净值和回撤指标都有明显改善。
- 下一步应把滚动窗口 LightGBM 作为阶段二 Alpha baseline，并继续补充基本面、行业、市值和科创特色因子，再进入风险模型与组合优化阶段。
