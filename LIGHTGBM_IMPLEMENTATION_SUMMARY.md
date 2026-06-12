# LightGBM固定窗口与滚动窗口调优总结

**完成时间**: 2026-06-12
**实施方式**: 基于 Parquet 文件的端到端实现
**状态**: ✅ 已完成并在 Spyder 跑通

---

## 实施内容

### 已完成的核心功能

✅ **数据处理**
- 从 `star50_daily_hfq_data_6yrs.parquet` 和 `star50_index_daily_6yrs.parquet` 读取数据。
- 构造基础量价因子、指数状态因子和未来超额收益标签。
- 标签使用 `label_5d`：个股未来 5 日收益相对科创 50 指数的超额收益。

✅ **截面预处理**
- 缺失值填充。
- MAD 截面去极值。
- 截面 Z-Score 标准化。

✅ **LightGBM 固定窗口训练**
- 训练窗口：756 个交易日。
- 预测窗口：20 个交易日。
- 调仓频率：5 个交易日。
- 选股方式：预测 Alpha 排名前 15% 等权持有。

✅ **参数优化**
- 完成默认参数、Optuna 贝叶斯优化、聚焦网格搜索三组对比。
- 当前最优参数来自聚焦网格搜索。

✅ **LightGBM 滚动窗口训练**
- 沿用网格搜索最优参数。
- 训练窗口仍为 756 个交易日。
- 预测窗口改为 5 个交易日。
- 每 5 个交易日重新训练并预测下一段。
- 目的：让预测周期和调仓周期一致，减少信号错配。

---

## 运行结果

### 结果对比

| 实验 | 累计净值 | 年化收益 | 最大回撤 | 卡玛比率 | 夏普比率 | IC | Rank IC |
|---|---:|---:|---:|---:|---:|---:|---:|
| 默认固定窗口 | 2.2476 | 37.97% | -40.72% | 0.9325 | 1.0447 | 0.0272 | 0.0190 |
| Optuna 固定窗口 | 2.6701 | 47.75% | -41.26% | 1.1573 | 1.2882 | 0.0231 | 0.0159 |
| 网格搜索固定窗口 | 3.2448 | 59.66% | -35.78% | 1.6672 | 1.6094 | 0.0313 | 0.0236 |
| 网格参数滚动窗口 | **8.8650** | **138.06%** | **-28.10%** | **4.9130** | **3.6114** | **0.0721** | **0.0550** |

### 最优参数

| 参数 | 值 |
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

---

## 核心发现

### 1. 固定窗口可以跑通，但回撤偏大

默认固定窗口的 IC 为 0.0272，说明基础量价因子已有一定截面预测能力；但最大回撤达到 -40.72%，风险调整后收益不够理想。

### 2. 网格搜索优于 Optuna 本轮结果

Optuna 提升了累计净值和卡玛比率，但最大回撤没有下降；聚焦网格搜索在收益、回撤和 IC 上更均衡，是固定窗口下的最佳版本。

### 3. 滚动窗口显著改善结果

滚动窗口将预测窗口从 20 日改为 5 日，并与 5 日调仓频率对齐。结果显示：

- IC 从 0.0313 提升到 0.0721。
- Rank IC 从 0.0236 提升到 0.0550。
- 最大回撤从 -35.78% 降到 -28.10%。
- 卡玛比率从 1.6672 提升到 4.9130。

---

## 文件结构

```text
star50-quant/
├── LIGHTGBM_IMPLEMENTATION_SUMMARY.md          # 本总结
├── docs/
│   └── LIGHTGBM_WINDOW_OPTIMIZATION_REPORT.md  # 更详细的实验记录
├── scripts/
│   ├── 05_lightgbm_fixed_window_backtest.py
│   ├── 06_optimize_lightgbm_fixed_window.py
│   ├── 07_grid_search_lightgbm_fixed_window.py
│   ├── 08_spyder_lightgbm_pipeline.py
│   └── 10_spyder_lightgbm_rolling_pipeline.py
├── tuning_results/
│   └── lightgbm_window/
│       ├── metrics_summary.csv
│       ├── best_params.csv
│       ├── rolling_metrics.csv
│       └── rolling_nav.png
└── results/
    └── lightgbm/
        └── lightgbm_results_20260612.json
```

---

## Spyder 运行方式

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

---

## 重要限制

这组结果是阶段二 Alpha 模型验证，不是最终指数增强实盘回测。

当前限制：

- 当前数据只有量价和指数行情，没有行业、市值、财务、研发、专利、分析师预期等因子。
- 当前脚本设置 `TRANSACTION_COST = 0.0`，尚未加入真实交易成本。
- 尚未加入科创板涨跌停、停牌、冲击成本等交易约束。
- 当前组合是 Top 15% 等权持有，还未接入组合优化、行业中性和风格约束。

---

## 最终结论

LightGBM 已完成固定窗口、参数优化和滚动窗口验证。当前最推荐作为阶段二 baseline 的版本是：

> **网格搜索最优参数 + 5 日滚动窗口 + Top 15% 等权组合**

下一步建议：

1. 加入交易成本和涨跌停约束，复核收益和回撤。
2. 补充行业、市值、基本面和科创特色因子。
3. 接入组合优化器，控制行业/风格暴露和个股偏离。
4. 与 XGBoost、CatBoost 或深度学习模型做 ensemble 对比。
