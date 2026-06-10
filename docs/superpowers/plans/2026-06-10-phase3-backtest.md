# Phase 3: 回测与部署 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建端到端回测系统，验证IC=0.0343的集成模型在真实交易环境下的表现

**Architecture:** 集成模型预测器 → 样本协方差估计器 → 组合优化器 → 回测引擎 → 性能评估。每周一再平衡，使用跟踪误差约束的均值方差优化。

**Tech Stack:** Python 3.12, LightGBM, cvxpy, pandas, numpy, pytest

---

## File Structure

### New Files
- `src/risk/__init__.py` - 风险模块初始化
- `src/risk/covariance_estimator.py` - 协方差估计器
- `src/models/ensemble_predictor.py` - 集成模型预测器
- `scripts/run_phase3_backtest.py` - 主回测脚本
- `configs/phase3_config.yaml` - 配置文件
- `tests/risk/test_covariance_estimator.py` - 协方差估计器测试
- `tests/models/test_ensemble_predictor.py` - 集成模型预测器测试

### Modified Files
- `src/backtest/backtest_engine.py` - 支持周度再平衡

### Directory Structure
```
star50-quant/
├── src/
│   ├── risk/                          # 新增
│   │   ├── __init__.py
│   │   └── covariance_estimator.py
│   ├── models/
│   │   └── ensemble_predictor.py      # 新增
│   ├── optimization/                   # 已有
│   └── backtest/                       # 修改
├── scripts/
│   └── run_phase3_backtest.py         # 新增
├── configs/
│   └── phase3_config.yaml             # 新增
├── tests/
│   ├── risk/                          # 新增
│   └── models/                        # 已有
└── results/
    └── phase3/                        # 新增
```

---

## Task 1: 协方差估计器

**Files:**
- Create: `src/risk/__init__.py`
- Create: `src/risk/covariance_estimator.py`
- Create: `tests/risk/__init__.py`
- Create: `tests/risk/test_covariance_estimator.py`

- [ ] **Step 1: 创建风险模块目录**

```bash
mkdir -p star50-quant/src/risk
mkdir -p star50-quant/tests/risk
touch star50-quant/src/risk/__init__.py
touch star50-quant/tests/risk/__init__.py
```

- [ ] **Step 2: 编写协方差估计器测试**

创建 `tests/risk/test_covariance_estimator.py`:

```python
"""
协方差估计器测试
"""

import numpy as np
import pandas as pd
import pytest
from src.risk.covariance_estimator import CovarianceEstimator


class TestCovarianceEstimator:
    """协方差估计器测试套件"""

    def test_sample_covariance_shape(self):
        """测试输出形状正确"""
        # 创建测试数据：10天 × 5只股票
        returns = pd.DataFrame(
            np.random.randn(10, 5),
            columns=['stock1', 'stock2', 'stock3', 'stock4', 'stock5']
        )
        
        estimator = CovarianceEstimator(window=10)
        cov_matrix = estimator.estimate(returns)
        
        # 验证形状
        assert cov_matrix.shape == (5, 5)
    
    def test_covariance_positive_definite(self):
        """测试协方差矩阵正定"""
        returns = pd.DataFrame(
            np.random.randn(50, 5),
            columns=[f'stock{i}' for i in range(5)]
        )
        
        estimator = CovarianceEstimator(window=50)
        cov_matrix = estimator.estimate(returns)
        
        # 验证正定：所有特征值>0
        eigenvalues = np.linalg.eigvals(cov_matrix)
        assert np.all(eigenvalues > 0)
    
    def test_covariance_symmetric(self):
        """测试协方差矩阵对称"""
        returns = pd.DataFrame(
            np.random.randn(30, 3),
            columns=['A', 'B', 'C']
        )
        
        estimator = CovarianceEstimator(window=30)
        cov_matrix = estimator.estimate(returns)
        
        # 验证对称
        assert np.allclose(cov_matrix, cov_matrix.T)
    
    def test_rolling_window(self):
        """测试滚动窗口计算"""
        # 创建100天数据
        returns = pd.DataFrame(
            np.random.randn(100, 3),
            columns=['X', 'Y', 'Z']
        )
        
        # 使用最近50天
        estimator = CovarianceEstimator(window=50)
        cov_matrix = estimator.estimate(returns)
        
        # 手工计算最近50天协方差
        recent_returns = returns.iloc[-50:]
        expected_cov = recent_returns.cov().values
        
        # 验证结果接近
        assert np.allclose(cov_matrix, expected_cov, rtol=1e-5)
    
    def test_missing_data_forward_fill(self):
        """测试缺失值处理（forward fill）"""
        # 创建带缺失值的数据
        returns = pd.DataFrame({
            'stock1': [0.01, 0.02, np.nan, 0.03, 0.01],
            'stock2': [0.02, np.nan, 0.01, 0.02, 0.03],
            'stock3': [0.01, 0.01, 0.02, 0.01, 0.02]
        })
        
        estimator = CovarianceEstimator(window=5)
        cov_matrix = estimator.estimate(returns)
        
        # 验证无NaN
        assert not np.isnan(cov_matrix).any()
        
        # 验证形状
        assert cov_matrix.shape == (3, 3)
```

- [ ] **Step 3: 运行测试确认失败**

```bash
cd star50-quant
pytest tests/risk/test_covariance_estimator.py -v
```

Expected: 全部FAIL，ModuleNotFoundError: No module named 'src.risk.covariance_estimator'

- [ ] **Step 4: 实现协方差估计器**

创建 `src/risk/covariance_estimator.py`:

```python
"""
协方差估计器
============

使用滚动窗口计算样本协方差矩阵。
"""

import numpy as np
import pandas as pd
from typing import Optional


class CovarianceEstimator:
    """
    样本协方差估计器
    
    使用历史收益率的滚动窗口计算协方差矩阵。
    """
    
    def __init__(self, window: int = 252):
        """
        初始化
        
        Args:
            window: 滚动窗口大小（交易日数），默认252天≈1年
        """
        self.window = window
    
    def estimate(
        self,
        returns: pd.DataFrame,
        method: str = 'sample'
    ) -> np.ndarray:
        """
        计算协方差矩阵
        
        Args:
            returns: 日收益率 DataFrame [n_days, n_stocks]
                     列名为股票代码，索引为日期
            method: 估计方法，目前只支持 'sample'
        
        Returns:
            协方差矩阵 [n_stocks, n_stocks]
        
        Raises:
            ValueError: 如果数据不足或method不支持
        """
        if method != 'sample':
            raise ValueError(f"不支持的方法: {method}，目前只支持 'sample'")
        
        # 处理缺失值：forward fill
        returns_filled = returns.fillna(method='ffill')
        
        # 如果仍有NaN（首行缺失），用0填充
        returns_filled = returns_filled.fillna(0)
        
        # 取最近window天
        if len(returns_filled) > self.window:
            recent_returns = returns_filled.iloc[-self.window:]
        else:
            recent_returns = returns_filled
        
        # 检查数据是否足够
        if len(recent_returns) < 2:
            raise ValueError(f"数据不足: 需要至少2天，实际{len(recent_returns)}天")
        
        # 计算样本协方差
        cov_matrix = recent_returns.cov().values
        
        # 确保正定性（添加小的正则化项）
        epsilon = 1e-8
        cov_matrix = cov_matrix + epsilon * np.eye(cov_matrix.shape[0])
        
        return cov_matrix
    
    def estimate_correlation(
        self,
        returns: pd.DataFrame
    ) -> np.ndarray:
        """
        计算相关系数矩阵
        
        Args:
            returns: 日收益率 DataFrame
        
        Returns:
            相关系数矩阵 [n_stocks, n_stocks]
        """
        returns_filled = returns.fillna(method='ffill').fillna(0)
        
        if len(returns_filled) > self.window:
            recent_returns = returns_filled.iloc[-self.window:]
        else:
            recent_returns = returns_filled
        
        corr_matrix = recent_returns.corr().values
        
        return corr_matrix
```

- [ ] **Step 5: 运行测试确认通过**

```bash
pytest tests/risk/test_covariance_estimator.py -v
```

Expected: 全部PASS (5 passed)

- [ ] **Step 6: 提交代码**

```bash
git add src/risk/ tests/risk/
git commit -m "feat: 实现协方差估计器

- 样本协方差计算，252天滚动窗口
- Forward fill处理缺失值
- 正则化确保正定性
- 完整单元测试覆盖"
```

---

## Task 2: 集成模型预测器

**Files:**
- Create: `src/models/ensemble_predictor.py`
- Create: `tests/models/test_ensemble_predictor.py`

- [ ] **Step 1: 编写集成模型预测器测试**

创建 `tests/models/test_ensemble_predictor.py`:

```python
"""
集成模型预测器测试
"""

import numpy as np
import pandas as pd
import pytest
import tempfile
import os
from src.models.ensemble_predictor import EnsemblePredictor


class TestEnsemblePredictor:
    """集成模型预测器测试套件"""
    
    @pytest.fixture
    def mock_model_dir(self):
        """创建模拟模型目录"""
        # 注意：这个测试需要真实的模型文件
        # 在实际环境中，应该使用phase2训练好的模型
        model_dir = 'models/phase2_ensemble/'
        if not os.path.exists(model_dir):
            pytest.skip(f"模型目录不存在: {model_dir}")
        return model_dir
    
    def test_load_models(self, mock_model_dir):
        """测试模型加载"""
        predictor = EnsemblePredictor(mock_model_dir)
        
        # 验证15个基础模型已加载
        assert len(predictor.base_models) == 15
        
        # 验证权重已加载
        assert len(predictor.weights) == 15
        assert np.isclose(predictor.weights.sum(), 1.0)
    
    def test_predict_shape(self, mock_model_dir):
        """测试预测输出形状"""
        predictor = EnsemblePredictor(mock_model_dir)
        
        # 创建测试特征：10只股票 × 9个因子
        features = pd.DataFrame(
            np.random.randn(10, 9),
            columns=[
                'momentum_5', 'momentum_10', 'momentum_20',
                'volatility_10', 'volatility_20',
                'volume_ratio', 'atr_ratio', 'ma_ratio', 'rsi14'
            ]
        )
        features['ts_code'] = [f'stock{i}' for i in range(10)]
        
        # 预测
        result = predictor.predict(features)
        
        # 验证输出
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 10
        assert 'ts_code' in result.columns
        assert 'alpha' in result.columns
    
    def test_ic_weighted_fusion(self, mock_model_dir):
        """测试IC加权融合"""
        predictor = EnsemblePredictor(mock_model_dir)
        
        # 创建测试特征
        features = pd.DataFrame(
            np.random.randn(5, 9),
            columns=[
                'momentum_5', 'momentum_10', 'momentum_20',
                'volatility_10', 'volatility_20',
                'volume_ratio', 'atr_ratio', 'ma_ratio', 'rsi14'
            ]
        )
        features['ts_code'] = [f'stock{i}' for i in range(5)]
        
        # 预测
        result = predictor.predict(features)
        
        # 验证alpha值合理（不是NaN或Inf）
        assert not result['alpha'].isna().any()
        assert not np.isinf(result['alpha']).any()
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/models/test_ensemble_predictor.py -v
```

Expected: FAIL, ModuleNotFoundError

- [ ] **Step 3: 实现集成模型预测器**

创建 `src/models/ensemble_predictor.py`:

```python
"""
集成模型预测器
==============

加载Phase 2训练好的15个LightGBM模型，使用IC加权融合生成Alpha预测。
"""

import numpy as np
import pandas as pd
import os
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False


class EnsemblePredictor:
    """
    集成模型预测器
    
    加载15个基础LightGBM模型 + IC加权元学习器，生成Alpha预测。
    """
    
    def __init__(self, model_dir: str):
        """
        初始化
        
        Args:
            model_dir: 模型目录路径，包含15个模型文件和权重文件
        """
        if not LGBM_AVAILABLE:
            raise ImportError("LightGBM未安装，请运行: pip install lightgbm")
        
        self.model_dir = model_dir
        self.base_models = {}
        self.weights = None
        
        # 窗口和配置
        self.windows = [1, 3, 5, 10, 20]
        self.configs = ['default', 'regularized', 'deep']
        
        # 加载模型
        self._load_models()
    
    def _load_models(self):
        """加载所有基础模型和权重"""
        print(f"加载集成模型从: {self.model_dir}")
        
        # 加载15个基础模型
        model_count = 0
        for window in self.windows:
            for config in self.configs:
                model_key = f'w{window}_{config}'
                model_path = os.path.join(self.model_dir, f'{model_key}.txt')
                
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"模型文件不存在: {model_path}")
                
                # 加载LightGBM模型
                model = lgb.Booster(model_file=model_path)
                self.base_models[model_key] = model
                model_count += 1
        
        print(f"  ✓ 加载 {model_count} 个基础模型")
        
        # 加载IC权重
        weights_path = os.path.join(self.model_dir, 'ic_weights.npy')
        if os.path.exists(weights_path):
            self.weights = np.load(weights_path)
            print(f"  ✓ 加载IC权重")
        else:
            # 如果没有权重文件，使用等权
            print(f"  ⚠ 权重文件不存在，使用等权")
            self.weights = np.ones(15) / 15
    
    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        生成Alpha预测
        
        Args:
            features: 特征DataFrame [n_stocks, 9]
                     必须包含列: momentum_5, momentum_10, momentum_20,
                                 volatility_10, volatility_20,
                                 volume_ratio, atr_ratio, ma_ratio, rsi14
                     可选列: ts_code (股票代码)
        
        Returns:
            预测结果 DataFrame: {ts_code, alpha}
        """
        # 提取特征列
        feature_cols = [
            'momentum_5', 'momentum_10', 'momentum_20',
            'volatility_10', 'volatility_20',
            'volume_ratio', 'atr_ratio', 'ma_ratio', 'rsi14'
        ]
        
        # 验证特征列存在
        missing_cols = set(feature_cols) - set(features.columns)
        if missing_cols:
            raise ValueError(f"缺失特征列: {missing_cols}")
        
        # 提取特征矩阵
        X = features[feature_cols].values
        
        # 检查NaN
        if np.isnan(X).any():
            # Forward fill处理NaN
            features_filled = features[feature_cols].fillna(method='ffill').fillna(0)
            X = features_filled.values
        
        # 收集所有基础模型预测
        base_predictions = []
        
        for window in self.windows:
            for config in self.configs:
                model_key = f'w{window}_{config}'
                model = self.base_models[model_key]
                
                # 预测
                pred = model.predict(X)
                base_predictions.append(pred)
        
        # 堆叠成矩阵 [n_stocks, 15]
        base_predictions = np.column_stack(base_predictions)
        
        # IC加权融合
        alpha = (base_predictions * self.weights).sum(axis=1)
        
        # 构建结果DataFrame
        result = pd.DataFrame({
            'alpha': alpha
        })
        
        # 如果有ts_code，添加到结果
        if 'ts_code' in features.columns:
            result['ts_code'] = features['ts_code'].values
            result = result[['ts_code', 'alpha']]  # 调整列顺序
        
        return result
    
    def save_weights(self, weights: np.ndarray, filepath: str):
        """
        保存IC权重
        
        Args:
            weights: IC权重数组 [15]
            filepath: 保存路径
        """
        np.save(filepath, weights)
        print(f"✓ 权重已保存到: {filepath}")
```

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/models/test_ensemble_predictor.py -v
```

Expected: PASS (如果模型文件存在) 或 SKIP (如果模型文件不存在)

- [ ] **Step 5: 提交代码**

```bash
git add src/models/ensemble_predictor.py tests/models/test_ensemble_predictor.py
git commit -m "feat: 实现集成模型预测器

- 加载15个LightGBM基础模型
- IC加权融合生成Alpha预测
- 处理缺失值和异常情况
- 完整单元测试"
```

---

(计划继续... 由于长度限制，后续任务将在下一个文件中继续)

## Task 3: 配置文件与回测引擎改造

**Files:**
- Create: `configs/phase3_config.yaml`
- Modify: `src/backtest/backtest_engine.py:66-76`

- [ ] **Step 1: 创建配置文件**

创建 `configs/phase3_config.yaml`:

```yaml
# Phase 3 回测配置

# 回测参数
backtest:
  start_date: '2024-12-28'
  end_date: '2025-12-31'
  initial_capital: 10000000  # 1000万
  rebalance_freq: 'W-MON'    # 每周一

# 风险估计参数
risk:
  estimation_window: 252      # 252天≈1年
  method: 'sample'            # 样本协方差

# 组合优化参数
optimization:
  risk_aversion: 2.0          # 风险厌恶系数
  max_tracking_error: 0.05    # 5%跟踪误差上限
  max_turnover: 0.30          # 30%换手率上限
  max_weight: 0.05            # 5%个股权重上限
  min_weight: 0.0

# 交易成本参数
trading:
  commission_rate: 0.0015     # 0.15%双边
  slippage: 0.0005            # 0.05%滑点
  price_limit: 0.20           # ±20%涨跌停

# 集成模型参数
ensemble:
  model_dir: 'models/phase2_ensemble/'
  n_base_models: 15
  windows: [1, 3, 5, 10, 20]

# 基准参数
benchmark:
  type: 'equal_weight'        # 等权基准
  universe: 'star50'          # 科创50

# 输出参数
output:
  results_dir: 'results/phase3/'
  save_trades: true
  save_positions: true
  save_plots: true
```

- [ ] **Step 2: 验证配置文件格式**

```bash
cd star50-quant
python -c "import yaml; yaml.safe_load(open('configs/phase3_config.yaml'))"
```

Expected: 无输出（格式正确）

- [ ] **Step 3: 修改回测引擎支持周度再平衡**

修改 `src/backtest/backtest_engine.py`，在 `run_backtest` 方法中添加再平衡频率过滤逻辑。

找到这段代码（约第66-76行）:

```python
        for i, date in enumerate(trade_dates):
            # 获取当日目标权重
            target_weights = weights_series[weights_series['date'] == date]

            if len(target_weights) == 0:
                continue
```

替换为:

```python
        # 判断再平衡日期
        rebalance_dates = self._get_rebalance_dates(trade_dates, rebalance_freq)
        
        for i, date in enumerate(trade_dates):
            # 只在再平衡日触发交易
            if date in rebalance_dates:
                # 获取当日目标权重
                target_weights = weights_series[weights_series['date'] == date]

                if len(target_weights) == 0:
                    continue
            else:
                # 非再平衡日，只更新市值
                target_weights = pd.DataFrame()  # 空DataFrame，不触发交易
```

- [ ] **Step 4: 添加再平衡日期计算方法**

在 `BacktestEngine` 类中添加新方法（约第277行后）:

```python
    def _get_rebalance_dates(
        self,
        trade_dates: List,
        freq: str
    ) -> set:
        """
        计算再平衡日期
        
        Args:
            trade_dates: 所有交易日列表
            freq: 再平衡频率
                  'D' = 每日
                  'W-MON' = 每周一
                  'M' = 每月第一个交易日
        
        Returns:
            再平衡日期集合
        """
        if freq == 'D':
            # 每日再平衡
            return set(trade_dates)
        
        elif freq == 'W-MON':
            # 每周一再平衡
            rebalance_dates = set()
            df = pd.DataFrame({'date': trade_dates})
            df['date'] = pd.to_datetime(df['date'])
            df['weekday'] = df['date'].dt.dayofweek
            
            # 找到每周的第一个交易日（周一=0）
            df['week'] = df['date'].dt.isocalendar().week
            df['year'] = df['date'].dt.year
            
            for (year, week), group in df.groupby(['year', 'week']):
                # 取该周第一个交易日
                first_date = group['date'].min()
                rebalance_dates.add(first_date.strftime('%Y-%m-%d'))
            
            return rebalance_dates
        
        elif freq == 'M':
            # 每月第一个交易日
            rebalance_dates = set()
            df = pd.DataFrame({'date': trade_dates})
            df['date'] = pd.to_datetime(df['date'])
            df['month'] = df['date'].dt.to_period('M')
            
            for month, group in df.groupby('month'):
                first_date = group['date'].min()
                rebalance_dates.add(first_date.strftime('%Y-%m-%d'))
            
            return rebalance_dates
        
        else:
            raise ValueError(f"不支持的再平衡频率: {freq}")
```

- [ ] **Step 5: 测试回测引擎改造**

创建简单测试脚本验证:

```bash
cd star50-quant
python -c "
from src.backtest.backtest_engine import BacktestEngine
import pandas as pd

engine = BacktestEngine()
dates = pd.date_range('2024-01-01', '2024-01-31', freq='D')
dates_str = [d.strftime('%Y-%m-%d') for d in dates if d.dayofweek < 5]

rebalance = engine._get_rebalance_dates(dates_str, 'W-MON')
print(f'交易日数: {len(dates_str)}')
print(f'再平衡日数: {len(rebalance)}')
print(f'再平衡日期: {sorted(rebalance)}')
"
```

Expected: 输出约4-5个周一日期

- [ ] **Step 6: 提交代码**

```bash
git add configs/phase3_config.yaml src/backtest/backtest_engine.py
git commit -m "feat: 添加Phase 3配置文件和周度再平衡支持

- 完整配置文件（回测、风险、优化、交易参数）
- 回测引擎支持W-MON周度再平衡
- 添加再平衡日期计算方法"
```

---

## Task 4: 主回测脚本

**Files:**
- Create: `scripts/run_phase3_backtest.py`

- [ ] **Step 1: 实现主回测脚本框架**

创建 `scripts/run_phase3_backtest.py`:

```python
#!/usr/bin/env python3
"""
Phase 3 完整回测脚本
===================

端到端流程：
1. 加载数据
2. 初始化组件（预测器、风险估计器、优化器、回测引擎）
3. 逐周生成Alpha预测和组合权重
4. 运行回测
5. 计算评估指标
6. 可视化结果
7. 验收检查
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from src.models.ensemble_predictor import EnsemblePredictor
from src.risk.covariance_estimator import CovarianceEstimator
from src.optimization.portfolio_optimizer import PortfolioOptimizer
from src.backtest.backtest_engine import BacktestEngine


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_data(start_date: str, end_date: str):
    """
    加载回测数据
    
    Returns:
        (prices_df, features_df)
    """
    print("="*80)
    print("1. 加载数据")
    print("="*80)
    
    # 加载价格数据
    prices_path = 'data/raw/star50_daily_hfq_data_6yrs.parquet'
    prices = pd.read_parquet(prices_path)
    
    # 过滤日期范围（扩展到start_date之前252天，用于协方差估计）
    start_expanded = pd.to_datetime(start_date) - pd.Timedelta(days=365)
    prices = prices[
        (prices['trade_date'] >= start_expanded.strftime('%Y-%m-%d')) &
        (prices['trade_date'] <= end_date)
    ]
    
    print(f"  价格数据: {len(prices)} 行, {prices['ts_code'].nunique()} 只股票")
    print(f"  日期范围: {prices['trade_date'].min()} ~ {prices['trade_date'].max()}")
    
    return prices


def calculate_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    计算9个核心因子
    
    Args:
        prices: 价格数据
    
    Returns:
        features DataFrame: {ts_code, factor_date, 9个因子}
    """
    print("\n计算核心因子...")
    
    result = []
    
    for ts_code, group in prices.groupby('ts_code'):
        group = group.sort_values('trade_date').copy()
        
        # 9个核心因子（与validate_final.py相同）
        group['momentum_5'] = group['close'].pct_change(5)
        group['momentum_10'] = group['close'].pct_change(10)
        group['momentum_20'] = group['close'].pct_change(20)
        
        group['volatility_10'] = group['close'].pct_change().rolling(10).std()
        group['volatility_20'] = group['close'].pct_change().rolling(20).std()
        
        group['volume_ratio'] = group['vol'] / group['vol'].rolling(20).mean()
        
        high_low = group['high'] - group['low']
        high_close = np.abs(group['high'] - group['close'].shift())
        low_close = np.abs(group['low'] - group['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        group['atr_ratio'] = true_range.rolling(14).mean() / group['close']
        
        group['ma5'] = group['close'].rolling(5).mean()
        group['ma20'] = group['close'].rolling(20).mean()
        group['ma_ratio'] = group['ma5'] / group['ma20']
        
        delta = group['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-6)
        group['rsi14'] = 100 - (100 / (1 + rs))
        
        group['factor_date'] = group['trade_date']
        result.append(group)
    
    features = pd.concat(result, ignore_index=True)
    
    # 选择特征列
    feature_cols = [
        'ts_code', 'factor_date',
        'momentum_5', 'momentum_10', 'momentum_20',
        'volatility_10', 'volatility_20',
        'volume_ratio', 'atr_ratio', 'ma_ratio', 'rsi14'
    ]
    
    features = features[feature_cols].dropna()
    
    print(f"  特征数据: {len(features)} 行")
    
    return features


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    计算日收益率
    
    Returns:
        returns DataFrame: {trade_date, ts_code, return}
    """
    returns_list = []
    
    for ts_code, group in prices.groupby('ts_code'):
        group = group.sort_values('trade_date').copy()
        group['return'] = group['close'].pct_change()
        returns_list.append(group[['trade_date', 'ts_code', 'return']])
    
    returns = pd.concat(returns_list, ignore_index=True)
    return returns


def calculate_benchmark_weights(prices: pd.DataFrame) -> pd.DataFrame:
    """
    计算基准权重（科创50等权）
    
    Returns:
        {ts_code, weight}
    """
    stocks = prices['ts_code'].unique()
    n_stocks = len(stocks)
    
    benchmark = pd.DataFrame({
        'ts_code': stocks,
        'weight': 1.0 / n_stocks
    })
    
    return benchmark


def get_rebalance_dates(start_date: str, end_date: str, freq: str = 'W-MON') -> list:
    """
    生成再平衡日期列表
    
    Args:
        start_date: 开始日期
        end_date: 结束日期
        freq: 频率 ('W-MON' = 每周一)
    
    Returns:
        日期列表
    """
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    return [d.strftime('%Y-%m-%d') for d in dates]


def validate_phase3(metrics: dict):
    """
    Phase 3验收检查
    
    验收标准:
    - 年化收益 >15%
    - 跟踪误差 <8%
    - 信息比率 >0.5
    - 最大回撤 <20%
    """
    print("\n" + "="*80)
    print("Phase 3 验收检查")
    print("="*80)
    
    checks = []
    
    # 1. 年化收益
    annual_return = metrics['annual_return']
    check1 = annual_return > 0.15
    checks.append(check1)
    status1 = "✓ PASS" if check1 else "✗ FAIL"
    print(f"\n1. 年化收益 >15%: {annual_return:.2%} ... {status1}")
    
    # 2. 跟踪误差
    if 'tracking_error' in metrics:
        te = metrics['tracking_error']
        check2 = te < 0.08
        checks.append(check2)
        status2 = "✓ PASS" if check2 else "✗ FAIL"
        print(f"2. 跟踪误差 <8%: {te:.2%} ... {status2}")
    else:
        print("2. 跟踪误差: 未计算")
        checks.append(False)
    
    # 3. 信息比率
    if 'information_ratio' in metrics:
        ir = metrics['information_ratio']
        check3 = ir > 0.5
        checks.append(check3)
        status3 = "✓ PASS" if check3 else "✗ FAIL"
        print(f"3. 信息比率 >0.5: {ir:.4f} ... {status3}")
    else:
        print("3. 信息比率: 未计算")
        checks.append(False)
    
    # 4. 最大回撤
    max_dd = metrics['max_drawdown']
    check4 = max_dd > -0.20
    checks.append(check4)
    status4 = "✓ PASS" if check4 else "✗ FAIL"
    print(f"4. 最大回撤 <20%: {max_dd:.2%} ... {status4}")
    
    # 总结
    print("\n" + "-"*80)
    if all(checks):
        print("✓ Phase 3验收通过！")
    else:
        print("✗ Phase 3验收未通过")
        print(f"通过: {sum(checks)}/{len(checks)} 项")


def main():
    """主函数"""
    print("\n" + "="*80)
    print("Phase 3: 完整回测验证")
    print("="*80)
    
    # 加载配置
    config = load_config('configs/phase3_config.yaml')
    print(f"\n配置加载成功")
    print(f"  回测期间: {config['backtest']['start_date']} ~ {config['backtest']['end_date']}")
    print(f"  再平衡频率: {config['backtest']['rebalance_freq']}")
    
    # 加载数据
    prices = load_data(
        config['backtest']['start_date'],
        config['backtest']['end_date']
    )
    
    # 计算特征
    features = calculate_features(prices)
    
    # 计算收益率
    returns = calculate_returns(prices)
    
    # 计算基准权重
    benchmark_weights = calculate_benchmark_weights(prices)
    print(f"\n基准权重: 科创50等权 ({len(benchmark_weights)}只股票)")
    
    # 初始化组件
    print("\n" + "="*80)
    print("2. 初始化组件")
    print("="*80)
    
    predictor = EnsemblePredictor(config['ensemble']['model_dir'])
    print("  ✓ 集成模型预测器")
    
    risk_estimator = CovarianceEstimator(
        window=config['risk']['estimation_window']
    )
    print("  ✓ 协方差估计器")
    
    optimizer = PortfolioOptimizer(
        risk_aversion=config['optimization']['risk_aversion'],
        max_weight=config['optimization']['max_weight'],
        max_turnover=config['optimization']['max_turnover']
    )
    print("  ✓ 组合优化器")
    
    backtester = BacktestEngine(
        initial_capital=config['backtest']['initial_capital'],
        commission_rate=config['trading']['commission_rate'],
        slippage=config['trading']['slippage'],
        price_limit=config['trading']['price_limit']
    )
    print("  ✓ 回测引擎")
    
    # 生成再平衡日期
    rebalance_dates = get_rebalance_dates(
        config['backtest']['start_date'],
        config['backtest']['end_date'],
        config['backtest']['rebalance_freq']
    )
    print(f"\n再平衡日期: {len(rebalance_dates)} 个")
    
    # 逐周生成权重
    print("\n" + "="*80)
    print("3. 生成组合权重")
    print("="*80)
    
    weights_series = []
    previous_weights = None
    
    for i, date in enumerate(rebalance_dates):
        print(f"\n[{i+1}/{len(rebalance_dates)}] {date}")
        
        # 3.1 风险估计
        date_dt = pd.to_datetime(date)
        lookback_start = (date_dt - pd.Timedelta(days=400)).strftime('%Y-%m-%d')
        
        historical_returns = returns[
            (returns['trade_date'] >= lookback_start) &
            (returns['trade_date'] < date)
        ]
        
        # 转为宽表
        returns_pivot = historical_returns.pivot(
            index='trade_date',
            columns='ts_code',
            values='return'
        )
        
        if len(returns_pivot) < 50:
            print(f"  ⚠ 数据不足，跳过")
            continue
        
        covariance = risk_estimator.estimate(returns_pivot)
        print(f"  ✓ 协方差估计 ({len(returns_pivot)}天)")
        
        # 3.2 Alpha预测
        daily_features = features[features['factor_date'] == date]
        
        if len(daily_features) == 0:
            print(f"  ⚠ 无特征数据，跳过")
            continue
        
        alpha_pred = predictor.predict(daily_features)
        print(f"  ✓ Alpha预测 ({len(alpha_pred)}只股票)")
        
        # 3.3 组合优化
        result = optimizer.optimize_with_tracking_error(
            alpha=alpha_pred['alpha'].values,
            covariance=covariance,
            benchmark_weights=benchmark_weights['weight'].values,
            max_tracking_error=config['optimization']['max_tracking_error'],
            previous_weights=previous_weights
        )
        
        if result['status'] not in ['optimal', 'optimal_inaccurate']:
            print(f"  ⚠ 优化失败: {result['status']}")
            continue
        
        print(f"  ✓ 组合优化 (TE={result['tracking_error']:.2%})")
        
        # 记录权重
        for j, ts_code in enumerate(alpha_pred['ts_code']):
            weights_series.append({
                'date': date,
                'ts_code': ts_code,
                'weight': result['weights'][j],
                'alpha': alpha_pred['alpha'].values[j]
            })
        
        previous_weights = result['weights']
    
    weights_df = pd.DataFrame(weights_series)
    print(f"\n✓ 生成 {len(rebalance_dates)} 期权重")
    
    # 运行回测
    print("\n" + "="*80)
    print("4. 运行回测")
    print("="*80)
    
    backtest_results = backtester.run_backtest(
        weights_series=weights_df,
        prices=prices,
        benchmark_weights=benchmark_weights,
        rebalance_freq=config['backtest']['rebalance_freq']
    )
    
    # 计算指标
    print("\n" + "="*80)
    print("5. 评估指标")
    print("="*80)
    
    metrics = backtester.calculate_metrics(backtest_results['portfolio'])
    
    # 验收检查
    validate_phase3(metrics)
    
    # 保存结果
    output_dir = config['output']['results_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    backtest_results['portfolio'].to_csv(
        os.path.join(output_dir, 'backtest_results.csv'),
        index=False
    )
    print(f"\n✓ 结果已保存到: {output_dir}")
    
    return backtest_results, metrics


if __name__ == '__main__':
    results, metrics = main()
```

- [ ] **Step 2: 测试脚本语法**

```bash
cd star50-quant
python -m py_compile scripts/run_phase3_backtest.py
```

Expected: 无输出（语法正确）

- [ ] **Step 3: 提交代码**

```bash
git add scripts/run_phase3_backtest.py
git commit -m "feat: 实现Phase 3主回测脚本

- 端到端流程：数据加载→组件初始化→权重生成→回测→评估
- 集成预测器、风险估计器、优化器、回测引擎
- 自动验收检查
- 结果保存"
```

---

(计划继续...)

## Task 5: 集成测试与执行

**Files:**
- Create: `tests/integration/test_phase3_pipeline.py`
- Execute: `scripts/run_phase3_backtest.py`

- [ ] **Step 1: 创建集成测试**

创建 `tests/integration/test_phase3_pipeline.py`:

```python
"""
Phase 3 端到端集成测试
"""

import numpy as np
import pandas as pd
import pytest
import os
from src.models.ensemble_predictor import EnsemblePredictor
from src.risk.covariance_estimator import CovarianceEstimator
from src.optimization.portfolio_optimizer import PortfolioOptimizer


class TestPhase3Pipeline:
    """Phase 3管道集成测试"""
    
    def test_end_to_end_single_period(self):
        """测试单期完整流程"""
        # 1. 模拟特征数据
        features = pd.DataFrame({
            'ts_code': [f'stock{i}' for i in range(10)],
            'momentum_5': np.random.randn(10) * 0.02,
            'momentum_10': np.random.randn(10) * 0.03,
            'momentum_20': np.random.randn(10) * 0.04,
            'volatility_10': np.abs(np.random.randn(10)) * 0.02,
            'volatility_20': np.abs(np.random.randn(10)) * 0.03,
            'volume_ratio': 0.8 + np.random.rand(10) * 0.4,
            'atr_ratio': np.abs(np.random.randn(10)) * 0.01,
            'ma_ratio': 0.95 + np.random.rand(10) * 0.1,
            'rsi14': 40 + np.random.rand(10) * 20
        })
        
        # 2. 模拟收益率数据（50天 × 10只股票）
        returns = pd.DataFrame(
            np.random.randn(50, 10) * 0.02,
            columns=[f'stock{i}' for i in range(10)]
        )
        
        # 3. 风险估计
        risk_estimator = CovarianceEstimator(window=50)
        covariance = risk_estimator.estimate(returns)
        
        assert covariance.shape == (10, 10)
        assert np.all(np.linalg.eigvals(covariance) > 0)
        
        # 4. Alpha预测（使用随机alpha代替真实预测）
        alpha = np.random.randn(10) * 0.01
        
        # 5. 组合优化
        optimizer = PortfolioOptimizer(
            risk_aversion=2.0,
            max_weight=0.15,
            max_turnover=0.5
        )
        
        benchmark_weights = np.ones(10) / 10
        
        result = optimizer.optimize_with_tracking_error(
            alpha=alpha,
            covariance=covariance,
            benchmark_weights=benchmark_weights,
            max_tracking_error=0.05
        )
        
        # 验证结果
        assert result['status'] in ['optimal', 'optimal_inaccurate']
        assert np.isclose(result['weights'].sum(), 1.0)
        assert np.all(result['weights'] >= 0)
        assert result['tracking_error'] <= 0.06  # 允许小误差
    
    def test_optimization_convergence(self):
        """测试优化器收敛性"""
        n_stocks = 20
        
        # 模拟数据
        alpha = np.random.randn(n_stocks) * 0.02
        returns = pd.DataFrame(
            np.random.randn(100, n_stocks) * 0.015,
            columns=[f'stock{i}' for i in range(n_stocks)]
        )
        
        risk_estimator = CovarianceEstimator(window=100)
        covariance = risk_estimator.estimate(returns)
        
        optimizer = PortfolioOptimizer(
            risk_aversion=1.5,
            max_weight=0.10,
            max_turnover=0.4
        )
        
        benchmark_weights = np.ones(n_stocks) / n_stocks
        
        # 优化
        result = optimizer.optimize_with_tracking_error(
            alpha=alpha,
            covariance=covariance,
            benchmark_weights=benchmark_weights,
            max_tracking_error=0.05
        )
        
        # 验证收敛
        assert result['status'] in ['optimal', 'optimal_inaccurate']
        
        # 验证约束满足
        assert np.isclose(result['weights'].sum(), 1.0, atol=1e-4)
        assert np.all(result['weights'] >= -1e-6)  # 允许小的负值误差
        assert np.all(result['weights'] <= 0.11)   # max_weight + 误差
```

- [ ] **Step 2: 运行集成测试**

```bash
cd star50-quant
pytest tests/integration/test_phase3_pipeline.py -v
```

Expected: 全部PASS

- [ ] **Step 3: 保存Phase 2集成模型权重**

在运行Phase 3之前，需要先保存Phase 2训练好的IC权重：

```bash
cd star50-quant
python -c "
import numpy as np
import os

# Phase 2训练得到的IC权重（从train_ensemble.py输出）
weights = np.array([
    0.0876, 0.0843, 0.0809,  # w1
    0.0444, 0.0539, 0.0280,  # w3
    0.0511, 0.0691, 0.0595,  # w5
    0.0903, 0.1083, 0.1053,  # w10
    0.0427, 0.0532, 0.0413   # w20
])

# 创建模型目录
os.makedirs('models/phase2_ensemble', exist_ok=True)

# 保存权重
np.save('models/phase2_ensemble/ic_weights.npy', weights)
print('✓ IC权重已保存')
print(f'权重和: {weights.sum():.4f}')
"
```

Expected: 
```
✓ IC权重已保存
权重和: 1.0000
```

- [ ] **Step 4: 执行完整回测**

```bash
cd star50-quant
python scripts/run_phase3_backtest.py
```

Expected: 完整输出，包括：
- 数据加载
- 组件初始化
- 逐周生成权重
- 回测执行
- 评估指标
- 验收检查

观察关键指标：
- 年化收益 >15%?
- 跟踪误差 <8%?
- 信息比率 >0.5?
- 最大回撤 <20%?

- [ ] **Step 5: 分析结果并调试**

如果验收未通过，检查：

**年化收益过低 (<15%)**
- 检查IC是否衰减：查看Alpha预测vs实际收益相关性
- 检查交易成本：累计成本是否过高
- 调整风险厌恶系数λ：降低λ提高风险敞口

**跟踪误差过大 (>8%)**
- 检查优化约束：5%可能过紧
- 查看优化失败次数
- 尝试放宽至6-7%

**信息比率过低 (<0.5)**
- IC转化效率问题
- 检查换手率：是否过高导致成本侵蚀
- 分析归因：Alpha贡献vs风险成本

调试命令：
```bash
# 检查中间结果
cd star50-quant
python -c "
import pandas as pd
results = pd.read_csv('results/phase3/backtest_results.csv')
print('净值曲线:')
print(results[['date', 'portfolio_value', 'cumulative_return']].head(10))
"
```

- [ ] **Step 6: 提交最终代码**

```bash
git add tests/integration/ models/phase2_ensemble/ic_weights.npy
git commit -m "test: 添加Phase 3集成测试并保存模型权重

- 端到端管道测试
- 优化器收敛性测试
- Phase 2 IC权重保存"
```

- [ ] **Step 7: 生成Phase 3结果报告**

创建 `docs/superpowers/reports/2026-06-10-phase3-results.md`，记录：

```markdown
# Phase 3 回测结果报告

## 回测配置
- 回测期间: 2024-12-28 ~ 2025-12-31
- 再平衡频率: 每周一
- 初始资金: 1000万

## 关键指标
- 年化收益: X.XX%
- 夏普比率: X.XX
- 最大回撤: X.XX%
- 跟踪误差: X.XX%
- 信息比率: X.XX

## 验收结果
- [ ] 年化收益 >15%
- [ ] 跟踪误差 <8%
- [ ] 信息比率 >0.5
- [ ] 最大回撤 <20%

## 分析与建议
...
```

- [ ] **Step 8: 最终提交**

```bash
git add docs/superpowers/reports/2026-06-10-phase3-results.md results/phase3/
git commit -m "docs: 添加Phase 3回测结果报告

- 完整回测指标
- 验收检查结果
- 问题分析与优化建议"
```

---

## 总结

### 完成标准

**代码交付**
- ✓ 协方差估计器 (`src/risk/covariance_estimator.py`)
- ✓ 集成模型预测器 (`src/models/ensemble_predictor.py`)
- ✓ 配置文件 (`configs/phase3_config.yaml`)
- ✓ 主回测脚本 (`scripts/run_phase3_backtest.py`)
- ✓ 回测引擎改造 (`src/backtest/backtest_engine.py`)
- ✓ 完整测试套件

**验收标准**
1. 回测系统正常运行（无报错）
2. 年化收益 >15%
3. 跟踪误差 <8%
4. 信息比率 >0.5
5. 最大回撤 <20%

### 关键技术点

1. **样本协方差估计**: 252天滚动窗口，正则化确保正定性
2. **IC加权融合**: 15个基础模型按验证集IC加权
3. **跟踪误差约束优化**: cvxpy实现均值方差优化
4. **周度再平衡**: 平衡信号利用和交易成本

### 常见问题

**Q: 模型文件不存在？**
A: 确保Phase 2已完成，15个模型文件在 `models/phase2_ensemble/` 目录

**Q: 优化频繁失败？**
A: 检查跟踪误差约束是否过紧，尝试放宽至6-8%

**Q: IC衰减严重？**
A: 外样本IC通常会降低，属于正常现象。如果降至<0.01，需要重新训练模型

**Q: 交易成本过高？**
A: 检查换手率，如果>40%考虑降低再平衡频率或收紧换手率约束

### 下一步

如果Phase 3验收通过，可以进行：
- 风险模型升级（Ledoit-Wolf收缩估计）
- 再平衡频率优化
- Alpha模型改进
- 多资产扩展

如果未通过，根据具体指标调整参数或改进模型。

---

**计划版本**: v1.0  
**创建时间**: 2026-06-10  
**预计工时**: 4-5小时  
**当前状态**: 待执行
