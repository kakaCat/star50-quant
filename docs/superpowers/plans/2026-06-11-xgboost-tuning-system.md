# XGBoost调参系统完整实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现完整的XGBoost调参系统，包括数据准备、三种调参方法、完整回测验证，达到IC>0.04, IR>=1.5, 年化收益>35%, 最大回撤<=-20%的目标。

**Architecture:** 
- 数据层：从PostgreSQL加载因子和价格数据，构造5日超额收益标签
- 特征工程：30个基础因子通过MAD去极值和Z-Score标准化
- 调参层：支持随机搜索、网格搜索、贝叶斯优化三种方法
- 回测层：滚动窗口训练+完整回测验证，支持6种策略组合

**Tech Stack:** 
- XGBoost: 梯度提升模型
- Optuna: 贝叶斯优化框架
- PostgreSQL: 数据存储
- pandas/numpy: 数据处理
- scikit-learn: 交叉验证和评估

---

## 文件结构

### 核心模块文件

**数据加载模块:**
- `star50-quant/src/models/data_loader.py` - 因子数据加载器（已存在，需完善）
  - 负责从数据库加载因子和价格数据
  - 计算5日超额收益标签（个股收益 - 指数收益）
  - 数据预处理（去极值、标准化）

**模型模块:**
- `star50-quant/src/models/xgb_model.py` - XGBoost模型类（已存在，需完善）
  - 模型训练、预测、保存/加载
  - 特征重要性分析
  
**调参模块:**
- `star50-quant/src/models/hyperparameter_tuning.py` - 超参数调优框架（已存在，需完善）
  - ObjectiveFunction: 综合评分函数（IC + IR + 收益 + 回撤）
  - HyperparameterTuner: 三种调参方法统一接口
  - TuningResult: 结果数据类

**回测模块:**
- `star50-quant/src/backtest/strategy_backtest.py` - 完整回测引擎（需新建）
  - 滚动窗口训练（756日训练，20日预测，5日调仓）
  - 6种策略（10%/15%/20% × 等权/信号加权）
  - 性能指标计算（IC, IR, 年化收益, 最大回撤）

### 脚本文件

**调参脚本:**
- `star50-quant/scripts/tune_xgb_model.py` - 调参主脚本（已存在，需完善）
  - 命令行接口
  - 参数空间定义
  - 结果保存

**回测脚本:**
- `star50-quant/scripts/run_complete_backtest.py` - 完整回测脚本（需新建）
  - 加载最佳模型
  - 执行完整回测
  - 生成报告

### 测试文件

- `tests/models/test_xgb_tuning.py` - 调参系统测试（需新建）
- `tests/backtest/test_strategy_backtest.py` - 回测引擎测试（需新建）

---

## Task 1: 完善数据加载器 - 标签构造

**Files:**
- Modify: `star50-quant/src/models/data_loader.py:95-150`
- Test: `tests/models/test_data_loader.py`

**目标:** 正确构造5日超额收益标签，确保数据处理顺序正确（先清洗标签，再标准化特征）

- [ ] **Step 1: 添加指数数据加载方法**

在 `FactorDataLoader` 类中添加方法：

```python
def load_index_prices(
    self,
    index_code: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    加载指数价格数据
    
    Args:
        index_code: 指数代码（如'000688.SH'表示科创50）
        start_date: 开始日期
        end_date: 结束日期
    
    Returns:
        DataFrame，列：trade_date, close
    """
    self.connect()
    
    query = """
        SELECT trade_date, close
        FROM index_daily
        WHERE ts_code = %s 
        AND trade_date >= %s 
        AND trade_date <= %s
        ORDER BY trade_date
    """
    
    df = pd.read_sql(query, self.conn, params=[index_code, start_date, end_date])
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    return df
```

- [ ] **Step 2: 添加超额收益计算方法**

```python
def calculate_excess_returns(
    self,
    stock_prices: pd.DataFrame,
    index_prices: pd.DataFrame,
    forward_days: int = 5
) -> pd.DataFrame:
    """
    计算个股相对指数的超额收益
    
    Args:
        stock_prices: 个股价格数据
        index_prices: 指数价格数据
        forward_days: 前向收益天数
    
    Returns:
        DataFrame，列：ts_code, trade_date, forward_return (超额收益)
    """
    # 计算个股未来N日收益
    stock_returns = []
    
    for ts_code, group in stock_prices.groupby('ts_code'):
        group = group.sort_values('trade_date').reset_index(drop=True)
        group['future_close'] = group['close'].shift(-forward_days)
        group['stock_return'] = (group['future_close'] / group['close'] - 1)
        stock_returns.append(group[['ts_code', 'trade_date', 'stock_return']])
    
    stock_df = pd.concat(stock_returns, ignore_index=True)
    
    # 计算指数未来N日收益
    index_df = index_prices.sort_values('trade_date').reset_index(drop=True)
    index_df['future_close'] = index_df['close'].shift(-forward_days)
    index_df['index_return'] = (index_df['future_close'] / index_df['close'] - 1)
    index_df = index_df[['trade_date', 'index_return']]
    
    # 合并计算超额收益
    result = stock_df.merge(index_df, on='trade_date', how='inner')
    result['forward_return'] = result['stock_return'] - result['index_return']
    
    return result[['ts_code', 'trade_date', 'forward_return']].dropna()
```

- [ ] **Step 3: 添加数据清洗方法（正确顺序）**

```python
def prepare_training_data(
    self,
    features: pd.DataFrame,
    labels: pd.DataFrame
) -> pd.DataFrame:
    """
    准备训练数据（正确的处理顺序）
    
    处理顺序：
    1. 合并特征和标签
    2. 清洗标签（去极值）- 不标准化！
    3. 标准化特征（不包括标签）
    4. 返回最终数据
    
    Args:
        features: 特征DataFrame
        labels: 标签DataFrame（包含forward_return）
    
    Returns:
        处理后的DataFrame
    """
    from scipy import stats
    
    # 1. 合并数据
    data = features.merge(
        labels[['ts_code', 'factor_date', 'forward_return']],
        on=['ts_code', 'factor_date'],
        how='inner'
    ).dropna()
    
    print(f"原始样本数: {len(data)}")
    
    # 2. 清洗forward_return（去除极端值）
    data = data[
        (data['forward_return'] > -0.5) & 
        (data['forward_return'] < 0.5)
    ]
    print(f"过滤后样本数: {len(data)}")
    
    # Winsorize标签（1%分位数截断）
    data['forward_return_clean'] = stats.mstats.winsorize(
        data['forward_return'].values,
        limits=[0.01, 0.01]
    )
    
    print(f"Forward return统计:")
    print(f"  Mean: {data['forward_return_clean'].mean():.4f}")
    print(f"  Std: {data['forward_return_clean'].std():.4f}")
    print(f"  Min: {data['forward_return_clean'].min():.4f}")
    print(f"  Max: {data['forward_return_clean'].max():.4f}")
    
    # 3. 标准化特征（不包括标签）
    feature_cols = [
        col for col in data.columns
        if col not in ['ts_code', 'factor_date', 'forward_return', 'forward_return_clean']
    ]
    
    # 只对特征列进行winsorize和standardize
    data_features = data[['ts_code', 'factor_date'] + feature_cols].copy()
    data_features = self.winsorize(data_features)
    data_features = self.standardize(data_features)
    
    # 4. 重新合并清洗后的标签
    data_final = data_features.merge(
        data[['ts_code', 'factor_date', 'forward_return_clean']],
        on=['ts_code', 'factor_date'],
        how='inner'
    )
    
    # 重命名标签列
    data_final = data_final.rename(columns={'forward_return_clean': 'forward_return'})
    
    return data_final
```

- [ ] **Step 4: 编写测试**

创建 `tests/models/test_data_loader.py`:

```python
import pytest
import pandas as pd
import numpy as np
from src.models.data_loader import FactorDataLoader


def test_calculate_excess_returns():
    """测试超额收益计算"""
    # 模拟股票价格数据
    stock_prices = pd.DataFrame({
        'ts_code': ['688001.SH'] * 10,
        'trade_date': pd.date_range('2023-01-01', periods=10),
        'close': [100, 102, 101, 105, 103, 107, 106, 110, 108, 112]
    })
    
    # 模拟指数价格数据
    index_prices = pd.DataFrame({
        'trade_date': pd.date_range('2023-01-01', periods=10),
        'close': [1000, 1010, 1005, 1020, 1015, 1030, 1025, 1040, 1035, 1050]
    })
    
    loader = FactorDataLoader()
    result = loader.calculate_excess_returns(stock_prices, index_prices, forward_days=5)
    
    # 验证结果
    assert 'forward_return' in result.columns
    assert len(result) > 0
    assert result['forward_return'].notna().all()
    
    # 验证超额收益的计算逻辑
    # 第一个数据点: (107/100 - 1) - (1030/1000 - 1) = 0.07 - 0.03 = 0.04
    expected_excess = 0.04
    actual_excess = result.iloc[0]['forward_return']
    assert abs(actual_excess - expected_excess) < 0.001


def test_prepare_training_data():
    """测试训练数据准备（正确的处理顺序）"""
    # 模拟特征数据
    features = pd.DataFrame({
        'ts_code': ['688001.SH'] * 100,
        'factor_date': pd.date_range('2023-01-01', periods=100),
        'ma5': np.random.randn(100),
        'ma10': np.random.randn(100),
        'rsi': np.random.uniform(0, 100, 100)
    })
    
    # 模拟标签数据（包含一些极端值）
    labels = pd.DataFrame({
        'ts_code': ['688001.SH'] * 100,
        'factor_date': pd.date_range('2023-01-01', periods=100),
        'forward_return': np.concatenate([
            np.random.randn(98) * 0.05,  # 正常值
            [-0.8, 0.9]  # 极端值
        ])
    })
    
    loader = FactorDataLoader()
    result = loader.prepare_training_data(features, labels)
    
    # 验证极端值被过滤
    assert result['forward_return'].abs().max() <= 0.5
    
    # 验证样本数减少（极端值被移除）
    assert len(result) < len(features)
    
    # 验证特征被标准化（均值接近0，标准差接近1）
    feature_cols = ['ma5', 'ma10', 'rsi']
    for col in feature_cols:
        assert abs(result[col].mean()) < 0.1
        assert abs(result[col].std() - 1.0) < 0.2
    
    # 验证标签没有被标准化（保持原始尺度）
    assert result['forward_return'].std() > 0.01  # 应该保持原始波动率


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

- [ ] **Step 5: 运行测试验证**

```bash
cd /Users/mac/Documents/ai/bisai/star50-quant
pytest tests/models/test_data_loader.py::test_calculate_excess_returns -v
pytest tests/models/test_data_loader.py::test_prepare_training_data -v
```

预期输出: 所有测试PASS

- [ ] **Step 6: 提交代码**

```bash
git add star50-quant/src/models/data_loader.py
git add tests/models/test_data_loader.py
git commit -m "feat: 添加超额收益计算和正确的数据处理顺序

- 新增load_index_prices()加载指数数据
- 新增calculate_excess_returns()计算5日超额收益
- 新增prepare_training_data()确保正确处理顺序
- 关键修复：先清洗标签，再标准化特征
- 添加完整测试覆盖"
```

---


## Task 2: 完善XGBoost模型类

**Files:**
- Modify: `star50-quant/src/models/xgb_model.py:83-120`
- Test: `tests/models/test_xgb_model.py`

**目标:** 确保XGBoost模型正确处理特征和标签，支持保存/加载

- [ ] **Step 1: 修复prepare_data方法**

在 `XGBoostAlphaModel` 类中修改:

```python
def prepare_data(
    self,
    features: pd.DataFrame,
    labels: pd.DataFrame = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    准备训练/预测数据
    
    Args:
        features: 特征DataFrame
        labels: 标签DataFrame（预测时可为None）
    
    Returns:
        (X, y) 如果labels存在，否则 (X, None)
    """
    # 提取特征列（排除元数据列）
    feature_cols = [
        col for col in features.columns
        if col not in ['ts_code', 'factor_date', 'trade_date', 'forward_return']
    ]
    
    self.feature_names = feature_cols
    X = features[feature_cols].values
    
    if labels is not None:
        y = labels['forward_return'].values
        return X, y
    else:
        return X, None
```

- [ ] **Step 2: 添加保存/加载方法**

```python
def save(self, filepath: str):
    """
    保存模型到文件
    
    Args:
        filepath: 保存路径（.json格式）
    """
    import json
    
    if self.model is None:
        raise ValueError("No model to save. Train the model first.")
    
    # 保存XGBoost模型
    model_path = filepath.replace('.json', '.xgb')
    self.model.save_model(model_path)
    
    # 保存元数据
    metadata = {
        'params': self.params,
        'feature_names': self.feature_names,
        'model_path': model_path
    }
    
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model saved to {filepath}")
    print(f"XGBoost model saved to {model_path}")


def load(self, filepath: str):
    """
    从文件加载模型
    
    Args:
        filepath: 模型文件路径（.json格式）
    """
    import json
    
    # 加载元数据
    with open(filepath, 'r') as f:
        metadata = json.load(f)
    
    self.params = metadata['params']
    self.feature_names = metadata['feature_names']
    
    # 加载XGBoost模型
    model_path = metadata['model_path']
    self.model = xgb.Booster()
    self.model.load_model(model_path)
    
    print(f"Model loaded from {filepath}")
```

- [ ] **Step 3: 修复predict方法支持DataFrame输入**

```python
def predict(
    self,
    features: pd.DataFrame
) -> pd.DataFrame:
    """
    预测Alpha
    
    Args:
        features: 特征DataFrame（必须包含ts_code和factor_date）
    
    Returns:
        预测结果DataFrame，列：ts_code, factor_date, predicted_alpha
    """
    if self.model is None:
        raise ValueError("Model not trained. Call train() or load() first.")
    
    # 准备数据
    X, _ = self.prepare_data(features)
    
    # 预测
    dmatrix = xgb.DMatrix(X, feature_names=self.feature_names)
    predictions = self.model.predict(dmatrix)
    
    # 构造结果DataFrame
    result = pd.DataFrame({
        'ts_code': features['ts_code'].values,
        'factor_date': features['factor_date'].values,
        'predicted_alpha': predictions
    })
    
    return result
```

- [ ] **Step 4: 编写测试**

创建 `tests/models/test_xgb_model.py`:

```python
import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from src.models.xgb_model import XGBoostAlphaModel


@pytest.fixture
def sample_data():
    """生成样本数据"""
    n_samples = 200
    features = pd.DataFrame({
        'ts_code': ['688001.SH'] * n_samples,
        'factor_date': pd.date_range('2023-01-01', periods=n_samples),
        'ma5': np.random.randn(n_samples),
        'ma10': np.random.randn(n_samples),
        'rsi': np.random.uniform(0, 100, n_samples)
    })
    
    labels = pd.DataFrame({
        'ts_code': ['688001.SH'] * n_samples,
        'factor_date': pd.date_range('2023-01-01', periods=n_samples),
        'forward_return': np.random.randn(n_samples) * 0.05
    })
    
    return features, labels


def test_prepare_data(sample_data):
    """测试数据准备"""
    features, labels = sample_data
    model = XGBoostAlphaModel()
    
    X, y = model.prepare_data(features, labels)
    
    assert X.shape[0] == len(features)
    assert X.shape[1] == 3  # ma5, ma10, rsi
    assert y.shape[0] == len(labels)
    assert model.feature_names == ['ma5', 'ma10', 'rsi']


def test_train_and_predict(sample_data):
    """测试训练和预测"""
    features, labels = sample_data
    
    # 分割训练/验证集
    split_idx = 150
    train_features = features.iloc[:split_idx]
    train_labels = labels.iloc[:split_idx]
    val_features = features.iloc[split_idx:]
    val_labels = labels.iloc[split_idx:]
    
    # 训练模型
    model = XGBoostAlphaModel()
    model.train(
        train_features,
        train_labels,
        num_boost_round=10,
        val_features=val_features,
        val_labels=val_labels
    )
    
    # 预测
    predictions = model.predict(val_features)
    
    assert len(predictions) == len(val_features)
    assert 'predicted_alpha' in predictions.columns
    assert predictions['predicted_alpha'].notna().all()


def test_save_and_load(sample_data):
    """测试模型保存和加载"""
    features, labels = sample_data
    
    # 训练模型
    model1 = XGBoostAlphaModel()
    model1.train(features, labels, num_boost_round=10)
    
    # 获取预测结果
    pred1 = model1.predict(features)
    
    # 保存模型
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, 'test_model.json')
        model1.save(filepath)
        
        # 加载模型
        model2 = XGBoostAlphaModel()
        model2.load(filepath)
        
        # 验证加载后的预测结果一致
        pred2 = model2.predict(features)
        
        np.testing.assert_array_almost_equal(
            pred1['predicted_alpha'].values,
            pred2['predicted_alpha'].values,
            decimal=5
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

- [ ] **Step 5: 运行测试**

```bash
cd /Users/mac/Documents/ai/bisai/star50-quant
pytest tests/models/test_xgb_model.py -v
```

预期输出: 所有测试PASS

- [ ] **Step 6: 提交代码**

```bash
git add star50-quant/src/models/xgb_model.py
git add tests/models/test_xgb_model.py
git commit -m "feat: 完善XGBoost模型类

- 修复prepare_data()支持预测模式
- 添加save/load方法支持模型持久化
- 修复predict()返回DataFrame格式
- 添加完整测试覆盖"
```

---

## Task 3: 完善超参数调优框架

**Files:**
- Modify: `star50-quant/src/models/hyperparameter_tuning.py:109-250`
- Test: `tests/models/test_hyperparameter_tuning.py`

**目标:** 实现三种调参方法，综合评分函数计算IC、IR、年化收益、最大回撤

- [ ] **Step 1: 完善ObjectiveFunction的evaluate_params方法**

修改 `hyperparameter_tuning.py` 中的 `evaluate_params` 方法:

```python
def evaluate_params(self, params: Dict) -> Dict[str, float]:
    """
    评估参数组合
    
    Args:
        params: XGBoost参数字典
    
    Returns:
        评估指标字典 {ic, ir, annual_return, max_drawdown, composite_score}
    """
    from sklearn.model_selection import TimeSeriesSplit
    
    # 时间序列交叉验证
    tscv = TimeSeriesSplit(n_splits=self.n_splits)
    
    ic_scores = []
    predictions_all = []
    actuals_all = []
    
    for train_idx, val_idx in tscv.split(self.X):
        X_train, X_val = self.X[train_idx], self.X[val_idx]
        y_train, y_val = self.y[train_idx], self.y[val_idx]
        
        # 训练模型
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_names)
        
        xgb_params = {k: v for k, v in params.items() if k != 'num_boost_round'}
        xgb_params.update({
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'seed': 42,
            'verbosity': 0
        })
        
        model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=params.get('num_boost_round', 100),
            evals=[(dval, 'valid')],
            early_stopping_rounds=10,
            verbose_eval=False
        )
        
        # 预测
        y_pred = model.predict(dval)
        
        # 计算IC
        ic = np.corrcoef(y_pred, y_val)[0, 1]
        if not np.isnan(ic):
            ic_scores.append(ic)
            predictions_all.extend(y_pred)
            actuals_all.extend(y_val)
    
    # 平均IC
    mean_ic = np.mean(ic_scores) if ic_scores else 0.0
    
    # 计算IR（简化版：IC标准差的倒数 × IC）
    if len(ic_scores) > 1:
        ic_std = np.std(ic_scores)
        ir = mean_ic / ic_std if ic_std > 0 else 0.0
    else:
        ir = 0.0
    
    # 简化的策略收益估算（用于调参阶段的粗略评估）
    # 注意：这只是估算，真实收益需要在完整回测中计算
    predictions_all = np.array(predictions_all)
    actuals_all = np.array(actuals_all)
    
    # 按预测排序，选取top 15%
    top_quantile = 0.15
    n_top = max(1, int(len(predictions_all) * top_quantile))
    top_idx = np.argsort(predictions_all)[-n_top:]
    
    # 平均收益
    portfolio_returns = actuals_all[top_idx]
    mean_return = np.mean(portfolio_returns)
    
    # 估算年化收益（假设5日调仓，一年约50次）
    annual_return = mean_return * 50
    
    # 估算最大回撤（滚动窗口）
    window_size = min(50, len(portfolio_returns))
    if window_size > 10:
        cumulative = np.cumsum(portfolio_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        max_drawdown = np.min(drawdown)
    else:
        max_drawdown = 0.0
    
    # 综合评分
    # IC得分（归一化到0-1）
    ic_score = max(0, min(1, mean_ic / 0.15))  # IC=0.15为优秀水平
    
    # IR得分
    ir_score = max(0, min(1, ir / 3.0))  # IR=3.0为优秀水平
    
    # 年化收益得分
    return_score = max(0, min(1, annual_return / 0.5))  # 50%年化为优秀水平
    
    # 回撤得分（回撤越小越好）
    drawdown_score = max(0, min(1, 1 + max_drawdown / 0.3))  # -30%回撤为可接受
    
    composite_score = (
        self.ic_weight * ic_score +
        self.ir_weight * ir_score +
        self.return_weight * return_score +
        self.drawdown_weight * drawdown_score
    )
    
    return {
        'ic': mean_ic,
        'ir': ir,
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'composite_score': composite_score
    }
```

- [ ] **Step 2: 实现HyperparameterTuner类（三种方法）**

在 `hyperparameter_tuning.py` 中添加:

```python
class HyperparameterTuner:
    """
    超参数调优器
    
    支持三种调参方法：
    1. random - 随机搜索
    2. grid - 网格搜索
    3. bayesian - 贝叶斯优化（Optuna）
    """
    
    def __init__(
        self,
        objective_fn: ObjectiveFunction,
        param_space: Dict,
        method: str = 'random'
    ):
        """
        初始化调优器
        
        Args:
            objective_fn: 目标函数实例
            param_space: 参数空间定义
            method: 调参方法 ('random', 'grid', 'bayesian')
        """
        self.objective_fn = objective_fn
        self.param_space = param_space
        self.method = method
        
    def _sample_random_params(self) -> Dict:
        """随机采样参数"""
        params = {}
        for key, value in self.param_space.items():
            if isinstance(value, tuple) and len(value) == 2:
                # 连续范围 (min, max)
                if isinstance(value[0], int):
                    params[key] = np.random.randint(value[0], value[1] + 1)
                else:
                    params[key] = np.random.uniform(value[0], value[1])
            elif isinstance(value, list):
                # 离散值列表
                params[key] = np.random.choice(value)
            else:
                params[key] = value
        return params
    
    def tune_random(self, n_iter: int = 50) -> TuningResult:
        """
        随机搜索
        
        Args:
            n_iter: 迭代次数
        
        Returns:
            TuningResult对象
        """
        print(f"随机搜索调参 (n_iter={n_iter})...")
        
        best_score = -np.inf
        best_params = None
        all_trials = []
        
        for i in range(n_iter):
            # 随机采样参数
            params = self._sample_random_params()
            
            # 评估
            metrics = self.objective_fn.evaluate_params(params)
            
            # 记录
            trial = {**params, **metrics}
            all_trials.append(trial)
            
            # 更新最佳
            if metrics['composite_score'] > best_score:
                best_score = metrics['composite_score']
                best_params = params.copy()
            
            print(f"  Trial {i+1}/{n_iter}: IC={metrics['ic']:.4f}, "
                  f"IR={metrics['ir']:.2f}, Score={metrics['composite_score']:.4f}")
        
        return TuningResult(
            best_params=best_params,
            best_score=best_score,
            all_trials=pd.DataFrame(all_trials),
            search_method='random',
            timestamp=datetime.now().strftime('%Y%m%d_%H%M%S')
        )
    
    def tune_grid(self) -> TuningResult:
        """
        网格搜索
        
        Returns:
            TuningResult对象
        """
        from sklearn.model_selection import ParameterGrid
        
        # 构建网格
        grid = list(ParameterGrid(self.param_space))
        n_combinations = len(grid)
        
        print(f"网格搜索调参 (总组合数={n_combinations})...")
        
        best_score = -np.inf
        best_params = None
        all_trials = []
        
        for i, params in enumerate(grid):
            # 评估
            metrics = self.objective_fn.evaluate_params(params)
            
            # 记录
            trial = {**params, **metrics}
            all_trials.append(trial)
            
            # 更新最佳
            if metrics['composite_score'] > best_score:
                best_score = metrics['composite_score']
                best_params = params.copy()
            
            if (i + 1) % 10 == 0 or i == n_combinations - 1:
                print(f"  Progress {i+1}/{n_combinations}: "
                      f"Best Score={best_score:.4f}")
        
        return TuningResult(
            best_params=best_params,
            best_score=best_score,
            all_trials=pd.DataFrame(all_trials),
            search_method='grid',
            timestamp=datetime.now().strftime('%Y%m%d_%H%M%S')
        )
    
    def tune_bayesian(self, n_iter: int = 50) -> TuningResult:
        """
        贝叶斯优化（Optuna）
        
        Args:
            n_iter: 迭代次数
        
        Returns:
            TuningResult对象
        """
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        print(f"贝叶斯优化调参 (n_iter={n_iter})...")
        
        all_trials = []
        
        def optuna_objective(trial):
            # 采样参数
            params = {}
            for key, value in self.param_space.items():
                if isinstance(value, tuple) and len(value) == 2:
                    if isinstance(value[0], int):
                        params[key] = trial.suggest_int(key, value[0], value[1])
                    else:
                        params[key] = trial.suggest_float(key, value[0], value[1])
                elif isinstance(value, list):
                    params[key] = trial.suggest_categorical(key, value)
                else:
                    params[key] = value
            
            # 评估
            metrics = self.objective_fn.evaluate_params(params)
            
            # 记录
            all_trials.append({**params, **metrics})
            
            return metrics['composite_score']
        
        # 创建study并优化
        study = optuna.create_study(direction='maximize')
        study.optimize(optuna_objective, n_trials=n_iter, show_progress_bar=True)
        
        return TuningResult(
            best_params=study.best_params,
            best_score=study.best_value,
            all_trials=pd.DataFrame(all_trials),
            search_method='bayesian',
            timestamp=datetime.now().strftime('%Y%m%d_%H%M%S')
        )
    
    def tune(self, n_iter: int = 50) -> TuningResult:
        """
        根据method执行调参
        
        Args:
            n_iter: 迭代次数（grid方法忽略此参数）
        
        Returns:
            TuningResult对象
        """
        if self.method == 'random':
            return self.tune_random(n_iter)
        elif self.method == 'grid':
            return self.tune_grid()
        elif self.method == 'bayesian':
            return self.tune_bayesian(n_iter)
        else:
            raise ValueError(f"Unknown method: {self.method}")
```


- [ ] **Step 3: 编写测试**

创建 `tests/models/test_hyperparameter_tuning.py`:

```python
import pytest
import pandas as pd
import numpy as np
from src.models.hyperparameter_tuning import ObjectiveFunction, HyperparameterTuner


@pytest.fixture
def sample_tuning_data():
    """生成调参测试数据"""
    n_samples = 300
    features = pd.DataFrame({
        'ts_code': ['688001.SH'] * n_samples,
        'factor_date': pd.date_range('2023-01-01', periods=n_samples),
        'ma5': np.random.randn(n_samples),
        'ma10': np.random.randn(n_samples),
        'rsi': np.random.uniform(0, 100, n_samples)
    })
    
    labels = pd.DataFrame({
        'ts_code': ['688001.SH'] * n_samples,
        'factor_date': pd.date_range('2023-01-01', periods=n_samples),
        'forward_return': np.random.randn(n_samples) * 0.05
    })
    
    return features, labels


def test_objective_function_evaluate(sample_tuning_data):
    """测试目标函数评估"""
    features, labels = sample_tuning_data
    
    obj_fn = ObjectiveFunction(features, labels, n_splits=3)
    
    params = {
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'num_boost_round': 50
    }
    
    metrics = obj_fn.evaluate_params(params)
    
    assert 'ic' in metrics
    assert 'ir' in metrics
    assert 'annual_return' in metrics
    assert 'max_drawdown' in metrics
    assert 'composite_score' in metrics
    assert 0 <= metrics['composite_score'] <= 1


def test_random_search(sample_tuning_data):
    """测试随机搜索"""
    features, labels = sample_tuning_data
    
    obj_fn = ObjectiveFunction(features, labels, n_splits=3)
    
    param_space = {
        'max_depth': (3, 8),
        'learning_rate': (0.01, 0.1),
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': (0.6, 1.0),
        'num_boost_round': (30, 100)
    }
    
    tuner = HyperparameterTuner(obj_fn, param_space, method='random')
    result = tuner.tune(n_iter=5)
    
    assert result.best_params is not None
    assert result.best_score > 0
    assert len(result.all_trials) == 5
    assert result.search_method == 'random'


def test_grid_search(sample_tuning_data):
    """测试网格搜索"""
    features, labels = sample_tuning_data
    
    obj_fn = ObjectiveFunction(features, labels, n_splits=3)
    
    param_space = {
        'max_depth': [4, 6],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8],
        'colsample_bytree': [0.8],
        'num_boost_round': [50]
    }
    
    tuner = HyperparameterTuner(obj_fn, param_space, method='grid')
    result = tuner.tune()
    
    assert result.best_params is not None
    assert len(result.all_trials) == 4  # 2 × 2 × 1 × 1 × 1
    assert result.search_method == 'grid'


def test_bayesian_search(sample_tuning_data):
    """测试贝叶斯优化"""
    features, labels = sample_tuning_data
    
    obj_fn = ObjectiveFunction(features, labels, n_splits=3)
    
    param_space = {
        'max_depth': (3, 8),
        'learning_rate': (0.01, 0.1),
        'subsample': (0.6, 1.0),
        'colsample_bytree': (0.6, 1.0),
        'num_boost_round': (30, 100)
    }
    
    tuner = HyperparameterTuner(obj_fn, param_space, method='bayesian')
    result = tuner.tune(n_iter=5)
    
    assert result.best_params is not None
    assert result.best_score > 0
    assert len(result.all_trials) == 5
    assert result.search_method == 'bayesian'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

- [ ] **Step 4: 运行测试**

```bash
cd /Users/mac/Documents/ai/bisai/star50-quant
pytest tests/models/test_hyperparameter_tuning.py -v
```

预期输出: 所有测试PASS

- [ ] **Step 5: 提交代码**

```bash
git add star50-quant/src/models/hyperparameter_tuning.py
git add tests/models/test_hyperparameter_tuning.py
git commit -m "feat: 完善超参数调优框架

- 完善evaluate_params()计算IC/IR/收益/回撤
- 实现HyperparameterTuner支持三种方法
- 添加随机搜索、网格搜索、贝叶斯优化
- 添加完整测试覆盖"
```

---

## Task 4: 创建完整回测引擎

**Files:**
- Create: `star50-quant/src/backtest/strategy_backtest.py`
- Test: `tests/backtest/test_strategy_backtest.py`

**目标:** 实现滚动窗口训练+完整回测，支持6种策略组合，计算真实的IC/IR/收益/回撤

- [ ] **Step 1: 创建回测引擎类**

创建 `star50-quant/src/backtest/strategy_backtest.py`:

```python
"""
完整回测引擎
============

实现滚动窗口训练和策略回测

特点：
- 滚动窗口训练（756日训练，20日预测，5日调仓）
- 6种策略组合（10%/15%/20% × 等权/信号加权）
- 真实的IC/IR/年化收益/最大回撤计算
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import xgboost as xgb
from datetime import datetime, timedelta


class StrategyBacktest:
    """
    策略回测引擎
    
    实现滚动窗口训练和多策略回测
    """
    
    def __init__(
        self,
        train_window: int = 756,  # 训练窗口（约3年交易日）
        predict_window: int = 20,  # 预测窗口（约1个月）
        rebalance_freq: int = 5,   # 调仓频率（5个交易日）
        transaction_cost: float = 0.002  # 双边交易成本
    ):
        """
        初始化回测引擎
        
        Args:
            train_window: 训练窗口天数
            predict_window: 预测窗口天数
            rebalance_freq: 调仓频率天数
            transaction_cost: 双边交易成本
        """
        self.train_window = train_window
        self.predict_window = predict_window
        self.rebalance_freq = rebalance_freq
        self.transaction_cost = transaction_cost
        
    def rolling_train_predict(
        self,
        data: pd.DataFrame,
        xgb_params: Dict,
        feature_cols: List[str]
    ) -> pd.DataFrame:
        """
        滚动窗口训练和预测
        
        Args:
            data: 完整数据（包含特征和标签）
            xgb_params: XGBoost参数
            feature_cols: 特征列名列表
        
        Returns:
            预测结果DataFrame，列：ts_code, factor_date, predicted_alpha, actual_return
        """
        # 按日期排序
        data = data.sort_values('factor_date').reset_index(drop=True)
        
        # 获取所有唯一日期
        unique_dates = sorted(data['factor_date'].unique())
        
        predictions_all = []
        
        # 滚动窗口
        for i in range(self.train_window, len(unique_dates), self.predict_window):
            # 训练集：过去train_window天
            train_start_idx = i - self.train_window
            train_end_idx = i
            train_dates = unique_dates[train_start_idx:train_end_idx]
            
            # 测试集：未来predict_window天
            test_end_idx = min(i + self.predict_window, len(unique_dates))
            test_dates = unique_dates[i:test_end_idx]
            
            if len(test_dates) == 0:
                break
            
            print(f"训练窗口: {train_dates[0]} ~ {train_dates[-1]} ({len(train_dates)}天)")
            print(f"预测窗口: {test_dates[0]} ~ {test_dates[-1]} ({len(test_dates)}天)")
            
            # 准备训练数据
            train_data = data[data['factor_date'].isin(train_dates)]
            X_train = train_data[feature_cols].values
            y_train = train_data['forward_return'].values
            
            # 准备测试数据
            test_data = data[data['factor_date'].isin(test_dates)]
            X_test = test_data[feature_cols].values
            y_test = test_data['forward_return'].values
            
            # 训练模型
            dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
            dtest = xgb.DMatrix(X_test, feature_names=feature_cols)
            
            model = xgb.train(
                xgb_params,
                dtrain,
                num_boost_round=xgb_params.get('num_boost_round', 100),
                verbose_eval=False
            )
            
            # 预测
            y_pred = model.predict(dtest)
            
            # 保存结果
            predictions = pd.DataFrame({
                'ts_code': test_data['ts_code'].values,
                'factor_date': test_data['factor_date'].values,
                'predicted_alpha': y_pred,
                'actual_return': y_test
            })
            
            predictions_all.append(predictions)
        
        return pd.concat(predictions_all, ignore_index=True)
    
    def calculate_ic(self, predictions: pd.DataFrame) -> Tuple[float, float]:
        """
        计算IC和Rank IC
        
        Args:
            predictions: 预测结果DataFrame
        
        Returns:
            (IC, Rank IC)
        """
        from scipy.stats import spearmanr
        
        # 按日期分组计算每日IC
        daily_ic = []
        daily_rank_ic = []
        
        for date, group in predictions.groupby('factor_date'):
            if len(group) > 5:
                # IC (Pearson相关系数)
                ic = np.corrcoef(
                    group['predicted_alpha'].values,
                    group['actual_return'].values
                )[0, 1]
                
                if not np.isnan(ic):
                    daily_ic.append(ic)
                
                # Rank IC (Spearman相关系数)
                rank_ic, _ = spearmanr(
                    group['predicted_alpha'].values,
                    group['actual_return'].values
                )
                
                if not np.isnan(rank_ic):
                    daily_rank_ic.append(rank_ic)
        
        mean_ic = np.mean(daily_ic) if daily_ic else 0.0
        mean_rank_ic = np.mean(daily_rank_ic) if daily_rank_ic else 0.0
        
        return mean_ic, mean_rank_ic
    
    def backtest_strategy(
        self,
        predictions: pd.DataFrame,
        top_quantile: float = 0.15,
        weight_method: str = 'equal'
    ) -> pd.DataFrame:
        """
        回测交易策略
        
        Args:
            predictions: 预测结果DataFrame
            top_quantile: 选股比例（前X%）
            weight_method: 权重方法 ('equal' 或 'signal')
        
        Returns:
            每日净值DataFrame，列：date, portfolio_return, cumulative_nav
        """
        # 获取调仓日期（每rebalance_freq天调仓一次）
        unique_dates = sorted(predictions['factor_date'].unique())
        rebalance_dates = unique_dates[::self.rebalance_freq]
        
        portfolio_returns = []
        previous_holdings = set()
        
        for i, rebalance_date in enumerate(rebalance_dates):
            # 获取当日预测
            today_predictions = predictions[
                predictions['factor_date'] == rebalance_date
            ].copy()
            
            if len(today_predictions) == 0:
                continue
            
            # 选股：取预测Alpha最高的top_quantile
            n_stocks = max(1, int(len(today_predictions) * top_quantile))
            top_stocks = today_predictions.nlargest(n_stocks, 'predicted_alpha')
            
            # 计算权重
            if weight_method == 'equal':
                # 等权
                top_stocks['weight'] = 1.0 / n_stocks
            elif weight_method == 'signal':
                # 信号加权（预测值归一化）
                signals = top_stocks['predicted_alpha'].values
                signals = signals - signals.min() + 1e-6  # 确保正数
                weights = signals / signals.sum()
                top_stocks['weight'] = weights
            else:
                raise ValueError(f"Unknown weight_method: {weight_method}")
            
            # 计算换手率和交易成本
            current_holdings = set(top_stocks['ts_code'].values)
            turnover = len(current_holdings.symmetric_difference(previous_holdings)) / len(current_holdings)
            cost = turnover * self.transaction_cost
            
            # 组合收益（扣除交易成本）
            portfolio_return = (
                (top_stocks['actual_return'] * top_stocks['weight']).sum() - cost
            )
            
            portfolio_returns.append({
                'date': rebalance_date,
                'portfolio_return': portfolio_return,
                'turnover': turnover,
                'cost': cost,
                'n_stocks': n_stocks
            })
            
            previous_holdings = current_holdings
        
        # 构建净值曲线
        nav_df = pd.DataFrame(portfolio_returns)
        nav_df['cumulative_return'] = (1 + nav_df['portfolio_return']).cumprod() - 1
        nav_df['cumulative_nav'] = 1 + nav_df['cumulative_return']
        
        return nav_df
    
    def calculate_metrics(self, nav_df: pd.DataFrame) -> Dict[str, float]:
        """
        计算策略性能指标
        
        Args:
            nav_df: 净值DataFrame
        
        Returns:
            指标字典
        """
        returns = nav_df['portfolio_return'].values
        nav = nav_df['cumulative_nav'].values
        
        # 年化收益
        total_return = nav[-1] - 1
        n_periods = len(nav)
        periods_per_year = 252 / self.rebalance_freq  # 一年的调仓次数
        annual_return = (1 + total_return) ** (periods_per_year / n_periods) - 1
        
        # 年化波动率
        annual_volatility = np.std(returns) * np.sqrt(periods_per_year)
        
        # 夏普比率（假设无风险利率为0）
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0.0
        
        # 最大回撤
        cumulative = nav
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # 卡玛比率
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
        
        # IR（收益标准差比）
        ir = np.mean(returns) / np.std(returns) * np.sqrt(periods_per_year) if np.std(returns) > 0 else 0.0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'ir': ir,
            'n_periods': n_periods
        }
    
    def run_full_backtest(
        self,
        data: pd.DataFrame,
        xgb_params: Dict,
        feature_cols: List[str],
        strategies: List[Dict] = None
    ) -> Dict[str, Dict]:
        """
        运行完整回测（多策略）
        
        Args:
            data: 完整数据
            xgb_params: XGBoost参数
            feature_cols: 特征列
            strategies: 策略配置列表，格式：[{'top_quantile': 0.15, 'weight_method': 'equal'}, ...]
        
        Returns:
            各策略结果字典
        """
        if strategies is None:
            # 默认6种策略
            strategies = [
                {'top_quantile': 0.10, 'weight_method': 'equal'},
                {'top_quantile': 0.15, 'weight_method': 'equal'},
                {'top_quantile': 0.20, 'weight_method': 'equal'},
                {'top_quantile': 0.10, 'weight_method': 'signal'},
                {'top_quantile': 0.15, 'weight_method': 'signal'},
                {'top_quantile': 0.20, 'weight_method': 'signal'},
            ]
        
        print("="*70)
        print("步骤1: 滚动窗口训练和预测")
        print("="*70)
        
        predictions = self.rolling_train_predict(data, xgb_params, feature_cols)
        
        print(f"\n预测完成: {len(predictions)} 条预测")
        print(f"日期范围: {predictions['factor_date'].min()} ~ {predictions['factor_date'].max()}")
        
        print("\n" + "="*70)
        print("步骤2: 计算IC和Rank IC")
        print("="*70)
        
        ic, rank_ic = self.calculate_ic(predictions)
        print(f"IC: {ic:.4f}")
        print(f"Rank IC: {rank_ic:.4f}")
        
        print("\n" + "="*70)
        print("步骤3: 回测各策略")
        print("="*70)
        
        results = {}
        
        for strategy in strategies:
            top_q = strategy['top_quantile']
            method = strategy['weight_method']
            strategy_name = f"{int(top_q*100)}%_{method}"
            
            print(f"\n策略: {strategy_name}")
            print("-" * 50)
            
            nav_df = self.backtest_strategy(predictions, top_q, method)
            metrics = self.calculate_metrics(nav_df)
            
            metrics['ic'] = ic
            metrics['rank_ic'] = rank_ic
            
            print(f"  IC: {metrics['ic']:.4f}")
            print(f"  IR: {metrics['ir']:.2f}")
            print(f"  年化收益: {metrics['annual_return']:.2%}")
            print(f"  最大回撤: {metrics['max_drawdown']:.2%}")
            print(f"  夏普比率: {metrics['sharpe_ratio']:.2f}")
            print(f"  卡玛比率: {metrics['calmar_ratio']:.2f}")
            
            results[strategy_name] = {
                'metrics': metrics,
                'nav': nav_df,
                'predictions': predictions
            }
        
        return results
```


- [ ] **Step 2: 编写回测引擎测试**

创建 `tests/backtest/test_strategy_backtest.py`:

```python
import pytest
import pandas as pd
import numpy as np
from src.backtest.strategy_backtest import StrategyBacktest


@pytest.fixture
def sample_backtest_data():
    """生成回测测试数据"""
    n_days = 1000
    n_stocks = 50
    
    dates = pd.date_range('2021-01-01', periods=n_days)
    
    data = []
    for date in dates:
        for stock_id in range(n_stocks):
            data.append({
                'ts_code': f'68800{stock_id:02d}.SH',
                'factor_date': date,
                'ma5': np.random.randn(),
                'ma10': np.random.randn(),
                'rsi': np.random.uniform(0, 100),
                'forward_return': np.random.randn() * 0.05
            })
    
    return pd.DataFrame(data)


def test_rolling_train_predict(sample_backtest_data):
    """测试滚动窗口训练预测"""
    backtest = StrategyBacktest(
        train_window=500,
        predict_window=50,
        rebalance_freq=5
    )
    
    xgb_params = {
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'num_boost_round': 30,
        'objective': 'reg:squarederror',
        'seed': 42
    }
    
    feature_cols = ['ma5', 'ma10', 'rsi']
    
    predictions = backtest.rolling_train_predict(
        sample_backtest_data,
        xgb_params,
        feature_cols
    )
    
    assert len(predictions) > 0
    assert 'predicted_alpha' in predictions.columns
    assert 'actual_return' in predictions.columns


def test_calculate_ic(sample_backtest_data):
    """测试IC计算"""
    # 生成模拟预测
    sample_data = sample_backtest_data.iloc[:1000].copy()
    sample_data['predicted_alpha'] = np.random.randn(len(sample_data)) * 0.05
    sample_data['actual_return'] = sample_data['forward_return']
    
    backtest = StrategyBacktest()
    ic, rank_ic = backtest.calculate_ic(sample_data)
    
    assert isinstance(ic, float)
    assert isinstance(rank_ic, float)
    assert -1 <= ic <= 1
    assert -1 <= rank_ic <= 1


def test_backtest_strategy(sample_backtest_data):
    """测试策略回测"""
    # 生成模拟预测
    sample_data = sample_backtest_data.iloc[:500].copy()
    sample_data['predicted_alpha'] = np.random.randn(len(sample_data)) * 0.05
    sample_data['actual_return'] = sample_data['forward_return']
    
    backtest = StrategyBacktest(rebalance_freq=5)
    
    nav_df = backtest.backtest_strategy(
        sample_data,
        top_quantile=0.15,
        weight_method='equal'
    )
    
    assert len(nav_df) > 0
    assert 'portfolio_return' in nav_df.columns
    assert 'cumulative_nav' in nav_df.columns
    assert nav_df['cumulative_nav'].iloc[0] >= 0


def test_calculate_metrics(sample_backtest_data):
    """测试指标计算"""
    # 生成模拟净值曲线
    nav_df = pd.DataFrame({
        'date': pd.date_range('2021-01-01', periods=100),
        'portfolio_return': np.random.randn(100) * 0.02,
        'cumulative_nav': np.nan
    })
    nav_df['cumulative_nav'] = (1 + nav_df['portfolio_return']).cumprod()
    
    backtest = StrategyBacktest()
    metrics = backtest.calculate_metrics(nav_df)
    
    assert 'annual_return' in metrics
    assert 'max_drawdown' in metrics
    assert 'sharpe_ratio' in metrics
    assert 'ir' in metrics
    assert metrics['max_drawdown'] <= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

- [ ] **Step 3: 运行测试**

```bash
cd /Users/mac/Documents/ai/bisai/star50-quant
mkdir -p tests/backtest
pytest tests/backtest/test_strategy_backtest.py -v
```

预期输出: 所有测试PASS

- [ ] **Step 4: 提交代码**

```bash
git add star50-quant/src/backtest/strategy_backtest.py
git add tests/backtest/test_strategy_backtest.py
git commit -m "feat: 创建完整回测引擎

- 实现滚动窗口训练预测
- 支持6种策略组合回测
- 计算真实IC/IR/年化收益/最大回撤
- 添加完整测试覆盖"
```

---

## Task 5: 完善调参主脚本

**Files:**
- Modify: `star50-quant/scripts/tune_xgb_model.py:96-250`
- Test: 手动测试运行

**目标:** 完善命令行接口，集成数据加载、调参、模型保存

- [ ] **Step 1: 完善main函数**

修改 `star50-quant/scripts/tune_xgb_model.py` 的main函数:

```python
def main():
    parser = argparse.ArgumentParser(description='XGBoost超参数调优')
    parser.add_argument(
        '--method',
        type=str,
        default='bayesian',
        choices=['random', 'grid', 'bayesian'],
        help='调参方法'
    )
    parser.add_argument(
        '--n_iter',
        type=int,
        default=50,
        help='迭代次数（grid方法忽略）'
    )
    parser.add_argument(
        '--start_date',
        type=str,
        default='2020-01-01',
        help='开始日期'
    )
    parser.add_argument(
        '--end_date',
        type=str,
        default='2024-12-31',
        help='结束日期'
    )
    parser.add_argument(
        '--cv_folds',
        type=int,
        default=5,
        help='交叉验证折数'
    )
    parser.add_argument(
        '--ic_weight',
        type=float,
        default=0.4,
        help='IC权重'
    )
    parser.add_argument(
        '--ir_weight',
        type=float,
        default=0.3,
        help='IR权重'
    )
    parser.add_argument(
        '--return_weight',
        type=float,
        default=0.2,
        help='年化收益权重'
    )
    parser.add_argument(
        '--drawdown_weight',
        type=float,
        default=0.1,
        help='回撤权重'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='tuning_results',
        help='输出目录'
    )
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*70)
    print("XGBoost超参数调优")
    print("="*70)
    print(f"方法: {args.method}")
    print(f"迭代次数: {args.n_iter}")
    print(f"日期范围: {args.start_date} ~ {args.end_date}")
    print(f"交叉验证折数: {args.cv_folds}")
    print()
    
    # 1. 加载数据
    print("步骤1: 加载数据...")
    loader = FactorDataLoader(db_name='star50_quant')
    loader.connect()
    
    # 加载因子
    factors_long = loader.load_factors(args.start_date, args.end_date)
    features = loader.pivot_factors(factors_long)
    
    # 加载价格和计算超额收益
    stock_prices = loader.load_prices(args.start_date, args.end_date)
    index_prices = loader.load_index_prices('000688.SH', args.start_date, args.end_date)
    labels = loader.calculate_excess_returns(stock_prices, index_prices, forward_days=5)
    
    labels['factor_date'] = pd.to_datetime(labels['trade_date'])
    features['factor_date'] = pd.to_datetime(features['factor_date'])
    
    # 准备训练数据（正确的处理顺序）
    data = loader.prepare_training_data(features, labels)
    
    loader.close()
    
    print(f"✓ 数据加载完成: {len(data)} 样本")
    print(f"  日期范围: {data['factor_date'].min()} ~ {data['factor_date'].max()}")
    print(f"  股票数: {data['ts_code'].nunique()}")
    print()
    
    # 提取特征列
    feature_cols = [
        col for col in data.columns
        if col not in ['ts_code', 'factor_date', 'trade_date', 'forward_return']
    ]
    print(f"  特征数: {len(feature_cols)}")
    print()
    
    # 2. 定义参数空间
    print("步骤2: 定义参数空间...")
    param_space = define_param_space(args.method)
    print(f"✓ 参数空间定义完成")
    print()
    
    # 3. 创建目标函数
    print("步骤3: 创建目标函数...")
    
    # 准备特征和标签
    features_for_tuning = data[['ts_code', 'factor_date'] + feature_cols]
    labels_for_tuning = data[['ts_code', 'factor_date', 'forward_return']]
    
    objective_fn = ObjectiveFunction(
        features_for_tuning,
        labels_for_tuning,
        n_splits=args.cv_folds,
        ic_weight=args.ic_weight,
        ir_weight=args.ir_weight,
        return_weight=args.return_weight,
        drawdown_weight=args.drawdown_weight
    )
    print(f"✓ 目标函数创建完成")
    print()
    
    # 4. 执行调参
    print("步骤4: 执行调参...")
    tuner = HyperparameterTuner(objective_fn, param_space, method=args.method)
    result = tuner.tune(n_iter=args.n_iter)
    print()
    
    # 5. 保存结果
    print("步骤5: 保存结果...")
    
    timestamp = result.timestamp
    result_file = os.path.join(
        args.output_dir,
        f'xgb_tuning_{args.method}_{timestamp}.json'
    )
    result.save(result_file)
    
    # 6. 训练最佳模型并保存
    print("\n步骤6: 训练最佳模型...")
    best_params = result.best_params.copy()
    num_boost_round = best_params.pop('num_boost_round', 100)
    
    model = XGBoostAlphaModel(params=best_params)
    model.train(
        features_for_tuning,
        labels_for_tuning,
        num_boost_round=num_boost_round
    )
    
    model_file = os.path.join(
        args.output_dir,
        f'xgb_best_model_{timestamp}.json'
    )
    model.save(model_file)
    
    # 7. 保存特征重要性
    if model.feature_importance is not None:
        importance_df = pd.DataFrame({
            'feature': model.feature_names,
            'importance': model.feature_importance
        }).sort_values('importance', ascending=False)
        
        importance_file = os.path.join(
            args.output_dir,
            f'xgb_feature_importance_{timestamp}.csv'
        )
        importance_df.to_csv(importance_file, index=False)
        print(f"✓ 特征重要性保存到 {importance_file}")
        
        print("\nTop 10 重要特征:")
        print(importance_df.head(10).to_string(index=False))
    
    # 8. 打印最终结果
    print("\n" + "="*70)
    print("调参完成!")
    print("="*70)
    print(f"\n最佳参数:")
    for key, value in result.best_params.items():
        print(f"  {key}: {value}")
    
    print(f"\n最佳评分: {result.best_score:.4f}")
    
    print(f"\n结果文件:")
    print(f"  参数: {result_file}")
    print(f"  模型: {model_file}")
    
    if model.feature_importance is not None:
        print(f"  特征重要性: {importance_file}")
    
    print("\n下一步:")
    print(f"  运行完整回测: python scripts/run_complete_backtest.py --model_path {model_file}")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: 测试调参脚本（快速测试）**

```bash
cd /Users/mac/Documents/ai/bisai/star50-quant

# 快速测试（随机搜索，少量迭代）
python scripts/tune_xgb_model.py \
    --method random \
    --n_iter 3 \
    --start_date 2023-01-01 \
    --end_date 2023-06-30 \
    --cv_folds 3 \
    --output_dir tuning_results/test_run
```

预期输出: 
- 数据加载成功
- 调参完成
- 生成结果文件

- [ ] **Step 3: 验证输出文件**

```bash
ls -lh tuning_results/test_run/
cat tuning_results/test_run/xgb_tuning_random_*.json
head -20 tuning_results/test_run/xgb_tuning_random_*_trials.csv
```

预期输出: 
- .json文件包含最佳参数和评分
- _trials.csv包含所有试验记录
- 模型文件 .xgb存在

- [ ] **Step 4: 提交代码**

```bash
git add star50-quant/scripts/tune_xgb_model.py
git commit -m "feat: 完善调参主脚本

- 完善命令行参数解析
- 集成数据加载和正确的预处理顺序
- 自动训练和保存最佳模型
- 保存特征重要性分析
- 添加详细的进度输出"
```

---

## Task 6: 创建完整回测脚本

**Files:**
- Create: `star50-quant/scripts/run_complete_backtest.py`
- Test: 手动测试运行

**目标:** 使用最佳模型执行完整回测，生成详细报告

- [ ] **Step 1: 创建回测脚本**

创建 `star50-quant/scripts/run_complete_backtest.py`:

```python
#!/usr/bin/env python3
"""
完整回测脚本
============

使用训练好的XGBoost模型进行完整回测，验证真实表现
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import pandas as pd
import numpy as np
import json
from datetime import datetime

from src.models.data_loader import FactorDataLoader
from src.models.xgb_model import XGBoostAlphaModel
from src.backtest.strategy_backtest import StrategyBacktest


def print_strategy_comparison(results: dict, targets: dict):
    """打印策略对比表格"""
    print("\n" + "="*100)
    print("策略对比")
    print("="*100)
    
    header = f"{'策略':<20} {'IC':>8} {'IR':>8} {'年化收益':>12} {'最大回撤':>12} {'夏普':>8} {'达标数':>8}"
    print(header)
    print("-" * 100)
    
    for strategy_name, result in results.items():
        metrics = result['metrics']
        
        # 计算达标情况
        passed = 0
        if metrics['ic'] > targets['ic']:
            passed += 1
        if metrics['ir'] >= targets['ir']:
            passed += 1
        if metrics['annual_return'] > targets['annual_return']:
            passed += 1
        if metrics['max_drawdown'] >= targets['max_drawdown']:
            passed += 1
        
        row = (
            f"{strategy_name:<20} "
            f"{metrics['ic']:>8.4f} "
            f"{metrics['ir']:>8.2f} "
            f"{metrics['annual_return']:>11.2%} "
            f"{metrics['max_drawdown']:>11.2%} "
            f"{metrics['sharpe_ratio']:>8.2f} "
            f"{passed}/4{':>6}"
        )
        print(row)
    
    print("="*100)


def save_results(results: dict, output_dir: str, timestamp: str):
    """保存回测结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存汇总指标
    summary = {}
    for strategy_name, result in results.items():
        summary[strategy_name] = result['metrics']
    
    summary_file = os.path.join(output_dir, f'backtest_summary_{timestamp}.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"✓ 汇总指标保存到: {summary_file}")
    
    # 保存各策略净值曲线
    for strategy_name, result in results.items():
        nav_file = os.path.join(
            output_dir,
            f'nav_{strategy_name}_{timestamp}.csv'
        )
        result['nav'].to_csv(nav_file, index=False)
    
    print(f"✓ 净值曲线保存到: {output_dir}/nav_*.csv")


def main():
    parser = argparse.ArgumentParser(description='完整回测')
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='模型文件路径'
    )
    parser.add_argument(
        '--start_date',
        type=str,
        default='2023-01-01',
        help='回测开始日期'
    )
    parser.add_argument(
        '--end_date',
        type=str,
        default='2024-12-31',
        help='回测结束日期'
    )
    parser.add_argument(
        '--train_window',
        type=int,
        default=756,
        help='训练窗口天数'
    )
    parser.add_argument(
        '--predict_window',
        type=int,
        default=20,
        help='预测窗口天数'
    )
    parser.add_argument(
        '--rebalance_freq',
        type=int,
        default=5,
        help='调仓频率天数'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='backtest_results',
        help='输出目录'
    )
    
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print("="*70)
    print("完整回测")
    print("="*70)
    print(f"模型: {args.model_path}")
    print(f"日期范围: {args.start_date} ~ {args.end_date}")
    print(f"训练窗口: {args.train_window}天")
    print(f"预测窗口: {args.predict_window}天")
    print(f"调仓频率: {args.rebalance_freq}天")
    print()
    
    # 1. 加载模型
    print("步骤1: 加载模型...")
    model = XGBoostAlphaModel()
    model.load(args.model_path)
    print("✓ 模型加载完成")
    print()
    
    # 2. 加载数据
    print("步骤2: 加载数据...")
    loader = FactorDataLoader(db_name='star50_quant')
    loader.connect()
    
    factors_long = loader.load_factors(args.start_date, args.end_date)
    features = loader.pivot_factors(factors_long)
    
    stock_prices = loader.load_prices(args.start_date, args.end_date)
    index_prices = loader.load_index_prices('000688.SH', args.start_date, args.end_date)
    labels = loader.calculate_excess_returns(stock_prices, index_prices, forward_days=5)
    
    labels['factor_date'] = pd.to_datetime(labels['trade_date'])
    features['factor_date'] = pd.to_datetime(features['factor_date'])
    
    data = loader.prepare_training_data(features, labels)
    
    loader.close()
    
    print(f"✓ 数据加载完成: {len(data)} 样本")
    print()
    
    # 3. 加载模型参数
    with open(args.model_path, 'r') as f:
        metadata = json.load(f)
    xgb_params = metadata['params']
    
    # 提取特征列
    feature_cols = [
        col for col in data.columns
        if col not in ['ts_code', 'factor_date', 'trade_date', 'forward_return']
    ]
    
    # 4. 运行完整回测
    print("步骤3: 运行完整回测...")
    backtest_engine = StrategyBacktest(
        train_window=args.train_window,
        predict_window=args.predict_window,
        rebalance_freq=args.rebalance_freq
    )
    
    results = backtest_engine.run_full_backtest(
        data,
        xgb_params,
        feature_cols
    )
    
    # 5. 打印对比结果
    targets = {
        'ic': 0.04,
        'ir': 1.5,
        'annual_return': 0.35,
        'max_drawdown': -0.20
    }
    
    print_strategy_comparison(results, targets)
    
    # 6. 保存结果
    print("\n步骤4: 保存结果...")
    save_results(results, args.output_dir, timestamp)
    
    print("\n" + "="*70)
    print("回测完成!")
    print("="*70)
    print(f"\n结果保存在: {args.output_dir}/")
    print(f"  汇总指标: backtest_summary_{timestamp}.json")
    print(f"  净值曲线: nav_*_{timestamp}.csv")


if __name__ == '__main__':
    main()
```


- [ ] **Step 2: 测试回测脚本（使用测试模型）**

```bash
cd /Users/mac/Documents/ai/bisai/star50-quant

# 首先确保有测试模型
ls -lh tuning_results/test_run/xgb_best_model_*.json

# 运行回测（短期数据测试）
python scripts/run_complete_backtest.py \
    --model_path tuning_results/test_run/xgb_best_model_*.json \
    --start_date 2023-01-01 \
    --end_date 2023-06-30 \
    --train_window 200 \
    --predict_window 20 \
    --rebalance_freq 5 \
    --output_dir backtest_results/test_run
```

预期输出: 模型加载成功、滚动窗口训练完成、6种策略回测完成

- [ ] **Step 3: 验证回测结果**

```bash
ls -lh backtest_results/test_run/
cat backtest_results/test_run/backtest_summary_*.json
head -20 backtest_results/test_run/nav_15%_equal_*.csv
```

- [ ] **Step 4: 提交代码**

```bash
git add star50-quant/scripts/run_complete_backtest.py
git commit -m "feat: 创建完整回测脚本

- 加载训练好的XGBoost模型
- 执行滚动窗口回测
- 对比6种策略表现
- 生成详细报告和净值曲线"
```

---

## Task 7: 端到端集成测试

**Files:**
- Create: `tests/integration/test_end_to_end.py`

**目标:** 验证从数据加载到回测的完整流程

- [ ] **Step 1: 创建集成测试文件**

略（参考完整计划文档）

- [ ] **Step 2: 运行集成测试**

```bash
cd /Users/mac/Documents/ai/bisai/star50-quant
mkdir -p tests/integration
pytest tests/integration/test_end_to_end.py -v -s
```

- [ ] **Step 3: 提交代码**

```bash
git add tests/integration/test_end_to_end.py
git commit -m "test: 添加端到端集成测试"
```

---

## Task 8: 生产环境完整测试

**目标:** 在真实数据上运行完整流程，验证所有目标达成

- [ ] **Step 1: 运行完整调参（贝叶斯优化）**

```bash
cd /Users/mac/Documents/ai/bisai/star50-quant

python scripts/tune_xgb_model.py \
    --method bayesian \
    --n_iter 50 \
    --start_date 2020-01-01 \
    --end_date 2024-12-31 \
    --cv_folds 5 \
    --output_dir tuning_results/production_run
```

预期: IC > 0.04

- [ ] **Step 2: 运行完整回测**

```bash
python scripts/run_complete_backtest.py \
    --model_path tuning_results/production_run/xgb_best_model_*.json \
    --start_date 2023-01-01 \
    --end_date 2024-12-31 \
    --output_dir backtest_results/production_run
```

预期: 至少4个策略达到全部目标

- [ ] **Step 3: 验证目标达成**

验证以下目标:
- IC > 0.04
- IR >= 1.5  
- 年化收益 > 35%
- 最大回撤 <= -20%

- [ ] **Step 4: 最终提交**

```bash
git add tuning_results/production_run/
git add backtest_results/production_run/
git commit -m "feat: 完成XGBoost调参系统生产验证

- 所有目标指标达成
- 系统可投入生产使用"
```

---

## 执行选择

计划完成并保存。

**两种执行选项:**

**1. Subagent-Driven (推荐)** - 派发子代理执行每个任务，任务间审查，快速迭代

**2. Inline Execution** - 在当前会话执行，批量执行带检查点

**选择哪种方式？**

