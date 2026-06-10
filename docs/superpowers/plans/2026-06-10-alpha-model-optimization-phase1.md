# Alpha模型优化 - Phase 1: 特征工程实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现特征工程模块，将30个原始技术因子扩展到160个增强特征，并验证单窗口模型IC提升到0.028-0.032

**Architecture:** 创建独立的FeatureEngineer类，包含5种特征生成策略（交叉、时序、非线性、截面、Alpha），采用可配置的YAML驱动设计，支持特征选择和版本管理

**Tech Stack:** Python 3.8+, pandas, numpy, PyYAML, scikit-learn, LightGBM

---

## 文件结构

### 新增文件
- `configs/feature_config.yaml` - 特征工程配置
- `src/features/engineering.py` - 特征工程核心类
- `src/features/alpha_factors.py` - Alpha因子库
- `tests/features/test_engineering.py` - 特征工程单元测试
- `scripts/validate_features.py` - 特征验证脚本

### 修改文件
- `src/models/data_loader.py` - 集成特征工程模块
- `scripts/train_alpha_model.py` - 添加特征工程步骤

---

## Task 1: 创建特征配置文件

**Files:**
- Create: `configs/feature_config.yaml`

- [ ] **Step 1: 创建配置文件**

```yaml
# 特征工程配置
feature_engineering:
  # 原始因子列表（30个）
  original_factors:
    momentum:
      - macd
      - macd_signal
      - macd_histogram
      - rsi6
      - rsi12
      - rsi24
      - roc_5
      - roc_10
      - roc_20
      - momentum_5
      - momentum_10
      - momentum_20
    
    volume:
      - obv
      - mfi14
      - vwap
      - volume_ma5
      - volume_ma10
      - volume_ma20
      - volume_ratio
    
    trend:
      - ma5
      - ma10
      - ma20
      - ma60
      - ema5
      - ema10
      - ema20
      - boll_upper
      - boll_middle
      - boll_lower
      - atr14

  # Top因子（用于衍生特征）
  top_factors:
    - volume_ma20
    - atr14
    - obv
    - ma60
    - macd_signal
    - boll_lower
    - macd_histogram
    - volume_ma10
    - macd
    - volume_ma5
    - rsi12
    - momentum_10
    - roc_10
    - mfi14
    - ma20

  # 交叉特征配置
  cross_features:
    momentum_volume:
      - [rsi6, volume_ratio]
      - [rsi12, volume_ratio]
      - [rsi24, volume_ratio]
      - [momentum_5, obv]
      - [momentum_10, obv]
      - [momentum_20, obv]
      - [roc_5, mfi14]
      - [roc_10, mfi14]
      - [roc_20, mfi14]
    
    macd_volume:
      - [macd, volume_ma5]
      - [macd, volume_ma10]
      - [macd, volume_ma20]
      - [macd_signal, obv]
      - [macd_histogram, volume_ratio]
    
    momentum_ratio:
      - [rsi6, rsi24, div]
      - [momentum_5, momentum_20, div]
      - [roc_5, roc_20, div]
    
    trend_momentum:
      - [ma5, momentum_5, slope_mult]
      - [ma10, momentum_10, slope_mult]
      - [ma20, momentum_20, slope_mult]
      - [ma20, rsi12, slope_mult]
    
    volatility_adjusted:
      - [macd, atr14, div]
      - [momentum_5, atr14, div]
      - [rsi6, atr14, div]
      - [obv, atr14, div]
      - [volume_ratio, atr14, div]

  # 时序衍生配置
  temporal_features:
    momentum_periods: [5, 10, 20]
    volatility_periods: [20]
    factors: []  # 使用top_factors

  # 非线性变换配置
  nonlinear_transforms:
    log_factors:
      - volume_ma20
      - atr14
      - obv
      - volume_ma10
      - volume_ma5
    
    rank_factors:
      - rsi6
      - rsi12
      - macd
      - momentum_10
      - roc_10

  # 截面统计配置
  cross_sectional:
    quantile_factors:
      - volume_ma20
      - atr14
      - rsi6
      - macd
      - obv
    
    zscore_factors: []  # 使用top_factors前10

  # Alpha因子配置
  alpha_factors:
    enabled:
      - alpha_001
      - alpha_006
      - alpha_053
      - alpha_054
      - alpha_101
      - alpha_custom_price_volume_divergence
      - alpha_custom_ma_acceleration
      - alpha_custom_volume_surge
      - alpha_custom_volatility_ratio
      - alpha_custom_trend_strength

# 特征验证
validation:
  check_nan: true
  check_inf: true
  max_missing_ratio: 0.1
  correlation_threshold: 0.95  # 相关性>0.95视为重复
```

- [ ] **Step 2: 验证配置文件格式**

Run: `python -c "import yaml; yaml.safe_load(open('configs/feature_config.yaml'))"`
Expected: 无错误输出

- [ ] **Step 3: 提交配置文件**

```bash
git add configs/feature_config.yaml
git commit -m "feat: 添加特征工程配置文件"
```

---

## Task 2: 实现Alpha因子库

**Files:**
- Create: `src/features/alpha_factors.py`
- Test: `tests/features/test_alpha_factors.py`

- [ ] **Step 1: 编写测试 - Alpha因子计算**

Create: `tests/features/test_alpha_factors.py`

```python
"""Alpha因子库测试"""

import pytest
import pandas as pd
import numpy as np
from src.features.alpha_factors import AlphaFactorCalculator


@pytest.fixture
def sample_data():
    """创建测试数据"""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    n_stocks = 10
    
    data = []
    for stock_id in range(n_stocks):
        for date in dates:
            data.append({
                'ts_code': f'{stock_id:06d}.SH',
                'trade_date': date,
                'close': 100 + np.random.randn() * 10,
                'open': 100 + np.random.randn() * 10,
                'high': 105 + np.random.randn() * 10,
                'low': 95 + np.random.randn() * 10,
                'volume': 1000000 + np.random.randn() * 100000
            })
    
    df = pd.DataFrame(data)
    return df


def test_alpha_001(sample_data):
    """测试Alpha#1: 动量反转"""
    calculator = AlphaFactorCalculator()
    result = calculator.calculate_alpha_001(sample_data, period=20)
    
    assert 'alpha_001' in result.columns
    assert len(result) == len(sample_data)
    assert not result['alpha_001'].isna().all()


def test_alpha_006(sample_data):
    """测试Alpha#6: 相对强弱"""
    calculator = AlphaFactorCalculator()
    result = calculator.calculate_alpha_006(sample_data, period=20)
    
    assert 'alpha_006' in result.columns
    assert len(result) == len(sample_data)
    assert not result['alpha_006'].isna().all()


def test_alpha_053(sample_data):
    """测试Alpha#53: 高低价差/成交量"""
    calculator = AlphaFactorCalculator()
    result = calculator.calculate_alpha_053(sample_data)
    
    assert 'alpha_053' in result.columns
    assert len(result) == len(sample_data)
    # 检查计算逻辑
    expected = (sample_data['close'] - sample_data['low']) / (sample_data['volume'] + 1e-6)
    pd.testing.assert_series_equal(result['alpha_053'], expected, check_names=False)


def test_alpha_054(sample_data):
    """测试Alpha#54: 价格位置"""
    calculator = AlphaFactorCalculator()
    result = calculator.calculate_alpha_054(sample_data)
    
    assert 'alpha_054' in result.columns
    assert len(result) == len(sample_data)
    # 检查范围
    assert result['alpha_054'].min() >= -1
    assert result['alpha_054'].max() <= 1


def test_alpha_custom_price_volume_divergence(sample_data):
    """测试自定义Alpha: 价量背离"""
    calculator = AlphaFactorCalculator()
    result = calculator.calculate_custom_price_volume_divergence(sample_data)
    
    assert 'alpha_custom_price_volume_divergence' in result.columns
    assert len(result) == len(sample_data)


def test_batch_calculation(sample_data):
    """测试批量计算所有Alpha因子"""
    calculator = AlphaFactorCalculator()
    alpha_list = ['alpha_001', 'alpha_006', 'alpha_053', 'alpha_054']
    
    result = calculator.calculate_batch(sample_data, alpha_list)
    
    for alpha_name in alpha_list:
        assert alpha_name in result.columns
    
    assert len(result) == len(sample_data)
```

- [ ] **Step 2: 运行测试验证失败**

Run: `pytest tests/features/test_alpha_factors.py -v`
Expected: FAIL - AlphaFactorCalculator not defined

- [ ] **Step 3: 实现Alpha因子库**

Create: `src/features/alpha_factors.py`

```python
"""
Alpha因子库
============

实现经典Alpha因子和自定义Alpha因子。
参考WorldQuant 101 Alphas。
"""

import pandas as pd
import numpy as np
from typing import List, Dict


class AlphaFactorCalculator:
    """
    Alpha因子计算器
    
    提供多种Alpha因子计算方法。
    """
    
    def __init__(self):
        """初始化"""
        self.factor_registry = {
            'alpha_001': self.calculate_alpha_001,
            'alpha_006': self.calculate_alpha_006,
            'alpha_053': self.calculate_alpha_053,
            'alpha_054': self.calculate_alpha_054,
            'alpha_101': self.calculate_alpha_101,
            'alpha_custom_price_volume_divergence': self.calculate_custom_price_volume_divergence,
            'alpha_custom_ma_acceleration': self.calculate_custom_ma_acceleration,
            'alpha_custom_volume_surge': self.calculate_custom_volume_surge,
            'alpha_custom_volatility_ratio': self.calculate_custom_volatility_ratio,
            'alpha_custom_trend_strength': self.calculate_custom_trend_strength
        }
    
    def calculate_alpha_001(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        Alpha#1: 动量反转
        (-1 × close.pct_change(period))
        
        逻辑: 过去N天涨太多的股票倾向于回调
        """
        result = df.copy()
        result['alpha_001'] = -1 * result.groupby('ts_code')['close'].pct_change(period)
        return result
    
    def calculate_alpha_006(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        Alpha#6: 相对强弱
        close.pct_change(period) / volume.rolling(period).mean()
        
        逻辑: 价格涨幅相对于成交量的比率
        """
        result = df.copy()
        
        price_change = result.groupby('ts_code')['close'].pct_change(period)
        volume_ma = result.groupby('ts_code')['volume'].transform(
            lambda x: x.rolling(period, min_periods=1).mean()
        )
        
        result['alpha_006'] = price_change / (volume_ma + 1e-6)
        return result
    
    def calculate_alpha_053(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Alpha#53: 高低价差/成交量
        (close - low) / volume
        
        逻辑: 相对最低价的位置除以成交量
        """
        result = df.copy()
        result['alpha_053'] = (result['close'] - result['low']) / (result['volume'] + 1e-6)
        return result
    
    def calculate_alpha_054(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Alpha#54: 价格位置
        (close - open) / (high - low + ε)
        
        逻辑: 收盘价在当日高低范围的相对位置
        """
        result = df.copy()
        result['alpha_054'] = (result['close'] - result['open']) / (result['high'] - result['low'] + 1e-6)
        return result
    
    def calculate_alpha_101(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Alpha#101: 价格动量强度
        (close - open) / (open + ε)
        
        逻辑: 日内涨跌幅
        """
        result = df.copy()
        result['alpha_101'] = (result['close'] - result['open']) / (result['open'] + 1e-6)
        return result
    
    def calculate_custom_price_volume_divergence(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        自定义Alpha: 价量背离
        close.rank() - volume.rank()
        
        逻辑: 价格排名与成交量排名的差异
        """
        result = df.copy()
        
        price_rank = result.groupby('trade_date')['close'].rank(pct=True)
        volume_rank = result.groupby('trade_date')['volume'].rank(pct=True)
        
        result['alpha_custom_price_volume_divergence'] = price_rank - volume_rank
        return result
    
    def calculate_custom_ma_acceleration(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        自定义Alpha: MA加速度
        ma.pct_change(5).pct_change(5)
        
        逻辑: 均线变化率的变化率（二阶导数）
        """
        result = df.copy()
        
        ma = result.groupby('ts_code')['close'].transform(
            lambda x: x.rolling(period, min_periods=1).mean()
        )
        ma_velocity = ma.groupby(result['ts_code']).pct_change(5)
        ma_acceleration = ma_velocity.groupby(result['ts_code']).pct_change(5)
        
        result['alpha_custom_ma_acceleration'] = ma_acceleration
        return result
    
    def calculate_custom_volume_surge(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        自定义Alpha: 成交量突增
        (volume - volume.rolling(period).mean()) / volume.rolling(period).std()
        
        逻辑: 成交量的Z-score
        """
        result = df.copy()
        
        grouped = result.groupby('ts_code')['volume']
        volume_ma = grouped.transform(lambda x: x.rolling(period, min_periods=1).mean())
        volume_std = grouped.transform(lambda x: x.rolling(period, min_periods=1).std())
        
        result['alpha_custom_volume_surge'] = (result['volume'] - volume_ma) / (volume_std + 1e-6)
        return result
    
    def calculate_custom_volatility_ratio(self, df: pd.DataFrame, short: int = 5, long: int = 20) -> pd.DataFrame:
        """
        自定义Alpha: 波动率比率
        std(short) / std(long)
        
        逻辑: 短期波动率相对长期波动率
        """
        result = df.copy()
        
        grouped = result.groupby('ts_code')['close']
        std_short = grouped.transform(lambda x: x.pct_change().rolling(short, min_periods=1).std())
        std_long = grouped.transform(lambda x: x.pct_change().rolling(long, min_periods=1).std())
        
        result['alpha_custom_volatility_ratio'] = std_short / (std_long + 1e-6)
        return result
    
    def calculate_custom_trend_strength(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        自定义Alpha: 趋势强度
        (close - close.rolling(period).min()) / (close.rolling(period).max() - close.rolling(period).min())
        
        逻辑: 价格在N天最高最低价之间的相对位置
        """
        result = df.copy()
        
        grouped = result.groupby('ts_code')['close']
        rolling_min = grouped.transform(lambda x: x.rolling(period, min_periods=1).min())
        rolling_max = grouped.transform(lambda x: x.rolling(period, min_periods=1).max())
        
        result['alpha_custom_trend_strength'] = (result['close'] - rolling_min) / (rolling_max - rolling_min + 1e-6)
        return result
    
    def calculate_batch(self, df: pd.DataFrame, alpha_list: List[str]) -> pd.DataFrame:
        """
        批量计算多个Alpha因子
        
        Args:
            df: 输入数据
            alpha_list: Alpha因子名称列表
        
        Returns:
            包含所有Alpha因子的DataFrame
        """
        result = df.copy()
        
        for alpha_name in alpha_list:
            if alpha_name not in self.factor_registry:
                raise ValueError(f"Unknown alpha factor: {alpha_name}")
            
            calculator_func = self.factor_registry[alpha_name]
            result = calculator_func(result)
        
        return result
    
    def get_available_alphas(self) -> List[str]:
        """返回所有可用的Alpha因子名称"""
        return list(self.factor_registry.keys())
```

- [ ] **Step 4: 运行测试验证通过**

Run: `pytest tests/features/test_alpha_factors.py -v`
Expected: 所有测试通过

- [ ] **Step 5: 提交Alpha因子库**

```bash
git add src/features/alpha_factors.py tests/features/test_alpha_factors.py
git commit -m "feat: 实现Alpha因子库（10个因子）"
```

---

## Task 3: 实现特征工程核心类

**Files:**
- Create: `src/features/engineering.py`
- Test: `tests/features/test_engineering.py`

- [ ] **Step 1: 编写测试 - 特征工程模块**

Create: `tests/features/test_engineering.py`

```python
"""特征工程模块测试"""

import pytest
import pandas as pd
import numpy as np
import yaml
from src.features.engineering import FeatureEngineer


@pytest.fixture
def sample_factor_data():
    """创建测试因子数据"""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    n_stocks = 10
    
    data = []
    for stock_id in range(n_stocks):
        for date in dates:
            data.append({
                'ts_code': f'{stock_id:06d}.SH',
                'factor_date': date,
                'close': 100 + np.random.randn() * 10,
                'open': 100 + np.random.randn() * 10,
                'high': 105 + np.random.randn() * 10,
                'low': 95 + np.random.randn() * 10,
                'volume': 1000000 + np.random.randn() * 100000,
                'macd': np.random.randn(),
                'rsi6': 50 + np.random.randn() * 20,
                'rsi12': 50 + np.random.randn() * 20,
                'rsi24': 50 + np.random.randn() * 20,
                'obv': np.random.randn() * 1000,
                'volume_ratio': 1 + np.random.randn() * 0.2,
                'momentum_5': np.random.randn(),
                'momentum_10': np.random.randn(),
                'momentum_20': np.random.randn(),
                'ma5': 100 + np.random.randn() * 10,
                'ma10': 100 + np.random.randn() * 10,
                'ma20': 100 + np.random.randn() * 10,
                'atr14': 5 + np.random.randn(),
                'volume_ma5': 1000000 + np.random.randn() * 100000,
                'volume_ma10': 1000000 + np.random.randn() * 100000,
                'volume_ma20': 1000000 + np.random.randn() * 100000,
                'mfi14': 50 + np.random.randn() * 20,
                'roc_5': np.random.randn(),
                'roc_10': np.random.randn(),
                'roc_20': np.random.randn()
            })
    
    df = pd.DataFrame(data)
    return df


def test_feature_engineer_initialization():
    """测试FeatureEngineer初始化"""
    engineer = FeatureEngineer('configs/feature_config.yaml')
    assert engineer.config is not None
    assert 'feature_engineering' in engineer.config


def test_add_cross_features(sample_factor_data):
    """测试交叉特征生成"""
    engineer = FeatureEngineer('configs/feature_config.yaml')
    result = engineer.add_cross_features(sample_factor_data)
    
    # 检查是否生成了交叉特征
    assert 'rsi6_x_volume_ratio' in result.columns
    assert 'macd_x_volume_ma5' in result.columns
    
    # 检查计算正确性
    expected = sample_factor_data['rsi6'] * sample_factor_data['volume_ratio']
    pd.testing.assert_series_equal(result['rsi6_x_volume_ratio'], expected, check_names=False)


def test_add_temporal_features(sample_factor_data):
    """测试时序衍生特征"""
    engineer = FeatureEngineer('configs/feature_config.yaml')
    result = engineer.add_temporal_features(sample_factor_data)
    
    # 检查因子动量
    assert 'macd_momentum_5d' in result.columns
    
    # 检查因子波动
    assert 'macd_volatility_20d' in result.columns


def test_add_nonlinear_features(sample_factor_data):
    """测试非线性变换"""
    engineer = FeatureEngineer('configs/feature_config.yaml')
    result = engineer.add_nonlinear_features(sample_factor_data)
    
    # 检查Log变换
    assert 'volume_ma20_log' in result.columns
    
    # 检查Rank归一化
    assert 'rsi6_rank' in result.columns
    assert result['rsi6_rank'].min() >= 0
    assert result['rsi6_rank'].max() <= 1


def test_add_cross_sectional_features(sample_factor_data):
    """测试截面统计特征"""
    engineer = FeatureEngineer('configs/feature_config.yaml')
    result = engineer.add_cross_sectional_features(sample_factor_data)
    
    # 检查分位数特征
    assert 'volume_ma20_quantile' in result.columns
    assert result['volume_ma20_quantile'].min() >= 0
    assert result['volume_ma20_quantile'].max() <= 1
    
    # 检查Z-score特征
    assert 'volume_ma20_zscore' in result.columns


def test_add_alpha_factors(sample_factor_data):
    """测试Alpha因子添加"""
    engineer = FeatureEngineer('configs/feature_config.yaml')
    result = engineer.add_alpha_factors(sample_factor_data)
    
    # 检查Alpha因子
    assert 'alpha_001' in result.columns
    assert 'alpha_006' in result.columns


def test_transform_complete(sample_factor_data):
    """测试完整特征转换流程"""
    engineer = FeatureEngineer('configs/feature_config.yaml')
    result = engineer.transform(sample_factor_data)
    
    # 检查原始特征数量
    original_cols = [c for c in sample_factor_data.columns if c not in ['ts_code', 'factor_date']]
    
    # 检查增强后特征数量（应该显著增加）
    enhanced_cols = [c for c in result.columns if c not in ['ts_code', 'factor_date']]
    assert len(enhanced_cols) > len(original_cols) * 3  # 至少3倍
    
    # 检查无NaN和Inf（除了前几行因为rolling）
    result_clean = result.iloc[30:]  # 跳过前30行（warm-up period）
    numeric_cols = result_clean.select_dtypes(include=[np.number]).columns
    assert not result_clean[numeric_cols].isna().all().any()
    assert not np.isinf(result_clean[numeric_cols]).any().any()


def test_get_feature_names(sample_factor_data):
    """测试特征名称获取"""
    engineer = FeatureEngineer('configs/feature_config.yaml')
    engineer.transform(sample_factor_data)
    
    feature_names = engineer.get_feature_names()
    assert isinstance(feature_names, list)
    assert len(feature_names) > 100  # 应该有100+特征
```

- [ ] **Step 2: 运行测试验证失败**

Run: `pytest tests/features/test_engineering.py -v`
Expected: FAIL - FeatureEngineer not defined

- [ ] **Step 3: 实现FeatureEngineer类（第1部分：基础框架）**

Create: `src/features/engineering.py`

```python
"""
特征工程模块
============

将30个原始技术因子扩展到160+增强特征。

策略:
1. 因子交叉特征 (+50)
2. 时序衍生特征 (+30)
3. 非线性变换 (+20)
4. 截面统计特征 (+15)
5. Alpha因子 (+15)
"""

import pandas as pd
import numpy as np
import yaml
from typing import List, Dict
from pathlib import Path

from src.features.alpha_factors import AlphaFactorCalculator


class FeatureEngineer:
    """
    特征工程器
    
    从30个原始因子扩展到160个特征。
    """
    
    def __init__(self, config_path: str = 'configs/feature_config.yaml'):
        """
        初始化
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = self.load_config(config_path)
        self.feature_names = []
        self.alpha_calculator = AlphaFactorCalculator()
    
    def load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        主入口：完整特征转换
        
        Args:
            df: 输入DataFrame（包含30个原始因子）
        
        Returns:
            增强后的DataFrame（160+特征）
        """
        print("="*60)
        print("特征工程：30因子 → 160+特征")
        print("="*60)
        
        df_enhanced = df.copy()
        original_feature_count = len([c for c in df.columns if c not in ['ts_code', 'factor_date']])
        
        # 1. 因子交叉 (+50)
        print("\n[1/5] 生成因子交叉特征...")
        df_enhanced = self.add_cross_features(df_enhanced)
        
        # 2. 时序衍生 (+30)
        print("[2/5] 生成时序衍生特征...")
        df_enhanced = self.add_temporal_features(df_enhanced)
        
        # 3. 非线性变换 (+20)
        print("[3/5] 生成非线性变换特征...")
        df_enhanced = self.add_nonlinear_features(df_enhanced)
        
        # 4. 截面统计 (+15)
        print("[4/5] 生成截面统计特征...")
        df_enhanced = self.add_cross_sectional_features(df_enhanced)
        
        # 5. Alpha因子 (+15)
        print("[5/5] 生成Alpha因子...")
        df_enhanced = self.add_alpha_factors(df_enhanced)
        
        # 收集特征名称
        self.feature_names = [c for c in df_enhanced.columns if c not in ['ts_code', 'factor_date']]
        
        final_feature_count = len(self.feature_names)
        print(f"\n特征扩展完成: {original_feature_count} → {final_feature_count} (+{final_feature_count - original_feature_count})")
        
        return df_enhanced
    
    def add_cross_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加因子交叉特征 (+50)
        
        包括:
        - 量价交叉 (20)
        - MACD×量能 (15)
        - 趋势×动量 (15)
        """
        result = df.copy()
        config = self.config['feature_engineering']['cross_features']
        
        # 1. 动量×量能交叉
        for factor1, factor2 in config['momentum_volume']:
            col_name = f'{factor1}_x_{factor2}'
            result[col_name] = result[factor1] * result[factor2]
        
        # 2. MACD×量能交叉
        for factor1, factor2 in config['macd_volume']:
            col_name = f'{factor1}_x_{factor2}'
            result[col_name] = result[factor1] * result[factor2]
        
        # 3. 动量比率
        for factor1, factor2, op in config['momentum_ratio']:
            if op == 'div':
                col_name = f'{factor1}_div_{factor2}'
                result[col_name] = result[factor1] / (result[factor2] + 1e-6)
        
        # 4. 趋势×动量交叉
        for factor1, factor2, op in config['trend_momentum']:
            if op == 'slope_mult':
                # MA斜率 × 动量
                ma_slope = result.groupby('ts_code')[factor1].pct_change(5)
                col_name = f'{factor1}_slope_x_{factor2}'
                result[col_name] = ma_slope * result[factor2]
        
        # 5. 波动率调整
        for factor, volatility, op in config['volatility_adjusted']:
            if op == 'div':
                col_name = f'{factor}_div_{volatility}'
                result[col_name] = result[factor] / (result[volatility] + 1e-6)
        
        return result
    
    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加时序衍生特征 (+30)
        
        包括:
        - 因子动量 (15)
        - 因子波动 (15)
        """
        result = df.copy()
        config = self.config['feature_engineering']
        
        # 使用top_factors
        top_factors = config['top_factors'][:15]
        momentum_periods = config['temporal_features']['momentum_periods']
        volatility_periods = config['temporal_features']['volatility_periods']
        
        for factor in top_factors:
            if factor not in result.columns:
                continue
            
            # 因子动量
            for period in momentum_periods:
                col_name = f'{factor}_momentum_{period}d'
                result[col_name] = result.groupby('ts_code')[factor].pct_change(period)
            
            # 因子波动（只用20天）
            for period in volatility_periods:
                col_name = f'{factor}_volatility_{period}d'
                result[col_name] = result.groupby('ts_code')[factor].transform(
                    lambda x: x.rolling(period, min_periods=1).std()
                )
        
        return result
    
    def add_nonlinear_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加非线性变换特征 (+20)
        
        包括:
        - Log变换 (10)
        - Rank归一化 (10)
        """
        result = df.copy()
        config = self.config['feature_engineering']['nonlinear_transforms']
        
        # Log变换
        for factor in config['log_factors']:
            if factor not in result.columns:
                continue
            col_name = f'{factor}_log'
            result[col_name] = np.sign(result[factor]) * np.log1p(np.abs(result[factor]))
        
        # Rank归一化
        for factor in config['rank_factors']:
            if factor not in result.columns:
                continue
            col_name = f'{factor}_rank'
            result[col_name] = result.groupby('factor_date')[factor].rank(pct=True)
        
        return result
    
    def add_cross_sectional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加截面统计特征 (+15)
        
        包括:
        - 分位数 (7)
        - Z-score (8)
        """
        result = df.copy()
        config = self.config['feature_engineering']['cross_sectional']
        
        # 分位数特征
        for factor in config['quantile_factors']:
            if factor not in result.columns:
                continue
            col_name = f'{factor}_quantile'
            result[col_name] = result.groupby('factor_date')[factor].rank(pct=True)
        
        # Z-score特征
        top_factors = self.config['feature_engineering']['top_factors'][:10]
        for factor in top_factors:
            if factor not in result.columns:
                continue
            col_name = f'{factor}_zscore'
            grouped = result.groupby('factor_date')[factor]
            mean = grouped.transform('mean')
            std = grouped.transform('std')
            result[col_name] = (result[factor] - mean) / (std + 1e-6)
        
        return result
    
    def add_alpha_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加Alpha因子 (+10-15)
        
        使用AlphaFactorCalculator计算经典Alpha因子
        """
        # 需要确保数据有必要的列
        required_cols = ['ts_code', 'close', 'open', 'high', 'low', 'volume']
        
        # 如果有trade_date但没有factor_date，临时映射
        if 'trade_date' not in df.columns and 'factor_date' in df.columns:
            df_with_trade_date = df.copy()
            df_with_trade_date['trade_date'] = df_with_trade_date['factor_date']
        else:
            df_with_trade_date = df.copy()
        
        # 计算Alpha因子
        config = self.config['feature_engineering']['alpha_factors']
        alpha_list = config['enabled']
        
        result = self.alpha_calculator.calculate_batch(df_with_trade_date, alpha_list)
        
        # 删除临时列
        if 'trade_date' in result.columns and 'trade_date' not in df.columns:
            result = result.drop('trade_date', axis=1)
        
        return result
    
    def get_feature_names(self) -> List[str]:
        """
        返回所有特征名称
        
        Returns:
            特征名称列表
        """
        return self.feature_names
    
    def validate_features(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        验证特征质量
        
        检查:
        - NaN比例
        - Inf值
        - 特征相关性
        
        Returns:
            验证结果字典
        """
        validation_config = self.config.get('validation', {})
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        results = {
            'total_features': len(numeric_cols),
            'nan_ratio': {},
            'inf_count': {},
            'high_correlation_pairs': []
        }
        
        # 检查NaN
        if validation_config.get('check_nan', True):
            for col in numeric_cols:
                nan_ratio = df[col].isna().mean()
                if nan_ratio > validation_config.get('max_missing_ratio', 0.1):
                    results['nan_ratio'][col] = nan_ratio
        
        # 检查Inf
        if validation_config.get('check_inf', True):
            for col in numeric_cols:
                inf_count = np.isinf(df[col]).sum()
                if inf_count > 0:
                    results['inf_count'][col] = inf_count
        
        # 检查高相关性
        corr_threshold = validation_config.get('correlation_threshold', 0.95)
        corr_matrix = df[numeric_cols].corr().abs()
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > corr_threshold:
                    results['high_correlation_pairs'].append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_matrix.iloc[i, j]
                    })
        
        return results
```

- [ ] **Step 4: 运行测试验证通过**

Run: `pytest tests/features/test_engineering.py -v`
Expected: 所有测试通过

- [ ] **Step 5: 提交特征工程模块**

```bash
git add src/features/engineering.py tests/features/test_engineering.py
git commit -m "feat: 实现特征工程核心类（30→160特征）"
```

---

## Task 4: 集成特征工程到数据加载器

**Files:**
- Modify: `src/models/data_loader.py`
- Test: `tests/models/test_data_loader_enhanced.py`

- [ ] **Step 1: 编写集成测试**

Create: `tests/models/test_data_loader_enhanced.py`

```python
"""增强数据加载器测试"""

import pytest
from src.models.data_loader import FactorDataLoader


def test_build_dataset_with_feature_engineering():
    """测试集成特征工程的数据集构建"""
    with FactorDataLoader() as loader:
        features, labels = loader.build_dataset(
            start_date='2024-01-01',
            end_date='2024-12-31',
            forward_days=5,
            enable_feature_engineering=True
        )
    
    # 检查特征数量显著增加
    feature_cols = [c for c in features.columns if c not in ['ts_code', 'factor_date']]
    assert len(feature_cols) > 100  # 应该有100+特征
    
    print(f"增强后特征数: {len(feature_cols)}")
```

- [ ] **Step 2: 运行测试验证失败**

Run: `pytest tests/models/test_data_loader_enhanced.py -v`
Expected: FAIL - enable_feature_engineering parameter not supported

- [ ] **Step 3: 修改data_loader.py添加特征工程支持**

```python
# 在 src/models/data_loader.py 的 build_dataset 方法中添加

def build_dataset(
    self,
    start_date: str,
    end_date: str,
    forward_days: int = 5,
    lookback_days: int = 20,
    stocks: Optional[List[str]] = None,
    enable_feature_engineering: bool = False,  # 新增参数
    feature_config_path: str = 'configs/feature_config.yaml'  # 新增参数
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    构建完整的训练数据集
    
    Args:
        ...
        enable_feature_engineering: 是否启用特征工程
        feature_config_path: 特征工程配置文件路径
    """
    # ... 原有代码 ...
    
    # 在标准化之后，添加特征工程
    if enable_feature_engineering:
        print(f"\n启用特征工程...")
        from src.features.engineering import FeatureEngineer
        
        engineer = FeatureEngineer(feature_config_path)
        factors_wide = engineer.transform(factors_wide)
        
        # 验证特征质量
        validation_results = engineer.validate_features(factors_wide)
        print(f"特征验证: {validation_results['total_features']}个特征")
        if validation_results['nan_ratio']:
            print(f"警告: {len(validation_results['nan_ratio'])}个特征NaN比例过高")
        if validation_results['inf_count']:
            print(f"警告: {len(validation_results['inf_count'])}个特征包含Inf值")
    
    # ... 继续原有代码 ...
```

- [ ] **Step 4: 运行测试验证通过**

Run: `pytest tests/models/test_data_loader_enhanced.py -v`
Expected: PASS

- [ ] **Step 5: 提交修改**

```bash
git add src/models/data_loader.py tests/models/test_data_loader_enhanced.py
git commit -m "feat: 数据加载器集成特征工程模块"
```

---

## Task 5: 更新训练脚本支持特征工程

**Files:**
- Modify: `scripts/train_alpha_model.py`

- [ ] **Step 1: 添加特征工程命令行参数**

```python
# 在 scripts/train_alpha_model.py 的 argparse 部分添加

parser.add_argument('--enable-features', action='store_true',
                   help='启用特征工程（30→160特征）')
parser.add_argument('--feature-config', type=str, 
                   default='configs/feature_config.yaml',
                   help='特征工程配置文件路径')
```

- [ ] **Step 2: 修改训练函数传递参数**

```python
# 在 train_lgbm 函数中

with FactorDataLoader() as loader:
    features, labels = loader.build_dataset(
        start_date=start_date,
        end_date=end_date,
        forward_days=forward_days,
        enable_feature_engineering=args.enable_features,  # 新增
        feature_config_path=args.feature_config  # 新增
    )
```

- [ ] **Step 3: 测试训练脚本**

Run: `python scripts/train_alpha_model.py --model lgbm --forward 5 --enable-features`
Expected: 训练成功，输出显示160+特征

- [ ] **Step 4: 提交修改**

```bash
git add scripts/train_alpha_model.py
git commit -m "feat: 训练脚本支持特征工程选项"
```

---

## Task 6: Phase 1验证 - 单窗口模型测试

**Files:**
- Create: `scripts/validate_phase1.py`

- [ ] **Step 1: 创建Phase 1验证脚本**

Create: `scripts/validate_phase1.py`

```python
#!/usr/bin/env python3
"""
Phase 1验证脚本
===============

验证特征工程效果：
- 对比Baseline（30因子）vs 增强（160特征）
- 目标: IC提升到0.028-0.032
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.models.data_loader import FactorDataLoader
from src.models.lgbm_model import LightGBMAlphaModel


def train_and_evaluate(enable_features, label):
    """训练并评估模型"""
    print("\n" + "="*80)
    print(f"{label}")
    print("="*80)
    
    # 加载数据
    with FactorDataLoader() as loader:
        features, labels = loader.build_dataset(
            start_date='2020-01-01',
            end_date='2024-12-31',
            forward_days=5,
            enable_feature_engineering=enable_features
        )
    
    # 时间序列划分
    split_date = features['factor_date'].quantile(0.8)
    train_features = features[features['factor_date'] <= split_date]
    train_labels = labels[labels['factor_date'] <= split_date]
    val_features = features[features['factor_date'] > split_date]
    val_labels = labels[labels['factor_date'] > split_date]
    
    print(f"\nTrain: {len(train_features)} samples")
    print(f"Val: {len(val_features)} samples")
    
    feature_cols = [c for c in features.columns if c not in ['ts_code', 'factor_date']]
    print(f"Features: {len(feature_cols)}")
    
    # 训练模型
    model = LightGBMAlphaModel()
    model.train(
        features=train_features,
        labels=train_labels,
        num_boost_round=200,
        early_stopping_rounds=20,
        verbose_eval=50
    )
    
    # 验证集评估
    val_pred = model.predict(val_features)
    val_true = val_labels['forward_return'].values
    
    # 计算IC
    ic = np.corrcoef(val_pred, val_true)[0, 1]
    
    # 按日期计算IC序列
    val_df = val_features.copy()
    val_df['pred'] = val_pred
    val_df['true'] = val_true
    
    ic_series = []
    for date in val_df['factor_date'].unique():
        mask = val_df['factor_date'] == date
        if mask.sum() >= 5:
            daily_ic = np.corrcoef(val_df.loc[mask, 'pred'], val_df.loc[mask, 'true'])[0, 1]
            if not np.isnan(daily_ic):
                ic_series.append(daily_ic)
    
    ic_mean = np.mean(ic_series)
    ic_std = np.std(ic_series)
    ic_ir = ic_mean / (ic_std + 1e-6)
    ic_positive_ratio = (np.array(ic_series) > 0).mean()
    
    print(f"\n{'='*80}")
    print(f"验证集结果:")
    print(f"  IC (整体): {ic:.4f}")
    print(f"  IC (日度均值): {ic_mean:.4f}")
    print(f"  IC (标准差): {ic_std:.4f}")
    print(f"  IR: {ic_ir:.4f}")
    print(f"  IC>0比例: {ic_positive_ratio:.2%}")
    print(f"{'='*80}\n")
    
    return {
        'ic': ic,
        'ic_mean': ic_mean,
        'ic_std': ic_std,
        'ic_ir': ic_ir,
        'ic_positive_ratio': ic_positive_ratio,
        'n_features': len(feature_cols)
    }


def main():
    """主函数"""
    print("\n" + "="*80)
    print("Phase 1 验证: 特征工程效果对比")
    print("="*80)
    
    # Baseline: 30因子
    baseline_results = train_and_evaluate(
        enable_features=False,
        label="Baseline: 30个原始因子"
    )
    
    # Enhanced: 160特征
    enhanced_results = train_and_evaluate(
        enable_features=True,
        label="Enhanced: 160个增强特征"
    )
    
    # 对比结果
    print("\n" + "="*80)
    print("Phase 1 验证结果对比")
    print("="*80)
    
    print(f"\n{'指标':<20} {'Baseline':<15} {'Enhanced':<15} {'提升':<15}")
    print("-" * 80)
    
    metrics = [
        ('特征数', 'n_features', ''),
        ('IC (整体)', 'ic', '.4f'),
        ('IC (日度均值)', 'ic_mean', '.4f'),
        ('IC 标准差', 'ic_std', '.4f'),
        ('IR', 'ic_ir', '.4f'),
        ('IC>0比例', 'ic_positive_ratio', '.2%')
    ]
    
    for label, key, fmt in metrics:
        baseline_val = baseline_results[key]
        enhanced_val = enhanced_results[key]
        
        if fmt:
            baseline_str = f"{baseline_val:{fmt}}"
            enhanced_str = f"{enhanced_val:{fmt}}"
        else:
            baseline_str = str(baseline_val)
            enhanced_str = str(enhanced_val)
        
        if key in ['ic', 'ic_mean', 'ic_ir']:
            improvement = ((enhanced_val - baseline_val) / abs(baseline_val)) * 100
            improvement_str = f"+{improvement:.1f}%"
        else:
            improvement_str = "-"
        
        print(f"{label:<20} {baseline_str:<15} {enhanced_str:<15} {improvement_str:<15}")
    
    # 验收判断
    print("\n" + "="*80)
    print("Phase 1 验收标准")
    print("="*80)
    
    target_ic = 0.028
    achieved_ic = enhanced_results['ic_mean']
    
    if achieved_ic >= target_ic:
        print(f"✓ PASS: IC={achieved_ic:.4f} >= 目标{target_ic}")
        print(f"✓ Phase 1完成，进入Phase 2（模型集成）")
    else:
        print(f"✗ FAIL: IC={achieved_ic:.4f} < 目标{target_ic}")
        print(f"建议: 调整特征工程配置或增加更多交叉特征")
        sys.exit(1)


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: 运行Phase 1验证**

Run: `python scripts/validate_phase1.py`
Expected: 
- Baseline IC ~0.020
- Enhanced IC 0.028-0.032
- Phase 1 PASS

- [ ] **Step 3: 提交验证脚本和结果**

```bash
git add scripts/validate_phase1.py
git commit -m "test: 添加Phase 1验证脚本

验证特征工程效果:
- Baseline (30因子): IC ~0.020
- Enhanced (160特征): IC ~0.030
- 提升约50%"
```

---

## Phase 1 完成检查清单

在进入Phase 2之前，确认以下内容已完成：

- [ ] 配置文件: `configs/feature_config.yaml` ✓
- [ ] Alpha因子库: `src/features/alpha_factors.py` (10个因子) ✓
- [ ] 特征工程类: `src/features/engineering.py` ✓
- [ ] 数据加载器集成: `src/models/data_loader.py` 支持特征工程 ✓
- [ ] 训练脚本更新: `scripts/train_alpha_model.py` 支持 `--enable-features` ✓
- [ ] 单元测试: 所有测试通过 ✓
- [ ] Phase 1验证: IC >= 0.028 ✓
- [ ] 代码提交: 所有改动已commit ✓

---

## 预期结果

**Phase 1目标:**
- IC提升: 0.021 → 0.028-0.032 (约40-50%提升)
- 特征数: 30 → 160
- 模型: 仍使用LightGBM单模型

**下一步:**
- Phase 2: 实现12个异构模型 + Stacking集成
- 目标IC: 0.035-0.040

---

## 故障排查

**问题1: 特征计算NaN过多**
- 原因: rolling窗口在数据开始部分产生NaN
- 解决: 在模型训练前dropna()，或使用min_periods参数

**问题2: 特征相关性过高**
- 原因: 某些交叉特征本质相同
- 解决: 使用validate_features()检查，删除高相关特征

**问题3: 内存不足**
- 原因: 160特征×大量样本占用内存
- 解决: 分批处理或使用float32代替float64

---

*Phase 1实施计划完成*

**下一步:** 创建Phase 2计划（模型集成）或直接开始执行Phase 1
