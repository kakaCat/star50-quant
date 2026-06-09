"""
因子计算基础类
==============

提供所有因子计算的基础功能：
- 输入验证
- 结果格式化
- 日志记录
- 装饰器支持
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Union, Dict, List, Any, Optional
import logging
from datetime import datetime
from functools import wraps
import time

from src.features.exceptions import DataValidationError, InsufficientDataError


# =========================================================================
# 装饰器
# =========================================================================

def validate_inputs(func):
    """验证输入数据的装饰器"""
    @wraps(func)
    def wrapper(self, klines: List[Dict[str, Any]], *args, **kwargs):
        if not klines:
            raise DataValidationError("K-line data cannot be empty", "klines")
        if not isinstance(klines, list):
            raise DataValidationError("K-line data must be a list", "klines")
        return func(self, klines, *args, **kwargs)
    return wrapper


def timing_decorator(func):
    """计时装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time

        # 将计时信息添加到结果中
        if isinstance(result, dict) and 'metadata' in result:
            result['metadata']['execution_time_ms'] = round(elapsed * 1000, 2)

        return result
    return wrapper


# =========================================================================
# 基础计算器
# =========================================================================

class BaseCalculator(ABC):
    """
    所有因子计算的抽象基类

    提供：
    - 输入验证框架
    - 标准化结果格式
    - 日志记录
    - 元数据追踪
    """

    def __init__(self, precision: int = 4):
        """
        初始化基础计算器

        Args:
            precision: 结果精度（小数位数）
        """
        self.precision = precision
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger(f"{self.__class__.__name__}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _create_result_dict(
        self,
        value: float,
        method: str,
        parameters: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        创建标准化的结果字典

        Args:
            value: 计算结果值
            method: 计算方法名称
            parameters: 计算参数
            metadata: 额外元数据

        Returns:
            标准化结果字典
        """
        result = {
            'value': round(value, self.precision) if not np.isnan(value) else None,
            'method': method,
            'parameters': parameters or {},
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat()
        }
        return result


# =========================================================================
# 技术因子计算器
# =========================================================================

class TechnicalFactorCalculator(BaseCalculator):
    """
    技术因子计算基类

    继承BaseCalculator，提供：
    - K线数据验证
    - OHLCV数据提取
    - 常用技术指标工具
    - 统一结果格式化

    所有技术因子应继承此类。
    """

    def __init__(self, precision: int = 4):
        """
        初始化技术因子计算器

        Args:
            precision: 结果精度（小数位数）
        """
        super().__init__(precision)

    # =========================================================================
    # 数据验证
    # =========================================================================

    def _validate_klines(self, klines: List[Dict], min_length: Optional[int] = None) -> None:
        """
        验证K线数据格式和内容

        Args:
            klines: K线数据列表
            min_length: 最小数据点数

        Raises:
            DataValidationError: 数据格式无效
            InsufficientDataError: 数据长度不足
        """
        if not klines:
            raise DataValidationError("K-line data cannot be empty", "klines")

        if not isinstance(klines, list):
            raise DataValidationError("K-line data must be a list", "klines")

        # 检查必需字段
        required_fields = ['open', 'high', 'low', 'close', 'volume']
        first_kline = klines[0]

        if not isinstance(first_kline, dict):
            raise DataValidationError("Each K-line must be a dictionary", "klines")

        missing_fields = [f for f in required_fields if f not in first_kline]
        if missing_fields:
            raise DataValidationError(
                f"Missing required fields: {', '.join(missing_fields)}",
                "klines"
            )

        # 检查最小长度
        if min_length is not None and len(klines) < min_length:
            raise InsufficientDataError(min_length, len(klines))

    def _validate_period(self, period: int, min_period: int = 1, max_period: int = 500) -> None:
        """
        验证周期参数

        Args:
            period: 周期值
            min_period: 最小周期
            max_period: 最大周期

        Raises:
            DataValidationError: 周期无效
        """
        if not isinstance(period, int):
            raise DataValidationError("Period must be an integer", "period")

        if period < min_period or period > max_period:
            raise DataValidationError(
                f"Period must be between {min_period} and {max_period}",
                "period"
            )

    # =========================================================================
    # 数据提取
    # =========================================================================

    def _extract_closes(self, klines: List[Dict]) -> np.ndarray:
        """提取收盘价"""
        return np.array([k['close'] for k in klines], dtype=np.float64)

    def _extract_opens(self, klines: List[Dict]) -> np.ndarray:
        """提取开盘价"""
        return np.array([k['open'] for k in klines], dtype=np.float64)

    def _extract_highs(self, klines: List[Dict]) -> np.ndarray:
        """提取最高价"""
        return np.array([k['high'] for k in klines], dtype=np.float64)

    def _extract_lows(self, klines: List[Dict]) -> np.ndarray:
        """提取最低价"""
        return np.array([k['low'] for k in klines], dtype=np.float64)

    def _extract_volumes(self, klines: List[Dict]) -> np.ndarray:
        """提取成交量"""
        return np.array([k['volume'] for k in klines], dtype=np.float64)

    # =========================================================================
    # 常用技术指标工具
    # =========================================================================

    def _sma(self, series: np.ndarray, period: int) -> float:
        """
        计算简单移动平均线（使用TA-Lib）

        Args:
            series: 价格序列
            period: 周期

        Returns:
            SMA值
        """
        if len(series) < period:
            raise InsufficientDataError(period, len(series))

        try:
            import talib
            sma_values = talib.SMA(series, timeperiod=period)
            sma = float(sma_values[-1]) if not np.isnan(sma_values[-1]) else 0.0
            return sma
        except ImportError:
            # 回退到numpy实现
            return float(np.mean(series[-period:]))

    def _ema(self, series: np.ndarray, period: int) -> float:
        """
        计算指数移动平均线（使用TA-Lib）

        Args:
            series: 价格序列
            period: 周期

        Returns:
            EMA值
        """
        n = len(series)
        if n < period:
            raise InsufficientDataError(period, n)

        try:
            import talib
            ema_values = talib.EMA(series, timeperiod=period)
            ema = float(ema_values[-1]) if not np.isnan(ema_values[-1]) else 0.0
            return ema
        except ImportError:
            # 回退到numpy实现
            alpha = 2.0 / (period + 1)
            ema = series[0]
            for price in series[1:]:
                ema = alpha * price + (1 - alpha) * ema
            return float(ema)

    def _std(self, series: np.ndarray, period: int) -> float:
        """
        计算标准差

        Args:
            series: 价格序列
            period: 周期

        Returns:
            标准差值
        """
        if len(series) < period:
            raise InsufficientDataError(period, len(series))

        return float(np.std(series[-period:], ddof=1))

    def _true_range_series(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray
    ) -> np.ndarray:
        """
        计算真实波幅序列

        Args:
            highs: 最高价序列
            lows: 最低价序列
            closes: 收盘价序列

        Returns:
            真实波幅序列
        """
        prev_close = np.roll(closes, 1)
        prev_close[0] = closes[0]

        tr1 = highs - lows
        tr2 = np.abs(highs - prev_close)
        tr3 = np.abs(lows - prev_close)

        return np.maximum(np.maximum(tr1, tr2), tr3)
