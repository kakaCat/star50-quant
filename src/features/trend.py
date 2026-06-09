"""
趋势因子模块
============

实现趋势类技术因子：
- MA (Moving Average) - 移动平均线
- EMA (Exponential Moving Average) - 指数移动平均线
- BOLL (Bollinger Bands) - 布林带
- ATR (Average True Range) - 平均真实波幅
"""

import numpy as np
from typing import Dict, Any, List

from src.features.base import TechnicalFactorCalculator, validate_inputs, timing_decorator
from src.features.exceptions import InsufficientDataError


class TrendFactors(TechnicalFactorCalculator):
    """
    趋势指标计算器

    提供MA、EMA、BOLL、ATR等趋势相关计算。
    使用TA-Lib实现高性能计算。
    """

    def get_supported_methods(self) -> List[str]:
        """返回支持的趋势指标列表"""
        return [
            'ma5', 'ma10', 'ma20', 'ma60',
            'ema5', 'ema10', 'ema20',
            'boll_upper', 'boll_middle', 'boll_lower',
            'atr14'
        ]

    # =========================================================================
    # MA (Simple Moving Average)
    # =========================================================================

    def _calc_ma(self, klines: List[Dict[str, Any]], period: int) -> float:
        """
        计算简单移动平均线（使用TA-Lib）

        MA = sum(Close) / period

        Args:
            klines: K线数据
            period: 周期

        Returns:
            MA值
        """
        closes = self._extract_closes(klines)
        n = len(closes)

        if n < period:
            raise InsufficientDataError(
                required=period,
                actual=n,
                message=f"MA requires at least {period} data points"
            )

        try:
            import talib
            ma_values = talib.SMA(closes, timeperiod=period)
            ma_value = float(ma_values[-1]) if not np.isnan(ma_values[-1]) else 0.0

        except ImportError:
            # 回退到numpy实现
            ma_value = float(np.mean(closes[-period:]))

        return ma_value

    @validate_inputs
    @timing_decorator
    def ma5(self, klines: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算5日均线"""
        ma_value = self._calc_ma(klines, period=5)

        return self._create_result_dict(
            value=ma_value,
            method='ma5',
            parameters={'period': 5},
            metadata={
                'data_points': len(klines),
                'current_price': float(klines[-1]['close'])
            }
        )

    @validate_inputs
    @timing_decorator
    def ma10(self, klines: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算10日均线"""
        ma_value = self._calc_ma(klines, period=10)

        return self._create_result_dict(
            value=ma_value,
            method='ma10',
            parameters={'period': 10},
            metadata={
                'data_points': len(klines),
                'current_price': float(klines[-1]['close'])
            }
        )

    @validate_inputs
    @timing_decorator
    def ma20(self, klines: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算20日均线"""
        ma_value = self._calc_ma(klines, period=20)

        return self._create_result_dict(
            value=ma_value,
            method='ma20',
            parameters={'period': 20},
            metadata={
                'data_points': len(klines),
                'current_price': float(klines[-1]['close'])
            }
        )

    @validate_inputs
    @timing_decorator
    def ma60(self, klines: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算60日均线"""
        ma_value = self._calc_ma(klines, period=60)

        return self._create_result_dict(
            value=ma_value,
            method='ma60',
            parameters={'period': 60},
            metadata={
                'data_points': len(klines),
                'current_price': float(klines[-1]['close'])
            }
        )

    # =========================================================================
    # EMA (Exponential Moving Average)
    # =========================================================================

    def _calc_ema(self, klines: List[Dict[str, Any]], period: int) -> float:
        """
        计算指数移动平均线（使用TA-Lib）

        EMA = alpha * Close + (1 - alpha) * EMA_prev
        alpha = 2 / (period + 1)

        Args:
            klines: K线数据
            period: 周期

        Returns:
            EMA值
        """
        closes = self._extract_closes(klines)
        n = len(closes)

        if n < period:
            raise InsufficientDataError(
                required=period,
                actual=n,
                message=f"EMA requires at least {period} data points"
            )

        try:
            import talib
            ema_values = talib.EMA(closes, timeperiod=period)
            ema_value = float(ema_values[-1]) if not np.isnan(ema_values[-1]) else 0.0

        except ImportError:
            # 回退到手动实现
            alpha = 2.0 / (period + 1)
            ema = closes[0]
            for price in closes[1:]:
                ema = alpha * price + (1 - alpha) * ema
            ema_value = float(ema)

        return ema_value

    @validate_inputs
    @timing_decorator
    def ema5(self, klines: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算5日指数移动平均"""
        ema_value = self._calc_ema(klines, period=5)

        return self._create_result_dict(
            value=ema_value,
            method='ema5',
            parameters={'period': 5},
            metadata={
                'data_points': len(klines),
                'current_price': float(klines[-1]['close'])
            }
        )

    @validate_inputs
    @timing_decorator
    def ema10(self, klines: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算10日指数移动平均"""
        ema_value = self._calc_ema(klines, period=10)

        return self._create_result_dict(
            value=ema_value,
            method='ema10',
            parameters={'period': 10},
            metadata={
                'data_points': len(klines),
                'current_price': float(klines[-1]['close'])
            }
        )

    @validate_inputs
    @timing_decorator
    def ema20(self, klines: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算20日指数移动平均"""
        ema_value = self._calc_ema(klines, period=20)

        return self._create_result_dict(
            value=ema_value,
            method='ema20',
            parameters={'period': 20},
            metadata={
                'data_points': len(klines),
                'current_price': float(klines[-1]['close'])
            }
        )

    # =========================================================================
    # Bollinger Bands
    # =========================================================================

    def _calc_bollinger_bands(
        self,
        klines: List[Dict[str, Any]],
        period: int = 20,
        num_std: float = 2.0
    ) -> Dict[str, float]:
        """
        计算布林带（使用TA-Lib）

        Middle Band = MA(period)
        Upper Band = Middle + (num_std * std)
        Lower Band = Middle - (num_std * std)

        Args:
            klines: K线数据
            period: 周期（默认20）
            num_std: 标准差倍数（默认2）

        Returns:
            包含upper、middle、lower的字典
        """
        closes = self._extract_closes(klines)
        n = len(closes)

        if n < period:
            raise InsufficientDataError(
                required=period,
                actual=n,
                message=f"Bollinger Bands requires at least {period} data points"
            )

        try:
            import talib
            upper, middle, lower = talib.BBANDS(
                closes,
                timeperiod=period,
                nbdevup=num_std,
                nbdevdn=num_std,
                matype=0  # SMA
            )

            upper_value = float(upper[-1]) if not np.isnan(upper[-1]) else 0.0
            middle_value = float(middle[-1]) if not np.isnan(middle[-1]) else 0.0
            lower_value = float(lower[-1]) if not np.isnan(lower[-1]) else 0.0

        except ImportError:
            # 回退到手动实现
            middle_value = float(np.mean(closes[-period:]))
            std_value = float(np.std(closes[-period:], ddof=1))
            upper_value = middle_value + num_std * std_value
            lower_value = middle_value - num_std * std_value

        return {
            'upper': upper_value,
            'middle': middle_value,
            'lower': lower_value
        }

    @validate_inputs
    @timing_decorator
    def boll_upper(self, klines: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算布林带上轨"""
        boll = self._calc_bollinger_bands(klines)

        return self._create_result_dict(
            value=boll['upper'],
            method='boll_upper',
            parameters={'period': 20, 'num_std': 2.0},
            metadata={
                'data_points': len(klines),
                'middle': boll['middle'],
                'lower': boll['lower'],
                'current_price': float(klines[-1]['close'])
            }
        )

    @validate_inputs
    @timing_decorator
    def boll_middle(self, klines: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算布林带中轨"""
        boll = self._calc_bollinger_bands(klines)

        return self._create_result_dict(
            value=boll['middle'],
            method='boll_middle',
            parameters={'period': 20},
            metadata={
                'data_points': len(klines),
                'upper': boll['upper'],
                'lower': boll['lower'],
                'current_price': float(klines[-1]['close'])
            }
        )

    @validate_inputs
    @timing_decorator
    def boll_lower(self, klines: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算布林带下轨"""
        boll = self._calc_bollinger_bands(klines)

        return self._create_result_dict(
            value=boll['lower'],
            method='boll_lower',
            parameters={'period': 20, 'num_std': 2.0},
            metadata={
                'data_points': len(klines),
                'middle': boll['middle'],
                'upper': boll['upper'],
                'current_price': float(klines[-1]['close'])
            }
        )

    # =========================================================================
    # ATR (Average True Range)
    # =========================================================================

    @validate_inputs
    @timing_decorator
    def atr14(self, klines: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        计算14日平均真实波幅（使用TA-Lib）

        TR = max(high - low, |high - prev_close|, |low - prev_close|)
        ATR = MA(TR, period)

        Args:
            klines: K线数据

        Returns:
            结果字典，包含ATR值
        """
        period = 14
        n = len(klines)

        if n < period + 1:
            raise InsufficientDataError(
                required=period + 1,
                actual=n,
                message=f"ATR requires at least {period + 1} data points"
            )

        highs = self._extract_highs(klines)
        lows = self._extract_lows(klines)
        closes = self._extract_closes(klines)

        try:
            import talib
            atr_values = talib.ATR(highs, lows, closes, timeperiod=period)
            atr_value = float(atr_values[-1]) if not np.isnan(atr_values[-1]) else 0.0

        except ImportError:
            # 回退到手动实现
            tr_series = self._true_range_series(highs, lows, closes)
            atr_value = float(np.mean(tr_series[-period:]))

        return self._create_result_dict(
            value=atr_value,
            method='atr14',
            parameters={'period': 14},
            metadata={
                'data_points': n,
                'current_price': float(closes[-1]),
                'volatility_pct': round((atr_value / closes[-1]) * 100, 2) if closes[-1] > 0 else 0.0
            }
        )
