"""
动量因子模块
============

实现动量类技术因子：
- MACD (Moving Average Convergence Divergence)
- RSI (Relative Strength Index)
- ROC (Rate of Change)
- Momentum
"""

import numpy as np
from typing import Dict, Any, List

from src.features.base import TechnicalFactorCalculator, validate_inputs, timing_decorator
from src.features.exceptions import InsufficientDataError


class MomentumFactors(TechnicalFactorCalculator):
    """
    动量指标计算器

    提供MACD、RSI、ROC和Momentum计算。
    使用TA-Lib实现高性能计算。
    """

    def get_supported_methods(self) -> List[str]:
        """返回支持的动量指标列表"""
        return [
            'macd', 'macd_signal', 'macd_histogram',
            'rsi6', 'rsi12', 'rsi24',
            'roc_5', 'roc_10', 'roc_20',
            'momentum_5', 'momentum_10', 'momentum_20'
        ]

    # =========================================================================
    # MACD (Moving Average Convergence Divergence)
    # =========================================================================

    def _calc_macd(self, klines: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        计算MACD、信号线和柱状图（使用TA-Lib）

        MACD = EMA12 - EMA26
        Signal = EMA9 of MACD
        Histogram = MACD - Signal

        Args:
            klines: K线数据

        Returns:
            包含macd、signal、histogram的字典
        """
        closes = self._extract_closes(klines)
        n = len(closes)

        if n < 26:
            raise InsufficientDataError(
                required=26,
                actual=n,
                message="MACD requires at least 26 data points"
            )

        try:
            import talib
            # 使用TA-Lib计算MACD（C实现，10倍速度提升）
            macd_line, signal_line, histogram = talib.MACD(
                closes,
                fastperiod=12,
                slowperiod=26,
                signalperiod=9
            )

            # 获取最后有效值（处理NaN）
            macd_value = float(macd_line[-1]) if not np.isnan(macd_line[-1]) else 0.0
            signal = float(signal_line[-1]) if not np.isnan(signal_line[-1]) else 0.0
            hist = float(histogram[-1]) if not np.isnan(histogram[-1]) else 0.0

        except ImportError:
            # 回退到手动实现
            ema12 = self._ema(closes, 12)
            ema26 = self._ema(closes, 26)
            macd_value = ema12 - ema26

            # 计算信号线（MACD的9日EMA）
            macd_series = np.zeros(n)
            for i in range(26, n):
                ema12_i = self._ema(closes[:i+1], 12)
                ema26_i = self._ema(closes[:i+1], 26)
                macd_series[i] = ema12_i - ema26_i

            signal = self._ema(macd_series[26:], 9)
            hist = macd_value - signal

        return {
            'macd': macd_value,
            'signal': signal,
            'histogram': hist
        }

    @validate_inputs
    @timing_decorator
    def macd(self, klines: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        计算MACD线（EMA12 - EMA26）

        Args:
            klines: K线数据，需包含'close'字段

        Returns:
            结果字典，包含MACD值
        """
        result = self._calc_macd(klines)

        return self._create_result_dict(
            value=result['macd'],
            method='macd',
            parameters={'fast_period': 12, 'slow_period': 26},
            metadata={
                'data_points': len(klines),
                'signal': result['signal'],
                'histogram': result['histogram']
            }
        )

    @validate_inputs
    @timing_decorator
    def macd_signal(self, klines: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        计算MACD信号线（MACD的EMA9）

        Args:
            klines: K线数据

        Returns:
            结果字典，包含信号线值
        """
        result = self._calc_macd(klines)

        return self._create_result_dict(
            value=result['signal'],
            method='macd_signal',
            parameters={'signal_period': 9},
            metadata={
                'data_points': len(klines),
                'macd': result['macd'],
                'histogram': result['histogram']
            }
        )

    @validate_inputs
    @timing_decorator
    def macd_histogram(self, klines: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        计算MACD柱状图（MACD - Signal）

        Args:
            klines: K线数据

        Returns:
            结果字典，包含柱状图值
        """
        result = self._calc_macd(klines)

        return self._create_result_dict(
            value=result['histogram'],
            method='macd_histogram',
            parameters={},
            metadata={
                'data_points': len(klines),
                'macd': result['macd'],
                'signal': result['signal']
            }
        )

    # =========================================================================
    # RSI (Relative Strength Index)
    # =========================================================================

    def _calc_rsi(self, klines: List[Dict[str, Any]], period: int = 14) -> float:
        """
        计算RSI指标（使用TA-Lib）

        RSI = 100 - 100 / (1 + RS)
        RS = Average Gain / Average Loss

        Args:
            klines: K线数据
            period: 周期（默认14）

        Returns:
            RSI值（0-100）
        """
        closes = self._extract_closes(klines)
        n = len(closes)

        if n < period + 1:
            raise InsufficientDataError(
                required=period + 1,
                actual=n,
                message=f"RSI requires at least {period + 1} data points"
            )

        try:
            import talib
            rsi_values = talib.RSI(closes, timeperiod=period)
            rsi_value = float(rsi_values[-1]) if not np.isnan(rsi_values[-1]) else 50.0

        except ImportError:
            # 回退到手动实现
            deltas = np.diff(closes)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)

            avg_gain = np.mean(gains[:period])
            avg_loss = np.mean(losses[:period])

            if avg_loss == 0:
                rsi_value = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi_value = 100.0 - (100.0 / (1.0 + rs))

        return rsi_value

    @validate_inputs
    @timing_decorator
    def rsi6(self, klines: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算6日RSI"""
        rsi_value = self._calc_rsi(klines, period=6)

        return self._create_result_dict(
            value=rsi_value,
            method='rsi6',
            parameters={'period': 6},
            metadata={
                'data_points': len(klines),
                'overbought': rsi_value > 80,
                'oversold': rsi_value < 20
            }
        )

    @validate_inputs
    @timing_decorator
    def rsi12(self, klines: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算12日RSI"""
        rsi_value = self._calc_rsi(klines, period=12)

        return self._create_result_dict(
            value=rsi_value,
            method='rsi12',
            parameters={'period': 12},
            metadata={
                'data_points': len(klines),
                'overbought': rsi_value > 70,
                'oversold': rsi_value < 30
            }
        )

    @validate_inputs
    @timing_decorator
    def rsi24(self, klines: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算24日RSI"""
        rsi_value = self._calc_rsi(klines, period=24)

        return self._create_result_dict(
            value=rsi_value,
            method='rsi24',
            parameters={'period': 24},
            metadata={
                'data_points': len(klines),
                'overbought': rsi_value > 70,
                'oversold': rsi_value < 30
            }
        )

    # =========================================================================
    # ROC (Rate of Change)
    # =========================================================================

    def _calc_roc(self, klines: List[Dict[str, Any]], period: int) -> float:
        """
        计算变动率指标（使用TA-Lib）

        ROC = (Close - Close[n periods ago]) / Close[n periods ago] * 100

        Args:
            klines: K线数据
            period: 周期

        Returns:
            ROC值（百分比）
        """
        closes = self._extract_closes(klines)
        n = len(closes)

        if n < period + 1:
            raise InsufficientDataError(
                required=period + 1,
                actual=n,
                message=f"ROC requires at least {period + 1} data points"
            )

        try:
            import talib
            roc_values = talib.ROC(closes, timeperiod=period)
            roc_value = float(roc_values[-1]) if not np.isnan(roc_values[-1]) else 0.0

        except ImportError:
            # 回退到手动实现
            current = closes[-1]
            past = closes[-period - 1]
            roc_value = ((current - past) / past) * 100.0 if past != 0 else 0.0

        return roc_value

    @validate_inputs
    @timing_decorator
    def roc_5(self, klines: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算5日ROC"""
        roc_value = self._calc_roc(klines, period=5)

        return self._create_result_dict(
            value=roc_value,
            method='roc_5',
            parameters={'period': 5},
            metadata={'data_points': len(klines)}
        )

    @validate_inputs
    @timing_decorator
    def roc_10(self, klines: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算10日ROC"""
        roc_value = self._calc_roc(klines, period=10)

        return self._create_result_dict(
            value=roc_value,
            method='roc_10',
            parameters={'period': 10},
            metadata={'data_points': len(klines)}
        )

    @validate_inputs
    @timing_decorator
    def roc_20(self, klines: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算20日ROC"""
        roc_value = self._calc_roc(klines, period=20)

        return self._create_result_dict(
            value=roc_value,
            method='roc_20',
            parameters={'period': 20},
            metadata={'data_points': len(klines)}
        )

    # =========================================================================
    # Momentum
    # =========================================================================

    def _calc_momentum(self, klines: List[Dict[str, Any]], period: int) -> float:
        """
        计算动量指标

        Momentum = Close - Close[n periods ago]

        Args:
            klines: K线数据
            period: 周期

        Returns:
            动量值
        """
        closes = self._extract_closes(klines)
        n = len(closes)

        if n < period + 1:
            raise InsufficientDataError(
                required=period + 1,
                actual=n,
                message=f"Momentum requires at least {period + 1} data points"
            )

        try:
            import talib
            mom_values = talib.MOM(closes, timeperiod=period)
            momentum = float(mom_values[-1]) if not np.isnan(mom_values[-1]) else 0.0

        except ImportError:
            # 回退到手动实现
            current = closes[-1]
            past = closes[-period - 1]
            momentum = current - past

        return momentum

    @validate_inputs
    @timing_decorator
    def momentum_5(self, klines: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算5日动量"""
        momentum = self._calc_momentum(klines, period=5)

        return self._create_result_dict(
            value=momentum,
            method='momentum_5',
            parameters={'period': 5},
            metadata={'data_points': len(klines)}
        )

    @validate_inputs
    @timing_decorator
    def momentum_10(self, klines: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算10日动量"""
        momentum = self._calc_momentum(klines, period=10)

        return self._create_result_dict(
            value=momentum,
            method='momentum_10',
            parameters={'period': 10},
            metadata={'data_points': len(klines)}
        )

    @validate_inputs
    @timing_decorator
    def momentum_20(self, klines: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算20日动量"""
        momentum = self._calc_momentum(klines, period=20)

        return self._create_result_dict(
            value=momentum,
            method='momentum_20',
            parameters={'period': 20},
            metadata={'data_points': len(klines)}
        )
