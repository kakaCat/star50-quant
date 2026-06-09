"""
量价因子模块
============

实现量价类技术因子：
- OBV (On Balance Volume)
- MFI (Money Flow Index)
- VWAP (Volume Weighted Average Price)
- 成交量均线
"""

import numpy as np
from typing import Dict, Any, List

from src.features.base import TechnicalFactorCalculator, validate_inputs, timing_decorator
from src.features.exceptions import InsufficientDataError


class VolumeFactors(TechnicalFactorCalculator):
    """
    量价指标计算器

    提供OBV、MFI、VWAP和成交量相关计算。
    使用TA-Lib实现高性能计算。
    """

    def get_supported_methods(self) -> List[str]:
        """返回支持的量价指标列表"""
        return [
            'obv', 'mfi14', 'vwap',
            'volume_ma5', 'volume_ma10', 'volume_ma20',
            'volume_ratio'
        ]

    # =========================================================================
    # OBV (On Balance Volume)
    # =========================================================================

    @validate_inputs
    @timing_decorator
    def obv(self, klines: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        计算能量潮指标（使用TA-Lib）

        OBV = 累计(成交量 * sign(价格变化))
        - 如果 close > prev_close: 加上成交量
        - 如果 close < prev_close: 减去成交量
        - 如果 close == prev_close: 不变

        Args:
            klines: K线数据，需包含'close'和'volume'字段

        Returns:
            结果字典，包含OBV值
        """
        n = len(klines)

        if n < 2:
            raise InsufficientDataError(
                required=2,
                actual=n,
                message="OBV requires at least 2 data points"
            )

        closes = self._extract_closes(klines)
        volumes = self._extract_volumes(klines)

        try:
            import talib
            obv_values = talib.OBV(closes, volumes)
            obv_value = float(obv_values[-1]) if not np.isnan(obv_values[-1]) else 0.0

        except ImportError:
            # 回退到手动实现
            obv = 0.0
            for i in range(1, n):
                if closes[i] > closes[i-1]:
                    obv += volumes[i]
                elif closes[i] < closes[i-1]:
                    obv -= volumes[i]
            obv_value = obv

        return self._create_result_dict(
            value=obv_value,
            method='obv',
            parameters={},
            metadata={
                'data_points': n,
                'latest_volume': float(volumes[-1]),
                'latest_close': float(closes[-1]),
                'trend': 'bullish' if obv_value > 0 else 'bearish'
            }
        )

    # =========================================================================
    # MFI (Money Flow Index)
    # =========================================================================

    @validate_inputs
    @timing_decorator
    def mfi14(self, klines: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        计算14日资金流量指标（使用TA-Lib）

        MFI = 100 - 100 / (1 + 资金流量比)
        资金流量比 = 正资金流量 / 负资金流量
        典型价格 = (最高价 + 最低价 + 收盘价) / 3
        资金流量 = 典型价格 * 成交量

        Args:
            klines: K线数据，需包含'high'、'low'、'close'、'volume'字段

        Returns:
            结果字典，包含MFI值（0-100）
        """
        period = 14
        n = len(klines)

        if n < period + 1:
            raise InsufficientDataError(
                required=period + 1,
                actual=n,
                message=f"MFI requires at least {period + 1} data points"
            )

        highs = self._extract_highs(klines)
        lows = self._extract_lows(klines)
        closes = self._extract_closes(klines)
        volumes = self._extract_volumes(klines)

        try:
            import talib
            mfi_values = talib.MFI(highs, lows, closes, volumes, timeperiod=period)
            mfi_value = float(mfi_values[-1]) if not np.isnan(mfi_values[-1]) else 50.0

        except ImportError:
            # 回退到手动实现
            typical_prices = (highs + lows + closes) / 3.0
            money_flow = typical_prices * volumes

            positive_flow = 0.0
            negative_flow = 0.0

            for i in range(n - period, n):
                if typical_prices[i] > typical_prices[i-1]:
                    positive_flow += money_flow[i]
                elif typical_prices[i] < typical_prices[i-1]:
                    negative_flow += money_flow[i]

            if negative_flow == 0:
                mfi_value = 100.0
            else:
                money_ratio = positive_flow / negative_flow
                mfi_value = 100.0 - (100.0 / (1.0 + money_ratio))

        return self._create_result_dict(
            value=mfi_value,
            method='mfi14',
            parameters={'period': 14},
            metadata={
                'data_points': n,
                'overbought': mfi_value > 80,
                'oversold': mfi_value < 20
            }
        )

    # =========================================================================
    # VWAP (Volume Weighted Average Price)
    # =========================================================================

    @validate_inputs
    @timing_decorator
    def vwap(self, klines: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        计算20日成交量加权平均价

        VWAP = sum(典型价格 * 成交量) / sum(成交量)
        典型价格 = (最高价 + 最低价 + 收盘价) / 3

        Args:
            klines: K线数据

        Returns:
            结果字典，包含VWAP值
        """
        period = 20
        n = len(klines)

        if n < period:
            raise InsufficientDataError(
                required=period,
                actual=n,
                message=f"VWAP requires at least {period} data points"
            )

        highs = self._extract_highs(klines)
        lows = self._extract_lows(klines)
        closes = self._extract_closes(klines)
        volumes = self._extract_volumes(klines)

        # 计算典型价格
        typical_prices = (highs + lows + closes) / 3.0

        # 计算VWAP
        recent_typical = typical_prices[-period:]
        recent_volumes = volumes[-period:]

        vwap_value = float(np.sum(recent_typical * recent_volumes) / np.sum(recent_volumes))

        return self._create_result_dict(
            value=vwap_value,
            method='vwap',
            parameters={'period': period},
            metadata={
                'data_points': n,
                'current_price': float(closes[-1]),
                'price_vs_vwap': 'above' if closes[-1] > vwap_value else 'below'
            }
        )

    # =========================================================================
    # 成交量均线
    # =========================================================================

    def _calc_volume_ma(self, klines: List[Dict[str, Any]], period: int) -> float:
        """
        计算成交量移动平均

        Args:
            klines: K线数据
            period: 周期

        Returns:
            成交量均线值
        """
        n = len(klines)

        if n < period:
            raise InsufficientDataError(
                required=period,
                actual=n,
                message=f"Volume MA requires at least {period} data points"
            )

        volumes = self._extract_volumes(klines)
        volume_ma = float(np.mean(volumes[-period:]))

        return volume_ma

    @validate_inputs
    @timing_decorator
    def volume_ma5(self, klines: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算5日成交量均线"""
        volume_ma = self._calc_volume_ma(klines, period=5)

        return self._create_result_dict(
            value=volume_ma,
            method='volume_ma5',
            parameters={'period': 5},
            metadata={
                'data_points': len(klines),
                'current_volume': float(klines[-1]['volume'])
            }
        )

    @validate_inputs
    @timing_decorator
    def volume_ma10(self, klines: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算10日成交量均线"""
        volume_ma = self._calc_volume_ma(klines, period=10)

        return self._create_result_dict(
            value=volume_ma,
            method='volume_ma10',
            parameters={'period': 10},
            metadata={
                'data_points': len(klines),
                'current_volume': float(klines[-1]['volume'])
            }
        )

    @validate_inputs
    @timing_decorator
    def volume_ma20(self, klines: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算20日成交量均线"""
        volume_ma = self._calc_volume_ma(klines, period=20)

        return self._create_result_dict(
            value=volume_ma,
            method='volume_ma20',
            parameters={'period': 20},
            metadata={
                'data_points': len(klines),
                'current_volume': float(klines[-1]['volume'])
            }
        )

    # =========================================================================
    # 量比
    # =========================================================================

    @validate_inputs
    @timing_decorator
    def volume_ratio(self, klines: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        计算量比

        量比 = 当前成交量 / 5日平均成交量

        Args:
            klines: K线数据

        Returns:
            结果字典，包含量比值
        """
        n = len(klines)

        if n < 6:
            raise InsufficientDataError(
                required=6,
                actual=n,
                message="Volume ratio requires at least 6 data points"
            )

        volumes = self._extract_volumes(klines)
        current_volume = volumes[-1]
        avg_volume = np.mean(volumes[-6:-1])  # 前5日平均

        ratio = float(current_volume / avg_volume) if avg_volume > 0 else 0.0

        return self._create_result_dict(
            value=ratio,
            method='volume_ratio',
            parameters={},
            metadata={
                'data_points': n,
                'current_volume': float(current_volume),
                'avg_volume': float(avg_volume),
                'status': 'high' if ratio > 2.0 else 'normal' if ratio > 0.5 else 'low'
            }
        )
