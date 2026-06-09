#!/usr/bin/env python3
"""
因子测试脚本
============

测试因子计算功能，验证计算结果。
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features.momentum import MomentumFactors
from src.features.volume import VolumeFactors
from src.features.trend import TrendFactors


def create_test_klines():
    """创建测试K线数据"""
    # 模拟60天K线数据
    klines = []
    base_price = 100.0

    for i in range(60):
        price = base_price + (i % 10) - 5  # 简单的波动模式
        klines.append({
            'open': price - 0.5,
            'high': price + 1.0,
            'low': price - 1.0,
            'close': price,
            'volume': 1000000 + (i * 10000),
            'amount': price * (1000000 + i * 10000)
        })

    return klines


def test_momentum_factors():
    """测试动量因子"""
    print("\n" + "="*60)
    print("测试动量因子")
    print("="*60)

    calculator = MomentumFactors(precision=4)
    klines = create_test_klines()

    # 测试MACD
    print("\n1. MACD测试:")
    try:
        result = calculator.macd(klines)
        print(f"   ✓ MACD: {result['value']}")
        print(f"     Signal: {result['metadata']['signal']}")
        print(f"     Histogram: {result['metadata']['histogram']}")
        print(f"     执行时间: {result['metadata'].get('execution_time_ms', 'N/A')}ms")
    except Exception as e:
        print(f"   ✗ MACD失败: {e}")

    # 测试RSI
    print("\n2. RSI测试:")
    for method_name in ['rsi6', 'rsi12', 'rsi24']:
        try:
            method = getattr(calculator, method_name)
            result = method(klines)
            print(f"   ✓ {method_name.upper()}: {result['value']}")
        except Exception as e:
            print(f"   ✗ {method_name.upper()}失败: {e}")

    # 测试ROC
    print("\n3. ROC测试:")
    for method_name in ['roc_5', 'roc_10', 'roc_20']:
        try:
            method = getattr(calculator, method_name)
            result = method(klines)
            print(f"   ✓ {method_name.upper()}: {result['value']}%")
        except Exception as e:
            print(f"   ✗ {method_name.upper()}失败: {e}")

    # 测试Momentum
    print("\n4. Momentum测试:")
    for method_name in ['momentum_5', 'momentum_10', 'momentum_20']:
        try:
            method = getattr(calculator, method_name)
            result = method(klines)
            print(f"   ✓ {method_name.upper()}: {result['value']}")
        except Exception as e:
            print(f"   ✗ {method_name.upper()}失败: {e}")


def test_volume_factors():
    """测试量价因子"""
    print("\n" + "="*60)
    print("测试量价因子")
    print("="*60)

    calculator = VolumeFactors(precision=4)
    klines = create_test_klines()

    # 测试OBV
    print("\n1. OBV测试:")
    try:
        result = calculator.obv(klines)
        print(f"   ✓ OBV: {result['value']}")
        print(f"     趋势: {result['metadata']['trend']}")
    except Exception as e:
        print(f"   ✗ OBV失败: {e}")

    # 测试MFI
    print("\n2. MFI测试:")
    try:
        result = calculator.mfi14(klines)
        print(f"   ✓ MFI14: {result['value']}")
        print(f"     超买: {result['metadata']['overbought']}")
        print(f"     超卖: {result['metadata']['oversold']}")
    except Exception as e:
        print(f"   ✗ MFI14失败: {e}")

    # 测试VWAP
    print("\n3. VWAP测试:")
    try:
        result = calculator.vwap(klines)
        print(f"   ✓ VWAP: {result['value']}")
        print(f"     当前价格: {result['metadata']['current_price']}")
        print(f"     价格vs VWAP: {result['metadata']['price_vs_vwap']}")
    except Exception as e:
        print(f"   ✗ VWAP失败: {e}")

    # 测试成交量均线
    print("\n4. 成交量均线测试:")
    for method_name in ['volume_ma5', 'volume_ma10', 'volume_ma20']:
        try:
            method = getattr(calculator, method_name)
            result = method(klines)
            print(f"   ✓ {method_name.upper()}: {result['value']:.0f}")
        except Exception as e:
            print(f"   ✗ {method_name.upper()}失败: {e}")

    # 测试量比
    print("\n5. 量比测试:")
    try:
        result = calculator.volume_ratio(klines)
        print(f"   ✓ 量比: {result['value']}")
        print(f"     状态: {result['metadata']['status']}")
    except Exception as e:
        print(f"   ✗ 量比失败: {e}")


def test_trend_factors():
    """测试趋势因子"""
    print("\n" + "="*60)
    print("测试趋势因子")
    print("="*60)

    calculator = TrendFactors(precision=4)
    klines = create_test_klines()

    # 测试MA
    print("\n1. 移动平均线测试:")
    for method_name in ['ma5', 'ma10', 'ma20', 'ma60']:
        try:
            method = getattr(calculator, method_name)
            result = method(klines)
            print(f"   ✓ {method_name.upper()}: {result['value']}")
        except Exception as e:
            print(f"   ✗ {method_name.upper()}失败: {e}")

    # 测试EMA
    print("\n2. 指数移动平均线测试:")
    for method_name in ['ema5', 'ema10', 'ema20']:
        try:
            method = getattr(calculator, method_name)
            result = method(klines)
            print(f"   ✓ {method_name.upper()}: {result['value']}")
        except Exception as e:
            print(f"   ✗ {method_name.upper()}失败: {e}")

    # 测试布林带
    print("\n3. 布林带测试:")
    for method_name in ['boll_upper', 'boll_middle', 'boll_lower']:
        try:
            method = getattr(calculator, method_name)
            result = method(klines)
            print(f"   ✓ {method_name.upper()}: {result['value']}")
        except Exception as e:
            print(f"   ✗ {method_name.upper()}失败: {e}")

    # 测试ATR
    print("\n4. ATR测试:")
    try:
        result = calculator.atr14(klines)
        print(f"   ✓ ATR14: {result['value']}")
        print(f"     波动率%: {result['metadata']['volatility_pct']}%")
    except Exception as e:
        print(f"   ✗ ATR14失败: {e}")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("科创50因子工程测试")
    print("="*60)

    # 测试各类因子
    test_momentum_factors()
    test_volume_factors()
    test_trend_factors()

    print("\n" + "="*60)
    print("测试完成!")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
