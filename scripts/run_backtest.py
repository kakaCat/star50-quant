#!/usr/bin/env python3
"""
完整策略回测脚本
================

整合Alpha预测、风险模型、组合优化和回测。

用法:
    python scripts/run_backtest.py --start 2024-01-01 --end 2024-12-31
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.models.data_loader import FactorDataLoader
from src.backtest.backtest_engine import BacktestEngine

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    from src.optimization.portfolio_optimizer import PortfolioOptimizer
    OPTIMIZER_AVAILABLE = True
except ImportError:
    OPTIMIZER_AVAILABLE = False


def load_data(start_date: str, end_date: str):
    """加载回测所需数据"""
    print("="*60)
    print("1. 加载数据")
    print("="*60)

    with FactorDataLoader() as loader:
        # 加载因子数据
        factors = loader.load_factors(start_date, end_date)

        # 加载价格数据
        prices = loader.load_prices(start_date, end_date)

    print(f"  ✓ 因子数据: {len(factors)} 条")
    print(f"  ✓ 价格数据: {len(prices)} 条")

    # 转换因子为宽表
    factors_pivot = factors.pivot_table(
        index=['ts_code', 'factor_date'],
        columns='factor_name',
        values='factor_value'
    ).reset_index()

    return factors_pivot, prices


def generate_signals(factors_pivot: pd.DataFrame, model_path: str):
    """生成Alpha信号"""
    print("\n" + "="*60)
    print("2. 生成Alpha信号")
    print("="*60)

    if not LGBM_AVAILABLE:
        print("错误: LightGBM未安装")
        sys.exit(1)

    # 加载模型
    model = lgb.Booster(model_file=model_path)
    print(f"  ✓ 加载模型: {model_path}")

    # 准备特征
    feature_cols = [col for col in factors_pivot.columns
                   if col not in ['ts_code', 'factor_date', 'return_5d']]

    # 逐日预测
    all_predictions = []

    for date in factors_pivot['factor_date'].unique():
        daily_factors = factors_pivot[factors_pivot['factor_date'] == date].copy()

        if len(daily_factors) == 0:
            continue

        X = daily_factors[feature_cols].values
        alpha = model.predict(X)

        predictions = pd.DataFrame({
            'date': date,
            'ts_code': daily_factors['ts_code'].values,
            'alpha': alpha
        })

        all_predictions.append(predictions)

    alpha_series = pd.concat(all_predictions, ignore_index=True)

    print(f"  ✓ 生成信号: {len(alpha_series)} 条")
    print(f"  ✓ 覆盖日期: {alpha_series['date'].nunique()} 天")

    return alpha_series


def load_risk_model():
    """加载风险模型"""
    print("\n" + "="*60)
    print("3. 加载风险模型")
    print("="*60)

    try:
        factor_exposures = pd.read_csv('models/risk/factor_exposures.csv', index_col=0)
        factor_cov = pd.read_csv('models/risk/factor_covariance.csv', index_col=0)
        specific_risk = pd.read_csv('models/risk/specific_risk.csv')

        print(f"  ✓ 因子暴露: {factor_exposures.shape}")
        print(f"  ✓ 因子协方差: {factor_cov.shape}")
        print(f"  ✓ 特质风险: {len(specific_risk)}")

        return factor_exposures, factor_cov, specific_risk

    except FileNotFoundError as e:
        print(f"错误: 风险模型文件未找到: {e}")
        sys.exit(1)


def calculate_market_timing(prices: pd.DataFrame, window: int = 20):
    """
    计算市场择时信号

    使用科创50等权指数的MA趋势判断市场状态

    Args:
        prices: 价格数据
        window: 移动平均窗口

    Returns:
        DataFrame with columns: date, market_signal (1=牛市, 0=熊市)
    """
    # 计算等权指数
    index_daily = prices.groupby('trade_date')['close'].mean()

    # 计算移动平均
    index_ma = index_daily.rolling(window=window).mean()

    # 生成信号: 价格 > MA20 = 牛市
    market_signal = (index_daily > index_ma).astype(int)

    return pd.DataFrame({
        'date': market_signal.index,
        'market_signal': market_signal.values,
        'index_close': index_daily.values,
        'index_ma': index_ma.values
    })


def optimize_portfolios(
    alpha_series: pd.DataFrame,
    factor_exposures: pd.DataFrame,
    factor_cov: pd.DataFrame,
    specific_risk: pd.DataFrame,
    risk_aversion: float,
    max_weight: float,
    prices: pd.DataFrame = None,
    rebalance_days: int = 5,
    market_timing: bool = False,
    alpha_weighted: bool = False
):
    """优化组合权重"""
    print("\n" + "="*60)
    print("4. 优化组合权重")
    print("="*60)

    if not OPTIMIZER_AVAILABLE:
        print("错误: cvxpy未安装")
        sys.exit(1)

    # 计算市场择时信号
    market_signals = None
    if market_timing and prices is not None:
        print("  ✓ 启用市场择时模块")
        market_signals = calculate_market_timing(prices, window=20)
        bull_days = (market_signals['market_signal'] == 1).sum()
        bear_days = (market_signals['market_signal'] == 0).sum()
        print(f"    牛市天数: {bull_days}, 熊市天数: {bear_days}")

    all_weights = []
    all_dates = sorted(alpha_series['date'].unique())

    print(f"  ✓ 调仓频率: 每{rebalance_days}天")
    print(f"  ✓ 预期调仓次数: ~{len(all_dates) // rebalance_days}")

    for i, date in enumerate(all_dates):
        # 降低调仓频率：只在指定的交易日调仓
        if i % rebalance_days != 0:
            # 复用上一期权重
            if all_weights:
                last_weights = all_weights[-1].copy()
                last_weights['date'] = date
                all_weights.append(last_weights)
            continue

        daily_alpha = alpha_series[alpha_series['date'] == date].copy()
        stock_codes = daily_alpha['ts_code'].tolist()

        # 构建协方差矩阵
        B = factor_exposures.loc[stock_codes].values
        F = factor_cov.values
        specific_risk_dict = dict(zip(specific_risk['ts_code'], specific_risk['specific_variance']))
        D = np.diag([specific_risk_dict.get(code, 0.0001) for code in stock_codes])
        covariance = B @ F @ B.T + D

        # 动态调整风险厌恶系数（市场择时）
        current_risk_aversion = risk_aversion
        if market_timing and market_signals is not None:
            market_state = market_signals[market_signals['date'] == date]
            if len(market_state) > 0:
                is_bull = market_state['market_signal'].iloc[0] == 1
                # 牛市降低风险厌恶，熊市提高
                current_risk_aversion = risk_aversion * 0.6 if is_bull else risk_aversion * 1.5

        optimizer = PortfolioOptimizer(
            risk_aversion=current_risk_aversion,
            max_weight=max_weight,
            max_turnover=1.0,
            alpha_weighted=alpha_weighted
        )

        # 优化
        result = optimizer.optimize(
            alpha=daily_alpha['alpha'].values,
            covariance=covariance
        )

        if result['status'] in ['optimal', 'optimal_inaccurate']:
            weights_df = pd.DataFrame({
                'date': date,
                'ts_code': stock_codes,
                'weight': result['weights']
            })
            all_weights.append(weights_df)

    weights_series = pd.concat(all_weights, ignore_index=True)

    print(f"  ✓ 优化完成: {len(weights_series['date'].unique())} 天")
    print(f"  ✓ 实际调仓次数: {len(all_dates) // rebalance_days}")

    return weights_series


def run_backtest_engine(
    weights_series: pd.DataFrame,
    prices: pd.DataFrame,
    initial_capital: float
):
    """运行回测"""
    print("\n" + "="*60)
    print("5. 运行回测")
    print("="*60)

    engine = BacktestEngine(initial_capital=initial_capital)

    # 等权基准
    stock_codes = weights_series['ts_code'].unique()
    benchmark_weights = pd.DataFrame({
        'ts_code': stock_codes,
        'weight': 1.0 / len(stock_codes)
    })

    results = engine.run_backtest(
        weights_series=weights_series,
        prices=prices,
        benchmark_weights=benchmark_weights
    )

    # 计算指标
    metrics = engine.calculate_metrics(results['portfolio'])

    return results, metrics


def save_results(results: dict, metrics: dict, output_dir: str):
    """保存回测结果"""
    print("\n" + "="*60)
    print("6. 保存结果")
    print("="*60)

    os.makedirs(output_dir, exist_ok=True)

    # 保存净值曲线
    results['portfolio'].to_csv(f'{output_dir}/portfolio_value.csv', index=False)
    print(f"  ✓ 净值曲线: {output_dir}/portfolio_value.csv")

    # 保存交易记录
    results['trades'].to_csv(f'{output_dir}/trades.csv', index=False)
    print(f"  ✓ 交易记录: {output_dir}/trades.csv")

    # 保存指标
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(f'{output_dir}/metrics.csv', index=False)
    print(f"  ✓ 业绩指标: {output_dir}/metrics.csv")


def plot_results(results: dict, output_dir: str):
    """绘制回测结果"""
    print("\n" + "="*60)
    print("7. 绘制图表")
    print("="*60)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("Warning: matplotlib未安装，跳过绘图")
        return

    portfolio_df = results['portfolio']

    # 1. 净值曲线
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # 累计收益
    axes[0].plot(portfolio_df['date'],
                portfolio_df['cumulative_return'] * 100,
                label='策略', linewidth=2)

    if 'cumulative_excess' in portfolio_df.columns:
        axes[0].plot(portfolio_df['date'],
                    portfolio_df['cumulative_excess'] * 100,
                    label='超额收益', linewidth=2, linestyle='--')

    axes[0].set_ylabel('累计收益 (%)')
    axes[0].set_title('策略净值曲线')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 每日收益
    axes[1].bar(portfolio_df['date'],
               portfolio_df['daily_return'] * 100,
               alpha=0.6, width=1)
    axes[1].set_ylabel('日收益率 (%)')
    axes[1].set_xlabel('日期')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance.png', dpi=150)
    print(f"  ✓ 业绩图表: {output_dir}/performance.png")

    plt.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='完整策略回测')
    parser.add_argument('--start', type=str, default='2024-01-01',
                       help='回测开始日期')
    parser.add_argument('--end', type=str, default='2024-12-31',
                       help='回测结束日期')
    parser.add_argument('--model', type=str, default='models/lgbm_alpha.txt',
                       help='Alpha模型路径')
    parser.add_argument('--capital', type=float, default=10000000.0,
                       help='初始资金')
    parser.add_argument('--risk_aversion', type=float, default=1.0,
                       help='风险厌恶系数')
    parser.add_argument('--max_weight', type=float, default=0.05,
                       help='最大个股权重')
    parser.add_argument('--rebalance_days', type=int, default=5,
                       help='调仓频率（天）')
    parser.add_argument('--market_timing', action='store_true',
                       help='启用市场择时')
    parser.add_argument('--alpha_weighted', action='store_true',
                       help='启用Alpha加权配置')
    parser.add_argument('--output', type=str, default='results/backtest',
                       help='输出目录')

    args = parser.parse_args()

    print("="*60)
    print("科创50指数增强 - 完整策略回测")
    print("="*60)
    print(f"回测区间: {args.start} 至 {args.end}")
    print(f"初始资金: {args.capital:,.0f}")
    print(f"风险厌恶: {args.risk_aversion}")
    print(f"最大权重: {args.max_weight}")
    print(f"调仓频率: 每{args.rebalance_days}天")
    print(f"市场择时: {'启用' if args.market_timing else '禁用'}")
    print(f"Alpha加权: {'启用' if args.alpha_weighted else '禁用'}")
    print("="*60)

    # 1. 加载数据
    factors_pivot, prices = load_data(args.start, args.end)

    # 2. 生成Alpha信号
    alpha_series = generate_signals(factors_pivot, args.model)

    # 3. 加载风险模型
    factor_exposures, factor_cov, specific_risk = load_risk_model()

    # 4. 优化组合
    weights_series = optimize_portfolios(
        alpha_series, factor_exposures, factor_cov, specific_risk,
        args.risk_aversion, args.max_weight,
        prices=prices,
        rebalance_days=args.rebalance_days,
        market_timing=args.market_timing,
        alpha_weighted=args.alpha_weighted
    )

    # 5. 运行回测
    results, metrics = run_backtest_engine(
        weights_series, prices, args.capital
    )

    # 6. 保存结果
    save_results(results, metrics, args.output)

    # 7. 绘制图表
    plot_results(results, args.output)

    print("\n" + "="*60)
    print("✓ 完整回测完成!")
    print("="*60)


if __name__ == '__main__':
    main()
