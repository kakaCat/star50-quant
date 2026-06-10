#!/usr/bin/env python3
"""
Phase 3 完整回测脚本
===================

端到端流程：
1. 加载数据
2. 初始化组件
3. 逐周生成Alpha预测和组合权重
4. 运行回测
5. 计算评估指标
6. 验收检查
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
    """加载回测数据"""
    print("="*80)
    print("1. 加载数据")
    print("="*80)
    
    # 加载价格数据
    prices_path = 'star50-quant/data/raw/star50_daily_hfq_data_6yrs.parquet'
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
    """计算9个核心因子"""
    print("\n计算核心因子...")
    
    result = []
    
    for ts_code, group in prices.groupby('ts_code'):
        group = group.sort_values('trade_date').copy()
        
        # 9个核心因子
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
    
    feature_cols = [
        'ts_code', 'factor_date',
        'momentum_5', 'momentum_10', 'momentum_20',
        'volatility_10', 'volatility_20',
        'volume_ratio', 'atr_ratio', 'ma_ratio', 'rsi14'
    ]
    
    features = features[feature_cols].dropna()
    print(f"  特征数据: {len(features)} 行")
    
    return features


print("Phase 3脚本已创建（1/2）")


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """计算日收益率"""
    returns_list = []
    
    for ts_code, group in prices.groupby('ts_code'):
        group = group.sort_values('trade_date').copy()
        group['return'] = group['close'].pct_change()
        returns_list.append(group[['trade_date', 'ts_code', 'return']])
    
    returns = pd.concat(returns_list, ignore_index=True)
    return returns


def calculate_benchmark_weights(prices: pd.DataFrame) -> pd.DataFrame:
    """计算基准权重（科创50等权）"""
    stocks = prices['ts_code'].unique()
    n_stocks = len(stocks)
    
    benchmark = pd.DataFrame({
        'ts_code': stocks,
        'weight': 1.0 / n_stocks
    })
    
    return benchmark


def get_rebalance_dates(start_date: str, end_date: str, freq: str = 'W-MON') -> list:
    """生成再平衡日期列表"""
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    return [d.strftime('%Y-%m-%d') for d in dates]


def validate_phase3(metrics: dict):
    """Phase 3验收检查"""
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
    
    print("\n" + "-"*80)
    if all(checks):
        print("✓ Phase 3验收通过！")
    else:
        print("✗ Phase 3验收未通过")
        print(f"通过: {sum(checks)}/{len(checks)} 项")


print("Phase 3脚本已创建（2/3）")


def main():
    """主函数"""
    print("\n" + "="*80)
    print("Phase 3: 完整回测验证")
    print("="*80)
    
    # 加载配置
    config = load_config('star50-quant/configs/phase3_config.yaml')
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
    
    predictor = EnsemblePredictor('star50-quant/' + config['ensemble']['model_dir'])
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
        
        # 风险估计
        date_dt = pd.to_datetime(date)
        lookback_start = (date_dt - pd.Timedelta(days=400)).strftime('%Y-%m-%d')
        
        historical_returns = returns[
            (returns['trade_date'] >= lookback_start) &
            (returns['trade_date'] < date)
        ]
        
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
        
        # Alpha预测
        daily_features = features[features['factor_date'] == date]
        
        if len(daily_features) == 0:
            print(f"  ⚠ 无特征数据，跳过")
            continue
        
        alpha_pred = predictor.predict(daily_features)
        print(f"  ✓ Alpha预测 ({len(alpha_pred)}只股票)")

        # 对齐股票列表：只使用有Alpha预测的股票
        alpha_stocks = set(alpha_pred['ts_code'])
        aligned_benchmark = benchmark_weights[benchmark_weights['ts_code'].isin(alpha_stocks)].copy()
        aligned_benchmark = aligned_benchmark.set_index('ts_code').loc[alpha_pred['ts_code']].reset_index()

        # 重新归一化基准权重
        aligned_benchmark['weight'] = aligned_benchmark['weight'] / aligned_benchmark['weight'].sum()

        # 对齐协方差矩阵（从returns_pivot列中选择对应股票）
        cov_stocks = list(returns_pivot.columns)
        common_stocks = [s for s in alpha_pred['ts_code'] if s in cov_stocks]

        if len(common_stocks) < len(alpha_pred):
            print(f"  ⚠ {len(alpha_pred) - len(common_stocks)} 只股票缺少协方差数据")

        # 过滤到共同股票
        alpha_pred_aligned = alpha_pred[alpha_pred['ts_code'].isin(common_stocks)].copy()
        aligned_benchmark_aligned = aligned_benchmark[aligned_benchmark['ts_code'].isin(common_stocks)].copy()
        aligned_benchmark_aligned['weight'] = aligned_benchmark_aligned['weight'] / aligned_benchmark_aligned['weight'].sum()

        # 对齐协方差矩阵
        stock_indices = [cov_stocks.index(s) for s in alpha_pred_aligned['ts_code']]
        covariance_aligned = covariance[np.ix_(stock_indices, stock_indices)]

        # 对齐previous_weights（如果股票列表变化，重置为None）
        if previous_weights is not None and len(previous_weights) != len(alpha_pred_aligned):
            previous_weights = None

        # 组合优化
        result = optimizer.optimize_with_tracking_error(
            alpha=alpha_pred_aligned['alpha'].values,
            covariance=covariance_aligned,
            benchmark_weights=aligned_benchmark_aligned['weight'].values,
            max_tracking_error=config['optimization']['max_tracking_error'],
            previous_weights=previous_weights
        )
        
        if result['status'] not in ['optimal', 'optimal_inaccurate']:
            print(f"  ⚠ 优化失败: {result['status']}")
            continue
        
        print(f"  ✓ 组合优化 (TE={result['tracking_error']:.2%})")

        # 记录权重（使用对齐后的股票列表）
        for j, ts_code in enumerate(alpha_pred_aligned['ts_code']):
            weights_series.append({
                'date': date,
                'ts_code': ts_code,
                'weight': result['weights'][j],
                'alpha': alpha_pred_aligned['alpha'].values[j]
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
    output_dir = 'star50-quant/' + config['output']['results_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    backtest_results['portfolio'].to_csv(
        os.path.join(output_dir, 'backtest_results.csv'),
        index=False
    )
    print(f"\n✓ 结果已保存到: {output_dir}")
    
    return backtest_results, metrics


if __name__ == '__main__':
    results, metrics = main()

