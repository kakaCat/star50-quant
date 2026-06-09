#!/usr/bin/env python3
"""
组合优化脚本
============

集成Alpha模型和风险模型，优化组合权重。

用法:
    python scripts/optimize_portfolio.py --date 2024-12-31
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.models.data_loader import FactorDataLoader

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    print("Warning: LightGBM not available")

try:
    from src.optimization.portfolio_optimizer import PortfolioOptimizer
    OPTIMIZER_AVAILABLE = True
except ImportError:
    OPTIMIZER_AVAILABLE = False
    print("错误: 组合优化需要cvxpy")
    print("请安装: pip install cvxpy")


def load_alpha_predictions(model_path: str, date: str) -> pd.DataFrame:
    """
    加载Alpha预测

    Args:
        model_path: 模型路径
        date: 预测日期

    Returns:
        DataFrame with columns: ts_code, alpha
    """
    print("="*60)
    print("1. 加载Alpha预测")
    print("="*60)

    if not LGBM_AVAILABLE:
        print("错误: LightGBM未安装")
        sys.exit(1)

    # 加载模型
    model = lgb.Booster(model_file=model_path)
    print(f"  ✓ 加载模型: {model_path}")

    # 加载特征数据
    with FactorDataLoader() as loader:
        # 获取date之前20天的数据用于计算因子
        end_date = pd.to_datetime(date)
        start_date = end_date - timedelta(days=60)

        factors = loader.load_factors(
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=date
        )

    # 转换为宽表格式
    factors_pivot = factors.pivot_table(
        index=['ts_code', 'factor_date'],
        columns='factor_name',
        values='factor_value'
    ).reset_index()

    # 取最新日期的因子 - 转换date字符串为datetime.date
    target_date = pd.to_datetime(date).date()
    latest_factors = factors_pivot[factors_pivot['factor_date'] == target_date].copy()

    if len(latest_factors) == 0:
        print(f"错误: {date}没有因子数据")
        sys.exit(1)

    print(f"  ✓ 加载因子数据: {len(latest_factors)}只股票")

    # 准备特征
    feature_cols = [col for col in latest_factors.columns
                   if col not in ['ts_code', 'factor_date', 'return_5d']]

    X = latest_factors[feature_cols].values

    # 预测
    alpha = model.predict(X)

    result = pd.DataFrame({
        'ts_code': latest_factors['ts_code'].values,
        'alpha': alpha
    })

    print(f"  ✓ Alpha预测完成")
    print(f"    Alpha统计: mean={alpha.mean():.6f}, std={alpha.std():.6f}")
    print(f"    Top 5 Alpha: {sorted(alpha, reverse=True)[:5]}")

    return result


def load_risk_model() -> tuple:
    """
    加载风险模型

    Returns:
        (factor_exposures, factor_cov, specific_risk)
    """
    print("\n" + "="*60)
    print("2. 加载风险模型")
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
        print("请先运行: python scripts/train_risk_model.py")
        sys.exit(1)


def construct_covariance_matrix(
    stock_codes: list,
    factor_exposures: pd.DataFrame,
    factor_cov: pd.DataFrame,
    specific_risk: pd.DataFrame
) -> np.ndarray:
    """
    构建协方差矩阵: Σ = B @ F @ B^T + D

    Args:
        stock_codes: 股票代码列表
        factor_exposures: 因子暴露矩阵
        factor_cov: 因子协方差矩阵
        specific_risk: 特质风险

    Returns:
        协方差矩阵 [n_stocks, n_stocks]
    """
    # 对齐股票顺序
    B = factor_exposures.loc[stock_codes].values
    F = factor_cov.values

    # 特质风险对角矩阵
    specific_risk_dict = dict(zip(specific_risk['ts_code'], specific_risk['specific_variance']))
    D = np.diag([specific_risk_dict.get(code, 0.0001) for code in stock_codes])

    # Σ = B @ F @ B^T + D
    covariance = B @ F @ B.T + D

    return covariance


def optimize_portfolio(
    alpha_df: pd.DataFrame,
    covariance: np.ndarray,
    risk_aversion: float = 1.0,
    max_weight: float = 0.05,
    max_turnover: float = 0.3
) -> dict:
    """
    优化组合

    Args:
        alpha_df: Alpha预测 DataFrame
        covariance: 协方差矩阵
        risk_aversion: 风险厌恶系数
        max_weight: 最大权重
        max_turnover: 最大换手率

    Returns:
        优化结果
    """
    print("\n" + "="*60)
    print("3. 优化组合权重")
    print("="*60)

    optimizer = PortfolioOptimizer(
        risk_aversion=risk_aversion,
        max_weight=max_weight,
        max_turnover=max_turnover
    )

    # 优化
    result = optimizer.optimize(
        alpha=alpha_df['alpha'].values,
        covariance=covariance
    )

    print(f"\n优化结果:")
    print(f"  状态: {result['status']}")
    print(f"  预期收益: {result['expected_return']:.4f}")
    print(f"  预期风险: {result['expected_volatility']:.4f}")
    print(f"  夏普比率: {result['sharpe_ratio']:.4f}")
    print(f"  换手率: {result['turnover']:.4f}")

    return result


def analyze_portfolio(
    weights: np.ndarray,
    stock_codes: list,
    alpha: np.ndarray,
    covariance: np.ndarray,
    factor_exposures: pd.DataFrame,
    factor_cov: pd.DataFrame,
    specific_risk: pd.DataFrame
) -> None:
    """
    分析组合特征

    Args:
        weights: 组合权重
        stock_codes: 股票代码
        alpha: Alpha值
        covariance: 协方差矩阵
        factor_exposures: 因子暴露
        factor_cov: 因子协方差
        specific_risk: 特质风险
    """
    print("\n" + "="*60)
    print("4. 组合分析")
    print("="*60)

    optimizer = PortfolioOptimizer()

    # 准备数据
    B = factor_exposures.loc[stock_codes].values
    F = factor_cov.values
    specific_risk_dict = dict(zip(specific_risk['ts_code'], specific_risk['specific_volatility']))
    D = np.array([specific_risk_dict.get(code, 0.01) for code in stock_codes])

    analytics = optimizer.compute_portfolio_analytics(
        weights=weights,
        alpha=alpha,
        covariance=covariance,
        factor_exposures=B,
        factor_covariance=F,
        specific_risk=D
    )

    print(f"\n风险分解:")
    print(f"  总风险: {analytics['total_risk']:.4f}")
    print(f"  系统性风险: {analytics['systematic_risk']:.4f} ({analytics['systematic_risk']/analytics['total_risk']*100:.2f}%)")
    print(f"  特质风险: {analytics['specific_risk']:.4f} ({analytics['specific_risk']/analytics['total_risk']*100:.2f}%)")

    print(f"\n组合集中度:")
    print(f"  赫芬达尔指数: {analytics['herfindahl_index']:.4f}")
    print(f"  有效股票数: {analytics['effective_n_stocks']:.2f}")
    print(f"  最大权重: {analytics['max_weight']:.4f}")
    print(f"  持仓数量: {analytics['n_holdings']}")

    # Top 10持仓
    portfolio_df = pd.DataFrame({
        'ts_code': stock_codes,
        'weight': weights,
        'alpha': alpha
    }).sort_values('weight', ascending=False)

    print(f"\nTop 10 持仓:")
    print(portfolio_df.head(10).to_string(index=False))


def save_results(
    weights: np.ndarray,
    stock_codes: list,
    alpha: np.ndarray,
    date: str,
    result: dict
) -> None:
    """
    保存优化结果

    Args:
        weights: 组合权重
        stock_codes: 股票代码
        alpha: Alpha值
        date: 日期
        result: 优化结果
    """
    print("\n" + "="*60)
    print("5. 保存结果")
    print("="*60)

    os.makedirs('results/portfolios', exist_ok=True)

    # 保存持仓
    portfolio_df = pd.DataFrame({
        'ts_code': stock_codes,
        'weight': weights,
        'alpha': alpha,
        'date': date
    }).sort_values('weight', ascending=False)

    output_file = f'results/portfolios/portfolio_{date}.csv'
    portfolio_df.to_csv(output_file, index=False)
    print(f"  ✓ 持仓保存至: {output_file}")

    # 保存优化指标
    metrics_df = pd.DataFrame([{
        'date': date,
        'expected_return': result['expected_return'],
        'expected_risk': result['expected_volatility'],
        'sharpe_ratio': result['sharpe_ratio'],
        'turnover': result['turnover'],
        'status': result['status']
    }])

    metrics_file = 'results/portfolios/optimization_metrics.csv'
    if os.path.exists(metrics_file):
        existing = pd.read_csv(metrics_file)
        metrics_df = pd.concat([existing, metrics_df], ignore_index=True)

    metrics_df.to_csv(metrics_file, index=False)
    print(f"  ✓ 指标保存至: {metrics_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='组合优化')
    parser.add_argument('--date', type=str, default='2024-12-31',
                       help='优化日期')
    parser.add_argument('--model', type=str, default='models/lgbm_alpha.txt',
                       help='Alpha模型路径')
    parser.add_argument('--risk_aversion', type=float, default=1.0,
                       help='风险厌恶系数')
    parser.add_argument('--max_weight', type=float, default=0.05,
                       help='最大个股权重')
    parser.add_argument('--max_turnover', type=float, default=0.3,
                       help='最大换手率')

    args = parser.parse_args()

    if not OPTIMIZER_AVAILABLE:
        print("错误: cvxpy未安装，无法进行组合优化")
        print("请安装: pip install cvxpy")
        sys.exit(1)

    print("="*60)
    print("科创50指数增强 - 组合优化")
    print("="*60)
    print(f"优化日期: {args.date}")
    print(f"风险厌恶: {args.risk_aversion}")
    print(f"最大权重: {args.max_weight}")
    print(f"最大换手: {args.max_turnover}")
    print("="*60)

    # 1. 加载Alpha预测
    alpha_df = load_alpha_predictions(args.model, args.date)

    # 2. 加载风险模型
    factor_exposures, factor_cov, specific_risk = load_risk_model()

    # 3. 构建协方差矩阵
    stock_codes = alpha_df['ts_code'].tolist()
    covariance = construct_covariance_matrix(
        stock_codes, factor_exposures, factor_cov, specific_risk
    )

    # 4. 优化组合
    result = optimize_portfolio(
        alpha_df, covariance,
        risk_aversion=args.risk_aversion,
        max_weight=args.max_weight,
        max_turnover=args.max_turnover
    )

    # 5. 分析组合
    analyze_portfolio(
        result['weights'], stock_codes, alpha_df['alpha'].values,
        covariance, factor_exposures, factor_cov, specific_risk
    )

    # 6. 保存结果
    save_results(
        result['weights'], stock_codes, alpha_df['alpha'].values,
        args.date, result
    )

    print("\n" + "="*60)
    print("✓ 组合优化完成!")
    print("="*60)


if __name__ == '__main__':
    main()
