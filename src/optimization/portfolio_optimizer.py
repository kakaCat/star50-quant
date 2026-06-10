"""
组合优化器
==========

使用cvxpy实现组合优化，集成Alpha信号和风险模型。

优化目标：
    maximize: w^T * alpha - lambda * w^T * Sigma * w

约束条件：
    1. 权重和为1: sum(w) = 1
    2. 做多约束: w >= 0
    3. 个股权重上限: w <= max_weight
    4. 换手率约束: ||w - w_prev|| <= max_turnover
    5. 行业中性（可选）
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import cvxpy as cp


class PortfolioOptimizer:
    """
    组合优化器

    使用凸优化求解最优权重，平衡Alpha收益和风险。
    """

    def __init__(
        self,
        risk_aversion: float = 1.0,
        max_weight: float = 0.05,
        max_turnover: float = 0.3,
        min_weight: float = 0.0,
        alpha_weighted: bool = False
    ):
        """
        初始化优化器

        Args:
            risk_aversion: 风险厌恶系数（lambda），越大越保守
            max_weight: 单个股票最大权重
            max_turnover: 最大换手率（相对于上期权重）
            min_weight: 单个股票最小权重（通常为0）
            alpha_weighted: 是否使用Alpha加权配置
        """
        self.risk_aversion = risk_aversion
        self.max_weight = max_weight
        self.max_turnover = max_turnover
        self.min_weight = min_weight
        self.alpha_weighted = alpha_weighted

    def _get_alpha_weighted_bounds(
        self,
        alpha: np.ndarray,
        n_holdings: int = 20
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        根据Alpha值计算动态权重上下限

        Args:
            alpha: Alpha预测值
            n_holdings: 持仓数量

        Returns:
            (lower_bounds, upper_bounds)
        """
        # 选择Top N个Alpha
        top_n_indices = np.argsort(alpha)[-n_holdings:]

        # 初始化上下限
        lower_bounds = np.zeros(len(alpha))
        upper_bounds = np.zeros(len(alpha))

        # 只对Top N设置权重
        top_alphas = alpha[top_n_indices]

        # 归一化Alpha到权重
        # 使用softmax确保正值
        alpha_normalized = np.exp(top_alphas - top_alphas.max())
        alpha_weights = alpha_normalized / alpha_normalized.sum()

        # 设置上限：基础权重的1.5倍，但不超过max_weight
        for i, idx in enumerate(top_n_indices):
            base_weight = alpha_weights[i]
            upper_bounds[idx] = min(base_weight * 1.5, self.max_weight)
            lower_bounds[idx] = max(base_weight * 0.5, 0.01)  # 至少1%

        return lower_bounds, upper_bounds

    def optimize(
        self,
        alpha: np.ndarray,
        covariance: np.ndarray,
        previous_weights: Optional[np.ndarray] = None,
        stock_universe: Optional[np.ndarray] = None,
        benchmark_weights: Optional[np.ndarray] = None,
        industry_matrix: Optional[np.ndarray] = None,
        industry_neutral: bool = False
    ) -> Dict:
        """
        优化组合权重

        Args:
            alpha: Alpha预测值 [n_stocks]
            covariance: 协方差矩阵 [n_stocks, n_stocks]
            previous_weights: 上期权重 [n_stocks]，用于换手率约束
            stock_universe: 可投资股票掩码 [n_stocks]，True表示可投资
            benchmark_weights: 基准权重 [n_stocks]，用于跟踪误差控制
            industry_matrix: 行业矩阵 [n_stocks, n_industries]
            industry_neutral: 是否行业中性

        Returns:
            结果字典：
                - weights: 最优权重 [n_stocks]
                - expected_return: 预期收益
                - expected_risk: 预期风险
                - turnover: 换手率
                - objective: 目标函数值
                - status: 求解状态
        """
        n_stocks = len(alpha)

        # 定义优化变量
        w = cp.Variable(n_stocks)

        # 目标函数: 最大化 (alpha - lambda * risk)
        expected_return = w @ alpha
        risk = cp.quad_form(w, covariance)
        objective = cp.Maximize(expected_return - self.risk_aversion * risk)

        # 约束条件
        constraints = []

        # 1. 权重和为1
        constraints.append(cp.sum(w) == 1)

        # 2. 权重范围约束
        if self.alpha_weighted:
            # Alpha加权：动态上下限
            lower_bounds, upper_bounds = self._get_alpha_weighted_bounds(alpha)
            for i in range(n_stocks):
                constraints.append(w[i] >= lower_bounds[i])
                constraints.append(w[i] <= upper_bounds[i])
        else:
            # 传统方式：固定上限
            constraints.append(w >= self.min_weight)
            constraints.append(w <= self.max_weight)

        # 3. 可投资股票约束
        if stock_universe is not None:
            for i in range(n_stocks):
                if not stock_universe[i]:
                    constraints.append(w[i] == 0)

        # 4. 换手率约束
        if previous_weights is not None:
            turnover = cp.norm(w - previous_weights, 1)
            constraints.append(turnover <= self.max_turnover)

        # 5. 行业中性约束
        if industry_neutral and industry_matrix is not None:
            # 组合行业暴露 = 基准行业暴露
            if benchmark_weights is not None:
                portfolio_industry = industry_matrix.T @ w
                benchmark_industry = industry_matrix.T @ benchmark_weights
                constraints.append(portfolio_industry == benchmark_industry)

        # 求解优化问题
        problem = cp.Problem(objective, constraints)

        try:
            problem.solve(solver=cp.OSQP, verbose=False)

            if problem.status not in ['optimal', 'optimal_inaccurate']:
                print(f"Warning: 优化未收敛，状态={problem.status}")
                return self._fallback_solution(n_stocks, alpha, previous_weights)

            # 提取结果
            optimal_weights = w.value

            # 计算指标
            opt_return = float(optimal_weights @ alpha)
            opt_risk = float(optimal_weights @ covariance @ optimal_weights)
            opt_turnover = 0.0
            if previous_weights is not None:
                opt_turnover = float(np.sum(np.abs(optimal_weights - previous_weights)))

            return {
                'weights': optimal_weights,
                'expected_return': opt_return,
                'expected_risk': opt_risk,
                'expected_volatility': np.sqrt(opt_risk),
                'turnover': opt_turnover,
                'objective': problem.value,
                'status': problem.status,
                'sharpe_ratio': opt_return / np.sqrt(opt_risk) if opt_risk > 0 else 0.0
            }

        except Exception as e:
            print(f"优化失败: {e}")
            return self._fallback_solution(n_stocks, alpha, previous_weights)

    def _fallback_solution(
        self,
        n_stocks: int,
        alpha: np.ndarray,
        previous_weights: Optional[np.ndarray]
    ) -> Dict:
        """
        优化失败时的备用方案：等权重或保持上期权重

        Args:
            n_stocks: 股票数量
            alpha: Alpha值
            previous_weights: 上期权重

        Returns:
            结果字典
        """
        if previous_weights is not None:
            weights = previous_weights
        else:
            weights = np.ones(n_stocks) / n_stocks

        return {
            'weights': weights,
            'expected_return': float(weights @ alpha),
            'expected_risk': 0.0,
            'expected_volatility': 0.0,
            'tracking_error': 0.0,
            'turnover': 0.0,
            'objective': 0.0,
            'status': 'fallback',
            'sharpe_ratio': 0.0
        }

    def optimize_with_tracking_error(
        self,
        alpha: np.ndarray,
        covariance: np.ndarray,
        benchmark_weights: np.ndarray,
        max_tracking_error: float = 0.05,
        previous_weights: Optional[np.ndarray] = None
    ) -> Dict:
        """
        带跟踪误差约束的优化

        Args:
            alpha: Alpha预测值
            covariance: 协方差矩阵
            benchmark_weights: 基准权重
            max_tracking_error: 最大跟踪误差（年化）
            previous_weights: 上期权重

        Returns:
            结果字典
        """
        n_stocks = len(alpha)
        w = cp.Variable(n_stocks)

        # 目标函数：最大化超额收益
        excess_return = w @ alpha
        objective = cp.Maximize(excess_return)

        # 约束条件
        constraints = []

        # 1. 权重和为1
        constraints.append(cp.sum(w) == 1)

        # 2. 权重范围
        constraints.append(w >= self.min_weight)
        constraints.append(w <= self.max_weight)

        # 3. 跟踪误差约束
        active_weights = w - benchmark_weights
        tracking_variance = cp.quad_form(active_weights, covariance)
        constraints.append(tracking_variance <= max_tracking_error ** 2)

        # 4. 换手率约束
        if previous_weights is not None:
            turnover = cp.norm(w - previous_weights, 1)
            constraints.append(turnover <= self.max_turnover)

        # 求解
        problem = cp.Problem(objective, constraints)

        try:
            # 先尝试OSQP
            try:
                problem.solve(solver=cp.OSQP, verbose=False, eps_abs=1e-4, eps_rel=1e-4)
            except Exception:
                # OSQP失败，尝试SCS
                problem.solve(solver=cp.SCS, verbose=False)

            if problem.status not in ['optimal', 'optimal_inaccurate']:
                return self._fallback_solution(n_stocks, alpha, previous_weights)

            optimal_weights = w.value
            active_weights_val = optimal_weights - benchmark_weights

            return {
                'weights': optimal_weights,
                'expected_return': float(optimal_weights @ alpha),
                'expected_risk': float(optimal_weights @ covariance @ optimal_weights),
                'tracking_error': float(np.sqrt(active_weights_val @ covariance @ active_weights_val)),
                'turnover': float(np.sum(np.abs(optimal_weights - previous_weights))) if previous_weights is not None else 0.0,
                'objective': problem.value,
                'status': problem.status,
                'active_weights': active_weights_val
            }

        except Exception as e:
            print(f"优化失败: {e}")
            return self._fallback_solution(n_stocks, alpha, previous_weights)

    def rebalance(
        self,
        current_weights: np.ndarray,
        target_weights: np.ndarray,
        transaction_cost: float = 0.0015,
        min_trade_size: float = 0.001
    ) -> Dict:
        """
        计算再平衡交易

        Args:
            current_weights: 当前权重
            target_weights: 目标权重
            transaction_cost: 交易成本（双边，0.15%）
            min_trade_size: 最小交易规模

        Returns:
            交易字典：
                - trades: 交易量 [n_stocks]
                - cost: 交易成本
                - turnover: 单边换手率
        """
        trades = target_weights - current_weights

        # 过滤小额交易
        trades[np.abs(trades) < min_trade_size] = 0

        # 计算成本
        turnover = np.sum(np.abs(trades))
        cost = turnover * transaction_cost

        return {
            'trades': trades,
            'cost': cost,
            'turnover': turnover,
            'n_trades': np.sum(trades != 0)
        }

    def compute_portfolio_analytics(
        self,
        weights: np.ndarray,
        alpha: np.ndarray,
        covariance: np.ndarray,
        benchmark_weights: Optional[np.ndarray] = None,
        factor_exposures: Optional[np.ndarray] = None,
        factor_covariance: Optional[np.ndarray] = None,
        specific_risk: Optional[np.ndarray] = None
    ) -> Dict:
        """
        计算组合分析指标

        Args:
            weights: 组合权重
            alpha: Alpha值
            covariance: 协方差矩阵
            benchmark_weights: 基准权重
            factor_exposures: 因子暴露 [n_stocks, n_factors]
            factor_covariance: 因子协方差 [n_factors, n_factors]
            specific_risk: 特质风险 [n_stocks]

        Returns:
            分析指标字典
        """
        analytics = {}

        # 预期收益
        analytics['expected_return'] = float(weights @ alpha)

        # 总风险
        total_variance = weights @ covariance @ weights
        analytics['total_risk'] = float(np.sqrt(total_variance))

        # 风险分解（如果提供因子模型）
        if factor_exposures is not None and factor_covariance is not None and specific_risk is not None:
            # 组合因子暴露
            portfolio_factor_exposure = weights @ factor_exposures

            # 系统性风险
            systematic_variance = portfolio_factor_exposure @ factor_covariance @ portfolio_factor_exposure

            # 特质风险
            specific_variance = weights**2 @ specific_risk**2

            analytics['systematic_risk'] = float(np.sqrt(systematic_variance))
            analytics['specific_risk'] = float(np.sqrt(specific_variance))
            analytics['factor_exposure'] = portfolio_factor_exposure

        # 相对基准指标
        if benchmark_weights is not None:
            active_weights = weights - benchmark_weights
            analytics['active_weights'] = active_weights

            # 跟踪误差
            tracking_variance = active_weights @ covariance @ active_weights
            analytics['tracking_error'] = float(np.sqrt(tracking_variance))

            # 主动风险分解
            if factor_exposures is not None and factor_covariance is not None:
                active_factor_exposure = active_weights @ factor_exposures
                active_systematic_var = active_factor_exposure @ factor_covariance @ active_factor_exposure
                active_specific_var = active_weights**2 @ specific_risk**2

                analytics['active_systematic_risk'] = float(np.sqrt(active_systematic_var))
                analytics['active_specific_risk'] = float(np.sqrt(active_specific_var))

        # 集中度指标
        analytics['herfindahl_index'] = float(np.sum(weights**2))
        analytics['effective_n_stocks'] = float(1.0 / np.sum(weights**2))
        analytics['max_weight'] = float(np.max(weights))
        analytics['n_holdings'] = int(np.sum(weights > 0.001))

        return analytics
