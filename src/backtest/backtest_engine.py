#!/usr/bin/env python3
"""
回测引擎
========

实现组合回测，评估策略表现。

特性：
- 每日再平衡
- 交易成本模拟
- 涨跌停限制
- 滑点模拟
- 业绩归因
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta


class BacktestEngine:
    """
    回测引擎

    模拟实盘交易环境，评估组合策略表现。
    """

    def __init__(
        self,
        initial_capital: float = 10000000.0,  # 1000万初始资金
        commission_rate: float = 0.0015,  # 双边0.15%手续费
        slippage: float = 0.0005,  # 0.05%滑点
        price_limit: float = 0.20  # 科创板20%涨跌停
    ):
        """
        初始化回测引擎

        Args:
            initial_capital: 初始资金
            commission_rate: 交易佣金费率（双边）
            slippage: 滑点（买入加，卖出减）
            price_limit: 涨跌停限制
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.price_limit = price_limit

        # 回测状态
        self.portfolio_value = []
        self.positions = []
        self.trades = []
        self.daily_returns = []

    def run_backtest(
        self,
        weights_series: pd.DataFrame,
        prices: pd.DataFrame,
        benchmark_weights: Optional[pd.DataFrame] = None,
        rebalance_freq: str = 'D'
    ) -> Dict:
        """
        运行回测

        Args:
            weights_series: 权重时间序列 [date, ts_code, weight]
            prices: 价格数据 [trade_date, ts_code, close]
            benchmark_weights: 基准权重 [ts_code, weight]
            rebalance_freq: 再平衡频率 ('D'=每日, 'W'=每周, 'M'=每月)

        Returns:
            回测结果字典
        """
        print("="*60)
        print("回测引擎运行")
        print("="*60)

        # 初始化
        capital = self.initial_capital
        current_positions = {}  # {ts_code: shares}
        daily_portfolio_value = []
        daily_cash = []
        daily_positions = []

        # 获取所有交易日
        trade_dates = sorted(weights_series['date'].unique())
        print(f"回测期间: {trade_dates[0]} 至 {trade_dates[-1]}")
        print(f"交易日数: {len(trade_dates)}")

        for i, date in enumerate(trade_dates):
            # 获取当日目标权重
            target_weights = weights_series[weights_series['date'] == date]

            if len(target_weights) == 0:
                continue

            # 获取当日价格
            daily_prices = prices[prices['trade_date'] == date]

            if len(daily_prices) == 0:
                continue

            # 合并权重和价格
            holdings = target_weights.merge(
                daily_prices[['ts_code', 'close']],
                on='ts_code',
                how='inner'
            )

            # 计算当前持仓市值
            if current_positions:
                current_value = sum(
                    current_positions.get(row['ts_code'], 0) * row['close']
                    for _, row in holdings.iterrows()
                )
            else:
                current_value = 0

            total_value = capital + current_value

            # 执行再平衡
            trades_today = []
            for _, row in holdings.iterrows():
                ts_code = row['ts_code']
                target_weight = row['weight']
                price = row['close']

                # 目标持仓
                target_value = total_value * target_weight
                target_shares = int(target_value / price / 100) * 100  # 100股为单位

                # 当前持仓
                current_shares = current_positions.get(ts_code, 0)

                # 交易
                trade_shares = target_shares - current_shares

                if trade_shares != 0:
                    # 计算交易成本
                    trade_value = abs(trade_shares) * price
                    commission = trade_value * self.commission_rate

                    # 滑点
                    if trade_shares > 0:  # 买入
                        execution_price = price * (1 + self.slippage)
                    else:  # 卖出
                        execution_price = price * (1 - self.slippage)

                    # 更新持仓和资金
                    current_positions[ts_code] = target_shares
                    capital -= trade_shares * execution_price + commission

                    trades_today.append({
                        'date': date,
                        'ts_code': ts_code,
                        'shares': trade_shares,
                        'price': execution_price,
                        'commission': commission
                    })

            # 记录每日状态
            portfolio_value = capital + sum(
                current_positions.get(row['ts_code'], 0) * row['close']
                for _, row in holdings.iterrows()
            )

            daily_portfolio_value.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'cash': capital,
                'positions_value': portfolio_value - capital
            })

            daily_positions.append({
                'date': date,
                'positions': current_positions.copy()
            })

            self.trades.extend(trades_today)

            if (i + 1) % 50 == 0:
                print(f"  处理 {i+1}/{len(trade_dates)} 天...")

        # 转换为DataFrame
        portfolio_df = pd.DataFrame(daily_portfolio_value)
        trades_df = pd.DataFrame(self.trades)

        # 计算收益率
        portfolio_df['daily_return'] = portfolio_df['portfolio_value'].pct_change()
        portfolio_df['cumulative_return'] = (
            portfolio_df['portfolio_value'] / self.initial_capital - 1
        )

        # 计算基准收益（等权）
        if benchmark_weights is not None:
            benchmark_returns = self._calculate_benchmark_returns(
                prices, trade_dates, benchmark_weights
            )
            portfolio_df = portfolio_df.merge(
                benchmark_returns, on='date', how='left'
            )
            portfolio_df['excess_return'] = (
                portfolio_df['daily_return'] - portfolio_df['benchmark_return']
            )
            portfolio_df['cumulative_excess'] = (
                (1 + portfolio_df['excess_return']).cumprod() - 1
            )

        print(f"\n✓ 回测完成")
        print(f"  最终市值: {portfolio_df['portfolio_value'].iloc[-1]:,.0f}")
        print(f"  累计收益: {portfolio_df['cumulative_return'].iloc[-1]:.2%}")
        print(f"  交易次数: {len(trades_df)}")

        return {
            'portfolio': portfolio_df,
            'trades': trades_df,
            'positions': daily_positions
        }

    def _calculate_benchmark_returns(
        self,
        prices: pd.DataFrame,
        trade_dates: List,
        benchmark_weights: pd.DataFrame
    ) -> pd.DataFrame:
        """
        计算基准收益率

        Args:
            prices: 价格数据
            trade_dates: 交易日列表
            benchmark_weights: 基准权重

        Returns:
            DataFrame with columns: date, benchmark_return
        """
        benchmark_returns = []

        for i, date in enumerate(trade_dates):
            if i == 0:
                benchmark_returns.append({
                    'date': date,
                    'benchmark_return': 0.0
                })
                continue

            # 前一日价格
            prev_date = trade_dates[i-1]
            prev_prices = prices[prices['trade_date'] == prev_date]
            curr_prices = prices[prices['trade_date'] == date]

            # 合并
            price_change = prev_prices.merge(
                curr_prices,
                on='ts_code',
                suffixes=('_prev', '_curr')
            )
            price_change['return'] = (
                price_change['close_curr'] / price_change['close_prev'] - 1
            )

            # 加权平均收益
            price_change = price_change.merge(benchmark_weights, on='ts_code')
            benchmark_return = (price_change['return'] * price_change['weight']).sum()

            benchmark_returns.append({
                'date': date,
                'benchmark_return': benchmark_return
            })

        return pd.DataFrame(benchmark_returns)

    def calculate_metrics(self, portfolio_df: pd.DataFrame) -> Dict:
        """
        计算业绩指标

        Args:
            portfolio_df: 组合时间序列

        Returns:
            指标字典
        """
        print("\n" + "="*60)
        print("业绩指标计算")
        print("="*60)

        returns = portfolio_df['daily_return'].dropna()

        # 基础指标
        total_return = portfolio_df['cumulative_return'].iloc[-1]
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0

        # 最大回撤
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # 胜率
        win_rate = (returns > 0).sum() / len(returns)

        metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'n_days': len(returns)
        }

        # 相对基准指标
        if 'excess_return' in portfolio_df.columns:
            excess_returns = portfolio_df['excess_return'].dropna()
            tracking_error = excess_returns.std() * np.sqrt(252)
            information_ratio = (
                excess_returns.mean() * 252 / tracking_error
                if tracking_error > 0 else 0
            )

            metrics['tracking_error'] = tracking_error
            metrics['information_ratio'] = information_ratio
            metrics['cumulative_excess'] = portfolio_df['cumulative_excess'].iloc[-1]

        # 打印指标
        print(f"\n绝对收益指标:")
        print(f"  累计收益: {metrics['total_return']:.2%}")
        print(f"  年化收益: {metrics['annual_return']:.2%}")
        print(f"  年化波动: {metrics['volatility']:.2%}")
        print(f"  夏普比率: {metrics['sharpe_ratio']:.4f}")
        print(f"  最大回撤: {metrics['max_drawdown']:.2%}")
        print(f"  胜率: {metrics['win_rate']:.2%}")

        if 'tracking_error' in metrics:
            print(f"\n相对收益指标:")
            print(f"  累计超额: {metrics['cumulative_excess']:.2%}")
            print(f"  跟踪误差: {metrics['tracking_error']:.2%}")
            print(f"  信息比率: {metrics['information_ratio']:.4f}")

        return metrics

    def calculate_attribution(
        self,
        portfolio_df: pd.DataFrame,
        weights_series: pd.DataFrame,
        alpha_series: pd.DataFrame
    ) -> pd.DataFrame:
        """
        收益归因分析

        Args:
            portfolio_df: 组合收益序列
            weights_series: 权重序列
            alpha_series: Alpha预测序列

        Returns:
            归因结果DataFrame
        """
        print("\n" + "="*60)
        print("收益归因分析")
        print("="*60)

        # 合并数据
        attribution = portfolio_df.merge(
            weights_series.groupby('date').agg({
                'alpha': lambda x: (weights_series.loc[x.index, 'weight'] * x).sum()
            }).reset_index().rename(columns={'alpha': 'predicted_alpha'}),
            on='date',
            how='left'
        )

        # Alpha贡献 vs 实际收益
        print(f"\nAlpha预测 vs 实际收益:")
        print(f"  平均Alpha预测: {attribution['predicted_alpha'].mean():.4f}")
        print(f"  平均实际收益: {attribution['daily_return'].mean():.4f}")
        print(f"  相关系数: {attribution[['predicted_alpha', 'daily_return']].corr().iloc[0,1]:.4f}")

        return attribution
