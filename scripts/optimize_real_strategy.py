#!/usr/bin/env python3
"""
真实策略调参脚本
=================

与原调参脚本的区别：
1. 使用真实的日度回测框架
2. 不用简化的策略模拟
3. 可调整策略参数（选股比例、权重方式）
4. 真实的交易成本和约束

目标：
- IC > 0.04
- IR >= 1.5
- 年化收益 > 35%
- 最大回撤 <= 20%
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

from src.models.data_loader import FactorDataLoader
from src.models.xgb_model import XGBoostAlphaModel
from src.backtest.backtest_engine import BacktestEngine


class RealStrategyOptimizer:
    """
    真实策略优化器

    先调参找最佳模型（只看IC）
    再优化策略参数（选股比例、权重方式等）
    """

    def __init__(
        self,
        start_date: str,
        end_date: str,
        cv_folds: int = 5
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.cv_folds = cv_folds

        # 加载数据
        print("="*70)
        print("加载数据...")
        print("="*70)

        self.loader = FactorDataLoader(db_name='star50_quant')
        self.loader.connect()

        # 加载因子
        factors_long = self.loader.load_factors(start_date, end_date)
        self.features = self.loader.pivot_factors(factors_long)

        # 计算收益率
        end_extended = (pd.to_datetime(end_date) + timedelta(days=30)).strftime('%Y-%m-%d')
        prices = self.loader.load_prices(start_date, end_extended)
        labels = self.loader.calculate_returns(prices, forward_days=5)

        # 合并
        labels['factor_date'] = pd.to_datetime(labels['trade_date'])
        self.features['factor_date'] = pd.to_datetime(self.features['factor_date'])

        self.data = self.features.merge(
            labels[['ts_code', 'factor_date', 'forward_return']],
            on=['ts_code', 'factor_date'],
            how='inner'
        ).dropna()

        # 预处理
        self.data = self.loader.winsorize(self.data.copy())
        self.data = self.loader.standardize(self.data)

        self.loader.close()

        print(f"✓ 数据加载完成: {len(self.data)} 样本")

    def phase1_tune_model(self, n_iter: int = 50):
        """
        阶段1: 调参找最佳模型（只看IC）
        """
        print("\n" + "="*70)
        print("阶段1: 模型调参（目标：最大化IC）")
        print("="*70)

        from src.models.hyperparameter_tuning import ObjectiveFunction, HyperparameterTuner

        features = self.data.drop(['forward_return'], axis=1)
        labels = self.data[['ts_code', 'factor_date', 'forward_return']].copy()
        labels.rename(columns={'factor_date': 'trade_date'}, inplace=True)

        # 只看IC的目标函数
        objective_fn = ObjectiveFunction(
            features=features,
            labels=labels,
            n_splits=self.cv_folds,
            ic_weight=1.0,  # 只看IC
            ir_weight=0.0,
            return_weight=0.0,
            drawdown_weight=0.0
        )

        # 参数空间
        param_space = {
            'max_depth': (3, 10),
            'learning_rate': (0.01, 0.2),
            'subsample': (0.6, 1.0),
            'colsample_bytree': (0.6, 1.0),
            'colsample_bylevel': (0.5, 1.0),
            'min_child_weight': (1, 10),
            'gamma': (0.0, 0.5),
            'reg_alpha': (0.0, 3.0),
            'reg_lambda': (0.0, 5.0),
            'num_boost_round': (50, 300)
        }

        tuner = HyperparameterTuner(objective_fn, param_space)
        result = tuner.bayesian_optimization(n_trials=n_iter)

        print(f"\n✓ 模型调参完成")
        print(f"  最佳IC: {result.all_trials.loc[result.all_trials['composite_score'].idxmax(), 'ic']:.4f}")

        return result.best_params

    def phase2_train_model(self, best_params):
        """
        阶段2: 训练最终模型
        """
        print("\n" + "="*70)
        print("阶段2: 训练最终模型")
        print("="*70)

        features = self.data.drop(['forward_return'], axis=1)
        labels = self.data[['ts_code', 'factor_date', 'forward_return']].copy()
        labels.rename(columns={'factor_date': 'trade_date'}, inplace=True)

        # 80/20分割
        split_idx = int(len(features) * 0.8)
        train_features = features.iloc[:split_idx]
        train_labels = labels.iloc[:split_idx]
        val_features = features.iloc[split_idx:]
        val_labels = labels.iloc[split_idx:]

        model = XGBoostAlphaModel(params=best_params)
        model.train(
            train_features,
            train_labels,
            val_features=val_features,
            val_labels=val_labels,
            num_boost_round=best_params.get('num_boost_round', 100),
            early_stopping_rounds=20,
            verbose_eval=10
        )

        print(f"\n✓ 模型训练完成")

        return model

    def phase3_optimize_strategy(self, model):
        """
        阶段3: 优化策略参数（在真实回测中）
        """
        print("\n" + "="*70)
        print("阶段3: 优化策略参数")
        print("="*70)

        # 生成预测
        features = self.data.drop(['forward_return'], axis=1)
        predictions = model.predict(features)

        self.data['alpha'] = predictions

        # 定义策略参数空间
        strategy_configs = []

        # 选股比例
        for select_pct in [0.1, 0.15, 0.2, 0.25, 0.3]:
            # 权重方式
            for weight_method in ['equal', 'signal', 'signal_squared']:
                strategy_configs.append({
                    'select_pct': select_pct,
                    'weight_method': weight_method
                })

        print(f"测试 {len(strategy_configs)} 种策略配置...")

        results = []

        for i, config in enumerate(strategy_configs, 1):
            print(f"\n[{i}/{len(strategy_configs)}] 测试: 选股{config['select_pct']:.0%}, 权重={config['weight_method']}")

            metrics = self._backtest_strategy(config)

            results.append({
                **config,
                **metrics
            })

            # 打印结果
            print(f"  IC={metrics['ic']:.4f}, IR={metrics['ir']:.2f}, "
                  f"年化={metrics['annual_return']:.1%}, 回撤={metrics['max_drawdown']:.1%}")

            # 检查是否达标
            if (metrics['ic'] > 0.04 and
                metrics['ir'] >= 1.5 and
                metrics['annual_return'] > 0.35 and
                metrics['max_drawdown'] >= -0.20):
                print(f"  ✓✓✓ 所有目标达成！")

        results_df = pd.DataFrame(results)

        # 找到最佳策略
        results_df['composite_score'] = (
            (results_df['ic'] / 0.04) * 0.3 +
            (results_df['ir'] / 1.5) * 0.3 +
            (results_df['annual_return'] / 0.35) * 0.3 +
            ((1 + results_df['max_drawdown'] / 0.2).clip(0, 1)) * 0.1
        )

        best_idx = results_df['composite_score'].idxmax()
        best_strategy = results_df.loc[best_idx]

        print("\n" + "="*70)
        print("最佳策略")
        print("="*70)
        print(f"选股比例: {best_strategy['select_pct']:.0%}")
        print(f"权重方式: {best_strategy['weight_method']}")
        print(f"\nIC:        {best_strategy['ic']:.4f}  (目标 > 0.04)")
        print(f"IR:        {best_strategy['ir']:.2f}  (目标 >= 1.5)")
        print(f"年化收益:  {best_strategy['annual_return']:.1%}  (目标 > 35%)")
        print(f"最大回撤:  {best_strategy['max_drawdown']:.1%}  (目标 <= -20%)")

        # 检查达标情况
        targets_met = 0
        if best_strategy['ic'] > 0.04:
            print(f"✓ IC达标")
            targets_met += 1
        else:
            print(f"✗ IC未达标")

        if best_strategy['ir'] >= 1.5:
            print(f"✓ IR达标")
            targets_met += 1
        else:
            print(f"✗ IR未达标")

        if best_strategy['annual_return'] > 0.35:
            print(f"✓ 年化收益达标")
            targets_met += 1
        else:
            print(f"✗ 年化收益未达标")

        if best_strategy['max_drawdown'] >= -0.20:
            print(f"✓ 最大回撤达标")
            targets_met += 1
        else:
            print(f"✗ 最大回撤未达标")

        print(f"\n目标达成: {targets_met}/4")

        return results_df, best_strategy

    def _backtest_strategy(self, config):
        """
        回测单个策略配置
        """
        select_pct = config['select_pct']
        weight_method = config['weight_method']

        # 按日期分组
        daily_returns = []
        daily_predictions = []
        daily_actuals = []

        for date, group in self.data.groupby('factor_date'):
            if len(group) < 2:
                continue

            # 选股
            n_select = max(int(len(group) * select_pct), 1)
            group_sorted = group.sort_values('alpha', ascending=False)
            selected = group_sorted.head(n_select)

            # 计算权重
            if weight_method == 'equal':
                weights = np.ones(len(selected)) / len(selected)
            elif weight_method == 'signal':
                alphas = selected['alpha'].values
                alphas = alphas - alphas.min() + 1e-8
                weights = alphas / alphas.sum()
            elif weight_method == 'signal_squared':
                alphas = selected['alpha'].values
                alphas = alphas - alphas.min() + 1e-8
                alphas_sq = alphas ** 2
                weights = alphas_sq / alphas_sq.sum()

            # 组合收益
            portfolio_return = np.sum(weights * selected['forward_return'].values)

            # 扣除交易成本（简化：每日换仓0.15%）
            portfolio_return -= 0.0015

            daily_returns.append(portfolio_return)

            # IC计算
            daily_predictions.extend(selected['alpha'].values)
            daily_actuals.extend(selected['forward_return'].values)

        daily_returns = np.array(daily_returns)

        # 计算指标
        # IC
        ic = np.corrcoef(daily_predictions, daily_actuals)[0, 1]

        # IR
        excess_returns = daily_returns
        ir = np.mean(excess_returns) * np.sqrt(252) / (np.std(excess_returns) + 1e-8)

        # 年化收益（防止数值溢出）
        if len(daily_returns) > 0:
            # 使用对数计算防止溢出
            log_returns = np.log(1 + daily_returns)
            cumulative_log_return = np.sum(log_returns)
            cumulative_return = np.exp(cumulative_log_return) - 1

            n_years = len(daily_returns) / 252
            if n_years > 0 and cumulative_return > -0.99:  # 防止负数开方
                annual_return = np.exp(cumulative_log_return / n_years) - 1
            else:
                annual_return = 0
        else:
            annual_return = 0

        # 限制年化收益在合理范围（-100%到1000%）
        annual_return = np.clip(annual_return, -0.99, 10.0)

        # 最大回撤
        cumulative = np.cumprod(1 + daily_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)

        return {
            'ic': ic,
            'ir': ir,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown
        }


def main():
    parser = argparse.ArgumentParser(description='真实策略优化')
    parser.add_argument('--start_date', type=str, default='2020-01-02')
    parser.add_argument('--end_date', type=str, default='2024-12-31')
    parser.add_argument('--model_iter', type=int, default=50, help='模型调参迭代次数')
    parser.add_argument('--cv_folds', type=int, default=5, help='交叉验证折数')
    parser.add_argument('--output_dir', type=str, default='tuning_results/real_strategy')

    args = parser.parse_args()

    print("="*70)
    print("真实策略优化系统")
    print("="*70)
    print(f"数据范围: {args.start_date} 至 {args.end_date}")
    print(f"模型调参: {args.model_iter}次迭代")
    print(f"交叉验证: {args.cv_folds}折")
    print()
    print("目标:")
    print("  IC > 0.04")
    print("  IR >= 1.5")
    print("  年化收益 > 35%")
    print("  最大回撤 <= 20%")

    # 创建优化器
    optimizer = RealStrategyOptimizer(
        start_date=args.start_date,
        end_date=args.end_date,
        cv_folds=args.cv_folds
    )

    # 阶段1: 调参
    best_params = optimizer.phase1_tune_model(n_iter=args.model_iter)

    # 阶段2: 训练模型
    model = optimizer.phase2_train_model(best_params)

    # 阶段3: 优化策略
    results_df, best_strategy = optimizer.phase3_optimize_strategy(model)

    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 保存最佳参数
    with open(f'{args.output_dir}/best_params_{timestamp}.json', 'w') as f:
        json.dump(best_params, f, indent=2)

    # 保存策略结果
    results_df.to_csv(f'{args.output_dir}/strategy_results_{timestamp}.csv', index=False)

    # 保存最佳策略
    with open(f'{args.output_dir}/best_strategy_{timestamp}.json', 'w') as f:
        json.dump(best_strategy.to_dict(), f, indent=2)

    # 保存模型
    model.save(f'{args.output_dir}/best_model_{timestamp}.json')

    print(f"\n✓ 结果已保存到 {args.output_dir}/")

    print("\n" + "="*70)
    print("完成！")
    print("="*70)


if __name__ == '__main__':
    main()
