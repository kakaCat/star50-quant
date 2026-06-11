#!/usr/bin/env python3
"""
XGBoost调参和回测完整流程
========================

一个脚本完成：数据加载 → 调参 → 训练 → 完整回测 → 结果报告
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import spearmanr
import json
from datetime import datetime
from pathlib import Path

# 导入数据加载器
from src.data.parquet_loader import ParquetDataLoader


class XGBoostTuner:
    """XGBoost超参数调优器"""

    def __init__(self, X, y, n_splits=5):
        self.X = X
        self.y = y
        self.n_splits = n_splits

    def evaluate_params(self, params):
        """评估参数组合"""
        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        ic_scores = []

        for train_idx, val_idx in tscv.split(self.X):
            X_train, X_val = self.X[train_idx], self.X[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]

            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)

            xgb_params = {k: v for k, v in params.items() if k != 'num_boost_round'}
            xgb_params.update({
                'objective': 'reg:squarederror',
                'seed': 42,
                'verbosity': 0
            })

            model = xgb.train(
                xgb_params,
                dtrain,
                num_boost_round=params.get('num_boost_round', 100),
                evals=[(dval, 'valid')],
                early_stopping_rounds=10,
                verbose_eval=False
            )

            y_pred = model.predict(dval)
            ic = np.corrcoef(y_pred, y_val)[0, 1]
            if not np.isnan(ic):
                ic_scores.append(ic)

        return np.mean(ic_scores) if ic_scores else 0.0

    def random_search(self, param_space, n_iter=20):
        """随机搜索"""
        print(f"\n随机搜索 (n_iter={n_iter})...")

        best_score = -np.inf
        best_params = None

        for i in range(n_iter):
            # 随机采样
            params = {}
            for key, value in param_space.items():
                if isinstance(value, tuple):
                    if isinstance(value[0], int):
                        params[key] = np.random.randint(value[0], value[1] + 1)
                    else:
                        params[key] = np.random.uniform(value[0], value[1])
                else:
                    params[key] = value

            score = self.evaluate_params(params)

            if score > best_score:
                best_score = score
                best_params = params.copy()

            print(f"  Trial {i+1}/{n_iter}: IC={score:.4f} (best={best_score:.4f})")

        return best_params, best_score


class StrategyBacktest:
    """策略回测引擎"""

    def __init__(self, train_window=756, predict_window=20, rebalance_freq=5):
        self.train_window = train_window
        self.predict_window = predict_window
        self.rebalance_freq = rebalance_freq

    def rolling_train_predict(self, data, xgb_params, feature_cols):
        """滚动窗口训练和预测"""
        data = data.sort_values('trade_date').reset_index(drop=True)
        unique_dates = sorted(data['trade_date'].unique())

        predictions_all = []

        for i in range(self.train_window, len(unique_dates), self.predict_window):
            train_dates = unique_dates[i - self.train_window:i]
            test_end_idx = min(i + self.predict_window, len(unique_dates))
            test_dates = unique_dates[i:test_end_idx]

            if len(test_dates) == 0:
                break

            print(f"  训练: {train_dates[0].date()} ~ {train_dates[-1].date()} | "
                  f"预测: {test_dates[0].date()} ~ {test_dates[-1].date()}")

            train_data = data[data['trade_date'].isin(train_dates)]
            test_data = data[data['trade_date'].isin(test_dates)]

            X_train = train_data[feature_cols].values
            y_train = train_data['forward_return'].values
            X_test = test_data[feature_cols].values
            y_test = test_data['forward_return'].values

            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test)

            model = xgb.train(
                xgb_params,
                dtrain,
                num_boost_round=xgb_params.get('num_boost_round', 100),
                verbose_eval=False
            )

            y_pred = model.predict(dtest)

            predictions = pd.DataFrame({
                'ts_code': test_data['ts_code'].values,
                'trade_date': test_data['trade_date'].values,
                'predicted_alpha': y_pred,
                'actual_return': y_test
            })

            predictions_all.append(predictions)

        return pd.concat(predictions_all, ignore_index=True)

    def calculate_ic(self, predictions):
        """计算IC和Rank IC"""
        daily_ic = []
        daily_rank_ic = []

        for date, group in predictions.groupby('trade_date'):
            if len(group) > 5:
                ic = np.corrcoef(
                    group['predicted_alpha'].values,
                    group['actual_return'].values
                )[0, 1]

                if not np.isnan(ic):
                    daily_ic.append(ic)

                rank_ic, _ = spearmanr(
                    group['predicted_alpha'].values,
                    group['actual_return'].values
                )

                if not np.isnan(rank_ic):
                    daily_rank_ic.append(rank_ic)

        return np.mean(daily_ic), np.mean(daily_rank_ic)

    def backtest_strategy(self, predictions, top_quantile=0.15, weight_method='equal'):
        """回测策略"""
        unique_dates = sorted(predictions['trade_date'].unique())
        rebalance_dates = unique_dates[::self.rebalance_freq]

        portfolio_returns = []

        for rebalance_date in rebalance_dates:
            today_pred = predictions[predictions['trade_date'] == rebalance_date].copy()

            if len(today_pred) == 0:
                continue

            n_stocks = max(1, int(len(today_pred) * top_quantile))
            top_stocks = today_pred.nlargest(n_stocks, 'predicted_alpha')

            if weight_method == 'equal':
                top_stocks['weight'] = 1.0 / n_stocks
            else:  # signal
                signals = top_stocks['predicted_alpha'].values
                signals = signals - signals.min() + 1e-6
                top_stocks['weight'] = signals / signals.sum()

            portfolio_return = (top_stocks['actual_return'] * top_stocks['weight']).sum()

            portfolio_returns.append({
                'date': rebalance_date,
                'portfolio_return': portfolio_return
            })

        nav_df = pd.DataFrame(portfolio_returns)
        nav_df['cumulative_nav'] = (1 + nav_df['portfolio_return']).cumprod()

        return nav_df

    def calculate_metrics(self, nav_df):
        """计算性能指标"""
        returns = nav_df['portfolio_return'].values
        nav = nav_df['cumulative_nav'].values

        total_return = nav[-1] - 1
        n_periods = len(nav)
        periods_per_year = 252 / self.rebalance_freq
        annual_return = (1 + total_return) ** (periods_per_year / n_periods) - 1

        annual_volatility = np.std(returns) * np.sqrt(periods_per_year)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0.0

        cumulative = nav
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)

        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
        ir = np.mean(returns) / np.std(returns) * np.sqrt(periods_per_year) if np.std(returns) > 0 else 0.0

        return {
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'calmar_ratio': calmar_ratio,
            'ir': ir
        }


def main():
    print("="*70)
    print("XGBoost调参和回测完整流程")
    print("="*70)

    # 1. 加载数据
    print("\n阶段1: 加载数据")
    print("-"*70)

    loader = ParquetDataLoader(data_dir='data/raw')
    data, feature_cols = loader.load_and_prepare(
        start_date='2020-01-01',
        end_date='2024-12-31',
        forward_days=5
    )

    # 准备X, y
    X = data[feature_cols].values
    y = data['forward_return'].values

    # 2. 超参数调优
    print("\n阶段2: 超参数调优")
    print("-"*70)

    param_space = {
        'max_depth': (4, 8),
        'learning_rate': (0.03, 0.1),
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': (1, 5),
        'gamma': 0.0,
        'reg_alpha': (0.0, 2.0),
        'reg_lambda': (0.5, 3.0),
        'num_boost_round': (80, 150)
    }

    tuner = XGBoostTuner(X, y, n_splits=5)
    best_params, best_ic = tuner.random_search(param_space, n_iter=20)

    print(f"\n最佳参数: {best_params}")
    print(f"最佳IC: {best_ic:.4f}")

    # 3. 完整回测
    print("\n阶段3: 完整回测")
    print("-"*70)

    xgb_params = {k: v for k, v in best_params.items() if k != 'num_boost_round'}
    xgb_params.update({
        'objective': 'reg:squarederror',
        'seed': 42,
        'verbosity': 0
    })

    backtest_engine = StrategyBacktest(
        train_window=756,
        predict_window=20,
        rebalance_freq=5
    )

    print("\n滚动窗口训练和预测...")
    predictions = backtest_engine.rolling_train_predict(data, xgb_params, feature_cols)

    print(f"\n生成预测: {len(predictions)} 条")

    ic, rank_ic = backtest_engine.calculate_ic(predictions)
    print(f"IC: {ic:.4f}")
    print(f"Rank IC: {rank_ic:.4f}")

    # 4. 多策略回测
    print("\n阶段4: 多策略对比")
    print("-"*70)

    strategies = [
        {'top_quantile': 0.10, 'weight_method': 'equal', 'name': '10%等权'},
        {'top_quantile': 0.15, 'weight_method': 'equal', 'name': '15%等权'},
        {'top_quantile': 0.20, 'weight_method': 'equal', 'name': '20%等权'},
        {'top_quantile': 0.10, 'weight_method': 'signal', 'name': '10%信号加权'},
        {'top_quantile': 0.15, 'weight_method': 'signal', 'name': '15%信号加权'},
        {'top_quantile': 0.20, 'weight_method': 'signal', 'name': '20%信号加权'},
    ]

    results = {}

    for strategy in strategies:
        print(f"\n策略: {strategy['name']}")

        nav_df = backtest_engine.backtest_strategy(
            predictions,
            strategy['top_quantile'],
            strategy['weight_method']
        )

        metrics = backtest_engine.calculate_metrics(nav_df)
        metrics['ic'] = ic
        metrics['rank_ic'] = rank_ic

        print(f"  IC: {metrics['ic']:.4f}")
        print(f"  IR: {metrics['ir']:.2f}")
        print(f"  年化收益: {metrics['annual_return']:.2%}")
        print(f"  最大回撤: {metrics['max_drawdown']:.2%}")
        print(f"  夏普比率: {metrics['sharpe_ratio']:.2f}")

        results[strategy['name']] = metrics

    # 5. 目标达成检查
    print("\n" + "="*70)
    print("目标达成情况")
    print("="*70)

    targets = {
        'ic': 0.04,
        'ir': 1.5,
        'annual_return': 0.35,
        'max_drawdown': -0.20
    }

    print(f"\n{'策略':<15} {'IC':<8} {'IR':<8} {'年化收益':<12} {'最大回撤':<12} {'达标':<8}")
    print("-"*70)

    for strategy_name, metrics in results.items():
        passed = 0
        if metrics['ic'] > targets['ic']:
            passed += 1
        if metrics['ir'] >= targets['ir']:
            passed += 1
        if metrics['annual_return'] > targets['annual_return']:
            passed += 1
        if metrics['max_drawdown'] >= targets['max_drawdown']:
            passed += 1

        status = "✓" if passed == 4 else f"{passed}/4"

        print(f"{strategy_name:<15} "
              f"{metrics['ic']:<8.4f} "
              f"{metrics['ir']:<8.2f} "
              f"{metrics['annual_return']:<11.2%} "
              f"{metrics['max_drawdown']:<11.2%} "
              f"{status:<8}")

    # 6. 保存结果
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    result_file = output_dir / f'xgboost_results_{timestamp}.json'
    with open(result_file, 'w') as f:
        json.dump({
            'best_params': best_params,
            'best_ic': best_ic,
            'strategies': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()},
            'timestamp': timestamp
        }, f, indent=2)

    print(f"\n结果已保存到: {result_file}")

    print("\n" + "="*70)
    print("完成!")
    print("="*70)


if __name__ == '__main__':
    main()
