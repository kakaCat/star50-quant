#!/usr/bin/env python3
"""
快速测试XGBoost调参系统
======================

使用小数据集快速验证调参流程是否正常工作
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.models.hyperparameter_tuning import (
    ObjectiveFunction,
    HyperparameterTuner
)


def generate_synthetic_data(n_samples=1000, n_features=50):
    """
    生成合成数据用于测试

    Args:
        n_samples: 样本数
        n_features: 特征数

    Returns:
        features, labels DataFrames
    """
    print("生成合成测试数据...")

    # 生成日期序列
    dates = pd.date_range('2023-01-01', periods=n_samples//10, freq='D')

    # 生成特征
    np.random.seed(42)

    data = []
    for date in dates:
        for stock_id in range(10):  # 10只股票
            # 生成特征
            features = np.random.randn(n_features)

            # 生成相关的收益率（添加一些信号）
            signal = features[:5].mean()  # 前5个特征有信号
            noise = np.random.randn() * 0.05
            forward_return = signal * 0.01 + noise

            row = {
                'ts_code': f'68000{stock_id}.SH',
                'factor_date': date,
                **{f'feature_{i}': features[i] for i in range(n_features)},
                'forward_return': forward_return
            }
            data.append(row)

    df = pd.DataFrame(data)

    # 分离特征和标签
    feature_cols = ['ts_code', 'factor_date'] + [f'feature_{i}' for i in range(n_features)]
    features = df[feature_cols]
    labels = df[['ts_code', 'factor_date', 'forward_return']]

    print(f"✓ 生成 {len(features)} 样本，{n_features} 特征")

    return features, labels


def test_objective_function():
    """测试目标函数"""
    print("\n" + "="*60)
    print("测试 1: 目标函数")
    print("="*60)

    features, labels = generate_synthetic_data(n_samples=500, n_features=20)

    obj_fn = ObjectiveFunction(
        features=features,
        labels=labels,
        n_splits=3,
        ic_weight=0.4,
        ir_weight=0.3,
        return_weight=0.2,
        drawdown_weight=0.1
    )

    # 测试参数
    test_params = {
        'max_depth': 4,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'gamma': 0.0,
        'reg_alpha': 0.0,
        'reg_lambda': 1.0,
        'num_boost_round': 50,
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'seed': 42,
        'verbosity': 0
    }

    print("评估测试参数...")
    scores = obj_fn.evaluate_params(test_params)

    print(f"\n结果:")
    print(f"  IC: {scores['ic']:.4f}")
    print(f"  IR: {scores['ir']:.4f}")
    print(f"  年化收益: {scores['annual_return']:.2%}")
    print(f"  最大回撤: {scores['max_drawdown']:.2%}")
    print(f"  综合评分: {scores['composite_score']:.4f}")

    print("\n✓ 目标函数测试通过")
    return True


def test_random_search():
    """测试随机搜索"""
    print("\n" + "="*60)
    print("测试 2: 随机搜索")
    print("="*60)

    features, labels = generate_synthetic_data(n_samples=500, n_features=20)

    obj_fn = ObjectiveFunction(
        features=features,
        labels=labels,
        n_splits=3
    )

    param_space = {
        'max_depth': (3, 6),
        'learning_rate': (0.01, 0.1),
        'subsample': (0.7, 0.9),
        'colsample_bytree': (0.7, 0.9),
        'min_child_weight': (1, 5),
        'gamma': (0.0, 0.2),
        'reg_alpha': (0.0, 1.0),
        'reg_lambda': (0.5, 2.0),
        'num_boost_round': (30, 100)
    }

    tuner = HyperparameterTuner(
        objective_fn=obj_fn,
        param_space=param_space
    )

    print("执行随机搜索 (5次迭代)...")
    result = tuner.random_search(n_iter=5)

    print("\n✓ 随机搜索测试通过")
    print(f"  最佳评分: {result.best_score:.4f}")
    print(f"  完成试验: {len(result.all_trials)}")

    return True


def test_grid_search():
    """测试网格搜索"""
    print("\n" + "="*60)
    print("测试 3: 网格搜索")
    print("="*60)

    features, labels = generate_synthetic_data(n_samples=500, n_features=20)

    obj_fn = ObjectiveFunction(
        features=features,
        labels=labels,
        n_splits=3
    )

    # 小网格空间用于快速测试
    param_space = {
        'max_depth': [4, 6],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8],
        'colsample_bytree': [0.8],
        'min_child_weight': [1],
        'gamma': [0.0],
        'reg_alpha': [0.0, 0.5],
        'reg_lambda': [1.0],
        'num_boost_round': [50]
    }

    tuner = HyperparameterTuner(
        objective_fn=obj_fn,
        param_space=param_space
    )

    print(f"执行网格搜索 (2×2×2=8个组合)...")
    result = tuner.grid_search()

    print("\n✓ 网格搜索测试通过")
    print(f"  最佳评分: {result.best_score:.4f}")
    print(f"  完成试验: {len(result.all_trials)}")

    return True


def test_bayesian_optimization():
    """测试贝叶斯优化"""
    print("\n" + "="*60)
    print("测试 4: 贝叶斯优化")
    print("="*60)

    try:
        import optuna
    except ImportError:
        print("✗ Optuna未安装，跳过贝叶斯优化测试")
        print("  安装命令: pip install optuna")
        return False

    features, labels = generate_synthetic_data(n_samples=500, n_features=20)

    obj_fn = ObjectiveFunction(
        features=features,
        labels=labels,
        n_splits=3
    )

    param_space = {
        'max_depth': (3, 6),
        'learning_rate': (0.01, 0.1),
        'subsample': (0.7, 0.9),
        'colsample_bytree': (0.7, 0.9),
        'min_child_weight': (1, 5),
        'gamma': (0.0, 0.2),
        'reg_alpha': (0.0, 1.0),
        'reg_lambda': (0.5, 2.0),
        'num_boost_round': (30, 100)
    }

    tuner = HyperparameterTuner(
        objective_fn=obj_fn,
        param_space=param_space
    )

    print("执行贝叶斯优化 (5次试验)...")
    result = tuner.bayesian_optimization(n_trials=5, n_startup_trials=2)

    print("\n✓ 贝叶斯优化测试通过")
    print(f"  最佳评分: {result.best_score:.4f}")
    print(f"  完成试验: {len(result.all_trials)}")

    return True


def main():
    """运行所有测试"""
    print("="*60)
    print("XGBoost调参系统快速测试")
    print("="*60)
    print("\n使用合成数据进行快速验证")

    results = []

    # 测试1: 目标函数
    try:
        results.append(("目标函数", test_objective_function()))
    except Exception as e:
        print(f"\n✗ 目标函数测试失败: {e}")
        results.append(("目标函数", False))

    # 测试2: 随机搜索
    try:
        results.append(("随机搜索", test_random_search()))
    except Exception as e:
        print(f"\n✗ 随机搜索测试失败: {e}")
        results.append(("随机搜索", False))

    # 测试3: 网格搜索
    try:
        results.append(("网格搜索", test_grid_search()))
    except Exception as e:
        print(f"\n✗ 网格搜索测试失败: {e}")
        results.append(("网格搜索", False))

    # 测试4: 贝叶斯优化
    try:
        results.append(("贝叶斯优化", test_bayesian_optimization()))
    except Exception as e:
        print(f"\n✗ 贝叶斯优化测试失败: {e}")
        results.append(("贝叶斯优化", False))

    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)

    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {name}: {status}")

    total_passed = sum(1 for _, p in results if p)
    total_tests = len(results)

    print(f"\n总计: {total_passed}/{total_tests} 测试通过")

    if total_passed == total_tests:
        print("\n🎉 所有测试通过！调参系统已就绪。")
        print("\n下一步:")
        print("  1. 准备真实数据: python scripts/calculate_factors.py")
        print("  2. 运行调参: python scripts/tune_xgb_model.py --method bayesian")
    else:
        print("\n⚠️ 部分测试失败，请检查依赖安装:")
        print("  pip install xgboost optuna scikit-learn pandas numpy")

    return total_passed == total_tests


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
