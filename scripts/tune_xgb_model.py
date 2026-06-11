#!/usr/bin/env python3
"""
XGBoost模型超参数调优脚本
========================

支持三种调参方法：
1. random - 随机搜索
2. grid - 网格搜索
3. bayesian - 贝叶斯优化

目标：
- IC > 0.04
- IR >= 1.5
- 年化收益 > 35%
- 最大回撤 <= 20%

使用方法:
    python scripts/tune_xgb_model.py --method bayesian --n_iter 50
    python scripts/tune_xgb_model.py --method random --n_iter 100
    python scripts/tune_xgb_model.py --method grid
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import pandas as pd
import numpy as np
from datetime import datetime

from src.models.data_loader import FactorDataLoader
from src.models.hyperparameter_tuning import (
    ObjectiveFunction,
    HyperparameterTuner,
    TuningResult
)
from src.models.xgb_model import XGBoostAlphaModel


def define_param_space(search_type: str) -> dict:
    """
    定义参数搜索空间

    Args:
        search_type: 搜索类型 ('random', 'grid', 'bayesian')

    Returns:
        参数空间字典
    """
    if search_type == 'grid':
        # 网格搜索：离散值列表
        param_space = {
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'min_child_weight': [1, 3, 5],
            'gamma': [0.0, 0.1, 0.2],
            'reg_alpha': [0.0, 0.1, 1.0],
            'reg_lambda': [0.5, 1.0, 2.0],
            'num_boost_round': [100]
        }
    elif search_type == 'random':
        # 随机搜索：范围或离散值
        param_space = {
            'max_depth': (3, 10),  # (min, max)
            'learning_rate': (0.01, 0.2),
            'subsample': (0.6, 1.0),
            'colsample_bytree': (0.6, 1.0),
            'colsample_bylevel': (0.6, 1.0),
            'min_child_weight': (1, 10),
            'gamma': (0.0, 0.5),
            'reg_alpha': (0.0, 2.0),
            'reg_lambda': (0.0, 3.0),
            'num_boost_round': (50, 200)
        }
    else:  # bayesian
        # 贝叶斯优化：连续或离散范围
        param_space = {
            'max_depth': (3, 10),
            'learning_rate': (0.005, 0.3),
            'subsample': (0.5, 1.0),
            'colsample_bytree': (0.5, 1.0),
            'colsample_bylevel': (0.5, 1.0),
            'min_child_weight': (1, 15),
            'gamma': (0.0, 1.0),
            'reg_alpha': (0.0, 3.0),
            'reg_lambda': (0.0, 5.0),
            'num_boost_round': (50, 300)
        }

    return param_space


def main():
    parser = argparse.ArgumentParser(description='XGBoost超参数调优')
    parser.add_argument(
        '--method',
        type=str,
        choices=['random', 'grid', 'bayesian'],
        default='bayesian',
        help='调参方法'
    )
    parser.add_argument(
        '--n_iter',
        type=int,
        default=50,
        help='迭代次数（random/bayesian）'
    )
    parser.add_argument(
        '--cv_folds',
        type=int,
        default=5,
        help='交叉验证折数'
    )
    parser.add_argument(
        '--start_date',
        type=str,
        default='2020-01-01',
        help='训练数据起始日期'
    )
    parser.add_argument(
        '--end_date',
        type=str,
        default='2024-12-31',
        help='训练数据结束日期'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='tuning_results',
        help='结果输出目录'
    )
    parser.add_argument(
        '--ic_weight',
        type=float,
        default=0.4,
        help='IC权重'
    )
    parser.add_argument(
        '--ir_weight',
        type=float,
        default=0.3,
        help='IR权重'
    )
    parser.add_argument(
        '--return_weight',
        type=float,
        default=0.2,
        help='年化收益权重'
    )
    parser.add_argument(
        '--drawdown_weight',
        type=float,
        default=0.1,
        help='最大回撤权重'
    )

    args = parser.parse_args()

    print("="*80)
    print("XGBoost超参数调优")
    print("="*80)
    print(f"调参方法: {args.method}")
    print(f"数据范围: {args.start_date} 至 {args.end_date}")
    print(f"交叉验证: {args.cv_folds} 折")
    print(f"目标权重: IC={args.ic_weight}, IR={args.ir_weight}, "
          f"Return={args.return_weight}, Drawdown={args.drawdown_weight}")

    # 1. 加载数据
    print("\n" + "="*80)
    print("步骤 1: 加载数据")
    print("="*80)

    loader = FactorDataLoader(db_name='star50_quant')

    try:
        loader.connect()

        # 1. 加载因子数据（长格式）
        print("  加载因子数据...")
        factors_long = loader.load_factors(args.start_date, args.end_date)
        print(f"    ✓ 加载 {len(factors_long):,} 条记录")

        # 2. 转换为宽格式
        print("  转换为宽格式...")
        features = loader.pivot_factors(factors_long)
        print(f"    ✓ 转换完成: {features.shape}")

        # 3. 加载价格数据并计算收益率
        print("  计算未来收益率...")
        # 需要加载更多数据以计算forward return
        from datetime import datetime, timedelta
        end_date_extended = (datetime.strptime(args.end_date, '%Y-%m-%d') + timedelta(days=30)).strftime('%Y-%m-%d')
        prices = loader.load_prices(args.start_date, end_date_extended)
        labels = loader.calculate_returns(prices, forward_days=5)
        print(f"    ✓ 计算完成: {labels.shape}")

        # 4. 合并数据
        print("  合并数据...")
        labels['factor_date'] = pd.to_datetime(labels['trade_date'])
        features['factor_date'] = pd.to_datetime(features['factor_date'])

        merged = features.merge(
            labels[['ts_code', 'factor_date', 'forward_return']],
            on=['ts_code', 'factor_date'],
            how='inner'
        )

        # 删除缺失值
        merged = merged.dropna()

        print(f"✓ 数据加载成功")
        print(f"  样本数: {len(merged)}")
        print(f"  特征数: {len(merged.columns) - 3}")  # 减去ts_code, factor_date, forward_return

        # 分离特征和标签
        features = merged.drop('forward_return', axis=1)
        labels = merged[['ts_code', 'factor_date', 'forward_return']].copy()
        labels.rename(columns={'factor_date': 'trade_date'}, inplace=True)

    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        print("\n提示: 请先运行以下命令准备数据:")
        print("  python scripts/calculate_factors.py --all --start 2020-01-02 --end 2024-12-31")
        if loader.conn:
            loader.close()
        return

    print("\n" + "="*80)
    print("步骤 2: 数据预处理")
    print("="*80)

    # 去极值
    print("  去极值（MAD方法）...")
    features_processed = loader.winsorize(features.copy(), n_sigma=3.0)

    # 标准化
    print("  标准化...")
    features_processed = loader.standardize(features_processed)

    print("✓ 预处理完成（去极值 + 标准化）")

    print(f"  特征形状: {features_processed.shape}")
    print(f"  标签形状: {labels.shape}")

    # 关闭数据库连接
    loader.close()

    # 3. 定义参数空间
    print("\n" + "="*80)
    print("步骤 3: 定义参数空间")
    print("="*80)

    param_space = define_param_space(args.method)

    print(f"参数空间:")
    for key, value in param_space.items():
        print(f"  {key}: {value}")

    # 4. 创建目标函数
    print("\n" + "="*80)
    print("步骤 4: 创建目标函数")
    print("="*80)

    objective_fn = ObjectiveFunction(
        features=features_processed,
        labels=labels,
        n_splits=args.cv_folds,
        ic_weight=args.ic_weight,
        ir_weight=args.ir_weight,
        return_weight=args.return_weight,
        drawdown_weight=args.drawdown_weight
    )
    print("✓ 目标函数创建完成")

    # 5. 创建调优器
    print("\n" + "="*80)
    print("步骤 5: 超参数搜索")
    print("="*80)

    tuner = HyperparameterTuner(
        objective_fn=objective_fn,
        param_space=param_space
    )

    # 执行调优
    if args.method == 'random':
        result = tuner.random_search(n_iter=args.n_iter)
    elif args.method == 'grid':
        result = tuner.grid_search()
    else:  # bayesian
        result = tuner.bayesian_optimization(n_trials=args.n_iter)

    # 6. 保存结果
    print("\n" + "="*80)
    print("步骤 6: 保存结果")
    print("="*80)

    os.makedirs(args.output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = os.path.join(
        args.output_dir,
        f'xgb_tuning_{args.method}_{timestamp}.json'
    )

    result.save(result_file)

    # 7. 使用最佳参数训练最终模型
    print("\n" + "="*80)
    print("步骤 7: 训练最终模型")
    print("="*80)

    print("使用最佳参数训练模型...")

    # 分割训练集和验证集（80/20）
    split_idx = int(len(features_processed) * 0.8)
    train_features = features_processed.iloc[:split_idx]
    train_labels = labels.iloc[:split_idx]
    val_features = features_processed.iloc[split_idx:]
    val_labels = labels.iloc[split_idx:]

    final_model = XGBoostAlphaModel(params=result.best_params)
    final_model.train(
        train_features,
        train_labels,
        val_features=val_features,
        val_labels=val_labels,
        num_boost_round=result.best_params.get('num_boost_round', 100),
        early_stopping_rounds=20
    )

    # 保存模型
    model_file = os.path.join(
        args.output_dir,
        f'xgb_best_model_{timestamp}.json'
    )
    final_model.save(model_file)

    # 保存特征重要性
    importance_file = os.path.join(
        args.output_dir,
        f'xgb_feature_importance_{timestamp}.csv'
    )
    final_model.get_feature_importance().to_csv(importance_file, index=False)
    print(f"✓ 特征重要性保存至: {importance_file}")

    # 8. 验证集评估
    print("\n" + "="*80)
    print("步骤 8: 验证集评估")
    print("="*80)

    val_predictions = final_model.predict(val_features)
    val_actuals = val_labels['forward_return'].values

    # 计算IC
    ic = np.corrcoef(val_predictions, val_actuals)[0, 1]
    print(f"验证集 IC: {ic:.4f}")

    # 保存预测结果
    predictions_df = pd.DataFrame({
        'ts_code': val_features['ts_code'].values,
        'factor_date': val_features['factor_date'].values,
        'prediction': val_predictions,
        'actual': val_actuals
    })

    predictions_file = os.path.join(
        args.output_dir,
        f'xgb_predictions_{timestamp}.csv'
    )
    predictions_df.to_csv(predictions_file, index=False)
    print(f"✓ 预测结果保存至: {predictions_file}")

    print("\n" + "="*80)
    print("调参完成!")
    print("="*80)
    print(f"\n结果文件:")
    print(f"  参数: {result_file}")
    print(f"  模型: {model_file}")
    print(f"  特征重要性: {importance_file}")
    print(f"  预测结果: {predictions_file}")

    print(f"\n下一步:")
    print(f"  1. 使用最佳模型进行完整回测")
    print(f"  2. 分析特征重要性，优化特征工程")
    print(f"  3. 如果指标未达标，尝试:")
    print(f"     - 调整目标函数权重")
    print(f"     - 扩大参数搜索空间")
    print(f"     - 增加迭代次数")
    print(f"     - 优化特征工程")


if __name__ == '__main__':
    main()
