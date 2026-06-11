#!/usr/bin/env python3
"""
MoE模型超参数调优脚本
=====================

使用网格搜索或随机搜索自动调优MoE模型超参数。

用法：
    # 网格搜索
    python scripts/tune_moe_hyperparams.py --method grid --max_trials 20

    # 随机搜索
    python scripts/tune_moe_hyperparams.py --method random --max_trials 50

    # 贝叶斯优化（需要安装optuna）
    python scripts/tune_moe_hyperparams.py --method bayesian --max_trials 30
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from itertools import product
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.alpha.moe_data_loader import MoEParquetDataLoader
from src.models.alpha.moe_model import MoEDataset, MoEAlphaTrainer
from torch.utils.data import DataLoader
import torch

try:
    import mlflow
    import mlflow.pytorch
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
    print("警告: 未安装MLflow，无法记录实验")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='MoE Model Hyperparameter Tuning')

    # 调优方法
    parser.add_argument('--method', type=str, default='grid',
                       choices=['grid', 'random', 'bayesian'],
                       help='调优方法：grid（网格搜索）, random（随机搜索）, bayesian（贝叶斯优化）')
    parser.add_argument('--max_trials', type=int, default=20,
                       help='最大试验次数')

    # 数据参数
    parser.add_argument('--stock_data', type=str,
                       default='data/raw/star50_daily_hfq_data_6yrs.parquet')
    parser.add_argument('--index_data', type=str,
                       default='data/raw/star50_index_daily_6yrs.parquet')
    parser.add_argument('--forward_days', type=int, default=5)
    parser.add_argument('--beta_window', type=int, default=60)

    # 训练参数
    parser.add_argument('--epochs', type=int, default=10,
                       help='每次试验的训练轮数（建议10-15以加快调优）')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--train_months', type=int, default=12,
                       help='训练窗口（月）')
    parser.add_argument('--val_months', type=int, default=3,
                       help='验证窗口（月）')

    # 设备
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--experiment_name', type=str, default='moe_tuning')

    return parser.parse_args()


def get_hyperparameter_space():
    """
    定义超参数搜索空间

    Returns:
        dict: 超参数范围定义
    """
    return {
        'hidden_dim': [32, 64, 96, 128],
        'dropout': [0.2, 0.3, 0.4, 0.5],
        'learning_rate': [0.001, 0.003, 0.005, 0.01],
        'weight_decay': [1e-5, 1e-4, 1e-3],
    }


def grid_search_combinations(space, max_trials):
    """
    生成网格搜索的参数组合

    Args:
        space: 超参数空间
        max_trials: 最大试验次数

    Returns:
        list: 参数组合列表
    """
    keys = list(space.keys())
    values = [space[k] for k in keys]

    all_combinations = list(product(*values))

    # 如果组合数超过max_trials，随机采样
    if len(all_combinations) > max_trials:
        import random
        all_combinations = random.sample(all_combinations, max_trials)

    # 转换为字典列表
    combinations = []
    for combo in all_combinations:
        combinations.append(dict(zip(keys, combo)))

    return combinations


def random_search_combinations(space, max_trials):
    """
    生成随机搜索的参数组合

    Args:
        space: 超参数空间
        max_trials: 最大试验次数

    Returns:
        list: 参数组合列表
    """
    import random

    combinations = []
    for _ in range(max_trials):
        combo = {}
        for key, values in space.items():
            combo[key] = random.choice(values)
        combinations.append(combo)

    return combinations


def evaluate_hyperparameters(
    params,
    stock_features,
    regime_features,
    labels,
    args,
    device
):
    """
    评估一组超参数

    Args:
        params: 超参数字典
        stock_features: 个股特征
        regime_features: 环境特征
        labels: 标签
        args: 命令行参数
        device: 计算设备

    Returns:
        dict: 评估指标
    """
    # 统一列名
    if 'trade_date' in stock_features.columns:
        stock_features = stock_features.rename(columns={'trade_date': 'factor_date'})
    if 'trade_date' in regime_features.columns:
        regime_features = regime_features.rename(columns={'trade_date': 'factor_date'})
    if 'trade_date' in labels.columns:
        labels = labels.rename(columns={'trade_date': 'factor_date'})

    # 合并数据
    full_data = stock_features.merge(labels, on=['ts_code', 'factor_date'], how='inner')
    full_data['factor_date'] = pd.to_datetime(full_data['factor_date'])
    full_data['year_month'] = full_data['factor_date'].dt.to_period('M')

    months = np.sort(full_data['year_month'].unique())

    # 使用固定的训练/验证分割
    train_end_idx = args.train_months
    val_end_idx = train_end_idx + args.val_months

    if val_end_idx > len(months):
        print(f"警告: 数据不足，跳过此组参数")
        return None

    train_months = months[:train_end_idx]
    val_months = months[train_end_idx:val_end_idx]

    train_data = full_data[full_data['year_month'].isin(train_months)]
    val_data = full_data[full_data['year_month'].isin(val_months)]

    if len(val_data) == 0:
        return None

    # 准备数据
    stock_cols = [c for c in stock_features.columns if c not in ['ts_code', 'factor_date']]
    regime_cols = [c for c in regime_features.columns if c != 'factor_date']

    train_stock = train_data[['ts_code', 'factor_date'] + stock_cols]
    train_regime = regime_features[regime_features['factor_date'].isin(train_data['factor_date'])]
    train_labels = train_data[['ts_code', 'factor_date', 'residual_return']]

    val_stock = val_data[['ts_code', 'factor_date'] + stock_cols]
    val_regime = regime_features[regime_features['factor_date'].isin(val_data['factor_date'])]
    val_labels = val_data[['ts_code', 'factor_date', 'residual_return']]

    # 创建数据集
    train_dataset = MoEDataset(train_stock, train_regime, train_labels)
    val_dataset = MoEDataset(val_stock, val_regime, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 创建训练器
    trainer = MoEAlphaTrainer(
        stock_dim=len(stock_cols),
        regime_dim=len(regime_cols),
        hidden_dim=params['hidden_dim'],
        dropout=params['dropout'],
        learning_rate=params['learning_rate'],
        weight_decay=params['weight_decay'],
        device=device
    )

    # 训练
    trainer.train(train_loader, val_loader=None, num_epochs=args.epochs, verbose=False)

    # 评估
    val_loss, val_ic, val_icir = trainer.evaluate(val_loader)

    return {
        'val_loss': val_loss,
        'val_ic': val_ic,
        'val_icir': val_icir,
        'val_samples': len(val_data)
    }


def tune_hyperparameters(args):
    """
    主调优函数

    Args:
        args: 命令行参数
    """
    print(f"\n{'='*70}")
    print(f"MoE模型超参数调优 - {args.method.upper()}搜索")
    print(f"{'='*70}\n")

    # 加载数据
    print("加载数据...")
    loader = MoEParquetDataLoader(
        stock_data_path=args.stock_data,
        index_data_path=args.index_data
    )

    stock_features, regime_features, labels = loader.build_moe_dataset(
        forward_days=args.forward_days,
        beta_window=args.beta_window
    )

    # 设备
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"使用设备: {device}\n")

    # 生成参数组合
    space = get_hyperparameter_space()

    if args.method == 'grid':
        param_combinations = grid_search_combinations(space, args.max_trials)
    elif args.method == 'random':
        param_combinations = random_search_combinations(space, args.max_trials)
    elif args.method == 'bayesian':
        print("贝叶斯优化需要安装optuna: pip install optuna")
        print("暂时使用随机搜索代替")
        param_combinations = random_search_combinations(space, args.max_trials)

    print(f"将测试 {len(param_combinations)} 组超参数\n")

    # MLflow设置
    if HAS_MLFLOW:
        mlflow.set_experiment(args.experiment_name)

    # 调优循环
    results = []
    best_ic = -float('inf')
    best_params = None

    for i, params in enumerate(param_combinations, 1):
        print(f"[{i}/{len(param_combinations)}] 测试参数: {params}")

        if HAS_MLFLOW:
            with mlflow.start_run(run_name=f"trial_{i}"):
                mlflow.log_params(params)

                metrics = evaluate_hyperparameters(
                    params, stock_features, regime_features, labels, args, device
                )

                if metrics:
                    mlflow.log_metrics(metrics)
                    results.append({**params, **metrics})

                    print(f"  结果: IC={metrics['val_ic']:.4f}, ICIR={metrics['val_icir']:.4f}, Loss={metrics['val_loss']:.6f}")

                    if metrics['val_ic'] > best_ic:
                        best_ic = metrics['val_ic']
                        best_params = params
                        print(f"  ✓ 新的最佳IC: {best_ic:.4f}")
                else:
                    print(f"  跳过（数据不足）")
        else:
            metrics = evaluate_hyperparameters(
                params, stock_features, regime_features, labels, args, device
            )

            if metrics:
                results.append({**params, **metrics})
                print(f"  结果: IC={metrics['val_ic']:.4f}, ICIR={metrics['val_icir']:.4f}")

                if metrics['val_ic'] > best_ic:
                    best_ic = metrics['val_ic']
                    best_params = params
                    print(f"  ✓ 新的最佳IC: {best_ic:.4f}")

        print()

    # 保存结果
    results_df = pd.DataFrame(results)
    output_dir = 'outputs/moe_tuning'
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"{output_dir}/tuning_results_{timestamp}.csv"
    results_df.to_csv(results_file, index=False)

    # 总结
    print(f"\n{'='*70}")
    print("调优完成!")
    print(f"{'='*70}\n")

    print(f"最佳参数:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print(f"\n最佳验证IC: {best_ic:.4f}")

    print(f"\n结果已保存: {results_file}")

    # Top 5参数组合
    if len(results_df) > 0:
        print(f"\nTop 5 参数组合（按IC排序）:")
        top5 = results_df.nlargest(5, 'val_ic')
        print(top5[['hidden_dim', 'dropout', 'learning_rate', 'val_ic', 'val_icir']].to_string(index=False))

    # 保存最佳参数到JSON
    best_params_file = f"{output_dir}/best_params_{timestamp}.json"
    with open(best_params_file, 'w') as f:
        json.dump({
            'best_params': best_params,
            'best_ic': float(best_ic),
            'search_method': args.method,
            'num_trials': len(results)
        }, f, indent=2)

    print(f"最佳参数已保存: {best_params_file}")

    if HAS_MLFLOW:
        print(f"\n查看MLflow实验: mlflow ui --port 5000")


if __name__ == '__main__':
    args = parse_args()
    tune_hyperparameters(args)
