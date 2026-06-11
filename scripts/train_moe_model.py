#!/usr/bin/env python3
"""
训练Neural MoE Alpha预测模型
============================

功能：
- 使用项目数据库和特征工程
- Beta剥离，预测真实Alpha（residual）
- Walk-forward验证
- MLflow实验追踪

用法：
    python scripts/train_moe_model.py --start_date 2019-01-01 --end_date 2024-12-31
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import torch
from torch.utils.data import DataLoader

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.alpha.moe_data_loader import MoEParquetDataLoader
from src.models.alpha.moe_model import MoEDataset, MoEAlphaTrainer
import mlflow
import mlflow.pytorch


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Train Neural MoE Alpha Model')

    # 数据参数
    parser.add_argument('--stock_data', type=str,
                       default='data/raw/star50_daily_hfq_data_6yrs.parquet',
                       help='股票数据Parquet文件路径')
    parser.add_argument('--index_data', type=str,
                       default='data/raw/star50_index_daily_6yrs.parquet',
                       help='指数数据Parquet文件路径')
    parser.add_argument('--forward_days', type=int, default=5,
                       help='预测未来N天收益率')
    parser.add_argument('--beta_window', type=int, default=60,
                       help='Beta计算滚动窗口')

    # 模型参数
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='隐藏层维度')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout比例')
    parser.add_argument('--learning_rate', type=float, default=0.005,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='L2正则化系数')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=25,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=1024,
                       help='批次大小')
    parser.add_argument('--train_months', type=int, default=24,
                       help='Walk-forward训练窗口（月）')
    parser.add_argument('--test_months', type=int, default=1,
                       help='Walk-forward测试窗口（月）')

    # 其他参数
    parser.add_argument('--device', type=str, default='auto',
                       help='设备（cpu/cuda/auto）')
    parser.add_argument('--experiment_name', type=str, default='moe_alpha',
                       help='MLflow实验名称')
    parser.add_argument('--no_mlflow', action='store_true',
                       help='禁用MLflow追踪')

    return parser.parse_args()


def split_stock_regime_features(features_wide: pd.DataFrame) -> tuple:
    """
    分离个股特征和环境特征

    Args:
        features_wide: 宽表格式特征DataFrame

    Returns:
        (stock_feature_names, regime_feature_names)
    """
    all_cols = [col for col in features_wide.columns
                if col not in ['ts_code', 'factor_date']]

    # 环境特征：以index_或market_开头的特征
    regime_cols = [col for col in all_cols
                   if col.startswith('index_') or col.startswith('market_')]

    # 个股特征：剩余特征
    stock_cols = [col for col in all_cols if col not in regime_cols]

    print(f"\n特征分离:")
    print(f"  - 个股特征: {len(stock_cols)}个")
    print(f"  - 环境特征: {len(regime_cols)}个")

    return stock_cols, regime_cols


def walk_forward_validation(
    stock_features: pd.DataFrame,
    regime_features: pd.DataFrame,
    labels: pd.DataFrame,
    args
) -> pd.DataFrame:
    """
    Walk-forward验证

    Args:
        stock_features: 个股特征
        regime_features: 环境特征
        labels: 标签
        args: 命令行参数

    Returns:
        测试集预测结果DataFrame
    """
    # 统一列名：trade_date → factor_date
    if 'trade_date' in stock_features.columns and 'factor_date' not in stock_features.columns:
        stock_features = stock_features.rename(columns={'trade_date': 'factor_date'})
    if 'trade_date' in regime_features.columns and 'factor_date' not in regime_features.columns:
        regime_features = regime_features.rename(columns={'trade_date': 'factor_date'})
    if 'trade_date' in labels.columns and 'factor_date' not in labels.columns:
        labels = labels.rename(columns={'trade_date': 'factor_date'})

    # 合并所有数据以便按月分割
    full_data = stock_features.merge(
        labels,
        on=['ts_code', 'factor_date'],
        how='inner'
    )

    # 按月分组
    full_data['factor_date'] = pd.to_datetime(full_data['factor_date'])
    full_data['year_month'] = full_data['factor_date'].dt.to_period('M')
    months = np.sort(full_data['year_month'].unique())

    print(f"\n数据时间范围: {months[0]} 到 {months[-1]} (共{len(months)}个月)")
    print(f"Walk-forward设置: {args.train_months}月训练 → {args.test_months}月测试\n")

    # 设备
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"使用设备: {device}\n")

    # Walk-forward循环
    predictions_list = []
    stock_cols = [col for col in stock_features.columns
                  if col not in ['ts_code', 'factor_date']]
    regime_cols = [col for col in regime_features.columns
                   if col != 'factor_date']

    for i in range(args.train_months, len(months), args.test_months):
        train_months_range = months[i - args.train_months: i]
        test_months_range = months[i: i + args.test_months]

        if len(test_months_range) == 0:
            break

        print(f"{'='*70}")
        print(f"训练窗口: {train_months_range[0]} → {train_months_range[-1]}")
        print(f"测试窗口: {test_months_range[0]} → {test_months_range[-1]}")

        # 分割数据
        train_data = full_data[full_data['year_month'].isin(train_months_range)]
        test_data = full_data[full_data['year_month'].isin(test_months_range)]

        if len(test_data) == 0:
            print("测试集为空，跳过")
            continue

        print(f"训练样本: {len(train_data)}, 测试样本: {len(test_data)}")

        # 构建数据集
        train_stock = train_data[['ts_code', 'factor_date'] + stock_cols]
        train_regime = regime_features[
            regime_features['factor_date'].isin(train_data['factor_date'])
        ]
        train_labels = train_data[['ts_code', 'factor_date', 'residual_return']]

        test_stock = test_data[['ts_code', 'factor_date'] + stock_cols]
        test_regime = regime_features[
            regime_features['factor_date'].isin(test_data['factor_date'])
        ]

        train_dataset = MoEDataset(train_stock, train_regime, train_labels)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True
        )

        # 创建训练器
        trainer = MoEAlphaTrainer(
            stock_dim=len(stock_cols),
            regime_dim=len(regime_cols),
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=device
        )

        # 训练模型
        trainer.train(
            train_loader=train_loader,
            val_loader=None,
            num_epochs=args.epochs,
            verbose=False
        )

        # 预测
        test_stock_np = test_stock[stock_cols].values
        test_regime_merged = test_stock.merge(
            test_regime,
            on='factor_date',
            how='left'
        )[regime_cols].values

        predictions = trainer.predict(test_stock_np, test_regime_merged)

        # 保存结果
        test_result = test_stock[['ts_code', 'factor_date']].copy()
        test_result['pred_alpha'] = predictions
        test_result = test_result.merge(
            test_data[['ts_code', 'factor_date', 'residual_return']],
            on=['ts_code', 'factor_date'],
            how='inner'
        )

        predictions_list.append(test_result)

        # 评估
        from scipy.stats import spearmanr
        valid_mask = ~(np.isnan(predictions) | np.isnan(test_result['residual_return'].values))
        if valid_mask.sum() > 0:
            ic, _ = spearmanr(
                predictions[valid_mask],
                test_result['residual_return'].values[valid_mask]
            )
            print(f"测试集 Rank IC: {ic:.4f}")

        print()

    # 合并所有预测
    all_predictions = pd.concat(predictions_list, ignore_index=True)
    return all_predictions


def evaluate_predictions(predictions_df: pd.DataFrame):
    """
    评估预测结果

    Args:
        predictions_df: 预测结果DataFrame
    """
    print(f"\n{'='*70}")
    print("全周期 Out-of-Sample 评估")
    print(f"{'='*70}\n")

    # 按日期分组计算IC
    ic_list = []
    for date, group in predictions_df.groupby('factor_date'):
        if len(group) > 2:
            from scipy.stats import spearmanr
            valid_mask = ~(np.isnan(group['pred_alpha']) | np.isnan(group['residual_return']))
            if valid_mask.sum() > 2:
                ic, _ = spearmanr(
                    group.loc[valid_mask, 'pred_alpha'],
                    group.loc[valid_mask, 'residual_return']
                )
                ic_list.append(ic)

    ic_series = pd.Series(ic_list)

    # 统计指标
    mean_ic = ic_series.mean()
    std_ic = ic_series.std()
    icir = mean_ic / std_ic if std_ic > 0 else 0

    print(f"Rank IC:")
    print(f"  - 均值: {mean_ic:.4f}")
    print(f"  - 标准差: {std_ic:.4f}")
    print(f"  - ICIR: {icir:.4f}")
    print(f"  - IC > 0 占比: {(ic_series > 0).sum() / len(ic_series):.2%}")
    print(f"  - IC > 0.02 占比: {(ic_series > 0.02).sum() / len(ic_series):.2%}")

    return {
        'mean_ic': mean_ic,
        'std_ic': std_ic,
        'icir': icir,
        'ic_positive_ratio': (ic_series > 0).sum() / len(ic_series)
    }


def main():
    """主函数"""
    args = parse_args()

    print(f"\n{'='*70}")
    print("Neural MoE Alpha Model Training")
    print(f"{'='*70}\n")

    # MLflow设置
    if not args.no_mlflow:
        mlflow.set_experiment(args.experiment_name)
        mlflow.start_run()
        mlflow.log_params({
            'stock_data': args.stock_data,
            'index_data': args.index_data,
            'forward_days': args.forward_days,
            'beta_window': args.beta_window,
            'hidden_dim': args.hidden_dim,
            'dropout': args.dropout,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'train_months': args.train_months,
            'test_months': args.test_months,
        })

    try:
        # 加载数据
        print("步骤 1/4: 加载数据...")
        loader = MoEParquetDataLoader(
            stock_data_path=args.stock_data,
            index_data_path=args.index_data
        )

        stock_features, regime_features, labels = loader.build_moe_dataset(
            forward_days=args.forward_days,
            beta_window=args.beta_window
        )

        # Walk-forward验证
        print("\n步骤 2/4: Walk-forward验证训练...")
        predictions_df = walk_forward_validation(
            stock_features,
            regime_features,
            labels,
            args
        )

        # 评估
        print("\n步骤 3/4: 评估预测结果...")
        metrics = evaluate_predictions(predictions_df)

        # 保存结果
        print("\n步骤 4/4: 保存结果...")
        output_dir = 'outputs/moe_predictions'
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"{output_dir}/predictions_{timestamp}.csv"
        predictions_df.to_csv(output_file, index=False)
        print(f"预测结果已保存: {output_file}")

        # MLflow记录
        if not args.no_mlflow:
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(output_file)
            mlflow.end_run()

        print(f"\n{'='*70}")
        print("训练完成!")
        print(f"{'='*70}\n")

    except Exception as e:
        if not args.no_mlflow:
            mlflow.end_run(status='FAILED')
        raise e


if __name__ == '__main__':
    main()
