#!/usr/bin/env python3
"""
Alpha模型训练脚本
=================

训练LightGBM和LSTM模型预测股票未来收益率。

用法:
    python scripts/train_alpha_model.py --model lgbm
    python scripts/train_alpha_model.py --model lstm
"""

import sys
import os
import argparse
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.data_loader import FactorDataLoader
from src.models.lgbm_model import LightGBMAlphaModel

# 可选导入PyTorch相关模块
try:
    from src.models.lstm_model import LSTMAlphaTrainer, StockSequenceDataset
    import torch
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def train_lgbm(
    start_date: str = '2020-01-01',
    end_date: str = '2024-12-31',
    forward_days: int = 5,
    save_path: str = 'models/lgbm_alpha.txt',
    enable_feature_engineering: bool = False,
    feature_config_path: str = '../configs/feature_config.yaml'
):
    """
    训练LightGBM模型

    Args:
        start_date: 开始日期
        end_date: 结束日期
        forward_days: 预测未来N天
        save_path: 模型保存路径
        enable_feature_engineering: 是否启用特征工程
        feature_config_path: 特征工程配置文件路径
    """
    print("="*60)
    print("训练LightGBM Alpha模型")
    print("="*60)

    # 加载数据
    with FactorDataLoader() as loader:
        features, labels = loader.build_dataset(
            start_date=start_date,
            end_date=end_date,
            forward_days=forward_days,
            enable_feature_engineering=enable_feature_engineering,
            feature_config_path=feature_config_path
        )

    print(f"\nDataset shape: {features.shape}")
    print(f"Date range: {features['factor_date'].min()} to {features['factor_date'].max()}")

    # 时间序列划分（80%训练，20%验证）
    split_date = features['factor_date'].quantile(0.8)
    train_features = features[features['factor_date'] <= split_date]
    train_labels = labels[labels['factor_date'] <= split_date]
    val_features = features[features['factor_date'] > split_date]
    val_labels = labels[labels['factor_date'] > split_date]

    print(f"\nTrain: {len(train_features)} samples")
    print(f"Val: {len(val_features)} samples")

    # 创建模型
    model = LightGBMAlphaModel()

    # 训练
    print("\n" + "="*60)
    print("开始训练...")
    print("="*60 + "\n")

    model.train(
        features=train_features,
        labels=train_labels,
        num_boost_round=200,
        early_stopping_rounds=20,
        verbose_eval=10
    )

    # 验证集评估
    print("\n" + "="*60)
    print("验证集评估")
    print("="*60 + "\n")

    val_pred = model.predict(val_features)
    val_true = val_labels['forward_return'].values

    import numpy as np
    mse = np.mean((val_pred - val_true) ** 2)
    rmse = np.sqrt(mse)
    ic = np.corrcoef(val_pred, val_true)[0, 1]

    print(f"Validation RMSE: {rmse:.6f}")
    print(f"Validation IC: {ic:.4f}")

    # 保存模型
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)

    # 保存特征重要性
    importance_path = save_path.replace('.txt', '_importance.csv')
    model.get_feature_importance().to_csv(importance_path, index=False)
    print(f"Feature importance saved to {importance_path}")

    print("\n✓ LightGBM模型训练完成!")


def train_lstm(
    start_date: str = '2020-01-01',
    end_date: str = '2024-12-31',
    forward_days: int = 5,
    lookback_days: int = 20,
    save_path: str = 'models/lstm_alpha.pth',
    enable_feature_engineering: bool = False,
    feature_config_path: str = '../configs/feature_config.yaml'
):
    """
    训练LSTM模型

    Args:
        start_date: 开始日期
        end_date: 结束日期
        forward_days: 预测未来N天
        lookback_days: 回看天数
        save_path: 模型保存路径
        enable_feature_engineering: 是否启用特征工程
        feature_config_path: 特征工程配置文件路径
    """
    print("="*60)
    print("训练LSTM Alpha模型")
    print("="*60)

    # 加载数据
    with FactorDataLoader() as loader:
        features, labels = loader.build_dataset(
            start_date=start_date,
            end_date=end_date,
            forward_days=forward_days,
            lookback_days=lookback_days,
            enable_feature_engineering=enable_feature_engineering,
            feature_config_path=feature_config_path
        )

    print(f"\nDataset shape: {features.shape}")
    print(f"Date range: {features['factor_date'].min()} to {features['factor_date'].max()}")

    # 时间序列划分（80%训练，20%验证）
    split_date = features['factor_date'].quantile(0.8)
    train_features = features[features['factor_date'] <= split_date]
    train_labels = labels[labels['factor_date'] <= split_date]
    val_features = features[features['factor_date'] > split_date]
    val_labels = labels[labels['factor_date'] > split_date]

    print(f"\nTrain: {len(train_features)} samples")
    print(f"Val: {len(val_features)} samples")

    # 创建数据集
    print("\n构建时序数据集...")
    train_dataset = StockSequenceDataset(train_features, train_labels, lookback_days)
    val_dataset = StockSequenceDataset(val_features, val_labels, lookback_days)

    print(f"Train sequences: {len(train_dataset)}")
    print(f"Val sequences: {len(val_dataset)}")

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    # 获取输入特征数
    input_size = train_dataset.sequences.shape[2]
    print(f"Input size (features): {input_size}")

    # 创建训练器
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    trainer = LSTMAlphaTrainer(
        input_size=input_size,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        learning_rate=0.001,
        device=device
    )

    # 训练
    print("\n" + "="*60)
    print("开始训练...")
    print("="*60 + "\n")

    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=50,
        early_stopping_patience=5
    )

    # 保存模型
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    trainer.save(save_path)

    print("\n✓ LSTM模型训练完成!")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练Alpha预测模型')
    parser.add_argument('--model', type=str, required=True, choices=['lgbm', 'lstm'],
                       help='模型类型：lgbm或lstm')
    parser.add_argument('--start', type=str, default='2020-01-01', help='开始日期')
    parser.add_argument('--end', type=str, default='2024-12-31', help='结束日期')
    parser.add_argument('--forward', type=int, default=5, help='预测未来N天')
    parser.add_argument('--lookback', type=int, default=20, help='LSTM回看天数')
    parser.add_argument('--enable-features', action='store_true',
                       help='启用特征工程（30→160特征）')
    parser.add_argument('--feature-config', type=str,
                       default='../configs/feature_config.yaml',
                       help='特征工程配置文件路径')

    args = parser.parse_args()

    if args.model == 'lgbm':
        train_lgbm(
            start_date=args.start,
            end_date=args.end,
            forward_days=args.forward,
            enable_feature_engineering=args.enable_features,
            feature_config_path=args.feature_config
        )
    elif args.model == 'lstm':
        if not TORCH_AVAILABLE:
            print("错误: PyTorch未安装，无法训练LSTM模型")
            print("请先安装: pip install torch")
            sys.exit(1)
        train_lstm(
            start_date=args.start,
            end_date=args.end,
            forward_days=args.forward,
            lookback_days=args.lookback,
            enable_feature_engineering=args.enable_features,
            feature_config_path=args.feature_config
        )


if __name__ == '__main__':
    main()
