#!/usr/bin/env python3
"""
风险模型训练脚本
================

训练深度风险模型，提取隐性风险因子。

用法:
    python scripts/train_risk_model.py --n_factors 10
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import pandas as pd
from src.models.data_loader import FactorDataLoader

try:
    from src.models.risk.deep_risk_model import DeepRiskModel
    RISK_MODEL_AVAILABLE = True
except ImportError:
    RISK_MODEL_AVAILABLE = False
    print("错误: 风险模型需要PyTorch")
    print("请安装: pip install torch")
    sys.exit(1)


def prepare_returns_matrix(
    start_date: str = '2020-01-01',
    end_date: str = '2024-12-31'
) -> tuple:
    """
    准备收益率矩阵

    Args:
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        (returns_matrix, stock_codes, dates)
    """
    print("="*60)
    print("准备收益率数据")
    print("="*60)

    with FactorDataLoader() as loader:
        # 加载价格数据
        prices = loader.load_prices(start_date, end_date)

    print(f"加载了 {len(prices)} 条价格记录")

    # 透视为宽表格式
    price_matrix = prices.pivot(
        index='trade_date',
        columns='ts_code',
        values='close'
    ).sort_index()

    # 计算收益率
    returns_matrix = price_matrix.pct_change().dropna()

    print(f"收益率矩阵形状: {returns_matrix.shape}")
    print(f"  交易日数: {len(returns_matrix)}")
    print(f"  股票数: {len(returns_matrix.columns)}")
    print(f"  日期范围: {returns_matrix.index[0]} 到 {returns_matrix.index[-1]}")

    return returns_matrix.values, list(returns_matrix.columns), list(returns_matrix.index)


def train_risk_model(
    n_factors: int = 10,
    hidden_dims: list = [64, 32],
    num_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001
):
    """
    训练风险模型

    Args:
        n_factors: 隐性风险因子数量
        hidden_dims: 隐藏层维度
        num_epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
    """
    print("\n" + "="*60)
    print("训练深度风险模型")
    print("="*60)

    # 准备数据
    returns_matrix, stock_codes, dates = prepare_returns_matrix()
    n_stocks = returns_matrix.shape[1]

    # 创建模型
    print(f"\n创建风险模型:")
    print(f"  股票数: {n_stocks}")
    print(f"  风险因子数: {n_factors}")
    print(f"  隐藏层: {hidden_dims}")

    model = DeepRiskModel(
        n_stocks=n_stocks,
        n_factors=n_factors,
        hidden_dims=hidden_dims,
        learning_rate=learning_rate,
        device='cuda' if __import__('torch').cuda.is_available() else 'cpu'
    )

    # 训练模型
    print("\n" + "="*60)
    print("开始训练...")
    print("="*60 + "\n")

    model.train(
        returns_data=returns_matrix,
        num_epochs=num_epochs,
        batch_size=batch_size,
        validation_split=0.2,
        early_stopping_patience=10
    )

    # 提取风险因子
    print("\n" + "="*60)
    print("提取风险因子暴露")
    print("="*60)

    factor_exposures, factor_returns = model.extract_risk_factors(returns_matrix)

    print(f"\n因子暴露矩阵形状: {factor_exposures.shape}")
    print(f"因子收益率矩阵形状: {factor_returns.shape}")

    # 计算风险协方差
    print("\n计算风险协方差矩阵...")
    factor_cov, specific_var = model.compute_risk_matrix(returns_matrix)

    print(f"因子协方差矩阵形状: {factor_cov.shape}")
    print(f"特质风险向量长度: {len(specific_var)}")

    # 保存结果
    print("\n" + "="*60)
    print("保存模型和结果")
    print("="*60)

    os.makedirs('models/risk', exist_ok=True)

    # 保存模型
    model.save('models/risk/deep_risk_model.pth')

    # 保存因子暴露
    factor_exposure_df = pd.DataFrame(
        factor_exposures,
        index=stock_codes,
        columns=[f'Factor_{i+1}' for i in range(n_factors)]
    )
    factor_exposure_df.to_csv('models/risk/factor_exposures.csv')
    print("  ✓ 因子暴露保存至: models/risk/factor_exposures.csv")

    # 保存因子收益率
    factor_returns_df = pd.DataFrame(
        factor_returns,
        index=dates,
        columns=[f'Factor_{i+1}' for i in range(n_factors)]
    )
    factor_returns_df.to_csv('models/risk/factor_returns.csv')
    print("  ✓ 因子收益率保存至: models/risk/factor_returns.csv")

    # 保存因子协方差
    factor_cov_df = pd.DataFrame(
        factor_cov,
        index=[f'Factor_{i+1}' for i in range(n_factors)],
        columns=[f'Factor_{i+1}' for i in range(n_factors)]
    )
    factor_cov_df.to_csv('models/risk/factor_covariance.csv')
    print("  ✓ 因子协方差保存至: models/risk/factor_covariance.csv")

    # 保存特质风险
    specific_risk_df = pd.DataFrame({
        'ts_code': stock_codes,
        'specific_variance': specific_var,
        'specific_volatility': np.sqrt(specific_var)
    })
    specific_risk_df.to_csv('models/risk/specific_risk.csv', index=False)
    print("  ✓ 特质风险保存至: models/risk/specific_risk.csv")

    # 分析结果
    print("\n" + "="*60)
    print("风险分析统计")
    print("="*60)

    print(f"\n因子暴露统计:")
    print(factor_exposure_df.describe())

    print(f"\n特质风险统计:")
    print(f"  均值: {specific_var.mean():.6f}")
    print(f"  标准差: {specific_var.std():.6f}")
    print(f"  最小值: {specific_var.min():.6f}")
    print(f"  最大值: {specific_var.max():.6f}")

    # 计算因子方差贡献
    total_var = np.trace(factor_cov) + specific_var.sum()
    systematic_var = np.trace(factor_cov)
    specific_total_var = specific_var.sum()

    print(f"\n风险分解:")
    print(f"  系统性风险占比: {systematic_var/total_var*100:.2f}%")
    print(f"  特质风险占比: {specific_total_var/total_var*100:.2f}%")

    print("\n" + "="*60)
    print("✓ 风险模型训练完成!")
    print("="*60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练深度风险模型')
    parser.add_argument('--n_factors', type=int, default=10,
                       help='隐性风险因子数量')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[64, 32],
                       help='隐藏层维度')
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='学习率')

    args = parser.parse_args()

    if not RISK_MODEL_AVAILABLE:
        print("错误: PyTorch未安装，无法训练风险模型")
        print("请安装: pip install torch")
        sys.exit(1)

    train_risk_model(
        n_factors=args.n_factors,
        hidden_dims=args.hidden_dims,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )


if __name__ == '__main__':
    main()
