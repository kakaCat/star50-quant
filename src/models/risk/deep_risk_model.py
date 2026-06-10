"""
深度风险模型模块
================

使用自编码器提取隐性风险因子。

核心思想：
1. Encoder: 将个股收益率压缩到低维隐性风险因子空间
2. Decoder: 从隐性因子重构个股收益率
3. 风险暴露: Encoder输出的隐含层即为股票的Beta
4. 特质风险: 重构误差的协方差矩阵
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import logging

# 尝试导入PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Risk model requires PyTorch.")


if TORCH_AVAILABLE:
    class RiskAutoencoder(nn.Module):
        """
        风险自编码器

        架构:
        - Encoder: 股票收益率 -> 隐性风险因子
        - Decoder: 隐性风险因子 -> 重构收益率
        """

        def __init__(
            self,
            input_dim: int,
            latent_dim: int = 10,
            hidden_dims: list = [64, 32]
        ):
            """
            初始化自编码器

            Args:
                input_dim: 输入维度（股票数量）
                latent_dim: 隐性风险因子数量
                hidden_dims: 隐藏层维度列表
            """
            super(RiskAutoencoder, self).__init__()

            self.input_dim = input_dim
            self.latent_dim = latent_dim

            # Encoder
            encoder_layers = []
            prev_dim = input_dim

            for hidden_dim in hidden_dims:
                encoder_layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                ])
                prev_dim = hidden_dim

            # 最后映射到隐性因子
            encoder_layers.append(nn.Linear(prev_dim, latent_dim))

            self.encoder = nn.Sequential(*encoder_layers)

            # Decoder
            decoder_layers = []
            prev_dim = latent_dim

            for hidden_dim in reversed(hidden_dims):
                decoder_layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                ])
                prev_dim = hidden_dim

            # 重构到原始维度
            decoder_layers.append(nn.Linear(prev_dim, input_dim))

            self.decoder = nn.Sequential(*decoder_layers)

        def encode(self, x):
            """提取隐性风险因子（Beta）"""
            return self.encoder(x)

        def decode(self, z):
            """从隐性因子重构收益率"""
            return self.decoder(z)

        def forward(self, x):
            """前向传播"""
            z = self.encode(x)
            x_reconstructed = self.decode(z)
            return x_reconstructed, z


    class DeepRiskModel:
        """
        深度风险模型

        使用自编码器学习隐性风险因子，并计算：
        1. 风险因子暴露 (Beta)
        2. 特质风险协方差矩阵
        3. 系统性风险
        """

        def __init__(
            self,
            n_stocks: int,
            n_factors: int = 10,
            hidden_dims: list = [64, 32],
            learning_rate: float = 0.001,
            device: str = 'cpu'
        ):
            """
            初始化风险模型

            Args:
                n_stocks: 股票数量
                n_factors: 隐性风险因子数量
                hidden_dims: 隐藏层维度
                learning_rate: 学习率
                device: 设备
            """
            self.n_stocks = n_stocks
            self.n_factors = n_factors
            self.device = torch.device(device)

            # 创建自编码器
            self.model = RiskAutoencoder(
                input_dim=n_stocks,
                latent_dim=n_factors,
                hidden_dims=hidden_dims
            ).to(self.device)

            # 优化器
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate
            )

            # 损失函数
            self.criterion = nn.MSELoss()

            self.logger = logging.getLogger(__name__)

        def train(
            self,
            returns_data: np.ndarray,
            num_epochs: int = 100,
            batch_size: int = 32,
            validation_split: float = 0.2,
            early_stopping_patience: int = 10
        ):
            """
            训练风险模型

            Args:
                returns_data: 收益率矩阵 [n_days, n_stocks]
                num_epochs: 训练轮数
                batch_size: 批次大小
                validation_split: 验证集比例
                early_stopping_patience: 早停耐心值
            """
            n_samples = len(returns_data)
            n_val = int(n_samples * validation_split)
            n_train = n_samples - n_val

            # 划分训练集和验证集
            train_data = returns_data[:n_train]
            val_data = returns_data[n_train:]

            # 创建DataLoader
            train_tensor = torch.FloatTensor(train_data).to(self.device)
            val_tensor = torch.FloatTensor(val_data).to(self.device)

            train_dataset = TensorDataset(train_tensor)
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True
            )

            best_val_loss = float('inf')
            patience_counter = 0

            print(f"Training Deep Risk Model...")
            print(f"  Stocks: {self.n_stocks}")
            print(f"  Risk Factors: {self.n_factors}")
            print(f"  Training samples: {n_train}")
            print(f"  Validation samples: {n_val}")

            for epoch in range(num_epochs):
                # 训练
                self.model.train()
                train_loss = 0.0

                for batch in train_loader:
                    batch_x = batch[0]

                    # 前向传播
                    reconstructed, _ = self.model(batch_x)
                    loss = self.criterion(reconstructed, batch_x)

                    # 反向传播
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    train_loss += loss.item()

                train_loss /= len(train_loader)

                # 验证
                self.model.eval()
                with torch.no_grad():
                    val_reconstructed, _ = self.model(val_tensor)
                    val_loss = self.criterion(val_reconstructed, val_tensor).item()

                # 打印进度
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs} - "
                          f"Train Loss: {train_loss:.6f}, "
                          f"Val Loss: {val_loss:.6f}")

                # 早停
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

            print("Training completed!")

        def extract_risk_factors(
            self,
            returns_data: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray]:
            """
            提取风险因子暴露

            Args:
                returns_data: 收益率矩阵 [n_days, n_stocks]

            Returns:
                (factor_exposures, factor_returns)
                - factor_exposures: [n_stocks, n_factors] Beta矩阵
                - factor_returns: [n_days, n_factors] 风险因子收益率
            """
            self.model.eval()

            with torch.no_grad():
                returns_tensor = torch.FloatTensor(returns_data).to(self.device)
                _, latent = self.model(returns_tensor)
                factor_returns = latent.cpu().numpy()

            # 计算因子暴露（回归系数）
            # returns = factor_returns @ factor_exposures.T + residuals
            # 使用最小二乘法: factor_exposures = (F'F)^-1 F' R
            factor_exposures = np.linalg.lstsq(
                factor_returns,
                returns_data,
                rcond=None
            )[0].T  # [n_stocks, n_factors]

            return factor_exposures, factor_returns

        def compute_risk_matrix(
            self,
            returns_data: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray]:
            """
            计算风险协方差矩阵

            Args:
                returns_data: 收益率矩阵 [n_days, n_stocks]

            Returns:
                (factor_cov, specific_var)
                - factor_cov: [n_factors, n_factors] 因子协方差矩阵
                - specific_var: [n_stocks] 特质风险（对角矩阵的对角元素）
            """
            # 提取风险因子
            factor_exposures, factor_returns = self.extract_risk_factors(returns_data)

            # 计算因子协方差矩阵
            factor_cov = np.cov(factor_returns, rowvar=False)

            # 计算残差
            reconstructed_returns = factor_returns @ factor_exposures.T
            residuals = returns_data - reconstructed_returns

            # 计算特质风险（残差方差）
            specific_var = np.var(residuals, axis=0)

            return factor_cov, specific_var

        def predict_covariance(
            self,
            returns_data: np.ndarray
        ) -> np.ndarray:
            """
            预测收益率协方差矩阵

            Cov = B @ F @ B.T + D

            Args:
                returns_data: 收益率矩阵

            Returns:
                协方差矩阵 [n_stocks, n_stocks]
            """
            factor_exposures, _ = self.extract_risk_factors(returns_data)
            factor_cov, specific_var = self.compute_risk_matrix(returns_data)

            # 系统性风险部分: B @ F @ B.T
            systematic_cov = factor_exposures @ factor_cov @ factor_exposures.T

            # 特质风险部分: D (对角矩阵)
            specific_cov = np.diag(specific_var)

            # 总协方差矩阵
            total_cov = systematic_cov + specific_cov

            return total_cov

        def save(self, filepath: str):
            """保存模型"""
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'n_stocks': self.n_stocks,
                'n_factors': self.n_factors,
            }, filepath)
            print(f"Risk model saved to {filepath}")

        def load(self, filepath: str):
            """加载模型"""
            checkpoint = torch.load(filepath)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Risk model loaded from {filepath}")

else:
    # PyTorch不可用时的占位类
    class DeepRiskModel:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for DeepRiskModel. Please install: pip install torch")
