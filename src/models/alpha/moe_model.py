"""
Neural Mixture of Experts (MoE) Alpha预测模型
===============================================

核心创新：
- 分离个股特征和环境特征处理
- 多专家架构捕捉不同市场状态
- Beta剥离，预测真实Alpha（residual）

架构：
1. Stock Expert：处理个股量价特征
2. Regime Expert：处理市场环境特征
3. Gating Network：动态加权专家输出
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class MoEDataset(Dataset):
    """MoE数据集，支持个股特征和环境特征分离"""

    def __init__(
        self,
        stock_features: pd.DataFrame,
        regime_features: pd.DataFrame,
        labels: pd.DataFrame
    ):
        """
        初始化数据集

        Args:
            stock_features: 个股特征DataFrame (ts_code, factor_date, ...)
            regime_features: 环境特征DataFrame (factor_date, ...)
            labels: 标签DataFrame (ts_code, factor_date, residual_return)
        """
        # 合并数据
        merged = stock_features.merge(
            regime_features,
            on='factor_date',
            how='inner'
        ).merge(
            labels,
            on=['ts_code', 'factor_date'],
            how='inner'
        )

        # 提取特征列
        stock_cols = [col for col in stock_features.columns
                     if col not in ['ts_code', 'factor_date']]
        regime_cols = [col for col in regime_features.columns
                      if col != 'factor_date']

        # 转换为numpy数组
        self.stock_features = merged[stock_cols].values.astype(np.float32)
        self.regime_features = merged[regime_cols].values.astype(np.float32)
        self.labels = merged['residual_return'].values.astype(np.float32)

        # 存储元数据
        self.meta = merged[['ts_code', 'factor_date']].copy()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.stock_features[idx],
            self.regime_features[idx],
            self.labels[idx]
        )


class NeuralMoE(nn.Module):
    """
    Neural Mixture of Experts 模型

    架构：
    - Stock Expert: 3层MLP处理个股特征
    - Regime Expert: 2层MLP处理环境特征
    - Gating Network: 动态融合两个专家
    """

    def __init__(
        self,
        stock_dim: int,
        regime_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.3
    ):
        """
        初始化模型

        Args:
            stock_dim: 个股特征维度
            regime_dim: 环境特征维度
            hidden_dim: 隐藏层维度
            dropout: Dropout比例
        """
        super(NeuralMoE, self).__init__()

        # Stock Expert - 处理个股量价特征
        self.stock_expert = nn.Sequential(
            nn.Linear(stock_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Regime Expert - 处理市场环境特征
        self.regime_expert = nn.Sequential(
            nn.Linear(regime_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Gating Network - 动态融合
        self.gating = nn.Sequential(
            nn.Linear(stock_dim + regime_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, stock_x, regime_x):
        """
        前向传播

        Args:
            stock_x: [batch_size, stock_dim] 个股特征
            regime_x: [batch_size, regime_dim] 环境特征

        Returns:
            [batch_size, 1] 预测的Alpha
        """
        # 专家预测
        stock_pred = self.stock_expert(stock_x)  # [batch, 1]
        regime_pred = self.regime_expert(regime_x)  # [batch, 1]

        # 门控权重
        combined = torch.cat([stock_x, regime_x], dim=1)
        weights = self.gating(combined)  # [batch, 2]

        # 加权融合
        stock_weight = weights[:, 0:1]  # [batch, 1]
        regime_weight = weights[:, 1:2]  # [batch, 1]

        output = stock_weight * stock_pred + regime_weight * regime_pred

        return output.squeeze(1)


class MoEAlphaTrainer:
    """MoE训练器"""

    def __init__(
        self,
        stock_dim: int,
        regime_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.3,
        learning_rate: float = 0.005,
        weight_decay: float = 1e-4,
        device: str = 'cpu'
    ):
        """
        初始化训练器

        Args:
            stock_dim: 个股特征维度
            regime_dim: 环境特征维度
            hidden_dim: 隐藏层维度
            dropout: Dropout比例
            learning_rate: 学习率
            weight_decay: L2正则化系数
            device: 设备（cpu/cuda）
        """
        self.device = torch.device(device)

        # 创建模型
        self.model = NeuralMoE(
            stock_dim=stock_dim,
            regime_dim=regime_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        ).to(self.device)

        # 损失函数和优化器
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 25,
        verbose: bool = True
    ):
        """
        训练模型

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮数
            verbose: 是否打印训练信息
        """
        for epoch in range(num_epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0

            for batch_stock, batch_regime, batch_y in train_loader:
                batch_stock = batch_stock.to(self.device)
                batch_regime = batch_regime.to(self.device)
                batch_y = batch_y.to(self.device)

                # 前向传播
                outputs = self.model(batch_stock, batch_regime)
                loss = self.criterion(outputs, batch_y)

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # 验证阶段
            if val_loader and verbose:
                val_loss, val_ic, val_icir = self.evaluate(val_loader)
                print(f"Epoch {epoch+1}/{num_epochs} - "
                      f"Train Loss: {train_loss:.6f}, "
                      f"Val Loss: {val_loss:.6f}, "
                      f"Val IC: {val_ic:.4f}, "
                      f"Val ICIR: {val_icir:.4f}")
            elif verbose:
                print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f}")

    def evaluate(self, data_loader: DataLoader) -> Tuple[float, float, float]:
        """
        评估模型

        Args:
            data_loader: 数据加载器

        Returns:
            (loss, IC, ICIR)
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_stock, batch_regime, batch_y in data_loader:
                batch_stock = batch_stock.to(self.device)
                batch_regime = batch_regime.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.model(batch_stock, batch_regime)
                loss = self.criterion(outputs, batch_y)

                total_loss += loss.item()
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())

        avg_loss = total_loss / len(data_loader)

        # 计算Rank IC和ICIR
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        # 过滤NaN
        valid_mask = ~(np.isnan(all_preds) | np.isnan(all_targets))
        all_preds = all_preds[valid_mask]
        all_targets = all_targets[valid_mask]

        if len(all_preds) > 0:
            from scipy.stats import spearmanr
            rank_ic, _ = spearmanr(all_preds, all_targets)
            # 简化的ICIR计算（假设IC序列标准差）
            icir = rank_ic / 0.1  # 占位符，实际需要时序IC
        else:
            rank_ic = 0.0
            icir = 0.0

        return avg_loss, rank_ic, icir

    def predict(
        self,
        stock_features: np.ndarray,
        regime_features: np.ndarray
    ) -> np.ndarray:
        """
        预测Alpha

        Args:
            stock_features: [N, stock_dim]
            regime_features: [N, regime_dim]

        Returns:
            预测值 [N]
        """
        self.model.eval()

        stock_tensor = torch.tensor(stock_features, dtype=torch.float32).to(self.device)
        regime_tensor = torch.tensor(regime_features, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            predictions = self.model(stock_tensor, regime_tensor)

        return predictions.cpu().numpy()

    def save(self, filepath: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {filepath}")
