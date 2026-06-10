"""
LSTM Alpha预测模型
==================

使用LSTM捕捉时序特征，预测股票未来超额收益率。

特点：
- 时序记忆性
- 多步历史输入
- 非线性模式识别
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class StockSequenceDataset(Dataset):
    """股票时序数据集"""

    def __init__(
        self,
        features: pd.DataFrame,
        labels: pd.DataFrame,
        lookback_days: int = 20
    ):
        """
        初始化数据集

        Args:
            features: 特征DataFrame
            labels: 标签DataFrame
            lookback_days: 回看天数
        """
        self.lookback_days = lookback_days
        self.sequences = []
        self.targets = []

        # 提取特征列
        feature_cols = [col for col in features.columns
                       if col not in ['ts_code', 'factor_date']]

        # 按股票分组构建序列
        for ts_code, stock_features in features.groupby('ts_code'):
            stock_features = stock_features.sort_values('factor_date')
            stock_labels = labels[labels['ts_code'] == ts_code].sort_values('factor_date')

            # 合并
            merged = stock_features.merge(
                stock_labels[['factor_date', 'forward_return']],
                on='factor_date',
                how='inner'
            )

            if len(merged) < lookback_days:
                continue

            # 构建滚动窗口
            X = merged[feature_cols].values
            y = merged['forward_return'].values

            for i in range(lookback_days, len(merged)):
                seq = X[i - lookback_days:i]  # [lookback_days, n_features]
                target = y[i]

                self.sequences.append(seq)
                self.targets.append(target)

        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class LSTMAlphaModel(nn.Module):
    """
    LSTM Alpha预测模型

    架构：
    - LSTM层捕捉时序特征
    - Dropout防止过拟合
    - 全连接层输出预测
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        """
        初始化模型

        Args:
            input_size: 输入特征数
            hidden_size: 隐藏层大小
            num_layers: LSTM层数
            dropout: Dropout比例
        """
        super(LSTMAlphaModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 全连接层
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        前向传播

        Args:
            x: [batch_size, seq_len, input_size]

        Returns:
            [batch_size, 1] 预测值
        """
        # LSTM
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, hidden_size]

        # 取最后一个时间步
        last_out = lstm_out[:, -1, :]  # [batch_size, hidden_size]

        # Dropout
        out = self.dropout(last_out)

        # 全连接
        out = self.fc(out)  # [batch_size, 1]

        return out.squeeze(1)


class LSTMAlphaTrainer:
    """LSTM训练器"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        device: str = 'cpu'
    ):
        """
        初始化训练器

        Args:
            input_size: 输入特征数
            hidden_size: 隐藏层大小
            num_layers: LSTM层数
            dropout: Dropout比例
            learning_rate: 学习率
            device: 设备（cpu/cuda）
        """
        self.device = torch.device(device)

        # 创建模型
        self.model = LSTMAlphaModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        ).to(self.device)

        # 损失函数和优化器
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 50,
        early_stopping_patience: int = 5
    ):
        """
        训练模型

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮数
            early_stopping_patience: 早停耐心值
        """
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # 前向传播
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # 验证阶段
            if val_loader:
                val_loss, val_ic = self.evaluate(val_loader)
                print(f"Epoch {epoch+1}/{num_epochs} - "
                      f"Train Loss: {train_loss:.6f}, "
                      f"Val Loss: {val_loss:.6f}, "
                      f"Val IC: {val_ic:.4f}")

                # 早停检查
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f}")

    def evaluate(self, data_loader: DataLoader) -> Tuple[float, float]:
        """
        评估模型

        Args:
            data_loader: 数据加载器

        Returns:
            (loss, IC)
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)

                total_loss += loss.item()
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())

        avg_loss = total_loss / len(data_loader)

        # 计算IC（信息系数）
        ic = np.corrcoef(all_preds, all_targets)[0, 1]

        return avg_loss, ic

    def predict(self, data_loader: DataLoader) -> np.ndarray:
        """
        预测

        Args:
            data_loader: 数据加载器

        Returns:
            预测值数组
        """
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch_x, _ in data_loader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x)
                predictions.extend(outputs.cpu().numpy())

        return np.array(predictions)

    def save(self, filepath: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {filepath}")
