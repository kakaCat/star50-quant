"""
测试MoE模型
===========

测试Neural MoE模型的各个组件。
"""

import pytest
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.models.alpha.moe_model import MoEDataset, NeuralMoE, MoEAlphaTrainer


class TestMoEDataset:
    """测试MoE数据集"""

    def test_dataset_creation(self):
        """测试数据集创建"""
        # 创建模拟数据
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        stocks = ['stock_1', 'stock_2', 'stock_3']

        # 个股特征
        stock_data = []
        for stock in stocks:
            for date in dates:
                stock_data.append({
                    'ts_code': stock,
                    'factor_date': date,
                    'feature_1': np.random.randn(),
                    'feature_2': np.random.randn()
                })
        stock_features = pd.DataFrame(stock_data)

        # 环境特征
        regime_data = []
        for date in dates:
            regime_data.append({
                'factor_date': date,
                'regime_1': np.random.randn(),
                'regime_2': np.random.randn()
            })
        regime_features = pd.DataFrame(regime_data)

        # 标签
        label_data = []
        for stock in stocks:
            for date in dates:
                label_data.append({
                    'ts_code': stock,
                    'factor_date': date,
                    'residual_return': np.random.randn() * 0.01
                })
        labels = pd.DataFrame(label_data)

        # 创建数据集
        dataset = MoEDataset(stock_features, regime_features, labels)

        # 验证
        assert len(dataset) == len(stocks) * len(dates)
        assert dataset.stock_features.shape[1] == 2
        assert dataset.regime_features.shape[1] == 2

    def test_dataset_getitem(self):
        """测试数据集索引"""
        # 简单数据
        stock_features = pd.DataFrame({
            'ts_code': ['A', 'B'],
            'factor_date': pd.to_datetime(['2020-01-01', '2020-01-01']),
            'f1': [1.0, 2.0]
        })
        regime_features = pd.DataFrame({
            'factor_date': pd.to_datetime(['2020-01-01']),
            'r1': [0.5]
        })
        labels = pd.DataFrame({
            'ts_code': ['A', 'B'],
            'factor_date': pd.to_datetime(['2020-01-01', '2020-01-01']),
            'residual_return': [0.01, -0.01]
        })

        dataset = MoEDataset(stock_features, regime_features, labels)

        stock_x, regime_x, y = dataset[0]
        assert isinstance(stock_x, np.ndarray)
        assert isinstance(regime_x, np.ndarray)
        assert isinstance(y, np.float32)


class TestNeuralMoE:
    """测试Neural MoE模型"""

    def test_model_forward(self):
        """测试模型前向传播"""
        stock_dim = 10
        regime_dim = 5
        batch_size = 32

        model = NeuralMoE(stock_dim, regime_dim)

        # 输入
        stock_x = torch.randn(batch_size, stock_dim)
        regime_x = torch.randn(batch_size, regime_dim)

        # 前向传播
        output = model(stock_x, regime_x)

        # 验证输出形状
        assert output.shape == (batch_size,)

    def test_model_components(self):
        """测试模型组件存在性"""
        model = NeuralMoE(stock_dim=10, regime_dim=5)

        # 验证组件存在
        assert hasattr(model, 'stock_expert')
        assert hasattr(model, 'regime_expert')
        assert hasattr(model, 'gating')

    def test_model_parameters(self):
        """测试模型参数可训练"""
        model = NeuralMoE(stock_dim=10, regime_dim=5)

        # 检查参数
        params = list(model.parameters())
        assert len(params) > 0

        # 检查所有参数需要梯度
        for param in params:
            assert param.requires_grad


class TestMoEAlphaTrainer:
    """测试MoE训练器"""

    def test_trainer_initialization(self):
        """测试训练器初始化"""
        trainer = MoEAlphaTrainer(
            stock_dim=10,
            regime_dim=5,
            device='cpu'
        )

        assert trainer.device.type == 'cpu'
        assert trainer.model is not None
        assert trainer.optimizer is not None

    def test_trainer_train(self):
        """测试训练过程"""
        # 创建模拟数据
        stock_features = pd.DataFrame({
            'ts_code': ['A'] * 100,
            'factor_date': pd.date_range('2020-01-01', periods=100),
            'f1': np.random.randn(100),
            'f2': np.random.randn(100)
        })
        regime_features = pd.DataFrame({
            'factor_date': pd.date_range('2020-01-01', periods=100),
            'r1': np.random.randn(100)
        })
        labels = pd.DataFrame({
            'ts_code': ['A'] * 100,
            'factor_date': pd.date_range('2020-01-01', periods=100),
            'residual_return': np.random.randn(100) * 0.01
        })

        dataset = MoEDataset(stock_features, regime_features, labels)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)

        # 训练
        trainer = MoEAlphaTrainer(stock_dim=2, regime_dim=1, device='cpu')
        trainer.train(loader, num_epochs=2, verbose=False)

        # 验证模型可以预测
        predictions = trainer.predict(
            stock_features[['f1', 'f2']].values,
            regime_features[['r1']].values
        )
        assert len(predictions) == 100

    def test_trainer_evaluate(self):
        """测试评估函数"""
        # 创建模拟数据
        stock_features = pd.DataFrame({
            'ts_code': ['A'] * 50,
            'factor_date': pd.date_range('2020-01-01', periods=50),
            'f1': np.random.randn(50)
        })
        regime_features = pd.DataFrame({
            'factor_date': pd.date_range('2020-01-01', periods=50),
            'r1': np.random.randn(50)
        })
        labels = pd.DataFrame({
            'ts_code': ['A'] * 50,
            'factor_date': pd.date_range('2020-01-01', periods=50),
            'residual_return': np.random.randn(50) * 0.01
        })

        dataset = MoEDataset(stock_features, regime_features, labels)
        loader = DataLoader(dataset, batch_size=16)

        trainer = MoEAlphaTrainer(stock_dim=1, regime_dim=1, device='cpu')
        loss, ic, icir = trainer.evaluate(loader)

        # 验证返回值类型
        assert isinstance(loss, float)
        assert isinstance(ic, float)
        assert isinstance(icir, float)

    def test_trainer_save_load(self, tmp_path):
        """测试模型保存和加载"""
        trainer = MoEAlphaTrainer(stock_dim=5, regime_dim=3, device='cpu')

        # 保存
        model_path = tmp_path / "test_model.pth"
        trainer.save(str(model_path))
        assert model_path.exists()

        # 加载
        new_trainer = MoEAlphaTrainer(stock_dim=5, regime_dim=3, device='cpu')
        new_trainer.load(str(model_path))

        # 验证参数一致
        for p1, p2 in zip(trainer.model.parameters(), new_trainer.model.parameters()):
            assert torch.allclose(p1, p2)


class TestIntegration:
    """集成测试"""

    def test_end_to_end(self):
        """端到端测试：数据->训练->预测"""
        # 1. 创建数据（确保日期对齐）
        dates = pd.date_range('2020-01-01', periods=100)
        stocks = ['A', 'B']

        stock_data = []
        for stock in stocks:
            for date in dates:
                stock_data.append({
                    'ts_code': stock,
                    'factor_date': date,
                    'momentum': np.random.randn(),
                    'volatility': np.random.randn()
                })
        stock_features = pd.DataFrame(stock_data)

        regime_features = pd.DataFrame({
            'factor_date': dates,
            'market_vol': np.random.randn(len(dates))
        })

        label_data = []
        for stock in stocks:
            for date in dates:
                label_data.append({
                    'ts_code': stock,
                    'factor_date': date,
                    'residual_return': np.random.randn() * 0.01
                })
        labels = pd.DataFrame(label_data)

        # 2. 分割训练/测试（按日期）
        train_dates = dates[:80]
        test_dates = dates[80:]

        train_stock = stock_features[stock_features['factor_date'].isin(train_dates)]
        train_regime = regime_features[regime_features['factor_date'].isin(train_dates)]
        train_labels = labels[labels['factor_date'].isin(train_dates)]

        test_stock = stock_features[stock_features['factor_date'].isin(test_dates)]
        test_regime = regime_features[regime_features['factor_date'].isin(test_dates)]

        # 3. 训练
        train_dataset = MoEDataset(train_stock, train_regime, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=32)

        trainer = MoEAlphaTrainer(stock_dim=2, regime_dim=1, device='cpu')
        trainer.train(train_loader, num_epochs=3, verbose=False)

        # 4. 预测（使用合并后的数据确保对齐）
        test_merged = test_stock.merge(test_regime, on='factor_date', how='inner')
        test_stock_x = test_merged[['momentum', 'volatility']].values
        test_regime_x = test_merged[['market_vol']].values
        predictions = trainer.predict(test_stock_x, test_regime_x)

        # 5. 验证
        assert len(predictions) == len(test_merged)
        assert not np.isnan(predictions).all()  # 至少有一些非NaN值
