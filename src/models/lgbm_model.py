"""
LightGBM Alpha预测模型
======================

使用LightGBM作为baseline模型预测股票未来超额收益率。

特点：
- 非线性因子交叉效应
- 特征重要性分析
- 快速训练
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
import pickle


class LightGBMAlphaModel:
    """
    LightGBM Alpha预测模型

    预测股票未来收益率，作为Alpha信号。
    """

    def __init__(self, params: Optional[Dict] = None):
        """
        初始化模型

        Args:
            params: LightGBM参数
        """
        self.params = params or {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }

        self.model = None
        self.feature_names = None
        self.feature_importance = None

    def prepare_data(
        self,
        features: pd.DataFrame,
        labels: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备训练数据

        Args:
            features: 特征DataFrame
            labels: 标签DataFrame

        Returns:
            (X, y) numpy arrays
        """
        # 提取特征列
        feature_cols = [col for col in features.columns
                       if col not in ['ts_code', 'factor_date']]

        self.feature_names = feature_cols

        X = features[feature_cols].values
        y = labels['forward_return'].values

        return X, y

    def train(
        self,
        features: pd.DataFrame,
        labels: pd.DataFrame,
        num_boost_round: int = 100,
        early_stopping_rounds: int = 10,
        verbose_eval: int = 10
    ):
        """
        训练模型

        Args:
            features: 特征DataFrame
            labels: 标签DataFrame
            num_boost_round: 提升轮数
            early_stopping_rounds: 早停轮数
            verbose_eval: 日志间隔
        """
        print("Preparing training data...")
        X, y = self.prepare_data(features, labels)

        print(f"Training LightGBM with {X.shape[0]} samples, {X.shape[1]} features...")

        # 创建Dataset
        train_data = lgb.Dataset(X, label=y, feature_name=self.feature_names)

        # 训练模型
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[train_data],
            valid_names=['train'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                lgb.log_evaluation(period=verbose_eval)
            ]
        )

        # 保存特征重要性
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)

        print("Training completed!")
        print("\nTop 10 important features:")
        print(self.feature_importance.head(10))

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        预测

        Args:
            features: 特征DataFrame

        Returns:
            预测的收益率数组
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")

        feature_cols = [col for col in features.columns
                       if col not in ['ts_code', 'factor_date']]

        X = features[feature_cols].values
        predictions = self.model.predict(X)

        return predictions

    def cross_validate(
        self,
        features: pd.DataFrame,
        labels: pd.DataFrame,
        n_splits: int = 5
    ) -> Dict[str, float]:
        """
        时间序列交叉验证

        Args:
            features: 特征DataFrame
            labels: 标签DataFrame
            n_splits: 折数

        Returns:
            评估指标字典
        """
        print(f"Cross-validating with {n_splits} splits...")

        X, y = self.prepare_data(features, labels)

        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            print(f"\nFold {fold}/{n_splits}")

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # 训练
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

            model = lgb.train(
                self.params,
                train_data,
                num_boost_round=100,
                valid_sets=[val_data],
                valid_names=['valid'],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=10),
                    lgb.log_evaluation(period=0)
                ]
            )

            # 预测
            y_pred = model.predict(X_val)

            # 计算指标
            mse = np.mean((y_pred - y_val) ** 2)
            rmse = np.sqrt(mse)
            ic = np.corrcoef(y_pred, y_val)[0, 1]

            scores.append({
                'rmse': rmse,
                'ic': ic
            })

            print(f"  RMSE: {rmse:.6f}, IC: {ic:.4f}")

        # 平均指标
        avg_scores = {
            'rmse_mean': np.mean([s['rmse'] for s in scores]),
            'rmse_std': np.std([s['rmse'] for s in scores]),
            'ic_mean': np.mean([s['ic'] for s in scores]),
            'ic_std': np.std([s['ic'] for s in scores]),
        }

        print(f"\nCross-validation results:")
        print(f"  RMSE: {avg_scores['rmse_mean']:.6f} ± {avg_scores['rmse_std']:.6f}")
        print(f"  IC: {avg_scores['ic_mean']:.4f} ± {avg_scores['ic_std']:.4f}")

        return avg_scores

    def save(self, filepath: str):
        """保存模型"""
        if self.model is None:
            raise ValueError("Model not trained yet!")

        self.model.save_model(filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """加载模型"""
        self.model = lgb.Booster(model_file=filepath)
        print(f"Model loaded from {filepath}")

    def get_feature_importance(self) -> pd.DataFrame:
        """获取特征重要性"""
        if self.feature_importance is None:
            raise ValueError("Model not trained yet!")

        return self.feature_importance
