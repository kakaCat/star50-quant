"""
XGBoost Alpha预测模型
=====================

使用XGBoost预测股票未来超额收益率。

特点：
- 强大的梯度提升框架
- 内置正则化（L1/L2）
- 支持自定义损失函数
- 特征重要性分析
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
import pickle


class XGBoostAlphaModel:
    """
    XGBoost Alpha预测模型

    预测股票未来收益率，作为Alpha信号。
    """

    def __init__(self, params: Optional[Dict] = None):
        """
        初始化模型

        Args:
            params: XGBoost参数
        """
        self.params = params or {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'booster': 'gbtree',
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'colsample_bylevel': 0.8,
            'min_child_weight': 1,
            'gamma': 0.0,
            'reg_alpha': 0.0,  # L1正则化
            'reg_lambda': 1.0,  # L2正则化
            'seed': 42,
            'n_jobs': -1,
            'verbosity': 0
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
        verbose_eval: int = 10,
        val_features: Optional[pd.DataFrame] = None,
        val_labels: Optional[pd.DataFrame] = None
    ):
        """
        训练模型

        Args:
            features: 训练特征DataFrame
            labels: 训练标签DataFrame
            num_boost_round: 提升轮数
            early_stopping_rounds: 早停轮数
            verbose_eval: 日志间隔
            val_features: 验证集特征
            val_labels: 验证集标签
        """
        print("Preparing training data...")
        X_train, y_train = self.prepare_data(features, labels)

        print(f"Training XGBoost with {X_train.shape[0]} samples, {X_train.shape[1]} features...")

        # 创建DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)

        # 设置验证集
        evals = [(dtrain, 'train')]
        if val_features is not None and val_labels is not None:
            X_val, y_val = self.prepare_data(val_features, val_labels)
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_names)
            evals.append((dval, 'valid'))

        # 训练模型
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval
        )

        # 保存特征重要性
        importance_dict = self.model.get_score(importance_type='gain')
        self.feature_importance = pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in importance_dict.items()
        ]).sort_values('importance', ascending=False)

        print("Training completed!")
        print(f"\nBest iteration: {self.model.best_iteration}")
        print(f"Best score: {self.model.best_score:.6f}")
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

        X = features[feature_cols]  # 保留DataFrame，不用.values
        dmatrix = xgb.DMatrix(X, feature_names=self.feature_names)
        predictions = self.model.predict(dmatrix)

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
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)

            model = xgb.train(
                self.params,
                dtrain,
                num_boost_round=100,
                evals=[(dval, 'valid')],
                early_stopping_rounds=10,
                verbose_eval=0
            )

            # 预测
            y_pred = model.predict(dval)

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
        self.model = xgb.Booster()
        self.model.load_model(filepath)
        print(f"Model loaded from {filepath}")

    def get_feature_importance(self) -> pd.DataFrame:
        """获取特征重要性"""
        if self.feature_importance is None:
            raise ValueError("Model not trained yet!")

        return self.feature_importance
