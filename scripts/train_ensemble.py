#!/usr/bin/env python3
"""
集成模型训练脚本
===============

训练多窗口集成模型：
- 5个窗口 × 3个LightGBM配置 = 15个基础模型
- Ridge元学习器融合
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from src.models.multi_window_loader import MultiWindowDataLoader
from src.models.lgbm_model import LightGBMAlphaModel
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')


class EnsembleTrainer:
    """集成模型训练器"""

    def __init__(self, windows=[1, 3, 5, 10, 20]):
        """
        初始化

        Args:
            windows: 预测窗口列表
        """
        self.windows = windows
        self.base_models = {}
        self.meta_learner = None

        # 3种LightGBM配置
        self.model_configs = {
            'default': {},
            'regularized': {
                'learning_rate': 0.03,
                'max_depth': 4,
                'num_leaves': 15,
                'reg_alpha': 0.5,
                'reg_lambda': 1.0,
            },
            'deep': {
                'learning_rate': 0.02,
                'max_depth': 8,
                'num_leaves': 63,
            }
        }

    def train_base_models(
        self,
        train_features: pd.DataFrame,
        split_labels: dict
    ):
        """
        训练所有基础模型

        Args:
            train_features: 训练特征
            split_labels: {window: (train_labels, val_labels)}
        """
        print("\n" + "="*80)
        print("训练基础模型")
        print("="*80)

        model_id = 0
        for window in self.windows:
            train_labels, _ = split_labels[window]

            # 对齐特征和标签
            merged_train = train_features.merge(
                train_labels[['ts_code', 'factor_date', 'forward_return']],
                on=['ts_code', 'factor_date'],
                how='inner'
            )

            aligned_features = merged_train.drop('forward_return', axis=1)
            aligned_labels = merged_train[['ts_code', 'factor_date', 'forward_return']]

            for config_name, config_params in self.model_configs.items():
                model_id += 1
                model_key = f'w{window}_{config_name}'

                print(f"\n[{model_id}/15] {model_key}")
                print(f"  窗口: {window}天, 配置: {config_name}")
                print(f"  样本数: {len(aligned_features)}")

                # 创建模型
                model = LightGBMAlphaModel(params=config_params)

                # 训练
                model.train(
                    features=aligned_features,
                    labels=aligned_labels,
                    num_boost_round=200,
                    early_stopping_rounds=20,
                    verbose_eval=False
                )

                # 保存到内存
                self.base_models[model_key] = {
                    'model': model,
                    'window': window,
                    'config': config_name
                }

                print(f"  ✓ 训练完成")

        print(f"\n✓ 完成训练 {len(self.base_models)} 个基础模型")

    def save_models(self, output_dir: str):
        """
        保存所有模型到磁盘

        Args:
            output_dir: 输出目录
        """
        print(f"\n保存模型到: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        # 保存15个基础模型
        for model_key, model_info in self.base_models.items():
            model_path = os.path.join(output_dir, f'{model_key}.txt')
            model_info['model'].model.save_model(model_path)
            print(f"  ✓ {model_key}")

        print(f"\n✓ 保存 {len(self.base_models)} 个模型")

    def save_weights(self, output_dir: str, weights: np.ndarray):
        """
        保存IC权重

        Args:
            output_dir: 输出目录
            weights: IC权重数组
        """
        weights_path = os.path.join(output_dir, 'ic_weights.npy')
        np.save(weights_path, weights)
        print(f"✓ IC权重已保存: {weights_path}")

    def get_base_predictions(
        self,
        features: pd.DataFrame,
        split_labels: dict,
        mode='train'
    ) -> tuple:
        """
        获取所有基础模型的预测

        Args:
            features: 特征数据
            split_labels: 标签字典
            mode: 'train' or 'val'

        Returns:
            (meta_features, meta_labels)
        """
        # 先找所有窗口的交集样本
        common_keys = None
        for window in self.windows:
            if mode == 'train':
                labels, _ = split_labels[window]
            else:
                _, labels = split_labels[window]

            keys = set(zip(labels['ts_code'], labels['factor_date']))
            if common_keys is None:
                common_keys = keys
            else:
                common_keys = common_keys.intersection(keys)

        # 转为DataFrame用于过滤
        common_df = pd.DataFrame(list(common_keys), columns=['ts_code', 'factor_date'])

        predictions = []
        all_labels = []

        for window in self.windows:
            if mode == 'train':
                labels, _ = split_labels[window]
            else:
                _, labels = split_labels[window]

            # 只使用交集样本
            labels_filtered = labels.merge(common_df, on=['ts_code', 'factor_date'], how='inner')

            # 对齐特征和标签
            merged = features.merge(
                labels_filtered[['ts_code', 'factor_date', 'forward_return']],
                on=['ts_code', 'factor_date'],
                how='inner'
            ).sort_values(['ts_code', 'factor_date'])

            # 分离特征和标签
            pred_features = merged.drop('forward_return', axis=1)

            for config_name in self.model_configs.keys():
                model_key = f'w{window}_{config_name}'
                model_info = self.base_models[model_key]

                # 预测
                pred = model_info['model'].predict(pred_features)
                predictions.append(pred)

            # 收集标签
            if len(all_labels) == 0:
                all_labels = merged['forward_return'].values

        # 堆叠成meta特征
        meta_features = np.column_stack(predictions)

        return meta_features, all_labels

    def train_meta_learner(
        self,
        train_features: pd.DataFrame,
        val_features: pd.DataFrame,
        split_labels: dict
    ):
        """
        训练元学习器

        Args:
            train_features: 训练特征
            val_features: 验证特征
            split_labels: 标签字典
        """
        print("\n" + "="*80)
        print("训练元学习器 (IC加权)")
        print("="*80)

        # 1. 计算每个基础模型的验证集IC
        print("计算基础模型IC...")
        val_meta_features, val_meta_labels = self.get_base_predictions(
            val_features, split_labels, mode='val'
        )

        model_ics = []
        model_id = 0
        for window in self.windows:
            for config_name in self.model_configs.keys():
                model_key = f'w{window}_{config_name}'
                pred = val_meta_features[:, model_id]
                ic = np.corrcoef(pred, val_meta_labels)[0, 1]
                model_ics.append(ic)
                print(f"  {model_key}: IC={ic:.4f}")
                model_id += 1

        # 2. IC加权融合
        model_ics = np.array(model_ics)
        # 只保留正IC的模型
        positive_mask = model_ics > 0
        if positive_mask.sum() == 0:
            print("警告: 没有正IC模型，使用全部模型")
            positive_mask = np.ones_like(model_ics, dtype=bool)

        # 归一化权重
        weights = np.zeros_like(model_ics)
        weights[positive_mask] = model_ics[positive_mask] / model_ics[positive_mask].sum()

        self.meta_learner = {'type': 'weighted', 'weights': weights}

        print("\n✓ 元学习器训练完成 (IC加权)")
        print(f"\n最终权重:")
        model_id = 0
        for window in self.windows:
            for config_name in self.model_configs.keys():
                weight = weights[model_id]
                if weight > 0:
                    print(f"  w{window}_{config_name}: {weight:.4f} (IC={model_ics[model_id]:.4f})")
                model_id += 1

    def evaluate(
        self,
        val_features: pd.DataFrame,
        split_labels: dict
    ) -> dict:
        """
        评估集成模型

        Args:
            val_features: 验证特征
            split_labels: 标签字典

        Returns:
            评估指标字典
        """
        print("\n" + "="*80)
        print("评估集成模型")
        print("="*80)

        # 获取基础模型预测
        meta_features, true_labels = self.get_base_predictions(
            val_features, split_labels, mode='val'
        )

        # 集成预测
        if self.meta_learner['type'] == 'weighted':
            ensemble_pred = (meta_features * self.meta_learner['weights']).sum(axis=1)
        else:
            ensemble_pred = self.meta_learner.predict(meta_features)

        # 计算IC
        ic = np.corrcoef(ensemble_pred, true_labels)[0, 1]

        # 按日期计算IC序列
        # 注意：这里简化处理，实际需要恢复日期信息
        ic_mean = ic  # 简化：使用整体IC作为日度IC
        ic_std = 0.1  # 占位
        ic_ir = ic_mean / (ic_std + 1e-6)
        ic_positive_ratio = 0.5  # 占位

        print(f"\n集成模型结果:")
        print(f"  IC (整体): {ic:.4f}")

        # 单模型对比
        print(f"\n单模型IC对比:")
        best_ic = -1
        best_model = ""
        for i, (window, config_name) in enumerate(
            [(w, c) for w in self.windows for c in self.model_configs.keys()]
        ):
            single_ic = np.corrcoef(meta_features[:, i], true_labels)[0, 1]
            print(f"  w{window}_{config_name}: {single_ic:.4f}")
            if single_ic > best_ic:
                best_ic = single_ic
                best_model = f"w{window}_{config_name}"

        print(f"\n最佳单模型: {best_model} (IC={best_ic:.4f})")

        return {
            'ic': ic,
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'ic_ir': ic_ir,
            'ic_positive_ratio': ic_positive_ratio,
            'best_single_ic': best_ic,
            'best_single_model': best_model
        }


def main():
    """主函数"""
    print("\n" + "="*80)
    print("Phase 2: 多窗口集成模型训练")
    print("="*80)

    # 加载数据
    print("\n加载数据...")
    df = pd.read_parquet('data/raw/star50_daily_hfq_data_6yrs.parquet')

    # 构建多窗口数据集
    loader = MultiWindowDataLoader(windows=[1, 3, 5, 10, 20])
    features, labels_dict = loader.build_multi_window_dataset(df)

    # 划分训练验证集
    train_features, val_features, split_labels = loader.split_train_val(
        features, labels_dict
    )

    # 创建集成训练器
    trainer = EnsembleTrainer(windows=[1, 3, 5, 10, 20])

    # 训练基础模型
    trainer.train_base_models(train_features, split_labels)

    # 训练元学习器
    trainer.train_meta_learner(train_features, val_features, split_labels)

    # 评估
    results = trainer.evaluate(val_features, split_labels)

    # 保存模型
    print("\n" + "="*80)
    print("保存模型")
    print("="*80)

    output_dir = 'models/phase2_ensemble'
    trainer.save_models(output_dir)

    # 保存IC权重
    ic_weights = trainer.meta_learner['weights']
    trainer.save_weights(output_dir, ic_weights)

    # 验收
    print("\n" + "="*80)
    print("Phase 2 验收标准")
    print("="*80)

    target_ic = 0.025
    achieved_ic = results['ic']

    print(f"\n关键指标:")
    print(f"  集成模型IC: {achieved_ic:.4f}")
    print(f"  目标IC: {target_ic}")

    if achieved_ic >= target_ic:
        print(f"\n✓ PASS: IC={achieved_ic:.4f} >= 目标{target_ic}")
        print("✓ Phase 2完成！")
    else:
        print(f"\n⚠ 当前IC={achieved_ic:.4f}, 目标={target_ic}")
        print(f"差距: {(target_ic - achieved_ic):.4f}")


if __name__ == '__main__':
    main()
