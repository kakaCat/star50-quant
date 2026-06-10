#!/usr/bin/env python3
"""
Phase 1验证脚本
===============

验证特征工程效果：
- 对比Baseline（30因子）vs 增强（160特征）
- 目标: IC提升到0.028-0.032
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.models.data_loader import FactorDataLoader
from src.models.lgbm_model import LightGBMAlphaModel


def train_and_evaluate(enable_features, label):
    """训练并评估模型"""
    print("\n" + "="*80)
    print(f"{label}")
    print("="*80)

    # 加载数据
    with FactorDataLoader() as loader:
        features, labels = loader.build_dataset(
            start_date='2020-01-01',
            end_date='2024-12-31',
            forward_days=5,
            enable_feature_engineering=enable_features,
            feature_config_path='../configs/feature_config.yaml'
        )

    # 时间序列划分
    split_date = features['factor_date'].quantile(0.8)
    train_features = features[features['factor_date'] <= split_date]
    train_labels = labels[labels['factor_date'] <= split_date]
    val_features = features[features['factor_date'] > split_date]
    val_labels = labels[labels['factor_date'] > split_date]

    print(f"\nTrain: {len(train_features)} samples")
    print(f"Val: {len(val_features)} samples")

    feature_cols = [c for c in features.columns if c not in ['ts_code', 'factor_date']]
    print(f"Features: {len(feature_cols)}")

    # 训练模型
    model = LightGBMAlphaModel()
    model.train(
        features=train_features,
        labels=train_labels,
        num_boost_round=200,
        early_stopping_rounds=20,
        verbose_eval=50
    )

    # 验证集评估
    val_pred = model.predict(val_features)
    val_true = val_labels['forward_return'].values

    # 计算IC
    ic = np.corrcoef(val_pred, val_true)[0, 1]

    # 按日期计算IC序列
    val_df = val_features.copy()
    val_df['pred'] = val_pred
    val_df['true'] = val_true

    ic_series = []
    for date in val_df['factor_date'].unique():
        mask = val_df['factor_date'] == date
        if mask.sum() >= 5:
            daily_ic = np.corrcoef(val_df.loc[mask, 'pred'], val_df.loc[mask, 'true'])[0, 1]
            if not np.isnan(daily_ic):
                ic_series.append(daily_ic)

    ic_mean = np.mean(ic_series)
    ic_std = np.std(ic_series)
    ic_ir = ic_mean / (ic_std + 1e-6)
    ic_positive_ratio = (np.array(ic_series) > 0).mean()

    print(f"\n{'='*80}")
    print(f"验证集结果:")
    print(f"  IC (整体): {ic:.4f}")
    print(f"  IC (日度均值): {ic_mean:.4f}")
    print(f"  IC (标准差): {ic_std:.4f}")
    print(f"  IR: {ic_ir:.4f}")
    print(f"  IC>0比例: {ic_positive_ratio:.2%}")
    print(f"{'='*80}\n")

    return {
        'ic': ic,
        'ic_mean': ic_mean,
        'ic_std': ic_std,
        'ic_ir': ic_ir,
        'ic_positive_ratio': ic_positive_ratio,
        'n_features': len(feature_cols)
    }


def main():
    """主函数"""
    print("\n" + "="*80)
    print("Phase 1 验证: 特征工程效果对比")
    print("="*80)

    # Baseline: 30因子
    baseline_results = train_and_evaluate(
        enable_features=False,
        label="Baseline: 30个原始因子"
    )

    # Enhanced: 160特征
    enhanced_results = train_and_evaluate(
        enable_features=True,
        label="Enhanced: 160个增强特征"
    )

    # 对比结果
    print("\n" + "="*80)
    print("Phase 1 验证结果对比")
    print("="*80)

    print(f"\n{'指标':<20} {'Baseline':<15} {'Enhanced':<15} {'提升':<15}")
    print("-" * 80)

    metrics = [
        ('特征数', 'n_features', ''),
        ('IC (整体)', 'ic', '.4f'),
        ('IC (日度均值)', 'ic_mean', '.4f'),
        ('IC 标准差', 'ic_std', '.4f'),
        ('IR', 'ic_ir', '.4f'),
        ('IC>0比例', 'ic_positive_ratio', '.2%')
    ]

    for label, key, fmt in metrics:
        baseline_val = baseline_results[key]
        enhanced_val = enhanced_results[key]

        if fmt:
            baseline_str = f"{baseline_val:{fmt}}"
            enhanced_str = f"{enhanced_val:{fmt}}"
        else:
            baseline_str = str(baseline_val)
            enhanced_str = str(enhanced_val)

        if key in ['ic', 'ic_mean', 'ic_ir']:
            improvement = ((enhanced_val - baseline_val) / abs(baseline_val)) * 100
            improvement_str = f"+{improvement:.1f}%"
        else:
            improvement_str = "-"

        print(f"{label:<20} {baseline_str:<15} {enhanced_str:<15} {improvement_str:<15}")

    # 验收判断
    print("\n" + "="*80)
    print("Phase 1 验收标准")
    print("="*80)

    target_ic = 0.028
    achieved_ic = enhanced_results['ic_mean']

    if achieved_ic >= target_ic:
        print(f"✓ PASS: IC={achieved_ic:.4f} >= 目标{target_ic}")
        print(f"✓ Phase 1完成，进入Phase 2（模型集成）")
    else:
        print(f"✗ FAIL: IC={achieved_ic:.4f} < 目标{target_ic}")
        print(f"建议: 调整特征工程配置或增加更多交叉特征")
        sys.exit(1)


if __name__ == '__main__':
    main()
