#!/usr/bin/env python3
"""
XGBoost调参系统完整性检查
========================

检查所有依赖、文件和配置是否就绪
"""

import sys
import os
from pathlib import Path


def check_dependencies():
    """检查Python依赖"""
    print("="*60)
    print("检查 Python 依赖")
    print("="*60)

    required = {
        'xgboost': '2.0.0',
        'optuna': '3.0.0',
        'sklearn': '1.2.0',
        'pandas': '1.5.0',
        'numpy': '1.23.0'
    }

    all_ok = True

    for package, min_version in required.items():
        try:
            if package == 'sklearn':
                import sklearn
                version = sklearn.__version__
            else:
                mod = __import__(package)
                version = mod.__version__

            print(f"  ✓ {package}: {version}")
        except ImportError:
            print(f"  ✗ {package}: 未安装")
            all_ok = False

    return all_ok


def check_files():
    """检查必要文件"""
    print("\n" + "="*60)
    print("检查项目文件")
    print("="*60)

    required_files = [
        'src/models/xgb_model.py',
        'src/models/hyperparameter_tuning.py',
        'scripts/tune_xgb_model.py',
        'scripts/test_tuning_system.py',
        'docs/xgboost-tuning-guide.md',
        'docs/xgboost-tuning-summary.md',
        'docs/README-XGBoost-Tuning.md',
    ]

    all_ok = True

    for filepath in required_files:
        if Path(filepath).exists():
            print(f"  ✓ {filepath}")
        else:
            print(f"  ✗ {filepath}: 文件不存在")
            all_ok = False

    return all_ok


def check_data():
    """检查数据可用性"""
    print("\n" + "="*60)
    print("检查数据")
    print("="*60)

    try:
        from src.models.data_loader import FactorDataLoader

        loader = FactorDataLoader(db_name='star50_quant')
        loader.connect()

        # 尝试加载少量数据
        loader.load_data(
            start_date='2024-01-01',
            end_date='2024-01-31'
        )

        if hasattr(loader, 'features') and len(loader.features) > 0:
            print(f"  ✓ 数据库连接正常")
            print(f"  ✓ 因子数据可用")
            print(f"    样本数: {len(loader.features)}")
            print(f"    特征数: {len(loader.features.columns) - 2}")
            loader.close()
            return True
        else:
            print(f"  ✗ 因子数据为空")
            loader.close()
            return False

    except Exception as e:
        print(f"  ✗ 数据检查失败: {e}")
        print(f"\n  提示: 请先准备数据")
        print(f"    python scripts/calculate_factors.py --start_date 2020-01-01 --end_date 2024-12-31")
        return False


def main():
    """运行所有检查"""
    print("\n" + "="*60)
    print("XGBoost调参系统完整性检查")
    print("="*60)

    checks = {
        '依赖': check_dependencies(),
        '文件': check_files(),
        '数据': check_data()
    }

    # 总结
    print("\n" + "="*60)
    print("检查总结")
    print("="*60)

    for name, passed in checks.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {name}: {status}")

    all_passed = all(checks.values())

    print("\n" + "="*60)

    if all_passed:
        print("🎉 系统就绪！")
        print("\n下一步:")
        print("  1. 快速测试: python scripts/test_tuning_system.py")
        print("  2. 真实调参: python scripts/tune_xgb_model.py --method bayesian --n_iter 50")
    else:
        print("⚠️ 系统未就绪，请解决上述问题")

        if not checks['依赖']:
            print("\n安装依赖:")
            print("  pip install -r requirements.txt")

        if not checks['数据']:
            print("\n准备数据:")
            print("  python scripts/calculate_factors.py --start_date 2020-01-01 --end_date 2024-12-31")

    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
