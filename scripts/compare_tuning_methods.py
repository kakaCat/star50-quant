#!/usr/bin/env python3
"""
对比三种调参方法的性能
=====================

使用相同的数据和参数空间，对比：
1. 随机搜索 (Random Search)
2. 网格搜索 (Grid Search)
3. 贝叶斯优化 (Bayesian Optimization)
"""

import subprocess
import time
import json
import pandas as pd

def run_tuning(method, n_iter=20):
    """运行调参并记录时间"""
    print(f"\n{'='*70}")
    print(f"运行 {method.upper()} 方法 ({n_iter}次试验)")
    print('='*70)

    start_time = time.time()

    cmd = [
        'python', 'star50-quant/scripts/tune_xgb_model.py',
        '--method', method,
        '--n_iter', str(n_iter),
        '--start_date', '2024-01-01',
        '--end_date', '2024-12-31',
        '--cv_folds', '3',
        '--output_dir', f'tuning_results/comparison_{method}'
    ]

    if method == 'grid':
        # 网格搜索不需要n_iter参数
        cmd = [
            'python', 'star50-quant/scripts/tune_xgb_model.py',
            '--method', 'grid',
            '--start_date', '2024-01-01',
            '--end_date', '2024-12-31',
            '--cv_folds', '3',
            '--output_dir', f'tuning_results/comparison_grid'
        ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    elapsed_time = time.time() - start_time

    return elapsed_time, result.returncode == 0

def analyze_results():
    """分析三种方法的结果"""
    methods = ['random', 'bayesian']  # grid暂时跳过（太慢）

    results = []

    for method in methods:
        try:
            # 读取结果文件（需要找到最新的）
            import glob
            result_files = glob.glob(f'tuning_results/comparison_{method}/xgb_tuning_{method}_*.json')
            if result_files:
                latest_file = max(result_files)
                with open(latest_file) as f:
                    data = json.load(f)

                # 读取试验数据
                trials_file = latest_file.replace('.json', '_trials.csv')
                trials = pd.read_csv(trials_file)

                best_trial = trials.loc[trials['composite_score'].idxmax()]

                results.append({
                    'method': method,
                    'best_ic': best_trial['ic'],
                    'best_ir': best_trial['ir'],
                    'best_return': best_trial['annual_return'],
                    'best_drawdown': best_trial['max_drawdown'],
                    'best_score': best_trial['composite_score'],
                    'n_trials': len(trials)
                })
        except Exception as e:
            print(f"Error processing {method}: {e}")

    return pd.DataFrame(results)

if __name__ == '__main__':
    print("="*70)
    print("三种调参方法对比实验")
    print("="*70)
    print("\n使用2024年数据，每种方法20次试验")
    print("(网格搜索跳过，因为组合数太多)")

    # 运行三种方法
    times = {}

    # 1. 随机搜索
    times['random'], success = run_tuning('random', 20)
    print(f"✓ 随机搜索完成，耗时 {times['random']:.1f}秒")

    # 2. 贝叶斯优化
    times['bayesian'], success = run_tuning('bayesian', 20)
    print(f"✓ 贝叶斯优化完成，耗时 {times['bayesian']:.1f}秒")

    # 3. 网格搜索（跳过）
    print("\n✗ 网格搜索跳过（参数组合数 > 1000，耗时过长）")

    # 分析结果
    print("\n" + "="*70)
    print("结果对比")
    print("="*70)

    df = analyze_results()
    print(df.to_string(index=False))

    print("\n" + "="*70)
    print("时间对比")
    print("="*70)
    for method, elapsed in times.items():
        print(f"{method:15s}: {elapsed:6.1f}秒")
