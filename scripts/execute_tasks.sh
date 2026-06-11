#!/bin/bash
# XGBoost调参系统 - 执行任务清单
# ====================================
#
# 本脚本列出了所有待执行的任务
# 可以逐个复制执行，或者直接运行整个脚本

set -e  # 遇到错误立即退出

echo "======================================================================"
echo "XGBoost调参系统 - 待执行任务"
echo "======================================================================"
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 任务状态
COMPLETED="✅"
PENDING="⏳"
FAILED="❌"

echo "======================================================================"
echo "当前状态检查"
echo "======================================================================"
echo ""

# 检查已完成的工作
echo -e "${GREEN}${COMPLETED} 阶段0: 数据准备${NC}"
echo "   - 因子数据已计算（120万+记录）"
echo ""

echo -e "${GREEN}${COMPLETED} 阶段1: 模型调参${NC}"
echo "   - 贝叶斯优化完成（50次试验）"
echo "   - IC = 0.0953"
echo ""

echo -e "${GREEN}${COMPLETED} 阶段2: 模型训练${NC}"
echo "   - 最佳模型已保存"
echo ""

echo -e "${YELLOW}${PENDING} 阶段3: 完整回测${NC} - 待执行"
echo -e "${YELLOW}${PENDING} 阶段4: 策略优化${NC} - 待执行"
echo -e "${YELLOW}${PENDING} 阶段5: 生产部署${NC} - 待执行"
echo ""

echo "======================================================================"
echo "任务1: 检查现有回测框架"
echo "======================================================================"
echo ""

# 任务1: 检查是否有回测脚本
if [ -f "star50-quant/scripts/run_backtest.py" ]; then
    echo -e "${GREEN}${COMPLETED} 发现回测脚本: star50-quant/scripts/run_backtest.py${NC}"
    echo ""
    echo "可以直接执行:"
    echo ""
    echo "python star50-quant/scripts/run_backtest.py \\"
    echo "    --model_path star50-quant/tuning_results/full_run_fixed/xgb_best_model_20260610_211911.json \\"
    echo "    --start_date 2020-01-02 \\"
    echo "    --end_date 2024-12-31"
    echo ""
else
    echo -e "${RED}${FAILED} 未找到回测脚本${NC}"
    echo ""
    echo "需要先实现回测框架。可以："
    echo ""
    echo "选项A: 检查是否有其他回测相关脚本"
    echo "  ls star50-quant/scripts/*backtest* 2>/dev/null || echo '无回测脚本'"
    echo ""
    echo "选项B: 查看是否有回测模块"
    echo "  ls star50-quant/src/backtest/ 2>/dev/null || echo '无回测模块'"
    echo ""
    echo "选项C: 创建简单的回测脚本（推荐）"
    echo "  参考文档: star50-quant/docs/WORKFLOW-COMPLETE.md 附录A"
    echo ""
fi

echo "======================================================================"
echo "任务2: 执行完整回测（如果回测框架可用）"
echo "======================================================================"
echo ""

cat << 'EOF'
# 步骤2.1: 运行基准回测
python star50-quant/scripts/run_backtest.py \
    --model_path star50-quant/tuning_results/full_run_fixed/xgb_best_model_20260610_211911.json \
    --start_date 2020-01-02 \
    --end_date 2024-12-31 \
    --select_pct 0.2 \
    --weight_method equal \
    --output_dir star50-quant/backtest_results/baseline

# 步骤2.2: 查看回测结果
python -c "
import pandas as pd
results = pd.read_csv('star50-quant/backtest_results/baseline/performance_metrics.csv')
print('回测结果:')
print(f'IC:        {results[\"ic\"].mean():.4f}')
print(f'IR:        {results[\"ir\"][0]:.2f}')
print(f'年化收益:  {results[\"annual_return\"][0]:.2%}')
print(f'最大回撤:  {results[\"max_drawdown\"][0]:.2%}')
"
EOF

echo ""
echo "======================================================================"
echo "任务3: 策略参数优化（如果回测结果未达标）"
echo "======================================================================"
echo ""

cat << 'EOF'
# 步骤3.1: 定义策略参数网格
cat > strategy_grid.yaml << 'YAML'
select_pct: [0.10, 0.15, 0.20, 0.25, 0.30]
weight_method: ['equal', 'signal', 'signal_squared']
rebalance: ['daily']
YAML

# 步骤3.2: 遍历所有策略配置
for select_pct in 0.10 0.15 0.20 0.25 0.30; do
  for weight_method in equal signal signal_squared; do
    echo "测试: 选股${select_pct}, 权重=${weight_method}"

    python star50-quant/scripts/run_backtest.py \
        --model_path star50-quant/tuning_results/full_run_fixed/xgb_best_model_20260610_211911.json \
        --start_date 2020-01-02 \
        --end_date 2024-12-31 \
        --select_pct $select_pct \
        --weight_method $weight_method \
        --output_dir star50-quant/backtest_results/select_${select_pct}_${weight_method}
  done
done

# 步骤3.3: 汇总所有结果
python -c "
import pandas as pd
import glob

results = []
for file in glob.glob('star50-quant/backtest_results/*/performance_metrics.csv'):
    df = pd.read_csv(file)
    df['config'] = file.split('/')[-2]
    results.append(df)

all_results = pd.concat(results)
all_results = all_results.sort_values('composite_score', ascending=False)

print('Top 5 策略配置:')
print(all_results.head()[['config', 'ic', 'ir', 'annual_return', 'max_drawdown']])
"
EOF

echo ""
echo "======================================================================"
echo "任务4: 创建简单回测脚本（如果没有现成的）"
echo "======================================================================"
echo ""

cat << 'EOF'
# 创建简单的回测脚本
cat > star50-quant/scripts/simple_backtest.py << 'PYTHON'
#!/usr/bin/env python3
"""
简单回测脚本
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import pandas as pd
import numpy as np
from datetime import timedelta
from src.models.xgb_model import XGBoostAlphaModel
from src.models.data_loader import FactorDataLoader

def run_backtest(model_path, start_date, end_date, select_pct=0.2, weight_method='equal'):
    """运行简单回测"""

    # 加载模型
    model = XGBoostAlphaModel()
    model.load(model_path)

    # 加载数据
    loader = FactorDataLoader(db_name='star50_quant')
    loader.connect()

    factors_long = loader.load_factors(start_date, end_date)
    features = loader.pivot_factors(factors_long)

    end_extended = (pd.to_datetime(end_date) + timedelta(days=30)).strftime('%Y-%m-%d')
    prices = loader.load_prices(start_date, end_extended)
    labels = loader.calculate_returns(prices, forward_days=5)

    labels['factor_date'] = pd.to_datetime(labels['trade_date'])
    features['factor_date'] = pd.to_datetime(features['factor_date'])

    data = features.merge(
        labels[['ts_code', 'factor_date', 'forward_return']],
        on=['ts_code', 'factor_date'],
        how='inner'
    ).dropna()

    data = loader.winsorize(data.copy())
    data = loader.standardize(data)
    loader.close()

    # 预测
    predict_features = data.drop(['forward_return'], axis=1)
    predictions = model.predict(predict_features)
    data['alpha'] = predictions

    # 回测
    daily_returns = []
    daily_ics = []

    for date, group in data.groupby('factor_date'):
        if len(group) < 5:
            continue

        n_select = max(int(len(group) * select_pct), 1)
        selected = group.nlargest(n_select, 'alpha')

        if weight_method == 'equal':
            weights = np.ones(len(selected)) / len(selected)
        elif weight_method == 'signal':
            alphas = selected['alpha'].values - selected['alpha'].values.min() + 0.01
            weights = alphas / alphas.sum()
        elif weight_method == 'signal_squared':
            alphas = selected['alpha'].values - selected['alpha'].values.min() + 0.01
            alphas_sq = alphas ** 2
            weights = alphas_sq / alphas_sq.sum()

        portfolio_return = np.dot(weights, selected['forward_return'].values)
        portfolio_return -= 0.002  # 交易成本

        daily_returns.append(portfolio_return)

        ic = np.corrcoef(selected['alpha'].values, selected['forward_return'].values)[0, 1]
        if not np.isnan(ic):
            daily_ics.append(ic)

    returns_array = np.array(daily_returns)

    # 计算指标
    mean_ic = np.mean(daily_ics)
    cumulative = np.cumprod(1 + returns_array)
    total_return = cumulative[-1] - 1

    n_days = len(returns_array)
    annual_return = (1 + total_return) ** (252 / n_days) - 1 if n_days > 0 else 0

    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = np.min(drawdown)

    mean_return = np.mean(returns_array)
    std_return = np.std(returns_array)
    ir = (mean_return * np.sqrt(252)) / std_return if std_return > 0 else 0

    # 打印结果
    print(f"\n回测结果:")
    print(f"  IC:        {mean_ic:.4f}")
    print(f"  IR:        {ir:.2f}")
    print(f"  年化收益:  {annual_return:.2%}")
    print(f"  最大回撤:  {max_drawdown:.2%}")

    targets_met = 0
    if mean_ic > 0.04:
        print(f"  ✓ IC达标")
        targets_met += 1
    if ir >= 1.5:
        print(f"  ✓ IR达标")
        targets_met += 1
    if annual_return > 0.35:
        print(f"  ✓ 年化收益达标")
        targets_met += 1
    if max_drawdown >= -0.20:
        print(f"  ✓ 最大回撤达标")
        targets_met += 1

    print(f"\n  目标达成: {targets_met}/4")

    return {
        'ic': mean_ic,
        'ir': ir,
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'targets_met': targets_met
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='简单回测')
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--start_date', default='2020-01-02')
    parser.add_argument('--end_date', default='2024-12-31')
    parser.add_argument('--select_pct', type=float, default=0.2)
    parser.add_argument('--weight_method', default='equal')

    args = parser.parse_args()

    run_backtest(
        args.model_path,
        args.start_date,
        args.end_date,
        args.select_pct,
        args.weight_method
    )
PYTHON

chmod +x star50-quant/scripts/simple_backtest.py

echo "✓ 简单回测脚本已创建"
EOF

echo ""
echo "======================================================================"
echo "快速测试命令（推荐立即执行）"
echo "======================================================================"
echo ""

echo "# 创建简单回测脚本并运行"
echo "python star50-quant/scripts/simple_backtest.py \\"
echo "    --model_path star50-quant/tuning_results/full_run_fixed/xgb_best_model_20260610_211911.json \\"
echo "    --start_date 2024-01-01 \\"
echo "    --end_date 2024-12-31 \\"
echo "    --select_pct 0.2 \\"
echo "    --weight_method equal"
echo ""

echo "======================================================================"
echo "总结"
echo "======================================================================"
echo ""
echo "当前进度:"
echo -e "${GREEN}${COMPLETED} 阶段0-2: 已完成${NC}"
echo -e "${YELLOW}${PENDING} 阶段3: 需要完整回测${NC}"
echo -e "${YELLOW}${PENDING} 阶段4-5: 等待阶段3${NC}"
echo ""
echo "下一步:"
echo "1. 检查是否有现成的回测框架"
echo "2. 如果没有，创建简单回测脚本"
echo "3. 运行回测得到真实指标"
echo "4. 根据结果决定是否需要策略优化"
echo ""
echo "详细文档: star50-quant/docs/WORKFLOW-COMPLETE.md"
echo ""
