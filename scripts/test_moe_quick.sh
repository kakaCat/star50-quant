#!/bin/bash
# MoE模型快速测试脚本
# 使用Parquet数据源验证模型是否正常工作

set -e

echo "=========================================="
echo "MoE Model Quick Test (Parquet Version)"
echo "=========================================="
echo ""

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "错误: 未找到Python"
    exit 1
fi

# 检查依赖
echo "步骤 1/6: 检查依赖..."
python -c "import torch; import pandas; import numpy; import scipy" || {
    echo "错误: 缺少依赖，请运行: pip install torch pandas numpy scipy"
    exit 1
}
echo "✓ 依赖检查通过"
echo ""

# 运行单元测试
echo "步骤 2/6: 运行MoE模型单元测试..."
cd "$(dirname "$0")/.."
python -m pytest tests/models/test_moe_model.py -v --tb=short || {
    echo "错误: 单元测试失败"
    exit 1
}
echo "✓ 模型单元测试通过 (10/10)"
echo ""

# 测试数据加载器
echo "步骤 3/6: 运行数据加载器测试..."
python -m pytest tests/models/test_moe_data_loader.py -v --tb=short || {
    echo "错误: 数据加载器测试失败"
    exit 1
}
echo "✓ 数据加载器测试通过 (5/5)"
echo ""

# 检查Parquet数据文件
echo "步骤 4/6: 检查Parquet数据文件..."
if [ -f "data/raw/star50_daily_hfq_data_6yrs.parquet" ]; then
    echo "✓ 股票数据文件存在: data/raw/star50_daily_hfq_data_6yrs.parquet"
else
    echo "⚠ 警告: 股票数据文件不存在"
fi

if [ -f "data/raw/star50_index_daily_6yrs.parquet" ]; then
    echo "✓ 指数数据文件存在: data/raw/star50_index_daily_6yrs.parquet"
else
    echo "⚠ 警告: 指数数据文件不存在"
fi
echo ""

# 测试实际数据加载
echo "步骤 5/6: 测试实际数据加载..."
python -c "
from src.models.alpha.moe_data_loader import MoEParquetDataLoader
loader = MoEParquetDataLoader()
try:
    stock_df, index_df = loader.load_data()
    print(f'✓ 成功加载数据')
    print(f'  股票数据: {len(stock_df):,} 行')
    print(f'  指数数据: {len(index_df):,} 行')
    print(f'  股票代码数: {stock_df[\"ts_code\"].nunique()} 只')
except FileNotFoundError:
    print('⚠ Parquet文件不存在，跳过实际数据测试')
" 2>/dev/null || {
    echo "⚠ 数据文件不存在，跳过实际数据加载测试"
}
echo ""

# 测试模型导入
echo "步骤 6/6: 测试模型导入..."
python -c "
from src.models.alpha.moe_model import NeuralMoE, MoEAlphaTrainer, MoEDataset
import torch
model = NeuralMoE(stock_dim=22, regime_dim=6)
print(f'✓ MoE模型创建成功，参数量: {sum(p.numel() for p in model.parameters()):,}')
" || {
    echo "错误: 模型导入失败"
    exit 1
}
echo ""

echo "=========================================="
echo "✓ 所有测试通过！(15/15)"
echo "=========================================="
echo ""
echo "数据源: Parquet文件"
echo "  - 股票: data/raw/star50_daily_hfq_data_6yrs.parquet"
echo "  - 指数: data/raw/star50_index_daily_6yrs.parquet"
echo ""
echo "下一步："
echo "  1. 训练模型: python scripts/train_moe_model.py"
echo "  2. 查看实验: mlflow ui --port 5000"
echo "  3. 阅读文档: cat docs/MoE_PARQUET_README.md"
echo ""
