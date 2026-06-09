#!/bin/bash
# Star50量化系统 - 数据集成验证脚本
# 用于验证Parquet数据集成是否成功

echo "================================================================"
echo "Star50量化系统 - Parquet数据集成验证"
echo "================================================================"
echo ""

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查数据文件
echo "1. 检查数据文件..."
if [ -f "data/raw/star50_daily_hfq_data_6yrs.parquet" ]; then
    echo -e "${GREEN}✓${NC} 股票数据文件存在"
    ls -lh data/raw/star50_daily_hfq_data_6yrs.parquet | awk '{print "  大小: " $5}'
else
    echo -e "${RED}✗${NC} 股票数据文件不存在"
    exit 1
fi

if [ -f "data/raw/star50_index_daily_6yrs.parquet" ]; then
    echo -e "${GREEN}✓${NC} 指数数据文件存在"
    ls -lh data/raw/star50_index_daily_6yrs.parquet | awk '{print "  大小: " $5}'
else
    echo -e "${RED}✗${NC} 指数数据文件不存在"
    exit 1
fi
echo ""

# 检查代码文件
echo "2. 检查代码文件..."
files=(
    "src/data/loaders.py"
    "src/models/data_loader_parquet.py"
    "scripts/calculate_factors_parquet.py"
    "examples/load_data_example.py"
    "quickstart_data.py"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓${NC} $file"
    else
        echo -e "${RED}✗${NC} $file 不存在"
    fi
done
echo ""

# 检查依赖
echo "3. 检查Python依赖..."
python -c "import pyarrow; print('✓ pyarrow 版本:', pyarrow.__version__)" 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} pyarrow 已安装"
else
    echo -e "${YELLOW}⚠${NC} pyarrow 未安装，请运行: pip install pyarrow>=12.0.0"
fi

python -c "import pandas; print('✓ pandas 版本:', pandas.__version__)" 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} pandas 已安装"
else
    echo -e "${RED}✗${NC} pandas 未安装"
fi
echo ""

# 运行快速测试
echo "4. 运行功能测试..."
echo "测试基础数据加载..."
python -c "
from src.data.loaders import DataLoader
loader = DataLoader()
stocks = loader.get_stock_list()
print(f'✓ 加载成功: {len(stocks)} 只股票')
" 2>/dev/null

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} 基础数据加载正常"
else
    echo -e "${RED}✗${NC} 基础数据加载失败"
fi

echo "测试数据集构建..."
python -c "
from src.models.data_loader_parquet import FactorDataLoader
loader = FactorDataLoader()
features, labels = loader.build_dataset('2024-06-01', '2024-06-30', use_basic_factors=True)
print(f'✓ 数据集构建成功: {features.shape[0]} 样本')
" 2>/dev/null

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} 数据集构建正常"
else
    echo -e "${RED}✗${NC} 数据集构建失败"
fi
echo ""

# 显示总结
echo "================================================================"
echo "验证完成"
echo "================================================================"
echo ""
echo "下一步操作:"
echo "  1. 运行快速示例: python quickstart_data.py"
echo "  2. 查看详细示例: python examples/load_data_example.py"
echo "  3. 计算因子: python scripts/calculate_factors_parquet.py --all"
echo "  4. 查看文档: cat MIGRATION_SUMMARY.md"
echo ""
echo "文档位置:"
echo "  - 数据集成说明: docs/DATA_INTEGRATION.md"
echo "  - 迁移指南: docs/MIGRATION_TO_PARQUET.md"
echo "  - 测试报告: TEST_REPORT.md"
echo ""
