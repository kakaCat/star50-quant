# 科创50指数增强策略系统

基于深度学习与强化学习的科创50指数增强量化交易策略系统。

## 项目概述

本项目通过以下三驱动模式实现超越科创50指数的收益表现：

1. **Alpha预测** - 深度学习模型捕捉多因子与未来收益率的非线性关系
2. **风险建模** - 深度风险模型进行有效的风险控制
3. **强化学习决策** - RL智能体优化动态调仓和交易执行策略

## 技术栈

- **语言**: Python 3.8+
- **数据存储**: PostgreSQL
- **深度学习**: PyTorch, PyTorch Lightning
- **强化学习**: Stable-Baselines3, Gymnasium
- **实验追踪**: MLflow
- **优化库**: cvxpy, scipy
- **数据处理**: pandas, numpy

## 项目结构

```
star50-quant/
├── data/              # 数据目录
├── src/               # 核心源代码
│   ├── data/         # 数据层
│   ├── features/     # 特征工程
│   ├── models/       # 模型层（Alpha/风险/RL）
│   ├── optimization/ # 组合优化
│   ├── backtest/     # 回测引擎
│   └── utils/        # 工具函数
├── notebooks/         # Jupyter notebooks
├── configs/           # 配置文件
├── scripts/           # 执行脚本
├── tests/             # 单元测试
└── mlruns/            # MLflow实验追踪
```

## 快速开始

### 1. 环境搭建

```bash
# 进入项目目录
cd star50-quant

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
pip install -e .
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑.env文件，填入数据库配置和API密钥
```

### 3. 初始化数据库

```bash
python scripts/setup_database.py
```

### 4. 数据采集

```bash
python scripts/collect_data.py --type all
```

### 5. 特征构建

```bash
python scripts/build_features.py
```

### 6. 模型训练

```bash
# 训练Alpha模型
python scripts/train_alpha.py --model lstm

# 训练风险模型
python scripts/train_risk.py --model autoencoder

# 训练强化学习智能体
python scripts/train_rl_agent.py --agent ppo --env rebalance
```

### 7. 回测评估

```bash
python scripts/run_backtest.py --config configs/backtest_config.yaml
```

### 8. 启动MLflow UI

```bash
mlflow ui --port 5000
# 浏览器访问 http://localhost:5000
```

## 运行测试

```bash
pytest tests/ -v --cov=src
```

## 文档

详细文档请参考 `../docs/` 目录：
- 设计文档: `../docs/superpowers/specs/2026-06-09-star50-project-structure-design.md`
- 实施计划: `../docs/superpowers/plans/2026-06-09-data-module-implementation.md`

项目内实验记录：
- LightGBM 固定窗口与滚动窗口调优总结: `LIGHTGBM_IMPLEMENTATION_SUMMARY.md`
- LightGBM 详细实验记录: `docs/LIGHTGBM_WINDOW_OPTIMIZATION_REPORT.md`
- LightGBM 调优结果: `tuning_results/lightgbm_window/`
- LightGBM 最终结果 JSON: `results/lightgbm/lightgbm_results_20260612.json`

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request!
