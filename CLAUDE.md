# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **科创50指数增强策略系统** (SSE STAR 50 Index Enhancement Strategy System) - a quantitative trading system that uses deep learning and reinforcement learning to generate alpha against the STAR 50 index. The system implements a three-driver approach:

1. **Alpha Prediction** - Deep learning models capture non-linear relationships between multi-factors and future returns
2. **Risk Modeling** - Deep risk models (autoencoders) for effective risk control
3. **Reinforcement Learning** - RL agents optimize dynamic rebalancing and execution strategies

## Architecture

The core system lives in `star50-quant/` with a modular pipeline architecture:

```
Data Layer → Feature Engineering → Models (Alpha/Risk/RL) → Portfolio Optimization → Backtesting
```

### Key Modules

- **`src/data/`** - Data collection (AkShare, Baostock), database management (PostgreSQL), and preprocessing
  - `loaders.py`: Main data loader for features and returns
  - `collectors/`: Multi-source market data collection
  - `database/`: SQLAlchemy models and connection management
  
- **`src/features/`** - Feature engineering pipeline that expands 30 technical factors to 160+ features
  - `engineering.py`: Main `FeatureEngineer` class orchestrates the 30→160 expansion
  - `momentum.py`, `trend.py`, `volume.py`: Calculate 30 base technical factors (MACD, RSI, MA, OBV, etc.)
  - `alpha_factors.py`: 10 WorldQuant-style alpha factors (alpha_001, alpha_006, alpha_053, alpha_054, alpha_101 + 5 custom)
  - Feature expansion strategies: cross features (+50), temporal derivatives (+30), nonlinear transforms (+20), cross-sectional stats (+15), alpha factors (+15)
  - Configuration driven via `configs/feature_config.yaml`

- **`src/models/`** - Machine learning models
  - `data_loader.py`: Factor data loader with train/validation/test splitting and preprocessing (deextreme, standardization)
  - `lstm_model.py`: LSTM for time-series alpha prediction
  - `lgbm_model.py`: LightGBM baseline model
  - `alpha/`: Alpha prediction models (LSTM, GRU, TFT)
  - `risk/`: Deep risk models (autoencoders for non-linear risk factors)
  - `rl/`: Reinforcement learning agents for portfolio rebalancing

- **`src/optimization/`** - Portfolio construction with cvxpy
  - Mean-variance optimization with constraints (industry neutralization, tracking error, individual stock limits)
  
- **`src/backtest/`** - Backtesting engine
  - Realistic transaction costs (0.15% two-way)
  - STAR market constraints (20% price limit, no short selling)
  - Survivorship bias handling

### Data Flow

1. **Data Collection**: Scripts pull OHLCV + fundamentals from multiple sources → PostgreSQL tables (`stock_daily`, `stock_minute`, `fundamentals`, `index_components`)
2. **Factor Calculation**: `calculate_factors.py` computes 30 technical factors → `factors` table
3. **Feature Engineering**: `FeatureEngineer.transform()` expands to 160 features (configured in `configs/feature_config.yaml`)
4. **Model Training**: `FactorDataLoader` loads data, models predict alpha/risk → MLflow tracking
5. **Portfolio Optimization**: Combine alpha + risk → constrained optimization → portfolio weights
6. **Backtesting**: Simulate trading with realistic constraints → evaluation metrics (IR, tracking error, max drawdown)

## Common Commands

### Environment Setup
```bash
cd star50-quant
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Database
```bash
# Initialize database schema
python scripts/setup_database.py

# Collect market data (AkShare + Baostock)
python scripts/collect_data.py --type all
```

### Feature Engineering
```bash
# Calculate 30 base factors from raw OHLCV
python scripts/calculate_factors.py --start_date 2019-01-01 --end_date 2024-12-31

# Parquet-based version (faster for large datasets)
python scripts/calculate_factors_parquet.py

# Validate Phase 1 completion (data + features)
python scripts/validate_phase1.py
```

### Model Training
```bash
# Train alpha prediction models
python scripts/train_alpha_model.py --model lstm --epochs 100
python scripts/train_alpha_model.py --model lgbm

# Train risk model (autoencoder)
python scripts/train_risk_model.py --model autoencoder --latent_dim 10

# Evaluate models
python scripts/evaluate_model.py --model_path mlruns/1/xxx/artifacts/model
python scripts/evaluate_risk_model.py --model_path mlruns/2/xxx/artifacts/model
```

### Portfolio Optimization & Backtesting
```bash
# Run portfolio optimization
python scripts/optimize_portfolio.py --alpha_path alpha_predictions.csv --risk_path risk_matrix.csv

# Run backtest with configuration
python scripts/run_backtest.py --config configs/backtest_config.yaml

# View MLflow experiments
mlflow ui --port 5000  # Open http://localhost:5000
```

### Testing
```bash
# Run all tests with coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# Run specific test modules
pytest tests/features/ -v
pytest tests/models/ -v

# Run single test file
pytest tests/features/test_engineering.py -v -k test_feature_expansion
```

## Development Workflow

### Working with Features
- Base factors are defined in `src/features/{momentum,trend,volume}.py` using the `FactorCalculator` base class
- Feature engineering expansion logic is in `src/features/engineering.py` - modify `transform()` method
- Configuration in `configs/feature_config.yaml` controls which factors/transformations are applied
- After modifying features, run `pytest tests/features/` and validate with `scripts/test_factors.py`

### Working with Models
- Models inherit from PyTorch Lightning's `LightningModule` for standardized training loops
- All training scripts use MLflow for experiment tracking (runs logged to `mlruns/`)
- `FactorDataLoader` handles data loading with proper temporal splitting (no look-ahead bias)
- Use walk-forward validation for realistic performance assessment

### Database Interaction
- Use the connection from `src/data/database/` rather than raw psycopg2
- Table schemas defined in `src/data/database/models.py`
- Environment variables for credentials: `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` (see `configs/db_config.yaml`)
- Default local setup: PostgreSQL on localhost with database `star50_quant`

## Key Constraints

### Index Enhancement Constraints
The portfolio optimizer enforces these constraints to maintain "index enhancement" characteristics:
- **Tracking error**: 4-6% annualized
- **Industry neutralization**: Sector exposure deviation from benchmark ≤ ±2%
- **Individual stock limits**: Position deviation from benchmark weight ≤ ±1%
- **Non-constituent stocks**: Max 10% allocation outside STAR 50 components

### STAR Market Trading Rules
Backtest engine accounts for STAR-specific rules:
- **Price limits**: ±20% daily (vs ±10% on main board)
- **No short selling**
- **Higher transaction costs**: 0.15% two-way (includes stamp tax, commission, slippage)
- **Impact costs**: Material for less liquid names

## Testing Philosophy

- **Unit tests** for feature calculations verify mathematical correctness
- **Integration tests** for data loaders ensure proper pipeline integration
- **Model tests** check training/inference shapes and loss convergence
- Use fixtures in `conftest.py` for reusable test data
- Mock database connections when testing data layer in isolation

## Important Notes

- **No look-ahead bias**: All training uses strict temporal splits. The `FactorDataLoader` enforces walk-forward validation.
- **Survivorship bias handling**: Include delisted stocks in sample universe (tracked via `index_components` table with historical membership)
- **Factor preprocessing**: Always apply cross-sectional deextreme (MAD method) + Z-score standardization before model training
- **Alpha factors**: WorldQuant-style alphas in `alpha_factors.py` are rank-based and require cross-sectional data
- **Configuration precedence**: YAML configs in `configs/` override defaults; environment variables override YAML
