#!/usr/bin/env python3
"""
阶段二：LightGBM 固定窗口贝叶斯优化

使用 Optuna 的 TPE 贝叶斯优化搜索 LightGBM 和组合构建参数，目标偏向：
- 提高卡玛比率（年化收益 / 最大回撤绝对值）
- 降低最大回撤
- 保留一定 IC / Rank IC 信号
"""

import argparse
import importlib.util
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import optuna
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXED_WINDOW_SCRIPT = PROJECT_ROOT / "scripts" / "05_lightgbm_fixed_window_backtest.py"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "alpha_fixed_window_optimized"


def load_fixed_window_module():
    """加载 05 脚本中的复用函数。"""
    spec = importlib.util.spec_from_file_location("fixed_window_backtest", FIXED_WINDOW_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metric_value(metrics: pd.DataFrame, name: str) -> float:
    """从指标表取数值。"""
    value = metrics.loc[metrics["metric"].eq(name), "LightGBM"].iloc[0]
    return float(value)


def suggest_params(trial: optuna.Trial) -> tuple[dict, dict]:
    """定义搜索空间。"""
    lgbm_params = {
        "n_estimators": trial.suggest_int("n_estimators", 40, 180),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 8, 63),
        "min_child_samples": trial.suggest_int("min_child_samples", 40, 180),
        "subsample": trial.suggest_float("subsample", 0.55, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 0.95),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
    }
    portfolio_params = {
        "rebalance_days": trial.suggest_categorical("rebalance_days", [5, 10, 20]),
        "top_quantile": trial.suggest_categorical(
            "top_quantile",
            [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
        ),
    }
    return lgbm_params, portfolio_params


def evaluate_once(
    fw,
    features: pd.DataFrame,
    forward_returns: pd.DataFrame,
    label: str,
    train_window_days: int,
    predict_window_days: int,
    transaction_cost: float,
    lgbm_params: dict,
    portfolio_params: dict,
) -> tuple[float, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """运行一次固定窗口训练与回测，返回优化目标和结果。"""
    predictions = fw.rolling_fixed_window_predict(
        features,
        label_column=label,
        train_window_days=train_window_days,
        predict_window_days=predict_window_days,
        lgbm_params=lgbm_params,
        verbose=False,
    )
    daily_returns = fw.build_daily_strategy_returns(
        predictions,
        forward_returns,
        rebalance_days=portfolio_params["rebalance_days"],
        top_quantile=portfolio_params["top_quantile"],
        transaction_cost=transaction_cost,
    )
    metrics = fw.calculate_metrics(daily_returns, predictions, label)

    calmar = metric_value(metrics, "卡玛比率")
    max_drawdown = metric_value(metrics, "最大回撤")
    annual_return = metric_value(metrics, "年化收益")
    rank_ic = metric_value(metrics, "Rank IC")

    # 目标：卡玛比为主，强惩罚超过40%的回撤；适度保留收益和Rank IC。
    drawdown_penalty = max(0.0, abs(max_drawdown) - 0.40) * 6.0
    drawdown_reward = max(0.0, 0.40 - abs(max_drawdown)) * 1.5
    objective = calmar + 0.15 * annual_return + 0.3 * rank_ic + drawdown_reward - drawdown_penalty
    return objective, metrics, predictions, daily_returns


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna 贝叶斯优化 LightGBM 固定窗口策略")
    parser.add_argument("--label", choices=["label_1d", "label_5d"], default="label_5d")
    parser.add_argument("--n-trials", type=int, default=12)
    parser.add_argument("--train-window-days", type=int, default=756)
    parser.add_argument("--predict-window-days", type=int, default=20)
    parser.add_argument("--transaction-cost", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fw = load_fixed_window_module()
    features = fw.load_features(args.label)
    forward_returns = fw.load_forward_returns()

    best_payload = {}

    def objective(trial: optuna.Trial) -> float:
        lgbm_params, portfolio_params = suggest_params(trial)
        score, metrics, predictions, daily_returns = evaluate_once(
            fw=fw,
            features=features,
            forward_returns=forward_returns,
            label=args.label,
            train_window_days=args.train_window_days,
            predict_window_days=args.predict_window_days,
            transaction_cost=args.transaction_cost,
            lgbm_params=lgbm_params,
            portfolio_params=portfolio_params,
        )

        for metric_name in ["卡玛比率", "年化收益", "最大回撤", "夏普比率", "IC", "Rank IC", "累计净值"]:
            trial.set_user_attr(metric_name, metric_value(metrics, metric_name))

        if not best_payload or score > best_payload["score"]:
            best_payload.clear()
            best_payload.update(
                {
                    "score": score,
                    "metrics": metrics,
                    "predictions": predictions,
                    "daily_returns": daily_returns,
                    "lgbm_params": lgbm_params,
                    "portfolio_params": portfolio_params,
                }
            )

        print(
            f"trial={trial.number:02d} score={score:.4f} "
            f"calmar={metric_value(metrics, '卡玛比率'):.4f} "
            f"mdd={metric_value(metrics, '最大回撤'):.2%} "
            f"ann={metric_value(metrics, '年化收益'):.2%} "
            f"rank_ic={metric_value(metrics, 'Rank IC'):.4f}"
        )
        return score

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.enqueue_trial(
        {
            "n_estimators": 120,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "min_child_samples": 80,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "rebalance_days": 5,
            "top_quantile": 0.2,
        }
    )
    study.optimize(objective, n_trials=args.n_trials)

    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
    trials_file = OUTPUT_DIR / f"optuna_trials_{args.label}.csv"
    best_metrics_file = OUTPUT_DIR / f"best_fixed_window_metrics_{args.label}.csv"
    best_predictions_file = OUTPUT_DIR / f"best_fixed_window_predictions_{args.label}.parquet"
    best_daily_file = OUTPUT_DIR / f"best_fixed_window_daily_nav_{args.label}.csv"
    best_params_file = OUTPUT_DIR / f"best_params_{args.label}.csv"
    best_plot_file = OUTPUT_DIR / f"best_fixed_window_nav_{args.label}.png"

    trials_df.to_csv(trials_file, index=False, encoding="utf-8-sig")
    best_payload["metrics"].to_csv(best_metrics_file, index=False, encoding="utf-8-sig")
    best_payload["predictions"].to_parquet(best_predictions_file, index=False)
    best_payload["daily_returns"].to_csv(best_daily_file, index=False, encoding="utf-8-sig")
    fw.plot_nav(best_payload["daily_returns"], best_plot_file)

    params_df = pd.DataFrame(
        [{"param": key, "value": value} for key, value in {
            **best_payload["lgbm_params"],
            **best_payload["portfolio_params"],
        }.items()]
    )
    params_df.to_csv(best_params_file, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print("贝叶斯优化完成")
    print(f"最佳目标值：{study.best_value:.6f}")
    print("最佳参数：")
    print(params_df.to_string(index=False))
    print("\n最佳指标：")
    print(best_payload["metrics"].to_string(index=False))
    print(f"\n试验记录：{trials_file}")
    print(f"最佳指标：{best_metrics_file}")
    print(f"最佳参数：{best_params_file}")
    print(f"最佳每日净值：{best_daily_file}")
    print(f"最佳净值图：{best_plot_file}")


if __name__ == "__main__":
    main()
